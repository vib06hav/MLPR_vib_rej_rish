# models.py

import torch
import torch.nn as nn
import torchvision.models as tv_models

from config import (
    NUM_CLASSES, FEATURE_DIM,
    DROPOUT_RATE, BATCHNORM_IN_PROJECTOR
)


# ════════════════════════════════════════════════════════════════════════════
# Base class — enforces get_features() interface across all three models
# ════════════════════════════════════════════════════════════════════════════

class BaseModel(nn.Module):
    """
    All three models inherit from this.
    Enforces that every model exposes:
        forward(x)       → class logits (B, NUM_CLASSES)
        get_features(x)  → 256-dim feature vector (B, FEATURE_DIM)

    get_features() is used by:
        - DANN domain classifier
        - t-SNE visualisation
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


# ════════════════════════════════════════════════════════════════════════════
# Squeeze-and-Excitation (SE) Block
# ════════════════════════════════════════════════════════════════════════════

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block for spatial-channel attention.
    Squeeze (GAP) → Excitation (FC-ReLU-FC-Sigmoid) → Scale
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# ════════════════════════════════════════════════════════════════════════════
# Projector Utility
# ════════════════════════════════════════════════════════════════════════════

def build_projector(input_dim: int, dropout_rate: float = None) -> nn.Sequential:
    """
    Builds a flexible projector head with optional BatchNorm and Dropout.
    Structure: FC → [BN] → ReLU → [Dropout]
    """
    layers = [
        nn.Flatten(),
        nn.Linear(input_dim, FEATURE_DIM),
    ]
    
    if BATCHNORM_IN_PROJECTOR:
        layers.append(nn.BatchNorm1d(FEATURE_DIM))
        
    layers.append(nn.ReLU(inplace=True))
    
    # Use explicitly passed dropout_rate if provided, else fall back to config
    dr = dropout_rate if dropout_rate is not None else DROPOUT_RATE
    if dr > 0.0:
        layers.append(nn.Dropout(dr))
        
    return nn.Sequential(*layers)


# ════════════════════════════════════════════════════════════════════════════
# 1. Custom CNN — trained from scratch
# ════════════════════════════════════════════════════════════════════════════

class CustomCNN(BaseModel):
    """
    Architecture:
        Input 224×224×3
        → Conv(32, 3×3) + BN + ReLU + MaxPool(2×2)
        → Conv(64, 3×3) + BN + ReLU + MaxPool(2×2)
        → Conv(128, 3×3) + BN + ReLU + MaxPool(2×2)
        → GlobalAveragePooling
        → FC(256) + ReLU + Dropout(0.5)   ← feature vector
        → FC(NUM_CLASSES)                  ← classifier head

    Uses dataset normalisation (not ImageNet).
    All layers trained at LR_BACKBONE_SCRATCH = 1e-3.
    """

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                         # 224 → 112

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                         # 112 → 56

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                         # 56 → 28
        )

        self.gap = nn.AdaptiveAvgPool2d(1)              # 28 → 1×1

        # Fix 3: CustomCNN must always use Dropout(0.5) regardless of config
        self.projector = build_projector(128, dropout_rate=0.5)

        self.classifier = nn.Linear(FEATURE_DIM, NUM_CLASSES)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.gap(x)
        x = self.projector(x)
        return x                                        # (B, 256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.get_features(x))   # (B, NUM_CLASSES)


# ════════════════════════════════════════════════════════════════════════════
# 2. ResNet18 — pretrained on ImageNet, head replaced
# ════════════════════════════════════════════════════════════════════════════

class ResNet18(BaseModel):
    """
    Backbone  : ResNet18 pretrained on ImageNet (frozen at LR 1e-4)
    Head      : original fc replaced with FC(512→256) + ReLU + FC(256→NUM_CLASSES)
    Features  : 256-dim output of the projection layer

    Why 512 input to projection?
        ResNet18's final average pool outputs 512-dim before the original fc.
        We intercept there and project down to our shared FEATURE_DIM=256.
    """

    def __init__(self):
        super().__init__()

        backbone = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)

        # Remove the original fully connected layer
        # backbone.fc was Linear(512, 1000) — we replace it with identity
        # and attach our own projection + classifier
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        # Output shape after backbone: (B, 512, 1, 1)

        self.projector = build_projector(512)

        self.classifier = nn.Linear(FEATURE_DIM, NUM_CLASSES)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)                           # (B, 512, 1, 1)
        x = self.projector(x)                          # (B, 256)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.get_features(x))   # (B, NUM_CLASSES)


# ════════════════════════════════════════════════════════════════════════════
# 3. EfficientNet-B0 — pretrained on ImageNet, head replaced
# ════════════════════════════════════════════════════════════════════════════

class EfficientNetB0(BaseModel):
    """
    Backbone  : EfficientNet-B0 pretrained on ImageNet (frozen at LR 1e-4)
    Head      : classifier replaced with FC(1280→256) + ReLU + FC(256→NUM_CLASSES)
    Features  : 256-dim output of the projection layer

    Why 1280 input to projection?
        EfficientNet-B0's adaptive avg pool outputs 1280-dim features.
        We intercept after the pooling layer and project to FEATURE_DIM=256.
    """

    def __init__(self):
        super().__init__()

        backbone = tv_models.efficientnet_b0(
            weights=tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )

        # EfficientNet structure:
        #   backbone.features  → conv layers
        #   backbone.avgpool   → adaptive avg pool → (B, 1280, 1, 1)
        #   backbone.classifier → Linear(1280, 1000) — we replace this

        self.features = backbone.features
        self.avgpool  = backbone.avgpool

        self.projector = build_projector(1280)

        self.classifier = nn.Linear(FEATURE_DIM, NUM_CLASSES)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)                           # (B, 1280, 7, 7)
        x = self.avgpool(x)                            # (B, 1280, 1, 1)
        x = self.projector(x)                          # (B, 256)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.get_features(x))   # (B, NUM_CLASSES)


# ════════════════════════════════════════════════════════════════════════════
# 4. ResNet50 — pretrained on ImageNet
# ════════════════════════════════════════════════════════════════════════════

class ResNet50(BaseModel):
    """
    Backbone  : ResNet50 pretrained on ImageNet
    Head      : FC(2048→FEATURE_DIM) + [BN] + ReLU + [Dropout] + FC(FEATURE_DIM→NUM_CLASSES)
    """

    def __init__(self):
        super().__init__()

        backbone = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        # Output shape: (B, 2048, 1, 1)

        self.projector = build_projector(2048)
        self.classifier = nn.Linear(FEATURE_DIM, NUM_CLASSES)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.projector(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.get_features(x))


# ════════════════════════════════════════════════════════════════════════════
# 5. EfficientNet-B3 — pretrained on ImageNet
# ════════════════════════════════════════════════════════════════════════════

class EfficientNetB3(BaseModel):
    """
    Backbone  : EfficientNet-B3 pretrained on ImageNet
    Head      : FC(1536→FEATURE_DIM) + [BN] + ReLU + [Dropout] + FC(FEATURE_DIM→NUM_CLASSES)
    """

    def __init__(self):
        super().__init__()

        backbone = tv_models.efficientnet_b3(
            weights=tv_models.EfficientNet_B3_Weights.IMAGENET1K_V1
        )
        self.features = backbone.features
        self.avgpool  = backbone.avgpool

        self.projector = build_projector(1536)
        self.classifier = nn.Linear(FEATURE_DIM, NUM_CLASSES)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = self.projector(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.get_features(x))


# ════════════════════════════════════════════════════════════════════════════
# Model factory + parameter group builder
# ════════════════════════════════════════════════════════════════════════════

def get_model(model_name: str) -> BaseModel:
    """
    Instantiate and return the correct model.

    Parameters
    ----------
    model_name : "custom_cnn" | "resnet18" | "efficientnet_b0"
    """
    models = {
        "custom_cnn":       CustomCNN,
        "resnet18":         ResNet18,
        "resnet50":         ResNet50,
        "efficientnet_b0":  EfficientNetB0,
        "efficientnet_b3":  EfficientNetB3,
    }
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. "
                         f"Choose from {list(models.keys())}")

    model = models[model_name]()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] {model_name}  |  trainable params: {n_params:,}")
    return model


def get_param_groups(model: BaseModel, model_name: str) -> list[dict]:
    """
    Return parameter groups with correct learning rates for Adam.

    Custom CNN      → single group at LR_BACKBONE_SCRATCH (1e-3)
    ResNet18        → backbone at LR_BACKBONE_PRETRAINED (1e-4)
                      projector + classifier at LR_HEAD (1e-3)
    EfficientNet-B0 → same split as ResNet18

    These are passed directly into the optimizer in train.py:
        optimizer = torch.optim.Adam(get_param_groups(model, model_name))
    """
    from config import LR_HEAD, LR_BACKBONE_PRETRAINED, LR_BACKBONE_SCRATCH

    if model_name == "custom_cnn":
        return [{"params": model.parameters(), "lr": LR_BACKBONE_SCRATCH}]

    elif model_name == "resnet18":
        return [
            {"params": model.backbone.parameters(),   "lr": LR_BACKBONE_PRETRAINED},
            {"params": model.projector.parameters(),  "lr": LR_HEAD},
            {"params": model.classifier.parameters(), "lr": LR_HEAD},
        ]

    elif model_name == "efficientnet_b0":
        return [
            {"params": model.features.parameters(),   "lr": LR_BACKBONE_PRETRAINED},
            {"params": model.avgpool.parameters(),     "lr": LR_BACKBONE_PRETRAINED},
            {"params": model.projector.parameters(),  "lr": LR_HEAD},
            {"params": model.classifier.parameters(), "lr": LR_HEAD},
        ]

    elif model_name == "resnet50":
        return [
            {"params": model.backbone.parameters(),   "lr": LR_BACKBONE_PRETRAINED},
            {"params": model.projector.parameters(),  "lr": LR_HEAD},
            {"params": model.classifier.parameters(), "lr": LR_HEAD},
        ]

    elif model_name == "efficientnet_b3":
        return [
            {"params": model.features.parameters(),   "lr": LR_BACKBONE_PRETRAINED},
            {"params": model.avgpool.parameters(),     "lr": LR_BACKBONE_PRETRAINED},
            {"params": model.projector.parameters(),  "lr": LR_HEAD},
            {"params": model.classifier.parameters(), "lr": LR_HEAD},
        ]

    else:
        raise ValueError(f"Unknown model: {model_name}")

"""A few things to note before we move on:
get_param_groups() — this is important. Without it, Adam would apply the same LR to the entire model. The pretrained backbone needs a gentle 1e-4 to avoid destroying ImageNet weights, while the fresh projection and classifier layers need the full 1e-3 to learn quickly. Custom CNN gets a single uniform group since nothing is pretrained.
get_features() vs forward() — every model separates these cleanly. forward() just calls get_features() then passes through the classifier. DANN and t-SNE call get_features() directly and never touch the classifier head.
EfficientNet avgpool — kept as a separate attribute rather than bundled into self.features so the parameter group split is clean. Both get backbone LR."""