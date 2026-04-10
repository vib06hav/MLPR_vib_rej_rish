# dann.py

import torch
import torch.nn as nn
from torch.autograd import Function

from config import (
    FEATURE_DIM, DANN_LAMBDA_MAX, DANN_LAMBDA_GAMMA,
    DANN_LAMBDA_SCHEDULE,
    DOMAIN_CLASSIFIER_DEPTH, CHECKPOINT_DIR, DROPOUT_RATE
)
import config
from models import BaseModel, get_model, get_param_groups


# ════════════════════════════════════════════════════════════════════════════
# 1. Gradient Reversal Layer
# ════════════════════════════════════════════════════════════════════════════

class GradientReversalFunction(Function):
    """
    Forward  : identity — passes input through unchanged
    Backward : multiplies gradient by -lambda

    This is the core trick of DANN. The feature extractor receives
    a reversed gradient from the domain classifier, forcing it to
    produce features that the domain classifier CANNOT distinguish.
    Meanwhile the class classifier receives a normal gradient, so
    the feature extractor still learns to classify vehicles correctly.

    Two objectives pulling on the same feature extractor simultaneously:
        - Minimise classification loss   (normal gradient)
        - Maximise domain loss           (reversed gradient via GRL)
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, lam: float) -> torch.Tensor:
        # Save lambda for use in backward
        ctx.save_for_backward(torch.tensor(lam))
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        lam, = ctx.saved_tensors
        # Flip and scale gradient — None for lam since it has no gradient
        return -lam * grad_output, None


class GradientReversal(nn.Module):
    """
    Thin nn.Module wrapper around GradientReversalFunction.
    Accepts lambda as a parameter so the DANN trainer can
    update it each step via the lambda scheduler.
    """

    def __init__(self, lam: float = 1.0):
        super().__init__()
        self.lam = lam

    def set_lambda(self, lam: float) -> None:
        self.lam = lam

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.lam)


# ════════════════════════════════════════════════════════════════════════════
# 2. Domain Classifier
# ════════════════════════════════════════════════════════════════════════════

class DomainClassifier(nn.Module):
    """
    Takes 256-dim features (after GRL) and predicts domain.

    Architecture (from experiment setup):
        FC(256 → 128) + ReLU
        FC(128 → 1)   + Sigmoid

    Output : scalar in [0, 1]
        0 → day
        1 → night

    Intentionally small — its only job is to distinguish two domains,
    not classify complex patterns. If it were too powerful it would
    overpower the feature extractor rather than guide it.
    """

    def __init__(self):
        super().__init__()

        if DOMAIN_CLASSIFIER_DEPTH == "deep":
            layers = [
                nn.Linear(FEATURE_DIM, 128),
                nn.ReLU(inplace=True),
            ]
            if DROPOUT_RATE > 0.0:
                layers.append(nn.Dropout(DROPOUT_RATE))
            layers.extend([
                nn.Linear(128, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 1)
            ])
            self.net = nn.Sequential(*layers)
        else:
            # "shallow" — existing structure
            self.net = nn.Sequential(
                nn.Linear(FEATURE_DIM, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)                             # (B, 1)


# ════════════════════════════════════════════════════════════════════════════
# 3. DANN Wrapper
# ════════════════════════════════════════════════════════════════════════════

class DANNModel(nn.Module):
    """
    Wraps any of the three backbones with:
        - The same classifier head as the standard model
        - A GradientReversal layer (lambda updated each step)
        - A DomainClassifier

    Data flow:

        Input Image
              ↓
        backbone.get_features()     →   256-dim feature vector
              ↓                                  ↓
        classifier head             GradientReversal (−λ on backward)
        FC(256 → NUM_CLASSES)                    ↓
              ↓                         DomainClassifier
        class logits                    FC(256→128)→FC(128→1)
                                             ↓
                                        domain prob [0,1]

    The feature extractor is pulled in two directions:
        - Toward better class discrimination   (via class loss)
        - Toward domain confusion              (via reversed domain loss)

    Parameters
    ----------
    backbone : instantiated BaseModel (CustomCNN, ResNet18, EfficientNetB0)
    """

    def __init__(self, backbone: BaseModel):
        super().__init__()

        self.backbone          = backbone
        self.grl               = GradientReversal(lam=0.0)   # starts at 0
        self.domain_classifier = DomainClassifier()
        self.semi_supervised   = config.DANN_SEMI_SUPERVISED

        # Classifier head is already part of backbone
        # We call backbone.get_features() + backbone.classifier separately
        # so DANN can branch after the feature vector

    def set_lambda(self, lam: float) -> None:
        """Called by the lambda scheduler each training step."""
        self.grl.set_lambda(lam)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Expose feature vector — used by t-SNE after training."""
        return self.backbone.get_features(x)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        class_logits  : (B, NUM_CLASSES)
        domain_probs  : (B, 1)
        """
        features     = self.backbone.get_features(x)      # (B, 256)
        class_logits = self.backbone.classifier(features) # (B, NUM_CLASSES)

        # GRL reverses gradient flowing back to feature extractor
        reversed_features = self.grl(features)            # (B, 256) forward unchanged
        domain_probs      = self.domain_classifier(reversed_features)  # (B, 1)

        return class_logits, domain_probs


# ════════════════════════════════════════════════════════════════════════════
# 4. Lambda Scheduler
# ════════════════════════════════════════════════════════════════════════════

def get_lambda(step: int, total_steps: int) -> float:
    """
    Core lambda computation based on the configured schedule.
    """
    import math
    p = step / max(total_steps, 1)

    if DANN_LAMBDA_SCHEDULE == "linear":
        return p
    elif DANN_LAMBDA_SCHEDULE == "constant":
        return 1.0
    else:
        # "sigmoid" — default GRL schedule
        return (2.0 / (1.0 + math.exp(-DANN_LAMBDA_GAMMA * p))) - 1.0

def compute_lambda(
    current_step: int,
    total_steps:  int,
) -> float:
    """
    Compute the GRL lambda value for the current training step.
    Wraps get_lambda and applies DANN_LAMBDA_MAX cap.
    """
    lam = get_lambda(current_step, total_steps)
    return min(lam, DANN_LAMBDA_MAX)


# ════════════════════════════════════════════════════════════════════════════
# 5. DANN factory + parameter groups
# ════════════════════════════════════════════════════════════════════════════

def get_dann_model(model_name: str, warmstart: bool | None = None) -> DANNModel:
    """
    Build and return a DANNModel wrapping the specified backbone.

    Parameters
    ----------
    model_name : "custom_cnn" | "resnet18" | "resnet50" | ...
    warmstart  : if True, load source_only weights into backbone
    """
    if warmstart is None:
        warmstart = config.DANN_WARMSTART

    backbone = get_model(model_name)

    if warmstart:
        from config import SEEDS
        current_seed = SEEDS[config.ACTIVE_SEED_INDEX]
        if config.EXP1_ACTIVE:
            path = CHECKPOINT_DIR / f"{model_name}_source_only_{config.EXP1_CONFIG_NAME}_seed{current_seed}_best.pth"
        else:
            path = CHECKPOINT_DIR / f"{model_name}_source_only_seed{current_seed}_best.pth"
        if path.exists():
            ckpt = torch.load(path, map_location="cpu")
            backbone.load_state_dict(ckpt["state_dict"])
            print(f"  [DANN] Warmstart: Loaded source_only weights from {path.name}")
        else:
            print(f"  [Warning] Warmstart requested but {path.name} not found. Starting from scratch.")

    model = DANNModel(backbone)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[DANN]  {model_name}  |  trainable params: {n_params:,}")
    return model


def get_dann_param_groups(model: DANNModel, model_name: str) -> list[dict]:
    """
    Parameter groups for the DANN model optimizer.

    Backbone groups   : same LR split as standard model (from models.py)
    Domain classifier : trained at LR_HEAD (1e-3) — it's a fresh head
    GRL               : no parameters to optimise

    Parameters
    ----------
    model      : DANNModel instance
    model_name : needed to determine backbone LR split
    """
    from config import LR_HEAD

    # Reuse backbone param groups from models.py
    backbone_groups = get_param_groups(model.backbone, model_name)

    # Add domain classifier as a separate group
    domain_group = {
        "params": model.domain_classifier.parameters(),
        "lr":     LR_HEAD,
    }

    return backbone_groups + [domain_group]

"""Why lambda starts at 0 — if you start with a strong domain signal from step one, the feature extractor gets confused before it has learned anything useful about vehicle classification. Starting at 0 and growing smoothly means the model first learns to classify, then gradually learns to also be domain-invariant. This is exactly what the sigmoid growth curve achieves.
forward() returns a tuple — unlike the standard models which return just logits, DANN returns (class_logits, domain_probs). The DANN trainer handles both. The standard trainer never touches this class so there's no confusion.
GRL has no parameters — it's a pure gradient manipulation layer. Nothing to optimise, nothing to initialise. It's invisible in the forward pass and only acts during backprop.
get_features() is exposed on DANNModel too — so t-SNE can call it identically on both standard models and DANN models without any special casing."""
