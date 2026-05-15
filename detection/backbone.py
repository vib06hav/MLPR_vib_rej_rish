from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
from torchvision.models.detection import RetinaNet
from torchvision.ops import FeaturePyramidNetwork
from torchvision.ops.feature_pyramid_network import LastLevelP6P7

from .config import DetectionConfig

FEATURE_BLOCKS = (3, 5, 7)
FEATURE_CHANNELS = (40, 112, 320)


def extract_dann_feature_state_dict(checkpoint_path: str | Path) -> OrderedDict[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint))
    prefix = "backbone.features."
    filtered: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith(prefix):
            filtered[key[len(prefix) :]] = value
    if not filtered:
        raise ValueError(f"No DANN EfficientNet feature weights found in checkpoint: {checkpoint_path}")
    return filtered


def verify_dann_checkpoint_coverage(checkpoint_path: str | Path) -> dict[str, Any]:
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    expected = model.features.state_dict()
    loaded = extract_dann_feature_state_dict(checkpoint_path)

    missing = sorted(set(expected.keys()) - set(loaded.keys()))
    unexpected = sorted(set(loaded.keys()) - set(expected.keys()))
    if missing or unexpected:
        raise ValueError(
            "DANN checkpoint feature-key mismatch.\n"
            f"Missing keys: {missing[:10]}\nUnexpected keys: {unexpected[:10]}"
        )

    model.features.load_state_dict(loaded, strict=True)
    return {
        "expected_key_count": len(expected),
        "loaded_key_count": len(loaded),
        "sample_keys": list(loaded.keys())[:10],
    }


class EfficientNetB0FPNBackbone(nn.Module):
    def __init__(self, cfg: DetectionConfig, init_mode: str) -> None:
        super().__init__()
        self.cfg = cfg
        self.init_mode = init_mode
        base = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.features = base.features
        if init_mode == "dann":
            state_dict = extract_dann_feature_state_dict(cfg.dann_checkpoint)
            self.features.load_state_dict(state_dict, strict=True)
        elif init_mode != "imagenet":
            raise ValueError(f"Unsupported backbone init mode: {init_mode}")

        self.fpn = FeaturePyramidNetwork(
            in_channels_list=list(FEATURE_CHANNELS),
            out_channels=256,
            extra_blocks=LastLevelP6P7(FEATURE_CHANNELS[-1], 256),
        )
        self.out_channels = 256

    def forward_features(self, x: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        outputs: OrderedDict[str, torch.Tensor] = OrderedDict()
        for idx, block in enumerate(self.features):
            x = block(x)
            if idx in FEATURE_BLOCKS:
                outputs[str(len(outputs))] = x
        return outputs

    def forward(self, x: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        raw = self.forward_features(x)
        return self.fpn(raw)


def build_retinanet_model(cfg: DetectionConfig, init_mode: str) -> RetinaNet:
    backbone = EfficientNetB0FPNBackbone(cfg, init_mode=init_mode)
    model = RetinaNet(
        backbone=backbone,
        num_classes=len(cfg.class_names) + 1,
        min_size=cfg.image_size,
        max_size=cfg.image_size,
        image_mean=list(cfg.image_mean),
        image_std=list(cfg.image_std),
        score_thresh=cfg.score_thresh,
        nms_thresh=cfg.nms_thresh,
        detections_per_img=cfg.detections_per_img,
    )
    return model
