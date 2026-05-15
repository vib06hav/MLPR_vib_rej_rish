from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch
from torch.utils.data import DataLoader

from .backbone import build_retinanet_model, verify_dann_checkpoint_coverage
from .dataset import build_detection_dataset
from .engine import collate_fn, collect_predictions, evaluate_model, run_bn_warmup


def verify_checkpoint(cfg) -> dict[str, Any]:
    return verify_dann_checkpoint_coverage(cfg.dann_checkpoint)


def verify_feature_taps(cfg) -> dict[str, Any]:
    model = build_retinanet_model(cfg, init_mode="dann")
    dummy = torch.randn(1, 3, cfg.image_size, cfg.image_size)
    raw = model.backbone.forward_features(dummy)
    pyramid = model.backbone(dummy)
    return {
        "raw_feature_shapes": {name: list(tensor.shape) for name, tensor in raw.items()},
        "fpn_feature_shapes": {name: list(tensor.shape) for name, tensor in pyramid.items()},
        "fpn_out_channels": model.backbone.out_channels,
    }


def verify_dataset(cfg) -> dict[str, Any]:
    dataset = build_detection_dataset(
        cfg=cfg,
        annotation_split="val",
        domain_filter="night",
        training=False,
        include_empty=cfg.include_empty_eval,
        max_samples=5,
    )
    samples = []
    for idx in range(min(3, len(dataset))):
        image, target = dataset[idx]
        samples.append(
            {
                "image_shape": list(image.shape),
                "boxes_shape": list(target["boxes"].shape),
                "labels": target["labels"].tolist(),
                "record": dataset.record_info(idx),
            }
        )
    return {"dataset_length": len(dataset), "samples": samples}


def verify_single_batch(cfg, init_mode: str) -> dict[str, Any]:
    dataset = build_detection_dataset(
        cfg=cfg,
        annotation_split="val",
        domain_filter="night",
        training=False,
        include_empty=False,
        max_samples=2,
    )
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_fn)
    images, targets = next(iter(loader))
    model = build_retinanet_model(cfg, init_mode=init_mode)
    model.train()
    loss_dict = model(list(images), list(targets))
    return {key: float(value.item()) for key, value in loss_dict.items()}


def verify_bn_recalibration(cfg, init_mode: str) -> dict[str, Any]:
    dataset = build_detection_dataset(
        cfg=cfg,
        annotation_split="val",
        domain_filter="night",
        training=False,
        include_empty=cfg.include_empty_eval,
        max_samples=4,
    )
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_fn)
    model = build_retinanet_model(cfg, init_mode=init_mode)
    before = []
    for module in model.backbone.features.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            before.append(module.running_mean.detach().clone())
            break
    stats = run_bn_warmup(
        model,
        loader,
        device=torch.device("cpu"),
        max_batches=2,
        momentum=cfg.bn_warmup_momentum,
    )
    after = []
    for module in model.backbone.features.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            after.append(module.running_mean.detach().clone())
            break
    changed = bool(before and after and not torch.equal(before[0], after[0]))
    return {"stats": stats, "running_mean_changed": changed}


def verify_eval_dry_run(cfg, init_mode: str) -> dict[str, Any]:
    dataset = build_detection_dataset(
        cfg=cfg,
        annotation_split="val",
        domain_filter="night",
        training=False,
        include_empty=cfg.include_empty_eval,
        max_samples=3,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)
    model = build_retinanet_model(cfg, init_mode=init_mode)
    metrics = evaluate_model(
        model,
        loader,
        device=torch.device("cpu"),
        class_names=cfg.class_names,
        max_batches=2,
    )
    return metrics


def run_all_verifications(cfg) -> dict[str, Any]:
    return {
        "checkpoint": verify_checkpoint(cfg),
        "features": verify_feature_taps(cfg),
        "dataset": verify_dataset(cfg),
        "single_batch_imagenet": verify_single_batch(cfg, "imagenet"),
        "single_batch_dann": verify_single_batch(cfg, "dann"),
        "bn_imagenet": verify_bn_recalibration(cfg, "imagenet"),
        "bn_dann": verify_bn_recalibration(cfg, "dann"),
        "eval_imagenet": verify_eval_dry_run(cfg, "imagenet"),
        "eval_dann": verify_eval_dry_run(cfg, "dann"),
    }
