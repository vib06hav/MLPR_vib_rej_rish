from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import summarize_detection_metrics


def collate_fn(batch):
    return tuple(zip(*batch))


def move_targets_to_device(targets, device: torch.device):
    moved = []
    for target in targets:
        moved.append({key: value.to(device) for key, value in target.items()})
    return moved


def freeze_backbone_features(model) -> None:
    for param in model.backbone.features.parameters():
        param.requires_grad = False


def unfreeze_backbone_blocks(model, trainable_blocks: tuple[int, ...]) -> None:
    freeze_backbone_features(model)
    for idx in trainable_blocks:
        if idx < 0 or idx >= len(model.backbone.features):
            continue
        for param in model.backbone.features[idx].parameters():
            param.requires_grad = True


def set_frozen_backbone_bn_eval(model, trainable_blocks: tuple[int, ...]) -> None:
    trainable = set(trainable_blocks)
    for idx, block in enumerate(model.backbone.features):
        if idx in trainable:
            continue
        for module in block.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()


def build_optimizer(model, head_lr: float, backbone_lr: float | None, weight_decay: float):
    param_groups = []
    head_params = []
    backbone_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("backbone.features"):
            backbone_params.append(param)
        else:
            head_params.append(param)

    if head_params:
        param_groups.append({"params": head_params, "lr": head_lr})
    if backbone_params and backbone_lr is not None:
        param_groups.append({"params": backbone_params, "lr": backbone_lr})

    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


def train_one_epoch(
    model,
    data_loader: DataLoader,
    optimizer,
    device: torch.device,
    grad_clip_norm: float,
    frozen_bn_blocks: tuple[int, ...] | None = None,
) -> dict[str, float]:
    model.train()
    if frozen_bn_blocks is not None:
        set_frozen_backbone_bn_eval(model, trainable_blocks=frozen_bn_blocks)
    running = {"loss_total": 0.0, "loss_classifier": 0.0, "loss_box_reg": 0.0}
    num_batches = 0

    for images, targets in tqdm(data_loader, leave=False):
        images = [img.to(device) for img in images]
        targets = move_targets_to_device(targets, device)
        optimizer.zero_grad(set_to_none=True)
        loss_dict = model(images, targets)
        total_loss = sum(loss_dict.values())
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        running["loss_total"] += float(total_loss.item())
        for key, value in loss_dict.items():
            running[key] = running.get(key, 0.0) + float(value.item())
        num_batches += 1

    if num_batches == 0:
        return running

    return {key: value / num_batches for key, value in running.items()}


@torch.no_grad()
def collect_predictions(model, data_loader: DataLoader, device: torch.device, max_batches: int | None = None):
    model.eval()
    predictions = []
    targets_cpu = []
    for batch_idx, (images, targets) in enumerate(tqdm(data_loader, leave=False)):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images_device = [img.to(device) for img in images]
        outputs = model(images_device)
        for pred, target in zip(outputs, targets):
            predictions.append({key: value.detach().cpu() for key, value in pred.items()})
            targets_cpu.append({key: value.detach().cpu() for key, value in target.items()})
    return predictions, targets_cpu


@torch.no_grad()
def evaluate_model(model, data_loader: DataLoader, device: torch.device, class_names, max_batches: int | None = None) -> dict[str, Any]:
    predictions, targets = collect_predictions(model, data_loader, device, max_batches=max_batches)
    return summarize_detection_metrics(predictions, targets, class_names=class_names)


@torch.no_grad()
def run_bn_warmup(model, data_loader: DataLoader, device: torch.device, max_batches: int, momentum: float) -> dict[str, Any]:
    bn_layers = []
    original_momentum = {}
    for module in model.backbone.features.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.reset_running_stats()
            original_momentum[id(module)] = module.momentum
            module.momentum = momentum
            module.train()
            bn_layers.append(module)

    model.backbone.features.train()
    batches_seen = 0
    for images, _ in data_loader:
        if batches_seen >= max_batches:
            break
        batch = torch.stack([img.to(device) for img in images], dim=0)
        model.backbone.forward_features(batch)
        batches_seen += 1

    for module in bn_layers:
        module.momentum = original_momentum[id(module)]

    return {"bn_layers": len(bn_layers), "batches_seen": batches_seen}


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
