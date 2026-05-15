from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from detection.backbone import build_retinanet_model
from detection.config import DetectionConfig
from detection.dataset import build_detection_dataset
from detection.engine import (
    build_optimizer,
    collate_fn,
    evaluate_model,
    freeze_backbone_features,
    run_bn_warmup,
    save_json,
    set_frozen_backbone_bn_eval,
    train_one_epoch,
    unfreeze_backbone_blocks,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Manual RetinaNet training entrypoint for BDD100K vehicle detection.")
    parser.add_argument("--init-mode", choices=["imagenet", "dann"], default="imagenet")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--train-domain", choices=["all", "day", "night"], default=None)
    parser.add_argument("--val-domain", choices=["all", "day", "night"], default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--phase1-max-epochs", type=int, default=None)
    parser.add_argument("--phase2-max-epochs", type=int, default=None)
    parser.add_argument("--phase1-patience", type=int, default=None)
    parser.add_argument("--phase2-patience", type=int, default=None)
    parser.add_argument("--head-lr", type=float, default=None)
    parser.add_argument("--backbone-lr", type=float, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--skip-bn-warmup", action="store_true")
    return parser.parse_args()


def run_phase(model, cfg, device, train_loader, val_loader, phase_name, max_epochs, patience, backbone_lr):
    best_metric = float("-inf")
    best_state = None
    epochs_without_improvement = 0
    history = []

    if phase_name == "phase1":
        freeze_backbone_features(model)
        frozen_bn_blocks: tuple[int, ...] = tuple()
    else:
        unfreeze_backbone_blocks(model, cfg.unfreeze_blocks)
        frozen_bn_blocks = cfg.unfreeze_blocks

    optimizer = build_optimizer(
        model,
        head_lr=cfg.head_lr,
        backbone_lr=backbone_lr,
        weight_decay=cfg.weight_decay,
    )

    for epoch in range(1, max_epochs + 1):
        if phase_name == "phase1":
            freeze_backbone_features(model)
            frozen_bn_blocks = tuple()
        else:
            unfreeze_backbone_blocks(model, cfg.unfreeze_blocks)
            frozen_bn_blocks = cfg.unfreeze_blocks

        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            device=device,
            grad_clip_norm=cfg.grad_clip_norm,
            frozen_bn_blocks=frozen_bn_blocks,
        )
        val_metrics = evaluate_model(
            model,
            val_loader,
            device=device,
            class_names=cfg.class_names,
        )
        monitor_metric = val_metrics["mAP@0.50:0.95"]
        history.append(
            {
                "phase": phase_name,
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
                "monitor_metric": monitor_metric,
            }
        )

        if monitor_metric > best_metric:
            best_metric = monitor_metric
            best_state = {
                "epoch": epoch,
                "phase": phase_name,
                "monitor_metric": monitor_metric,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state["model_state_dict"])
    return history, best_state


def main():
    args = parse_args()
    cfg = DetectionConfig.from_args(
        backbone_init=args.init_mode,
        device=args.device,
        train_domain=args.train_domain,
        val_domain=args.val_domain,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        phase1_max_epochs=args.phase1_max_epochs,
        phase2_max_epochs=args.phase2_max_epochs,
        phase1_patience=args.phase1_patience,
        phase2_patience=args.phase2_patience,
        head_lr=args.head_lr,
        backbone_lr=args.backbone_lr,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
    )
    cfg.ensure_output_dirs()
    run_name = args.run_name or f"retinanet_{args.init_mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = cfg.checkpoints_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    device_name = cfg.device if (cfg.device == "cpu" or torch.cuda.is_available()) else "cpu"
    device = torch.device(device_name)

    train_dataset = build_detection_dataset(
        cfg=cfg,
        annotation_split="train",
        domain_filter=cfg.train_domain,
        training=True,
        include_empty=cfg.include_empty_train,
        max_samples=cfg.max_train_samples,
    )
    val_dataset = build_detection_dataset(
        cfg=cfg,
        annotation_split="val",
        domain_filter=cfg.val_domain,
        training=False,
        include_empty=cfg.include_empty_eval,
        max_samples=cfg.max_eval_samples,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
    )

    model = build_retinanet_model(cfg, init_mode=args.init_mode).to(device)

    bn_warmup_stats = None
    if not args.skip_bn_warmup:
        warmup_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            collate_fn=collate_fn,
        )
        bn_warmup_stats = run_bn_warmup(
            model,
            warmup_loader,
            device=device,
            max_batches=cfg.bn_warmup_batches,
            momentum=cfg.bn_warmup_momentum,
        )

    phase1_history, phase1_best = run_phase(
        model,
        cfg,
        device,
        train_loader,
        val_loader,
        phase_name="phase1",
        max_epochs=cfg.phase1_max_epochs,
        patience=cfg.phase1_patience,
        backbone_lr=None,
    )
    phase2_history, phase2_best = run_phase(
        model,
        cfg,
        device,
        train_loader,
        val_loader,
        phase_name="phase2",
        max_epochs=cfg.phase2_max_epochs,
        patience=cfg.phase2_patience,
        backbone_lr=cfg.backbone_lr,
    )

    final_state = phase2_best or phase1_best
    checkpoint_payload = {
        "run_name": run_name,
        "init_mode": args.init_mode,
        "config": cfg.to_dict(),
        "bn_warmup": bn_warmup_stats,
        "phase1_best": phase1_best,
        "phase2_best": phase2_best,
        "history": phase1_history + phase2_history,
    }
    json_summary = {
        "run_name": run_name,
        "init_mode": args.init_mode,
        "config": cfg.to_dict(),
        "bn_warmup": bn_warmup_stats,
        "phase1_best": {
            "epoch": phase1_best["epoch"],
            "phase": phase1_best["phase"],
            "monitor_metric": phase1_best["monitor_metric"],
        } if phase1_best else None,
        "phase2_best": {
            "epoch": phase2_best["epoch"],
            "phase": phase2_best["phase"],
            "monitor_metric": phase2_best["monitor_metric"],
        } if phase2_best else None,
        "history": phase1_history + phase2_history,
    }
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metadata": checkpoint_payload,
        },
        run_dir / "best_model.pth",
    )
    save_json(run_dir / "training_summary.json", json_summary)
    print(json.dumps({"run_dir": str(run_dir), "best_monitor_metric": final_state["monitor_metric"] if final_state else None}, indent=2))


if __name__ == "__main__":
    main()
