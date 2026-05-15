from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from detection.backbone import build_retinanet_model
from detection.config import DetectionConfig
from detection.dataset import build_detection_dataset
from detection.engine import collate_fn, evaluate_model, save_json


def parse_args():
    parser = argparse.ArgumentParser(description="Manual RetinaNet evaluation entrypoint for BDD100K vehicle detection.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--init-mode", choices=["imagenet", "dann"], default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--domain", choices=["all", "day", "night"], default="night")
    parser.add_argument("--annotation-split", choices=["train", "val"], default="val")
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    metadata = checkpoint.get("metadata", {})
    init_mode = args.init_mode or metadata.get("init_mode", "imagenet")
    cfg = DetectionConfig.from_args(
        device=args.device,
        eval_domain=args.domain,
        max_eval_samples=args.max_eval_samples,
    )
    cfg.ensure_output_dirs()

    device_name = cfg.device if (cfg.device == "cpu" or torch.cuda.is_available()) else "cpu"
    device = torch.device(device_name)

    dataset = build_detection_dataset(
        cfg=cfg,
        annotation_split=args.annotation_split,
        domain_filter=args.domain,
        training=False,
        include_empty=cfg.include_empty_eval,
        max_samples=cfg.max_eval_samples,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
    )

    model = build_retinanet_model(cfg, init_mode=init_mode)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)

    metrics = evaluate_model(
        model,
        loader,
        device=device,
        class_names=cfg.class_names,
        max_batches=args.max_batches,
    )
    output_path = Path(args.output_json) if args.output_json else cfg.results_dir / f"eval_{Path(args.checkpoint).stem}_{args.domain}.json"
    payload = {
        "checkpoint": str(args.checkpoint),
        "init_mode": init_mode,
        "annotation_split": args.annotation_split,
        "domain": args.domain,
        "metrics": metrics,
    }
    save_json(output_path, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
