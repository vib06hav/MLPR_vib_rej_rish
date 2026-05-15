from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch
from torchvision.ops import box_iou


def compute_average_precision(recalls: np.ndarray, precisions: np.ndarray) -> float:
    if recalls.size == 0 or precisions.size == 0:
        return 0.0
    recall_points = np.linspace(0.0, 1.0, 101)
    precision_interp = np.zeros_like(recall_points)
    for idx, point in enumerate(recall_points):
        mask = recalls >= point
        precision_interp[idx] = precisions[mask].max() if np.any(mask) else 0.0
    return float(precision_interp.mean())


def evaluate_class_at_iou(
    predictions: list[dict[str, torch.Tensor]],
    targets: list[dict[str, torch.Tensor]],
    class_id: int,
    iou_threshold: float,
) -> float:
    gt_by_image: dict[int, torch.Tensor] = {}
    gt_count = 0
    for target in targets:
        mask = target["labels"] == class_id
        image_id = int(target["image_id"].item())
        boxes = target["boxes"][mask]
        gt_by_image[image_id] = boxes
        gt_count += int(mask.sum().item())

    if gt_count == 0:
        return 0.0

    detections: list[tuple[int, float, torch.Tensor]] = []
    for pred, target in zip(predictions, targets):
        image_id = int(target["image_id"].item())
        mask = pred["labels"] == class_id
        boxes = pred["boxes"][mask]
        scores = pred["scores"][mask]
        for box, score in zip(boxes, scores):
            detections.append((image_id, float(score.item()), box))

    detections.sort(key=lambda item: item[1], reverse=True)

    matched = {
        image_id: torch.zeros((boxes.shape[0],), dtype=torch.bool)
        for image_id, boxes in gt_by_image.items()
    }
    tps: list[float] = []
    fps: list[float] = []

    for image_id, _, pred_box in detections:
        gt_boxes = gt_by_image.get(image_id)
        if gt_boxes is None or gt_boxes.numel() == 0:
            tps.append(0.0)
            fps.append(1.0)
            continue

        ious = box_iou(pred_box.unsqueeze(0), gt_boxes).squeeze(0)
        best_iou, best_idx = (ious.max(dim=0) if ious.numel() > 0 else (torch.tensor(0.0), torch.tensor(0)))
        if best_iou.item() >= iou_threshold and not matched[image_id][best_idx]:
            matched[image_id][best_idx] = True
            tps.append(1.0)
            fps.append(0.0)
        else:
            tps.append(0.0)
            fps.append(1.0)

    if not tps:
        return 0.0

    tps_np = np.cumsum(np.asarray(tps, dtype=np.float32))
    fps_np = np.cumsum(np.asarray(fps, dtype=np.float32))
    recalls = tps_np / max(gt_count, 1)
    precisions = tps_np / np.maximum(tps_np + fps_np, 1e-8)
    return compute_average_precision(recalls, precisions)


def summarize_detection_metrics(
    predictions: list[dict[str, torch.Tensor]],
    targets: list[dict[str, torch.Tensor]],
    class_names: list[str] | tuple[str, ...],
    iou_thresholds: list[float] | None = None,
) -> dict[str, Any]:
    if iou_thresholds is None:
        iou_thresholds = [round(x, 2) for x in np.arange(0.5, 0.96, 0.05)]

    per_class = defaultdict(dict)
    map_values: list[float] = []
    ap50_values: list[float] = []

    for class_id, class_name in enumerate(class_names, start=1):
        class_aps: list[float] = []
        ap50 = 0.0
        for threshold in iou_thresholds:
            ap = evaluate_class_at_iou(predictions, targets, class_id, threshold)
            per_class[class_name][f"AP@{threshold:.2f}"] = ap
            class_aps.append(ap)
            if abs(threshold - 0.5) < 1e-9:
                ap50 = ap
        class_map = float(np.mean(class_aps)) if class_aps else 0.0
        per_class[class_name]["mAP@0.50:0.95"] = class_map
        per_class[class_name]["AP@0.50"] = ap50
        map_values.append(class_map)
        ap50_values.append(ap50)

    return {
        "mAP@0.50:0.95": float(np.mean(map_values)) if map_values else 0.0,
        "mAP@0.50": float(np.mean(ap50_values)) if ap50_values else 0.0,
        "per_class": dict(per_class),
        "iou_thresholds": iou_thresholds,
        "num_images": len(targets),
    }
