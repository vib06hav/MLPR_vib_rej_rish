from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F

from .config import DetectionConfig

TIMEOFDAY_TO_DOMAIN = {
    "daytime": "day",
    "night": "night",
    "dawn/dusk": "day",
}


@dataclass(frozen=True)
class DetectionRecord:
    image_name: str
    annotation_split: str
    domain: str
    boxes: tuple[tuple[float, float, float, float], ...]
    labels: tuple[int, ...]


def normalize_domain(timeofday: str, use_dawn_dusk_as_day: bool) -> str | None:
    if timeofday == "dawn/dusk" and use_dawn_dusk_as_day:
        return "day"
    return TIMEOFDAY_TO_DOMAIN.get(timeofday)


@lru_cache(maxsize=2)
def load_annotation_entries(annotation_path: str) -> list[dict[str, Any]]:
    with Path(annotation_path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=1)
def build_image_index(image_root: str) -> dict[str, str]:
    root = Path(image_root)
    index: dict[str, str] = {}
    for image_path in root.rglob("*.jpg"):
        index[image_path.name] = str(image_path)
    if not index:
        raise FileNotFoundError(f"No .jpg files found under detection image root: {root}")
    return index


class BDDDetectionDataset(Dataset):
    def __init__(
        self,
        cfg: DetectionConfig,
        annotation_split: str,
        domain_filter: str = "all",
        training: bool = False,
        include_empty: bool = True,
        max_samples: int | None = None,
    ) -> None:
        super().__init__()
        if annotation_split not in {"train", "val"}:
            raise ValueError(f"Unsupported annotation_split: {annotation_split}")
        if domain_filter not in {"all", "day", "night"}:
            raise ValueError(f"Unsupported domain_filter: {domain_filter}")

        self.cfg = cfg
        self.annotation_split = annotation_split
        self.domain_filter = domain_filter
        self.training = training
        self.include_empty = include_empty
        self.max_samples = max_samples
        self.label_to_idx = {name: idx + 1 for idx, name in enumerate(cfg.class_names)}
        self.records = self._build_records()

    def _build_records(self) -> list[DetectionRecord]:
        annotation_path = self.cfg.train_annotations if self.annotation_split == "train" else self.cfg.val_annotations
        entries = load_annotation_entries(str(annotation_path))
        records: list[DetectionRecord] = []

        for entry in entries:
            domain = normalize_domain(
                entry.get("attributes", {}).get("timeofday", ""),
                use_dawn_dusk_as_day=self.cfg.use_dawn_dusk_as_day,
            )
            if domain is None:
                continue
            if self.domain_filter != "all" and domain != self.domain_filter:
                continue

            boxes: list[tuple[float, float, float, float]] = []
            labels: list[int] = []
            for label in entry.get("labels", []):
                category = label.get("category")
                if category not in self.label_to_idx:
                    continue
                box2d = label.get("box2d")
                if not box2d:
                    continue
                x1 = float(box2d["x1"])
                y1 = float(box2d["y1"])
                x2 = float(box2d["x2"])
                y2 = float(box2d["y2"])
                if x2 <= x1 or y2 <= y1:
                    continue
                boxes.append((x1, y1, x2, y2))
                labels.append(self.label_to_idx[category])

            if not boxes and not self.include_empty:
                continue

            records.append(
                DetectionRecord(
                    image_name=entry["name"],
                    annotation_split=self.annotation_split,
                    domain=domain,
                    boxes=tuple(boxes),
                    labels=tuple(labels),
                )
            )

            if self.max_samples is not None and len(records) >= self.max_samples:
                break

        return records

    def __len__(self) -> int:
        return len(self.records)

    def image_path_for(self, idx: int) -> Path:
        record = self.records[idx]
        image_index = build_image_index(str(self.cfg.image_root))
        image_path = image_index.get(record.image_name)
        if image_path is None:
            raise FileNotFoundError(
                f"Image not found in indexed BDD image tree: {record.image_name}"
            )
        return Path(image_path)

    def record_info(self, idx: int) -> dict[str, Any]:
        record = self.records[idx]
        return {
            "image_name": record.image_name,
            "domain": record.domain,
            "annotation_split": record.annotation_split,
            "num_boxes": len(record.boxes),
        }

    def __getitem__(self, idx: int):
        record = self.records[idx]
        image_path = self.image_path_for(idx)
        image = Image.open(image_path).convert("RGB")
        orig_w, orig_h = image.size

        image = F.resize(image, [self.cfg.image_size, self.cfg.image_size])
        image_tensor = F.to_tensor(image)

        if record.boxes:
            boxes = torch.tensor(record.boxes, dtype=torch.float32)
            scale_x = self.cfg.image_size / float(orig_w)
            scale_y = self.cfg.image_size / float(orig_h)
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y
            boxes[:, 0::2].clamp_(0, self.cfg.image_size)
            boxes[:, 1::2].clamp_(0, self.cfg.image_size)
            labels = torch.tensor(record.labels, dtype=torch.int64)
            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)

        if self.training and self.cfg.horizontal_flip_prob > 0.0:
            if torch.rand(1).item() < self.cfg.horizontal_flip_prob:
                image_tensor = torch.flip(image_tensor, dims=[2])
                if boxes.numel() > 0:
                    x1 = boxes[:, 0].clone()
                    x2 = boxes[:, 2].clone()
                    boxes[:, 0] = self.cfg.image_size - x2
                    boxes[:, 2] = self.cfg.image_size - x1

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": area,
            "iscrowd": iscrowd,
        }
        return image_tensor, target


def build_detection_dataset(
    cfg: DetectionConfig,
    annotation_split: str,
    domain_filter: str,
    training: bool,
    include_empty: bool,
    max_samples: int | None = None,
) -> BDDDetectionDataset:
    return BDDDetectionDataset(
        cfg=cfg,
        annotation_split=annotation_split,
        domain_filter=domain_filter,
        training=training,
        include_empty=include_empty,
        max_samples=max_samples,
    )
