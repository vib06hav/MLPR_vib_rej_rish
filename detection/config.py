from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import config as legacy_config


@dataclass
class DetectionConfig:
    image_root: Path = legacy_config.DATASET_ROOT / "bdd100k" / "bdd100k" / "images" / "100k"
    labels_root: Path = legacy_config.DATASET_ROOT / "bdd100k_labels_release" / "bdd100k" / "labels"
    train_annotations: Path = legacy_config.DATASET_ROOT / "bdd100k_labels_release" / "bdd100k" / "labels" / "bdd100k_labels_images_train.json"
    val_annotations: Path = legacy_config.DATASET_ROOT / "bdd100k_labels_release" / "bdd100k" / "labels" / "bdd100k_labels_images_val.json"
    dann_checkpoint: Path = legacy_config.DATASET_ROOT / "checkpoints_datasetB_final" / "efficientnet_b0_dann_warmstart_seed42_best.pth"
    output_root: Path = legacy_config.DATASET_ROOT / "retinanet_detection"
    checkpoints_dir: Path = legacy_config.DATASET_ROOT / "retinanet_detection" / "checkpoints"
    results_dir: Path = legacy_config.DATASET_ROOT / "retinanet_detection" / "results"
    verification_dir: Path = legacy_config.DATASET_ROOT / "retinanet_detection" / "verification"
    cache_dir: Path = legacy_config.DATASET_ROOT / "retinanet_detection" / "cache"
    class_names: tuple[str, ...] = tuple(legacy_config.CLASSES)
    image_size: int = 512
    batch_size: int = 2
    num_workers: int = 0
    device: str = "cuda"
    train_domain: str = "all"
    val_domain: str = "night"
    eval_domain: str = "night"
    backbone_init: str = "imagenet"
    head_lr: float = 1e-4
    backbone_lr: float = 1e-5
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    phase1_patience: int = 3
    phase1_max_epochs: int = 12
    phase2_patience: int = 5
    phase2_max_epochs: int = 20
    bn_warmup_batches: int = 100
    bn_warmup_momentum: float = 0.01
    score_thresh: float = 0.05
    nms_thresh: float = 0.5
    detections_per_img: int = 300
    unfreeze_blocks: tuple[int, ...] = (6, 7, 8)
    image_mean: tuple[float, float, float] = tuple(legacy_config.IMAGENET_MEAN)
    image_std: tuple[float, float, float] = tuple(legacy_config.IMAGENET_STD)
    include_empty_train: bool = False
    include_empty_eval: bool = True
    use_dawn_dusk_as_day: bool = bool(getattr(legacy_config, "FOLD_DAWN_DUSK", True))
    horizontal_flip_prob: float = 0.5
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def ensure_output_dirs(self) -> None:
        for path in (
            self.output_root,
            self.checkpoints_dir,
            self.results_dir,
            self.verification_dir,
            self.cache_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key, value in list(data.items()):
            if isinstance(value, Path):
                data[key] = str(value)
            elif isinstance(value, tuple):
                data[key] = list(value)
        return data

    @classmethod
    def from_args(cls, **kwargs: Any) -> "DetectionConfig":
        cfg = cls()
        for key, value in kwargs.items():
            if value is None or not hasattr(cfg, key):
                continue
            current = getattr(cfg, key)
            if isinstance(current, Path):
                setattr(cfg, key, Path(value))
            elif isinstance(current, tuple) and not isinstance(value, tuple):
                setattr(cfg, key, tuple(value))
            else:
                setattr(cfg, key, value)
        return cfg
