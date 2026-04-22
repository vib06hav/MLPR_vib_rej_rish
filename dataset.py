# dataset.py

import json
import random
from collections import Counter
from pathlib import Path
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from config import (
    PROCESSED_ROOT, CLASSES, SPLITS,
    INPUT_SIZE, AUG_PAD_SIZE,
    AUG_FLIP_PROB, AUG_ROTATION_DEGREES,
    AUG_BRIGHTNESS, AUG_CONTRAST,
    AUG_GRAYSCALE, AUG_GAUSSIAN_BLUR, AUG_TRANSLATE,
    IMAGENET_MEAN, IMAGENET_STD,
    DATASET_MEAN, DATASET_STD,
    BATCH_SIZE, SEEDS, UNLABELED_CLASS_INDEX,
)
import config

# ── Label maps ────────────────────────────────────────────────────────────────
CLASS_TO_IDX  = {cls: i for i, cls in enumerate(sorted(CLASSES))}
# bus→0, car→1, truck→2

DOMAIN_TO_IDX = {"day": 0, "night": 1}


# ── Worker seed function — must be top-level for Windows pickling ─────────────
def seed_worker(worker_id: int) -> None:
    current_seed = SEEDS[config.ACTIVE_SEED_INDEX]
    worker_seed = current_seed + worker_id
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def apply_target_label_ratio(
    records: list[dict],
    ratio: float,
    seed: int,
) -> list[dict]:
    """
    Mark a deterministic, class-balanced subset of target-domain samples
    as labelled for Direction 1.

    Unlabelled samples keep their images and domain labels for domain loss,
    but their class label is replaced with UNLABELED_CLASS_INDEX so the
    trainer can exclude them from target classification loss.
    """
    ratio = max(0.0, min(1.0, float(ratio)))
    updated_records = [dict(record) for record in records]

    if ratio >= 1.0:
        for record in updated_records:
            record["_is_target_labeled"] = True
        return updated_records

    rng = random.Random(seed)
    indices_by_class = {}
    for idx, record in enumerate(updated_records):
        indices_by_class.setdefault(record["category"], []).append(idx)

    labelled_indices = set()
    for category, indices in indices_by_class.items():
        shuffled = indices[:]
        rng.shuffle(shuffled)

        keep_count = int(round(len(indices) * ratio))
        if ratio > 0.0 and keep_count == 0:
            keep_count = 1

        selected = shuffled[:keep_count]
        labelled_indices.update(selected)
        print(
            f"[Dataset] Target labels kept for {category:<5}: "
            f"{len(selected):>4}/{len(indices)} ({ratio:.0%})"
        )

    for idx, record in enumerate(updated_records):
        is_labelled = idx in labelled_indices
        record["_is_target_labeled"] = is_labelled
        if not is_labelled:
            record["_class_label"] = UNLABELED_CLASS_INDEX

    total_labelled = sum(record["_is_target_labeled"] for record in updated_records)
    print(
        f"[Dataset] Applied target label ratio {ratio:.0%}: "
        f"{total_labelled}/{len(updated_records)} labelled night samples"
    )
    return updated_records


# ════════════════════════════════════════════════════════════════════════════
# Core Dataset Class
# ════════════════════════════════════════════════════════════════════════════

class VehicleDataset(Dataset):
    """
    Reads vehicle crops from processed_dataset using metadata.json files.

    Each item returns:
        image        : transformed PIL image as tensor (C x H x W)
        class_label  : int  — bus=0, car=1, truck=2
        domain_label : int  — day=0, night=1

    Parameters
    ----------
    domains  : list of domains to include, e.g. ["day"] or ["day", "night"]
    split    : "train", "val", or "test"
    norm     : "imagenet" for pretrained models, "dataset" for custom CNN
    augment  : True only for training split
    """

    def __init__(
        self,
        domains:  list[str],
        split:    str,
        norm:     str  = "imagenet",
        augment:  bool = False,
        weather_include: list[str] = None,
        scene_include:   list[str] = None,
    ):
        assert split  in SPLITS,              f"Invalid split: {split}"
        assert norm   in ("imagenet", "dataset"), f"Invalid norm: {norm}"

        self.domains = domains
        self.split   = split
        self.norm    = norm
        self.augment = augment
        self.weather_include = weather_include
        self.scene_include   = scene_include

        # Build transform pipelines for both domains
        self.transform_day = self._build_transform(domain="day")
        self.transform_night = self._build_transform(domain="night")

        # Load all records from metadata.json files
        self.records = self._load_records()

        print(f"[Dataset] split={split:<6}  domains={domains}  "
              f"norm={norm:<8}  augment={augment}  "
              f"samples={len(self.records)}")

    # ── Record loading ────────────────────────────────────────────────────────
    
    def _load_records(self) -> list[dict]:
        """
        Walk the actual disk directory for (domain, split, category)
        and collect all images. Metadata is used for auxiliary info
        (weather/scene) but not for locating files.
        """
        records = []
        
        for domain in self.domains:
            for category in sorted(CLASSES):
                images_dir = (
                    PROCESSED_ROOT / domain / self.split / category / "images"
                )
                meta_path = (
                    PROCESSED_ROOT / domain / self.split / category / "metadata.json"
                )

                if not images_dir.exists():
                    print(f"[WARN] Images directory not found: {images_dir}")
                    continue

                # Load metadata lookup once per folder
                meta_lookup = {}
                if meta_path.exists():
                    try:
                        with meta_path.open("r", encoding="utf-8") as f:
                            meta_entries = json.load(f)
                        meta_lookup = {e["crop_name"]: e for e in meta_entries}
                    except Exception as e:
                        print(f"[WARN] Failed to load metadata {meta_path}: {e}")

                # Collect all actual images on disk
                image_files = list(images_dir.glob("*.jpg"))
                for img_path in image_files:
                    crop_name = img_path.name
                    entry = meta_lookup.get(crop_name, {})
                    
                    # ── Filtering (Experiment 1) ───────────────────────────────────────
                    if self.weather_include and entry.get("weather") not in self.weather_include:
                        continue
                    if self.scene_include and entry.get("scene") not in self.scene_include:
                        continue

                    # Create standard record
                    record = {
                        "crop_name":      crop_name,
                        "domain":         domain,
                        "split":          self.split,
                        "category":       category,
                        "_img_path":      img_path,
                        "_class_label":   CLASS_TO_IDX[category],
                        "_domain_label":  DOMAIN_TO_IDX[domain],
                    }
                    # Merge existing metadata fields (attributes, weather, etc)
                    for k, v in entry.items():
                        if k not in record:
                            record[k] = v
                    
                    records.append(record)

        if self.weather_include or self.scene_include:
            print(f"[Dataset] Filter active: weather={self.weather_include} scene={self.scene_include}")
            print(f"          Records remaining: {len(records)}")
            class_counts = Counter(record["category"] for record in records)
            print(f"          Per-class counts: {dict(class_counts)}")

        return records

    # ── Transform builder ─────────────────────────────────────────────────────
    def _build_transform(self, domain: str = "day") -> T.Compose:
        """
        Train split  → augmentation + normalisation
        Val/test     → resize + normalisation only

        Normalisation uses ImageNet stats for pretrained models,
        dataset stats for the custom CNN.
        """
        mean, std = (
            (IMAGENET_MEAN, IMAGENET_STD)
            if self.norm == "imagenet"
            else (DATASET_MEAN, DATASET_STD)
        )

        normalise = T.Normalize(mean=mean, std=std)

        if self.augment:
            # 1. Original augmentation pipeline (must be preserved exactly)
            transforms = [
                T.Resize((AUG_PAD_SIZE, AUG_PAD_SIZE)),
                T.RandomCrop(INPUT_SIZE),
                T.RandomHorizontalFlip(p=AUG_FLIP_PROB),
                T.RandomRotation(degrees=AUG_ROTATION_DEGREES),
                T.ColorJitter(
                    brightness=AUG_BRIGHTNESS,
                    contrast=AUG_CONTRAST,
                ),
            ]

            # 2. New strictly additive augmentations (only applied if True)
            if AUG_GRAYSCALE:
                transforms.append(T.RandomGrayscale(p=0.1))
            if AUG_GAUSSIAN_BLUR:
                transforms.append(T.GaussianBlur(kernel_size=3))
            if AUG_TRANSLATE:
                transforms.append(T.RandomAffine(degrees=0, translate=(0.1, 0.1)))

            # Asymmetric augmentation for night data
            if getattr(config, "ASYMMETRIC_AUGMENTATION", False) and domain == "night":
                # Stronger jitter and specific night blurring for scarce domain yield
                transforms.append(T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1))
                transforms.append(T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)))

            # 3. Terminate with conversion and normalisation
            transforms.extend([
                T.ToTensor(),
                normalise,
            ])
            return T.Compose(transforms)
        else:
            return T.Compose([
                T.Resize((INPUT_SIZE, INPUT_SIZE)),
                T.ToTensor(),
                normalise,
            ])

    # ── Dataset protocol ──────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int, int]:
        rec = self.records[idx]

        # Load image — crops are already 224×224 JPEGs on disk
        img = Image.open(rec["_img_path"]).convert("RGB")
        
        # Asymmetric logic: apply domain-specific transform
        if rec["domain"] == "night":
            img = self.transform_night(img)
        else:
            img = self.transform_day(img)

        return img, rec["_class_label"], rec["_domain_label"], idx


# ════════════════════════════════════════════════════════════════════════════
# DataLoader Factory
# Returns the right loaders for each experiment type
# ════════════════════════════════════════════════════════════════════════════

def get_loaders(
    experiment: str,
    model_name: str,
) -> dict[str, DataLoader]:
    """
    Build and return all DataLoaders needed for a given experiment.

    Experiment → domains used per split
    ─────────────────────────────────────────────────────────
    target_only  : train=night, val=night, test=night
    source_only  : train=day,   val=day,   test=night
    finetune     : train=night, val=night, test=night
                   (fine-tune starts from source_only checkpoint)
    dann         : train=day+night, val=night, test=night

    Note on finetune:
        The finetune experiment loads the source_only checkpoint in
        main.py before training — dataset here is the same as target_only.

    Parameters
    ----------
    experiment : one of target_only | source_only | finetune | dann
    model_name : one of custom_cnn | resnet18 | efficientnet_b0
                 determines which normalisation stats to use

    Returns
    -------
    dict with keys: "train", "val", "test"
    DANN also gets "train_day" and "train_night" for the domain
    discriminator step — see dann.py for how these are used.
    """

    # Normalisation strategy depends on model
    norm = "dataset" if model_name == "custom_cnn" else "imagenet"

    # ── Domain assignments per experiment ─────────────────────────────────────
    if experiment == "target_only":
        train_domains = ["night"]
        val_domains   = ["night"]

    elif experiment == "source_only":
        train_domains = ["day"]
        val_domains   = ["day"]

    elif experiment == "finetune":
        # Same data as target_only — main.py loads source checkpoint first
        train_domains = ["night"]
        val_domains   = ["night"]

    elif experiment == "dann":
        train_domains = ["day", "night"]
        val_domains   = ["night"]

    else:
        raise ValueError(f"Unknown experiment: {experiment}")

    # Test is always night — never evaluated on day
    test_domains = ["night"]

    # ── Experiment 1 Filters ──────────────────────────────────────────────────
    weather_filter = None
    scene_filter   = None
    
    if config.EXP1_ACTIVE:
        weather_filter = config.SOURCE_WEATHER_INCLUDE
        scene_filter   = config.SOURCE_SCENE_INCLUDE

    # ── Build datasets ────────────────────────────────────────────────────────
    # Filters only apply to SOURCE (day) TRAINING domain
    train_ds = VehicleDataset(
        train_domains, "train", norm=norm, augment=True,
        weather_include=weather_filter if "day" in train_domains else None,
        scene_include=scene_filter     if "day" in train_domains else None
    )
    val_ds   = VehicleDataset(val_domains,   "val",   norm=norm, augment=False)
    test_ds  = VehicleDataset(test_domains,  "test",  norm=norm, augment=False)

    # Shared worker init for reproducibility
    current_seed = SEEDS[config.ACTIVE_SEED_INDEX]
    g = torch.Generator()
    g.manual_seed(current_seed)

    # ── Build loaders ─────────────────────────────────────────────────────────
    loaders = {
        "train": DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
            worker_init_fn=seed_worker,
            generator=g,
        ),
        "val": DataLoader(
            val_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
        ),
        "test": DataLoader(
            test_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
        ),
    }

    # ── DANN needs separate day and night loaders for domain loss ─────────────
    # The DANN trainer zips these together so it always has one batch
    # from each domain per step
    if experiment == "dann":
        day_ds   = VehicleDataset(
            ["day"], "train", norm=norm, augment=True,
            weather_include=weather_filter,
            scene_include=scene_filter
        )
        night_ds = VehicleDataset(["night"], "train", norm=norm, augment=True)
        if config.EXP1_ACTIVE is True and len(night_ds.records) > len(day_ds.records):
            random.Random(current_seed).shuffle(night_ds.records)
            night_ds.records = night_ds.records[:len(day_ds.records)]
            print(f"[Dataset] Truncated DANN night train set to {len(night_ds.records)} samples to match filtered day set.")

        target_label_ratio = max(0.0, min(1.0, float(config.DANN_TARGET_LABEL_RATIO)))
        if target_label_ratio < 1.0:
            night_ds.records = apply_target_label_ratio(
                night_ds.records,
                ratio=target_label_ratio,
                seed=current_seed,
            )

        loaders["train_day"] = DataLoader(
            day_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
            worker_init_fn=seed_worker,
            generator=g,
        )
        loaders["train_night"] = DataLoader(
            night_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
            worker_init_fn=seed_worker,
            generator=g,
        )
        viz_ds = VehicleDataset(
            ["day", "night"],
            "val",
            norm=norm,
            augment=False,
        )
        loaders["viz"] = DataLoader(
            viz_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
        )

    # ── Weighted Random Sampler support ───────────────────────────────────────
    from config import USE_WEIGHTED_SAMPLER
    if USE_WEIGHTED_SAMPLER:
        print("[Dataset] Using WeightedRandomSampler for training ...")
        
        # 1. Get class labels for all training samples
        train_labels = [r["_class_label"] for r in train_ds.records]
        
        # 2. Compute class counts
        class_counts = np.bincount(train_labels, minlength=len(CLASSES))
        
        # 3. Compute weights for each class
        # weight = 1.0 / count
        class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float32)
        
        # 4. Assign weight to each sample
        sample_weights = class_weights[train_labels]
        
        # 5. Create sampler
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        
        # 6. Update train loader to use sampler
        loaders["train"] = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            sampler=sampler,
            shuffle=False,      # shuffle must be False with sampler
            num_workers=2,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g,
        )
    train_class_counts = Counter(record["_class_label"] for record in train_ds.records)
    print(
        f"[Loader] train class counts: "
        f"bus={train_class_counts[CLASS_TO_IDX['bus']]} "
        f"car={train_class_counts[CLASS_TO_IDX['car']]} "
        f"truck={train_class_counts[CLASS_TO_IDX['truck']]}"
    )

    return loaders




# ════════════════════════════════════════════════════════════════════════════
# Dataset Mean / Std Calculator
# Run once, paste values into config.py DATASET_MEAN / DATASET_STD
# ════════════════════════════════════════════════════════════════════════════

def compute_dataset_stats() -> tuple[list, list]:
    """
    Compute per-channel mean and std over the entire day+night training set.
    Uses raw pixel values (ToTensor only, no normalisation).

    Run this once:
        from dataset import compute_dataset_stats
        mean, std = compute_dataset_stats()
        print(mean, std)

    Then paste the values into config.py.
    """
    ds = VehicleDataset(
        domains=["day", "night"],
        split="train",
        norm="imagenet",   # norm doesn't matter — we override transform below
        augment=False,
    )

    # Override transform to raw tensor only
    ds.transform_day = T.Compose([
        T.Resize((INPUT_SIZE, INPUT_SIZE)),
        T.ToTensor(),
    ])
    ds.transform_night = ds.transform_day

    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)

    mean = torch.zeros(3)
    std  = torch.zeros(3)
    n_samples = 0

    print("[Stats] Computing dataset mean and std ...")
    for imgs, _, _, _ in loader:
        # imgs: (B, C, H, W)
        batch_size = imgs.size(0)
        imgs = imgs.view(batch_size, 3, -1)          # (B, C, H*W)
        mean += imgs.mean(dim=[0, 2]) * batch_size
        std  += imgs.std(dim=[0, 2])  * batch_size
        n_samples += batch_size

    mean /= n_samples
    std  /= n_samples

    mean_list = mean.tolist()
    std_list  = std.tolist()

    print(f"[Stats] DATASET_MEAN = {[round(v, 4) for v in mean_list]}")
    print(f"[Stats] DATASET_STD  = {[round(v, 4) for v in std_list]}")
    print("[Stats] Paste these into config.py")

    return mean_list, std_list
