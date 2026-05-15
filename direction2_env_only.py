# direction2_env_only.py
#
# Direction 2 — Environment-Only Arm (Arm A)
#
# Isolates the pure environment shift component of the IDD transfer study.
# Uses only the three classes shared between BDD100K and IDD:
#   bus, car, truck
#
# This separates the "new environment" challenge from the "novel class"
# challenge present in the full 5-class direction2_idd.py experiment.
#
# Design:
#   - Load same BDD backbone checkpoints as direction2_idd.py
#   - Replace head with fresh Linear(512 → 3) — 3 IDD classes only
#   - K-shot fine-tune on IDD bus/car/truck train images
#   - Evaluate on IDD bus/car/truck test set (234 per class)
#   - Sweep: 8 backbone configs × 5 K-shots × 3 seeds = 120 runs
#
# Outputs → C:\Users\vibha\Downloads\idd-20k-II\results_direction2_env_only\
#
# Usage:
#   python direction2_env_only.py

import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# ── Paths ─────────────────────────────────────────────────────────────────────
IDD_ROOT          = Path(r"C:\Users\vibha\Downloads\idd-20k-II")
IDD_FINAL         = IDD_ROOT / "idd_final"

# BDD backbone checkpoint sources — same as direction2_idd.py
ARCHIVE_ROOT      = Path(r"C:\Users\vibha\Downloads\archive")
CKPT_DATASET_B    = ARCHIVE_ROOT / "checkpoints_datasetB_final"   # source_only, finetune
CKPT_LABEL_SWEEP  = ARCHIVE_ROOT / "checkpoints_51k_label_sweep"  # DANN ratios

# Env-only outputs
CKPT_OUT_DIR      = IDD_ROOT / "checkpoints_direction2_env_only"
RESULTS_DIR       = IDD_ROOT / "results_direction2_env_only"

# ── Experiment settings ───────────────────────────────────────────────────────
# ONLY the 3 classes shared between BDD and IDD
IDD_CLASSES       = sorted(["bus", "car", "truck"])
IDD_CLASS_TO_IDX  = {cls: i for i, cls in enumerate(IDD_CLASSES)}
# bus=0, car=1, truck=2
NUM_IDD_CLASSES   = len(IDD_CLASSES)   # 3

SHOT_COUNTS       = [1, 5, 10, 25, 50]
SEEDS             = [42, 43, 44]
MODEL_NAME        = "efficientnet_b0"

# ── Model architecture constants (must match BDD training) ────────────────────
FEATURE_DIM       = 512
INPUT_SIZE        = 224
AUG_PAD_SIZE      = 256

IMAGENET_MEAN     = [0.485, 0.456, 0.406]
IMAGENET_STD      = [0.229, 0.224, 0.225]

# ── Training hyperparameters (locked — same as BDD training) ─────────────────
BATCH_SIZE        = 32
MAX_EPOCHS        = 50
EARLY_STOP_PAT    = 10
LR_PATIENCE       = 5
LR_FACTOR         = 0.5
WEIGHT_DECAY      = 1e-3
GRAD_CLIP_NORM    = 1.0
BACKBONE_LR       = 2e-4
HEAD_LR           = 2e-3

# ── Backbone configs to sweep ─────────────────────────────────────────────────
# ratio_tag must match the actual checkpoint filename stem exactly.
# source_only and finetune → CKPT_DATASET_B
# direction1_ratio_* → CKPT_LABEL_SWEEP
BACKBONES = [
    {"display": "Source Only",     "tag": "source_only",            "dir": CKPT_DATASET_B},
    {"display": "Finetune",        "tag": "finetune",               "dir": CKPT_DATASET_B},
    {"display": "DANN 0% labels",  "tag": "direction1_ratio_000",   "dir": CKPT_LABEL_SWEEP},
    {"display": "DANN 5% labels",  "tag": "direction1_ratio_005",   "dir": CKPT_LABEL_SWEEP},
    {"display": "DANN 25% labels", "tag": "direction1_ratio_025",   "dir": CKPT_LABEL_SWEEP},
    {"display": "DANN 50% labels", "tag": "direction1_ratio_050",   "dir": CKPT_LABEL_SWEEP},
    {"display": "DANN 75% labels", "tag": "direction1_ratio_075",   "dir": CKPT_LABEL_SWEEP},
    {"display": "DANN 100% labels","tag": "direction1_ratio_100",   "dir": CKPT_LABEL_SWEEP},
]


# ════════════════════════════════════════════════════════════════════════════
# IDD Dataset — bus/car/truck only
# ════════════════════════════════════════════════════════════════════════════

class IDDEnvDataset(Dataset):
    """
    Loads IDD images for the 3 shared classes (bus, car, truck) only.
    Mirrors VehicleDataset interface — returns (img, class_label, domain_label, idx).
    """

    def __init__(self, split: str, k_shot: int = None, seed: int = 42,
                 augment: bool = False):
        assert split in ("train", "val", "test")
        self.split   = split
        self.k_shot  = k_shot
        self.seed    = seed
        self.augment = augment
        self.transform = self._build_transform()
        self.records   = self._load_records()
        print(f"[IDDEnvDataset] split={split:<6}  k_shot={k_shot}  "
              f"seed={seed}  n={len(self.records)}")

    def _build_transform(self) -> T.Compose:
        norm = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        if self.augment:
            return T.Compose([
                T.Resize((AUG_PAD_SIZE, AUG_PAD_SIZE)),
                T.RandomCrop(INPUT_SIZE),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=10),
                T.ColorJitter(brightness=0.2, contrast=0.2),
                T.ToTensor(), norm,
            ])
        return T.Compose([
            T.Resize((INPUT_SIZE, INPUT_SIZE)),
            T.ToTensor(), norm,
        ])

    def _load_records(self) -> list:
        rng     = random.Random(self.seed)
        records = []
        for category in IDD_CLASSES:
            img_dir = IDD_FINAL / "day" / self.split / category / "images"
            if not img_dir.exists():
                print(f"  [WARN] Missing: {img_dir}")
                continue
            files = sorted(img_dir.glob("*.jpg"))
            if self.split == "train" and self.k_shot is not None:
                files = list(files)
                rng.shuffle(files)
                files = files[:self.k_shot]
            for f in files:
                records.append({
                    "_img_path":    f,
                    "_class_label": IDD_CLASS_TO_IDX[category],
                    "_domain_label": 0,
                    "category":     category,
                })
        return records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = Image.open(rec["_img_path"]).convert("RGB")
        return self.transform(img), rec["_class_label"], rec["_domain_label"], idx


def get_env_loaders(k_shot: int, seed: int) -> dict:
    train_ds = IDDEnvDataset("train", k_shot=k_shot, seed=seed, augment=True)
    val_ds   = IDDEnvDataset("val",   k_shot=None,   seed=seed, augment=False)
    test_ds  = IDDEnvDataset("test",  k_shot=None,   seed=seed, augment=False)
    kw = dict(num_workers=2, pin_memory=True)
    return {
        "train": DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  **kw),
        "val":   DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, **kw),
        "test":  DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, **kw),
    }


# ════════════════════════════════════════════════════════════════════════════
# Model — EfficientNet-B0 with fresh 3-class IDD head
# ════════════════════════════════════════════════════════════════════════════

class EfficientNetB0_3Class(nn.Module):
    """EfficientNet-B0 backbone + projector + 3-class head."""
    def __init__(self):
        super().__init__()
        import torchvision.models as tv
        b = tv.efficientnet_b0(weights=None)
        self.features   = b.features
        self.avgpool    = b.avgpool
        self.projector  = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280, FEATURE_DIM),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(FEATURE_DIM, NUM_IDD_CLASSES)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.projector(x)
        return self.classifier(x)


def get_param_groups(model: nn.Module) -> list:
    """Differential LRs: backbone slow, head fast."""
    backbone_params = list(model.features.parameters()) + \
                      list(model.avgpool.parameters())
    head_params     = list(model.projector.parameters()) + \
                      list(model.classifier.parameters())
    return [
        {"params": backbone_params, "lr": BACKBONE_LR},
        {"params": head_params,     "lr": HEAD_LR},
    ]


def load_bdd_backbone(tag: str, ckpt_dir: Path, seed: int) -> nn.Module:
    """
    Load BDD checkpoint, strip DANN prefix if present,
    replace 3-class BDD head with fresh 3-class IDD head.
    (Both are 3-class, but class semantics are re-initialized.)
    """
    ckpt_path = ckpt_dir / f"{MODEL_NAME}_{tag}_seed{seed}_best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"BDD checkpoint not found: {ckpt_path}")

    model = EfficientNetB0_3Class()
    ckpt  = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["state_dict"]

    # Strip "backbone." prefix (DANN checkpoints)
    stripped = {}
    for k, v in state.items():
        stripped[k[len("backbone."):] if k.startswith("backbone.") else k] = v

    # strict=False: ignores domain_classifier keys + allows head mismatch
    model.load_state_dict(stripped, strict=False)

    # Reinitialize the classifier head — IDD class indices differ from BDD
    model.classifier = nn.Linear(FEATURE_DIM, NUM_IDD_CLASSES)
    nn.init.xavier_uniform_(model.classifier.weight)
    nn.init.zeros_(model.classifier.bias)

    print(f"  [Backbone] Loaded: {ckpt_path.name}")
    return model


# ════════════════════════════════════════════════════════════════════════════
# Early Stopping
# ════════════════════════════════════════════════════════════════════════════

class EarlyStopping:
    def __init__(self, patience: int = EARLY_STOP_PAT):
        self.patience    = patience
        self.best_loss   = float("inf")
        self.counter     = 0
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# ════════════════════════════════════════════════════════════════════════════
# Training Loop
# ════════════════════════════════════════════════════════════════════════════

def finetune(model: nn.Module, loaders: dict, ckpt_save_path: Path) -> None:
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(get_param_groups(model), weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=LR_PATIENCE, factor=LR_FACTOR
    )
    scaler     = torch.amp.GradScaler("cuda")
    early_stop = EarlyStopping()
    best_val_loss = float("inf")

    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()

        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        tr_loss = tr_correct = tr_total = 0
        for imgs, labels, _, _ in loaders["train"]:
            imgs, labels = imgs.to(device, non_blocking=True), \
                           labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                logits = model(imgs)
                loss   = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
            tr_loss    += loss.item() * imgs.size(0)
            tr_correct += (logits.argmax(1) == labels).sum().item()
            tr_total   += imgs.size(0)

        # ── Val ────────────────────────────────────────────────────────────
        model.eval()
        vl_loss = vl_correct = vl_total = 0
        with torch.no_grad():
            for imgs, labels, _, _ in loaders["val"]:
                imgs, labels = imgs.to(device, non_blocking=True), \
                               labels.to(device, non_blocking=True)
                with torch.amp.autocast("cuda"):
                    logits = model(imgs)
                    loss   = criterion(logits, labels)
                vl_loss    += loss.item() * imgs.size(0)
                vl_correct += (logits.argmax(1) == labels).sum().item()
                vl_total   += imgs.size(0)

        vl_loss /= vl_total
        scheduler.step(vl_loss)

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            torch.save({"epoch": epoch, "val_loss": vl_loss,
                        "state_dict": model.state_dict()}, ckpt_save_path)

        elapsed = time.time() - t0
        print(f"    Ep {epoch:>3}/{MAX_EPOCHS} | "
              f"tr {tr_correct/tr_total:.4f} | val {vl_correct/vl_total:.4f} | "
              f"vl_loss {vl_loss:.4f} | {elapsed:.1f}s")

        if early_stop.step(vl_loss):
            print(f"    [EarlyStop] Stopped at epoch {epoch}")
            break


# ════════════════════════════════════════════════════════════════════════════
# Evaluation
# ════════════════════════════════════════════════════════════════════════════

def evaluate(model: nn.Module, test_loader: DataLoader,
             ckpt_save_path: Path) -> dict:
    from sklearn.metrics import f1_score, accuracy_score

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(ckpt_save_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model  = model.to(device)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels, _, _ in test_loader:
            imgs = imgs.to(device, non_blocking=True)
            with torch.amp.autocast("cuda"):
                preds = model(imgs).argmax(1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())

    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    per_cls  = f1_score(all_labels, all_preds, average=None, zero_division=0)

    results = {"accuracy": accuracy, "macro_f1": macro_f1}
    for i, cls in enumerate(IDD_CLASSES):
        results[f"{cls}_f1"] = float(per_cls[i]) if i < len(per_cls) else 0.0
    return results


# ════════════════════════════════════════════════════════════════════════════
# Main Sweep
# ════════════════════════════════════════════════════════════════════════════

def run_env_only():
    CKPT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*72}")
    print("  DIRECTION 2 — ENVIRONMENT ONLY ARM (bus / car / truck on IDD)")
    print(f"{'='*72}")
    print(f"  Backbones   : {len(BACKBONES)}")
    print(f"  Shot counts : {SHOT_COUNTS}")
    print(f"  Seeds       : {SEEDS}")
    print(f"  Total runs  : {len(BACKBONES)} × {len(SHOT_COUNTS)} × {len(SEEDS)} = "
          f"{len(BACKBONES) * len(SHOT_COUNTS) * len(SEEDS)}")
    print(f"{'='*72}\n")

    results_rows = []

    for bb in BACKBONES:
        tag      = bb["tag"]
        ckpt_dir = bb["dir"]
        display  = bb["display"]

        for k_shot in SHOT_COUNTS:
            for seed in SEEDS:
                print(f"\n{'-'*60}")
                print(f"  Backbone: {display:<25}  k={k_shot:<4}  seed={seed}")
                print(f"{'-'*60}")

                # ── Reproducibility ─────────────────────────────────────────
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)

                run_tag = (f"{MODEL_NAME}_env_{tag}"
                           f"_shot{k_shot:02d}_seed{seed}")
                ckpt_save = CKPT_OUT_DIR / f"{run_tag}_best.pth"

                try:
                    model   = load_bdd_backbone(tag, ckpt_dir, seed)
                    loaders = get_env_loaders(k_shot=k_shot, seed=seed)

                    finetune(model, loaders, ckpt_save)
                    results = evaluate(model, loaders["test"], ckpt_save)

                    print(f"  [Result] acc={results['accuracy']:.4f}  "
                          f"f1={results['macro_f1']:.4f}")
                    for cls in IDD_CLASSES:
                        print(f"    {cls:<8}: f1={results[f'{cls}_f1']:.4f}")

                    results_rows.append({
                        "backbone":   display,
                        "bdd_tag":    tag,
                        "k_shot":     k_shot,
                        "seed":       seed,
                        "accuracy":   round(results["accuracy"],  6),
                        "macro_f1":   round(results["macro_f1"],  6),
                        **{f"{cls}_f1": round(results[f"{cls}_f1"], 6)
                           for cls in IDD_CLASSES},
                    })

                except FileNotFoundError as e:
                    print(f"  [SKIP] {e}")
                    continue
                except torch.cuda.OutOfMemoryError:
                    print("  [OOM] Skipping this run — reduce BATCH_SIZE if persistent")
                    torch.cuda.empty_cache()
                    continue
                except Exception as e:
                    import traceback
                    print(f"  [ERROR] {e}")
                    traceback.print_exc()
                    continue

    if not results_rows:
        print("[WARN] No results collected.")
        return

    # ── Write raw CSV ──────────────────────────────────────────────────────
    raw_path = RESULTS_DIR / "env_only_raw.csv"
    fields   = list(results_rows[0].keys())
    with open(raw_path, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()
        csv.DictWriter(f, fieldnames=fields).writerows(results_rows)
    print(f"\n[Done] Raw results → {raw_path}")

    # ── Write summary CSV (mean ± std over seeds) ──────────────────────────
    from collections import defaultdict
    grouped = defaultdict(list)
    for row in results_rows:
        grouped[(row["backbone"], row["bdd_tag"], row["k_shot"])].append(row)

    summary_rows = []
    for (backbone, tag, k_shot), grp in sorted(grouped.items()):
        s = {"backbone": backbone, "bdd_tag": tag, "k_shot": k_shot,
             "n_seeds": len(grp)}
        for col in ["accuracy", "macro_f1"] + [f"{c}_f1" for c in IDD_CLASSES]:
            vals = [r[col] for r in grp]
            s[f"{col}_mean"] = round(float(np.mean(vals)), 6)
            s[f"{col}_std"]  = round(float(np.std(vals)),  6)
        summary_rows.append(s)

    summary_path = RESULTS_DIR / "env_only_summary.csv"
    sum_fields   = list(summary_rows[0].keys())
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=sum_fields).writeheader()
        csv.DictWriter(f, fieldnames=sum_fields).writerows(summary_rows)
    print(f"[Done] Summary → {summary_path}")

    # ── Console summary table ──────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  SUMMARY — Accuracy mean over seeds")
    print(f"{'='*80}")
    print(f"  {'Backbone':<26} {'K':<6} {'Overall':<10} "
          f"{'Bus':>8} {'Car':>8} {'Truck':>8}")
    print(f"  {'-'*78}")
    for row in summary_rows:
        print(f"  {row['backbone']:<26} {row['k_shot']:<6} "
              f"{row['accuracy_mean']:<10.4f} "
              f"{row['bus_f1_mean']:>8.4f} "
              f"{row['car_f1_mean']:>8.4f} "
              f"{row['truck_f1_mean']:>8.4f}")

    print(f"\n{'='*72}")
    print("  ENVIRONMENT-ONLY SWEEP COMPLETE")
    print(f"  Results: {RESULTS_DIR}")
    print(f"{'='*72}")


if __name__ == "__main__":
    run_env_only()
