# direction2_idd.py
#
# Direction 2: IDD Few-Shot Transfer Study
#
# Standalone runner that:
#   1. Loads the best BDD checkpoint for each Direction 1 label ratio.
#   2. Replaces the 3-class BDD head with a fresh 5-class IDD head.
#   3. Fine-tunes on K IDD shots (K in DIRECTION2_SHOT_COUNTS).
#   4. Evaluates on the full IDD test set.
#   5. Saves all results to IDD_RESULTS_DIR with clean naming.
#
# ALL backbone hyperparameters (LR, batch size, augmentation, weight decay,
# early stopping) are inherited from config.py unchanged.
# Nothing in config.py, train.py, models.py, or evaluate.py is modified.
#
# Usage:
#   In config.py, set RUN_MODE = "direction2", then run: python main.py

import csv
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

import config
from config import (
    IDD_PROCESSED_ROOT, IDD_CHECKPOINT_DIR, IDD_RESULTS_DIR,
    IDD_CLASSES, IDD_NUM_CLASSES,
    CHECKPOINT_DIR,
    INPUT_SIZE, AUG_PAD_SIZE,
    AUG_FLIP_PROB, AUG_ROTATION_DEGREES, AUG_BRIGHTNESS, AUG_CONTRAST,
    IMAGENET_MEAN, IMAGENET_STD,
    BATCH_SIZE, WEIGHT_DECAY, GRAD_CLIP_NORM,
    MAX_EPOCHS, EARLY_STOP_PATIENCE, LR_PATIENCE, LR_SCHEDULER,
    DIRECTION2_BDD_RATIOS, DIRECTION2_SHOT_COUNTS, DIRECTION2_SEEDS,
    DIRECTION2_MODEL_NAME,
)
from models import get_model, get_param_groups


# -- IDD Class Index Map =======================================================-
IDD_CLASS_TO_IDX = {cls: i for i, cls in enumerate(sorted(IDD_CLASSES))}
# autorickshaw=0, bus=1, car=2, motorcycle=3, truck=4


# ==============================================================================
# IDD Dataset - K-Shot aware loader
# ==============================================================================

class IDDDataset(Dataset):
    """
    Reads IDD crops from idd_final/day/[split]/[class]/images/.
    Mirrors the VehicleDataset interface exactly (returns img, class_label,
    domain_label, idx) so it is compatible with the existing train_standard()
    and evaluate() functions without any modification.

    Parameters
    ----------
    split   : "train" | "val" | "test"
    k_shot  : if split == "train", sample exactly k images per class.
              if None, use all available images.
    seed    : random seed used for k-shot sampling (reproducible).
    augment : True for training split only.
    """

    def __init__(
        self,
        split:   str,
        k_shot:  int  = None,
        seed:    int  = 42,
        augment: bool = False,
    ):
        assert split in ("train", "val", "test"), f"Invalid split: {split}"
        self.split   = split
        self.k_shot  = k_shot
        self.seed    = seed
        self.augment = augment

        self.transform = self._build_transform()
        self.records   = self._load_records()

        print(f"[IDDDataset] split={split:<6}  k_shot={k_shot}  "
              f"seed={seed}  samples={len(self.records)}")

    def _build_transform(self) -> T.Compose:
        """Identical augmentation logic to VehicleDataset._build_transform(day)."""
        mean, std = IMAGENET_MEAN, IMAGENET_STD
        normalise  = T.Normalize(mean=mean, std=std)

        if self.augment:
            return T.Compose([
                T.Resize((AUG_PAD_SIZE, AUG_PAD_SIZE)),
                T.RandomCrop(INPUT_SIZE),
                T.RandomHorizontalFlip(p=AUG_FLIP_PROB),
                T.RandomRotation(degrees=AUG_ROTATION_DEGREES),
                T.ColorJitter(brightness=AUG_BRIGHTNESS, contrast=AUG_CONTRAST),
                T.ToTensor(),
                normalise,
            ])
        else:
            return T.Compose([
                T.Resize((INPUT_SIZE, INPUT_SIZE)),
                T.ToTensor(),
                normalise,
            ])

    def _load_records(self) -> list[dict]:
        records = []
        rng = random.Random(self.seed)

        for category in sorted(IDD_CLASSES):
            images_dir = IDD_PROCESSED_ROOT / "day" / self.split / category / "images"
            if not images_dir.exists():
                print(f"[WARN] IDDDataset: Directory not found: {images_dir}")
                continue

            files = sorted(images_dir.glob("*.jpg"))

            # K-shot sampling: pick exactly k images per class for train split
            if self.split == "train" and self.k_shot is not None:
                files = list(files)
                rng.shuffle(files)
                files = files[:self.k_shot]

            for img_path in files:
                records.append({
                    "_img_path":    img_path,
                    "_class_label": IDD_CLASS_TO_IDX[category],
                    "_domain_label": 0,  # IDD is treated as "day" domain
                    "category":     category,
                })

        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        img = Image.open(rec["_img_path"]).convert("RGB")
        img = self.transform(img)
        return img, rec["_class_label"], rec["_domain_label"], idx


def get_idd_loaders(k_shot: int, seed: int) -> dict:
    """
    Build IDD DataLoaders for a given K-shot experiment.

    Returns dict with keys: "train" (k_shot images/class),
    "val" (all 50/class), "test" (all ~234/class).
    """
    train_ds = IDDDataset(split="train", k_shot=k_shot, seed=seed, augment=True)
    val_ds   = IDDDataset(split="val",   k_shot=None,   seed=seed, augment=False)
    test_ds  = IDDDataset(split="test",  k_shot=None,   seed=seed, augment=False)

    num_workers = 2
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return {"train": train_loader, "val": val_loader, "test": test_loader}


# ==============================================================================
# Checkpoint Loading - BDD backbone + fresh IDD head
# ==============================================================================

def load_bdd_checkpoint_with_idd_head(
    model_name: str,
    ratio_tag: str,
    seed: int,
) -> nn.Module:
    """
    Load the best BDD checkpoint for (model_name, ratio_tag, seed),
    then replace the 3-class BDD head with a fresh 5-class IDD head.

    The backbone (features, avgpool, projector) weights are preserved exactly.
    Only the final Linear(FEATURE_DIM -> NUM_CLASSES) is replaced.
    """

    # For the Supervised FT and Source Only baselines, we look in the datasetB_final folder
    # For DANN ratios, we look in the label_sweep folder
    if ratio_tag in ["finetune", "source_only"]:
        search_dir = CHECKPOINT_DIR.parent / "checkpoints_datasetB_final"
    else:
        search_dir = CHECKPOINT_DIR

    ckpt_path = search_dir / f"{model_name}_{ratio_tag}_seed{seed}_best.pth"

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"BDD checkpoint not found: {ckpt_path}\n"
            f"Check if the file exists in {search_dir}"
        )

    # Create model with original 3-class head to load weights
    model = get_model(model_name)
    ckpt  = torch.load(ckpt_path, map_location="cpu")
    
    state_dict = ckpt["state_dict"]
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("backbone."):
            # DANN checkpoints prefix the backbone with "backbone."
            new_state_dict[k[len("backbone."):]] = v
        else:
            new_state_dict[k] = v

    # Load with strict=False because the DANN checkpoint also contains 
    # domain_classifier weights which the standard model does not have.
    model.load_state_dict(new_state_dict, strict=False)
    print(f"  [Dir2] Loaded BDD weights (stripped prefix) from: {ckpt_path.name}")

    # Replace the classifier head for 5 IDD classes
    from config import FEATURE_DIM
    model.classifier = nn.Linear(FEATURE_DIM, IDD_NUM_CLASSES)
    nn.init.xavier_uniform_(model.classifier.weight)
    nn.init.zeros_(model.classifier.bias)
    print(f"  [Dir2] Replaced classifier head: 3 -> {IDD_NUM_CLASSES} classes")

    return model


# ==============================================================================
# Fine-Tuning - reuses train_standard() logic exactly
# ==============================================================================

def finetune_on_idd(
    model:      nn.Module,
    loaders:    dict,
    model_name: str,
    ckpt_tag:   str,
) -> dict:
    """
    Fine-tune a model on IDD K-shot training data.
    Uses the exact same training loop as train_standard() - all
    hyperparameters (LR, weight decay, scheduler, early stopping)
    are inherited from config.py.

    Saves best checkpoint to IDD_CHECKPOINT_DIR.
    Returns history dict.
    """
    import time
    from train import EarlyStopping

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(get_param_groups(model, model_name), weight_decay=WEIGHT_DECAY)

    if LR_SCHEDULER == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=LR_PATIENCE, factor=0.5
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=MAX_EPOCHS, eta_min=1e-6
        )

    scaler     = torch.amp.GradScaler("cuda")
    early_stop = EarlyStopping()

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  [],
    }
    best_val_loss = float("inf")
    IDD_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_save_path = IDD_CHECKPOINT_DIR / f"{ckpt_tag}_best.pth"

    print(f"\n  [Dir2] Fine-tuning: {ckpt_tag}")
    print(f"         Train batches: {len(loaders['train'])}  "
          f"Val batches: {len(loaders['val'])}")

    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()

        # -- Training ========================================================-
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for imgs, labels, _, _ in loaders["train"]:
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                logits = model(imgs)
                loss   = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()

            train_loss    += loss.item() * imgs.size(0)
            train_correct += (logits.argmax(1) == labels).sum().item()
            train_total   += imgs.size(0)

        train_loss /= train_total
        train_acc   = train_correct / train_total

        # -- Validation ======================================================-
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels, _, _ in loaders["val"]:
                imgs   = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with torch.amp.autocast("cuda"):
                    logits = model(imgs)
                    loss   = criterion(logits, labels)
                val_loss    += loss.item() * imgs.size(0)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total   += imgs.size(0)

        val_loss /= val_total
        val_acc   = val_correct / val_total

        elapsed = time.time() - t0
        print(f"    Epoch {epoch:>3}/{MAX_EPOCHS} | "
              f"train loss: {train_loss:.4f}  acc: {train_acc:.4f} | "
              f"val loss: {val_loss:.4f}  acc: {val_acc:.4f} | "
              f"time: {elapsed:.1f}s")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"epoch": epoch, "val_loss": val_loss,
                        "state_dict": model.state_dict()}, ckpt_save_path)

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

        if early_stop.step(val_loss):
            print(f"    [EarlyStop] Stopped at epoch {epoch}.")
            break

    print(f"  [Dir2] Best val loss: {best_val_loss:.4f}  -> {ckpt_save_path.name}")
    return history


# ==============================================================================
# Evaluation on IDD Test Set
# ==============================================================================

def evaluate_idd(model: nn.Module, test_loader: DataLoader, ckpt_path: Path) -> dict:
    """
    Load best checkpoint, evaluate on IDD test set.
    Returns accuracy and per-class F1 scores.
    """
    from sklearn.metrics import f1_score, accuracy_score

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load best saved checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels, _, _ in test_loader:
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.amp.autocast("cuda"):
                logits = model(imgs)
            preds = logits.argmax(1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    accuracy  = accuracy_score(all_labels, all_preds)
    macro_f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)

    results = {
        "accuracy":  accuracy,
        "macro_f1":  macro_f1,
    }
    for i, cls in enumerate(sorted(IDD_CLASSES)):
        results[f"{cls}_f1"] = float(per_class[i]) if i < len(per_class) else 0.0

    return results


# ==============================================================================
# Main Sweep Runner
# ==============================================================================

def run_direction2() -> None:
    """
    Overnight sweep runner for Direction 2: IDD Few-Shot Transfer.

    Grid:
        BDD ratios  : DIRECTION2_BDD_RATIOS  (6 values)
        IDD shots   : DIRECTION2_SHOT_COUNTS (5 values)
        Seeds       : DIRECTION2_SEEDS       (3 values)
    Total runs: 6 x 5 x 3 = 90 runs (+ 1 x 5 x 3 = 15 for finetune = 105)
    """
    import random as _random
    import numpy as np
    import torch as _torch

    print(f"\n{'=' * 72}")
    print("  DIRECTION 2: IDD FEW-SHOT TRANSFER STUDY")
    print(f"{'=' * 72}")
    print(f"  BDD ratios  : {DIRECTION2_BDD_RATIOS}")
    print(f"  Shot counts : {DIRECTION2_SHOT_COUNTS}")
    print(f"  Seeds       : {DIRECTION2_SEEDS}")
    print(f"  Output dir  : {IDD_RESULTS_DIR}")
    print(f"{'=' * 72}\n")

    IDD_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    IDD_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    results_rows = []

    # -- Define the set of BDD backbones to test -------------------------------
    # FOR THIS RUN: We are ONLY running the 'source_only' (Pure Day) baseline.
    backbones_to_test = [
        {
            "ratio_val": "source_only",
            "ratio_tag": "source_only",
            "display":   "Source Only"
        }
    ]

    for bb in backbones_to_test:
        ratio_val = bb["ratio_val"]
        ratio_tag = bb["ratio_tag"]

        for k_shot in DIRECTION2_SHOT_COUNTS:
            for seed in DIRECTION2_SEEDS:

                print(f"\n{'-' * 60}")
                print(f"  Backbone={bb['display']}  |  k_shot={k_shot}  |  seed={seed}")
                print(f"{'-' * 60}")

                # -- Reproducibility --------------------------------------------
                _random.seed(seed)
                np.random.seed(seed)
                _torch.manual_seed(seed)
                if _torch.cuda.is_available():
                    _torch.cuda.manual_seed(seed)

                # -- Build unique tag for this run ------------------------------
                ckpt_tag = (f"{DIRECTION2_MODEL_NAME}_idd_{ratio_tag}"
                            f"_shot{k_shot:02d}_seed{seed}")

                ckpt_save_path = IDD_CHECKPOINT_DIR / f"{ckpt_tag}_best.pth"

                try:
                    # -- Load BDD backbone + fresh IDD head --------------------
                    model = load_bdd_checkpoint_with_idd_head(
                        DIRECTION2_MODEL_NAME, ratio_tag, seed
                    )

                    # -- Get IDD loaders ======================================-
                    loaders = get_idd_loaders(k_shot=k_shot, seed=seed)

                    # -- Fine-tune on IDD =====================================-
                    finetune_on_idd(model, loaders, DIRECTION2_MODEL_NAME, ckpt_tag)

                    # -- Evaluate on IDD test set =============================-
                    results = evaluate_idd(model, loaders["test"], ckpt_save_path)

                    print(f"  [Result] accuracy={results['accuracy']:.4f}  "
                          f"macro_f1={results['macro_f1']:.4f}")

                    results_rows.append({
                        "bdd_ratio":   ratio_val,
                        "k_shot":      k_shot,
                        "seed":        seed,
                        "accuracy":    results["accuracy"],
                        "macro_f1":    results["macro_f1"],
                        **{k: v for k, v in results.items()
                           if k not in ("accuracy", "macro_f1")},
                    })

                except FileNotFoundError as e:
                    print(f"  [SKIP] {e}")
                    continue
                except Exception as e:
                    import traceback
                    print(f"  [ERROR] {e}")
                    traceback.print_exc()
                    continue

    # -- Write full results CSV ------------------------------------------------
    if results_rows:
        csv_path = IDD_RESULTS_DIR / "direction2_source_only_baseline_raw.csv"
        fieldnames = list(results_rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results_rows)
        print(f"\n[Dir2] Raw baseline results saved to: {csv_path}")

        # -- Aggregate mean/std across seeds =================================-
        _aggregate_direction2_results(results_rows)

    print(f"\n{'=' * 72}")
    print("  DIRECTION 2 SWEEP COMPLETE")
    print(f"  Results: {IDD_RESULTS_DIR}")
    print(f"{'=' * 72}")


def _aggregate_direction2_results(rows: list[dict]) -> None:
    """
    Aggregate per-seed results into mean   std CSV.
    Groups by (bdd_ratio, k_shot) and computes statistics across seeds.
    """
    import numpy as np
    from collections import defaultdict

    grouped = defaultdict(list)
    for row in rows:
        key = (row["bdd_ratio"], row["k_shot"])
        grouped[key].append(row)

    summary_rows = []
    for (ratio, k_shot), group_rows in sorted(grouped.items()):
        accs = [r["accuracy"] for r in group_rows]
        f1s  = [r["macro_f1"] for r in group_rows]
        summary_rows.append({
            "bdd_ratio":     ratio,
            "k_shot":        k_shot,
            "n_seeds":       len(group_rows),
            "accuracy_mean": round(float(np.mean(accs)), 6),
            "accuracy_std":  round(float(np.std(accs)),  6),
            "macro_f1_mean": round(float(np.mean(f1s)),  6),
            "macro_f1_std":  round(float(np.std(f1s)),   6),
        })

    csv_path = IDD_RESULTS_DIR / "direction2_source_only_baseline_summary.csv"
    fieldnames = list(summary_rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"[Dir2] Summary (mean std) saved to: {csv_path}")
    print(f"\n  {'bdd_ratio':<15} {'k_shot':<8} {'accuracy_mean':<16} {'macro_f1_mean'}")
    for row in summary_rows:
        # Handle both float ratios and string tags like 'source_only'
        r_val = row['bdd_ratio']
        r_str = f"{r_val:.2f}" if isinstance(r_val, (int, float)) else str(r_val)
        print(f"  {r_str:<15} {row['k_shot']:<8} "
              f"{row['accuracy_mean']:<16.4f} {row['macro_f1_mean']:.4f}")
