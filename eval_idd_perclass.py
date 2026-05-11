# eval_idd_perclass.py
#
# Targeted evaluation script — loads every saved IDD checkpoint across all
# three experimental arms and computes per-class accuracy + confusion matrix.
#
# Arms covered:
#   1. source_only  (15 checkpoints) — checkpoints_direction2_idd/
#   2. finetune     (15 checkpoints) — checkpoints_direction2_idd_ft/
#   3. label_sweep  (90 checkpoints) — checkpoints_direction2_idd_labelsweep/
#
# Outputs (all written to idd-20k-II root):
#   idd_perclass_accuracy_raw.csv     — one row per checkpoint, per-class accuracy
#   idd_perclass_accuracy_summary.csv — mean ± std over seeds, per (arm, ratio, k_shot)
#   idd_confusion_matrices.txt        — readable text dump of every confusion matrix
#
# Usage:
#   python eval_idd_perclass.py
#   (run from the model Experiment directory, no config changes needed)

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# ── Paths ─────────────────────────────────────────────────────────────────────
IDD_ROOT        = Path(r"C:\Users\vibha\Downloads\idd-20k-II")
IDD_FINAL       = IDD_ROOT / "idd_final"

CKPT_SOURCE     = IDD_ROOT / "checkpoints_direction2_idd"
CKPT_FINETUNE   = IDD_ROOT / "checkpoints_direction2_idd_ft"
CKPT_SWEEP      = IDD_ROOT / "checkpoints_direction2_idd_labelsweep"

OUT_RAW         = IDD_ROOT / "idd_perclass_accuracy_raw.csv"
OUT_SUMMARY     = IDD_ROOT / "idd_perclass_accuracy_summary.csv"
OUT_CONFMAT     = IDD_ROOT / "idd_confusion_matrices.txt"

# ── IDD class order (sorted = alphabetical, matching training) ─────────────────
IDD_CLASSES     = sorted(["autorickshaw", "bus", "car", "motorcycle", "truck"])
IDD_CLASS_TO_IDX = {cls: i for i, cls in enumerate(IDD_CLASSES)}
NUM_IDD_CLASSES = len(IDD_CLASSES)   # 5

# ── ImageNet normalisation (same as BDD pipeline) ─────────────────────────────
IMAGENET_MEAN   = [0.485, 0.456, 0.406]
IMAGENET_STD    = [0.229, 0.224, 0.225]
INPUT_SIZE      = 224

# ── Model parameters (must match training) ────────────────────────────────────
FEATURE_DIM     = 512   # projector output before final classifier


# ════════════════════════════════════════════════════════════════════════════
# IDD Test Dataset (read-only, no k-shot sampling needed)
# ════════════════════════════════════════════════════════════════════════════

class IDDTestDataset(Dataset):
    def __init__(self):
        self.transform = T.Compose([
            T.Resize((INPUT_SIZE, INPUT_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        self.records = []
        for category in IDD_CLASSES:
            img_dir = IDD_FINAL / "day" / "test" / category / "images"
            if not img_dir.exists():
                print(f"[WARN] Missing: {img_dir}")
                continue
            for img_path in sorted(img_dir.glob("*.jpg")):
                self.records.append({
                    "_img_path":    img_path,
                    "_class_label": IDD_CLASS_TO_IDX[category],
                    "category":     category,
                })
        print(f"[Dataset] IDD test set: {len(self.records)} images "
              f"across {NUM_IDD_CLASSES} classes")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = Image.open(rec["_img_path"]).convert("RGB")
        return self.transform(img), rec["_class_label"]


# ════════════════════════════════════════════════════════════════════════════
# Model builder — EfficientNet-B0 with 5-class IDD head
# ════════════════════════════════════════════════════════════════════════════

def build_idd_model() -> nn.Module:
    """
    Build EfficientNet-B0 with the 5-class IDD head.
    Architecture matches direction2_idd.py exactly.
    """
    import torchvision.models as tv_models

    backbone = tv_models.efficientnet_b0(
        weights=None   # weights will be loaded from checkpoint
    )

    # Replicate the EfficientNetB0 class from models.py
    class _IDD_EfficientNetB0(nn.Module):
        def __init__(self, b):
            super().__init__()
            self.features  = b.features
            self.avgpool   = b.avgpool
            self.projector = nn.Sequential(
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

    return _IDD_EfficientNetB0(backbone)


# ════════════════════════════════════════════════════════════════════════════
# Checkpoint loader — handles both standard and DANN state_dict prefixes
# ════════════════════════════════════════════════════════════════════════════

def load_checkpoint(model: nn.Module, ckpt_path: Path) -> nn.Module:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["state_dict"]

    # Strip "backbone." prefix if present (DANN checkpoints)
    stripped = {}
    for k, v in state.items():
        if k.startswith("backbone."):
            stripped[k[len("backbone."):]] = v
        else:
            stripped[k] = v

    # strict=False: ignores domain_classifier keys not present in IDD model
    missing, unexpected = model.load_state_dict(stripped, strict=False)
    if unexpected:
        # Only warn on non-domain-classifier unexpected keys
        real_unexpected = [k for k in unexpected if "domain_classifier" not in k]
        if real_unexpected:
            print(f"  [WARN] Unexpected keys: {real_unexpected[:3]}...")
    return model


# ════════════════════════════════════════════════════════════════════════════
# Evaluation — returns per-class accuracy + confusion matrix
# ════════════════════════════════════════════════════════════════════════════

def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    n = NUM_IDD_CLASSES
    conf_mat = np.zeros((n, n), dtype=int)

    with torch.no_grad():
        for imgs, labels in loader:
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.amp.autocast("cuda"):
                logits = model(imgs)
            preds = logits.argmax(1).cpu().numpy()
            lbls  = labels.cpu().numpy()
            for t, p in zip(lbls, preds):
                conf_mat[t][p] += 1

    # Per-class accuracy = diagonal / row sum
    row_sums = conf_mat.sum(axis=1)
    per_class_acc = {
        IDD_CLASSES[i]: float(conf_mat[i][i]) / float(row_sums[i])
        if row_sums[i] > 0 else 0.0
        for i in range(n)
    }
    overall_acc = float(np.trace(conf_mat)) / float(conf_mat.sum())

    return {
        "overall_accuracy": overall_acc,
        "per_class_accuracy": per_class_acc,
        "confusion_matrix": conf_mat,
    }


# ════════════════════════════════════════════════════════════════════════════
# Checkpoint discovery — parse filename to extract metadata
# ════════════════════════════════════════════════════════════════════════════

def parse_checkpoint(ckpt_path: Path, arm: str) -> dict | None:
    """
    Parse filename into (arm, bdd_ratio, k_shot, seed).
    Patterns:
      source_only: efficientnet_b0_idd_source_only_shot{K}_seed{S}_best.pth
      finetune:    efficientnet_b0_idd_finetune_shot{K}_seed{S}_best.pth
      sweep:       efficientnet_b0_idd_ratio_{R}_shot{K}_seed{S}_best.pth
    """
    name = ckpt_path.stem  # strip .pth

    if arm == "source_only":
        m = re.search(r"shot(\d+)_seed(\d+)", name)
        if not m:
            return None
        return {
            "arm":       "source_only",
            "bdd_ratio": "source_only",
            "k_shot":    int(m.group(1)),
            "seed":      int(m.group(2)),
            "ckpt_path": ckpt_path,
        }

    elif arm == "finetune":
        m = re.search(r"shot(\d+)_seed(\d+)", name)
        if not m:
            return None
        return {
            "arm":       "finetune",
            "bdd_ratio": "finetune",
            "k_shot":    int(m.group(1)),
            "seed":      int(m.group(2)),
            "ckpt_path": ckpt_path,
        }

    elif arm == "sweep":
        m = re.search(r"ratio_(\d+)_shot(\d+)_seed(\d+)", name)
        if not m:
            return None
        ratio_int = int(m.group(1))   # 000, 005, 025, 050, 075, 100
        ratio_val = ratio_int / 100.0
        return {
            "arm":       "sweep",
            "bdd_ratio": ratio_val,
            "k_shot":    int(m.group(2)),
            "seed":      int(m.group(3)),
            "ckpt_path": ckpt_path,
        }

    return None


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Eval] Device: {device}")

    # Build test loader once — shared across all checkpoints
    test_ds = IDDTestDataset()
    test_loader = DataLoader(
        test_ds, batch_size=64, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    # Collect all checkpoints
    all_checkpoints = []
    for ckpt in sorted(CKPT_SOURCE.glob("*.pth")):
        parsed = parse_checkpoint(ckpt, "source_only")
        if parsed:
            all_checkpoints.append(parsed)
    for ckpt in sorted(CKPT_FINETUNE.glob("*.pth")):
        parsed = parse_checkpoint(ckpt, "finetune")
        if parsed:
            all_checkpoints.append(parsed)
    for ckpt in sorted(CKPT_SWEEP.glob("*.pth")):
        parsed = parse_checkpoint(ckpt, "sweep")
        if parsed:
            all_checkpoints.append(parsed)

    total = len(all_checkpoints)
    print(f"[Eval] Found {total} checkpoints to evaluate")

    raw_rows = []
    confmat_lines = []

    for i, meta in enumerate(all_checkpoints, 1):
        arm       = meta["arm"]
        bdd_ratio = meta["bdd_ratio"]
        k_shot    = meta["k_shot"]
        seed      = meta["seed"]
        ckpt_path = meta["ckpt_path"]

        print(f"\n[{i:>3}/{total}]  arm={arm:<12}  ratio={str(bdd_ratio):<12}  "
              f"k={k_shot:<4}  seed={seed}  |  {ckpt_path.name}")

        # Build fresh model and load checkpoint
        model = build_idd_model().to(device)
        model = load_checkpoint(model, ckpt_path)

        # Evaluate
        results = evaluate(model, test_loader, device)

        overall    = results["overall_accuracy"]
        per_class  = results["per_class_accuracy"]
        conf_mat   = results["confusion_matrix"]

        print(f"  Overall acc: {overall:.4f}")
        for cls in IDD_CLASSES:
            print(f"  {cls:<14}: {per_class[cls]:.4f}")

        # Build raw row
        row = {
            "arm":              arm,
            "bdd_ratio":        bdd_ratio,
            "k_shot":           k_shot,
            "seed":             seed,
            "overall_accuracy": round(overall, 6),
        }
        for cls in IDD_CLASSES:
            row[f"{cls}_acc"] = round(per_class[cls], 6)
        raw_rows.append(row)

        # Confusion matrix entry
        confmat_lines.append(
            f"\n{'='*70}\n"
            f"arm={arm}  ratio={bdd_ratio}  k_shot={k_shot}  seed={seed}\n"
            f"ckpt: {ckpt_path.name}\n"
            f"Overall accuracy: {overall:.4f}\n\n"
            f"Confusion matrix (rows=true, cols=pred):\n"
            f"Classes: {IDD_CLASSES}\n"
        )
        # Header row
        col_w = 14
        header = " " * col_w + "".join(f"{c:>{col_w}}" for c in IDD_CLASSES)
        confmat_lines.append(header + "\n")
        for r_idx, row_cls in enumerate(IDD_CLASSES):
            row_str = f"{row_cls:<{col_w}}"
            for c_idx in range(NUM_IDD_CLASSES):
                row_str += f"{conf_mat[r_idx][c_idx]:>{col_w}}"
            confmat_lines.append(row_str + "\n")

        # Free GPU memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Write raw CSV ──────────────────────────────────────────────────────────
    fieldnames = ["arm", "bdd_ratio", "k_shot", "seed", "overall_accuracy"] + \
                 [f"{cls}_acc" for cls in IDD_CLASSES]
    with open(OUT_RAW, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(raw_rows)
    print(f"\n[Done] Raw results → {OUT_RAW}")

    # ── Write summary CSV (mean ± std over seeds) ──────────────────────────────
    grouped = defaultdict(list)
    for row in raw_rows:
        key = (row["arm"], str(row["bdd_ratio"]), row["k_shot"])
        grouped[key].append(row)

    summary_rows = []
    for (arm, ratio, k_shot), group in sorted(grouped.items()):
        s_row = {"arm": arm, "bdd_ratio": ratio, "k_shot": k_shot,
                 "n_seeds": len(group)}
        # Overall accuracy
        accs = [r["overall_accuracy"] for r in group]
        s_row["overall_acc_mean"] = round(float(np.mean(accs)), 6)
        s_row["overall_acc_std"]  = round(float(np.std(accs)),  6)
        # Per-class accuracy
        for cls in IDD_CLASSES:
            vals = [r[f"{cls}_acc"] for r in group]
            s_row[f"{cls}_acc_mean"] = round(float(np.mean(vals)), 6)
            s_row[f"{cls}_acc_std"]  = round(float(np.std(vals)),  6)
        summary_rows.append(s_row)

    summary_fields = ["arm", "bdd_ratio", "k_shot", "n_seeds",
                      "overall_acc_mean", "overall_acc_std"] + \
                     [f"{cls}_acc_mean" for cls in IDD_CLASSES] + \
                     [f"{cls}_acc_std"  for cls in IDD_CLASSES]
    with open(OUT_SUMMARY, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"[Done] Summary results → {OUT_SUMMARY}")

    # ── Write confusion matrix text file ──────────────────────────────────────
    with open(OUT_CONFMAT, "w", encoding="utf-8") as f:
        f.writelines(confmat_lines)
    print(f"[Done] Confusion matrices → {OUT_CONFMAT}")

    # ── Print summary table to console ────────────────────────────────────────
    print(f"\n{'='*90}")
    print(f"  SUMMARY — Per-Class Accuracy (mean over seeds)")
    print(f"{'='*90}")
    header = f"{'arm':<14} {'ratio':<12} {'K':<6} {'overall':<10}" + \
             "".join(f"{cls[:5]:>10}" for cls in IDD_CLASSES)
    print(header)
    print("-" * 90)
    for row in summary_rows:
        per_cls = "".join(f"{row[f'{cls}_acc_mean']:>10.4f}" for cls in IDD_CLASSES)
        print(f"{row['arm']:<14} {str(row['bdd_ratio']):<12} {row['k_shot']:<6} "
              f"{row['overall_acc_mean']:<10.4f}{per_cls}")


if __name__ == "__main__":
    main()
