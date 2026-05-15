# run_tsne_bdd.py
#
# Standalone t-SNE visualisation script — no config.py changes needed.
#
# Generates three figures saved to:
#   C:\Users\vibha\Downloads\archive\results_datasetB_final\figures\
#
#   1. bdd_tsne_strategy_comparison.png
#      2×5 grid — all 5 BDD strategies (source_only, target_only, finetune,
#      dann_warmstart, semi_dann) coloured by domain (row 0) and class (row 1)
#
#   2. bdd_tsne_domain_alignment.png
#      1×2 — source_only vs dann_warmstart side by side, coloured by domain
#      The headline figure showing domain alignment effect
#
#   3. bdd_tsne_direction1_sweep.png
#      2×7 — DANN at each label ratio (0%, 5%, 10%, 25%, 50%, 75%, 100%)
#      by domain (row 0) and by class (row 1)
#
# Usage:
#   python run_tsne_bdd.py
#   (run from the model Experiment directory — uses venv)

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE

# ── Add project root to path ─────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from models import get_model
from dann import get_dann_model
from dataset import VehicleDataset

# ── Paths (hardcoded — never touches config.py) ───────────────────────────────
ARCHIVE         = Path(r"C:\Users\vibha\Downloads\archive")
PROCESSED_ROOT  = ARCHIVE / "processed_dataset_51k"
CKPT_B          = ARCHIVE / "checkpoints_datasetB_final"
CKPT_SWEEP      = ARCHIVE / "checkpoints_51k_label_sweep"
FIGURES_DIR     = ARCHIVE / "results_datasetB_final" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42  # single seed for visualisation — enough for feature space analysis

# ── Colour palettes ───────────────────────────────────────────────────────────
DOMAIN_COLOURS = {"day": "#F4A261", "night": "#264653"}
CLASS_COLOURS  = {"bus": "#E76F51", "car": "#2A9D8F", "truck": "#E9C46A"}
IDX_TO_CLASS   = {0: "bus", 1: "car", 2: "truck"}
IDX_TO_DOMAIN  = {0: "day", 1: "night"}

# ── Dataset B strategies (seed 42) ────────────────────────────────────────────
STRATEGIES = [
    {"tag": "source_only",    "display": "Source Only",    "is_dann": False, "dir": CKPT_B},
    {"tag": "target_only",    "display": "Target Only",    "is_dann": False, "dir": CKPT_B},
    {"tag": "finetune",       "display": "Finetune",       "is_dann": False, "dir": CKPT_B},
    {"tag": "dann_warmstart", "display": "DANN Warmstart", "is_dann": True,  "dir": CKPT_B},
    {"tag": "semi_dann",      "display": "Semi-DANN",      "is_dann": True,  "dir": CKPT_B},
]

# ── Direction 1 label sweep strategies ───────────────────────────────────────
SWEEP_RATIOS = [
    {"tag": "direction1_ratio_000", "display": "DANN 0%"},
    {"tag": "direction1_ratio_005", "display": "DANN 5%"},
    {"tag": "direction1_ratio_010", "display": "DANN 10%"},
    {"tag": "direction1_ratio_025", "display": "DANN 25%"},
    {"tag": "direction1_ratio_050", "display": "DANN 50%"},
    {"tag": "direction1_ratio_075", "display": "DANN 75%"},
    {"tag": "direction1_ratio_100", "display": "DANN 100%"},
]


# ════════════════════════════════════════════════════════════════════════════
# Checkpoint loader
# ════════════════════════════════════════════════════════════════════════════

def load_model(tag: str, is_dann: bool, ckpt_dir: Path) -> nn.Module:
    ckpt_path = ckpt_dir / f"efficientnet_b0_{tag}_seed{SEED}_best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Not found: {ckpt_path}")

    if is_dann:
        model = get_dann_model("efficientnet_b0")
    else:
        model = get_model("efficientnet_b0")

    ckpt  = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    print(f"  [Loaded] {ckpt_path.name}  (epoch {ckpt.get('epoch', '?')})")
    return model


# ════════════════════════════════════════════════════════════════════════════
# Feature extractor — uses model.get_features()
# ════════════════════════════════════════════════════════════════════════════

def extract_features(model: nn.Module, loader: DataLoader) -> tuple:
    """Returns (features, class_labels, domain_labels) as numpy arrays."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)
    model.eval()

    all_f, all_c, all_d = [], [], []
    with torch.no_grad():
        for imgs, class_labels, domain_labels, _ in loader:
            imgs = imgs.to(device)
            with torch.amp.autocast("cuda"):
                feats = model.get_features(imgs)   # (B, 512)
            all_f.append(feats.cpu().numpy())
            all_c.append(class_labels.numpy())
            all_d.append(domain_labels.numpy())

    features      = np.concatenate(all_f, axis=0)
    class_labels  = np.concatenate(all_c, axis=0)
    domain_labels = np.concatenate(all_d, axis=0)
    print(f"  [Features] {features.shape[0]} samples × {features.shape[1]} dims")
    return features, class_labels, domain_labels


# ════════════════════════════════════════════════════════════════════════════
# t-SNE
# ════════════════════════════════════════════════════════════════════════════

def run_tsne(features: np.ndarray) -> np.ndarray:
    print(f"  [t-SNE] Running on {features.shape[0]} samples...")
    tsne = TSNE(
        n_components=2,
        perplexity=40,
        max_iter=1000,
        random_state=SEED,
        n_jobs=-1,
    )
    emb = tsne.fit_transform(features)
    print(f"  [t-SNE] Done.")
    return emb


# ════════════════════════════════════════════════════════════════════════════
# Scatter plot helpers
# ════════════════════════════════════════════════════════════════════════════

def scatter_by_domain(ax: plt.Axes, emb: np.ndarray,
                      domain_labels: np.ndarray, title: str) -> None:
    for d_idx, d_name in IDX_TO_DOMAIN.items():
        mask = domain_labels == d_idx
        ax.scatter(emb[mask, 0], emb[mask, 1],
                   c=DOMAIN_COLOURS[d_name], s=6, alpha=0.55,
                   label=d_name, rasterized=True)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])


def scatter_by_class(ax: plt.Axes, emb: np.ndarray,
                     class_labels: np.ndarray, title: str) -> None:
    for c_idx, c_name in IDX_TO_CLASS.items():
        mask = class_labels == c_idx
        ax.scatter(emb[mask, 0], emb[mask, 1],
                   c=CLASS_COLOURS[c_name], s=6, alpha=0.55,
                   label=c_name, rasterized=True)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])


def make_legends() -> tuple:
    domain_handles = [
        mpatches.Patch(color=v, label=k) for k, v in DOMAIN_COLOURS.items()
    ]
    class_handles = [
        mpatches.Patch(color=v, label=k) for k, v in CLASS_COLOURS.items()
    ]
    return domain_handles, class_handles


# ════════════════════════════════════════════════════════════════════════════
# Figure 1 — All 5 BDD strategies (2 × 5)
# ════════════════════════════════════════════════════════════════════════════

def figure_strategy_comparison(loader: DataLoader) -> None:
    print(f"\n{'='*60}")
    print("  Figure 1: All-strategy comparison (2×5)")
    print(f"{'='*60}")

    n_cols = len(STRATEGIES)
    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8))
    fig.suptitle(
        "BDD100K Feature Space — t-SNE\nAll Strategies  |  Seed 42  |  Day + Night Test Set",
        fontsize=12, fontweight="bold", y=1.01,
    )

    domain_handles, class_handles = make_legends()

    for col, strat in enumerate(STRATEGIES):
        print(f"\n  [{col+1}/{n_cols}] {strat['display']}")
        try:
            model = load_model(strat["tag"], strat["is_dann"], strat["dir"])
            features, class_labels, domain_labels = extract_features(model, loader)
            emb = run_tsne(features)

            scatter_by_domain(axes[0, col], emb, domain_labels, strat["display"])
            scatter_by_class( axes[1, col], emb, class_labels,  strat["display"])

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
            axes[0, col].set_title(f"{strat['display']}\n(missing)", fontsize=8)
            axes[1, col].set_title(f"{strat['display']}\n(missing)", fontsize=8)

    # Row labels
    axes[0, 0].set_ylabel("Coloured by Domain", fontsize=10, labelpad=8)
    axes[1, 0].set_ylabel("Coloured by Class",  fontsize=10, labelpad=8)

    # Shared legends
    fig.legend(handles=domain_handles, title="Domain",
               loc="lower left",  bbox_to_anchor=(0.01, -0.06), ncol=2, fontsize=9)
    fig.legend(handles=class_handles,  title="Class",
               loc="lower right", bbox_to_anchor=(0.99, -0.06), ncol=3, fontsize=9)

    plt.tight_layout()
    save_path = FIGURES_DIR / "bdd_tsne_strategy_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  [Saved] {save_path}")


# ════════════════════════════════════════════════════════════════════════════
# Figure 2 — Headline: source_only vs DANN (1 × 2, by domain only)
# ════════════════════════════════════════════════════════════════════════════

def figure_domain_alignment(loader: DataLoader) -> None:
    print(f"\n{'='*60}")
    print("  Figure 2: Headline domain alignment (1×2)")
    print(f"{'='*60}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(
        "Domain Alignment: Source-Only vs DANN Warmstart\n"
        "BDD100K Day + Night Test Set  |  Seed 42",
        fontsize=12, fontweight="bold",
    )

    domain_handles, _ = make_legends()

    for col, (tag, is_dann, title) in enumerate([
        ("source_only",    False, "Source Only\n(No Domain Adaptation)"),
        ("dann_warmstart", True,  "DANN Warmstart\n(Adversarial Alignment)"),
    ]):
        print(f"\n  [{col+1}/2] {title.splitlines()[0]}")
        model = load_model(tag, is_dann, CKPT_B)
        features, class_labels, domain_labels = extract_features(model, loader)
        emb = run_tsne(features)
        scatter_by_domain(axes[col], emb, domain_labels, title)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    fig.legend(handles=domain_handles, title="Domain",
               loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=10)
    plt.tight_layout()

    save_path = FIGURES_DIR / "bdd_tsne_domain_alignment.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  [Saved] {save_path}")


# ════════════════════════════════════════════════════════════════════════════
# Figure 3 — Direction 1 label sweep (2 × 7)
# ════════════════════════════════════════════════════════════════════════════

def figure_direction1_sweep(loader: DataLoader) -> None:
    print(f"\n{'='*60}")
    print("  Figure 3: Direction 1 label sweep (2×7)")
    print(f"{'='*60}")

    n_cols = len(SWEEP_RATIOS)
    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8))
    fig.suptitle(
        "Direction 1 — Feature Space Evolution Across Night Label Ratios\n"
        "DANN Warmstart  |  Seed 42  |  Day + Night Test Set",
        fontsize=12, fontweight="bold", y=1.01,
    )

    domain_handles, class_handles = make_legends()

    for col, ratio in enumerate(SWEEP_RATIOS):
        print(f"\n  [{col+1}/{n_cols}] {ratio['display']}")
        try:
            model = load_model(ratio["tag"], is_dann=True, ckpt_dir=CKPT_SWEEP)
            features, class_labels, domain_labels = extract_features(model, loader)
            emb = run_tsne(features)

            scatter_by_domain(axes[0, col], emb, domain_labels, ratio["display"])
            scatter_by_class( axes[1, col], emb, class_labels,  ratio["display"])

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
            axes[0, col].set_title(f"{ratio['display']}\n(missing)", fontsize=8)
            axes[1, col].set_title(f"{ratio['display']}\n(missing)", fontsize=8)

    axes[0, 0].set_ylabel("Coloured by Domain", fontsize=10, labelpad=8)
    axes[1, 0].set_ylabel("Coloured by Class",  fontsize=10, labelpad=8)

    fig.legend(handles=domain_handles, title="Domain",
               loc="lower left",  bbox_to_anchor=(0.01, -0.06), ncol=2, fontsize=9)
    fig.legend(handles=class_handles,  title="Class",
               loc="lower right", bbox_to_anchor=(0.99, -0.06), ncol=3, fontsize=9)

    plt.tight_layout()
    save_path = FIGURES_DIR / "bdd_tsne_direction1_sweep.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  [Saved] {save_path}")


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    print(f"\n{'='*60}")
    print("  BDD t-SNE Visualisation")
    print(f"  Output: {FIGURES_DIR}")
    print(f"{'='*60}\n")

    # Build a shared test loader — day + night, no augmentation
    # This is the same data used for evaluation in all experiments
    test_ds = VehicleDataset(
        domains=["day", "night"],
        split="test",
        norm="imagenet",
        augment=False,
    )
    loader = DataLoader(
        test_ds,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    print(f"[Dataset] Test set: {len(test_ds)} images (day + night)")

    figure_strategy_comparison(loader)
    figure_domain_alignment(loader)
    figure_direction1_sweep(loader)

    print(f"\n{'='*60}")
    print("  ALL FIGURES SAVED")
    print(f"  {FIGURES_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
