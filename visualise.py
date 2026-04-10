# visualise.py

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import DataLoader, Subset
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
try:
    import umap
except ImportError:
    umap = None

from config import (
    CLASSES, DATASET_ROOT, RESULTS_DIR, CHECKPOINT_DIR,
    SEED, BATCH_SIZE,
)
from models import get_model
from dann import get_dann_model
from train import get_device, load_checkpoint
from dataset import VehicleDataset


# ── Output directory for all figures ─────────────────────────────────────────
FIGURES_DIR = RESULTS_DIR / "figures"
FEATURES_DIR = RESULTS_DIR / "features"

# ── Colour palettes ───────────────────────────────────────────────────────────
DOMAIN_COLOURS = {
    "day":   "#F4A261",   # warm orange
    "night": "#264653",   # dark teal
}

CLASS_COLOURS = {
    "bus":   "#E76F51",   # coral
    "car":   "#2A9D8F",   # teal
    "truck": "#E9C46A",   # gold
}

IDX_TO_CLASS  = {i: cls for i, cls in enumerate(sorted(CLASSES))}
IDX_TO_DOMAIN = {0: "day", 1: "night"}


# ════════════════════════════════════════════════════════════════════════════
# Feature Extractor
# ════════════════════════════════════════════════════════════════════════════

def extract_features(
    model:      torch.nn.Module,
    loader:     DataLoader,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pass all samples through the model's feature extractor and
    collect 256-dim vectors along with class and domain labels.

    Works with both BaseModel and DANNModel since both expose
    get_features().

    Parameters
    ----------
    model  : trained model with get_features() method
    loader : DataLoader — typically the night test set

    Returns
    -------
    features     : (N, 256) float32 array
    class_labels : (N,)     int array
    domain_labels: (N,)     int array
    """
    device = get_device()
    model  = model.to(device)
    model.eval()

    all_features = []
    all_classes  = []
    all_domains  = []

    with torch.no_grad():
        for imgs, class_labels, domain_labels, _ in loader:
            imgs = imgs.to(device)

            with torch.amp.autocast("cuda"):
                feats = model.get_features(imgs)        # (B, 256)

            all_features.append(feats.cpu().numpy())
            all_classes.append(class_labels.numpy())
            all_domains.append(domain_labels.numpy())

    features      = np.concatenate(all_features, axis=0)
    class_labels  = np.concatenate(all_classes,  axis=0)
    domain_labels = np.concatenate(all_domains,  axis=0)

    print(f"[tSNE] Extracted {features.shape[0]} feature vectors "
          f"of dim {features.shape[1]}")

    return features, class_labels, domain_labels


# ════════════════════════════════════════════════════════════════════════════
# t-SNE Reducer
# ════════════════════════════════════════════════════════════════════════════

def run_tsne(features: np.ndarray) -> np.ndarray:
    """
    Reduce 256-dim features to 2D using t-SNE.

    Parameters are standard for visual diagnostic use:
        perplexity = 30  — good balance for ~1000-2000 samples
        n_iter     = 1000
        random_state = SEED for reproducibility

    Returns
    -------
    embedding : (N, 2) float32 array
    """
    print(f"[tSNE] Running t-SNE on {features.shape[0]} samples ...")
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        max_iter=1000,
        random_state=SEED,
        n_jobs=-1,
    )
    embedding = tsne.fit_transform(features)
    print(f"[tSNE] Done. Embedding shape: {embedding.shape}")
    return embedding


def run_umap(features: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.1) -> np.ndarray:
    """
    Reduce 256-dim features to 2D using UMAP.
    """
    if umap is None:
        print("[Warning] umap-learn not installed. Skipping UMAP.")
        return np.zeros((features.shape[0], 2))

    print(f"[UMAP] Running UMAP on {features.shape[0]} samples ...")
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=SEED,
    )
    embedding = reducer.fit_transform(features)
    print(f"[UMAP] Done. Embedding shape: {embedding.shape}")
    return embedding


# ════════════════════════════════════════════════════════════════════════════
# Plot Functions
# ════════════════════════════════════════════════════════════════════════════

def _base_scatter(
    ax:        plt.Axes,
    embedding: np.ndarray,
    colours:   np.ndarray,
    alpha:     float = 0.6,
    size:      float = 8.0,
) -> None:
    """Draw the scatter points onto an existing axes."""
    ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=colours,
        alpha=alpha,
        s=size,
        linewidths=0,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)


def plot_by_domain(
    embedding:     np.ndarray,
    domain_labels: np.ndarray,
    title:         str,
    ax:            plt.Axes,
) -> None:
    """
    Colour each point by domain (day=orange, night=teal).
    Good alignment after DANN = orange and teal points mixed together.
    Bad alignment (source-only) = two distinct clusters.
    """
    colours = np.array([
        DOMAIN_COLOURS[IDX_TO_DOMAIN[d]] for d in domain_labels
    ])
    _base_scatter(ax, embedding, colours)

    # Legend
    patches = [
        mpatches.Patch(color=DOMAIN_COLOURS[d], label=d)
        for d in ("day", "night")
    ]
    ax.legend(handles=patches, loc="lower right", fontsize=8)
    ax.set_title(title, fontsize=10, pad=6)


def plot_by_class(
    embedding:    np.ndarray,
    class_labels: np.ndarray,
    title:        str,
    ax:           plt.Axes,
) -> None:
    """
    Colour each point by vehicle class (bus/car/truck).
    Class clusters should stay tight and separated regardless of
    domain — if they blur together the model is not discriminating well.
    """
    colours = np.array([
        CLASS_COLOURS[IDX_TO_CLASS[c]] for c in class_labels
    ])
    _base_scatter(ax, embedding, colours)

    # Legend
    patches = [
        mpatches.Patch(color=CLASS_COLOURS[cls], label=cls)
        for cls in sorted(CLASSES)
    ]
    ax.legend(handles=patches, loc="lower right", fontsize=8)
    ax.set_title(title, fontsize=10, pad=6)


# ════════════════════════════════════════════════════════════════════════════
# Side-by-side comparison: source_only vs dann
# One figure per model — 2 rows x 2 cols
#   row 0: coloured by domain
#   row 1: coloured by class
#   col 0: source_only
#   col 1: dann
# ════════════════════════════════════════════════════════════════════════════

def plot_comparison(
    model_name:  str,
    norm:        str,
) -> None:
    """
    Produce the 2×2 comparison figure for one model:

        Source-only / by domain  |  DANN / by domain
        Source-only / by class   |  DANN / by class

    Saves to results/figures/<model_name>_tsne_comparison.png

    Parameters
    ----------
    model_name : "custom_cnn" | "resnet18" | "efficientnet_b0"
    norm       : "dataset" for custom_cnn, "imagenet" for others
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ── Build test loader — day + night so we see both domains in t-SNE ───────
    # We use both domains here so the plot shows how well domains mix.
    # Evaluation uses night only — this is visualisation only.
    test_ds = VehicleDataset(
        domains=["day", "night"],
        split="test",
        norm=norm,
        augment=False,
    )
    loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        f"t-SNE Feature Space — {model_name}",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    for col, experiment in enumerate(["source_only", "dann"]):

        print(f"\n[tSNE] Processing {model_name} / {experiment} ...")

        # ── Load model + checkpoint ───────────────────────────────────────────
        if experiment == "dann":
            model = get_dann_model(model_name)
        else:
            model = get_model(model_name)

        model = load_checkpoint(model, model_name, experiment)

        # ── Extract features + run t-SNE ──────────────────────────────────────
        features, class_labels, domain_labels = extract_features(model, loader)
        embedding = run_tsne(features)

        # ── Row 0: coloured by domain ─────────────────────────────────────────
        plot_by_domain(
            embedding, domain_labels,
            title=f"{experiment} — by domain",
            ax=axes[0, col],
        )

        # ── Row 1: coloured by class ──────────────────────────────────────────
        plot_by_class(
            embedding, class_labels,
            title=f"{experiment} — by class",
            ax=axes[1, col],
        )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    save_path = FIGURES_DIR / f"{model_name}_tsne_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[tSNE] Saved → {save_path}")


def plot_all_four_experiments(
    model_name: str,
    norm: str,
) -> None:
    """
    Produce a 2×4 figure showing all four experiments for one model:
    (source_only, target_only, finetune, dann)
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    experiments = ["source_only", "target_only", "finetune", "dann"]
    
    test_ds = VehicleDataset(
        domains=["day", "night"],
        split="test",
        norm=norm,
        augment=False,
    )
    loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f"t-SNE All Experiments — {model_name}", fontsize=16, fontweight="bold")

    for col, experiment in enumerate(experiments):
        print(f"[tSNE] Multi-plot processing: {experiment}")
        if experiment == "dann":
            model = get_dann_model(model_name)
        else:
            model = get_model(model_name)
        
        try:
            model = load_checkpoint(model, model_name, experiment)
            features, class_labels, domain_labels = extract_features(model, loader)
            embedding = run_tsne(features)
            
            plot_by_domain(embedding, domain_labels, title=f"{experiment}\n(Domain)", ax=axes[0, col])
            plot_by_class(embedding, class_labels, title=f"{experiment}\n(Class)", ax=axes[1, col])
        except Exception as e:
            print(f"  [Error] Failed to plot {experiment}: {e}")
            axes[0, col].set_title(f"{experiment} (Failed)")
            axes[1, col].set_title(f"{experiment} (Failed)")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = FIGURES_DIR / f"{model_name}_tsne_all_experiments.png"
    plt.savefig(save_path, dpi=150)
    plt.close()


# ════════════════════════════════════════════════════════════════════════════
# Training Curve Plotter
# Separate utility — call after each training run
# ════════════════════════════════════════════════════════════════════════════

def plot_training_curves(
    history:    dict,
    model_name: str,
    experiment: str,
) -> None:
    """
    Plot training and validation loss (and accuracy if available)
    over epochs. Saves to results/figures/<model>_<experiment>_curves.png

    Works with both standard history (train_loss/val_loss/train_acc/val_acc)
    and DANN history (train_class_loss/train_domain_loss/val_loss/val_acc).

    Parameters
    ----------
    history    : dict returned by train_standard() or train_dann()
    model_name : for title and filename
    experiment : for title and filename
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    is_dann = "train_class_loss" in history

    if is_dann:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Loss components
        axes[0].plot(history["train_class_loss"],  label="class loss",  color="#E76F51")
        axes[0].plot(history["train_domain_loss"], label="domain loss", color="#264653")
        axes[0].set_title("Train Losses")
        axes[0].set_xlabel("Epoch")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # Total train loss vs val loss
        axes[1].plot(history["train_total_loss"], label="train total", color="#2A9D8F")
        axes[1].plot(history["val_loss"],         label="val loss",    color="#E9C46A")
        axes[1].set_title("Total Loss vs Val Loss")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        # Val accuracy
        axes[2].plot(history["val_acc"], color="#F4A261")
        axes[2].set_title("Val Accuracy (night)")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Accuracy")
        axes[2].grid(alpha=0.3)

    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        # Loss
        axes[0].plot(history["train_loss"], label="train", color="#2A9D8F")
        axes[0].plot(history["val_loss"],   label="val",   color="#E76F51")
        axes[0].set_title("Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # Accuracy
        axes[1].plot(history["train_acc"], label="train", color="#2A9D8F")
        axes[1].plot(history["val_acc"],   label="val",   color="#E76F51")
        axes[1].set_title("Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

    fig.suptitle(
        f"Training Curves — {model_name} / {experiment}",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()

    save_path = FIGURES_DIR / f"{model_name}_{experiment}_curves.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[Curves] Saved → {save_path}")


# ════════════════════════════════════════════════════════════════════════════
# Run all t-SNE comparisons after all 12 runs complete
# ════════════════════════════════════════════════════════════════════════════

def run_all_tsne() -> None:
    """
    Generate t-SNE comparison figures for all three models.
    Call this once after all 12 training runs have completed.

    Requires source_only and dann checkpoints for each model.
    """
    configs = [
        ("custom_cnn",      "dataset"),
        ("resnet18",        "imagenet"),
        ("efficientnet_b0", "imagenet"),
    ]

    for model_name, norm in configs:
        print(f"\n{'─' * 60}")
        print(f"[tSNE] Generating comparison for {model_name} ...")
        plot_comparison(model_name, norm)

    print(f"\n[tSNE] All figures saved to {FIGURES_DIR}")


# ════════════════════════════════════════════════════════════════════════════
# Additional Visualisation Utilities
# ════════════════════════════════════════════════════════════════════════════

def plot_confusion_heatmap(cm: list, class_names: list, save_name: str = "confusion_matrix.png"):
    """
    Plot annotated heatmap of confusion matrix.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix Heatmap")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    
    save_path = FIGURES_DIR / save_name
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_reliability_diagram(confidence: np.ndarray, accuracy: np.ndarray, save_name: str = "reliability_diagram.png"):
    """
    Plot reliability diagram (calibration).
    """
    from sklearn.calibration import calibration_curve
    prob_true, prob_pred = calibration_curve(accuracy, confidence, n_bins=10)
    
    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker='o', label='Model')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
    plt.title("Reliability Diagram")
    plt.xlabel("Average Confidence")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)
    
    save_path = FIGURES_DIR / save_name
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_domain_gap_scatter(results_dict: dict, save_name: str = "domain_gap_scatter.png"):
    """
    Plot domain gap vs improvement.
    results_dict: {model_name: {experiment: accuracy}}
    """
    plt.figure(figsize=(8, 6))
    for model, res in results_dict.items():
        gap = res.get("target_only", 0) - res.get("source_only", 0)
        impr = res.get("dann", 0) - res.get("source_only", 0)
        plt.scatter(gap, impr, label=model, s=100)
        plt.annotate(model, (gap, impr), xytext=(5, 5), textcoords='offset points')
    
    plt.title("Domain Gap vs DANN Improvement")
    plt.xlabel("Domain Gap (Target Accuracy - Source Accuracy)")
    plt.ylabel("Improvement (DANN Accuracy - Source Accuracy)")
    plt.legend()
    plt.grid(alpha=0.3)
    
    save_path = FIGURES_DIR / save_name
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_radar_weather(results_by_weather: dict, save_name: str = "weather_radar.png"):
    """
    Plot radar chart for different weather conditions.
    results_by_weather: {weather: accuracy}
    """
    categories = list(results_by_weather.keys())
    values = list(results_by_weather.values())
    
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='teal', alpha=0.25)
    ax.plot(angles, values, color='teal', linewidth=2)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    plt.title("Performance by Weather")
    
    save_path = FIGURES_DIR / save_name
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_exp1_results() -> None:
    """
    Read exp1_analysis.csv and produce two summary plots:
    1. Accuracy vs Domain Gap size
    2. DANN Advantage vs Domain Gap size
    """
    import pandas as pd
    import seaborn as sns
    from scipy import stats

    csv_path = RESULTS_DIR / "exp1_analysis.csv"
    if not csv_path.exists():
        print("[Error] exp1_analysis.csv not found. Run analyze_exp1_results first.")
        return

    df = pd.read_csv(csv_path)
    
    # --- Figure 1: Performance vs Gap ---
    plt.figure(figsize=(10, 6))
    plt.plot(df["domain_gap"], df["source_only_mean"], 'o-', label="Source Only", color="#E76F51", linewidth=2)
    plt.plot(df["domain_gap"], df["finetune_mean"],    's-', label="Fine-tune",   color="#2A9D8F", linewidth=2)
    plt.plot(df["domain_gap"], df["dann_warmstart_mean"], '^-', label="DANN Warmstart", color="#264653", linewidth=2)
    
    plt.title("Performance vs Domain Gap Size — ResNet18", fontsize=14, fontweight="bold")
    plt.xlabel("Domain Gap (Target-only - Source-only)", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    
    save_path1 = FIGURES_DIR / "exp1_performance_vs_gap.png"
    plt.savefig(save_path1, dpi=150, bbox_inches="tight")
    plt.close()

    # --- Figure 2: DANN Advantage ---
    plt.figure(figsize=(10, 6))
    x = df["domain_gap"].values
    y = df["dann_vs_finetune_diff"].values
    
    plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
    plt.scatter(x, y, color="#264653", s=100)
    
    # Annotate points
    for i, txt in enumerate(df["config_name"]):
        # simplify name for plot
        label = txt.replace("config", "C")
        plt.annotate(label, (x[i], y[i]), xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    # Add trend line
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    plt.plot(x, slope*x + intercept, color="#E76F51", linestyle=':', alpha=0.7, label=f"Trend (R={r_value:.2f})")
    
    plt.title("DANN Warmstart Advantage over Finetune vs Domain Gap", fontsize=14, fontweight="bold")
    plt.xlabel("Domain Gap (Target-only - Source-only)", fontsize=12)
    plt.ylabel("DANN Improvement over Fine-tune", fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    
    save_path2 = FIGURES_DIR / "exp1_dann_advantage_vs_gap.png"
    plt.savefig(save_path2, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[Visualise] Experiment 1 plots saved to {FIGURES_DIR}")


def save_epoch_features(model, loader, model_name, epoch, seed, run_label: str = "default"):
    """
    Extract features and labels and save to disk for t-SNE animation.
    Filename: <model_name>_<run_label>_seed<seed>_epoch<epoch>.npz
    """
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    safe_label = str(run_label).replace(" ", "_").replace("/", "_")
    save_path = FEATURES_DIR / f"{model_name}_{safe_label}_seed{seed}_epoch{epoch}.npz"
    
    features, classes, domains = extract_features(model, loader)
    np.savez_compressed(
        save_path,
        features=features.astype(np.float32),
        classes=classes.astype(np.int16),
        domains=domains.astype(np.int8),
        epoch=np.array([epoch], dtype=np.int16),
        seed=np.array([seed], dtype=np.int32),
        model=np.array([model_name]),
        run_label=np.array([safe_label]),
    )
    print(f"[Visualise] Saved epoch {epoch} features to {save_path}")


def plot_direction1_results(summary_path: Path | None = None) -> None:
    """
    Plot accuracy and macro-F1 against target label ratio for Direction 1.
    """
    import pandas as pd

    if summary_path is None:
        summary_path = RESULTS_DIR / "direction1_results_summary.csv"

    if not summary_path.exists():
        print(f"[Visualise] Direction 1 summary not found: {summary_path}")
        return

    df = pd.read_csv(summary_path)
    if df.empty:
        print("[Visualise] Direction 1 summary is empty. Skipping plot.")
        return

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    metrics = [
        ("accuracy_mean", "accuracy_std", "Accuracy"),
        ("macro_f1_mean", "macro_f1_std", "Macro-F1"),
    ]

    palette = {
        "efficientnet_b0": "#264653",
        "resnet18": "#E76F51",
        "custom_cnn": "#2A9D8F",
    }

    for ax, (mean_col, std_col, title) in zip(axes, metrics):
        for model_name in sorted(df["model"].unique()):
            model_df = df[df["model"] == model_name].sort_values("target_label_ratio")
            x = model_df["target_label_ratio"].to_numpy() * 100.0
            y = model_df[mean_col].to_numpy()
            yerr = model_df[std_col].fillna(0.0).to_numpy()
            colour = palette.get(model_name, "#1f77b4")

            ax.plot(x, y, marker="o", linewidth=2, color=colour, label=model_name)
            ax.fill_between(x, y - yerr, y + yerr, alpha=0.15, color=colour)

        ax.set_title(title)
        ax.set_xlabel("Labelled Night Data (%)")
        ax.set_ylabel(title)
        ax.grid(alpha=0.3)
        ax.set_xticks(sorted(df["target_label_ratio"].unique() * 100.0))

    axes[0].legend()
    fig.suptitle("Direction 1: Target Label Ratio Sweep", fontsize=13, fontweight="bold")
    plt.tight_layout()

    save_path = FIGURES_DIR / "direction1_label_ratio_curve.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Visualise] Saved Direction 1 plot to {save_path}")


def build_mixed_domain_loader(
    model_name: str,
    split: str = "test",
    max_samples: int | None = None,
    seed: int = SEED,
) -> DataLoader:
    """
    Build a loader containing both day and night domains for visualisation.
    """
    norm = "dataset" if model_name == "custom_cnn" else "imagenet"
    dataset = VehicleDataset(
        domains=["day", "night"],
        split=split,
        norm=norm,
        augment=False,
    )
    if max_samples is not None and len(dataset) > max_samples:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(len(dataset), size=max_samples, replace=False))
        dataset = Subset(dataset, indices.tolist())
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )


def load_model_from_checkpoint_path(
    model_name: str,
    checkpoint_path: Path,
    is_dann: bool = False,
) -> torch.nn.Module:
    """
    Load a model from an explicit checkpoint path without mutating config paths.
    """
    if is_dann:
        import config
        original_warmstart = config.DANN_WARMSTART
        try:
            config.DANN_WARMSTART = False
            model = get_dann_model(model_name)
        finally:
            config.DANN_WARMSTART = original_warmstart
    else:
        model = get_model(model_name)

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    print(f"[Visualise] Loaded checkpoint -> {checkpoint_path}")
    return model


def fit_shared_pca(
    features_by_label: dict[str, np.ndarray],
    n_components: int = 2,
) -> PCA:
    """
    Fit one shared PCA projection so all frames live in the same 2D space.
    """
    stacked = np.concatenate(list(features_by_label.values()), axis=0)
    pca = PCA(n_components=n_components, random_state=SEED)
    pca.fit(stacked)
    return pca


def render_embedding_panel(
    embedding: np.ndarray,
    class_labels: np.ndarray,
    domain_labels: np.ndarray,
    title_prefix: str,
    subtitle: str,
    save_path: Path,
) -> None:
    """
    Save a 1x2 figure with domain-coloured and class-coloured views.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_by_domain(embedding, domain_labels, f"{title_prefix}\nBy Domain", axes[0])
    plot_by_class(embedding, class_labels, f"{title_prefix}\nBy Class", axes[1])
    fig.suptitle(subtitle, fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Visualise] Saved panel -> {save_path}")


def create_gif_from_pngs(
    frame_paths: list[Path],
    output_path: Path,
    duration_ms: int = 1200,
) -> None:
    """
    Create a GIF from an ordered list of PNG frames using Pillow.
    """
    if not frame_paths:
        print("[Visualise] No frames found for GIF generation.")
        return

    images = [Image.open(path).convert("RGB") for path in frame_paths]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
    )
    for image in images:
        image.close()
    print(f"[Visualise] Saved GIF -> {output_path}")


def generate_direction1_ratio_visuals(
    model_name: str = "efficientnet_b0",
    seed: int = 42,
    checkpoint_root: Path | None = None,
    max_samples: int = 800,
) -> None:
    """
    Generate static frames and a GIF across saved Direction 1 checkpoints.
    """
    import config

    if checkpoint_root is None:
        checkpoint_root = CHECKPOINT_DIR

    checkpoint_specs = [
        ("0% Labels", "direction1_ratio_000", True),
        ("5% Labels", "direction1_ratio_005", True),
        ("10% Labels", "direction1_ratio_010", True),
        ("25% Labels", "direction1_ratio_025", True),
        ("50% Labels", "direction1_ratio_050", True),
        ("75% Labels", "direction1_ratio_075", True),
        ("100% Labels", "direction1_ratio_100", True),
    ]

    seed_index = config.SEEDS.index(seed)
    old_seed_index = config.ACTIVE_SEED_INDEX
    config.ACTIVE_SEED_INDEX = seed_index

    try:
        loader = build_mixed_domain_loader(
            model_name,
            split="test",
            max_samples=max_samples,
            seed=seed,
        )
        features_by_label = {}
        labels_lookup = {}

        for display_name, experiment_tag, is_dann in checkpoint_specs:
            checkpoint_path = checkpoint_root / f"{model_name}_{experiment_tag}_seed{seed}_best.pth"
            if not checkpoint_path.exists():
                print(f"[Visualise] Missing checkpoint, skipping: {checkpoint_path}")
                continue

            model = load_model_from_checkpoint_path(model_name, checkpoint_path, is_dann=is_dann)
            features, class_labels, domain_labels = extract_features(model, loader)
            features_by_label[display_name] = features
            labels_lookup[display_name] = (class_labels, domain_labels)

        if not features_by_label:
            print("[Visualise] No Direction 1 checkpoints available for visuals.")
            return

        pca = fit_shared_pca(features_by_label)
        frame_dir = FIGURES_DIR / "direction1_ratio_frames"
        frame_paths = []

        for display_name, _, _ in checkpoint_specs:
            if display_name not in features_by_label:
                continue
            embedding = pca.transform(features_by_label[display_name])
            class_labels, domain_labels = labels_lookup[display_name]
            save_path = frame_dir / f"{model_name}_seed{seed}_{display_name.replace('%', 'pct').replace(' ', '_').lower()}.png"
            render_embedding_panel(
                embedding,
                class_labels,
                domain_labels,
                title_prefix=display_name,
                subtitle=f"Direction 1 Ratio Progression ({model_name}, seed {seed})",
                save_path=save_path,
            )
            frame_paths.append(save_path)

        gif_path = FIGURES_DIR / f"{model_name}_seed{seed}_direction1_ratio_progression.gif"
        create_gif_from_pngs(frame_paths, gif_path, duration_ms=1200)
    finally:
        config.ACTIVE_SEED_INDEX = old_seed_index


def generate_baseline_direction1_comparison(
    model_name: str = "efficientnet_b0",
    seed: int = 42,
    baseline_checkpoint_root: Path | None = None,
    direction1_checkpoint_root: Path | None = None,
    max_samples: int = 800,
) -> None:
    """
    Generate a static comparison figure spanning baseline and Direction 1 checkpoints.
    """
    import config

    if baseline_checkpoint_root is None:
        baseline_checkpoint_root = DATASET_ROOT / "Final Baseline" / "checkpoints"
    if direction1_checkpoint_root is None:
        direction1_checkpoint_root = CHECKPOINT_DIR

    specs = [
        ("Source Only", baseline_checkpoint_root / f"{model_name}_source_only_seed{seed}_best.pth", False),
        ("Fine-tune", baseline_checkpoint_root / f"{model_name}_finetune_seed{seed}_best.pth", False),
        ("Old DANN", baseline_checkpoint_root / f"{model_name}_dann_seed{seed}_best.pth", True),
        ("Dir1 0%", direction1_checkpoint_root / f"{model_name}_direction1_ratio_000_seed{seed}_best.pth", True),
        ("Dir1 50%", direction1_checkpoint_root / f"{model_name}_direction1_ratio_050_seed{seed}_best.pth", True),
        ("Dir1 100%", direction1_checkpoint_root / f"{model_name}_direction1_ratio_100_seed{seed}_best.pth", True),
    ]

    seed_index = config.SEEDS.index(seed)
    old_seed_index = config.ACTIVE_SEED_INDEX
    config.ACTIVE_SEED_INDEX = seed_index

    try:
        loader = build_mixed_domain_loader(
            model_name,
            split="test",
            max_samples=max_samples,
            seed=seed,
        )
        features_by_label = {}
        labels_lookup = {}

        for label, checkpoint_path, is_dann in specs:
            if not checkpoint_path.exists():
                print(f"[Visualise] Missing checkpoint, skipping: {checkpoint_path}")
                continue
            model = load_model_from_checkpoint_path(model_name, checkpoint_path, is_dann=is_dann)
            features, class_labels, domain_labels = extract_features(model, loader)
            features_by_label[label] = features
            labels_lookup[label] = (class_labels, domain_labels)

        if not features_by_label:
            print("[Visualise] No checkpoints available for baseline comparison figure.")
            return

        pca = fit_shared_pca(features_by_label)
        labels_in_order = [label for label, _, _ in specs if label in features_by_label]
        fig, axes = plt.subplots(2, len(labels_in_order), figsize=(4 * len(labels_in_order), 8))

        if len(labels_in_order) == 1:
            axes = np.array(axes).reshape(2, 1)

        for col, label in enumerate(labels_in_order):
            embedding = pca.transform(features_by_label[label])
            class_labels, domain_labels = labels_lookup[label]
            plot_by_domain(embedding, domain_labels, label, axes[0, col])
            plot_by_class(embedding, class_labels, label, axes[1, col])

        fig.suptitle(
            f"Baseline to Direction 1 Comparison ({model_name}, seed {seed})",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        save_path = FIGURES_DIR / f"{model_name}_seed{seed}_baseline_direction1_comparison.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[Visualise] Saved comparison figure -> {save_path}")
    finally:
        config.ACTIVE_SEED_INDEX = old_seed_index

"""t-SNE uses both domains — unlike evaluation which is night test only, the t-SNE loader loads both day and night test crops. This is intentional — the whole point of the plot is to see whether day and night features mix after DANN. If you only load night you'd have nothing to compare.
2×2 layout per model — top row shows domain separation, bottom row shows class separation. Left column is source_only, right is DANN. This gives you four direct visual comparisons per model in a single figure.
What good alignment looks like — in the domain plot, source_only should show two distinct orange and teal blobs. After DANN they should mix into one cloud. In the class plot, bus/car/truck clusters should stay tight and separated in both columns — if they blur under DANN it means alignment came at the cost of discrimination.
Training curves auto-detect DANN — plot_training_curves() checks which keys are in the history dict and switches layout accordingly. You call it the same way for all 12 runs.
run_all_tsne() — single call after everything is trained. Loops all three models and saves three figures."""
