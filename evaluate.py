# evaluate.py

import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from config import (
    CLASSES, RESULTS_DIR, CHECKPOINT_DIR,
    SEEDS,
)
import config
from models import BaseModel, get_model
from dann import DANNModel, get_dann_model
from train import get_device, load_checkpoint


# ── Label map ─────────────────────────────────────────────────────────────────
IDX_TO_CLASS = {i: cls for i, cls in enumerate(sorted(CLASSES))}
# 0→bus, 1→car, 2→truck


def get_results_csv_path():
    current_seed = SEEDS[config.ACTIVE_SEED_INDEX]
    if config.DIRECTION1_ACTIVE:
        return RESULTS_DIR / f"direction1_results_seed{current_seed}.csv"
    if config.EXP1_ACTIVE:
        return RESULTS_DIR / f"all_results_{config.EXP1_CONFIG_NAME}_seed{current_seed}.csv"
    return RESULTS_DIR / f"all_results_seed{current_seed}.csv"


# ════════════════════════════════════════════════════════════════════════════
# Core Evaluation Function
# Called identically for all 12 runs
# ════════════════════════════════════════════════════════════════════════════

def evaluate(
    model:      torch.nn.Module,
    loader:     DataLoader,
    model_name: str,
    experiment: str,
    split:      str = "test",
) -> dict:
    """
    Run evaluation on the given loader and compute all metrics.
    Always evaluated on night test set — enforced in main.py by
    passing the correct loader here.

    Parameters
    ----------
    model      : trained BaseModel or DANNModel
    loader     : DataLoader for the evaluation split
    model_name : for results naming
    experiment : for results naming
    split      : label for logging only — always "test" in practice

    Returns
    -------
    results : dict containing all metrics, ready for CSV logging
    """
    device = get_device()
    model  = model.to(device)
    model.eval()

    all_preds   = []
    all_labels  = []
    all_probs   = []
    all_indices = []

    with torch.no_grad():
        for imgs, class_labels, _, indices in loader:
            imgs         = imgs.to(device)
            class_labels = class_labels.to(device)

            with torch.amp.autocast("cuda"):
                # Handle both standard model and DANN model
                output = model(imgs)
                if isinstance(output, tuple):
                    # DANN returns (class_logits, domain_probs)
                    logits = output[0]
                else:
                    logits = output

            preds = logits.argmax(dim=1)
            probs = torch.softmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(class_labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_indices.extend(indices.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.concatenate(all_probs, axis=0)
    all_indices = np.array(all_indices)

    # ── Metrics ───────────────────────────────────────────────────────────────
    accuracy    = accuracy_score(all_labels, all_preds)
    
    # Top-2 Accuracy
    top2_correct = 0
    top2_preds = torch.topk(torch.from_numpy(all_probs), k=2, dim=1).indices.numpy()
    for i in range(len(all_labels)):
        if all_labels[i] in top2_preds[i]:
            top2_correct += 1
    top2_acc = top2_correct / len(all_labels)

    macro_f1    = f1_score(all_labels, all_preds, average="macro")
    per_class_f1 = f1_score(all_labels, all_preds, average=None).tolist()
    conf_matrix  = confusion_matrix(all_labels, all_preds).tolist()

    # Per-class F1 as a named dict for readability in results
    per_class_f1_named = {
        IDX_TO_CLASS[i]: round(per_class_f1[i], 4)
        for i in range(len(per_class_f1))
    }

    # Full sklearn classification report for console printing
    report = classification_report(
        all_labels,
        all_preds,
        target_names=sorted(CLASSES),
    )

    results = {
        "model":            model_name,
        "experiment":       experiment,
        "split":            split,
        "accuracy":         round(accuracy, 4),
        "top2_acc":         round(top2_acc, 4),
        "macro_f1":         round(macro_f1, 4),
        "per_class_f1":     per_class_f1_named,
        "confusion_matrix": conf_matrix,
        # Infrastructure for failure analysis
        "indices":          all_indices.tolist(),
        "probs":            all_probs.tolist(),
        "labels":           all_labels.tolist(),
    }

    # ── Console output ────────────────────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print(f"  Evaluation — {model_name} / {experiment} / {split}")
    print(f"{'═' * 60}")
    print(f"  Accuracy  : {accuracy:.4f} (Top-2: {top2_acc:.4f})")
    print(f"  Macro F1  : {macro_f1:.4f}")
    print(f"\n  Per-class F1:")
    for cls, f1 in per_class_f1_named.items():
        print(f"    {cls:<10} : {f1:.4f}")
    print(f"\n  Confusion Matrix (rows=true, cols=pred):")
    print(f"  Classes: {sorted(CLASSES)}")
    for i, row in enumerate(conf_matrix):
        print(f"    {IDX_TO_CLASS[i]:<10} : {row}")
    print(f"\n  Full Report:\n{report}")
    print(f"{'═' * 60}\n")

    return results


# ════════════════════════════════════════════════════════════════════════════
# Results Logger
# Appends one row per run to a single CSV
# ════════════════════════════════════════════════════════════════════════════

def save_results(results: dict) -> None:
    """
    Append evaluation results for one run to all_results.csv.
    Creates the file with headers if it doesn't exist.

    CSV columns:
        model, experiment, accuracy, macro_f1,
        bus_f1, car_f1, truck_f1, confusion_matrix

    Confusion matrix is stored as a JSON string in the CSV cell
    so it stays human-readable without extra files.
    """
    import csv

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = get_results_csv_path()

    fieldnames = [
        "model", "experiment", "run_family", "config_name", "seed",
        "warmstart", "semi_supervised", "target_label_ratio",
        "accuracy", "macro_f1",
        "bus_f1", "car_f1", "truck_f1",
        "confusion_matrix",
    ]

    if config.DIRECTION1_ACTIVE:
        config_name = config.DIRECTION1_EXPERIMENT_NAME
        run_family = "direction1"
    elif config.EXP1_ACTIVE:
        config_name = config.EXP1_CONFIG_NAME
        run_family = "exp1"
    else:
        config_name = "baseline"
        run_family = "baseline"

    row = {
        "model":            results["model"],
        "experiment":       results["experiment"],
        "run_family":       run_family,
        "config_name":      config_name,
        "seed":             SEEDS[config.ACTIVE_SEED_INDEX],
        "warmstart":        config.DANN_WARMSTART,
        "semi_supervised":  config.DANN_SEMI_SUPERVISED,
        "target_label_ratio": round(float(config.DANN_TARGET_LABEL_RATIO), 4),
        "accuracy":         results["accuracy"],
        "macro_f1":         results["macro_f1"],
        "bus_f1":           results["per_class_f1"].get("bus",   ""),
        "car_f1":           results["per_class_f1"].get("car",   ""),
        "truck_f1":         results["per_class_f1"].get("truck", ""),
        "confusion_matrix": json.dumps(results["confusion_matrix"]),
    }

    # Check if file exists to decide whether to write headers
    write_header = not csv_path.exists()

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"[Results] Appended to {csv_path}")


# ════════════════════════════════════════════════════════════════════════════
# Convenience: Load best checkpoint and evaluate in one call
# Used in main.py after training completes
# ════════════════════════════════════════════════════════════════════════════

def load_and_evaluate(
    model_name: str,
    experiment: str,
    test_loader: DataLoader,
    experiment_label: str | None = None,
    checkpoint_experiment: str | None = None,
    is_dann: bool | None = None,
) -> dict:
    """
    Load the best checkpoint for a completed run and evaluate on
    the night test set.

    Handles both standard and DANN models automatically.

    Parameters
    ----------
    model_name   : "custom_cnn" | "resnet18" | "efficientnet_b0"
    experiment   : "target_only" | "source_only" | "finetune" | "dann"
    test_loader  : DataLoader for night test set

    Returns
    -------
    results dict — also saved to CSV automatically
    """
    checkpoint_name = checkpoint_experiment or experiment
    use_dann = is_dann if is_dann is not None else checkpoint_name == "dann"

    # Instantiate correct model type
    if use_dann:
        model = get_dann_model(model_name)
    else:
        model = get_model(model_name)

    # Load best weights from training
    model = load_checkpoint(model, model_name, checkpoint_name)

    # Evaluate and save
    results = evaluate(model, test_loader, model_name, experiment_label or experiment)
    save_results(results)

    return results


# ════════════════════════════════════════════════════════════════════════════
# Domain Gap + Gap Closure Tables
# Derived from all_results.csv after all 12 runs complete
# ════════════════════════════════════════════════════════════════════════════

def print_analysis_tables() -> None:
    """
    Read all_results.csv and print the two analysis tables:

    Table 1 — Domain Gap
        model | target_only | source_only | gap

    Table 2 — Gap Closure
        model | source_only | finetune | dann

    Call this once all 12 runs have completed.
    """
    import csv

    csv_path = RESULTS_DIR / "all_results.csv"
    if not csv_path.exists():
        print("[Error] all_results.csv not found. Run all experiments first.")
        return

    # Read all rows into a lookup: (model, experiment) → accuracy
    lookup = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["model"], row["experiment"])
            lookup[key] = float(row["accuracy"])

    models = ["custom_cnn", "resnet18", "efficientnet_b0"]

    # ── Table 1: Domain Gap ───────────────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print("  Table 1 — Domain Gap  (accuracy on night test set)")
    print(f"{'─' * 60}")
    print(f"  {'Model':<20} {'Target-only':>12} {'Source-only':>12} {'Gap':>8}")
    print(f"{'─' * 60}")

    for m in models:
        target = lookup.get((m, "target_only"), float("nan"))
        source = lookup.get((m, "source_only"), float("nan"))
        gap    = target - source
        print(f"  {m:<20} {target:>12.4f} {source:>12.4f} {gap:>8.4f}")

    print(f"{'═' * 60}")

    # ── Table 2: Gap Closure ──────────────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print("  Table 2 — Gap Closure  (accuracy on night test set)")
    print(f"{'─' * 60}")
    print(f"  {'Model':<20} {'Source-only':>12} {'Fine-tune':>10} {'DANN':>8}")
    print(f"{'─' * 60}")

    for m in models:
        source   = lookup.get((m, "source_only"), float("nan"))
        finetune = lookup.get((m, "finetune"),    float("nan"))
        dann     = lookup.get((m, "dann"),         float("nan"))
        print(f"  {m:<20} {source:>12.4f} {finetune:>10.4f} {dann:>8.4f}")

    print(f"{'═' * 60}\n")


# ════════════════════════════════════════════════════════════════════════════
# Calibration & Statistical Tests
# ════════════════════════════════════════════════════════════════════════════

def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE).
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels)

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Calculated |confidence - accuracy| in each bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece


def mcnemar_test(preds_a: np.ndarray, preds_b: np.ndarray, labels: np.ndarray):
    """
    Perform McNemar's test to compare two models.
    Returns (statistic, p_value).
    """
    from statsmodels.stats.contingency_tables import mcnemar

    correct_a = (preds_a == labels)
    correct_b = (preds_b == labels)

    # Contingency table:
    #              B Correct | B Incorrect
    # A Correct   |    n00   |    n01
    # A Incorrect |    n10   |    n11
    n01 = np.sum(correct_a & ~correct_b)
    n10 = np.sum(~correct_a & correct_b)

    table = [[0, n01], [n10, 0]]
    result = mcnemar(table, exact=True)
    return result.statistic, result.pvalue


# ════════════════════════════════════════════════════════════════════════════
# Granular Evaluation
# ════════════════════════════════════════════════════════════════════════════

def evaluate_by_weather(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    loader_config: dict,
) -> dict:
    """
    Evaluate model per weather condition.
    """
    # Assuming dataset is a VehicleDataset or similar that supports filters
    unique_weather = ["clear", "rainy", "foggy", "overcast"] # Example set
    # In practice, we would extract unique values from dataset.metadata
    
    weather_results = {}
    for w in unique_weather:
        print(f"[Eval] Filtering by weather: {w}")
        dataset.weather_filter = w
        dataset.records = dataset._load_records() # Refresh records
        
        if len(dataset) == 0:
            print(f"  No records for weather: {w}. Skipping.")
            continue
            
        loader = DataLoader(dataset, **loader_config)
        metrics = evaluate(model, loader, "dummy", "dummy", split=f"weather_{w}")
        weather_results[w] = metrics
        
    # Reset filter
    dataset.weather_filter = None
    dataset.records = dataset._load_records()
    return weather_results


def evaluate_by_scene(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    loader_config: dict,
) -> dict:
    """
    Evaluate model per scene type.
    """
    unique_scenes = ["city", "highway", "residential"] # Example set
    
    scene_results = {}
    for s in unique_scenes:
        print(f"[Eval] Filtering by scene: {s}")
        dataset.scene_filter = s
        dataset.records = dataset._load_records()
        
        if len(dataset) == 0:
            print(f"  No records for scene: {s}. Skipping.")
            continue
            
        loader = DataLoader(dataset, **loader_config)
        metrics = evaluate(model, loader, "dummy", "dummy", split=f"scene_{s}")
        scene_results[s] = metrics
        
    dataset.scene_filter = None
    dataset.records = dataset._load_records()
    return scene_results

def aggregate_seed_results() -> None:
    """
    Read all_results_seed42.csv, all_results_seed43.csv, all_results_seed44.csv.
    Compute mean and std for each (model, experiment) pair.
    Write to all_results_summary.csv.
    """
    import csv
    import pandas as pd
    import numpy as np

    cols_to_agg = ["accuracy", "macro_f1", "bus_f1", "car_f1", "truck_f1"]
    all_data = []

    for seed in SEEDS:
        path = RESULTS_DIR / f"all_results_seed{seed}.csv"
        if not path.exists():
            print(f"[Warning] Seed file {path} not found. Skipping.")
            continue
        
        try:
            df = pd.read_csv(path)
            all_data.append(df)
        except Exception as e:
            print(f"[Error] Failed to read {path}: {e}")

    if not all_data:
        print("[Error] No seed result files found.")
        return

    combined = pd.concat(all_data)
    
    # Calculate Mean and Std
    summary = combined.groupby(["model", "experiment"])[cols_to_agg].agg(["mean", "std"]).reset_index()
    
    # Flatten multi-index columns: accuracy_mean, accuracy_std, etc.
    summary.columns = [
        f"{col[0]}_{col[1]}" if col[1] else col[0]
        for col in summary.columns
    ]

    out_path = RESULTS_DIR / "all_results_summary.csv"
    summary.to_csv(out_path, index=False)
    print(f"[Results] Statistical summary saved to {out_path}")


def aggregate_direction1_results() -> None:
    """
    Aggregate Direction 1 ratio-sweep results across seeds.
    Writes direction1_results_summary.csv grouped by model and label ratio.
    """
    import pandas as pd

    all_data = []
    for seed in SEEDS:
        path = RESULTS_DIR / f"direction1_results_seed{seed}.csv"
        if not path.exists():
            print(f"[Warning] Direction 1 seed file not found: {path}")
            continue

        try:
            df = pd.read_csv(path)
            all_data.append(df)
        except Exception as e:
            print(f"[Error] Failed to read {path}: {e}")

    if not all_data:
        print("[Error] No Direction 1 result files found.")
        return

    combined = pd.concat(all_data, ignore_index=True)
    cols_to_agg = ["accuracy", "macro_f1", "bus_f1", "car_f1", "truck_f1"]
    summary = (
        combined
        .groupby(["model", "target_label_ratio"])[cols_to_agg]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values(["model", "target_label_ratio"])
    )

    summary.columns = [
        f"{col[0]}_{col[1]}" if col[1] else col[0]
        for col in summary.columns
    ]

    out_path = RESULTS_DIR / "direction1_results_summary.csv"
    summary.to_csv(out_path, index=False)
    print(f"[Results] Direction 1 summary saved to {out_path}")


def analyze_exp1_results() -> None:
    """
    Aggregate results for Experiment 1 across all 5 configs and 3 seeds.
    Computes domain gap, means, and writes exp1_analysis.csv.
    """
    import pandas as pd
    
    configs = [
        "config1_all_weather",
        "config2_clear_overcast_rainy_snowy",
        "config3_clear_overcast_rainy",
        "config4_clear_overcast",
        "config5_clear_only",
    ]
    
    # 1. Load Baseline summary for target_only ResNet18
    baseline_path = RESULTS_DIR / "all_results_summary.csv"
    if not baseline_path.exists():
        print("[Error] all_results_summary.csv not found. Run baseline first.")
        return
    
    summary_df = pd.read_csv(baseline_path)
    # Get mean accuracy for resnet18 target_only
    target_only_row = summary_df[
        (summary_df["model"] == "resnet18") & 
        (summary_df["experiment"] == "target_only")
    ]
    if target_only_row.empty:
        print("[Error] target_only ResNet18 not found in baseline summary.")
        return
    target_only_mean = target_only_row["accuracy_mean"].values[0]

    rows = []
    for cfg in configs:
        cfg_accuracies = {"source_only": [], "finetune": [], "dann_warmstart": []}
        
        for seed in SEEDS:
            path = RESULTS_DIR / f"all_results_{cfg}_seed{seed}.csv"
            if not path.exists():
                continue
            
            df = pd.read_csv(path)
            for exp in cfg_accuracies.keys():
                acc = df[df["experiment"] == exp]["accuracy"].values
                if len(acc) > 0:
                    cfg_accuracies[exp].append(acc[0])
        
        # Compute means
        if len(cfg_accuracies["source_only"]) == 0:
            print(f"[Warning] No results found for config: {cfg}")
            continue
            
        source_mean = np.mean(cfg_accuracies["source_only"])
        fine_mean   = np.mean(cfg_accuracies["finetune"]) if cfg_accuracies["finetune"] else 0.0
        dann_mean   = np.mean(cfg_accuracies["dann_warmstart"]) if cfg_accuracies["dann_warmstart"] else 0.0
        
        gap = target_only_mean - source_mean
        
        rows.append({
            "config_name": cfg,
            "domain_gap": round(gap, 4),
            "source_only_mean": round(source_mean, 4),
            "finetune_mean": round(fine_mean, 4),
            "dann_warmstart_mean": round(dann_mean, 4),
            "dann_vs_finetune_diff": round(dann_mean - fine_mean, 4)
        })

    analysis_df = pd.DataFrame(rows)
    out_path = RESULTS_DIR / "exp1_analysis.csv"
    analysis_df.to_csv(out_path, index=False)
    print(f"\n[Analysis] Experiment 1 aggregated results saved to {out_path}")
    print(analysis_df.to_string())


"""evaluate() handles both model types — it checks if the output is a tuple to detect DANN. This means main.py can call evaluate() identically for all 12 runs without any special casing.
CSV append mode — every run appends one row. If you re-run an experiment it adds a second row for that combination. That's intentional — you can see if results are stable across runs. If you want clean results just delete all_results.csv and start fresh.
print_analysis_tables() — call this once after all 12 runs finish. It reads the CSV and prints both tables directly in the terminal. The gap in Table 1 tells you how severe the domain shift is per model. Table 2 tells you how much each technique recovered.
Confusion matrix stored as JSON string in CSV — keeps everything in one file. When you want to read it back just json.loads(row["confusion_matrix"])."""
