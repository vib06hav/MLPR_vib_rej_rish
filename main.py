# main.py

import torch
import random
import numpy as np

from config import (
    MODEL_NAME, EXPERIMENT, SEED,
    RUN_MODE, SEEDS, MODELS_TO_RUN,
    LR_GRID_SEARCH, BACKBONE_LR_GRID, HEAD_LR_GRID,
    DANN_WARMSTART, DANN_SEMI_SUPERVISED,
    EXP1_ACTIVE, EXP1_CONFIG_NAME, SOURCE_WEATHER_INCLUDE, SOURCE_SCENE_INCLUDE
)
import config  # Imported as module to allow patching for grid search
from dataset import get_loaders
from models import get_model
from dann import get_dann_model
from train import train_standard, train_dann, load_checkpoint
from evaluate import load_and_evaluate, print_analysis_tables, evaluate
from visualise import plot_training_curves, run_all_tsne, plot_direction1_results


# ════════════════════════════════════════════════════════════════════════════
# Reproducibility — set everywhere before anything else
# ════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False


# ════════════════════════════════════════════════════════════════════════════
# Individual Experiment Runners
# One function per experiment type — main() dispatches to the right one
# ════════════════════════════════════════════════════════════════════════════

def run_target_only(model_name: str) -> None:
    """
    Train and evaluate on night data only.
    Purpose: establish upper bound — best possible night performance.

    Train : night train
    Val   : night val
    Test  : night test
    """
    print(f"\n{'═' * 60}")
    print(f"  EXPERIMENT: target_only  |  MODEL: {model_name}")
    print(f"{'═' * 60}")

    loaders = get_loaders("target_only", model_name)
    model   = get_model(model_name)

    history = train_standard(model, loaders, model_name, "target_only")
    plot_training_curves(history, model_name, "target_only")

    results = load_and_evaluate(model_name, "target_only", loaders["test"])

    print(f"\n[Done] target_only — {model_name}")
    print(f"       Night test accuracy : {results['accuracy']:.4f}")
    print(f"       Night test macro F1 : {results['macro_f1']:.4f}")


def run_source_only(model_name: str) -> None:
    """
    Train on day data, evaluate on night test.
    Purpose: measure the domain gap — how much accuracy drops
             when the model never sees night images.

    Train : day train
    Val   : day val
    Test  : night test
    """
    print(f"\n{'═' * 60}")
    print(f"  EXPERIMENT: source_only  |  MODEL: {model_name}")
    print(f"{'═' * 60}")

    loaders = get_loaders("source_only", model_name)
    model   = get_model(model_name)

    history = train_standard(model, loaders, model_name, "source_only")
    plot_training_curves(history, model_name, "source_only")

    results = load_and_evaluate(model_name, "source_only", loaders["test"])

    print(f"\n[Done] source_only — {model_name}")
    print(f"       Night test accuracy : {results['accuracy']:.4f}")
    print(f"       Night test macro F1 : {results['macro_f1']:.4f}")


def run_finetune(model_name: str) -> None:
    """
    Start from source_only checkpoint, fine-tune on night data.
    Purpose: transfer learning baseline — does adapting with labels help?

    Requires source_only checkpoint to exist.
    Run source_only first.

    Train : night train  (starting from day-trained weights)
    Val   : night val
    Test  : night test
    """
    print(f"\n{'═' * 60}")
    print(f"  EXPERIMENT: finetune  |  MODEL: {model_name}")
    print(f"{'═' * 60}")

    loaders = get_loaders("finetune", model_name)

    # Load source_only checkpoint as starting point
    model = get_model(model_name)
    model = load_checkpoint(model, model_name, "source_only")
    print(f"  [Finetune] Loaded source_only weights — continuing on night data ...")

    history = train_standard(model, loaders, model_name, "finetune")
    plot_training_curves(history, model_name, "finetune")

    results = load_and_evaluate(model_name, "finetune", loaders["test"])

    print(f"\n[Done] finetune — {model_name}")
    print(f"       Night test accuracy : {results['accuracy']:.4f}")
    print(f"       Night test macro F1 : {results['macro_f1']:.4f}")


def run_dann(model_name: str, experiment_label: str = "dann") -> None:
    """
    Train with Domain Adversarial Neural Network.
    Purpose: domain adaptation — can we close the gap without
             explicit night labels driving the main loss?

    Note: both domains are labelled here so class loss uses both.
    Domain discriminator uses domain labels (day=0, night=1) only.

    Train : day + night train
    Val   : night val
    Test  : night test
    """
    print(f"\n{'═' * 60}")
    print(f"  EXPERIMENT: {experiment_label}  |  MODEL: {model_name}")
    print(f"{'═' * 60}")

    loaders = get_loaders("dann", model_name)
    model   = get_dann_model(model_name)

    history = train_dann(model, loaders, model_name, experiment_tag=experiment_label)
    plot_training_curves(history, model_name, experiment_label)

    results = load_and_evaluate(
        model_name,
        "dann",
        loaders["test"],
        experiment_label=experiment_label,
        checkpoint_experiment=experiment_label,
        is_dann=True,
    )

    print(f"\n[Done] dann — {model_name}")
    print(f"       Night test accuracy : {results['accuracy']:.4f}")
    print(f"       Night test macro F1 : {results['macro_f1']:.4f}")


# ════════════════════════════════════════════════════════════════════════════
# Orchestration Helpers
# ════════════════════════════════════════════════════════════════════════════

def format_direction1_ratio_tag(ratio: float) -> str:
    ratio_pct = int(round(float(ratio) * 100))
    return f"direction1_ratio_{ratio_pct:03d}"


def source_checkpoint_exists(model_name: str) -> bool:
    current_seed = SEEDS[config.ACTIVE_SEED_INDEX]
    path = config.CHECKPOINT_DIR / f"{model_name}_source_only_seed{current_seed}_best.pth"
    return path.exists()


def ensure_source_checkpoint_exists(model_name: str) -> None:
    if source_checkpoint_exists(model_name):
        print(f"  [Direction1] Found source_only checkpoint for {model_name}.")
        return

    if not config.DIRECTION1_AUTO_SOURCE_ONLY:
        raise FileNotFoundError(
            f"Missing source_only checkpoint for {model_name} and auto-generation is disabled."
        )

    print(f"  [Direction1] source_only checkpoint missing for {model_name}. Training it first ...")
    orig_direction1_active = config.DIRECTION1_ACTIVE
    try:
        config.DIRECTION1_ACTIVE = False
        run_source_only(model_name)
    finally:
        config.DIRECTION1_ACTIVE = orig_direction1_active


def run_single(model_name: str, experiment: str, save_all_ckpt: bool = False) -> dict:
    """
    Run a single (model, experiment) pair.
    """
    dispatch = {
        "target_only": run_target_only,
        "source_only": run_source_only,
        "finetune":    run_finetune,
        "dann":        run_dann,
    }

    # Handle new experiment variants
    if experiment == "dann_warmstart":
        print("  [Main] Multi-run: Enabling DANN_WARMSTART")
        config.DANN_WARMSTART = True
        config.DANN_SEMI_SUPERVISED = False
        run_dann(model_name, experiment_label="dann_warmstart")
    elif experiment == "semi_dann":
        print("  [Main] Multi-run: Enabling DANN_SEMI_SUPERVISED")
        config.DANN_WARMSTART = True
        config.DANN_SEMI_SUPERVISED = True
        run_dann(model_name, experiment_label="semi_dann")
        config.DANN_WARMSTART = True
        config.DANN_SEMI_SUPERVISED = False
    elif experiment in dispatch:
        dispatch[experiment](model_name)
    else:
        raise ValueError(f"Unknown experiment: {experiment}")

    # For now, evaluate.py handles loading best and returning results
    # We return results for higher-level loops if needed
    return {}


def run_all_experiments(model_name: str) -> None:
    """
    Run standard suite for one model.
    """
    print(f"\n{'#' * 60}")
    print(f"#  Running all experiments for: {model_name}")
    print(f"{'#' * 60}")

    # 1. Source Only (Prerequisite for finetune)
    source_failed = False
    try:
        run_source_only(model_name)
    except Exception as e:
        print(f"\n[Error] source_only failed for {model_name}: {e}")
        import traceback
        traceback.print_exc()
        source_failed = True

    # 2. Target Only
    try:
        run_target_only(model_name)
    except Exception as e:
        print(f"\n[Error] target_only failed for {model_name}: {e}")
        import traceback
        traceback.print_exc()

    # 3. Finetune (Skip if source failed)
    if source_failed:
        print(f"\n[Skip] Skipping finetune for {model_name} because source_only failed.")
    else:
        try:
            run_finetune(model_name)
        except Exception as e:
            print(f"\n[Error] finetune failed for {model_name}: {e}")
            import traceback
            traceback.print_exc()

    # 4. DANN Warmstart
    try:
        config.DANN_WARMSTART = True
        config.DANN_SEMI_SUPERVISED = False
        run_dann(model_name, experiment_label="dann_warmstart")
    except Exception as e:
        print(f"\n[Error] dann_warmstart failed for {model_name}: {e}")
        import traceback
        traceback.print_exc()

    # 5. Semi-supervised DANN
    try:
        config.DANN_WARMSTART = True
        config.DANN_SEMI_SUPERVISED = True
        run_dann(model_name, experiment_label="semi_dann")
    except Exception as e:
        print(f"\n[Error] semi_dann failed for {model_name}: {e}")
        import traceback
        traceback.print_exc()
    config.DANN_WARMSTART = True
    config.DANN_SEMI_SUPERVISED = False


def run_all_models() -> None:
    """
    Run suite for all models in MODELS_TO_RUN.
    """
    for model_name in MODELS_TO_RUN:
        try:
            run_all_experiments(model_name)
        except Exception as e:
            print(f"\n[Critical Error] run_all_experiments failed for {model_name}: {e}")
            import traceback
            traceback.print_exc()


def run_all_seeds() -> None:
    """
    Loop over all SEEDS and run all models.
    """
    for i, seed in enumerate(SEEDS):
        try:
            print(f"\n{'=' * 60}")
            print(f"  SEED RUN: {seed} (Index {i})")
            print(f"{'=' * 60}")
            
            # Patch config index so all modules see the current seed
            config.ACTIVE_SEED_INDEX = i
            
            set_seed(seed)
            for model_name in config.MODELS_TO_RUN:
                run_all_experiments(model_name)
        except Exception as e:
            print(f"\n[Critical Error] SEED RUN {seed} failed: {e}")
            import traceback
            traceback.print_exc()

    # After all seeds finish, aggregate stats
    print(f"\n{'═' * 60}")
    print("  All seed runs complete. Aggregating summary ...")
    print(f"{'═' * 60}")
    from evaluate import aggregate_seed_results
    aggregate_seed_results()


def run_direction1() -> None:
    """
    One-click overnight runner for the Direction 1 label-ratio sweep.
    """
    print(f"\n{'=' * 72}")
    print("  DIRECTION 1: TARGET LABEL RATIO SWEEP")
    print(f"{'=' * 72}")

    orig_direction1_active = config.DIRECTION1_ACTIVE
    orig_save_epoch_features = config.SAVE_EPOCH_FEATURES
    orig_save_features_every_n = config.SAVE_FEATURES_EVERY_N
    orig_dann_warmstart = config.DANN_WARMSTART
    orig_dann_semi_supervised = config.DANN_SEMI_SUPERVISED
    orig_target_label_ratio = config.DANN_TARGET_LABEL_RATIO

    try:
        config.DIRECTION1_ACTIVE = True
        config.SAVE_EPOCH_FEATURES = config.DIRECTION1_SAVE_EPOCH_FEATURES
        config.SAVE_FEATURES_EVERY_N = config.DIRECTION1_FEATURE_SAVE_EVERY_N

        for i, seed in enumerate(SEEDS):
            print(f"\n{'-' * 72}")
            print(f"  Direction 1 seed: {seed}")
            print(f"{'-' * 72}")

            config.ACTIVE_SEED_INDEX = i
            set_seed(seed)

            for model_name in config.DIRECTION1_MODELS_TO_RUN:
                print(f"\n[Direction1] Model: {model_name}")
                set_seed(seed)

                if config.DIRECTION1_REQUIRE_WARMSTART:
                    ensure_source_checkpoint_exists(model_name)

                for ratio in config.DIRECTION1_LABEL_RATIOS:
                    ratio_tag = format_direction1_ratio_tag(ratio)
                    set_seed(seed)
                    config.DANN_WARMSTART = config.DIRECTION1_REQUIRE_WARMSTART
                    config.DANN_SEMI_SUPERVISED = ratio == 0.0
                    config.DANN_TARGET_LABEL_RATIO = ratio

                    print(
                        f"\n[Direction1] Running {model_name} | "
                        f"ratio={ratio:.0%} | tag={ratio_tag}"
                    )
                    run_dann(model_name, experiment_label=ratio_tag)

        from evaluate import aggregate_direction1_results

        aggregate_direction1_results()
        plot_direction1_results()
        print("\n[Direction1] Overnight sweep complete.")

    finally:
        config.DIRECTION1_ACTIVE = orig_direction1_active
        config.SAVE_EPOCH_FEATURES = orig_save_epoch_features
        config.SAVE_FEATURES_EVERY_N = orig_save_features_every_n
        config.DANN_WARMSTART = orig_dann_warmstart
        config.DANN_SEMI_SUPERVISED = orig_dann_semi_supervised
        config.DANN_TARGET_LABEL_RATIO = orig_target_label_ratio


def run_lr_grid_search(model_name: str, experiment: str) -> None:
    """
    Search over Backbone and Head Learning Rates.
    """
    print(f"\n[Grid Search] Starting LR grid search for {model_name} / {experiment}")
    
    orig_backbone_pre = config.LR_BACKBONE_PRETRAINED
    orig_backbone_scratch = config.LR_BACKBONE_SCRATCH
    orig_head = config.LR_HEAD

    for b_lr in BACKBONE_LR_GRID:
        for h_lr in HEAD_LR_GRID:
            print(f"\n  [Grid] Testing Backbone LR: {b_lr} | Head LR: {h_lr}")
            
            # Patch config
            config.LR_BACKBONE_PRETRAINED = b_lr
            config.LR_BACKBONE_SCRATCH    = b_lr
            config.LR_HEAD                = h_lr
            
            # Custom tag for results logging to avoid overwriting
            exp_tag = f"{experiment}_lr_{b_lr}_{h_lr}"
            
            # We bypass the standard run_x functions to use the tag
            loaders = get_loaders(experiment, model_name)
            if experiment == "dann":
                model = get_dann_model(model_name)
                history = train_dann(model, loaders, model_name)
            else:
                model = get_model(model_name)
                if experiment == "finetune":
                    model = load_checkpoint(model, model_name, "source_only")
                history = train_standard(model, loaders, model_name, experiment)
            
            # Evaluate using the tag
            results = evaluate(model, loaders["test"], model_name, exp_tag)
            # save_results is called inside load_and_evaluate usually, 
            # here we might need a direct call or just rely on evaluate's return
            from evaluate import save_results
            save_results(results)

    # Restore originals
    config.LR_BACKBONE_PRETRAINED = orig_backbone_pre
    config.LR_BACKBONE_SCRATCH    = orig_backbone_scratch
    config.LR_HEAD                = orig_head


def run_exp1_config(config_name: str, weather_include: list, scene_include: list = None) -> None:
    """
    Runner for a single Exp 1 weather configuration across all 3 seeds.
    Sets config flags, runs source_only, finetune, and dann_warmstart.
    """
    print(f"\n{'=' * 70}")
    print(f"  EXPERIMENT 1: {config_name}")
    print(f"{'=' * 70}")
    
    # 1. Patch config
    config.EXP1_ACTIVE = True
    config.EXP1_CONFIG_NAME = config_name
    config.SOURCE_WEATHER_INCLUDE = weather_include
    config.SOURCE_SCENE_INCLUDE = scene_include
    
    # 2. Run for all seeds
    for i, seed in enumerate(SEEDS):
        print(f"\n--- SEED {seed} (Config: {config_name}) ---")
        config.ACTIVE_SEED_INDEX = i
        set_seed(seed)
        
        # ResNet18 only for Experiment 1
        model_name = "resnet18"
        
        # Experiments to run
        run_source_only(model_name)
        run_single(model_name, "finetune")
        run_single(model_name, "dann_warmstart")
        
    # 3. Reset
    config.EXP1_ACTIVE = False


def run_all_exp1() -> None:
    """
    Run all 5 Experiment 1 configurations and aggregate results.
    """
    exp1_configs = [
        #("config1_all_weather", ["clear", "overcast", "rainy", "snowy", "partly cloudy", "foggy"]),
       # ("config2_clear_overcast_rainy_snowy", ["clear", "overcast", "rainy", "snowy"]),
       # ("config3_clear_overcast_rainy", ["clear", "overcast", "rainy"]),
        ("config4_clear_overcast", ["clear", "overcast"]),
        #("config5_clear_only", ["clear"]),
    ]
    
    for name, weather in exp1_configs:
        run_exp1_config(name, weather)
        
    print(f"\n{'=' * 70}")
    print("  Experiment 1 Pipeline Complete. Analyzing results ...")
    print(f"{'=' * 70}")
    from evaluate import analyze_exp1_results
    analyze_exp1_results()
    
    from visualise import plot_exp1_results
    plot_exp1_results()


def run_analysis() -> None:
    """
    Stub for consolidated analysis (tables + figures).
    """
    print("\n[Main] Running final analysis...")
    print_analysis_tables()
    run_all_tsne()


# ════════════════════════════════════════════════════════════════════════════
# Run all 12 experiments sequentially
# Call this once you are ready to run the full pipeline
# ════════════════════════════════════════════════════════════════════════════

def run_all() -> None:
    """
    Run all 12 experiments in the correct dependency order:

        For each model:
            1. source_only   — must run before finetune
            2. target_only   — independent
            3. finetune      — requires source_only checkpoint
            4. dann          — independent

    Prints analysis tables after all runs complete.
    Generates all t-SNE figures after all runs complete.

    Total: 12 runs across 3 models × 4 experiments.
    Estimated time: 4-5 hours on a 6GB GPU.
    """
    models = ["custom_cnn", "resnet18", "efficientnet_b0"]

    for model_name in models:
        print(f"\n{'#' * 60}")
        print(f"#  Starting all experiments for: {model_name}")
        print(f"{'#' * 60}")

        set_seed()
        run_source_only(model_name)   # must come before finetune

        set_seed()
        run_target_only(model_name)

        set_seed()
        run_finetune(model_name)      # depends on source_only checkpoint

        set_seed()
        run_dann(model_name)

    # ── Post-training analysis ────────────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print("  All 12 runs complete. Generating analysis ...")
    print(f"{'═' * 60}")

    print_analysis_tables()
    run_all_tsne()

    print("\n[Done] Full pipeline complete.")
    print(f"       Results CSV   : see results/all_results.csv")
    print(f"       Figures       : see results/figures/")
    print(f"       Checkpoints   : see checkpoints/")


# ════════════════════════════════════════════════════════════════════════════
# Main entry point
# Reads MODEL_NAME and EXPERIMENT from config.py
# Change those two values to run a specific experiment
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """
    Core entry point. Dispatches based on RUN_MODE in config.py.
    """
    print(f"\n{'═' * 60}")
    print(f"  BDD100K Domain Adaptation Pipeline")
    print(f"{'═' * 60}")
    print(f"  Mode       : {RUN_MODE}")
    print(f"  Device     : {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"{'═' * 60}")

    if LR_GRID_SEARCH:
        set_seed(SEED)
        run_lr_grid_search(MODEL_NAME, EXPERIMENT)
        return

    if RUN_MODE == "single":
        set_seed(SEED)
        run_single(MODEL_NAME, EXPERIMENT)

    elif RUN_MODE == "all_experiments":
        set_seed(SEED)
        run_all_experiments(MODEL_NAME)

    elif RUN_MODE == "all_models":
        set_seed(SEED)
        run_all_models()

    elif RUN_MODE == "all_seeds":
        run_all_seeds()

    elif RUN_MODE == "direction1":
        run_direction1()

    elif RUN_MODE == "exp1":
        run_all_exp1()

    else:
        print(f"[Error] Unknown RUN_MODE: {RUN_MODE}")

    # Optional final analysis
    # run_analysis()


if __name__ == "__main__":
    main()

"""Dependency order matters — source_only must always run before finetune because finetune loads the source_only checkpoint. run_all() handles this automatically. If you run experiments individually just keep this in mind.
Two ways to use main.py — single experiment by setting MODEL_NAME and EXPERIMENT in config.py and running python main.py, or full pipeline by calling run_all() directly. The commented lines at the bottom of main() are the post-training calls you uncomment once everything is done.
set_seed() is called before every experiment — ensures each of the 12 runs starts from the exact same random state regardless of what ran before it."""
