# config.py
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
DATASET_ROOT   = Path(r"C:\Users\vibha\Downloads\archive")
PROCESSED_ROOT = DATASET_ROOT / "processed_dataset_51k"
RESULTS_DIR    = DATASET_ROOT / "results_51k_label_sweep"
CHECKPOINT_DIR = DATASET_ROOT / "checkpoints_51k_label_sweep"

# ── Dataset ───────────────────────────────────────────────────────────────────
CLASSES        = ["bus", "car", "truck"]   # sorted — index 0,1,2
NUM_CLASSES    = 3
DOMAINS        = ["day", "night"]          # day=0, night=1
SPLITS         = ["train", "val", "test"]
INPUT_SIZE     = 224

CAP            = 1246                      # crops per class per domain after balancing

# ── Extraction Enhancements / Ablations ───────────────────────────────────────
FOLD_DAWN_DUSK = True                      # Map dawn/dusk -> Day (zero-cost data)
CLASS_RATIO_ABLATION = {"car": 4, "truck": 2, "bus": 1} # Relative class volume weights
NIGHT_BUS_MIN_AREA_OVERRIDE = 2500         # Lower size floor strictly for bottleneck
ASYMMETRIC_AUGMENTATION = True             # Stronger jitter/blur applied on night only

ASYMMETRIC_AUGMENTATION = True             # Stronger jitter/blur applied on night only

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE     = 32
MAX_EPOCHS     = 50
EARLY_STOP_PATIENCE = 10
LR_PATIENCE    = 5                         # ReduceLROnPlateau patience
USE_WEIGHTED_SAMPLER = False
# ── Seeds ─────────────────────────────────────────────────────────────────────
SEEDS = [42, 43, 44]
ACTIVE_SEED_INDEX = 0
SEED = SEEDS[ACTIVE_SEED_INDEX]

# ── Learning rates ────────────────────────────────────────────────────────────
LR_HEAD        = 2e-3                      # classifier head — optimized winner
LR_BACKBONE_PRETRAINED = 2e-4             # EfficientNet-B0 backbone — optimized winner
LR_BACKBONE_SCRATCH    = 1e-3             # Custom CNN — trained fully from scratch

BACKBONE_LR_GRID = [5e-5, 1e-4, 2e-4]
HEAD_LR_GRID     = [5e-4, 1e-3, 2e-3]
LR_GRID_SEARCH   = False

# ── Phase 2 Tier 1 Grid section ───────────────────────────────────────────────
LAMBDA_MAX_GRID    = [0.5, 1.0, 1.5]
GAMMA_GRID         = [5.0, 10.0, 20.0]
LAMBDA_GRID_SEARCH = False


# ── Regularisation section ────────────────────────────────────────────────────
WEIGHT_DECAY           = 1e-3              # Optimized winner
GRAD_CLIP_NORM         = 1.0
DROPOUT_RATE           = 0.0
BATCHNORM_IN_PROJECTOR = False

# ── Scheduler section ─────────────────────────────────────────────────────────
LR_SCHEDULER = "plateau"  # alternative: "cosine"

# ── Model ─────────────────────────────────────────────────────────────────────
FEATURE_DIM    = 512                       # Optimized winner
FEATURE_DIM_SEARCH = [128, 256, 512]
RESIZE_OPTIONS     = [224, 256, 320]

# ── Normalisation ─────────────────────────────────────────────────────────────
# Used for ResNet18 and EfficientNet-B0 (pretrained on ImageNet)
IMAGENET_MEAN  = [0.485, 0.456, 0.406]
IMAGENET_STD   = [0.229, 0.224, 0.225]

# Used for Custom CNN (computed from your dataset)
# These will be calculated by dataset.py on first run and cached here
# placeholder — update after compute
DATASET_MEAN = [0.2918, 0.2789, 0.2684]
DATASET_STD  = [0.2266, 0.212, 0.2048]

# ── Augmentation ──────────────────────────────────────────────────────────────
AUG_FLIP_PROB        = 0.5
AUG_ROTATION_DEGREES = 10
AUG_BRIGHTNESS       = 0.2
AUG_CONTRAST         = 0.2
AUG_GRAYSCALE        = False
AUG_GAUSSIAN_BLUR    = False
AUG_TRANSLATE        = False
AUG_PAD_SIZE         = 256               # pad to this before random crop to INPUT_SIZE

# ── DANN ──────────────────────────────────────────────────────────────────────
DANN_LAMBDA_MAX      = 0.5               # lambda grows from 0 → MAX (Winner Tier 1)
DANN_LAMBDA_GAMMA    = 5.0               # growth speed (Winner Tier 1)

DANN_WARMSTART       = True
DANN_SEMI_SUPERVISED = False
DANN_LAMBDA_SCHEDULE = "sigmoid"         # alternatives: "linear", "constant"
DANN_DEEP_CLASSIFIER = False

# ── Domain classifier depth section ───────────────────────────────────────────
DOMAIN_CLASSIFIER_DEPTH = "shallow"      # alternative: "deep"

# ── Experiment control ────────────────────────────────────────────────────────
# Change these two values to switch between the 12 runs

# Options: "custom_cnn" | "resnet18" | "efficientnet_b0"
MODEL_NAME   = "efficientnet_b0"

# Options: "target_only" | "source_only" | "finetune" | "dann"
EXPERIMENT   = "dann"

# ── Run control section ───────────────────────────────────────────────────────
# "single" — runs the one experiment defined by MODEL_NAME and EXPERIMENT
# "all_experiments" — runs all 4 experiments for the one model defined by MODEL_NAME
# "all_models" — runs all 4 experiments for all 3 models sequentially
# "all_seeds" — runs the full 12-run pipeline for all 3 seeds back to back
# "direction1" — overnight target-label-ratio sweep for DANN
# "lambda_grid" — Phase 2 Tier 1: Lambda/Gamma sweep
RUN_MODE = "direction1"

# Models to include in pipeline runs
MODELS_TO_RUN = ["efficientnet_b0"]
SAVE_EPOCH_FEATURES = False
SAVE_FEATURES_EVERY_N = 3    # save features every N epochs (epoch 1 always saved)

# ── Direction 1 (Night Label Ratio Study) ───────────────────────────────────
# Sweeps target-domain label availability from 0% → 100% inside warmstarted DANN.
# Existing baselines (source_only / target_only / finetune) remain unchanged.
DIRECTION1_ACTIVE = True
DIRECTION1_EXPERIMENT_NAME = "direction1_label_ratio"
DIRECTION1_MODELS_TO_RUN = ["efficientnet_b0"]
DIRECTION1_LABEL_RATIOS = [0.00, 0.05, 0.10, 0.25, 0.50, 0.75, 1.00]
DIRECTION1_SAVE_EPOCH_FEATURES = True
DIRECTION1_FEATURE_SAVE_EVERY_N = 2
DIRECTION1_REQUIRE_WARMSTART = True
DIRECTION1_AUTO_SOURCE_ONLY = True

# Effective proportion of labelled NIGHT train samples used in DANN class loss.
# 1.0 = all labelled, 0.0 = no night class labels, intermediate values = partial labels.
DANN_TARGET_LABEL_RATIO = 1.0
UNLABELED_CLASS_INDEX = -100

# ── Experiment 1 (Gap Size Control) ──────────────────────────────────────────
EXP1_ACTIVE       = False                # when False, all existing behavior is unchanged
EXP1_CONFIG_NAME  = "baseline" # appended to CSV/PTH filenames
SOURCE_WEATHER_INCLUDE = None            # None = all, or list: ["clear", "rainy", etc]
SOURCE_SCENE_INCLUDE   = None            # None = all

# ── Results logging ───────────────────────────────────────────────────────────
RESULTS_CSV  = RESULTS_DIR / "all_results.csv"

"""The dataset mean/std placeholders — those two lines marked placeholder need to be computed from your actual training crops once. dataset.py will have a utility function to do that. You run it once, get the values, paste them here, and never touch them again.
The two values to switch experiments — MODEL_NAME and EXPERIMENT at the bottom are the only two lines you ever change between the 12 runs. Everything else reads from them.
DANN lambda constants — gamma=10.0 and max=1.0 are directly from the original DANN paper (Ganin et al. 2016). No guesswork there.
Ready for dataset.py? That's the most complex one so we'll go through it carefully."""
