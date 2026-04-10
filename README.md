# BDD100K Day-to-Night Vehicle Classification and Domain Adaptation

This project studies domain shift in vehicle classification using BDD100K. The core task is to classify cropped vehicle images into `bus`, `car`, and `truck`, while adapting from a daytime source domain to a nighttime target domain.

The main research question is:

How much does performance degrade when a model trained on day images is tested on night images, and how much of that gap can be recovered using supervised fine-tuning and domain adaptation?

The project also includes a follow-up Direction 1 study that sweeps the amount of labeled night data available to DANN from `0%` to `100%`.

## Project Structure

- `Preprocessdata.py`: builds the processed crop dataset from BDD100K annotations and images
- `dataset.py`: dataset loading, transforms, DANN loaders, target-label masking for Direction 1
- `models.py`: model definitions and classifier heads
- `dann.py`: DANN architecture and gradient reversal setup
- `train.py`: standard training and DANN training
- `evaluate.py`: metrics, CSV logging, seed aggregation
- `visualise.py`: curves, feature plots, checkpoint-based visuals, GIF generation
- `main.py`: experiment runner and orchestration entrypoint
- `config.py`: all paths, hyperparameters, experiment switches, and Direction 1 controls
- `run_direction1.bat`: one-click Windows launcher for the Direction 1 overnight sweep

## Dataset Source

The raw data comes from the BDD100K image dataset and label release.

This project uses:

- BDD100K image annotations from the image label JSON files
- The `timeofday` field to split data into `daytime` and `night`
- Vehicle bounding boxes from labeled objects
- Only the three target classes: `bus`, `car`, and `truck`

The preprocessing script assumes the archive is available locally and that `config.py` points to it through:

```python
DATASET_ROOT = Path(r"C:\Users\vibha\Downloads\archive")
```

If you clone this repo on another machine, that path is the first thing you should update.

## Data Preprocessing

All preprocessing logic is implemented in `Preprocessdata.py`.

### Preprocessing Goals

- build a clean vehicle crop dataset from BDD100K
- remove tiny and unreliable boxes
- keep day and night separate as source and target domains
- avoid train/val/test leakage at the image level
- balance classes within each domain and split

### Pipeline Summary

The script runs in stages:

1. Load BDD100K train and val JSON annotation files.
2. Index all source images from the BDD100K image folders.
3. Group entries by `timeofday` into `day` and `night`.
4. Create an image-level `70/15/15` train/val/test split independently for each domain.
5. Extract only `bus`, `car`, and `truck` bounding boxes.
6. Discard boxes with area below `4096 px^2`.
7. Add `15%` padding around each valid bounding box.
8. Resize each crop to `224 x 224`.
9. Save metadata for each crop, including domain, split, class, weather, scene, and original image.
10. Apply class balancing with domain-specific caps while preserving image-level split integrity.

### Important Preprocessing Decisions

#### 1. Area Filtering

The minimum box area was reduced from `9216` to `4096`.

Reason:

- `64 x 64` equivalent crops were still large enough to preserve vehicle structure after resizing
- the higher threshold was discarding too many useful `bus` examples

#### 2. Image-Level Split to Prevent Leakage

Splitting is done at the image level, not the crop level.

That means:

- all crops from one original image stay in the same split
- a source image cannot contribute one crop to train and another to test
- this closes a serious leakage path for cropped-object datasets

#### 3. Domain-Specific Caps Instead of One Global Cap

The script does not force day and night to use the same cap.

Instead it uses asymmetric caps:

- Day train cap per class: `3728`
- Night train cap per class: `1714`
- Validation and test caps are proportional to those train caps

This was chosen because the day domain naturally has more surviving data than the night domain after filtering. Rather than throwing away a huge amount of useful day data, the pipeline preserves that asymmetry and lets DANN handle the domain imbalance during training.

### Final Dataset

From the project summary:

- Pre-cap valid crops: `250,546`
- Images processed: `73,886`
- Final balanced dataset size: `21,132` crops total

Final post-cap class counts per split:

| Class | Day Train | Day Val | Day Test | Night Train | Night Val | Night Test |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bus | 3728 | 785 | 776 | 1246 | 266 | 243 |
| car | 3728 | 785 | 776 | 1246 | 266 | 243 |
| truck | 3728 | 785 | 776 | 1246 | 266 | 243 |

This means:

- classes are balanced within each domain and split
- day remains the richer source domain
- night remains the smaller target domain

## Problem Setup

The experiments are framed as domain adaptation:

- Source domain: `day`
- Target domain: `night`
- Labels: `bus`, `car`, `truck`

The evaluation target is always the night test set.

This lets the project measure:

- the day-to-night domain gap
- how much target supervision helps
- whether adversarial alignment improves transfer

## Models

The codebase supports:

- `custom_cnn`
- `resnet18`
- `efficientnet_b0`

Most of the reported results focus on `resnet18` and `efficientnet_b0`, with EfficientNet-B0 emerging as the strongest backbone.

## Training Setup

Main defaults from `config.py`:

- input size: `224`
- batch size: `32`
- max epochs: `50`
- early stopping patience: `10`
- scheduler: `ReduceLROnPlateau` with patience `5`
- weight decay: `1e-4`
- gradient clipping: `1.0`
- seeds: `42, 43, 44`

Augmentation used for training:

- resize to `256`
- random crop to `224`
- random horizontal flip
- random rotation up to `10` degrees
- brightness jitter
- contrast jitter

Normalization:

- ImageNet stats for pretrained backbones
- dataset-specific stats for the custom CNN

## Baseline Experiments

The main baseline experiments are:

- `source_only`: train on day, test on night
- `target_only`: train on night, test on night
- `finetune`: start from `source_only`, then continue training on labeled night data
- `semi_dann`: warmstarted DANN with the low-label or unlabeled target setting used in the baseline sweep
- `dann_warmstart`: DANN initialized from a source-only checkpoint

### What Each Baseline Means

`source_only`

- measures the raw domain gap
- no target-domain supervision in the classification objective

`target_only`

- gives a supervised target-domain reference point
- useful as a practical upper baseline

`finetune`

- tests whether supervised transfer from a day-trained initialization helps

`dann_warmstart`

- tests whether adversarial domain alignment plus warmstarting improves adaptation

## Baseline Results

From `all_results_summary.csv`:

| Model | Experiment | Accuracy Mean | Accuracy Std | Macro-F1 Mean |
| --- | --- | ---: | ---: | ---: |
| efficientnet_b0 | source_only | 0.7572 | 0.0116 | 0.7519 |
| efficientnet_b0 | target_only | 0.8018 | 0.0088 | 0.8011 |
| efficientnet_b0 | finetune | 0.8251 | 0.0048 | 0.8224 |
| efficientnet_b0 | semi_dann | 0.7819 | 0.0116 | 0.7808 |
| efficientnet_b0 | dann_warmstart | 0.8340 | 0.0058 | 0.8334 |
| resnet18 | source_only | 0.7600 | 0.0117 | 0.7540 |
| resnet18 | target_only | 0.7936 | 0.0009 | 0.7915 |
| resnet18 | finetune | 0.8176 | 0.0155 | 0.8159 |
| resnet18 | semi_dann | 0.7812 | 0.0030 | 0.7815 |
| resnet18 | dann_warmstart | 0.8161 | 0.0310 | 0.8147 |

### Baseline Interpretation

- `source_only` confirms a clear day-to-night domain gap.
- `target_only` shows that direct night supervision substantially improves results.
- `finetune` improves strongly over `source_only`.
- `dann_warmstart` is the strongest overall baseline, especially for EfficientNet-B0.
- EfficientNet-B0 is the best-performing backbone overall.

## Direction 1: Target Label Ratio Sweep

Direction 1 is the main follow-up study in this repo.

It answers:

How does DANN performance change as the amount of labeled night training data increases from none to full supervision?

### Direction 1 Design

Direction 1 uses warmstarted DANN and sweeps:

- `0%`
- `5%`
- `10%`
- `25%`
- `50%`
- `75%`
- `100%`

Implementation details:

- seeds: `42, 43, 44`
- default overnight model: `efficientnet_b0`
- source-only checkpoint is auto-trained first if missing
- target labels are masked deterministically and class-balancely inside `dataset.py`
- unlabeled target samples still contribute to domain loss
- only labeled target samples contribute to target classification loss

### Direction 1 Results

From `direction1_results_summary.csv` for `efficientnet_b0`:

| Target Label Ratio | Accuracy Mean | Accuracy Std | Macro-F1 Mean |
| --- | ---: | ---: | ---: |
| 0.00 | 0.7851 | 0.0199 | 0.7836 |
| 0.05 | 0.7901 | 0.0143 | 0.7888 |
| 0.10 | 0.7993 | 0.0070 | 0.7971 |
| 0.25 | 0.8052 | 0.0118 | 0.8035 |
| 0.50 | 0.8180 | 0.0096 | 0.8167 |
| 0.75 | 0.8262 | 0.0057 | 0.8253 |
| 1.00 | 0.8285 | 0.0041 | 0.8275 |

### Direction 1 Interpretation

The Direction 1 sweep shows a clean trend:

- even `0%` labeled night data already beats `source_only`
- adding target labels improves performance steadily
- the curve starts flattening after about `50%` to `75%`
- `75%` labels is already very close to `100%`

The practical takeaway is that most of the gains can be recovered before full target annotation is reached.

## Visual Outputs

The project includes saved visual analysis outputs for the Direction 1 study.

These include:

- accuracy and macro-F1 ratio curves
- per-ratio checkpoint comparison figures
- checkpoint-based feature-space visualizations
- a ratio-progression GIF built from saved checkpoints

The checkpoint-based GIF is not an epoch-by-epoch training GIF. It is a progression across saved final checkpoints for different target-label ratios.

## How to Run

### 1. Set the dataset path

Update `DATASET_ROOT` in `config.py` if your archive is in a different location.

### 2. Run preprocessing

```powershell
python Preprocessdata.py
```

This creates the processed crop dataset under the archive root.

### 3. Run a single experiment

Set these fields in `config.py`:

- `MODEL_NAME`
- `EXPERIMENT`
- `RUN_MODE = "single"`

Then run:

```powershell
python main.py
```

### 4. Run the full Direction 1 overnight sweep

Set:

```python
RUN_MODE = "direction1"
```

Then either:

```powershell
python main.py
```

or double-click:

```text
run_direction1.bat
```

## Outputs

By default, outputs are written under the archive folder configured in `config.py`:

- checkpoints: `checkpoints/`
- results CSVs: `results/`
- figures and GIFs: `results/figures/`

Typical important files:

- `all_results_summary.csv`
- `direction1_results_summary.csv`
- `direction1_label_ratio_curve.png`

## Reproducibility Notes

- all major experiments use seeds `42`, `43`, and `44`
- image-level splitting avoids crop leakage
- Direction 1 target-label selection is deterministic for a given seed
- source-only checkpoints are used as warmstarts where required

## Current State of the Repo

This repository contains:

- the preprocessing pipeline
- training and evaluation code
- baseline and Direction 1 result summaries
- visualization code for report figures and GIF generation

Large generated artifacts such as processed data, checkpoints, full result dumps, and virtual environments are intentionally excluded from git through `.gitignore`.

## Notes for Future Cleanup

This project currently uses absolute local Windows paths in `config.py`. For a more portable public version, the next improvement would be:

- convert all paths to relative or environment-driven paths
- add a `requirements.txt` or `pyproject.toml`
- document package installation explicitly
- separate raw data, processed data, and experiment outputs more cleanly
