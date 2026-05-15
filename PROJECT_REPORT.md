# BDD100K Domain Adaptation Project: Final Report

## 1. Project Objective
The primary goal of this project is to solve the **Day-to-Night domain shift** in vehicle classification using the BDD100K dataset. We aim to classify vehicles into three categories (`bus`, `car`, `truck`) while ensuring the model performs reliably in low-light nighttime conditions using **Domain-Adversarial Neural Networks (DANN)**.

---

## 2. Directory & Storage Map

### Workspace Root
`c:\Users\vibha\OneDrive\Desktop\model Experiment\`
Contains all core logic, training scripts, and configuration files.

### Data Archive (Storage)
`C:\Users\vibha\Downloads\archive\`
This is the primary storage for large artifacts.

| Folder | Contents |
| :--- | :--- |
| `bdd100k/` | Raw driving images (100k and 10k sets). |
| `bdd100k_labels_release/` | Original JSON annotation files (1.4GB train, 200MB val). |
| `processed_dataset_51k/` | 224x224 cropped vehicle patches (balanced and filtered). |
| `checkpoints_datasetB_final/` | Best performing model weights for all baseline experiments. |
| `checkpoints_51k_label_sweep/` | Weights for the Direction 1 (0% to 100% labels) study. |
| `results_datasetB_final/` | CSV metrics, confusion matrices, and t-SNE figures. |

---

## 3. Preprocessing Pipeline (`Preprocessdata.py`)
- **Source:** BDD100K raw images and object labels.
- **Filtering:** Minimum area of 4096 px² to ensure vehicle structure is preserved.
- **Splitting:** Performed at the **image level** to prevent leakage (all crops from one image stay in the same split).
- **Balancing:** Applied asymmetric caps (Day: 3728/class, Night: 1714/class) to preserve as much daytime data as possible while maintaining class balance.
- **Output:** 21,132 total balanced crops for the primary study.

---

## 4. Codebase Guide

### Core Logic
- `config.py`: The "Brain." Contains all paths, hyperparams, and experiment flags.
- `dataset.py`: Handles domain-specific loading and target-label masking.
- `models.py`: Architecture definitions for ResNet18 and EfficientNet-B0.
- `dann.py`: Implements the Gradient Reversal Layer (GRL) and Domain Classifier.

### Execution
- `main.py`: The experiment orchestrator.
- `train.py`: Logic for standard and adversarial training.
- `evaluate.py`: Computes Accuracy, F1, ECE, and McNemar's tests.
- `visualise.py`: Generates training curves and GIFs.
- `run_tsne_bdd.py`: Standalone script for feature space mapping.

---

## 5. Experimental Results Summary

### Baseline Performance (EfficientNet-B0)
| Experiment | Accuracy (Mean) | Macro-F1 (Mean) | Std |
| :--- | :--- | :--- | :--- |
| **Source Only** | 0.7572 | 0.7519 | 0.0116 |
| **Target Only** | 0.8018 | 0.8011 | 0.0088 |
| **Finetune** | 0.8251 | 0.8224 | 0.0048 |
| **DANN Warmstart**| **0.8340** | **0.8334** | **0.0058** |

### Direction 1: Label Ratio Sweep
- **Observation:** Adding just 5% night labels significantly boosts performance over 0%.
- **Conclusion:** The performance curve flattens after 75% labels, suggesting full annotation of the target domain is not necessary for high-quality results.

---

## 6. Legacy & Useless Files
The following files are either scratch scripts or related to an adjacent project (IDD) and are not required for the core BDD100K pipeline:

- **IDD Related:** `PreprocessIDD.py`, `FinalizeIDD.py`, `CullPool.py`. (Used for preparing the Indian Driving Dataset).
- **Scratch Scripts:** `minimal_test.py` (logic test), `small.py` (counter script), `verify_exp1.py` (verification tool).
- **Redundant:** `visualise.py` is largely superseded for t-SNE purposes by the more robust `run_tsne_bdd.py`.

---

## 7. Next Steps: EfficientDet Extension
The project is poised to move from **Classification** to **Detection**.
- **Target:** EfficientDet-D0.
- **Hypothesis:** Initialising an object detector with the DANN-trained backbone will yield better night-mAP than ImageNet initialisation.
- **Blocker:** Must resolve `timm` vs `torchvision` naming conventions before weight transfer.
