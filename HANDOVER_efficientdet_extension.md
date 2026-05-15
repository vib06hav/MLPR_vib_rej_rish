# LLM Handover — EfficientDet Detection Extension
# BDD100K Domain Adaptation Project

---

## 1. What This Project Is (Context for the Incoming LLM)

This repository (`model Experiment/`) studies **day-to-night domain adaptation** for vehicle
classification using the **BDD100K** driving dataset. The core task is classifying cropped
vehicle images into `bus`, `car`, and `truck` while bridging the performance gap between a
model trained on daytime images and tested on nighttime images.

### Key results achieved so far

| Model | Experiment | Accuracy Mean | Macro-F1 Mean | Std |
| :--- | :--- | ---: | ---: | ---: |
| efficientnet_b0 | source_only | 0.7572 | 0.7519 | 0.0116 |
| efficientnet_b0 | dann_warmstart | 0.8340 | 0.8334 | 0.0058 |
| efficientnet_b0 | finetune | 0.8251 | 0.8224 | 0.0048 |

DANN Warmstart is the strongest baseline. Results are averaged across seeds 42, 43, 44.
Full numbers are in `all_results_summary.csv` and `direction1_results_summary.csv`.

### What DANN actually did to the backbone

The EfficientNet-B0 backbone was surgically modified:
- Original ImageNet classifier (1280→1000) was **removed entirely**
- A custom projector `FC(1280→256) + ReLU` was added
- A custom classifier `FC(256→3)` was added (3 = bus/car/truck)

During DANN training, two losses pulled the backbone simultaneously:
1. **Classification loss** — normal gradients, pushes backbone to recognise vehicle types
2. **Domain adversarial loss** — reversed gradients via GRL, pushes backbone to produce
   features that look the same regardless of whether the image is day or night

The t-SNE plots (`run_tsne_bdd.py`) visualise the 256-dim feature vectors.
The `dann_warmstart` checkpoint is the one being considered for transfer.

---

## 2. The New Direction — EfficientDet Detection Extension

### The Research Hypothesis

> A DANN-adapted EfficientNet-B0 backbone provides a domain-invariant vehicle representation
> that serves as a stronger initialisation for night-condition vehicle detectors compared to
> standard ImageNet pretraining — reducing the domain gap before detection training even begins.

The experiment is:
- Train **EfficientDet-D0** (detection model) with ImageNet backbone init → **Baseline**
- Train **EfficientDet-D0** with your DANN checkpoint as backbone init → **Experimental**
- Evaluate both on **night BDD100K detection** (bus, car, truck only)
- Primary metric: **Night mAP** (mean Average Precision)
- Control metric: **Day mAP** (to verify the DANN backbone didn't lose general capability)

### Why EfficientDet-D0 specifically, not YOLOv8

EfficientDet-D0 was chosen because it uses **EfficientNet-B0 as its backbone natively**.
YOLOv8 uses a custom CSP-based architecture (C2f blocks) — not EfficientNet — making direct
weight transfer impossible without a full architectural remap.

EfficientDet-D0 architecture:
```
Input Image
    ↓
EfficientNet-B0 backbone (features blocks 0–8)
    ↓ taps at blocks 3, 5, 7 → P3, P4, P5 feature maps
BiFPN neck (multi-scale feature fusion)
    ↓
Detection head (class + box regression per anchor)
```

Your backbone contributes `features.0` through `features.8`. Everything after that (BiFPN,
head) starts from scratch regardless of init. This is expected and fine.

---

## 3. Raw Data Available on Disk

Everything needed for the detection experiment exists locally:

| Asset | Location |
| :--- | :--- |
| Full BDD100K driving images (100k split) | `C:\Users\vibha\Downloads\archive\bdd100k\bdd100k\images\100k\` |
| Training annotations (JSON, 1.4 GB) | `C:\Users\vibha\Downloads\archive\bdd100k_labels_release\bdd100k\labels\bdd100k_labels_images_train.json` |
| Validation annotations (JSON, 208 MB) | `C:\Users\vibha\Downloads\archive\bdd100k_labels_release\bdd100k\labels\bdd100k_labels_images_val.json` |
| DANN checkpoint (seed 42, best) | `C:\Users\vibha\Downloads\archive\checkpoints_datasetB_final\efficientnet_b0_dann_warmstart_seed42_best.pth` |

The annotation JSONs already contain:
- Full bounding box coordinates per object
- `timeofday` field (`daytime` / `night`) for domain filtering
- Class labels including `bus`, `car`, `truck`

No new data collection is needed.

---

## 4. Unresolved Problems — Full List

### 🔴 Hard Blockers (Must resolve before writing training code)

#### Blocker 1 — Library naming mismatch (timm vs torchvision)

**The problem:**
Your DANN backbone was built with **torchvision's EfficientNet-B0**. Most EfficientDet
implementations use **timm's EfficientNet-B0** internally. Both implement the same
architecture but with different internal naming conventions for layers.

`load_state_dict` with `strict=False` will silently skip all mismatched keys and load nothing
— no error, no warning. You would be running ImageNet weights thinking you loaded DANN weights.

**What to check:**
Inspect the top-level keys in your DANN checkpoint and compare to the EfficientDet
implementation's backbone keys. Look for patterns like:

- torchvision style: `features.0.0.weight`, `features.1.0.block.0.weight`
- timm style: `conv_stem.weight`, `blocks.0.0.conv_pw.weight`

**Resolution path:**
The correct fix is to **match libraries, not remap weights**. Either:
- Find an EfficientDet implementation that explicitly uses torchvision's EfficientNet, OR
- Use `timm`'s EfficientNet in your DANN model and retrain (cleanest long-term)

Positional remapping (zipping source keys and target keys by index) is risky because
torchvision and timm may order parameters differently — you could silently load weights
into wrong layers.

#### Blocker 2 — BiFPN feature pyramid tap points and channel dimensions

**The problem:**
EfficientDet's BiFPN neck taps feature maps from specific intermediate blocks of the backbone.
If the EfficientDet implementation hardcodes different block indices than what your backbone
actually produces, the neck receives semantically wrong inputs.

Additionally, each tap point must produce the correct number of channels:
- Block 3 → 40 channels (P3)
- Block 5 → 112 channels (P4)
- Block 7 → 320 channels (P5)

If the implementation was built for a different backbone version, the BiFPN input projection
layers will have wrong input sizes and will crash immediately.

**Resolution path:**
Before writing any training code, run a dummy forward pass through your backbone with hooks
registered on each block. Print both the spatial resolution and channel dimension at each block.
Cross-reference these against the EfficientDet implementation's expected input dimensions.
This must be verified and confirmed before any training starts.

---

### 🟡 Significant (Will hurt results or scientific validity if ignored)

#### Problem 3 — Crops vs full scenes (training distribution mismatch)

**The problem:**
Your backbone was trained on 224×224 close-up vehicle crops where the vehicle fills most of
the frame. EfficientDet runs on full 512×512 driving scenes where vehicles are small, distant,
and partially occluded.

This is both a **technical** and **scientific validity** problem:
- Technically: the backbone's learned statistics are biased toward "large close-up vehicle"
- Scientifically: any performance delta between your DANN model and the ImageNet baseline
  might reflect this training distribution gap rather than the domain adaptation benefit

**Mitigation:**
- Use progressive unfreezing (last blocks first) to let the backbone re-adapt to full scenes
- Use a much lower learning rate on backbone layers vs BiFPN and head
- Acknowledge this confound explicitly in how you frame results — it cannot be fully eliminated

#### Problem 4 — Batch Normalisation running statistics

**The problem:**
The BN layers in your transferred backbone have `running_mean` and `running_var` baked in from
training on 224×224 crops. Running full 512×512 scenes through frozen BN layers will produce
incorrectly normalised feature maps.

**Resolution:**
Run a BN warmup pass before detection training begins:
- Set backbone to `train()` mode (BN updates running stats in train mode)
- Reset running stats on all BN layers to zero
- Set BN momentum to 0.01 (slow, stable accumulation — default is 0.1)
- Forward pass ~100 batches of detection images with no backprop
- BN running stats now reflect the actual detection data distribution

This is a one-time preprocessing step, takes minutes, and should be done for **BOTH** the
DANN model and the ImageNet baseline to keep the comparison fair.

#### Problem 5 — Freezing strategy and differential learning rates

**The problem:**
If the backbone is unfrozen immediately, the large detection dataset will overwrite your DANN
representations within the first few epochs. You would effectively be training from ImageNet
init despite loading DANN weights.

**Correct approach (signal-driven, not epoch-driven):**
- Phase 1: Freeze backbone entirely, train BiFPN + head only. Monitor validation mAP.
  When val mAP plateaus, move to Phase 2.
- Phase 2: Unfreeze backbone at 10x lower LR than BiFPN/head. Progressive unfreezing
  (last blocks first) is preferable to unfreezing all at once.

Do NOT hardcode epoch numbers — use validation mAP as the signal for phase transitions.

> **CRITICAL:** Both the DANN model and the ImageNet baseline must use the **exact same**
> freezing schedule, BN warmup, and learning rate structure. Any asymmetry in training
> protocol confounds results.

---

### 🟢 Minor (Verify once, then move on)

#### Problem 6 — Input normalisation

Almost certainly both models use ImageNet normalisation (`mean=[0.485, 0.456, 0.406]`,
`std=[0.229, 0.224, 0.225]`). Verify this matches between your data pipeline and the
EfficientDet config. If it matches, no action needed.

#### Problem 7 — Vehicle specialisation of the backbone

DANN pushed the backbone toward vehicle-specific, illumination-invariant features. This means
it may have suppressed features useful for non-vehicle classes (pedestrians, traffic lights).

**Recommendation:** Scope the detection experiment to `bus`, `car`, `truck` only. This keeps
the scientific comparison clean and avoids the backbone's specialisation working against you.
It also keeps the night data volume healthy since vehicle labels are the most dense category
in BDD100K night scenes.

#### Problem 8 — Projector removal effect on block 8

Block 8 of your EfficientNet was jointly trained with the `FC(1280→256)` projector that came
after it. Removing the projector and attaching BiFPN instead means block 8's output is consumed
differently than it was optimised for. No action required — just expect block 8 to take a few
extra epochs to stabilise, which the progressive unfreezing schedule handles naturally.

---

## 5. The Correct Order of Operations

```
1. Verify timm vs torchvision key naming
   → If mismatch: resolve library compatibility (do NOT use positional remapping)
   ↓

2. Verify BiFPN tap points and channel dimensions
   → Hook test on backbone, cross-reference with EfficientDet implementation
   ↓

3. Convert BDD100K to detection format
   → Full images + COCO format bounding boxes
   → Filter: bus/car/truck only
   → Split by timeofday: daytime and night splits
   → Source: bdd100k_labels_images_train.json and _val.json (already on disk)
   → Reuse JSON-parsing logic from Preprocessdata.py as reference
   ↓

4. BN warmup pass (for BOTH models — DANN and ImageNet baseline)
   → Reset BN running stats, forward ~100 batches, no backprop
   ↓

5. Phase 1 training (BOTH models, identical protocol)
   → Backbone frozen, train BiFPN + head only
   → Monitor val mAP, transition when plateau
   ↓

6. Phase 2 training (BOTH models, identical protocol)
   → Unfreeze backbone at 10x lower LR (progressive, last blocks first)
   → Continue until convergence
   ↓

7. Evaluation
   → Night mAP per class (primary metric)
   → Day mAP per class (control)
   → Report delta: DANN init vs ImageNet init on both day and night
```

---

## 6. What a Clean Scientific Result Looks Like

The claim that can be made if the experiment succeeds:

> *"Domain-adversarial pretraining on vehicle classification transfers to vehicle detection,
> improving night-condition mAP by X% relative to ImageNet pretraining, without any additional
> night-specific detection labels. Day mAP is maintained within Y%, confirming that domain
> alignment did not degrade general visual capability."*

If night mAP improves but day mAP drops significantly — the story becomes more nuanced and
must be framed carefully. Both numbers must be reported regardless of outcome.

---

## 7. Files in the Repo Relevant to This Extension

| File | Role in Extension |
| :--- | :--- |
| `models.py` | EfficientNetB0 class definition — `self.features`, `self.avgpool`, `self.projector`, `self.classifier` — source of weights to extract |
| `dann.py` | DANNModel wrapper — use `model.backbone.features.state_dict()` for extraction |
| `config.py` | Paths, normalization settings, checkpoint directories |
| `run_tsne_bdd.py` | Shows exact checkpoint loading pattern — reuse the `load_model()` function |
| `Preprocessdata.py` | Shows BDD100K JSON parsing logic — reuse for detection dataset builder |
| `checkpoints_datasetB_final/` | Contains trained DANN checkpoints for seeds 42, 43, 44 |

---

## 8. Open Decisions That Need to Be Made

1. **Which EfficientDet implementation to use?**
   Options:
   - `rwightman/efficientdet-pytorch` (uses timm — naming mismatch likely)
   - Google's automl repo (TensorFlow — incompatible)
   - Write a minimal custom wrapper around torchvision EfficientNet with a BiFPN neck
     (cleanest, full naming control, most work upfront)

2. **Single seed or three seeds?**
   The classification experiment used seeds 42, 43, 44 for statistical rigour. For a fair
   comparison, the detection experiment should too. This triples compute time.

3. **Frame as completed experiment or proposed future work?**
   If time and compute allow: run the experiment and report concrete mAP numbers.
   If time is limited: frame as a fully specified proposed extension with theoretical
   justification — this is scientifically honest and still adds value to the write-up.

---

*Handover generated: 2026-05-15. Conversation ID: ba368d47-33b2-413e-8743-11d1391eed4d.*
