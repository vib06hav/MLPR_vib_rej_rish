"""
BDD100K Vehicle Crop Preprocessing Pipeline — Stages 1 through 8

Changes from previous version:
  - MIN_AREA lowered from 9216 to 4096 (64x64 px equivalent)
    Justification: 64x64 crop upscaled to 224x224 preserves vehicle structural
    geometry (aspect ratio, roofline, proportions) sufficient for type classification.
    Previous threshold discarded ~60% of viable bus annotations.

  - Stage 8 cap logic changed from global minimum to domain-specific caps:
      DAY_CAP   = 3728  (bottleneck: day bus surviving area filter after split)
      NIGHT_CAP = 1714  (bottleneck: night bus surviving area filter after split)
    This reflects real domain asymmetry rather than forcing artificial equality.
    DANN loader handles the imbalance explicitly at training time.

  - Stage 8 redistribution fixed to respect image-level split boundaries from Stage 3.
    Previously: pooled all crops across train/val/test, resampled, redistributed.
    Problem: crops from the same source image could appear in both train and test.
    Fix: cap is applied per-split respecting the original split assignment of each
    crop's source image. Crops from train-assigned images stay in train. This closes
    the data leakage path.

  - Verbose per-image print in Stage 5/6 loop suppressed with progress counter
    to avoid flooding console on 79k images.
"""

import json
import random
import cv2
from pathlib import Path
from collections import defaultdict
import config

# ── Reproducibility ───────────────────────────────────────────────────────────
MASTER_SEED = 42
random.seed(MASTER_SEED)

# ── Root paths ────────────────────────────────────────────────────────────────
DATASET_ROOT = Path(r"C:\Users\vibha\Downloads\archive")

LABELS_DIR = DATASET_ROOT / "bdd100k_labels_release" / "bdd100k" / "labels"
TRAIN_JSON  = LABELS_DIR / "bdd100k_labels_images_train.json"
VAL_JSON    = LABELS_DIR / "bdd100k_labels_images_val.json"

IMAGE_SCAN_DIRS = [
    DATASET_ROOT / "bdd100k" / "bdd100k" / "images" / "100k" / "train",
    DATASET_ROOT / "bdd100k" / "bdd100k" / "images" / "100k" / "val",
]

# ── Preprocessing constants ───────────────────────────────────────────────────
MIN_AREA      = 4096            # 64x64 px equivalent — lowered from 9216
PADDING_RATIO = 0.15
RESIZE_TO     = (224, 224)      # (W, H) for cv2.resize

ALLOWED_CATEGORIES: set[str] = {"car", "truck", "bus"}

OUTPUT_ROOT = DATASET_ROOT / "processed_dataset"

# ── Asymmetric domain caps (derived from Section 7 of annotation report) ─────
# Day bottleneck   : day bus after 4096 filter + 70/15/15 split = 3728 train
# Night bottleneck : night bus after 4096 filter + 70/15/15 split = 1714 train
# These caps apply to the TRAIN split only.
# Val and test are capped proportionally at 15% of the domain cap.
DAY_CAP_TRAIN   = 3728
NIGHT_CAP_TRAIN = 1714
DAY_CAP_VAL     = int(DAY_CAP_TRAIN   * (15 / 70))   # ~798
NIGHT_CAP_VAL   = int(NIGHT_CAP_TRAIN * (15 / 70))   # ~367
DAY_CAP_TEST    = int(DAY_CAP_TRAIN   * (15 / 70))   # ~798
NIGHT_CAP_TEST  = int(NIGHT_CAP_TRAIN * (15 / 70))   # ~367

DOMAIN_SPLIT_CAPS = {
    "day":   {"train": DAY_CAP_TRAIN,   "val": DAY_CAP_VAL,   "test": DAY_CAP_TEST},
    "night": {"train": NIGHT_CAP_TRAIN, "val": NIGHT_CAP_VAL, "test": NIGHT_CAP_TEST},
}


# ════════════════════════════════════════════════════════════════════════════
# STAGE 1 — JSON Loading + Image Index
# ════════════════════════════════════════════════════════════════════════════

def load_json(path: Path) -> list:
    print(f"[INFO] Loading: {path}")
    if not path.exists():
        raise FileNotFoundError(f"JSON not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"       → {len(data)} entries loaded.")
    return data


train_entries = load_json(TRAIN_JSON)
val_entries   = load_json(VAL_JSON)

all_entries = train_entries + val_entries
print(f"\n[INFO] Total combined entries: {len(all_entries)}")


def count_by_timeofday(entries: list) -> dict:
    counts: dict[str, int] = {}
    for entry in entries:
        tod = entry.get("attributes", {}).get("timeofday", "unknown")
        counts[tod] = counts.get(tod, 0) + 1
    return counts


tod_counts = count_by_timeofday(all_entries)
print("\n[INFO] Entry counts by timeofday:")
print(f"       daytime : {tod_counts.get('daytime', 0)}")
print(f"       night   : {tod_counts.get('night', 0)}")
print(f"       other   : {sum(v for k, v in tod_counts.items() if k not in ('daytime', 'night'))}")
print(f"       (full breakdown: {tod_counts})")


def build_image_index(scan_dirs: list) -> dict:
    index: dict[str, Path] = {}
    for scan_dir in scan_dirs:
        if not scan_dir.exists():
            print(f"[WARN] Scan directory not found, skipping: {scan_dir}")
            continue
        print(f"[INFO] Scanning: {scan_dir}")
        found = 0
        for img_path in scan_dir.rglob("*.jpg"):
            filename = img_path.name
            if filename in index:
                print(f"[WARN] Duplicate filename '{filename}' — overwriting.")
            index[filename] = img_path
            found += 1
        print(f"       → {found} images found.")
    return index


print("\n[INFO] Building image index ...")
image_index = build_image_index(IMAGE_SCAN_DIRS)
print(f"\n[INFO] Total images indexed: {len(image_index)}")


# ════════════════════════════════════════════════════════════════════════════
# STAGE 2 — Day / Night Grouping
# ════════════════════════════════════════════════════════════════════════════

def group_entries_by_timeofday(entries: list) -> tuple:
    day_entries, night_entries, skipped_entries = [], [], []
    for entry in entries:
        tod = entry.get("attributes", {}).get("timeofday", "unknown")
        if tod == "daytime" or (getattr(config, "FOLD_DAWN_DUSK", False) and tod == "dawn/dusk"):
            day_entries.append(entry)
        elif tod == "night":
            night_entries.append(entry)
        else:
            skipped_entries.append(entry)
    return day_entries, night_entries, skipped_entries


print("\n" + "─" * 60)
print("[STAGE 2] Grouping entries by timeofday ...")

day_entries, night_entries, skipped_entries = group_entries_by_timeofday(all_entries)
print(f"\n[INFO] Day    entries : {len(day_entries)}")
print(f"[INFO] Night  entries : {len(night_entries)}")
print(f"[INFO] Skipped entries: {len(skipped_entries)}")


def extract_image_names(entries: list) -> set:
    return {entry["name"] for entry in entries if "name" in entry}


day_image_names:   set = extract_image_names(day_entries)
night_image_names: set = extract_image_names(night_entries)

print(f"\n[INFO] Unique day   image names: {len(day_image_names)}")
print(f"[INFO] Unique night image names: {len(night_image_names)}")

overlap = day_image_names & night_image_names
if overlap:
    print(f"[WARN] {len(overlap)} filename(s) appear in BOTH day and night sets!")
else:
    print("[INFO] Sets are disjoint — no filename belongs to both day and night. ✓")

print("\n[DONE] Stage 2 complete.")


# ════════════════════════════════════════════════════════════════════════════
# STAGE 3 — Image-Level Train / Val / Test Splitting
#
# Split is performed at the IMAGE level, not the crop level.
# This guarantees that all crops from a given source image land in the
# same split — eliminating the data leakage path where the model could
# see partial context from a test image during training.
#
# This split map is the authority for all subsequent crop assignment.
# Stage 8 does NOT redistribute crops across splits — it only downsamples
# within each split bucket.
# ════════════════════════════════════════════════════════════════════════════

def make_split_map(
    image_names: set,
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
    seed:        int   = 42,
) -> dict:
    """
    Assigns each image name to train / val / test.
    Sort before shuffle for deterministic ordering regardless of set iteration order.
    Returns dict: image_name -> "train" | "val" | "test"
    """
    names = sorted(image_names)
    rng = random.Random(seed)
    rng.shuffle(names)

    total     = len(names)
    train_end = int(total * train_ratio)
    val_end   = train_end + int(total * val_ratio)

    split_map = {}
    for name in names[:train_end]:
        split_map[name] = "train"
    for name in names[train_end:val_end]:
        split_map[name] = "val"
    for name in names[val_end:]:
        split_map[name] = "test"

    return split_map


def print_split_summary(domain: str, split_map: dict) -> None:
    total     = len(split_map)
    train_cnt = sum(1 for v in split_map.values() if v == "train")
    val_cnt   = sum(1 for v in split_map.values() if v == "val")
    test_cnt  = sum(1 for v in split_map.values() if v == "test")
    print(f"\n[INFO] {domain.upper()} split summary:")
    print(f"       Total images : {total}")
    print(f"       Train        : {train_cnt}  ({train_cnt/total*100:.1f}%)")
    print(f"       Val          : {val_cnt}  ({val_cnt/total*100:.1f}%)")
    print(f"       Test         : {test_cnt}  ({test_cnt/total*100:.1f}%)")
    print(f"       Sum check    : {train_cnt+val_cnt+test_cnt}  "
          f"{'✓' if train_cnt+val_cnt+test_cnt == total else '✗ MISMATCH!'}")


def verify_split_integrity(domain: str, split_map: dict) -> None:
    buckets = {"train": set(), "val": set(), "test": set()}
    for name, split in split_map.items():
        buckets[split].add(name)
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    all_clean = True
    for a, b in pairs:
        ov = buckets[a] & buckets[b]
        if ov:
            print(f"[WARN] {domain}: {len(ov)} image(s) overlap between '{a}' and '{b}'!")
            all_clean = False
    if all_clean:
        print(f"[INFO] {domain.upper()}: No overlap between splits. ✓")
    total_in = sum(len(v) for v in buckets.values())
    if total_in == len(split_map):
        print(f"[INFO] {domain.upper()}: All {len(split_map)} images accounted for. ✓")
    else:
        print(f"[WARN] {domain.upper()}: Expected {len(split_map)} but got {total_in}!")


print("\n" + "─" * 60)
print("[STAGE 3] Creating image-level train / val / test splits ...")

day_split_map   = make_split_map(day_image_names,   seed=MASTER_SEED)
night_split_map = make_split_map(night_image_names, seed=MASTER_SEED)

print_split_summary("day",   day_split_map)
print_split_summary("night", night_split_map)
print()
verify_split_integrity("day",   day_split_map)
verify_split_integrity("night", night_split_map)

print("\n[DONE] Stage 3 complete.")


# ════════════════════════════════════════════════════════════════════════════
# STAGE 5 / 6 — Crop Extraction + Metadata Writing
# ════════════════════════════════════════════════════════════════════════════

def get_domain(entry: dict):
    tod = entry.get("attributes", {}).get("timeofday", "")
    if tod == "daytime" or (getattr(config, "FOLD_DAWN_DUSK", False) and tod == "dawn/dusk"):
        return "day"
    if tod == "night":
        return "night"
    return None


def get_split(image_name: str, domain: str):
    if domain == "day":
        return day_split_map.get(image_name)
    if domain == "night":
        return night_split_map.get(image_name)
    return None


def get_output_dir(domain: str, split: str, category: str) -> Path:
    out_dir = OUTPUT_ROOT / domain / split / category / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def process_entry(
    entry:                 dict,
    domain:                str,
    split:                 str,
    small_box_counter:     list,
    invalid_box_counter:   list,
    unknown_cat_counter:   list,
    metadata_store:        dict,
) -> int:
    """
    Extract all valid vehicle crops from one JSON entry.
    Applies MIN_AREA filter, padding, resize, saves crops, appends metadata.
    Returns count of crops saved.
    """
    image_name = entry.get("name", "")

    img_path = image_index.get(image_name)
    if img_path is None:
        return 0

    img = cv2.imread(str(img_path))
    if img is None:
        return 0

    img_height, img_width = img.shape[:2]
    labels    = entry.get("labels") or []
    basename  = Path(image_name).stem
    img_attrs = entry.get("attributes") or {}

    crops_saved = 0
    vehicle_idx = 0

    for label in labels:
        category = label.get("category", "")

        if category not in ALLOWED_CATEGORIES:
            unknown_cat_counter[0] += 1
            continue

        vehicle_idx += 1

        box2d = label.get("box2d")
        if box2d is None:
            invalid_box_counter[0] += 1
            continue

        x1 = int(box2d["x1"])
        y1 = int(box2d["y1"])
        x2 = int(box2d["x2"])
        y2 = int(box2d["y2"])

        if x2 <= x1 or y2 <= y1:
            invalid_box_counter[0] += 1
            continue

        area = (x2 - x1) * (y2 - y1)
        
        current_min_area = MIN_AREA
        if domain == "night" and category == "bus" and getattr(config, "NIGHT_BUS_MIN_AREA_OVERRIDE", None):
            current_min_area = config.NIGHT_BUS_MIN_AREA_OVERRIDE
            
        if area < current_min_area:
            small_box_counter[0] += 1
            continue

        width  = x2 - x1
        height = y2 - y1
        pad_x  = int(width  * PADDING_RATIO)
        pad_y  = int(height * PADDING_RATIO)

        x1_p = max(0,          x1 - pad_x)
        y1_p = max(0,          y1 - pad_y)
        x2_p = min(img_width,  x2 + pad_x)
        y2_p = min(img_height, y2 + pad_y)

        if x2_p <= x1_p or y2_p <= y1_p:
            invalid_box_counter[0] += 1
            continue

        crop = img[y1_p:y2_p, x1_p:x2_p]
        if crop.size == 0:
            invalid_box_counter[0] += 1
            continue

        crop_resized = cv2.resize(crop, RESIZE_TO, interpolation=cv2.INTER_LANCZOS4)

        out_dir   = get_output_dir(domain, split, category)
        crop_name = f"{basename}_{category}{vehicle_idx}.jpg"
        out_path  = out_dir / crop_name

        cv2.imwrite(str(out_path), crop_resized)
        crops_saved += 1

        label_attrs = label.get("attributes") or {}

        entry_dict = {
            "crop_name":      crop_name,
            "original_image": image_name,
            "domain":         domain,
            "split":          split,
            "weather":        img_attrs.get("weather"),
            "scene":          img_attrs.get("scene"),
            "occluded":       label_attrs.get("occluded"),
            "truncated":      label_attrs.get("truncated"),
            "bbox_area":      area,
            "bbox_original":  {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "bbox_padded":    {"x1": x1_p, "y1": y1_p, "x2": x2_p, "y2": y2_p},
        }

        metadata_store[domain][split][category].append(entry_dict)

    return crops_saved


# ── Counters ──────────────────────────────────────────────────────────────────
small_box_count        = [0]
invalid_box_count      = [0]
unknown_category_count = [0]

_empty_cat_dict = lambda: {cat: [] for cat in ALLOWED_CATEGORIES}

metadata_store: dict = {
    "day":   {"train": _empty_cat_dict(), "val": _empty_cat_dict(), "test": _empty_cat_dict()},
    "night": {"train": _empty_cat_dict(), "val": _empty_cat_dict(), "test": _empty_cat_dict()},
}

print("\n" + "─" * 60)
print(f"[STAGE 5/6] Crop extraction — {len(all_entries)} entries ...")
print(f"[INFO] MIN_AREA     : {MIN_AREA} px²  (was 9216)")
print(f"[INFO] PADDING      : {PADDING_RATIO}")
print(f"[INFO] RESIZE_TO    : {RESIZE_TO}")
print(f"[INFO] OUTPUT_ROOT  : {OUTPUT_ROOT}")

total_processed   = 0
total_crops_saved = 0
MAX_IMAGES        = len(all_entries)

for i, entry in enumerate(all_entries[:MAX_IMAGES]):

    image_name = entry.get("name", "")
    domain     = get_domain(entry)
    if domain is None:
        continue

    split = get_split(image_name, domain)
    if split is None:
        continue

    n_crops = process_entry(
        entry, domain, split,
        small_box_count, invalid_box_count,
        unknown_category_count, metadata_store,
    )

    total_processed   += 1
    total_crops_saved += n_crops

    # Progress print every 5000 images instead of per-image
    if (i + 1) % 5000 == 0:
        print(f"  [PROG] Processed {i+1}/{MAX_IMAGES} entries | crops so far: {total_crops_saved}")

print(f"\n[STAGE 5/6] Complete.")
print(f"  Images processed  : {total_processed}")
print(f"  Crops saved       : {total_crops_saved}")
print(f"  Small boxes skip  : {small_box_count[0]}")
print(f"  Invalid boxes skip: {invalid_box_count[0]}")
print(f"  Unknown cats skip : {unknown_category_count[0]}")

# ── Write initial metadata.json files (pre-cap) ───────────────────────────────
print(f"\n[STAGE 6] Writing pre-cap metadata.json files ...")
meta_files_written = 0
for domain in metadata_store:
    for split in metadata_store[domain]:
        for category, records in metadata_store[domain][split].items():
            if not records:
                continue
            meta_dir  = OUTPUT_ROOT / domain / split / category
            meta_dir.mkdir(parents=True, exist_ok=True)
            meta_path = meta_dir / "metadata.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2)
            meta_files_written += 1
print(f"  Metadata files written: {meta_files_written}")


# ════════════════════════════════════════════════════════════════════════════
# STAGE 7 — Summary Report
# ════════════════════════════════════════════════════════════════════════════

def compute_summary_stats(metadata_store, total_processed, total_crops_saved,
                          small_box_count, invalid_box_count,
                          unknown_category_count, skipped_tod_count) -> dict:
    per_class:   dict = {}
    per_domain:  dict = {}
    per_weather: dict = {}

    for domain in metadata_store:
        for split in metadata_store[domain]:
            for category, records in metadata_store[domain][split].items():
                for rec in records:
                    per_class[category] = per_class.get(category, 0) + 1
                    dom = rec.get("domain", "unknown")
                    per_domain[dom] = per_domain.get(dom, 0) + 1
                    weather = rec.get("weather") or "unknown"
                    per_weather[weather] = per_weather.get(weather, 0) + 1

    return {
        "total_processed_images":   total_processed,
        "total_vehicles_extracted": total_crops_saved,
        "per_class":                per_class,
        "per_domain":               per_domain,
        "per_weather":              per_weather,
        "small_boxes_discarded":    small_box_count[0],
        "invalid_boxes_discarded":  invalid_box_count[0],
        "unknown_categories_seen":  unknown_category_count[0],
        "skipped_timeofday":        skipped_tod_count,
    }


def format_summary(stats: dict) -> str:
    sep  = "=" * 60
    sep2 = "-" * 40
    lines = []
    lines.append(sep)
    lines.append("  BDD100K Vehicle Crop Dataset — Summary Report")
    lines.append(f"  Area filter : >= {MIN_AREA} px²")
    lines.append(f"  Day cap     : train={DAY_CAP_TRAIN} val={DAY_CAP_VAL} test={DAY_CAP_TEST}")
    lines.append(f"  Night cap   : train={NIGHT_CAP_TRAIN} val={NIGHT_CAP_VAL} test={NIGHT_CAP_TEST}")
    lines.append(sep)
    lines.append("")
    lines.append("EXTRACTION STATS (pre-cap)")
    lines.append(sep2)
    lines.append(f"  Total images processed    : {stats['total_processed_images']}")
    lines.append(f"  Total vehicles extracted  : {stats['total_vehicles_extracted']}")
    lines.append("")
    lines.append("  Per-class counts:")
    for cat in sorted(stats["per_class"]):
        lines.append(f"    {cat:<10} : {stats['per_class'][cat]}")
    lines.append("")
    lines.append("  Per-domain counts:")
    for dom in sorted(stats["per_domain"]):
        lines.append(f"    {dom:<10} : {stats['per_domain'][dom]}")
    lines.append("")
    lines.append("  Per-weather counts:")
    for weather in sorted(stats["per_weather"]):
        lines.append(f"    {weather:<20} : {stats['per_weather'][weather]}")
    lines.append("")
    lines.append("DISCARD STATS")
    lines.append(sep2)
    lines.append(f"  Small boxes (< {MIN_AREA} px²) : {stats['small_boxes_discarded']}")
    lines.append(f"  Invalid / degenerate boxes  : {stats['invalid_boxes_discarded']}")
    lines.append(f"  Unknown categories seen     : {stats['unknown_categories_seen']}")
    lines.append(f"  Skipped time-of-day entries : {stats['skipped_timeofday']}")
    lines.append("")
    lines.append(sep)
    lines.append("  END OF REPORT")
    lines.append(sep)
    return "\n".join(lines)


skipped_tod_count = sum(1 for e in all_entries[:MAX_IMAGES] if get_domain(e) is None)

stats        = compute_summary_stats(
    metadata_store, total_processed, total_crops_saved,
    small_box_count, invalid_box_count,
    unknown_category_count, skipped_tod_count,
)
summary_text = format_summary(stats)
print("\n" + summary_text)

summary_path = OUTPUT_ROOT / "dataset_summary_precap.txt"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
with summary_path.open("w", encoding="utf-8") as f:
    f.write(summary_text + "\n")
print(f"\n[INFO] Pre-cap summary written to: {summary_path}")

print("\n[DONE] Stage 7 complete.")


# ════════════════════════════════════════════════════════════════════════════
# STAGE 8 — Class Balancing with Asymmetric Domain Caps
#
# Key design decisions:
#
# 1. ASYMMETRIC CAPS — day and night are capped independently.
#    Day is richer (day bus train ~3728) so day uses a higher cap.
#    Night is the scarce domain (night bus train ~1714) so night uses lower cap.
#    This reflects real domain asymmetry and avoids throwing away useful day data.
#
# 2. IMAGE-LEVEL LEAKAGE FIX — caps are applied per split, not by pooling
#    across splits. Each crop already has the correct split assignment from
#    Stage 3 (inherited from its source image's split_map entry). We never
#    move crops between splits. We only downsample within each split bucket.
#    This guarantees that all crops from source image X remain in whichever
#    split X was assigned to in Stage 3 — no leakage possible.
#
# 3. PER-CLASS BALANCE WITHIN EACH SPLIT — after downsampling, every
#    (domain, split, category) bucket is capped to the same count so that
#    class balance is maintained within every split independently.
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("[STAGE 8] Asymmetric class balancing ...")
print(f"  Day   caps — train: {DAY_CAP_TRAIN}  val: {DAY_CAP_VAL}  test: {DAY_CAP_TEST}")
print(f"  Night caps — train: {NIGHT_CAP_TRAIN}  val: {NIGHT_CAP_VAL}  test: {NIGHT_CAP_TEST}")

# ── 8a. Delete stale motor/bike files if any remain from previous runs ────────
STALE_CATEGORIES = {"motor", "bike"}
stale_deleted = 0
for domain in ("day", "night"):
    for split in ("train", "val", "test"):
        for cat in STALE_CATEGORIES:
            stale_dir = OUTPUT_ROOT / domain / split / cat / "images"
            if stale_dir.exists():
                for f in list(stale_dir.glob("*.jpg")):
                    f.unlink()
                    stale_deleted += 1
                try:
                    stale_dir.rmdir()
                    stale_dir.parent.rmdir()
                except OSError:
                    pass
print(f"\n  [CLEAN] Deleted {stale_deleted} stale motor/bike files.")

# ── 8b. Print pre-cap counts per (domain, split, category) ───────────────────
print("\n  [COUNT] Pre-cap counts per domain/split/category:")
print(f"  {'domain':<8} {'split':<8} {'bus':>8} {'car':>8} {'truck':>8}")
for domain in ("day", "night"):
    for split in ("train", "val", "test"):
        row = f"  {domain:<8} {split:<8}"
        for cat in ("bus", "car", "truck"):
            n = len(metadata_store[domain][split][cat])
            row += f"  {n:>8}"
        print(row)

# ── 8c. Apply caps per (domain, split, category) — NO cross-split pooling ─────
#
# Ablation: apply CLASS_RATIO_ABLATION and stratified priority sampling.
# If config.CLASS_RATIO_ABLATION is not set, fallback to global min cap.

random.seed(MASTER_SEED)

total_kept    = 0
total_removed = 0

def stratified_priority_sample(records, k):
    if len(records) <= k:
        return records
    
    # Sort by area
    records_sorted = sorted(records, key=lambda r: r.get("bbox_area", 0))
    
    # Bin into 4 size buckets
    bucket_count = 4
    buckets = [[] for _ in range(bucket_count)]
    for i, r in enumerate(records_sorted):
        bucket_idx = min(bucket_count - 1, int(i / len(records_sorted) * bucket_count))
        # Prioritize hard cases by placing them at the beginning
        if r.get("occluded") or r.get("truncated"):
            buckets[bucket_idx].insert(0, r)
        else:
            buckets[bucket_idx].append(r)
            
    kept = []
    # Pull from buckets round-robin
    idx = 0
    while len(kept) < k:
        b_idx = idx % bucket_count
        if buckets[b_idx]:
            kept.append(buckets[b_idx].pop(0))
        idx += 1
        
    return kept

for domain in ("day", "night"):
    for split in ("train", "val", "test"):

        target_cap = DOMAIN_SPLIT_CAPS[domain][split]
        class_ratios = getattr(config, "CLASS_RATIO_ABLATION", {"bus": 1, "car": 1, "truck": 1})
        
        # Calculate maximum possible multiplier
        # For each class, multiplier can't exceed actual / ratio
        max_multiplier = target_cap # fallback default
        for cat in ALLOWED_CATEGORIES:
            records = metadata_store[domain][split][cat]
            actual = len(records)
            ratio = class_ratios.get(cat, 1)
            limit = actual / ratio
            if cat == "bus":
                max_multiplier = limit
            else:
                max_multiplier = min(max_multiplier, limit)
                
        # The base amount refers to ratio=1
        base_amount = int(min(max_multiplier, target_cap))
        
        print(f"\n  [{domain}/{split}] target_cap={target_cap} base_amount={base_amount}")

        for cat in sorted(ALLOWED_CATEGORIES):
            records = metadata_store[domain][split][cat]
            ratio = class_ratios.get(cat, 1)
            effective_cap = int(base_amount * ratio)

            if len(records) > effective_cap:
                kept    = stratified_priority_sample(records, effective_cap)
                removed = [r for r in records if r not in kept]

                # Delete image files for removed records
                for rec in removed:
                    img_path = (OUTPUT_ROOT / domain / split / cat
                                / "images" / rec["crop_name"])
                    if img_path.exists():
                        img_path.unlink()
                        total_removed += 1

                metadata_store[domain][split][cat] = kept
            else:
                kept = records

            total_kept += len(kept)
            print(f"    {cat:<8} : kept {len(kept)}  (cap was {effective_cap})")

print(f"\n  [TRIM] Total kept    : {total_kept}")
print(f"  [TRIM] Total removed : {total_removed}")

# ── 8d. Rewrite all metadata.json files with capped records ──────────────────
print(f"\n  [META] Rewriting metadata.json files after capping ...")
meta_rewritten = 0
for domain in metadata_store:
    for split in metadata_store[domain]:
        for category, records in metadata_store[domain][split].items():
            meta_dir  = OUTPUT_ROOT / domain / split / category
            meta_dir.mkdir(parents=True, exist_ok=True)
            meta_path = meta_dir / "metadata.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2)
            meta_rewritten += 1
print(f"  Rewrote {meta_rewritten} metadata.json files.")

# ── 8e. Final balance verification ───────────────────────────────────────────
print(f"\n  [FINAL] Post-cap balance verification:")
print(f"  {'category':<10}  {'day/train':>10}  {'day/val':>8}  {'day/test':>9}"
      f"  {'night/train':>12}  {'night/val':>10}  {'night/test':>11}")

grand_total = 0
for cat in sorted(ALLOWED_CATEGORIES):
    row = f"  {cat:<10}"
    for domain in ("day", "night"):
        for split in ("train", "val", "test"):
            n = len(metadata_store[domain][split][cat])
            row += f"  {n:>10}"
            grand_total += n
    print(row)

print(f"\n  [FINAL] Grand total crops in balanced dataset: {grand_total}")

# ── 8f. Verify no cross-split leakage ────────────────────────────────────────
print(f"\n  [VERIFY] Checking image-level split integrity ...")
for domain in ("day", "night"):
    split_to_images: dict = defaultdict(set)
    for split in ("train", "val", "test"):
        for cat in ALLOWED_CATEGORIES:
            for rec in metadata_store[domain][split][cat]:
                split_to_images[split].add(rec["original_image"])

    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    domain_clean = True
    for a, b in pairs:
        ov = split_to_images[a] & split_to_images[b]
        if ov:
            print(f"  [WARN] {domain}: {len(ov)} source image(s) appear in both "
                  f"'{a}' and '{b}' splits!")
            domain_clean = False
    if domain_clean:
        print(f"  [OK] {domain}: No source image appears in more than one split. ✓")

# ── 8g. Write final post-cap summary ─────────────────────────────────────────
final_summary_lines = [
    "=" * 60,
    "  BDD100K Vehicle Crop Dataset — Post-Cap Summary",
    f"  Area filter : >= {MIN_AREA} px²",
    f"  Day   caps  : train={DAY_CAP_TRAIN} val={DAY_CAP_VAL} test={DAY_CAP_TEST}",
    f"  Night caps  : train={NIGHT_CAP_TRAIN} val={NIGHT_CAP_VAL} test={NIGHT_CAP_TEST}",
    "=" * 60,
    "",
    f"  {'category':<10}  {'day/train':>10}  {'day/val':>8}  {'day/test':>9}"
    f"  {'night/train':>12}  {'night/val':>10}  {'night/test':>11}",
]
for cat in sorted(ALLOWED_CATEGORIES):
    row = f"  {cat:<10}"
    for domain in ("day", "night"):
        for split in ("train", "val", "test"):
            n = len(metadata_store[domain][split][cat])
            row += f"  {n:>10}"
    final_summary_lines.append(row)

final_summary_lines += [
    "",
    f"  Grand total: {grand_total}",
    "",
    "=" * 60,
    "  END OF REPORT",
    "=" * 60,
]

final_summary_path = OUTPUT_ROOT / "dataset_summary_postcap.txt"
with final_summary_path.open("w", encoding="utf-8") as f:
    f.write("\n".join(final_summary_lines) + "\n")

print(f"\n  Post-cap summary written to: {final_summary_path}")
print("\n[DONE] Stage 8 complete — dataset is balanced and ready for training.")
print(f"\n  Config summary:")
print(f"    MIN_AREA        : {MIN_AREA} px² (was 9216)")
print(f"    Day train cap   : {DAY_CAP_TRAIN} per class")
print(f"    Night train cap : {NIGHT_CAP_TRAIN} per class")
print(f"    Split leakage   : closed — image-level split respected throughout")
print(f"    Total crops     : {grand_total}")