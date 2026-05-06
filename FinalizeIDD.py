# FinalizeIDD.py
import os
import json
import random
import shutil
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
POOL_ROOT  = Path(r"C:\Users\vibha\Downloads\idd-20k-II\idd_pool")
FINAL_ROOT = Path(r"C:\Users\vibha\Downloads\idd-20k-II\idd_final")

# ── Few-Shot Split Configuration ─────────────────────────────────────────────
# Fixed counts rather than ratios — supports 1-shot to 50-shot sweeps.
# The test set gets ALL remaining images, maximizing evaluation robustness.
TRAIN_COUNT = 50   # Support pool (sample K from here at experiment time)
VAL_COUNT   = 50   # Validation set (fixed)
# Test gets everything else (~234 per class at current pool size)

def finalize_dataset():
    print("[INFO] Finalizing IDD Dataset...")
    
    # 1. Gather counts and find bottleneck
    class_files = {}
    for class_dir in POOL_ROOT.iterdir():
        if class_dir.is_dir():
            files = list(class_dir.glob("*.jpg"))
            if len(files) > 0:
                class_files[class_dir.name] = files
    
    if not class_files:
        print("[ERROR] No images found in pool. Did you clean them out?")
        return

    min_count = min(len(f) for f in class_files.values())
    print(f"[INFO] Bottleneck class has {min_count} images. Balancing all classes to this count.")
    
    required = TRAIN_COUNT + VAL_COUNT + 1  # At least 1 test image
    if min_count < required:
        print(f"[ERROR] Need at least {required} images per class but only have {min_count}. Clean your pool more carefully.")
        return

    test_count = min_count - TRAIN_COUNT - VAL_COUNT
    print(f"[INFO] Split — Train: {TRAIN_COUNT} | Val: {VAL_COUNT} | Test: {test_count}")

    # 2. Process each class
    for category, files in class_files.items():
        random.shuffle(files)
        kept_files = files[:min_count]
        
        splits = {
            "train": kept_files[:TRAIN_COUNT],
            "val":   kept_files[TRAIN_COUNT:TRAIN_COUNT + VAL_COUNT],
            "test":  kept_files[TRAIN_COUNT + VAL_COUNT:]
        }
        
        for split_name, split_files in splits.items():
            # IDD is treated as "day" domain to match existing BDD code logic
            dest_dir = FINAL_ROOT / "day" / split_name / category / "images"
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            metadata_records = []
            
            for f in split_files:
                # Copy image
                shutil.copy(f, dest_dir / f.name)
                
                # Create metadata record (matches BDD structure)
                record = {
                    "crop_name": f.name,
                    "original_image": f.name.split('_')[0], # Heuristic
                    "domain": "day",
                    "split": split_name,
                    "category": category
                }
                metadata_records.append(record)
            
            # Write metadata.json for this leaf folder
            meta_path = FINAL_ROOT / "day" / split_name / category / "metadata.json"
            with open(meta_path, 'w', encoding='utf-8') as mf:
                json.dump(metadata_records, mf, indent=2)
                
            print(f"  [OK] {category:<12} | {split_name:<6} | {len(split_files)} images")

    print("\n" + "="*40)
    print("IDD FINALIZATION COMPLETE")
    print(f"Final Dataset: {FINAL_ROOT}")
    print("="*40)

if __name__ == "__main__":
    finalize_dataset()
