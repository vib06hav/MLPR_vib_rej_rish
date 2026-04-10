import sys
import os
from pathlib import Path

# Mock config to avoid full pipeline import if possible
class MockConfig:
    EXP1_ACTIVE = True
    SOURCE_WEATHER_INCLUDE = ["clear"]
    SOURCE_SCENE_INCLUDE = None
    DATASET_ROOT = Path(r"C:\Users\vibha\Downloads\archive")
    PROCESSED_ROOT = DATASET_ROOT / "processed_dataset"
    CLASSES = ["bus", "car", "truck"]
    DOMAINS = ["day", "night"]
    SPLITS = ["train", "val", "test"]

# Inject mock config into sys.modules if needed, 
# but dataset.py might already be using the real one.
# Let's just use the real one but be careful.

print("--- Minimal Filtering Test ---")
try:
    from dataset import VehicleDataset
    print("Importing VehicleDataset successful")
    
    # Test with explicit arguments
    ds = VehicleDataset(["day"], "train", norm="imagenet", augment=False, 
                        weather_include=["clear"])
    print(f"Count with 'clear' filter: {len(ds.records)}")
    
    ds_all = VehicleDataset(["day"], "train", norm="imagenet", augment=False)
    print(f"Count without filter: {len(ds_all.records)}")
    
    if len(ds.records) < len(ds_all.records):
        print("SUCCESS: Filtering works.")
    else:
        print("FAILURE: Filtering did not reduce record count.")

except Exception as e:
    import traceback
    traceback.print_exc()
