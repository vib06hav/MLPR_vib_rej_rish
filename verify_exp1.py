import config
# Set config BEFORE importing dataset if dataset uses it at module level (it doesn't for filters, but good practice)
config.EXP1_ACTIVE = True
config.SOURCE_WEATHER_INCLUDE = ["clear"]

print("\n--- Verification: Experiment 1 Filtering ---")
from dataset import VehicleDataset

# Test 1: Clear only
# We must pass the include list to the constructor
ds = VehicleDataset(["day"], "train", norm="imagenet", augment=False, 
                    weather_include=config.SOURCE_WEATHER_INCLUDE)
count_clear = len(ds.records)
print(f"Clear only records: {count_clear}")

# Test 2: All weather (reset filter)
config.SOURCE_WEATHER_INCLUDE = None
ds2 = VehicleDataset(["day"], "train", norm="imagenet", augment=False,
                     weather_include=config.SOURCE_WEATHER_INCLUDE)
count_all = len(ds2.records)
print(f"All weather records: {count_all}")

# Verify logic
if count_clear < count_all and count_clear > 0:
    print("\n[SUCCESS] Filtering logic verified correctly.")
    print(f"Reduction: {count_all} -> {count_clear} ({(count_clear/count_all)*100:.1f}%)")
else:
    print("\n[FAILURE] Filtering logic did not behave as expected.")
    print(f"DEBUG: count_clear={count_clear}, count_all={count_all}")

# Check main.py orchestration
import main
if hasattr(main, 'run_all_exp1'):
    print("[SUCCESS] main.run_all_exp1 exists.")
