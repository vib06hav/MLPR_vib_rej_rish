# CullPool.py
import os
import random
from pathlib import Path

POOL_ROOT = Path(r"C:\Users\vibha\Downloads\idd-20k-II\idd_pool")
TARGET_COUNT = 500

def cull():
    print(f"[INFO] Culling folders to {TARGET_COUNT} images each...")
    
    for class_dir in POOL_ROOT.iterdir():
        if not class_dir.is_dir():
            continue
            
        files = list(class_dir.glob("*.jpg"))
        count = len(files)
        
        if count <= TARGET_COUNT:
            print(f"  [SKIP] {class_dir.name} already has {count} images.")
            continue
            
        print(f"  [CULL] {class_dir.name}: {count} -> {TARGET_COUNT}")
        
        # Shuffle and pick files to delete
        random.shuffle(files)
        to_delete = files[TARGET_COUNT:]
        
        for f in to_delete:
            f.unlink()
            
    print("[DONE] Pool is now slimmed down and ready for manual cleaning!")

if __name__ == "__main__":
    cull()
