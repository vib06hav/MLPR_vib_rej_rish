# PreprocessIDD.py
import json
import cv2
import numpy as np
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
IDD_ROOT   = Path(r"C:\Users\vibha\Downloads\idd-20k-II")
ANN_DIR    = IDD_ROOT / "idd20kII" / "gtFine"
IMG_DIR    = IDD_ROOT / "idd20kII" / "leftImg8bit"
POOL_ROOT  = IDD_ROOT / "idd_pool"

# ── Preprocessing Constants (Mirrored from BDD Pipeline) ──────────────────────
MIN_AREA      = 50000           # ~224x224 px (Extreme quality, very close vehicles)
PADDING_RATIO = 0.15
RESIZE_TO     = (224, 224)      # (W, H) for cv2.resize

# We include both Base and Novel classes
ALLOWED_CATEGORIES = {"car", "bus", "truck", "autorickshaw", "motorcycle"}

def polygon_to_bbox(polygon):
    """Convert IDD polygon vertices to a standard bounding box."""
    if not polygon or len(polygon) == 0:
        return None
    points = np.array(polygon)
    x1, y1 = np.min(points, axis=0)
    x2, y2 = np.max(points, axis=0)
    return int(x1), int(y1), int(x2), int(y2)

def process_idd():
    print(f"[INFO] Starting IDD Extraction...")
    print(f"[INFO] Min Area: {MIN_AREA} | Padding: {PADDING_RATIO}")
    
    # 1. Map all images for fast lookup
    print("[INFO] Indexing images...")
    image_index = {}
    for img_path in IMG_DIR.rglob("*.jpg"):
        # IDD structure: leftImg8bit/split/folder/filename_leftImg8bit.jpg
        # Annotation structure: gtFine/split/folder/filename_gtFine_polygons.json
        # The common part is the filename before the suffix
        base_name = img_path.name.replace("_leftImg8bit.jpg", "")
        image_index[base_name] = img_path
    
    print(f"[INFO] Found {len(image_index)} images.")

    # 2. Iterate through annotations
    json_files = list(ANN_DIR.rglob("*_polygons.json"))
    print(f"[INFO] Processing {len(json_files)} annotation files...")

    total_crops = 0
    skipped_small = 0
    
    for json_path in json_files:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        base_name = json_path.name.replace("_gtFine_polygons.json", "")
        img_path = image_index.get(base_name)
        
        if not img_path:
            continue

        img = None # Lazy load image only if valid objects found
        
        img_h = data['imgHeight']
        img_w = data['imgWidth']

        for i, obj in enumerate(data['objects']):
            label = obj['label']
            if label not in ALLOWED_CATEGORIES:
                continue
            
            # Convert Polygon to BBox
            bbox = polygon_to_bbox(obj['polygon'])
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            
            # Area Filter
            area = (x2 - x1) * (y2 - y1)
            if area < MIN_AREA:
                skipped_small += 1
                continue
            
            # Apply Padding
            width = x2 - x1
            height = y2 - y1
            pad_x = int(width * PADDING_RATIO)
            pad_y = int(height * PADDING_RATIO)
            
            x1_p = max(0, x1 - pad_x)
            y1_p = max(0, y1 - pad_y)
            x2_p = min(img_w, x2 + pad_x)
            y2_p = min(img_h, y2 + pad_y)

            # Load image if not already loaded for this file
            if img is None:
                img = cv2.imread(str(img_path))
                if img is None: break

            # Crop and Resize
            crop = img[y1_p:y2_p, x1_p:x2_p]
            if crop.size == 0: continue
            
            crop_resized = cv2.resize(crop, RESIZE_TO, interpolation=cv2.INTER_LANCZOS4)
            
            # Save to class-specific pool
            out_dir = POOL_ROOT / label
            out_dir.mkdir(parents=True, exist_ok=True)
            
            crop_name = f"{base_name}_{label}_{i}.jpg"
            cv2.imwrite(str(out_dir / crop_name), crop_resized)
            total_crops += 1

        if (total_crops % 100 == 0) and total_crops > 0:
            print(f"  [PROG] Extracted {total_crops} crops...")

    print("\n" + "="*40)
    print("IDD EXTRACTION COMPLETE")
    print(f"Total Crops Saved: {total_crops}")
    print(f"Small Boxes Skipped: {skipped_small}")
    print(f"Pool Directory: {POOL_ROOT}")
    print("="*40)

if __name__ == "__main__":
    process_idd()
