# data_prep_2.py
# -------------------------------------------------------------
# Generate region-aware dataset for DME VQA project.
#  - Resizes fundus & mask to 224×224
#  - Detects lesion location (in 3x3 grid)
#  - Updates question text accordingly
#  - Keeps 0/1/2 grading questions unchanged
# -------------------------------------------------------------

import os, json
import numpy as np
from PIL import Image
from tqdm import tqdm

# --- [1] CONFIGURATION ---
ROOT = "/dme_vqa"
SPLIT = "train"  # or "val", "test"
IMG_ROOT = os.path.join(ROOT, "visual", SPLIT)
MASK_ROOT = os.path.join(ROOT, "masks", SPLIT, "maskA")
QA_PATH = os.path.join(ROOT, "qa", f"{SPLIT}qa.json")

OUT_PATH = os.path.join(ROOT, f"{SPLIT}_region_aware.json")

# --- [2] REGION LABEL MAP ---
REGION_NAMES = [
    "top-left", "top-middle", "top-right",
    "middle-left", "center", "middle-right",
    "bottom-left", "bottom-middle", "bottom-right"
]


# --- [3] FUNCTION TO DETECT LESION REGION ---
def detect_region(mask_path):
    try:
        mask = Image.open(mask_path).convert("L").resize((224, 224))
        mask_np = np.array(mask) / 255.0
        h, w = mask_np.shape
        h3, w3 = h // 3, w // 3

        sums = []
        for i in range(3):
            for j in range(3):
                cell = mask_np[i*h3:(i+1)*h3, j*w3:(j+1)*w3]
                sums.append(cell.sum())

        region_index = int(np.argmax(sums))
        return REGION_NAMES[region_index]
    except Exception as e:
        print(f"⚠️ Error processing mask {mask_path}: {e}")
        return None


# --- [4] LOAD DATA ---
with open(QA_PATH, "r") as f:
    data = json.load(f)

print(f"✅ Loaded {len(data)} QA samples from {QA_PATH}")

updated_records = []

# --- [5] PROCESS EACH ENTRY ---
for item in tqdm(data, desc="Processing samples"):
    question = item["question"].strip().lower()
    answer = item["answer"]

    # Initialize updated record
    updated_item = item.copy()

    # Handle only region-based binary questions
    if "in this region" in question and answer.lower() in ["yes", "no"]:
        img_name = item.get("image_name") or os.path.basename(item["image"])
        mask_name = item.get("mask_name") or img_name.replace(".jpg", ".tif")
        mask_path = os.path.join(MASK_ROOT, mask_name)

        if os.path.exists(mask_path):
            region_name = detect_region(mask_path)
            if region_name:
                updated_question = item["question"].replace(
                    "in this region", f"in the {region_name} region"
                )
                updated_item["question"] = updated_question
                updated_item["region"] = region_name
        else:
            print(f"⚠️ Mask not found for {img_name}")
            updated_item["region"] = None

    # For numeric (0/1/2) or non-region questions → keep as is
    updated_records.append(updated_item)

# --- [6] SAVE NEW JSON ---
with open(OUT_PATH, "w") as f:
    json.dump(updated_records, f, indent=2)

print(f"💾 Saved {len(updated_records)} region-aware samples to {OUT_PATH}")
print("✅ Processing complete!")

