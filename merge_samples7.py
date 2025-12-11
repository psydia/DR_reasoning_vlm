import json

CLEAN_PATH = "/clean_val_samples_reasoning.json"
REGEN_PATH = "/bad_val_reasoning_samples_regenerated.json"
OUT_PATH = "/final_val_reasoning_samples.json"

# Load clean samples
with open(CLEAN_PATH, "r") as f:
    clean = json.load(f)

# Load regenerated samples
with open(REGEN_PATH, "r") as f:
    regen = json.load(f)

# Merge
final = clean + regen

num_clean = len(clean)
num_regen = len(regen)
num_total = len(final)

print("Clean samples:", num_clean)
print("Regenerated corrected samples:", num_regen)
print("Total merged samples:", num_total)

# Save merged dataset
with open(OUT_PATH, "w") as f:
    json.dump(final, f, indent=2)

print(f"\nSaved merged dataset to {OUT_PATH}")
print(f"Total samples written to final file: {num_total}")