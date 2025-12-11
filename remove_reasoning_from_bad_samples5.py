import json

BAD_IN = "/bad_val_samples_reasoning.json"
BAD_OUT = "/bad_val_samples_for_regen.json"

with open(BAD_IN, "r") as f:
    bad_samples = json.load(f)

cleaned = []
for sample in bad_samples:
    new_sample = {
        "image": sample.get("image"),
        "question": sample.get("question"),
        "answer": sample.get("answer")
    }
    cleaned.append(new_sample)

print("Original bad samples:", len(bad_samples))
print("Prepared for regeneration:", len(cleaned))

with open(BAD_OUT, "w") as f:
    json.dump(cleaned, f, indent=2)

print(f"Saved cleaned bad samples to {BAD_OUT}")
