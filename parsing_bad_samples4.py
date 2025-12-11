import json
import re

INPUT_PATH = "/val_reasoning_samples_cleaned.json"
BAD_OUT = "/bad_val_samples_reasoning.json"
GOOD_OUT = "/clean_val_samples_reasoning.json"

# ---------------------------------------------------------
# Minimal explicit contradiction flags (high precision)
# ---------------------------------------------------------
EXPLICIT_FLAGS = [
    r"\bincorrect\b",
    r"\bnot correct\b",
    r"\bwrong\b",
    r"\binconsistent\b",
    r"\binconsistency\b",
    r"\bdoes not match\b",
    r"\bdoesn't match\b",
    r"\bcontradict",   # catches contradict / contradictory / contradicting
]


def is_contradictory(sample):
    """Return True if reasoning contains explicit contradiction keywords."""
    reasoning = json.dumps(sample.get("reasoning", ""), indent=2).lower()

    for flag in EXPLICIT_FLAGS:
        if re.search(flag, reasoning):
            return True

    return False


# ---------------------------------------------------------
# Main Script
# ---------------------------------------------------------

with open(INPUT_PATH, "r") as f:
    data = json.load(f)

bad, good = [], []

for sample in data:
    if is_contradictory(sample):
        bad.append(sample)
    else:
        good.append(sample)

print("Total samples:", len(data))
print("Bad samples:", len(bad))
print("Clean samples:", len(good))

with open(BAD_OUT, "w") as f:
    json.dump(bad, f, indent=2)

with open(GOOD_OUT, "w") as f:
    json.dump(good, f, indent=2)

print(f"\nSaved bad samples to {BAD_OUT}")
print(f"Saved clean samples to {GOOD_OUT}")
