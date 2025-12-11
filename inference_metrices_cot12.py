import pandas as pd
import json


# ====================================================
# 🔧 Set CSV path here
# ====================================================
CSV_PATH = "/test_predictions.csv"


# ====================================================
# Robust JSON extractor (uses bracket counting)
# ====================================================
def extract_last_json(text):
    """
    Extracts the LAST complete {...} JSON block in a messy text field.
    Works even if multiple JSON blocks exist in the same cell.
    """
    if not isinstance(text, str):
        return {}

    json_blocks = []
    stack = 0
    start = None

    for i, ch in enumerate(text):
        if ch == '{':
            if stack == 0:
                start = i
            stack += 1
        elif ch == '}':
            if stack > 0:
                stack -= 1
                if stack == 0 and start is not None:
                    json_blocks.append(text[start:i+1])
                    start = None

    if not json_blocks:
        return {}

    # Use the LAST complete JSON object
    last_block = json_blocks[-1]

    try:
        return json.loads(last_block)
    except:
        return {}


# ====================================================
# Normalize final_answer for comparison
# ====================================================
def normalize(x):
    return str(x).strip().lower()


# ====================================================
# Direct-answer detection
# ====================================================
reasoning_fields = [
    "observation",
    "interpretation",
    "diagnosis",
    "recommendation"
]

def is_direct_answer(pred):
    """
    A prediction is considered DIRECT ANSWER if:
    - JSON fails to parse
    - ANY reasoning field missing
    - OR reasoning field too short
    """
    if not isinstance(pred, dict) or len(pred) == 0:
        return True

    for f in reasoning_fields:
        if f not in pred:
            return True
        if len(str(pred.get(f, "")).strip()) < 10:
            return True

    return False


# ====================================================
# Metric computation
# ====================================================
def compute_metrics(csv_path):

    df = pd.read_csv(csv_path)

    # ---------------------------------------
    # Parse JSON using robust extractor
    # ---------------------------------------
    df["gt"] = df["target_json"].apply(extract_last_json)
    df["pred"] = df["predicted_json"].apply(extract_last_json)

    df["gt_ans"] = df["gt"].apply(lambda x: normalize(x.get("final_answer", "")))
    df["pred_ans"] = df["pred"].apply(lambda x: normalize(x.get("final_answer", "")))

    # ---------------------------------------
    # 1. Overall accuracy
    # ---------------------------------------
    overall_accuracy = (df["gt_ans"] == df["pred_ans"]).mean()

    # ---------------------------------------
    # 2. Yes/No metrics
    # ---------------------------------------
    yes_no_mask = df["gt_ans"].isin(["yes", "no"])
    df_yn = df[yes_no_mask]

    if len(df_yn) > 0:

        yes_no_accuracy = (df_yn["gt_ans"] == df_yn["pred_ans"]).mean()

        df_yes = df_yn[df_yn["gt_ans"] == "yes"]
        df_no  = df_yn[df_yn["gt_ans"] == "no"]

        yes_accuracy = (df_yes["pred_ans"] == "yes").mean() if len(df_yes) > 0 else None
        no_accuracy  = (df_no["pred_ans"] == "no").mean() if len(df_no) > 0 else None

    else:
        yes_no_accuracy = None
        yes_accuracy = None
        no_accuracy = None
        df_yes = df_no = []

    # ---------------------------------------
    # 3. Direct-answer ratio
    # ---------------------------------------
    df["direct_answer"] = df["pred"].apply(is_direct_answer)
    direct_answer_ratio = df["direct_answer"].mean()

    # ---------------------------------------
    # Return all metrics
    # ---------------------------------------
    return {
        "num_samples": len(df),
        "overall_accuracy": overall_accuracy,

        "num_yes_no_samples": len(df_yn),
        "num_yes_samples": len(df_yes),
        "num_no_samples": len(df_no),

        "yes_no_accuracy": yes_no_accuracy,
        "yes_accuracy": yes_accuracy,
        "no_accuracy": no_accuracy,

        "direct_answer_ratio": direct_answer_ratio,
    }


# ====================================================
# Run the script
# ====================================================
if __name__ == "__main__":

    print(f"\nReading CSV: {CSV_PATH}")
    metrics = compute_metrics(CSV_PATH)

    print("\n========== METRICS ==========")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    print("=============================\n")
