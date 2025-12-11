import os
import json
import csv
from tqdm import tqdm
from PIL import Image
import torch

# project imports
from src.dataset import QwenVLJsonlDataset
from src.model_wrapper import QwenVLModelWrapper, ModelSettings, LoRASettings
from transformers import AutoProcessor
from peft import PeftModel


# ============================================================
# HARD-CODE PATHS
# ============================================================
MODEL_CFG = "/configs/model.yaml"
CHECKPOINT_DIR = "/outputs_qwen_sft/checkpoints/best_model"
TEST_JSONL = "/Qwen_reasoning_samples_dme/qwen_test.jsonl"
IMAGE_ROOT = "/DME_VQA_Dataset/dme_vqa/visual/test"
OUTPUT_DIR = "/results_qwen_sft_test"


# ============================================================
# YAML Loader
# ============================================================
def load_yaml(path):
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ============================================================
# Build Model Settings
# ============================================================
def build_model_settings(model_cfg):

    # LoRA config
    lora_cfg = None
    lora_section = model_cfg.get("lora", {})

    if lora_section.get("enabled", False):
        lora_cfg = LoRASettings(
            enabled=True,
            r=lora_section.get("r", 8),
            alpha=lora_section.get("alpha", 16),
            dropout=lora_section.get("dropout", 0.05),
            target_modules=lora_section.get(
                "target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"]
            ),
            lora_bias=lora_section.get("lora_bias", "none"),
            task_type=lora_section.get("task_type", "CAUSAL_LM"),
        )

    dtype_cfg = model_cfg.get("dtype", {})
    vision_cfg = model_cfg.get("vision", {})
    gc_cfg = model_cfg.get("gradient_checkpointing", {})

    return ModelSettings(
        model_name=model_cfg["model"]["name"],
        use_bf16=dtype_cfg.get("use_bf16", True),
        use_fp16=dtype_cfg.get("use_fp16", False),
        freeze_vision=vision_cfg.get("freeze_vision", True),
        gradient_checkpointing=gc_cfg.get("enabled", True),
        lora=lora_cfg,
    )


# ============================================================
# Reasoning Polarity
# ============================================================
def extract_reasoning_polarity(text):
    t = text.lower()

    neg_terms = ["no ", "absent", "not seen", "without"]
    pos_terms = ["present", "seen", "visible", "identified", "observed"]

    for n in neg_terms:
        if n in t:
            return 0

    for p in pos_terms:
        if p in t:
            return 1

    return -1


# ============================================================
# Load model correctly (NO double LoRA)
# ============================================================
def load_model():

    model_cfg_dict = load_yaml(MODEL_CFG)

    # 🔴 Disable LoRA here — we will load real LoRA weights from checkpoint
    if "lora" in model_cfg_dict:
        model_cfg_dict["lora"]["enabled"] = False

    model_settings = build_model_settings(model_cfg_dict)

    # Load base model only
    print("\n[Inference] Loading base Qwen3-VL model (without LoRA)...")
    model_wrapper = QwenVLModelWrapper(model_settings)
    processor = model_wrapper.processor

    # Attach LoRA adapters
    print(f"[Inference] Loading LoRA fine-tuned checkpoint: {CHECKPOINT_DIR}")
    model_wrapper.model = PeftModel.from_pretrained(
        model_wrapper.model,
        CHECKPOINT_DIR
    )

    model_wrapper.model.eval()
    return model_wrapper, processor


# ============================================================
# Inference loop
# ============================================================
def run_inference(model_wrapper, processor, test_dataset):

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    predictions = []
    qualitative = []

    print("\n[Inference] Running inference...\n")

    for idx in tqdm(range(len(test_dataset))):

        item = test_dataset[idx]

        img_path = item["image_path"]
        convo = item["conversations"]

        user_text = convo[0]["value"].replace("<image>", "").strip()
        target_json = convo[1]["value"]

        image = Image.open(img_path).convert("RGB")

        # Build conversation exactly like collator (user only)
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_text},
                ],
            }
        ]

        chat_batch = [conversation]

        # Encode
        enc = processor.apply_chat_template(
            chat_batch,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=False
        )

        # Move tensors to device
        for k in enc:
            if torch.is_tensor(enc[k]):
                enc[k] = enc[k].to(model_wrapper.model.device)

        # Generate
        out_ids = model_wrapper.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            pixel_values=enc["pixel_values"],
            image_grid_thw=enc.get("image_grid_thw", None),
            max_new_tokens=768
        )

        decoded = processor.decode(out_ids[0], skip_special_tokens=True)

        record = {
            "image_path": img_path,
            "question": user_text,
            "target_json": target_json,
            "predicted_json": decoded
        }

        predictions.append(record)

        if len(qualitative) < 50:
            qualitative.append(record)

    # Save qualitative samples
    qpath = os.path.join(OUTPUT_DIR, "test_samples.txt")
    with open(qpath, "w") as f:
        for i, s in enumerate(qualitative):
            f.write(f"[{i}] IMAGE: {s['image_path']}\n")
            f.write(f"QUESTION: {s['question']}\n\n")
            f.write(f"TARGET:\n{s['target_json']}\n\n")
            f.write(f"PREDICTED:\n{s['predicted_json']}\n\n")
            f.write("-----------------------------------------------\n\n")

    print(f"[Inference] Saved qualitative samples → {qpath}")

    # Save all predictions
    csv_path = os.path.join(OUTPUT_DIR, "test_predictions.csv")
    with open(csv_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "question", "target_json", "predicted_json"])
        for p in predictions:
            writer.writerow([p["image_path"], p["question"], p["target_json"], p["predicted_json"]])

    print(f"[Inference] Saved full predictions → {csv_path}")

    return predictions


# ============================================================
# Metric Computation
# ============================================================
def compute_metrics(predictions):

    total = len(predictions)
    correct = 0

    tp = tn = fp = fn = 0
    json_valid = 0
    complete = 0
    consistency = 0
    valid_pol = 0

    for p in predictions:

        tgt = json.loads(p["target_json"])
        try:
            pred = json.loads(p["predicted_json"])
        except:
            pred = {}

        if pred:
            json_valid += 1

        needed = ["observation", "interpretation", "diagnosis", "recommendation", "final_answer"]
        if all(k in pred and pred[k] not in ["", None] for k in needed):
            complete += 1

        gt = str(tgt["final_answer"]).strip().lower()
        pr = str(pred.get("final_answer", "")).strip().lower()

        if gt == pr:
            correct += 1

        if gt == "yes":
            if pr == "yes": tp += 1
            else: fn += 1
        else:
            if pr == "no": tn += 1
            else: fp += 1

        if pred and "interpretation" in pred:
            pol = extract_reasoning_polarity(pred["interpretation"])
            if pol != -1:
                valid_pol += 1
                if pol == 1 and pr == "yes":
                    consistency += 1
                if pol == 0 and pr == "no":
                    consistency += 1

    acc = correct / total
    sens = tp / (tp + fn + 1e-9)
    spec = tn / (tn + fp + 1e-9)
    json_rate = json_valid / total
    comp_rate = complete / total
    cons_rate = consistency / max(valid_pol, 1)

    out = os.path.join(OUTPUT_DIR, "test_metrics.csv")
    with open(out, "w") as f:
        f.write("metric,value\n")
        f.write(f"accuracy,{acc:.4f}\n")
        f.write(f"sensitivity,{sens:.4f}\n")
        f.write(f"specificity,{spec:.4f}\n")
        f.write(f"json_validity,{json_rate:.4f}\n")
        f.write(f"field_completeness,{comp_rate:.4f}\n")
        f.write(f"reasoning_answer_consistency,{cons_rate:.4f}\n")

    print(f"[Inference] Saved metrics → {out}")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    print("\n========== Qwen3-VL Inference ==========\n")

    test_dataset = QwenVLJsonlDataset(TEST_JSONL, IMAGE_ROOT)

    model_wrapper, processor = load_model()

    predictions = run_inference(model_wrapper, processor, test_dataset)

    compute_metrics(predictions)

    print("\n[Inference] DONE.\n")
