import os
import json
import csv
from tqdm import tqdm
from PIL import Image
import torch

# project imports
from src.dataset import QwenVLJsonlDataset
from src.model_wrapper import QwenVLModelWrapper, ModelSettings, LoRASettings
from peft import PeftModel


# ============================================================
# HARD-CODE PATHS
# ============================================================
MODEL_CFG = "/configs/model.yaml"
CHECKPOINT_DIR = "/outputs_qwen_sft/checkpoints/best_model"
TEST_JSONL = "/Qwen_reasoning_samples_dme/qwen_test.jsonl"
IMAGE_ROOT = "/DME_VQA_Dataset/dme_vqa/visual/test"
OUTPUT_DIR = "/results_qwen_sft_test_without_cot"


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
                ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"],
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
# Load model (base + LoRA)
# ============================================================
def load_model():

    model_cfg_dict = load_yaml(MODEL_CFG)

    # Disable LoRA inside config since we load real adapters
    if "lora" in model_cfg_dict:
        model_cfg_dict["lora"]["enabled"] = False

    settings = build_model_settings(model_cfg_dict)

    print("\n[Inference] Loading base Qwen3-VL model ...")
    model_wrapper = QwenVLModelWrapper(settings)
    processor = model_wrapper.processor

    print(f"[Inference] Loading LoRA checkpoint: {CHECKPOINT_DIR}")
    model_wrapper.model = PeftModel.from_pretrained(
        model_wrapper.model,
        CHECKPOINT_DIR
    )

    model_wrapper.model.eval()
    return model_wrapper, processor


# ============================================================
# Extract true question text (ignore JSON schema in test prompt)
# ============================================================
def extract_question_text(raw_user: str) -> str:
    raw = raw_user.replace("<image>", "").strip()
    idx = raw.rfind("Question:")
    if idx != -1:
        return raw[idx:].strip()
    return raw  # fallback if no "Question:" found


# ============================================================
# Extract final label (supports yes/no + numeric grades)
# ============================================================
def extract_final_label(decoded: str) -> str:
    text = decoded.lower().strip()

    # 1. Yes/no
    if text.startswith("yes"):
        return "yes"
    if text.startswith("no"):
        return "no"

    # 2. Numeric grades
    import re
    m = re.search(r"\b([0-9])\b", text)
    if m:
        return m.group(1)

    # 3. Check space-separated tokens
    for tok in ["0", "1", "2", "3", "4", "5"]:
        if tok in text.split():
            return tok

    return ""


# ============================================================
# Direct classification inference
# ============================================================
def run_inference(model_wrapper, processor, test_dataset):

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    csv_path = os.path.join(OUTPUT_DIR, "direct_predictions2.csv")
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["image_path", "question", "gt_final_answer", "predicted_final_answer"])

    predictions = []

    print("\n[Inference] Running DIRECT classification...\n")

    for i in tqdm(range(len(test_dataset))):

        item = test_dataset[i]
        img_path = item["image_path"]
        convo = item["conversations"]

        # Ground truth final_answer (inside JSON)
        gt_json = json.loads(convo[1]["value"])
        gt_ans = str(gt_json["final_answer"]).strip().lower()

        # Extract real question
        raw_user = convo[0]["value"]
        question_part = extract_question_text(raw_user)

        # New user prompt (no JSON reasoning!)
        user_text = (
            "You are a medical vision-language model specialized in diabetic "
            "retinopathy and diabetic macular edema.\n\n"
            "Provide ONLY the final answer label for the question below. "
            "Do NOT provide reasoning, explanation, or JSON.\n"
            "The answer may be yes/no OR a numeric disease grade (0,1,2,...).\n\n"
            f"{question_part}"
        )

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Qwen-VL structured user message
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_text},
                ],
            }
        ]

        # Encode
        enc = processor.apply_chat_template(
            [conversation],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=False,
        )

        # Move to GPU
        for k in enc:
            if torch.is_tensor(enc[k]):
                enc[k] = enc[k].to(model_wrapper.model.device)

        # Generate direct label
        out_ids = model_wrapper.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            pixel_values=enc["pixel_values"],
            image_grid_thw=enc.get("image_grid_thw"),
            max_new_tokens=8,
            do_sample=False,
        )

        # Keep only generated tokens (remove prompt)
        gen_tokens = out_ids[0][enc["input_ids"].shape[1]:]
        decoded = processor.decode(gen_tokens, skip_special_tokens=True)

        pred_ans = extract_final_label(decoded)

        # Print for human debugging
        print(f"\n[{i}]")
        print("Image:", img_path)
        print("Question:", question_part)
        print("GT:", gt_ans)
        print("Raw model output:", repr(decoded))
        print("Pred:", pred_ans)

        # Save row
        writer.writerow([img_path, question_part, gt_ans, pred_ans])

        predictions.append({
            "image_path": img_path,
            "question": question_part,
            "gt_final_answer": gt_ans,
            "predicted_final_answer": pred_ans,
        })

    csv_file.close()
    print(f"\n[Inference] Saved CSV → {csv_path}")

    return predictions


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    print("\n========== DIRECT ANSWER INFERENCE ==========\n")

    test_dataset = QwenVLJsonlDataset(TEST_JSONL, IMAGE_ROOT)
    model_wrapper, processor = load_model()

    run_inference(model_wrapper, processor, test_dataset)

    print("\n[Inference] DONE.\n")
