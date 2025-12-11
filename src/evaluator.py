import json
import torch
from typing import Dict, Any, List
from tqdm import tqdm
from PIL import Image


def safe_parse_json(text: str) -> dict:
    try:
        return json.loads(text)
    except:
        return {}


class QwenVLEvaluator:

    def __init__(self, model_wrapper, processor, max_new_tokens=256):
        self.model = model_wrapper.model
        self.processor = processor
        self.max_new_tokens = max_new_tokens

    # ------------------------------------------------------------
    # 1. Validation loss
    # ------------------------------------------------------------
    @torch.no_grad()
    def compute_val_loss(self, model_wrapper, val_loader) -> float:
        model_wrapper.eval()

        total_loss = 0
        total_count = 0

        for batch in tqdm(val_loader, desc="Validating", leave=False):
            for k in batch:
                if torch.is_tensor(batch[k]):
                    batch[k] = batch[k].to(self.model.device)

            outputs = model_wrapper(**batch)
            total_loss += outputs.loss.item()
            total_count += 1

        return total_loss / max(total_count, 1)

    # ------------------------------------------------------------
    # 2. Generate validation examples (NEW)
    # ------------------------------------------------------------
    @torch.no_grad()
    def generate_samples(self, model_wrapper, val_dataset, num_samples=3):
        """
        Uses correct Qwen3-VL structured multimodal messages and
        processor.apply_chat_template().
        """

        samples = []

        for i in range(num_samples):
            item = val_dataset[i]
            image_path = item["image_path"]

            # Load image
            image = Image.open(item["image_path"]).convert("RGB")

            # Build structured messages (SAME as collator)
            user_text = item["conversations"][0]["value"].replace("<image>", "").strip()
            target_json = item["conversations"][1]["value"]

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text",  "text": user_text},
                    ],
                }
            ]

            # Encode prompt
            enc = self.processor.apply_chat_template(
                [messages],                     # batch of 1
                tokenize=True,
                add_generation_prompt=True,     # enable generation prompt
                return_dict=True,
                return_tensors="pt"
            )

            # Prepare model inputs
            enc = {k: (v.to(self.model.device) if torch.is_tensor(v) else v)
                   for k, v in enc.items()}

            # Generate output
            out_ids = model_wrapper.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                pixel_values=enc["pixel_values"],
                image_grid_thw=enc.get("image_grid_thw", None),
                max_new_tokens=self.max_new_tokens
            )

            decoded = self.processor.decode(out_ids[0], skip_special_tokens=True)

            samples.append({
                "image_path": image_path,
                "input_question": user_text,
                "target_json": target_json,
                "generated_json": decoded
            })

        return samples

    # ------------------------------------------------------------
    # 3. JSON validity + answer accuracy
    # ------------------------------------------------------------
    def compute_metrics(self, outputs: List[str], targets: List[str]) -> Dict[str, Any]:

        valid = 0
        correct = 0
        total = len(outputs)

        for pred, gt in zip(outputs, targets):
            jp = safe_parse_json(pred)
            jt = safe_parse_json(gt)

            if jp:
                valid += 1

            if "final_answer" in jp and "final_answer" in jt:
                if str(jp["final_answer"]).lower().strip() == str(jt["final_answer"]).lower().strip():
                    correct += 1

        return {
            "json_validity": valid / max(total, 1),
            "answer_accuracy": correct / max(total, 1)
        }
