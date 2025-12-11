import torch
from PIL import Image
from typing import List, Dict
from transformers import AutoProcessor


class QwenVLDataCollator:
    """
    Correct collator for Qwen3-VL SFT.
    Uses apply_chat_template() to produce:
      - input_ids
      - attention_mask
      - pixel_values
      - image_grid_thw
      - labels (during SFT)
    """

    def __init__(self, model_name: str, max_length: int):
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        self.max_length = max_length

    def __call__(self, batch: List[Dict]):
        """
        Batch is list of samples:
        {
          "image_path": "...",
          "conversations": [
              {"from": "user", "value": "..."},
              {"from": "assistant", "value": "..."}
          ]
        }
        """
        all_messages = []

        for sample in batch:
            # Load image
            img_path = sample["image_path"]
            image = Image.open(img_path).convert("RGB")

            user_raw = sample["conversations"][0]["value"]
            asst_raw = sample["conversations"][1]["value"]

            # Remove <image> tag because processor handles image separately
            user_text = user_raw.replace("<image>", "").strip()

            # Build Qwen3-VL style messages with structured multimodal content
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text",  "text": user_text},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": asst_raw},
                    ],
                },
            ]

            all_messages.append(messages)

        # Apply chat template — the ONLY correct way for Qwen3-VL
        encoded = self.processor.apply_chat_template(
            all_messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        # Processor returns EVERYTHING we need
        input_ids      = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        pixel_values   = encoded["pixel_values"]
        image_grid_thw = encoded.get("image_grid_thw", None)

        # Simple SFT: predict all tokens in sequence
        # Later: can improve to mask user tokens if needed
        labels = input_ids.clone()

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "pixel_values":   pixel_values,
            "image_grid_thw": image_grid_thw,
            "labels":         labels,
        }
