import os
from dataclasses import dataclass
from typing import Optional, Dict

import torch
from torch import nn
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model


# ==============================================================
# Settings dataclasses
# ==============================================================

@dataclass
class LoRASettings:
    enabled: bool
    r: int
    alpha: int
    dropout: float
    target_modules: Optional[list]
    lora_bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class ModelSettings:
    model_name: str
    use_bf16: bool = True
    use_fp16: bool = False
    freeze_vision: bool = True
    gradient_checkpointing: bool = True
    lora: Optional[LoRASettings] = None


# ==============================================================
# Qwen3-VL Model Wrapper
# ==============================================================

class QwenVLModelWrapper(nn.Module):
    """
    Proper wrapper for Qwen3-VL-4B-Instruct.
    Uses:
        - Qwen3VLForConditionalGeneration
        - AutoProcessor (with trust_remote_code=True)
        - LoRA (optional)
        - BF16 support
        - image_grid_thw support for correct vision embeddings
    """

    def __init__(self, settings: ModelSettings):
        super().__init__()
        self.settings = settings

        # ----------------------------------------
        # Select dtype
        # ----------------------------------------
        if settings.use_bf16:
            torch_dtype = torch.bfloat16
        elif settings.use_fp16:
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

        # ----------------------------------------
        # Load processor
        # ----------------------------------------
        self.processor = AutoProcessor.from_pretrained(
            settings.model_name,
            trust_remote_code=True
        )

        # ----------------------------------------
        # Load model (This is the ONLY correct class)
        # ----------------------------------------
        print(f"[ModelWrapper] Loading {settings.model_name} ...")
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            settings.model_name,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )

        # ----------------------------------------
        # Enable gradient checkpointing
        # ----------------------------------------
        if settings.gradient_checkpointing:
            print("[ModelWrapper] Enabling gradient checkpointing...")
            self.model.gradient_checkpointing_enable()

        # ----------------------------------------
        # Freeze vision tower
        # ----------------------------------------
        if settings.freeze_vision:
            self._freeze_vision()

        # ----------------------------------------
        # Apply LoRA
        # ----------------------------------------
        if settings.lora is not None and settings.lora.enabled:
            self._apply_lora(settings.lora)

    # ==============================================================
    # Freeze visual backbone
    # ==============================================================

    def _freeze_vision(self):
        print("[ModelWrapper] Freezing vision tower...")
        frozen = 0
        total = 0
        for name, p in self.model.named_parameters():
            total += 1
            lname = name.lower()
            if "vision" in lname or "visual" in lname or "image" in lname:
                p.requires_grad = False
                frozen += 1
        print(f"[ModelWrapper] Frozen {frozen}/{total} parameters from vision.")

    # ==============================================================
    # Apply LoRA adapters
    # ==============================================================

    def _apply_lora(self, cfg: LoRASettings):
        print("[ModelWrapper] Applying LoRA...")
        print("  Target modules:", cfg.target_modules)

        lora_cfg = LoraConfig(
            r=cfg.r,
            lora_alpha=cfg.alpha,
            lora_dropout=cfg.dropout,
            bias=cfg.lora_bias,
            task_type=cfg.task_type,
            target_modules=cfg.target_modules,
        )

        self.model = get_peft_model(self.model, lora_cfg)
        self.model.print_trainable_parameters()

    # ==============================================================
    # Forward pass for trainer
    # ==============================================================

    def forward(self, **batch: Dict[str, torch.Tensor]):
        """
        Qwen3VL forward() requires:
            - input_ids
            - attention_mask
            - pixel_values
            - image_grid_thw  (VERY IMPORTANT)
            - labels
        """
        return self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch["pixel_values"],
            image_grid_thw=batch.get("image_grid_thw", None),
            labels=batch["labels"],
        )

    # ==============================================================
    # Generation for evaluator / inference
    # ==============================================================

    @torch.no_grad()
    def generate(self, **kwargs):
        return self.model.generate(**kwargs)

    # ==============================================================
    # Save model + processor
    # ==============================================================

    def save_pretrained(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.processor.save_pretrained(output_dir)
        print(f"[ModelWrapper] Saved model + processor to: {output_dir}")
