import os
import sys
import argparse
import yaml
import torch
from torch.utils.data import DataLoader

# Make project root importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset import QwenVLJsonlDataset
from src.collator import QwenVLDataCollator
from src.model_wrapper import QwenVLModelWrapper, ModelSettings, LoRASettings
from src.trainer import QwenVLTrainer


# -------------------------------------------------------------
# Helpers
# -------------------------------------------------------------
def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_model_settings(model_cfg: dict) -> ModelSettings:
    """
    Convert your ORIGINAL nested model.yaml → ModelSettings.
    
    Expected structure:
      model:
        name: ...
      dtype:
        use_bf16: true
        use_fp16: false
      vision:
        freeze_vision: true
      gradient_checkpointing:
        enabled: true
      lora:
        enabled: true
        lora_bias: "none"
        ...
    """

    # ------------------------
    # LoRA Settings
    # ------------------------
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

    # ------------------------
    # Core Model Settings
    # ------------------------
    model_name = model_cfg["model"]["name"]

    dtype_cfg = model_cfg.get("dtype", {})
    use_bf16 = dtype_cfg.get("use_bf16", True)
    use_fp16 = dtype_cfg.get("use_fp16", False)

    vision_cfg = model_cfg.get("vision", {})
    freeze_vision = vision_cfg.get("freeze_vision", True)

    gc_cfg = model_cfg.get("gradient_checkpointing", {})
    gradient_checkpointing = gc_cfg.get("enabled", True)

    return ModelSettings(
        model_name=model_name,
        use_bf16=use_bf16,
        use_fp16=use_fp16,
        freeze_vision=freeze_vision,
        gradient_checkpointing=gradient_checkpointing,
        lora=lora_cfg,
    )


# -------------------------------------------------------------
# Main Training Script
# -------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train Qwen3-VL SFT on DR/DME dataset.")
    parser.add_argument("--model_cfg", type=str, default="configs/model.yaml")
    parser.add_argument("--data_cfg", type=str, default="configs/data.yaml")
    parser.add_argument("--train_cfg", type=str, default="configs/train.yaml")
    parser.add_argument("--output_dir", type=str, default="/outputs_qwen_sft")
    args = parser.parse_args()

    # ---------------------------------------------------------
    # 1. Load configs
    # ---------------------------------------------------------
    model_cfg_dict = load_yaml(args.model_cfg)
    data_cfg = load_yaml(args.data_cfg)
    train_cfg = load_yaml(args.train_cfg)

    os.makedirs(args.output_dir, exist_ok=True)

    # ---------------------------------------------------------
    # 2. Resolve dataset + loader config (NESTED)
    # ---------------------------------------------------------
    paths_cfg = data_cfg["paths"]
    train_jsonl = paths_cfg["train_jsonl"]
    val_jsonl = paths_cfg["val_jsonl"]
    image_root = paths_cfg["image_root"]

    batching_cfg = data_cfg.get("batching", {})
    train_batch_size = batching_cfg.get("train_batch_size", 1)
    val_batch_size = batching_cfg.get("val_batch_size", 1)

    dataloader_cfg = data_cfg.get("dataloader", {})
    num_workers = dataloader_cfg.get("num_workers", 4)

    token_cfg = data_cfg.get("tokenization", {})
    max_length = token_cfg.get("max_length", 512)

    print("\n[Script] Loading datasets...")
    train_dataset = QwenVLJsonlDataset(train_jsonl, image_root)
    val_dataset   = QwenVLJsonlDataset(val_jsonl, image_root)

    # ---------------------------------------------------------
    # 3. Collator (uses model.model.name)
    # ---------------------------------------------------------
    collator = QwenVLDataCollator(
        model_name=model_cfg_dict["model"]["name"],
        max_length=max_length,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
    )

    # ---------------------------------------------------------
    # 4. Build model wrapper
    # ---------------------------------------------------------
    print("\n[Script] Building model settings...")
    model_settings = build_model_settings(model_cfg_dict)

    print("\n[Script] Initializing Qwen3-VL model wrapper...")
    model_wrapper = QwenVLModelWrapper(model_settings)
    processor = model_wrapper.processor

    # ---------------------------------------------------------
    # 5. Trainer
    # ---------------------------------------------------------
    trainer = QwenVLTrainer(
        model_wrapper=model_wrapper,
        train_loader=train_loader,
        val_loader=val_loader,
        train_config=train_cfg,
        model_config=model_settings,
        output_dir=args.output_dir,
        processor=processor,
        val_dataset=val_dataset,
    )

    # ---------------------------------------------------------
    # 6. Train
    # ---------------------------------------------------------
    trainer.train()
    print("\n[Script] Training finished.\n")


if __name__ == "__main__":
    main()
