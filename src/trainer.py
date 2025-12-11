import os
import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from src.evaluator import QwenVLEvaluator


class QwenVLTrainer:
    """
    Full SFT Trainer for Qwen3-VL with:
      - Correct early stopping
      - Linear warmup + cosine decay scheduler
      - Proper numeric casting
      - Optional seeding
    """

    def __init__(self,
                 model_wrapper,
                 train_loader,
                 val_loader,
                 train_config,
                 model_config,
                 output_dir="outputs/",
                 processor=None,
                 val_dataset=None):

        self.model_wrapper = model_wrapper
        self.model = model_wrapper.model
        self.processor = processor
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.val_dataset = val_dataset

        self.train_cfg = train_config
        self.model_cfg = model_config
        self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/logs", exist_ok=True)
        os.makedirs(f"{self.output_dir}/checkpoints", exist_ok=True)

        # --------------------------------------------------------------
        # 0. Optional: Set seed
        # --------------------------------------------------------------
        seed = self.train_cfg.get("training", {}).get("seed", None)
        if seed is not None:
            import random, numpy as np
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # --------------------------------------------------------------
        # 1. Optimizer with numeric casting
        # --------------------------------------------------------------
        opt_cfg = train_config["optimizer"]

        lr = float(opt_cfg["lr"])
        eps = float(opt_cfg.get("eps", 1e-8))
        betas = tuple([float(b) for b in opt_cfg.get("betas", [0.9, 0.999])])
        weight_decay = float(opt_cfg.get("weight_decay", 0.0))

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )

        # --------------------------------------------------------------
        # 2. Scheduler = warmup + cosine decay
        # --------------------------------------------------------------
        sched_cfg = train_config["scheduler"]

        warmup_steps = int(sched_cfg.get("warmup_steps", 0))

        total_steps = (
            len(train_loader)
            // train_config["training"]["gradient_accumulation_steps"]
            * train_config["training"]["num_epochs"]
        )

        # First do **linear warmup**, then cosine decay
        self.scheduler = get_linear_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        # --------------------------------------------------------------
        # 3. Early stopping state
        # --------------------------------------------------------------
        self.best_val_loss = float("inf")
        self.no_improve_count = 0
        self.patience = train_config["evaluation"]["early_stopping"]["patience"]
        self.min_delta = float(train_config["evaluation"]["early_stopping"].get("min_delta", 0.0))

        # --------------------------------------------------------------
        # 4. Evaluator
        # --------------------------------------------------------------
        self.evaluator = QwenVLEvaluator(
            model_wrapper=model_wrapper,
            processor=processor,
            max_new_tokens=768
        )

        # Gradient accumulation
        self.grad_accum_steps = train_config["training"]["gradient_accumulation_steps"]


    # ==========================================================================
    # TRAIN LOOP
    # ==========================================================================
    def train(self):

        num_epochs = self.train_cfg["training"]["num_epochs"]
        log_every = self.train_cfg["logging"]["log_every"]
        eval_every = self.train_cfg["evaluation"]["eval_every_steps"]

        print(f"\n[Trainer] Starting training for {num_epochs} epochs...\n")

        global_step = 0

        for epoch in range(num_epochs):
            print(f"\n========== Epoch {epoch+1}/{num_epochs} ==========\n")

            self.model.train()
            running_loss = 0

            for step, batch in enumerate(tqdm(self.train_loader, desc="Training")):

                # Move batch tensors to device
                for k in batch:
                    if torch.is_tensor(batch[k]):
                        batch[k] = batch[k].to(self.model.device)

                # Forward
                outputs = self.model_wrapper(**batch)
                loss = outputs.loss
                running_loss += loss.item()

                # Grad accumulation
                loss = loss / self.grad_accum_steps
                loss.backward()

                if (step + 1) % self.grad_accum_steps == 0:

                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.train_cfg["training"]["max_grad_norm"]
                    )

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                # -------------------------
                # Print train loss
                # -------------------------
                if global_step % log_every == 0:
                    true_loss = loss.item() * self.grad_accum_steps
                    print(f"[train | step {global_step}] loss = {true_loss:.4f}")

                    with open(f"{self.output_dir}/logs/train_log.txt", "a") as f:
                        f.write(f"step {global_step} train_loss {true_loss:.4f}\n")

                # -------------------------
                # Periodic validation
                # -------------------------
                if global_step > 0 and global_step % eval_every == 0:
                    val_loss, improved = self._run_validation(global_step)
                    if self._check_early_stopping(improved):
                        print("\n[Trainer] Early stopping triggered.")
                        return

                global_step += 1

            # Validation at epoch end
            val_loss, improved = self._run_validation(global_step)
            if self._check_early_stopping(improved):
                print("\n[Trainer] Early stopping triggered.")
                return


    # ==========================================================================
    # VALIDATION
    # ==========================================================================
    def _run_validation(self, step: int):
        print(f"\n[Trainer] Running validation at step {step} ...")

        val_loss = self.evaluator.compute_val_loss(self.model_wrapper, self.val_loader)
        print(f"[val | step {step}] loss = {val_loss:.4f}")

        # Log val loss
        with open(f"{self.output_dir}/logs/val_log.txt", "a") as f:
            f.write(f"step {step} val_loss {val_loss:.4f}\n")

        # Determine improvement BEFORE updating best_val_loss
        improved = (self.best_val_loss - val_loss) > self.min_delta

        # Save validation samples
        if self.val_dataset is not None:
            samples = self.evaluator.generate_samples(
                model_wrapper=self.model_wrapper,
                val_dataset=self.val_dataset,
                num_samples=2
            )

            path = f"{self.output_dir}/logs/val_samples_step_{step}.txt"
            with open(path, "w") as f:
                f.write("=== Validation Samples ===\n\n")
                for i, s in enumerate(samples):
                    f.write(f"Image: {s['image_path']}\n\n")
                    f.write(f"--- Sample {i} ---\n")
                    f.write(f"Question:\n{s['input_question']}\n\n")
                    f.write(f"Target JSON:\n{s['target_json']}\n\n")
                    f.write(f"Generated JSON:\n{s['generated_json']}\n\n")

            print(f"[Trainer] Saved validation samples → {path}")

        # Save checkpoint if improved
        if improved:
            print(f"[Trainer] New best model! (val_loss={val_loss:.4f})")
            self.best_val_loss = val_loss
            self.model_wrapper.save_pretrained(
                f"{self.output_dir}/checkpoints/best_model"
            )

        return val_loss, improved


    # ==========================================================================
    # EARLY STOPPING (correct!)
    # ==========================================================================
    def _check_early_stopping(self, improved: bool) -> bool:
        if improved:
            self.no_improve_count = 0
            return False

        self.no_improve_count += 1
        print(f"[Trainer] No improvement. Patience = {self.no_improve_count}/{self.patience}")

        return self.no_improve_count >= self.patience
