🩺 DR-Reason: Structured Clinical Reasoning for Automated Diabetic Retinopathy Assessment

This repository contains the full pipeline for DR-Reason, a system that enhances diabetic retinopathy (DR) and diabetic macular edema (DME) assessment using structured clinical reasoning on top of modern multimodal vision–language models (VLMs).
The project reformulates retinal VQA into a step-by-step reasoning task, improving interpretability, safety, and clinical grounding.

📌 Key Contributions
🔍 1. Structured Clinical Reasoning Framework

Each answer is generated using a 5-stage chain:

Observation – visible retinal findings

Interpretation – clinical meaning

Diagnosis – DR/DME decision

Recommendation – next clinical step

Final Answer – question-specific response

This ensures transparent and auditable predictions.

🗺️ 2. Region-Aware Question Enhancement

The original DME-VQA dataset contains ambiguous phrases such as “in this region”.
This project fixes that by:

Processing lesion masks

Dividing the retina into a 3×3 grid

Detecting lesion concentration (e.g., top-left, center)

Rewriting questions with explicit region references

This improves spatial grounding for both training and inference.

🧠 3. Reasoning Dataset Generation

The pipeline includes:

Automatic reasoning generation using GPT

Filtering inconsistent reasoning

Regenerating incorrect samples

Final cleaned reasoning dataset for supervised fine-tuning (SFT)

⚙️ 4. Qwen3-VL Fine-Tuning

Includes:

LoRA adapters

Frozen vision tower

Chat-template–based multimodal collator

Gradient checkpointing

Early stopping

Validation sample generation

The training scripts support both reasoning-based SFT and baseline (no-CoT) variants.

📂 Repository Structure

The repository includes:

configs/
    model.yaml
    data.yaml
    train.yaml

src/
    model_wrapper.py
    trainer.py
    collator.py
    dataset.py
    evaluator.py

data_prep_1.py
data_prep_for_qwen3vl8.py
download_qwen_model9.py
generate_reasoning_samples_gpt_cleaned3.py
inference_metrices_cot12.py
inference_metrics_without_cot12_2.py
inference_qwen_sft_CoT11.py
inference_qwen_sft_without_cot11_2.py
merge_samples7.py
parsing_bad_samples4.py
regen_reasoning_gpt6.py
region_aware_dataset_2.py
remove_reasoning_from_bad_samples5.py
train_qwen_sft_main10.py

README.md


Scripts overview:

Script	Purpose
region_aware_dataset_2.py	Region-aware question rewriting
generate_reasoning_samples_gpt_cleaned3.py	Multi-step reasoning generation
parsing_bad_samples4.py	Flags and removes contradictory reasoning
regen_reasoning_gpt6.py	Regenerates corrected reasoning
merge_samples7.py	Merges clean and regenerated samples
data_prep_for_qwen3vl8.py	Converts data into Qwen-VL JSONL format
train_qwen_sft_main10.py	Qwen3-VL supervised fine-tuning with LoRA
inference_qwen_sft_CoT11.py	Inference with reasoning
inference_metrices_cot12.py	Computes accuracy, JSON validity, consistency
📦 Dataset

We use and enhance the DME-VQA dataset
👉 https://zenodo.org/records/6784358

Enhancements include:

Region-aware question rewriting

Structured reasoning generation

Automated cleaning

JSONL formatting for Qwen3-VL training

Citation:

@dataset{dme_vqa_2022,
  title        = {DME-VQA: A Retinal Visual Question Answering Dataset},
  year         = {2022},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.6784358}
}


🚀 Training

To train the reasoning model:

python train_qwen_sft_main10.py \
  --model_cfg configs/model.yaml \
  --data_cfg configs/data.yaml \
  --train_cfg configs/train.yaml


Features:

LoRA fine-tuning

Frozen vision encoder

Structured reasoning supervision

Validation + early stopping

🔎 Inference
python inference_qwen_sft_CoT11.py


Outputs:

Predicted reasoning JSON

Side-by-side comparison with ground truth

50-sample qualitative evaluation

Metrics saved in CSV format

📊 Evaluation Metrics

The evaluation computes:

Accuracy

Yes/No accuracy

JSON validity rate

Field completeness

Reasoning–answer consistency

Direct-answer ratio

📝 Example Result Highlights
Metric	With Reasoning	Without Reasoning
Overall Accuracy	0.73	0.76
Yes Accuracy	0.87	0.91
No Accuracy	0.64	0.65
Direct Answer Ratio	0.0%	N/A

Interpretability improves dramatically due to structured reasoning.

🔮 Future Work

RLHF for stable reasoning

Multi-lesion spatial grounding

OCT dataset extension

Temporal disease progression modeling