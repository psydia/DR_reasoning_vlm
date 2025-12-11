from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import torch

model_id = "Qwen/Qwen3-VL-4B-Instruct"

print("Loading model...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

print("Loading processor...")
processor = AutoProcessor.from_pretrained(model_id)

print("Model + Processor loaded successfully!")
print("Model dtype:", next(model.parameters()).dtype)
