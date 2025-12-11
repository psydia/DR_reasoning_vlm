import json
from pathlib import Path
from typing import Dict


def build_user_message(question: str) -> str:
    """
    Create the standardized user message for Qwen3-VL SFT.
    Includes <image> token and the final domain-tuned reasoning instructions.
    """
    template = (
        "<image>\n"
        "You are a medical vision-language model specialized in diabetic retinopathy "
        "and diabetic macular edema analysis.\n\n"
        "Given the fundus image above, answer the question using structured clinical reasoning.\n"
        "If the question refers to a specific region (e.g., center, top-left, inferior), "
        "ensure your reasoning explicitly evaluates that region.\n\n"
        "Respond ONLY in valid JSON using the following schema:\n"
        "{\n"
        '  "observation": "",\n'
        '  "interpretation": "",\n'
        '  "diagnosis": "",\n'
        '  "recommendation": "",\n'
        '  "final_answer": ""\n'
        "}\n\n"
        f"Question: {question}"
    )
    return template


def build_assistant_message(reasoning: Dict, final_answer: str) -> str:
    """
    Construct the assistant response JSON exactly as required.
    No extra text allowed — only the JSON object.
    """
    output = {
        "observation": reasoning.get("observation", ""),
        "interpretation": reasoning.get("interpretation", ""),
        "diagnosis": reasoning.get("diagnosis", ""),
        "recommendation": reasoning.get("recommendation", ""),
        "final_answer": final_answer
    }
    return json.dumps(output, indent=2)


def convert_split(input_json_path: str, output_jsonl_path: str, image_root: str = ""):
    """
    Convert your reasoning-augmented dataset into Qwen-VL chat format.
    
    Parameters:
      - input_json_path: raw dataset path (train/val/test)
      - output_jsonl_path: location to save Qwen-compatible JSONL
      - image_root: optional prefix for image paths
    """
    input_json_path = Path(input_json_path)
    output_jsonl_path = Path(output_jsonl_path)
    image_root = Path(image_root) if image_root else None

    with open(input_json_path, "r") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} samples from {input_json_path}")

    with open(output_jsonl_path, "w") as fout:
        for sample in data:
            img_path = str(image_root / sample["image"]) if image_root else sample["image"]

            user_msg = build_user_message(sample["question"])
            assistant_msg = build_assistant_message(sample["reasoning"], sample["answer"])

            qwen_item = {
                "image": img_path,
                "conversations": [
                    {"from": "user", "value": user_msg},
                    {"from": "assistant", "value": assistant_msg}
                ]
            }

            fout.write(json.dumps(qwen_item, ensure_ascii=False) + "\n")

    print(f"Saved Qwen-format JSONL → {output_jsonl_path}")
    print("Done.")


if __name__ == "__main__":
    # Example usage — modify these paths when running
    convert_split(
        input_json_path="/final_test_reasoning_samples.json",
        output_jsonl_path="/qwen_test.jsonl",
        image_root="/visual/test"   # or "/absolute/path/to/images"
    )
