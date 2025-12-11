import json
from typing import List, Dict
from torch.utils.data import Dataset

class QwenVLJsonlDataset(Dataset):
    """
    A simple dataset that reads Qwen-style JSONL SFT data.
    Each line in the JSONL file should contain:
    
    {
        "image": "/path/to/image.jpg",
        "conversations": [
            {"from": "user", "value": "<image>\n..."},
            {"from": "assistant", "value": "{...JSON...}"}
        ]
    }

    This dataset DOES NOT:
        - tokenize text
        - load images
        - pad sequences
    
    These are handled in collator.py.
    """

    def __init__(self, jsonl_path: str, image_root: str):
        super().__init__()
        self.samples = []
        self.image_root = image_root

        # Load JSONL into memory
        with open(jsonl_path, "r") as f:
            for line in f:
                if line.strip():
                    sample = json.loads(line)
                    self.samples.append(sample)

        print(f"[QwenVLJsonlDataset] Loaded {len(self.samples)} samples from {jsonl_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        """
        Returns:
            {
              "image_path": "/abs/path/to/img.jpg",
              "conversations": [
                   {"from": "user", "value": "..."},
                   {"from": "assistant", "value": "..."}
              ]
            }
        """
        item = self.samples[idx]

        image_path = item["image"]
        # Ensure full absolute path if dataset uses relative paths
        if not image_path.startswith("/"):
            image_path = f"{self.image_root}/{image_path}"

        return {
            "image_path": image_path,
            "conversations": item["conversations"],
        }
