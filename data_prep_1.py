# -----------------------------------------------
# [01] CONFIGURATION & IMPORTS
# -----------------------------------------------
import os, json, random
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Path setup (update ROOT to your actual location)
ROOT = "/dme_vqa"
QA_PATH = os.path.join(ROOT, "qa")

# Choose dataset split
SPLIT = "train"  # or "val", "test"
IMG_ROOT = os.path.join(ROOT, "visual", "train")
JSON_PATH = os.path.join(QA_PATH, f"{SPLIT}qa.json")
# -------------------------------
# GLOBAL FONT SETTINGS FOR MATPLOTLIB
# -------------------------------
plt.rcParams.update({
    "font.size": 20,          # Increase overall font size
    "axes.titlesize": 20,     # Title font size
    "axes.labelsize": 20,     # Axis label size
    "xtick.labelsize": 16,    # X-axis tick size
    "ytick.labelsize": 16,    # Y-axis tick size
    "font.weight": "bold",    # Make all fonts bold
    "axes.titleweight": "bold",
    "axes.labelweight": "bold"
})
# -----------------------------------------------
# [02] LOAD JSON METADATA
# -----------------------------------------------
def load_json_metadata(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    print(f"✅ Loaded {len(data)} samples from {os.path.basename(json_path)}")
    print("Example entry:")
    print(json.dumps(data[0], indent=2))
    return data

# ✅ this will now work
data = load_json_metadata(JSON_PATH)

# -----------------------------------------------
# [03] DEFINE PYTORCH DATASET CLASS
# -----------------------------------------------
class DMEVQADataset(Dataset):
    """Custom Dataset for DME_VQA project"""
    def __init__(self, data, img_root, transform=None):
        self.data = data
        self.img_root = img_root
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = os.path.join(self.img_root, item["image_name"])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return {
            "image": image,
            "question": item["question"],
            "answer": item["answer"],
            "img_path": img_path
        }
# -----------------------------------------------
# [04] INITIALIZE DATASET AND INSPECT SAMPLE
# -----------------------------------------------
dataset = DMEVQADataset(data, IMG_ROOT)
print(f"Dataset size: {len(dataset)} samples")

sample = dataset[0]
print(f"Q: {sample['question']}")
print(f"A: {sample['answer']}")
print(f"Image path: {sample['img_path']}")
# -----------------------------------------------
# [05] VISUALIZE RANDOM SAMPLES
# -----------------------------------------------
def visualize_samples(dataset, n=3):
    fig, axes = plt.subplots(1, n, figsize=(15, 5))
    for i in range(n):
        idx = random.randint(0, len(dataset) - 1)
        s = dataset[idx]
        img = s["image"].permute(1, 2, 0)
        axes[i].imshow(img)
        axes[i].set_title(f"Q: {s['question']}\nA: {s['answer']}", fontsize=8)
        axes[i].axis("off")
    plt.tight_layout()
    plt.show()

# Preview sample images and questions
visualize_samples(dataset, n=5)
# -----------------------------------------------
# [06] DATA ANALYSIS & DISTRIBUTION STATS
# -----------------------------------------------
df = pd.DataFrame(data)
print("\nDataframe preview:")
print(df.head())

print("\nAnswer distribution:")
print(df['answer'].value_counts())

# Plot distribution
df['answer'].value_counts().plot(kind='bar', title='Answer Distribution', figsize=(5,3))
plt.show()

# Optional: classify question type
def classify_question(q):
    ql = q.lower()
    if "lesion" in ql:
        return "lesion"
    elif "region" in ql:
        return "region"
    else:
        return "global"

df['type'] = df['question'].apply(classify_question)
df['type'].value_counts().plot(kind='bar', title='Question Type Distribution', figsize=(5,3))
plt.show()

