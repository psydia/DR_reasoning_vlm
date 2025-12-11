import os
import json
import random
import base64
import re
import time
from tqdm import tqdm
from openai import OpenAI, APIError, RateLimitError

# --------------------------------------------------------
# 🔑 1. Setup your API key
# --------------------------------------------------------
client = OpenAI(api_key="")
client.timeout = 60  # prevent long hangs

# --------------------------------------------------------
# 📂 2. Define paths
# --------------------------------------------------------
ROOT = ""
SPLIT = "train"
DATA_PATH = os.path.join(ROOT, f"{SPLIT}_region_aware.json")
IMG_ROOT = os.path.join(ROOT, "visual", SPLIT)
OUTPUT_DIR = ""
OUT_PATH = os.path.join(OUTPUT_DIR, f"{SPLIT}_reasoning_samples_cleaned.json")

#N_SAMPLES = 10

# --------------------------------------------------------
# ⏳ Checkpointing and Error Log
# --------------------------------------------------------
CHECKPOINT_FILE = f"{SPLIT}_completed_ids.txt"
ERROR_LOG = f"{SPLIT}_errors.log"

def load_completed():
    if not os.path.exists(CHECKPOINT_FILE):
        return set()
    with open(CHECKPOINT_FILE) as f:
        return set(line.strip() for line in f)

def save_completed(sample_id):
    with open(CHECKPOINT_FILE, "a") as f:
        f.write(sample_id + "\n")

def log_error(msg):
    with open(ERROR_LOG, "a") as f:
        f.write(msg + "\n")

# --------------------------------------------------------
# 🧩 3. Utility: encode image to base64
# --------------------------------------------------------
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# --------------------------------------------------------
# 🧠 4. Prompt builder
# --------------------------------------------------------
def build_prompt(question, answer):
    return f"""
You are a medical reasoning assistant specialized in diabetic retinopathy.

Analyze the provided fundus image and the given QA pair, and produce a 4-part reasoning chain:

1. Observation — describe visible findings in the fundus image.
2. Interpretation — what these findings indicate.
3. Diagnosis (with confidence) — infer DME or DR status.
4. Recommendation — suggest the next clinical step.

If the question references a specific anatomical region (e.g., "top-left region", "center"), explicitly focus your observation and interpretation on that region while still considering the global context of the eye.


Output strictly in JSON:
{{
  "observation": "...",
  "interpretation": "...",
  "diagnosis": "...",
  "recommendation": "..."
}}

QUESTION: {question}
ANSWER: {answer}
"""

# --------------------------------------------------------
# 🧹 5. Helper: clean & parse model output
# --------------------------------------------------------
def clean_reasoning_output(raw_text: str):
    cleaned = re.sub(r"```json|```", "", raw_text, flags=re.IGNORECASE).strip()

    try:
        parsed = json.loads(cleaned)
        if all(k in parsed for k in ["observation", "interpretation", "diagnosis", "recommendation"]):
            return parsed
        else:
            return {"raw_text": raw_text.strip()}
    except Exception:
        return {"raw_text": raw_text.strip()}

# --------------------------------------------------------
# 🔄 6. Safe API call with retries
# --------------------------------------------------------
def safe_chat_completion(**kwargs):
    retries = 5
    delay = 2

    for attempt in range(retries):
        try:
            return client.chat.completions.create(**kwargs)

        except (APIError, RateLimitError) as e:
            tqdm.write(f"⚠️ API error: {e}. Retrying in {delay}s...")
            time.sleep(delay)
            delay *= 2

        except Exception as e:
            tqdm.write(f"❌ Unexpected error: {e}")
            log_error(f"Unexpected: {e}")
            return None

    tqdm.write("❌ Max retries reached, skipping sample.")
    return None

# --------------------------------------------------------
# 💬 7. Multimodal reasoning generator
# --------------------------------------------------------
def get_reasoning_with_image(image_path, question, answer):
    try:
        base64_img = encode_image_to_base64(image_path)
        prompt = build_prompt(question, answer)

        response = safe_chat_completion(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": "You are a concise, medically accurate assistant."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
                    ],
                },
            ],
            temperature=1,
        )

        if not response:
            return None

        content = response.choices[0].message.content.strip()
        return clean_reasoning_output(content)

    except Exception as e:
        msg = f"Error processing {image_path}: {e}"
        tqdm.write("⚠️ " + msg)
        log_error(msg)
        return None

# --------------------------------------------------------
# 🚀 8. Main loop with checkpointing + periodic saving
# --------------------------------------------------------
if __name__ == "__main__":
    with open(DATA_PATH, "r") as f:
        data = json.load(f)

    #subset = random.sample(data, N_SAMPLES)
    subset=data
    completed = load_completed()
    results = []

    SAVE_INTERVAL = 5
    counter = 0

    for item in tqdm(subset, desc="Generating reasoning with images"):
        img_name = item.get("image_name") or item.get("image")
        img_path = os.path.join(IMG_ROOT, img_name)

        if img_name in completed:
            continue

        if not os.path.exists(img_path):
            msg = f"Image not found: {img_path}"
            tqdm.write(f"⚠️ {msg}")
            log_error(msg)
            continue

        q, a = item["question"], item["answer"]
        reasoning = get_reasoning_with_image(img_path, q, a)

        if reasoning and isinstance(reasoning, dict):
            results.append({
                "image": img_path,
                "question": q,
                "answer": a,
                "reasoning": reasoning
            })

        save_completed(img_name)  # checkpoint
        counter += 1

        # periodic auto-save
        if counter % SAVE_INTERVAL == 0:
            with open(OUT_PATH, "w") as f:
                json.dump(results, f, indent=2)
            tqdm.write(f"💾 Auto-saved {len(results)} samples to {OUT_PATH}")

    # final save
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Saved {len(results)} cleaned multimodal reasoning samples to {OUT_PATH}")
    if results:
        print("Example output:\n", json.dumps(results[0], indent=2))
