import json
import os
from literai.util import get_output_dir
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm

def generate_image_descriptions(title: str):
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl", device_map="auto")

    base_dir = get_output_dir(title)
    parts = [os.path.join(base_dir, f) for f in os.listdir(
        base_dir) if f.startswith("part") and f.endswith(".json")]
    
    for part in tqdm(parts, desc="Part"):
        obj = json.load(open(part, "r", encoding="utf8"))

        for line in tqdm(obj['lines'], "Line"):
            summary = line.get("summary", None)
            if summary is None:
                continue

            