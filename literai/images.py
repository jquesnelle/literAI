import json
import os
import torch
from diffusers import StableDiffusionPipeline
from literai.util import get_output_dir
from transformers import AutoTokenizer, T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm
from tqdm.contrib import tenumerate

SUMMARY_MODEL_ID = "allenai/cosmo-xl"
DESCRIBE_MODEL_ID = "google/flan-t5-xl"
DRAW_MODEL_ID = "prompthero/openjourney"

DESCRIBE_PROMPT = \
    r"""passage: Even the Duchess sneezed occasionally; and as for the baby, it was sneezing and howling alternately without a moment's pause. The only things in the kitchen that did not sneeze, were the cook, and a large cat which was sitting on the hearth and grinning from ear to ear. "Please would you tell me," said Alice, a little timidly, for she was not quite sure whether it was good manners for her to speak first, "why your cat grins like that?" "It's a Cheshire cat," said the Duchess, "and that's why. Pig!"
image: girl in a kitchen looking at a large grinning cheshire cat sitting on a hearth

passage: The door of the Doctor's room opened, and he came out with Charles Darnay. He was so deadly pale--which had not been the case when they went in together--that no vestige of colour was to be seen in his face. But, in the composure of his manner he was unaltered, except that to the shrewd glance of Mr. Lorry it disclosed some shadowy indication that the old air of avoidance and dread had lately passed over him, like a cold wind. 
image: man with a very pale face standing in a doorway, second man looking at him with a shrewed glance

passage: {passage}
image: """


def generate_image_descriptions(title: str, txt: str, summarize_batch_length=2048, summary_batch_stride=16, describe_batch_length=164, describe_batch_stride=16):
    summarize_tokenizer = AutoTokenizer.from_pretrained(SUMMARY_MODEL_ID)

    # re-create the batches used for summarization
    input_text = open(txt, "r", encoding="utf-8").read()
    summary_encodings = summarize_tokenizer.encode_plus(
        input_text,
        padding="max_length",
        truncation=True,
        max_length=summarize_batch_length,
        stride=summary_batch_stride,
        return_overflowing_tokens=True,
        add_special_tokens=False,
    )

    batch_tokenizer = AutoTokenizer.from_pretrained(DESCRIBE_MODEL_ID)
    tokenizer = T5Tokenizer.from_pretrained(DESCRIBE_MODEL_ID)
    model = T5ForConditionalGeneration.from_pretrained(
        DESCRIBE_MODEL_ID, device_map="auto")

    base_dir = get_output_dir(title)
    parts = [os.path.join(base_dir, f) for f in os.listdir(
        base_dir) if f.startswith("part") and f.endswith(".json")]

    for part in tqdm(parts, desc="Part", leave=False):
        obj = json.load(open(part, "r", encoding="utf8"))

        last_batch = None
        for summary in tqdm(obj['summaries'], desc="Summary", leave=False):
            batch = summary['batch']
            if batch == last_batch:
                continue
            last_batch = batch

            summary_offsets = summary_encodings[batch].offsets
            if summary_offsets[len(summary_offsets) - 1] == (0, 0):
                summary_offsets.pop()
            summary_text_start = summary_offsets[0][0]
            summary_text_end = summary_offsets[len(summary_offsets) - 1][1]

            text = input_text[summary_text_start:summary_text_end+1]

            describe_encodings = batch_tokenizer.encode_plus(
                text,
                padding="max_length",
                truncation=True,
                max_length=describe_batch_length,
                stride=describe_batch_stride,
                return_overflowing_tokens=True,
                add_special_tokens=False,
            )

            for batch in tqdm(describe_encodings.encodings, desc="Batch", leave=False):
                if batch.offsets[len(batch.offsets) - 1] == (0, 0):
                    batch.offsets.pop()
                batch_text_start = batch.offsets[0][0]
                batch_text_end = batch.offsets[len(batch.offsets) - 1][1]

                batch_text = input_text[batch_text_start:
                                        batch_text_end+1].replace('\n', '')
                prompt = DESCRIBE_PROMPT.format(passage=batch_text)

                input_ids = tokenizer(
                    prompt, return_tensors="pt").input_ids.to("cuda")
                outputs = model.generate(
                    input_ids, max_new_tokens=64)
                result = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

                if len(result) > 10:
                    if 'descriptions' not in summary:
                        summary['descriptions'] = []
                    summary['descriptions'].append(result)

        json.dump(obj, open(part, "w", encoding="utf8"), indent=2)


def generate_images(title: str): 
    pipe = StableDiffusionPipeline.from_pretrained(DRAW_MODEL_ID, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    base_dir = get_output_dir(title)
    parts = [os.path.join(base_dir, f) for f in os.listdir(
        base_dir) if f.startswith("part") and f.endswith(".json")]

    images = get_output_dir(title, "images")

    for part in tqdm(parts, desc="Part", leave=False):
        obj = json.load(open(part, "r", encoding="utf8"))

        part_base = os.path.basename(part)
        part_base = part_base[0:part_base.rfind('.')]

        for summary, summary_index in tenumerate(obj['summaries'], desc="Summary", leave=False):
            for description in tqdm(summary['descriptions'], desc="Description", leave=False):
                prompt = f"{description}, mdjrny-v4 style"
                image = pipe(prompt).images[0]

                image_filename = f"{part_base}-{summary_index}.png"
                image.save(os.path.join(images, image_filename))

                if "images" not in summary:
                    summary["images"] = []
                summary["images"].append(f"images/{image_filename}")

        json.dump(obj, open(part, "w", encoding="utf8"), indent=2)