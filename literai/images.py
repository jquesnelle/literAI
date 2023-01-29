import json
import os
from typing import Optional
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
from literai.util import get_output_dir, logger_error
from transformers import AutoTokenizer, T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm, trange
from tqdm.contrib import tenumerate

SUMMARY_MODEL_ID = "pszemraj/long-t5-tglobal-xl-16384-book-summary"
DESCRIBE_MODEL_ID = "google/flan-t5-xl"
DEFAULT_DRAW_MODEL_ID = "dreamlike-art/dreamlike-diffusion-1.0"

DESCRIBE_PROMPT = \
    r"""passage: There was certainly too much of it in the air. Even the Duchess sneezed occasionally; and as for the baby, it was sneezing and howling alternately without a moment's pause. The only things in the kitchen that did not sneeze, were the cook, and a large cat which was sitting on the hearth and grinning from ear to ear. "Please would you tell me," said Alice, a little timidly, for she was not quite sure whether it was good manners for her to speak first, "why your cat grins like that?" "It's a Cheshire cat," said the Duchess, "and that's why. Pig!"
scene: a kitchen filled with sneezing people, a large cat sitting on the hearth and grinning from ear to ear

passage: The door of the Doctor's room opened, and he came out with Charles Darnay. He was so deadly pale--which had not been the case when they went in together--that no vestige of colour was to be seen in his face. But, in the composure of his manner he was unaltered, except that to the shrewd glance of Mr. Lorry it disclosed some shadowy indication that the old air of avoidance and dread had lately passed over him, like a cold wind. 
scene: doctor and a man coming out of a room, the doctor pale and with an indication of avoidance and dread in his manner

passage: Then today, at sunrise, we saw a white flame among the trees, high on a sheer peak before us. We thought that it was a fire andstopped. But the flame was unmoving, yet blinding as liquid metal. So we climbed toward it through the rocks. And there, before us, on a broad summit, with the mountains rising behind it, stood a house such as we had never seen, and the white fire came from the sun on the glass of its windows. The house had two stories and a strange roof flat as a floor. There was more window than wall upon its walls, and the windows went on straight around the corners, though how this kept the house standing we could not guess. The walls were hard and smooth, of that stone unlike stone which we had seen in our tunnel.
scene: a white flame high on a sheer peak, a house with two stories and a strange flat roof, walls made of a strange stone, many windows going straight around the corners

passage: Close to our bows, strange forms in the water darted hither and thither before us; while thick in our rear flew the inscrutable sea-ravens. And every morning, perched on our stays, rows of these birds were seen; and spite of our hootings, for a long time obstinately clung to the hemp, as though they deemed our ship some drifting, uninhabited craft; a thing appointed to desolation, and therefore fit roosting-place for their homeless selves. And heaved and heaved, still unrestingly heaved the black sea, as if its vast tides were a conscience; and the great mundane soul were in anguish and remorse for the long sin and suffering it had bred.
scene: strange forms in the water, sea-ravens perched on the ship's stays, the black sea heaving as if in anguish and remorse

passage: For the main road was a boiling stream of people, a torrent of human beings rushing northward, one pressing on another. A great bank of dust, white and luminous in the blaze of the sun, made everything within twenty feet of the ground grey and indistinct and was perpetually renewed by the hurrying feet of a dense crowd of horses and of men and women on foot, and by the wheels of vehicles of every description.
scene: a boiling stream of people rushing northward, a great bank of dust, horses, people, and vehicles of all kinds

passage: A piercing, bloodcurdling shriek split the silence — the book was screaming! Harry snapped it shut, but the shriek went on and on, one high, unbroken, earsplitting note. He stumbled backward and knocked over his lamp, which went out at once. Panicking, he heard footsteps coming down the corridor outside — stuffing the shrieking book back on the shelf, he ran for it. He passed Filch in the doorway; Filch's pale, wild eyes looked straight through him, and Harry slipped under Filch's outstretched arm and streaked off up the corridor, the book's shrieks still ringing in his ears.
scene: boy in a dark corridor, book screaming, running away from a pale-eyed man in the doorway

passage: little shriek, and went on: "—that begins with an M, such as mouse-traps, and the moon, and memory, and muchness—you know you say things are "much of a muchness"—did you ever see such a thing as a drawing of a muchness?" "Really, now you ask me," said Alice, very much confused, "I don't think—" "Then you shouldn't talk," said the Hatter. This piece of rudeness was more than Alice could bear: she got up in great disgust, and walked off; the Dormouse fell asleep instantly, and neither of the others took the least notice of her going, though she looked back once or twice, half hoping that they would call after her: the last time she saw them, they were trying to put the Dormouse into the teapot.
scene: girl walking away from two people trying to put a dormouse into a teapot, after being insulted by one of them

passage: He was a man of about sixty, handsomely dressed, haughty in manner, and with a face like a fine mask. A face of a transparent paleness; every feature in it clearly defined; one set expression on it. The nose, beautifully formed otherwise, was very slightly pinched at the top of each nostril. In those two compressions, or dints, the only little change that the face ever showed, resided. They persisted in changing colour sometimes, and they would be occasionally dilated and contracted by something like a faint pulsation; then, they gave a look of treachery, and cruelty, to the whole countenance. Examined with attention, its capacity of helping such a look was to be found in the line of the mouth, and the lines of the orbits of the eyes, being much too horizontal and thin; still, in the effect of the face made, it was a handsome face, and a remarkable one.
scene: a man with a handsome face, pale complexion, and thin horizontal lines around his mouth and eyes, giving a look of treachery and cruelty

passage: {passage}
scene: """

DEFAULT_DRAW_PROMPT = \
    "dreamlikeart, {description}, in the style of artgerm and charlie bowater and atey ghailan and mike mignola, vibrant colors and hard shadows and strong rim light, comic cover art, epic scene, plain background, trending on artstation"


@logger_error
def generate_image_descriptions(
        title: str,
        txt: str,
        summarize_batch_length=2048,
        summary_batch_stride=16,
        describe_batch_length=256,
        describe_batch_stride=16,
        print_descriptions=False):
    summarize_tokenizer = AutoTokenizer.from_pretrained(SUMMARY_MODEL_ID)

    # re-create the batches used for summarization
    input_text = open(txt, "r", encoding="utf-8").read()
    summary_encodings = summarize_tokenizer.encode_plus(
        input_text,
        truncation=True,
        max_length=summarize_batch_length,
        stride=summary_batch_stride,
        return_overflowing_tokens=True,
        add_special_tokens=False,
    )

    batch_tokenizer = AutoTokenizer.from_pretrained(DESCRIBE_MODEL_ID)
    tokenizer = T5Tokenizer.from_pretrained(
        DESCRIBE_MODEL_ID, model_max_length=2048)
    model = T5ForConditionalGeneration.from_pretrained(
        DESCRIBE_MODEL_ID, device_map="auto")

    base_dir = get_output_dir(title)
    parts = [os.path.join(base_dir, f) for f in os.listdir(
        base_dir) if f.startswith("part") and f.endswith(".json")]

    for part in tqdm(parts, desc="Part"):
        obj = json.load(open(part, "r", encoding="utf8"))

        last_batch = None
        for summary in tqdm(obj['summaries'], desc="Summary", leave=False):
            batch = summary['batch']
            if batch == last_batch:
                continue
            last_batch = batch

            summary_offsets = summary_encodings[batch].offsets
            summary_text_start = summary_offsets[0][0]
            summary_text_end = summary_offsets[len(summary_offsets) - 1][1]

            summary_text = input_text[summary_text_start:summary_text_end+1]

            describe_encodings = batch_tokenizer.encode_plus(
                summary_text,
                truncation=True,
                max_length=describe_batch_length,
                stride=describe_batch_stride,
                return_overflowing_tokens=True,
                add_special_tokens=False,
            )

            summary['descriptions'] = []
            for batch in tqdm(describe_encodings.encodings, desc="Batch", leave=False):
                batch_text_start = batch.offsets[0][0]
                batch_text_end = batch.offsets[len(batch.offsets) - 1][1]

                batch_text = summary_text[batch_text_start:
                                          batch_text_end+1].replace('\n', '')
                prompt = DESCRIBE_PROMPT.format(passage=batch_text)

                input_ids = tokenizer(
                    prompt, return_tensors="pt").input_ids.to("cuda")
                outputs = model.generate(
                    input_ids, max_new_tokens=24)
                result: str = tokenizer.decode(
                    outputs[0], skip_special_tokens=True)

                if len(result) > 10:
                    summary['descriptions'].append(result)
                    if print_descriptions:
                        print(result)

        json.dump(obj, open(part, "w", encoding="utf8"), indent=2)


@logger_error
def generate_images(title: str, draw_model_id, draw_prompt, num_title_images=12, num_images_per_description=1, single_part: Optional[str] = None):
    draw_pipe = StableDiffusionPipeline.from_pretrained(draw_model_id)
    draw_pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
        draw_pipe.scheduler.config)
    draw_pipe.set_progress_bar_config(disable=True)
    draw_pipe = draw_pipe.to("cuda")

    base_dir = get_output_dir(title)
    parts = [os.path.join(base_dir, f) for f in os.listdir(
        base_dir) if f.startswith("part" if single_part is None else f"{single_part}.") and f.endswith(".json")]

    images = get_output_dir(title, "images")

    for part in tqdm(parts, desc="Part"):
        obj = json.load(open(part, "r", encoding="utf8"))

        part_base = os.path.basename(part)
        part_base = part_base[0:part_base.rfind('.')]

        obj["images"] = []
        for title_image_index in trange(0, num_title_images, desc="Image", leave=False):
            image = draw_pipe(draw_prompt.format(description=title), height=768, width=512,
                              num_inference_steps=50, guidance_scale=7).images[0]
            image_filename = f"{part_base}-title-{title_image_index}.png"
            image.save(os.path.join(images, image_filename))
            obj["images"].append(f"images/{image_filename}")

        for summary_index, summary in tenumerate(obj['summaries'], desc="Summary", leave=False):
            summary["images"] = []

            for description_index, description in tenumerate(summary['descriptions'], desc="Description", leave=False):
                # remove some unneccesary verbage / cleanup for prompt
                description = description.replace(' are', '').replace(
                    ' is', '').replace('scene: ', '').replace(' was', '').replace(' were', '').strip()

                for description_image_index in trange(0, num_images_per_description, desc="Image", leave=False):
                    image = draw_pipe(draw_prompt.format(description=description), height=768, width=512,
                                      num_inference_steps=50, guidance_scale=7).images[0]
                    image_filename = f"{part_base}-{summary_index}-{description_index}-{description_image_index}.png"
                    image.save(os.path.join(images, image_filename))
                    summary["images"].append(f"images/{image_filename}")

        json.dump(obj, open(part, "w", encoding="utf8"), indent=2)
