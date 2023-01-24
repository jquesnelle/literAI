import json
import os
import torch
from typing import List, Optional
from langchain import HuggingFacePipeline
from langchain.llms.base import BaseLLM
from langchain.schema import LLMResult
from langchain.prompts import BasePromptTemplate
from langchain.chains import LLMChain
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from tqdm import tqdm
from tqdm.auto import trange
from .util import get_output_dir

MODEL_ID = "allenai/cosmo-xl"
SITUATION = "Alice and Bob are hosts of a literary criticism podcast. They are discussing a passage from the book \"{title}\" by \"{author}\" that they both recently read. The conversation is academic, intelligent, nuanced, and elaborate. They are currently discussing the following passage:\n{passage}"
INSTRUCTION_ALICE = "Imagine you are Alice and ask Bob questions using specific details from the passage"
INSTRUCTION_BOB = "Imagine you are Bob and respond to Alice"


class NullLLM(BaseLLM):
    def _generate(
        self, prompts: List[str], stop: Optional[List[str]] = None
    ) -> LLMResult:
        return LLMResult([[]])

    @property
    def _llm_type(self) -> str:
        return "NullLLM"


class CosmoPromptTemplate(BasePromptTemplate):

    input_variables: List[str] = [
        "narrative", "instruction", "dialogue_history"]

    def _set_input(narrative: str, instruction: str, dialogue_history: List[str]) -> str:
        input_text = " <turn> ".join(dialogue_history)

        if instruction != "":
            input_text = instruction + " <sep> " + input_text

        if narrative != "":
            input_text = narrative + " <sep> " + input_text

        return input_text

    def format(self, **kwargs) -> str:
        return CosmoPromptTemplate._set_input(kwargs['narrative'], kwargs['instruction'], kwargs['dialogue_history'])


def ordinal(n: int):
    if 11 <= (n % 100) <= 13:
        suffix = 'th'
    else:
        suffix = ['th', 'st', 'nd', 'rd', 'th'][min(n % 10, 4)]
    return str(n) + suffix


def generate_scripts(
        title: str,
        author: str,
        summary_long="summary-2048-256.txt",
        summary_short="summary-2048-128.txt",
        sections_per_part=20,
        max_dialogue_tokens=128,
        summary_context_tokens=128,
        dialogue_history_len=3,
        dialogues_per_passage=3,
        sections_per_refresher=4,
        temperature=1.0,
        top_p=0.95,
        print_dialogue=False):

    summary_long = [x.strip() for x in open(os.path.join(get_output_dir(
        title, "summaries"), summary_long), "r", encoding="utf-8").readlines()]
    summary_short = [x.strip() for x in open(os.path.join(get_output_dir(
        title, "summaries"), summary_short), "r", encoding="utf-8").readlines()]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    parts_long = [summary_long[i:i+sections_per_part]
                  for i in range(0, len(summary_long), sections_per_part)]
    parts_short = [summary_short[i:i+sections_per_part]
                   for i in range(0, len(summary_short), sections_per_part)]

    # make last part longer
    if len(parts_long) > 1:
        parts_long[len(parts_long) - 2].extend(parts_long.pop())
        parts_short[len(parts_short) - 2].extend(parts_short.pop())

    prompt = CosmoPromptTemplate()

    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
    pipe = pipeline("text2text-generation", device=0 if torch.cuda.is_available()
                    else -1, model=model, tokenizer=tokenizer, max_length=None, max_new_tokens=max_dialogue_tokens, temperature=temperature, top_p=top_p, do_sample=True, clean_up_tokenization_spaces=False)
    llm = HuggingFacePipeline(pipeline=pipe)
    chain = LLMChain(llm=llm, prompt=prompt)

    num_parts = len(parts_long)
    batch = 0

    for part in trange(0, num_parts, desc="Part", leave=False):

        obj = {
            'book_title': title,
            'book_author': author,
            'part': part + 1,
            'num_parts': num_parts,
            'speakers': ['Alice', 'Bob'],
            'summaries': [],
            'lines': []
        }

        full_dialogue = [
            "Alice: Welcome to liter A I, your local neighborhood literary podcast where we discuss, disect, and dismantle interesting novels and stories",
            "Bob: I'm Bob, a computer generated personality",
            f"Alice: And I'm Alice, and I'm also a computer generated personality. This is the {ordinal(part+1)} of our {num_parts} part series on \"{title}\" by \"{author}\"",
            "Bob: Please support researchers that facilitate equal access to human knowledge by open sourcing their A I models",
            "Alice: That's right, Bob. Remember there's nothing open about A I that's stuck behind an A P I! Okay, let's get started"
        ]
        if print_dialogue:
            for x in full_dialogue:
                print(x)

        obj['lines'].extend([{
            'speaker': 0,
            'text': full_dialogue[0][7:],
        }, {
            'speaker': 1,
            'text': full_dialogue[1][5:]
        }, {
            'speaker': 0,
            'text': full_dialogue[2][7:],
        }, {
            'speaker': 1,
            'text': full_dialogue[3][5:],
        }, {
            'speaker': 0,
            'text': full_dialogue[4][7:],
        }])

        num_sections = len(parts_long[part])
        since_last_refresher = sections_per_refresher
        refreshers = 0
        for section in trange(0, num_sections, desc="Section", leave=False):

            if since_last_refresher == sections_per_refresher:
                full_dialogue.append(
                    f"Bob: Alright, let's talk about our {ordinal(refreshers + 1) if section + sections_per_refresher < num_sections else 'last'} section today. Here's a bit of a refresher of where we are: {parts_short[part][section]}")

                obj['lines'].append({
                    'speaker': 1,
                    'text': full_dialogue[len(full_dialogue) - 1],
                })

                if print_dialogue:
                    print(full_dialogue[len(full_dialogue)-1])

                since_last_refresher = 0
                refreshers += 1
            else:
                since_last_refresher += 1

            # split passages along token boundaries
            part_summary_long = parts_long[part][section]
            tokenized_offsets = tokenizer.encode_plus(
                part_summary_long, return_offsets_mapping=True).encodings[0].offsets
            tokenized_offsets.pop()  # remove end special token
            passage_offsets = [tokenized_offsets[i:i+summary_context_tokens]
                               for i in range(0, len(tokenized_offsets), summary_context_tokens)]
            for tokenized_passage in tqdm(passage_offsets, desc="Passage", leave=False):

                summary_start = tokenized_passage[0][0]
                summary_end = tokenized_passage[len(tokenized_passage) - 1][1]

                summary = part_summary_long[summary_start:summary_end+1]

                situation = SITUATION.format(title=title, author=author, passage=summary)
                obj['summaries'].append({"text": summary, "batch": batch})
                summary_index = len(obj['summaries']) - 1

                alice_dialogue = []
                bob_dialogue = []
                passage_dialogue = []

                # one trick here is to only pass in the opposite side of the dialogue as history.
                # it seems that cosmo can effectively "infer" what one side would have said given the
                # other, and this lets is double the history (which is needed given cosmo's very short)
                # input token limit (512)

                for _ in trange(dialogues_per_passage, desc="Dialogue", leave=False):
                    alice_history = bob_dialogue[-dialogue_history_len:]
                    response = chain.run(
                        narrative=situation, instruction=INSTRUCTION_ALICE, dialogue_history=alice_history)

                    obj['lines'].append({
                        'speaker': 0,
                        'text': response,
                        'summary': summary_index
                    })

                    response = f"Alice: {response}"
                    alice_dialogue.append(response)
                    passage_dialogue.append(response)
                    full_dialogue.append(response)
                    if print_dialogue:
                        print(response)

                    bob_history = alice_dialogue[-dialogue_history_len:]
                    response = chain.run(
                        narrative=situation, instruction=INSTRUCTION_BOB, dialogue_history=bob_history)

                    obj['lines'].append({
                        'speaker': 1,
                        'text': response,
                        'summary': summary_index
                    })

                    response = f"Bob: {response}"
                    bob_dialogue.append(response)
                    passage_dialogue.append(response)
                    full_dialogue.append(response)
                    if print_dialogue:
                        print(response)

            batch += 1

        full_dialogue.append(
            "Alice: Okay, that's all the time we have for today!")
        obj['lines'].append({
            'speaker': 0,
            'text': full_dialogue[len(full_dialogue) - 1][7:],
        })
        full_dialogue.append(
            f'Bob: Thanks for listening to our series on \"{title}\" by \"{author}\" and be sure to be on the lookout for future episodes of liter A I')
        obj['lines'].append({
            'speaker': 1,
            'text': full_dialogue[len(full_dialogue) - 1][5:],
        })

        open(os.path.join(get_output_dir(title, "scripts"), f"part{part+1}.txt"), "w",
             encoding="utf8").writelines([x + "\n" for x in full_dialogue])

        json.dump(obj, open(os.path.join(get_output_dir(
            title), f"part{part+1}.json"), "w", encoding="utf8"), indent=2)
