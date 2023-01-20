import os
import torch
from typing import List, Optional
from langchain import HuggingFacePipeline
from langchain.llms.base import BaseLLM
from langchain.schema import LLMResult
from langchain.prompts import BasePromptTemplate
from langchain.chains import LLMChain
from gpt_index import GPTTreeIndex, LLMPredictor
from gpt_index.data_structs import IndexGraph
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from tqdm import tqdm
from tqdm.auto import trange
from tqdm.contrib import tenumerate
from .util import get_output_dir

MODEL_ID = "allenai/cosmo-xl"
MAX_DIALOGUE_TOKENS = 160
TEMPERATURE = 1.0
TOP_P = 0.95
DIALOGUES_PER_PASSAGE = 5

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


SITUATION = \
    r"""Alice and Bob are hosts of a literary criticism podcast where the themes, motivations, and implications of novels are discussed. They are discussing a passage from a book they both recently read. The conversation is intelligent, nuanced, and elaborate.
They are currently discussing the following passage:
"""

INSTRUCTION_ALICE = "Imagine you are Alice and ask Bob questions about the passage"
INSTRUCTION_BOB = "Imagine you are Bob and speak to Alice"


def generate_scripts(title: str, part_glob=6, print_dialogue=False):
    gpt_indexed = get_output_dir(title, "gpt-indexed")
    scripts = get_output_dir(title, "scripts")

    summary = GPTTreeIndex.load_from_disk(
        os.path.join(gpt_indexed, "gpt-indexed-summary.json"), llm_predictor=LLMPredictor(NullLLM()))
    summary_index: IndexGraph = summary.index_struct

    summary_creative = GPTTreeIndex.load_from_disk(
        os.path.join(gpt_indexed, "gpt-indexed-creative.json"), llm_predictor=LLMPredictor(NullLLM()))
    summary_creative_index: IndexGraph = summary_creative.index_struct

    part_indicies = []
    for root_node in summary_index.root_nodes.values():
        part_indicies.extend(root_node.child_indices)
    part_indicies = sorted(part_indicies)

    joined_part_indicies: List[List[int]] = []
    for part_indicies_index in range(0, len(part_indicies), part_glob):
        if part_indicies_index + part_glob < len(part_indicies):
            joined_part_indicies.append(
                [part_indicies[part_indicies_index + i] for i in range(0, part_glob)])
        else:
            joined_part_indicies[len(
                joined_part_indicies) - 1].extend(part_indicies[part_indicies_index:])

    prompt = CosmoPromptTemplate()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
    pipe = pipeline("text2text-generation", device=0 if torch.cuda.is_available()
                    else -1, model=model, tokenizer=tokenizer, max_length=None, max_new_tokens=MAX_DIALOGUE_TOKENS, temperature=TEMPERATURE, top_p=TOP_P, do_sample=True, clean_up_tokenization_spaces=False)
    llm = HuggingFacePipeline(pipeline=pipe)
    chain = LLMChain(llm=llm, prompt=prompt)

    for part, part_sections in tenumerate(joined_part_indicies, desc="Part", leave=False):

        full_dialogue = [
            "Alice: Welcome to liter-AI, your local neighborhood literary podcast where we discuss, disect, and dismantle interesting novels and stories",
            "Bob: I'm Bob, a computer generated personality",
            f"Alice: And I'm Alice, and I'm also a computer generated personality. This is the {ordinal(part+1)} of our {len(joined_part_indicies)} part series on \"{title}\"",
            "Bob: Please support researchers that facilitate equal access to human knowledge by open sourcing their AI models",
            "Alice: That's right, Bob. Remember there's nothing open about AI that's stuck behind an API! Okay, let's get started"
        ]
        if print_dialogue:
            for x in full_dialogue:
                print(x)

        for section, section_index in tenumerate(part_sections, desc="Section", leave=False):

            section_node = summary_index.all_nodes[section_index]
            full_dialogue.append(
                f"Bob: Alright, let's talk about our {ordinal(section+1) if section+1 != len(part_sections) else 'last'} section today. Here's a bit of a refresher on what happened: {section_node.text}")
            if print_dialogue:
                print(full_dialogue[len(full_dialogue)-1])

            for passage_index in tqdm(section_node.child_indices, desc="Passage", leave=False):

                alice_situation = SITUATION + \
                    summary_index.all_nodes[passage_index].text
                bob_situation = SITUATION + \
                    summary_creative_index.all_nodes[passage_index].text

                alice_dialogue = []
                bob_dialogue = []
                passage_dialogue = []

                for _ in trange(DIALOGUES_PER_PASSAGE, desc="Dialogue", leave=False):
                    response = chain.run(
                        narrative=alice_situation, instruction=INSTRUCTION_ALICE, dialogue_history=bob_dialogue)
                    response = f"Alice: {response}"
                    alice_dialogue.append(response)
                    passage_dialogue.append(response)
                    full_dialogue.append(response)
                    if print_dialogue:
                        print(response)

                    response = chain.run(
                        narrative=bob_situation, instruction=INSTRUCTION_BOB, dialogue_history=alice_dialogue)
                    response = f"Bob: {response}"
                    bob_dialogue.append(response)
                    passage_dialogue.append(response)
                    full_dialogue.append(response)
                    if print_dialogue:
                        print(response)

        open(os.path.join(scripts, f"part{part+1}.txt"), "w",
             encoding="utf8").writelines([x + "\n" for x in full_dialogue])
