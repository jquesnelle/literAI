import os
from typing import Dict, List
os.environ['OPENAI_API_KEY'] = "NOTAVALIDOPENAIKEY"
os.environ['OPENAI_API_BASE'] = 'http://127.0.0.1:5000/v1'

from gpt_index import GPTTreeIndex
from gpt_index.data_structs import IndexGraph, Node
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("allenai/cosmo-xl")
model = AutoModelForSeq2SeqLM.from_pretrained("allenai/cosmo-xl").to('cuda')

def set_input(narrative, instruction, dialogue_history):
    input_text = " <turn> ".join(dialogue_history)

    if instruction != "":
        input_text = instruction + " <sep> " + input_text

    if narrative != "":
        input_text = narrative + " <sep> " + input_text
    

    return input_text

def generate(narrative, instruction, dialogue_history):
    """
    narrative: the description of situation/context with the characters included (e.g., "David goes to an amusement park")
    instruction: the perspective/speaker instruction (e.g., "Imagine you are David and speak to his friend Sarah").
    dialogue history: the previous utterances in the dialogue in a list
    """

    input_text = set_input(narrative, instruction, dialogue_history) 

    inputs = tokenizer([input_text], return_tensors="pt").to('cuda')
    outputs = model.generate(inputs["input_ids"], max_new_tokens=128, temperature=1.0, top_p=.95, do_sample=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)

    return response

def ordinal(n: int):
    if 11 <= (n % 100) <= 13:
        suffix = 'th'
    else:
        suffix = ['th', 'st', 'nd', 'rd', 'th'][min(n % 10, 4)]
    return str(n) + suffix

summary = GPTTreeIndex.load_from_disk("gpt-indexed-summary.json")
summary_index: IndexGraph = summary.index_struct

summary_creative = GPTTreeIndex.load_from_disk("gpt-indexed-summary-creative.json")
summary_creative_index: IndexGraph = summary_creative.index_struct

part_indicies = []
for root_node in summary_index.root_nodes.values():
    part_indicies.extend(root_node.child_indices)
part_indicies = sorted(part_indicies)

part_glob = 3
joined_part_indicies: List[List[int]] = []
for part_indicies_index in range(0, len(part_indicies), part_glob):
    if part_indicies_index + part_glob < len(part_indicies):
        joined_part_indicies.append([part_indicies[part_indicies_index + i] for i in range(0, part_glob)])
    else:
        joined_part_indicies[len(joined_part_indicies) - 1].extend(part_indicies[part_indicies_index:])

title = "Harry Potter and the Sorcerer's Stone"

situation = \
r"""Alice and Bob are hosts of a literary criticism podcast where the themes, motivations, and implications of novels are discussed. They are discussing a passage from a book they both recently read. The conversation is intelligent, nuanced, and elaborate.
They are currently discussing the following passage:
"""

instruction_alice = \
r"""Imagine you are Alice and ask Bob questions about the passage"""
instruction_bob = \
r"""Imagine you are Bob and speak to Alice"""

for part, part_sections in enumerate(joined_part_indicies):

    full_dialogue = [
        "Alice: Welcome to liter-AI, your local neighborhood literary podcast where we discuss, disect, and dismantle interesting novels and stories",
        "Bob: I'm Bob, a computer generated personality",
        f"Alice: And I'm Alice, and I'm also a computer generated personality. This is the {ordinal(part+1)} of our {len(joined_part_indicies)} part series on \"{title}\"",
        "Alice: Okay, let's get started"
    ]

    for section, section_index in enumerate(part_sections):

        section_node = summary_index.all_nodes[section_index]
        full_dialogue.append(f"Bob: Alright, let's talk about our {ordinal(section+1) if section+1 != len(part_sections) else 'last'} section today. Here's a bit of a refresher on what happened: {section_node.text}")

        for passage_index in section_node.child_indices:

            alice_situation = situation + summary_index.all_nodes[passage_index].text
            bob_situation = situation + summary_creative_index.all_nodes[passage_index].text

            partial_dialogue = []

            for i in range(0, 5):
                response = generate(alice_situation, instruction_alice, partial_dialogue)
                response = f"Alice: {response}"
                partial_dialogue.append(response)
                full_dialogue.append(response)
                print(response)

                response = generate(bob_situation, instruction_bob, partial_dialogue)
                response = f"Bob: {response}"
                full_dialogue.append(response)
                print(response)

    open(f"part{part+1}.txt", "w", encoding="utf8").writelines([x + "\n" for x in full_dialogue])