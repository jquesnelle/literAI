import shutil
import os
import torch
from gpt_index.langchain_helpers.chain_wrapper import LLMMetadata
from gpt_index import GPTTreeIndex, SimpleDirectoryReader, SummaryPrompt, LLMPredictor
from langchain import HuggingFacePipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from .util import get_output_dir

MODEL_ID = "pszemraj/long-t5-tglobal-xl-16384-book-summary"


def gpt_index_summarize(title: str, text_path: str, summary_tag: str, **kwargs):
    to_gpt_index = get_output_dir(title, "to-gpt-index")
    shutil.copy2(text_path, os.path.join(to_gpt_index, "novel.txt"))
    documents = SimpleDirectoryReader(to_gpt_index).load_data()

    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    pipe = pipeline("text2text-generation", device=0 if torch.cuda.is_available()
                    else -1, model=model, tokenizer=tokenizer, **kwargs)
    llm = HuggingFacePipeline(pipeline=pipe)

    empty_summary_prompt = SummaryPrompt("{context_str}")

    summary_predictor = LLMPredictor(llm)
    summary_predictor.get_llm_metadata = lambda: LLMMetadata(2048)

    index = GPTTreeIndex(documents, summary_template=empty_summary_prompt,
                         llm_predictor=summary_predictor, num_children=8)

    gpt_indexed = get_output_dir(title, "gpt-indexed")
    index.save_to_disk(os.path.join(
        gpt_indexed, f"gpt-indexed-{summary_tag}.json"))
