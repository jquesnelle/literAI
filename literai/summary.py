import shutil
import os
import torch
from gpt_index.langchain_helpers.chain_wrapper import LLMMetadata
from gpt_index import GPTTreeIndex, SimpleDirectoryReader, SummaryPrompt, LLMPredictor, PromptHelper
from langchain import HuggingFacePipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from .util import get_output_dir

MODEL_ID = "pszemraj/long-t5-tglobal-xl-16384-book-summary"
MAX_SUMMARIZE_TOKENS = 1024
TOKEN_OVERLAP = 16
MIN_TOKENS = 8
MAX_TOKENS = 128


def gpt_index_summarize(title: str, text_path: str, summary_tag: str, **kwargs):
    to_gpt_index = get_output_dir(title, "to-gpt-index")
    shutil.copy2(text_path, os.path.join(to_gpt_index, "novel.txt"))
    documents = SimpleDirectoryReader(to_gpt_index).load_data()

    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    pipe = pipeline("text2text-generation", device=0 if torch.cuda.is_available()
                    else -1, model=model, tokenizer=tokenizer, max_length=MAX_TOKENS, min_length=MIN_TOKENS, **kwargs)
    llm = HuggingFacePipeline(pipeline=pipe)

    empty_summary_prompt = SummaryPrompt("{context_str}")

    summary_predictor = LLMPredictor(llm)
    summary_predictor.get_llm_metadata = lambda: LLMMetadata(
        MAX_SUMMARIZE_TOKENS)

    prompt_helper = PromptHelper(
        MAX_SUMMARIZE_TOKENS, 0, TOKEN_OVERLAP, tokenizer=tokenizer)

    index = GPTTreeIndex(documents, summary_template=empty_summary_prompt,
                         llm_predictor=summary_predictor, num_children=8,
                         prompt_helper=prompt_helper)

    gpt_indexed = get_output_dir(title, "gpt-indexed")
    index.save_to_disk(os.path.join(
        gpt_indexed, f"gpt-indexed-{summary_tag}.json"))
