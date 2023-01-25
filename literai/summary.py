
import os
from textsum.summarize import Summarizer
from literai.util import get_output_dir, logger_error

MODEL_ID = "pszemraj/long-t5-tglobal-xl-16384-book-summary"

@logger_error
def summarize(title: str, text_path: str, token_batch_length, max_new_tokens):
    summarizer = Summarizer(MODEL_ID, token_batch_length=token_batch_length,
                            max_length_ratio=(max_new_tokens / token_batch_length), encoder_no_repeat_ngram_size=None, no_repeat_ngram_size=None)
    input = open(text_path, "r", encoding="utf8").read()
    summary = summarizer.summarize_string(input)
    open(os.path.join(get_output_dir(title, "summaries"),
         f"summary-{token_batch_length}-{max_new_tokens}.txt"), "w", encoding="utf-8").write(summary)
