import os
os.environ['OPENAI_API_KEY'] = "NOTAVALIDOPENAIKEY"
os.environ['OPENAI_API_BASE'] = 'http://127.0.0.1:5000/v1'

from gpt_index.langchain_helpers.chain_wrapper import LLMMetadata
from gpt_index import GPTTreeIndex, SimpleDirectoryReader, SummaryPrompt, LLMPredictor

documents = SimpleDirectoryReader('to-gpt-index').load_data()

empty_summary_prompt = SummaryPrompt("{context_str}")

summary_predictor = LLMPredictor()
summary_predictor.get_llm_metadata = lambda: LLMMetadata(2048)

index = GPTTreeIndex(documents, summary_template=empty_summary_prompt,
                     llm_predictor=summary_predictor, num_children=8)

index.save_to_disk('gpt-indexed-summary.json')
