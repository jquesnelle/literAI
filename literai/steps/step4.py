import argparse
import sys
from typing import Optional
from literai.images import generate_images, DEFAULT_DRAW_MODEL_ID, DEFAULT_DRAW_PROMPT
from literai.steps.util import free_memory_after

@free_memory_after
def step4(title: str, draw_model: str, draw_prompt: str, single_part: Optional[str]=None):
    print("------------- STEP 4 (Draw images) ------------- ")
    generate_images(title, draw_model, draw_prompt, single_part=single_part)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("title", help="title of the novel")
    parser.add_argument(
        "--draw_model", help="stablediffusion model", default=DEFAULT_DRAW_MODEL_ID)
    parser.add_argument(
        "--draw_prompt", help="prompt template, with {description} to be replaced with description", default=DEFAULT_DRAW_PROMPT)
    parser.add_argument("--single_part", help="only run for single part (e.g. pass 'part2'")
    args = parser.parse_args()

    step4(args.title, args.draw_model, args.draw_prompt, args.single_part)


if __name__ == '__main__':
    sys.exit(main())
