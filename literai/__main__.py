import argparse
import sys
from .steps.step1 import step1
from .steps.step2 import step2
from .steps.step3 import step3
from .steps.step4 import step4, DEFAULT_DRAW_MODEL_ID, DEFAULT_DRAW_PROMPT
from .steps.step5 import step5
from .steps.step6 import step6


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("title", help="title of the novel")
    parser.add_argument("author", help="author of the novel")
    parser.add_argument("txt", help="path to raw .txt of novel")
    parser.add_argument("--alice_voice", default="train_daws")
    parser.add_argument("--bob_voice", default="train_dotrice")
    parser.add_argument(
        "--draw_model", help="stablediffusion model", default=DEFAULT_DRAW_MODEL_ID)
    parser.add_argument(
        "--draw_prompt", help="prompt template, with {description} to be replaced with description", default=DEFAULT_DRAW_PROMPT)
    args = parser.parse_args()

    step1(args.title, args.txt)
    step2(args.title, args.author)
    step3(args.title, args.txt)
    step4(args.title, args.draw_model, args.draw_prompt)
    step5(args.title, args.alice_voice, args.bob_voice)
    step6(args.title, args.author)


if __name__ == '__main__':
    sys.exit(main())
