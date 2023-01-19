import sys
from literai.util import arg_parser
from literai.summary import gpt_index_summarize


def main():
    args = arg_parser().parse_args()

    print("Generating base summary")
    gpt_index_summarize(args.title, args.txt, "summary",
                        max_length=256, min_length=8)
    print("Generating creative summary")
    gpt_index_summarize(args.title, args.txt, "creative", max_length=256,
                        min_length=8, early_stopping=True, length_penalty=0.4, num_beams=5)


if __name__ == '__main__':
    sys.exit(main())
