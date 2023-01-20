import sys
from literai.util import arg_parser
from literai.summary import gpt_index_summarize


def main():
    args = arg_parser().parse_args()

    print("------------- PART 1 (Generate summaries) ------------- ")

    print("Generating base summary")
    gpt_index_summarize(args.title, args.txt, "summary")
    print("Generating creative summary")
    gpt_index_summarize(args.title, args.txt, "creative",
                        early_stopping=True, length_penalty=0.4, num_beams=8)


if __name__ == '__main__':
    sys.exit(main())
