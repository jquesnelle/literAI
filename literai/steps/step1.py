import argparse
import sys
from literai.summary import summarize
from literai.steps.util import free_memory_after

@free_memory_after
def step1(title: str, txt: str):
    print("------------- STEP 1 (Summarize text) ------------- ")
    summarize(title, txt, 2048, 512)
    summarize(title, txt, 2048, 256)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("title", help="title of the novel")
    parser.add_argument("txt", help="path to raw .txt of novel")
    args = parser.parse_args()

    step1(args.title, args.txt)


if __name__ == '__main__':
    sys.exit(main())
