import argparse
import sys
from literai.summary import summarize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("title", help="title of the novel")
    parser.add_argument("txt", help="path to raw .txt of novel")
    args = parser.parse_args()

    print("------------- PART 1 (Summarize text) ------------- ")
    summarize(args.title, args.txt, 2048, 512)
    summarize(args.title, args.txt, 2048, 256)


if __name__ == '__main__':
    sys.exit(main())
