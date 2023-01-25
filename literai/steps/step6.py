import argparse
import sys
from literai.steps.util import free_memory_after

@free_memory_after
def step6(title: str, txt: str):
    print("------------- STEP 6 (Finalize) ------------- ")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("title", help="title of the novel")
    parser.add_argument("author", help="author of the novel")
    args = parser.parse_args()

    step6(args.title, args.txt)


if __name__ == '__main__':
    sys.exit(main())
