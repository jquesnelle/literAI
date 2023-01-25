import argparse
import sys
from literai.script import generate_scripts
from literai.steps.util import free_memory_after

@free_memory_after
def step2(title: str, author: str):
    print("------------- STEP 2 (Write scripts) ------------- ")
    generate_scripts(title, author)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("title", help="title of the novel")
    parser.add_argument("author", help="author of the novel")
    args = parser.parse_args()

    step2(args.title, args.author)


if __name__ == '__main__':
    sys.exit(main())
