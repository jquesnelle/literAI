import argparse
import sys
from literai.images import generate_image_descriptions
from literai.steps.util import free_memory_after

@free_memory_after
def step3(title: str, txt: str):
    print("------------- STEP 3 (Make image descriptions) ------------- ")
    generate_image_descriptions(title, txt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("title", help="title of the novel")
    parser.add_argument("txt", help="path to raw .txt of novel")
    args = parser.parse_args()

    step3(args.title, args.txt)


if __name__ == '__main__':
    sys.exit(main())
