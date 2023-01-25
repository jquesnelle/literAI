import argparse
import sys
from literai.images import generate_images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("title", help="title of the novel")
    args = parser.parse_args()

    print("------------- PART 5 (Draw images) ------------- ")
    generate_images(args.title)


if __name__ == '__main__':
    sys.exit(main())
