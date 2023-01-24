import argparse
import sys
from literai.images import generate_image_descriptions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("title", help="title of the novel")
    parser.add_argument("txt", help="path to raw .txt of novel")
    args = parser.parse_args()

    print("------------- PART 4 (Make image descriptions) ------------- ")
    generate_image_descriptions(args.title, args.txt)


if __name__ == '__main__':
    sys.exit(main())
