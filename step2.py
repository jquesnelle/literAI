import argparse
import sys
from literai.script import generate_scripts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("title", help="title of the novel")
    parser.add_argument("author", help="author of the novel")
    args = parser.parse_args()

    print("------------- PART 2 (Write scripts) ------------- ")
    generate_scripts(args.title, args.author)


if __name__ == '__main__':
    sys.exit(main())
