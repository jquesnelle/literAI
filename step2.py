import sys
from literai.util import arg_parser
from literai.script import generate_scripts


def main():
    args = arg_parser().parse_args()

    print("------------- PART 2 (Write scripts) ------------- ")

    generate_scripts(args.title)


if __name__ == '__main__':
    sys.exit(main())
