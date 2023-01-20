import sys
from literai.util import arg_parser
from literai.record import record_podcast


def main():
    parser = arg_parser()
    parser.add_argument("--alice_voice", default="train_grace")
    parser.add_argument("--bob_voice", default="train_dotrice")
    args = parser.parse_args()

    print("------------- PART 3 (Record audio) ------------- ")

    record_podcast(args.title, args.alice_voice, args.bob_voice, save_recorded_lines=True)


if __name__ == '__main__':
    sys.exit(main())
