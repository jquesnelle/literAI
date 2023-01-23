import argparse
import sys
from literai.record import record_podcast


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("title", help="title of the novel")
    parser.add_argument("--alice_voice", default="train_daws")
    parser.add_argument("--bob_voice", default="train_dotrice")
    args = parser.parse_args()

    print("------------- PART 3 (Record audio) ------------- ")
    record_podcast(args.title, [args.alice_voice, args.bob_voice])


if __name__ == '__main__':
    sys.exit(main())
