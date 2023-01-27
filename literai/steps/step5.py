import argparse
import sys
from typing import Optional
from literai.record import record_podcast
from literai.steps.util import free_memory_after

@free_memory_after
def step5(title: str, alice_voice: str, bob_voice: str, single_part: Optional[str]):
    print("------------- STEP 5 (Record audio) ------------- ")
    record_podcast(title, [alice_voice, bob_voice], single_part=single_part)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("title", help="title of the novel")
    parser.add_argument("--alice_voice", default="train_daws")
    parser.add_argument("--bob_voice", default="train_dotrice")
    parser.add_argument("--single_part", help="only run for single part (e.g. pass 'part2'")
    args = parser.parse_args()

    step5(args.title, args.alice_voice, args.bob_voice, single_part=args.single_part)


if __name__ == '__main__':
    sys.exit(main())
