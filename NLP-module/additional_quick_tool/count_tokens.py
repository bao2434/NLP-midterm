
import argparse
from pathlib import Path


def count_max_characters(input_file):
    input_path = Path(input_file)

    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    char_counts = [
        len(line.strip())
        for line in lines
        if line.strip()
    ]

    if not char_counts:
        print("The file contains no non-empty lines.")
        return

    max_chars = max(char_counts)
    max_line = char_counts.index(max_chars) + 1

    print(f"Total non-empty lines: {len(char_counts)}")
    print(f"Maximum characters: {max_chars}")
    print(f"Line number: {max_line}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find the maximum number of characters per line."
    )

    parser.add_argument("input", help="Input text file")
    args = parser.parse_args()

    count_max_characters(args.input)
