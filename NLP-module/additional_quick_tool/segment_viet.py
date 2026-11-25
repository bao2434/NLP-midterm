
import argparse
import re
from pathlib import Path


# Common Vietnamese abbreviations
ABBREVIATIONS = [
    "PGS.", "GS.", "TS.", "ThS.", "BS.", "DS.",
    "KS.", "CN.", "TP.", "Q.", "P.",
    "Tp.", "Mr.", "Mrs.", "Dr.",
    "v.v.", "vv.", "etc.",
]

def segment_text(text):
    # Temporary placeholder for protected periods
    placeholder = "\uE000"

    # Protect decimal points: 3.14, 10.5, etc.
    text = re.sub(
        r"(?<=\d)\.(?=\d)",
        placeholder,
        text
    )

    # Protect common abbreviations
    for i, abbreviation in enumerate(ABBREVIATIONS):
        protected = abbreviation.replace(".", placeholder)
        text = re.sub(
            re.escape(abbreviation),
            protected,
            text,
            flags=re.IGNORECASE
        )

    # Protect initials in names, e.g. N.V.A.
    text = re.sub(
        r"\b(?:[A-ZĐ]\.){2,}",
        lambda m: m.group().replace(".", placeholder),
        text
    )

    # Split after punctuation when followed by whitespace
    # Punctuation is preserved.
    segments = re.split(
        r"(?<=[.!?;:])\s+",
        text
    )

    # Restore protected periods
    segments = [
        segment.replace(placeholder, ".").strip()
        for segment in segments
    ]

    # Remove empty segments
    return [s for s in segments if s]


def process_file(input_path):
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    output_path = input_path.with_name(
        input_path.stem + "_segment" + input_path.suffix
    )

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    segments = segment_text(text)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(segments))

    print(f"Input file: {input_path}")
    print(f"Output file: {output_path}")
    print(f"Total segments: {len(segments)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Vietnamese text segmentation"
    )

    parser.add_argument(
        "input",
        help="Input text file"
    )

    args = parser.parse_args()

    process_file(args.input)
