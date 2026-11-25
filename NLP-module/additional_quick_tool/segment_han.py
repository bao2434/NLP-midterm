
import argparse
import re
from pathlib import Path


def segment_han_text(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    results = []

    for line in text.split("\n"):
        line = line.strip()

        if not line:
            continue

        # Split after Han sentence-boundary punctuation
        parts = re.split(r'(?<=[。！？；!?])', line)

        for part in parts:
            part = part.strip()
            if part:
                results.append(part)

    return results


def process_file(input_file, output_dir):
    input_path = Path(input_file)
    output_dir = Path(output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate output filename
    output_path = output_dir / (
        input_path.stem + "_segment" + input_path.suffix
    )

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    sentences = segment_han_text(text)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(sentences))

        if sentences:
            f.write("\n")

    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Total sentences: {len(sentences)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Segment classical Chinese text."
    )

    parser.add_argument("input", help="Input text file")
    parser.add_argument("output_dir", help="Output folder")

    args = parser.parse_args()

    process_file(args.input, args.output_dir)
