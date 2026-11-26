
import argparse
import re
from pathlib import Path


def fix_segmentation(file_path):
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    text = path.read_text(encoding="utf-8")

    # Merge standalone numbered lines with the following line.
    # Supports: 1. / 2. / 10. / 1) / 1-
    text = re.sub(
        r"(?m)^(\s*\d+[.)-])\s*\n\s*([^\n]+)",
        r"\1 \2",
        text
    )

    path.write_text(text, encoding="utf-8")
    print(f"Fixed and saved: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge standalone numbering lines with the next line."
    )
    parser.add_argument("file", help="Input text file path")
    args = parser.parse_args()

    fix_segmentation(args.file)
