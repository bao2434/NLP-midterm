
import argparse

def cut_lines(input_file, output_file, start_line, end_line):
    if start_line < 1 or end_line < start_line:
        raise ValueError("Invalid line range.")

    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Remove lines from start_line to end_line (inclusive)
    remaining_lines = lines[start_line - 1:end_line]

    with open(output_file, "w", encoding="utf-8") as f:
        f.writelines(remaining_lines)

    print(f"Removed lines {start_line}-{end_line}.")
    print(f"Output saved to: {output_file}")
    print(f"Original lines: {len(lines)}")
    print(f"Remaining lines: {len(remaining_lines)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Remove a range of lines from a text file."
    )

    parser.add_argument("input", help="Input text file")
    parser.add_argument("output", help="Output text file")
    parser.add_argument("s", type=int, help="Starting line (1-based)")
    parser.add_argument("e", type=int, help="Ending line (1-based, inclusive)")

    args = parser.parse_args()

    cut_lines(args.input, args.output, args.s, args.e)
