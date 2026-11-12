
import argparse
from pathlib import Path

from ocr_tool import PaddleOCR_Tool


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run PaddleOCR on documents and export OCR results."
    )

    # Input and output
    parser.add_argument(
        "--input-dir",
        type=str,
        default="/kaggle/input",
        help="Directory containing input documents.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/kaggle/working/ocr_raw",
        help="Directory for raw OCR results.",
    )
    parser.add_argument(
        "--txt-dir",
        type=str,
        default="/kaggle/working/txt",
        help="Directory for exported text files.",
    )

    # OCR configuration
    parser.add_argument(
        "--engine",
        type=str,
        default="paddle",
        choices=["paddle"],
        help="OCR engine.",
    )
    parser.add_argument(
        "--paddle-version",
        type=str,
        default="v1.6",
        help="PaddleOCR model/pipeline version.",
    )
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--max-side", type=int, default=1600)
    parser.add_argument("--max-new-tokens", type=int, default=6000)

    # Document processing
    parser.add_argument(
        "--pages",
        type=str,
        default="",
        help='Pages to process. Empty string means all pages.',
    )
    parser.add_argument(
        "--budget-hours",
        type=float,
        default=10.5,
        help="Maximum processing budget in hours.",
    )

    # Parallel processing
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs to use.",
    )
    parser.add_argument(
        "--no-subprocess",
        action="store_true",
        help="Disable subprocess-based multi-GPU execution.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.num_gpus < 1:
        raise ValueError("--num-gpus must be at least 1")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.txt_dir).mkdir(parents=True, exist_ok=True)

    # Initialize OCR
    ocr = PaddleOCR_Tool(
        engine=args.engine,
        paddle_version=args.paddle_version,
        dpi=args.dpi,
        max_side=args.max_side,
        max_new_tokens=args.max_new_tokens,
    )

    # Discover documents
    books = ocr.discover(
        args.input_dir,
        pages=args.pages,
    )

    print(f"Discovered {len(books)} documents.")

    # Run OCR
    ocr.run(
        books,
        output_dir=args.output_dir,
        budget_hours=args.budget_hours,
        num_gpus=args.num_gpus,
        workers_use_subprocess=not args.no_subprocess,
    )

    # Export OCR results to text files
    ocr.export_txt(
        books,
        output_dir=args.output_dir,
        txt_dir=args.txt_dir,
    )

    print("OCR processing and text export completed.")


if __name__ == "__main__":
    main()
