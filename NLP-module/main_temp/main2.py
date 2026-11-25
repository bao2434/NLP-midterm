"""CLI entry point for running Kim Han-Nom OCR API on a folder of PDFs."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from ocr_api import APIConfig, HanNomOCR
from ocr_api.api_client import DEFAULT_BASE_URL, DEFAULT_USER_AGENT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect PDFs in a folder and OCR each page sequentially using the "
            "Kim Han-Nom API (https://kimhannom.fit.hcmus.edu.vn/)."
        )
    )

    # Input and output paths
    parser.add_argument(
        "--input-dir",
        type=str,
        default=".",
        help="Directory (or single PDF file) containing input PDFs.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./ocr_api_raw",
        help="Directory for incremental JSONL OCR results.",
    )
    parser.add_argument(
        "--txt-dir",
        type=str,
        default="./ocr_api_txt",
        help="Directory for exported plain-text files.",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only search the top-level input directory for PDFs.",
    )

    # API authentication & endpoint
    parser.add_argument(
        "--token",
        type=str,
        default=os.environ.get("HANNOM_API_TOKEN"),
        help=(
            "Bearer token from the website cookie 'token' "
            "(defaults to HANNOM_API_TOKEN env var)."
        ),
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=DEFAULT_BASE_URL,
        help="API base URL.",
    )
    parser.add_argument(
        "--user-agent",
        type=str,
        default=DEFAULT_USER_AGENT,
        help="Custom User-Agent header (must not start with 'python-requests').",
    )

    # OCR mode & parameters
    parser.add_argument(
        "--mode",
        type=str,
        default="hannom",
        choices=["hannom", "viet"],
        help="OCR endpoint mode: 'hannom' (image-upload + image-ocr) or 'viet' (viet-ocr).",
    )
    parser.add_argument(
        "--ocr-id",
        type=int,
        default=1,
        choices=[1, 2, 3, 4],
        help="Han-Nom OCR type: 1=normal vertical, 2=administrative, 3=scene, 4=normal horizontal.",
    )
    parser.add_argument(
        "--lang-type",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Language type: 0=unknown, 1=Han, 2=Nom.",
    )
    parser.add_argument(
        "--reading-direction",
        type=int,
        default=1,
        choices=[1, 2],
        help="Reading direction: 1=vertical (default), 2=horizontal.",
    )
    parser.add_argument(
        "--epitaph",
        type=int,
        default=0,
        choices=[0, 1],
        help="Document style: 0=normal document, 1=epitaph (van bia).",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional specialized model_id tag assigned by admin.",
    )
    parser.add_argument(
        "--auto-classify",
        action="store_true",
        help="Call /structure-classification before /image-ocr to auto-detect ocr_id/lang_type/direction (+1 req/page).",
    )

    # Rendering & page selection
    parser.add_argument(
        "--pages",
        type=str,
        default="",
        help="1-based pages to process, e.g. '1-5,10'. Empty string means all pages.",
    )
    parser.add_argument("--dpi", type=int, default=200, help="PDF render DPI.")
    parser.add_argument(
        "--max-side",
        type=int,
        default=1600,
        help="Resize rendered page so longest side <= max-side (0 disables resizing).",
    )
    parser.add_argument(
        "--image-format",
        type=str,
        default="JPEG",
        choices=["JPEG", "PNG"],
        help="Image format sent to the API.",
    )

    # Rate-limiting & execution controls
    parser.add_argument(
        "--rate-limit-mode",
        type=str,
        default="limited",
        choices=["limited", "unlimited"],
        help=(
            "API call limit mode: 'limited' restricts calls according to configured limits; "
            "'unlimited' disables local request limits (default: limited)."
        ),
    )
    parser.add_argument(
        "--unlimited",
        action="store_true",
        help="Shorthand for --rate-limit-mode unlimited (allow unlimited API calls).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=3.0,
        help="Delay in seconds between consecutive API requests (default: 3.0; 0 disables).",
    )
    parser.add_argument(
        "--max-requests-per-minute",
        type=int,
        default=25,
        help="Maximum API requests per minute in limited mode (default: 25; 0 disables).",
    )
    parser.add_argument(
        "--max-requests-per-hour",
        type=int,
        default=50,
        help="Maximum API requests per hour in limited mode (default: 50; 0 disables).",
    )
    parser.add_argument(
        "--max-requests-per-day",
        type=int,
        default=200,
        help="Maximum API requests per day in limited mode (default: 200; 0 disables).",
    )
    parser.add_argument(
        "--max-requests-per-week",
        type=int,
        default=600,
        help="Maximum API requests per week in limited mode (default: 600; 0 disables).",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=None,
        help="Optional maximum number of API requests for this run (ignored in unlimited mode).",
    )
    parser.add_argument(
        "--budget-hours",
        type=float,
        default=None,
        help="Optional maximum runtime budget in hours.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="HTTP request timeout in seconds.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retries on transient network/5xx errors.",
    )
    parser.add_argument(
        "--max-consecutive-errors",
        type=int,
        default=5,
        help="Stop early after N consecutive page errors to preserve API quota (0 disables).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Re-process pages even if already saved in output-dir.",
    )
    parser.add_argument(
        "--no-rate-limit-guard",
        action="store_true",
        help="Deprecated alias for --unlimited (disable local request limits).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover PDFs and estimate API requests without calling the API.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rate_limit_mode = (
        "unlimited"
        if (args.unlimited or args.no_rate_limit_guard or args.rate_limit_mode == "unlimited")
        else "limited"
    )

    config = APIConfig(
        base_url=args.base_url,
        token=args.token,
        user_agent=args.user_agent,
        mode=args.mode,
        ocr_id=args.ocr_id,
        lang_type=args.lang_type,
        reading_direction=args.reading_direction,
        epitaph=args.epitaph,
        tag=args.tag,
        auto_classify=args.auto_classify,
        rate_limit_mode=rate_limit_mode,
        request_delay=args.delay,
        max_requests_per_minute=args.max_requests_per_minute,
        max_requests_per_hour=args.max_requests_per_hour,
        max_requests_per_day=args.max_requests_per_day,
        max_requests_per_week=args.max_requests_per_week,
        max_requests_per_run=args.max_requests,
        max_consecutive_errors=args.max_consecutive_errors,
        timeout=args.timeout,
        max_retries=args.max_retries,
        dpi=args.dpi,
        max_side=args.max_side,
        image_format=args.image_format,
    )

    with HanNomOCR(config) as ocr:
        books = ocr.discover(
            args.input_dir,
            pages=args.pages,
            recursive=not args.no_recursive,
        )
        print(f"Discovered {len(books)} PDF document(s) in {Path(args.input_dir).resolve()}:")
        total_pages = 0
        for name, (src, selected) in books.items():
            total_pages += len(selected)
            print(f"  - {name}: {len(selected)} page(s) ({src})")

        if not books:
            print("No PDF files found.")
            return

        reqs_per_page = 1 if args.mode == "viet" else (3 if args.auto_classify else 2)
        est_requests = total_pages * reqs_per_page
        print(
            f"Total selected pages: {total_pages} "
            f"(~{est_requests} API request(s) at {reqs_per_page} req/page). "
            f"API call policy: {config.rate_limit_summary()}, delay={config.request_delay}s."
        )

        if args.dry_run:
            print("Dry run complete. No API requests were sent.")
            return

        if not config.token:
            print(
                "[WARNING] No --token / HANNOM_API_TOKEN provided. "
                "Authenticated endpoints may reject requests without a Bearer token."
            )

        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        Path(args.txt_dir).mkdir(parents=True, exist_ok=True)

        ocr.run(
            books,
            output_dir=args.output_dir,
            resume=not args.no_resume,
            budget_hours=args.budget_hours,
        )
        ocr.export_txt(
            books,
            output_dir=args.output_dir,
            txt_dir=args.txt_dir,
        )
        print("OCR API processing and text export completed.")


if __name__ == "__main__":
    main()
