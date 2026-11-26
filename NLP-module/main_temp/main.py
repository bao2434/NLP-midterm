"""CLI entry point for running Kim Han-Nom Sentence Segmentation API on a folder of text files."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from segmentation_api import HanNomSegmenter, SegmentationAPIConfig
from segmentation_api.api_client import (
    DEFAULT_BASE_URL,
    DEFAULT_USER_AGENT,
    VALID_ENDPOINTS,
    VALID_GENRES,
    VALID_PUNCTUATION_GROUPS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Discover all text files in a folder, divide each file into non-empty line chunks "
            "(under the 10,000-token limit), send each chunk sequentially to the Kim Han-Nom API "
            "(https://kimhannom.clc.hcmus.edu.vn/), and write concatenated segmented sentences "
            "to the output directory."
        )
    )

    # Input and output paths
    parser.add_argument(
        "--input-dir",
        type=str,
        default=".",
        help="Directory (or single text file) containing input text files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./seg_api_output",
        help="Directory where received segmented text files will be written.",
    )
    parser.add_argument(
        "--extensions",
        type=str,
        default=".txt",
        help="Comma-separated file extensions to include (default: '.txt').",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only search the top-level input directory for text files.",
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
        help="API base URL (default: https://kimhannom.clc.hcmus.edu.vn).",
    )
    parser.add_argument(
        "--user-agent",
        type=str,
        default=DEFAULT_USER_AGENT,
        help="Custom User-Agent header (must not start with 'python-requests').",
    )

    # Segmentation API parameters
    parser.add_argument(
        "--endpoint",
        type=str,
        default="separate-sentences",
        choices=list(VALID_ENDPOINTS),
        help=(
            "Kim Han-Nom API endpoint: 'separate-sentences' (default: punctuation + sentence splitting) "
            "or 'sentence-punctuation' (punctuation only)."
        ),
    )
    parser.add_argument(
        "--genre",
        type=str,
        default="medical",
        choices=list(VALID_GENRES),
        help="Document genre for the API model (default: 'medical' for traditional medicine).",
    )
    parser.add_argument(
        "--punctuation-group",
        type=str,
        default="all",
        choices=list(VALID_PUNCTUATION_GROUPS),
        help=(
            "Punctuation marks to retain: 'all' (default: 。？！，：、；), "
            "'fullstop' (。？！), or 'other_punctuation' (，：、；)."
        ),
    )

    # Rate-limiting & execution controls
    parser.add_argument(
        "--rate-limit-mode",
        type=str,
        default="limited",
        choices=["limited", "unlimited"],
        help=(
            "API call limit mode: 'limited' restricts calls according to configured limits; "
            "'unlimited' calls the API continuously and sequentially without caring about limits "
            "(default: limited)."
        ),
    )
    parser.add_argument(
        "--unlimited",
        action="store_true",
        help=(
            "Call the API continuously and sequentially without caring about rate limits "
            "(disables all local caps, error-quota stops, and default delay)."
        ),
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=None,
        help=(
            "Delay in seconds between consecutive API requests "
            "(default: 0.0 in --unlimited mode, 3.0 in limited mode)."
        ),
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
        default=300.0,
        help="HTTP request timeout in seconds (default: 300.0).",
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
        help="Stop early after N consecutive errors in limited mode (0 disables; ignored in unlimited mode).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Re-process files even if an output file already exists in output-dir.",
    )
    parser.add_argument(
        "--no-rate-limit-guard",
        action="store_true",
        help="Deprecated alias for --unlimited.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover text files and list them without calling the API.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rate_limit_mode = (
        "unlimited"
        if (args.unlimited or args.no_rate_limit_guard or args.rate_limit_mode == "unlimited")
        else "limited"
    )

    extensions = [ext.strip() for ext in args.extensions.split(",") if ext.strip()]

    config = SegmentationAPIConfig(
        base_url=args.base_url,
        token=args.token,
        user_agent=args.user_agent,
        endpoint=args.endpoint,
        genre=args.genre,
        punctuation_group=args.punctuation_group,
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
    )

    with HanNomSegmenter(config) as segmenter:
        docs = segmenter.discover(
            args.input_dir,
            extensions=extensions,
            recursive=not args.no_recursive,
        )
        print(
            f"Discovered {len(docs)} text file(s) in {Path(args.input_dir).resolve()}:"
        )
        total_chunks = 0
        for name, src in docs.items():
            raw_text = Path(src).read_text(encoding="utf-8-sig", errors="replace")
            chunks = segmenter.split_chunks(raw_text)
            total_chunks += len(chunks)
            max_chunk_len = max((len(c) for c in chunks), default=0)
            print(
                f"  - {name}: {len(raw_text)} char(s), {len(chunks)} line chunk(s) "
                f"(max chunk: {max_chunk_len} chars) ({src})"
            )

        if not docs:
            print("No text files found.")
            return

        print(
            f"Total files: {len(docs)} ({total_chunks} line-chunk API request(s)). "
            f"Endpoint: {config.endpoint}, genre: {config.genre}, "
            f"punctuation: {config.punctuation_group}, "
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

        written = segmenter.run(
            docs,
            output_dir=args.output_dir,
            resume=not args.no_resume,
            budget_hours=args.budget_hours,
        )
        print(
            f"Completed: {len(written)}/{len(docs)} file(s) written to "
            f"{Path(args.output_dir).resolve()}."
        )


if __name__ == "__main__":
    main()
