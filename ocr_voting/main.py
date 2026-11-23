"""CLI entry point for 5-step Levenshtein OCR voting and disagreement reporting."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from file_matcher import discover_document_groups
from voter import DocumentVoteResult, vote_document


def _fmt_cell(text: str) -> str:
    if text == "":
        return "`[EMPTY]`"
    escaped = (
        text.replace("|", "\\|")
        .replace("\n", "\\n")
        .replace("\r", "")
        .replace("`", "'")
    )
    return f"`{escaped}`"


def _parse_weights(raw_weights: list[str] | None) -> dict[str, float] | None:
    if not raw_weights:
        return None
    parsed: dict[str, float] = {}
    for item in raw_weights:
        if "=" not in item:
            raise ValueError(
                f"Invalid --weights entry '{item}'. Expected format: model_name=weight (e.g. ocr_hunyuang=1.2)"
            )
        name, val = item.split("=", 1)
        w = float(val)
        if w <= 0:
            raise ValueError(f"Weight for '{name}' must be > 0, got {w}")
        parsed[name.strip()] = w
    return parsed


def write_markdown_report(
    result: DocumentVoteResult,
    output_path: Path,
    max_entries: int = 0,
) -> None:
    """Write a human-readable Markdown report of all disagreements with confidence scores."""
    summary = result.to_summary_dict()
    weight_str = ", ".join(
        f"{m}={summary['model_weights'][m]:.3f}" for m in result.models
    )

    lines: list[str] = [
        f"# OCR Voting Disagreement Report: {result.doc_name}",
        "",
        "## Summary",
        "",
        f"- **Models & Weights**: {weight_str}",
        f"- **Confidence threshold**: {summary['confidence_threshold']:.2f}",
        f"- **Aligned blocks (pages)**: {summary['total_blocks']}",
        f"- **Total aligned character columns**: {summary['total_columns']:,}",
        f"- **Unanimous columns**: {summary['unanimous_columns']:,} ({summary['agreement_rate'] * 100:.2f}%)",
        f"- **Average column confidence**: {summary['avg_confidence'] * 100:.2f}%",
        f"- **High-confidence disagreement columns**: {summary['high_confidence_disagreement_columns']:,}",
        f"- **Low-confidence disagreement columns**: {summary['low_confidence_disagreement_columns']:,}",
        f"- **Disagreement entries**: {summary['total_disagreement_entries']:,} "
        f"(high_confidence: {summary['high_confidence_entries']:,}, "
        f"low_confidence: {summary['low_confidence_entries']:,})",
        "",
        "## Disagreements",
        "",
    ]

    if not result.disagreements:
        lines.append("All models agreed 100% on this document.")
    else:
        shown = (
            result.disagreements[:max_entries]
            if max_entries > 0
            else result.disagreements
        )
        if max_entries > 0 and len(result.disagreements) > max_entries:
            lines.append(
                f"> Showing first {max_entries:,} of {len(result.disagreements):,} "
                f"disagreements. See the corresponding `.json` file for the complete list."
            )
            lines.append("")

        model_headers = " | ".join(result.models)
        sep_headers = " | ".join(["---"] * len(result.models))
        lines.append(
            f"| # | Block | Pos (block) | Resolution | Conf | Context (`【voted】`) | Voted | {model_headers} |"
        )
        lines.append(f"| --- | --- | --- | --- | ---: | --- | --- | {sep_headers} |")

        for d in shown:
            voted_marker = d.voted_text if d.voted_text else "∅"
            ctx = f"{d.context_before}【{voted_marker}】{d.context_after}"
            ctx_escaped = ctx.replace("|", "\\|").replace("\n", "\\n")
            model_cells = " | ".join(_fmt_cell(d.votes[m]) for m in result.models)
            lines.append(
                f"| {d.index} | {d.block_index} | {d.block_start}-{d.block_end} | "
                f"{d.resolution} | {d.confidence:.2f} | {ctx_escaped} | "
                f"{_fmt_cell(d.voted_text)} | {model_cells} |"
            )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary_markdown(summaries: list[dict], output_path: Path) -> None:
    lines = [
        "# OCR Voting Overall Summary (Levenshtein + Weighted Confidence)",
        "",
        "| Document | Blocks | Aligned Cols | Unanimous % | Avg Conf % | Disagreements | High-Conf | Low-Conf | Model Weights |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for s in summaries:
        w_str = ", ".join(f"{m}:{w:.2f}" for m, w in s["model_weights"].items())
        lines.append(
            f"| {s['doc_name']} | {s['total_blocks']} | {s['total_columns']:,} | "
            f"{s['agreement_rate'] * 100:.2f}% | {s['avg_confidence'] * 100:.2f}% | "
            f"{s['total_disagreement_entries']:,} | {s['high_confidence_entries']:,} | "
            f"{s['low_confidence_entries']:,} | {w_str} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    default_input = SCRIPT_DIR.parent / "res"
    default_output = SCRIPT_DIR / "output"

    parser = argparse.ArgumentParser(
        description="5-step Levenshtein OCR voting with weighted confidence resolution."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=default_input,
        help=f"Directory containing model subfolders (default: {default_input})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help=f"Directory to save voted texts and disagreement reports (default: {default_output})",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Optional substring filter to process only matching document(s).",
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        default=None,
        help="Optional per-model weights, e.g. ocr_hunyuang=1.2 ocr_paddle=1.1 ocr_api=0.8. "
        "If omitted, weights are automatically estimated from pairwise agreement.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Confidence threshold separating high_confidence vs low_confidence disagreements (default: 0.5).",
    )
    parser.add_argument(
        "--per-char",
        action="store_true",
        help="Report every disagreeing column individually instead of grouping adjacent disagreements.",
    )
    parser.add_argument(
        "--keep-whitespace",
        action="store_true",
        help="Keep internal whitespace/line breaks inside blocks during normalization.",
    )
    parser.add_argument(
        "--keep-punctuation",
        action="store_true",
        help="Do not normalize punctuation variants during Step 1 normalization.",
    )
    parser.add_argument(
        "--context",
        type=int,
        default=15,
        help="Surrounding consensus characters to show around each disagreement (default: 15).",
    )
    parser.add_argument(
        "--max-md-disagreements",
        type=int,
        default=0,
        help="Max disagreements to render in Markdown table (0 = all; JSON always has all).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    explicit_weights = _parse_weights(args.weights)

    model_names, groups = discover_document_groups(
        args.input_dir, file_filter=args.file
    )
    print(f"Discovered models: {', '.join(model_names)}")
    print(f"Matched {len(groups)} document(s) in {args.input_dir}\n")

    if not groups:
        print("No matching documents found.")
        return

    all_summaries: list[dict] = []

    for group in groups:
        if len(group.model_files) < 2:
            print(
                f"[SKIP] {group.doc_key}: only present in {list(group.model_files.keys())}"
            )
            continue

        print(f"Processing: {group.doc_key} ({len(group.model_files)} models)...")
        model_texts = {
            m: path.read_text(encoding="utf-8")
            for m, path in group.model_files.items()
        }

        result = vote_document(
            doc_name=group.doc_key,
            model_texts_full=model_texts,
            model_weights=explicit_weights,
            confidence_threshold=args.confidence_threshold,
            ignore_whitespace=not args.keep_whitespace,
            normalize_punctuation=not args.keep_punctuation,
            group_adjacent=not args.per_char,
            context_window=args.context,
        )

        summary = result.to_summary_dict()
        all_summaries.append(summary)

        voted_path = args.output_dir / f"{group.stem}_voted.txt"
        voted_path.write_text(result.voted_text, encoding="utf-8")

        json_path = args.output_dir / f"{group.stem}_disagreements.json"
        payload = {
            "summary": summary,
            "disagreements": [d.to_dict() for d in result.disagreements],
        }
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        md_path = args.output_dir / f"{group.stem}_disagreements.md"
        write_markdown_report(
            result, md_path, max_entries=args.max_md_disagreements
        )

        w_str = ", ".join(f"{m}={w:.2f}" for m, w in summary["model_weights"].items())
        print(
            f"  -> Blocks: {summary['total_blocks']} | "
            f"Unanimous: {summary['agreement_rate'] * 100:.2f}% | "
            f"Avg Conf: {summary['avg_confidence'] * 100:.2f}% | "
            f"Disagreements: {summary['total_disagreement_entries']:,} "
            f"(high: {summary['high_confidence_entries']:,}, low: {summary['low_confidence_entries']:,}) | "
            f"Weights: [{w_str}]"
        )

    (args.output_dir / "summary.json").write_text(
        json.dumps(all_summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_summary_markdown(all_summaries, args.output_dir / "summary.md")
    print(f"\nDone! Outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
