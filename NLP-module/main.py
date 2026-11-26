"""CLI entry point for running 3-way ensemble Han-Viet sentence alignment on paired folders."""
from __future__ import annotations

import argparse
import json
import os
import sys
import unicodedata
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from alignment_tool.EnsembleHanVietAligner import (  # noqa: E402
    EnsembleHanVietAligner,
    split_parallel_texts_by_lines,
    split_sentences,
    write_alignment_outputs,
    write_parallel_parts,
)


def _normalize_key(stem: str) -> str:
    return unicodedata.normalize("NFC", stem)


def _collect_files(folder: Path, extensions: set[str], recursive: bool) -> dict[str, Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Directory not found: {folder}")
    iterator = folder.rglob("*") if recursive else folder.iterdir()
    found: dict[str, Path] = {}
    for path in sorted(iterator):
        if not path.is_file():
            continue
        if path.suffix.lower() not in extensions:
            continue
        key = _normalize_key(path.stem)
        if key in found:
            raise ValueError(
                f"Duplicate filename stem {path.stem!r} in {folder}: "
                f"{found[key]} and {path}"
            )
        found[key] = path
    return found


def discover_pairs(
    han_dir: Path,
    vi_dir: Path,
    *,
    extensions: set[str],
    recursive: bool = True,
) -> tuple[list[tuple[str, Path, Path]], list[str], list[str]]:
    """Discover all (stem, han_path, vi_path) pairs sharing the same filename stem."""
    han_files = _collect_files(han_dir, extensions, recursive)
    vi_files = _collect_files(vi_dir, extensions, recursive)

    shared_keys = sorted(set(han_files) & set(vi_files))
    only_han = sorted(set(han_files) - set(vi_files))
    only_vi = sorted(set(vi_files) - set(han_files))

    pairs = [(han_files[k].stem, han_files[k], vi_files[k]) for k in shared_keys]
    return pairs, only_han, only_vi


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")


def _load_units(path: Path, language: str, split_mode: str) -> list[str]:
    text = _read_text(path)
    if split_mode == "lines":
        return [
            " ".join(line.split())
            for line in text.splitlines()
            if line.strip()
        ]
    return split_sentences(text, language)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Discover all same-named file pairs across a Han-Nom folder and a Vietnamese folder, "
            "split each pair into line-bounded text parts (each pair <= max_chars), run 3-way "
            "voting alignment (Bertalign + Vecalign + CLC API) on each part pair, and write "
            "merged <name>.tsv, <name>.xlsx, and <name>_note.xlsx to the target output directory."
        )
    )

    parser.add_argument(
        "--han-dir",
        type=Path,
        required=True,
        help="Folder containing Han-Nom input text files.",
    )
    parser.add_argument(
        "--vi-dir",
        type=Path,
        required=True,
        help="Folder containing Vietnamese input text files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./alignment_output"),
        help="Target folder where <name>.tsv, <name>.xlsx, and <name>_note.xlsx will be written.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=5000,
        help=(
            "Maximum combined characters (Han + Vi) per split part pair, cut strictly at "
            "newline boundaries (default: 5000)."
        ),
    )
    parser.add_argument(
        "--max-units",
        type=int,
        default=80,
        help=(
            "Maximum sentence/line units per side in each split part pair, matching CLC API's "
            "per-request limit (default: 80)."
        ),
    )
    parser.add_argument(
        "--extensions",
        type=str,
        default=".txt",
        help="Comma-separated file extensions to match (default: '.txt').",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only search the top-level of --han-dir and --vi-dir.",
    )
    parser.add_argument(
        "--split-mode",
        choices=("sentences", "lines"),
        default="sentences",
        help="How to split each part file into units: 'sentences' (default) or 'lines' (pre-segmented).",
    )

    # Engine toggles (all 3 enabled by default for 3-way voting)
    parser.add_argument("--disable-bertalign", action="store_true", help="Disable Bertalign engine.")
    parser.add_argument("--disable-vecalign", action="store_true", help="Disable Vecalign engine.")
    parser.add_argument("--disable-api", action="store_true", help="Disable CLC API engine.")

    # API options
    parser.add_argument(
        "--api-token",
        type=str,
        default=os.environ.get("CLC_API_TOKEN") or os.environ.get("HAN_NOM_API_TOKEN"),
        help="Optional Bearer token for CLC API (defaults to CLC_API_TOKEN / HAN_NOM_API_TOKEN).",
    )
    parser.add_argument(
        "--api-response-dir",
        type=Path,
        default=None,
        help=(
            "Optional folder of saved CLC API JSON responses for offline replay "
            "(<name>_part_001.json or <name>.json for single-part files)."
        ),
    )
    parser.add_argument(
        "--api-interval",
        type=float,
        default=3.0,
        help="Minimum interval in seconds between CLC API requests (default: 3.0).",
    )

    # Vecalign options
    parser.add_argument(
        "--vec-model",
        type=str,
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="SentenceTransformer model for Vecalign.",
    )
    parser.add_argument("--overlap-n", type=int, default=4, help="Vecalign overlap size (default: 4).")
    parser.add_argument("--batch-size", type=int, default=16, help="Vecalign embedding batch size (default: 16).")

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover paired files and preview line-level part splits without running engines.",
    )
    return parser.parse_args()


def _resolve_api_response(
    api_response_dir: Path | None,
    stem: str,
    part_idx: int,
    total_parts: int,
) -> Path | None:
    if api_response_dir is None:
        return None
    part_candidate = api_response_dir / f"{stem}_part_{part_idx:03d}.json"
    if part_candidate.is_file():
        return part_candidate
    if total_parts == 1:
        single_candidate = api_response_dir / f"{stem}.json"
        if single_candidate.is_file():
            return single_candidate
    raise FileNotFoundError(f"Missing API replay file: {part_candidate}")


def main() -> None:
    args = parse_args()
    extensions = {
        ext.strip().lower() if ext.strip().startswith(".") else f".{ext.strip().lower()}"
        for ext in args.extensions.split(",")
        if ext.strip()
    }

    pairs, only_han, only_vi = discover_pairs(
        args.han_dir,
        args.vi_dir,
        extensions=extensions,
        recursive=not args.no_recursive,
    )

    print(f"Discovered {len(pairs)} matched pair(s):")
    for stem, han_path, vi_path in pairs:
        print(f"  - {stem}: {han_path}  <->  {vi_path}")
    if only_han:
        print(f"[WARNING] Unmatched in --han-dir ({len(only_han)}): {', '.join(only_han)}")
    if only_vi:
        print(f"[WARNING] Unmatched in --vi-dir ({len(only_vi)}): {', '.join(only_vi)}")

    if not pairs:
        print("No matching file pairs found.")
        return

    if args.dry_run:
        for stem, han_path, vi_path in pairs:
            parts = split_parallel_texts_by_lines(
                _read_text(han_path),
                _read_text(vi_path),
                max_chars=args.max_chars,
                max_units=args.max_units,
                split_mode=args.split_mode,
            )
            print(
                f"  [DRY-RUN] {stem}: {len(parts)} part(s) "
                f"(max_chars={args.max_chars}, max_units={args.max_units})"
            )
            for idx, (h_part, v_part) in enumerate(parts, start=1):
                combined = len(h_part) + len(v_part)
                print(
                    f"    part {idx:03d}: Han={len(h_part)} chars, "
                    f"Vi={len(v_part)} chars, total={combined} chars"
                )
        print("Dry run complete. No alignment engines were executed.")
        return

    aligner = EnsembleHanVietAligner(
        vec_model=args.vec_model,
        overlap_n=args.overlap_n,
        batch_size=args.batch_size,
        api_token=args.api_token or None,
        api_interval_s=args.api_interval,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    failures: list[tuple[str, str]] = []

    for index, (stem, han_path, vi_path) in enumerate(pairs, start=1):
        print(f"\n[{index}/{len(pairs)}] Aligning: {stem}")
        try:
            parts_dir = args.output_dir / "_parts" / stem
            part_files = write_parallel_parts(
                _read_text(han_path),
                _read_text(vi_path),
                parts_dir,
                stem=stem,
                max_chars=args.max_chars,
                max_units=args.max_units,
                split_mode=args.split_mode,
            )
            print(f"  Split into {len(part_files)} part file pair(s) in {parts_dir}")

            part_work_root = args.output_dir / "_details" / stem / "parts"
            part_work_root.mkdir(parents=True, exist_ok=True)

            all_rows: list[dict] = []
            all_disagreements: list[dict] = []
            part_summaries: list[dict] = []
            total_accepted = 0
            all_parts_ok = True
            next_disagree_idx = 1

            for part_idx, (h_part_path, v_part_path) in enumerate(part_files, start=1):
                h_text = _read_text(h_part_path)
                v_text = _read_text(v_part_path)
                han_units = _load_units(h_part_path, "han", args.split_mode)
                vi_units = _load_units(v_part_path, "vi", args.split_mode)
                print(
                    f"  [Part {part_idx}/{len(part_files)}] "
                    f"chars={len(h_text) + len(v_text)} (Han={len(h_text)}, Vi={len(v_text)}), "
                    f"units=(Han={len(han_units)}, Vi={len(vi_units)})"
                )

                api_resp = _resolve_api_response(
                    args.api_response_dir, stem, part_idx, len(part_files)
                )

                part_result = aligner.align_section(
                    han_units,
                    vi_units,
                    out_dir=part_work_root / f"part_{part_idx:03d}",
                    enable_bertalign=not args.disable_bertalign,
                    enable_vecalign=not args.disable_vecalign,
                    enable_api=not args.disable_api,
                    api_response=api_resp,
                    output_name=f"{stem}_part_{part_idx:03d}",
                    disagreement_start_index=next_disagree_idx,
                )

                part_ok = all(part_result.methods_ok.values())
                all_parts_ok = all_parts_ok and part_ok
                total_accepted += len(part_result.accepted)
                next_disagree_idx += len(part_result.disagreements)

                for row in part_result.rows:
                    all_rows.append({**row, "part": part_idx})
                for item in part_result.disagreements:
                    all_disagreements.append({**item, "part": part_idx})

                part_summaries.append({
                    "part": part_idx,
                    "han_file": str(h_part_path),
                    "vi_file": str(v_part_path),
                    "han_chars": len(h_text),
                    "vi_chars": len(v_text),
                    "combined_chars": len(h_text) + len(v_text),
                    "methods_ok": part_result.methods_ok,
                    "method_errors": part_result.method_errors,
                    "stats": part_result.stats,
                })

            tsv_path, xlsx_path, notes_path = write_alignment_outputs(
                args.output_dir,
                all_rows,
                all_disagreements,
                all_ok=all_parts_ok,
                output_name=stem,
            )

            summary_path = args.output_dir / "_details" / stem / "summary.json"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(
                json.dumps(
                    {
                        "stem": stem,
                        "max_chars": args.max_chars,
                        "total_parts": len(part_files),
                        "all_parts_ok": all_parts_ok,
                        "total_accepted_pairs": total_accepted,
                        "total_disagreement_blocks": len(all_disagreements),
                        "parts": part_summaries,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            print(f"  Status:              {'VOTING_READY_REVIEW' if all_parts_ok else 'INCOMPLETE_METHODS'}")
            print(f"  Accepted (>=2/3):    {total_accepted}")
            print(f"  Disagreement blocks: {len(all_disagreements)}")
            print(f"  Result files:        {tsv_path}, {xlsx_path}")
            print(f"  Note file:           {notes_path}")
        except Exception as exc:
            failures.append((stem, f"{type(exc).__name__}: {exc}"))
            print(f"  [ERROR] {stem}: {type(exc).__name__}: {exc}")

    print("\n=== BATCH SUMMARY ===")
    print(f"Processed: {len(pairs) - len(failures)}/{len(pairs)} pair(s)")
    print(f"Output directory: {args.output_dir.resolve()}")
    if failures:
        print("Failures:")
        for stem, err in failures:
            print(f"  - {stem}: {err}")


if __name__ == "__main__":
    main()
