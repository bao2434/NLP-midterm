"""Read OCR disagreement JSON files and resolve conflicts using explicit 3-tier rules:

1. If all three models agree -> accept the result.
2. If two models agree       -> calculate their combined confidence and decide.
3. If all three disagree     -> resolve using confidence-weighted voting.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ConflictResolution:
    """Resolved decision for a single disagreement entry."""

    index: int
    block_index: int
    block_char_range: list[int]
    global_char_range: list[int]
    rule_applied: str  # "all_agree" | "two_agree" | "all_disagree"
    resolved_text: str
    previous_voted_text: str
    changed_from_voted: bool
    winning_confidence: float
    combined_pair_confidence: float | None
    needs_review: bool
    votes: dict[str, str]
    candidate_scores: dict[str, float]
    context_before: str
    context_after: str

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "block_index": self.block_index,
            "block_char_range": self.block_char_range,
            "global_char_range": self.global_char_range,
            "rule_applied": self.rule_applied,
            "resolved_text": self.resolved_text,
            "previous_voted_text": self.previous_voted_text,
            "changed_from_voted": self.changed_from_voted,
            "winning_confidence": round(self.winning_confidence, 4),
            "combined_pair_confidence": (
                round(self.combined_pair_confidence, 4)
                if self.combined_pair_confidence is not None
                else None
            ),
            "needs_review": self.needs_review,
            "votes": self.votes,
            "candidate_scores": {
                k: round(v, 4) for k, v in self.candidate_scores.items()
            },
            "context": {
                "before": self.context_before,
                "resolved": self.resolved_text,
                "after": self.context_after,
            },
        }


def resolve_single_conflict(
    votes: dict[str, str],
    model_confidences: dict[str, float],
    pair_confidence_threshold: float = 0.5,
    review_threshold: float = 0.5,
) -> tuple[str, str, float, float | None, bool, dict[str, float]]:
    """Apply the 3-tier conflict resolution rules to a single set of model votes.

    Args:
        votes: Mapping ``model_name -> text`` (typically 3 models).
        model_confidences: Positive confidence/reliability score per model.
        pair_confidence_threshold: Minimum combined confidence required for an agreeing
            pair to win outright without flagging for review.
        review_threshold: Winning confidence below this value sets ``needs_review=True``.

    Returns:
        ``(rule_applied, resolved_text, winning_confidence, combined_pair_confidence,
           needs_review, candidate_scores)``
    """
    # Group models by the text they produced.
    groups: dict[str, list[str]] = {}
    for model, text in votes.items():
        groups.setdefault(text, []).append(model)

    total_conf = sum(model_confidences[m] for m in votes)
    if total_conf <= 0:
        total_conf = float(len(votes))
        weights = {m: 1.0 for m in votes}
    else:
        weights = {m: model_confidences[m] for m in votes}

    # Normalized confidence score per distinct candidate text.
    candidate_scores = {
        text: sum(weights[m] for m in ms) / total_conf
        for text, ms in groups.items()
    }
    ranked = sorted(
        candidate_scores.items(),
        key=lambda kv: (-kv[1], kv[0] == "", kv[0]),
    )

    num_distinct = len(groups)

    # Rule 1: All models agree -> accept immediately.
    if num_distinct == 1:
        winner = ranked[0][0]
        return "all_agree", winner, 1.0, None, False, dict(ranked)

    # Rule 2: Two models agree -> calculate their combined confidence.
    pair_candidates = [text for text, ms in groups.items() if len(ms) >= 2]
    if pair_candidates:
        pair_text = max(pair_candidates, key=lambda t: candidate_scores[t])
        combined_conf = candidate_scores[pair_text]

        # Compare the agreeing pair's combined confidence against the dissenting model.
        top_text, top_conf = ranked[0]
        if pair_text == top_text:
            needs_review = combined_conf < pair_confidence_threshold
            return (
                "two_agree",
                pair_text,
                combined_conf,
                combined_conf,
                needs_review,
                dict(ranked),
            )
        else:
            # Rare case where the single dissenting model has higher weight than both
            # agreeing models combined: pick highest confidence and flag for review.
            return (
                "two_agree",
                top_text,
                top_conf,
                combined_conf,
                True,
                dict(ranked),
            )

    # Rule 3: All three disagree -> confidence-weighted voting.
    winner, win_conf = ranked[0]
    needs_review = win_conf < review_threshold
    return (
        "all_disagree",
        winner,
        win_conf,
        None,
        needs_review,
        dict(ranked),
    )


def _apply_resolutions_to_voted_text(
    voted_text: str,
    resolutions: list[ConflictResolution],
) -> str:
    """Apply resolved replacements onto the voted text blocks safely (reverse offset order)."""
    blocks = voted_text.split("\n\n")
    by_block: dict[int, list[ConflictResolution]] = {}
    for r in resolutions:
        if r.changed_from_voted:
            by_block.setdefault(r.block_index, []).append(r)

    for block_idx, edits in by_block.items():
        if not (1 <= block_idx <= len(blocks)):
            continue
        text = blocks[block_idx - 1]
        # Apply from right to left so earlier character offsets remain valid.
        for edit in sorted(edits, key=lambda e: e.block_char_range[0], reverse=True):
            start, end = edit.block_char_range
            text = text[:start] + edit.resolved_text + text[end:]
        blocks[block_idx - 1] = text

    return "\n\n".join(blocks)


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


def write_resolution_markdown(
    doc_name: str,
    models: list[str],
    model_confidences: dict[str, float],
    resolutions: list[ConflictResolution],
    output_path: Path,
    only_review: bool = False,
    max_entries: int = 0,
) -> None:
    """Write a human-readable Markdown report of resolved conflicts."""
    all_agree_n = sum(1 for r in resolutions if r.rule_applied == "all_agree")
    two_agree_n = sum(1 for r in resolutions if r.rule_applied == "two_agree")
    all_disagree_n = sum(1 for r in resolutions if r.rule_applied == "all_disagree")
    changed_n = sum(1 for r in resolutions if r.changed_from_voted)
    review_n = sum(1 for r in resolutions if r.needs_review)
    conf_str = ", ".join(f"{m}={model_confidences[m]:.3f}" for m in models)

    lines = [
        f"# Conflict Resolution Report: {doc_name}",
        "",
        "## Summary",
        "",
        f"- **Model confidence weights**: {conf_str}",
        f"- **Total conflicts processed**: {len(resolutions):,}",
        f"- **Rule 1 (`all_agree`)**: {all_agree_n:,}",
        f"- **Rule 2 (`two_agree` — combined confidence)**: {two_agree_n:,}",
        f"- **Rule 3 (`all_disagree` — confidence-weighted voting)**: {all_disagree_n:,}",
        f"- **Changed from initial voted text**: {changed_n:,}",
        f"- **Flagged for manual review (`needs_review`)**: {review_n:,}",
        "",
        "## Resolved Conflicts",
        "",
    ]

    rows = [r for r in resolutions if r.needs_review] if only_review else resolutions
    if max_entries > 0:
        rows = rows[:max_entries]

    if not rows:
        lines.append("No conflicts to display.")
    else:
        model_headers = " | ".join(models)
        sep_headers = " | ".join(["---"] * len(models))
        lines.append(
            f"| # | Block | Pos | Rule | Win Conf | Pair Conf | Review | Context (`【resolved】`) | Resolved | {model_headers} |"
        )
        lines.append(
            f"| --- | --- | --- | --- | ---: | ---: | --- | --- | --- | {sep_headers} |"
        )

        for r in rows:
            marker = r.resolved_text if r.resolved_text else "∅"
            ctx = f"{r.context_before}【{marker}】{r.context_after}"
            ctx_escaped = ctx.replace("|", "\\|").replace("\n", "\\n")
            pair_str = (
                f"{r.combined_pair_confidence:.2f}"
                if r.combined_pair_confidence is not None
                else "-"
            )
            review_str = "YES" if r.needs_review else "no"
            model_cells = " | ".join(_fmt_cell(r.votes[m]) for m in models)
            lines.append(
                f"| {r.index} | {r.block_index} | {r.block_char_range[0]}-{r.block_char_range[1]} | "
                f"`{r.rule_applied}` | {r.winning_confidence:.2f} | {pair_str} | "
                f"{review_str} | {ctx_escaped} | {_fmt_cell(r.resolved_text)} | {model_cells} |"
            )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def resolve_disagreement_file(
    disagreement_json_path: Path,
    output_dir: Path,
    explicit_confidences: dict[str, float] | None = None,
    pair_confidence_threshold: float = 0.5,
    review_threshold: float = 0.5,
    only_review_md: bool = False,
    max_md_entries: int = 0,
) -> dict:
    """Read one ``*_disagreements.json`` file, resolve all conflicts, and save outputs."""
    data = json.loads(disagreement_json_path.read_text(encoding="utf-8"))
    summary = data["summary"]
    disagreements = data["disagreements"]

    doc_name: str = summary["doc_name"]
    models: list[str] = summary["models"]
    stored_weights: dict[str, float] = summary.get(
        "model_weights", {m: 1.0 for m in models}
    )

    if explicit_confidences:
        model_confidences = {
            m: float(explicit_confidences.get(m, stored_weights.get(m, 1.0)))
            for m in models
        }
    else:
        model_confidences = {m: float(stored_weights.get(m, 1.0)) for m in models}

    resolutions: list[ConflictResolution] = []
    for entry in disagreements:
        votes: dict[str, str] = entry["votes"]
        rule, resolved, win_conf, pair_conf, needs_review, scores = (
            resolve_single_conflict(
                votes=votes,
                model_confidences=model_confidences,
                pair_confidence_threshold=pair_confidence_threshold,
                review_threshold=review_threshold,
            )
        )
        prev_voted = entry["voted_text"]
        ctx = entry.get("context", {})
        resolutions.append(
            ConflictResolution(
                index=entry["index"],
                block_index=entry["block_index"],
                block_char_range=entry["block_char_range"],
                global_char_range=entry["global_char_range"],
                rule_applied=rule,
                resolved_text=resolved,
                previous_voted_text=prev_voted,
                changed_from_voted=(resolved != prev_voted),
                winning_confidence=win_conf,
                combined_pair_confidence=pair_conf,
                needs_review=needs_review,
                votes=votes,
                candidate_scores=scores,
                context_before=ctx.get("before", ""),
                context_after=ctx.get("after", ""),
            )
        )

    stem = disagreement_json_path.name.removesuffix("_disagreements.json")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Reconstruct final resolved text if the sibling *_voted.txt exists.
    voted_txt_path = disagreement_json_path.with_name(f"{stem}_voted.txt")
    resolved_txt_path = output_dir / f"{stem}_resolved.txt"
    if voted_txt_path.is_file():
        voted_text = voted_txt_path.read_text(encoding="utf-8")
        resolved_text = _apply_resolutions_to_voted_text(voted_text, resolutions)
        resolved_txt_path.write_text(resolved_text, encoding="utf-8")

    res_summary = {
        "doc_name": doc_name,
        "models": models,
        "model_confidences": {m: round(model_confidences[m], 4) for m in models},
        "total_conflicts": len(resolutions),
        "all_agree": sum(1 for r in resolutions if r.rule_applied == "all_agree"),
        "two_agree": sum(1 for r in resolutions if r.rule_applied == "two_agree"),
        "all_disagree": sum(1 for r in resolutions if r.rule_applied == "all_disagree"),
        "changed_from_voted": sum(1 for r in resolutions if r.changed_from_voted),
        "needs_review": sum(1 for r in resolutions if r.needs_review),
    }

    # Save resolved conflicts JSON.
    json_out = output_dir / f"{stem}_resolved_conflicts.json"
    json_out.write_text(
        json.dumps(
            {
                "summary": res_summary,
                "resolutions": [r.to_dict() for r in resolutions],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # Save resolved conflicts Markdown.
    md_out = output_dir / f"{stem}_resolved_conflicts.md"
    write_resolution_markdown(
        doc_name=doc_name,
        models=models,
        model_confidences=model_confidences,
        resolutions=resolutions,
        output_path=md_out,
        only_review=only_review_md,
        max_entries=max_md_entries,
    )

    return res_summary


def _parse_confidences(raw: list[str] | None) -> dict[str, float] | None:
    if not raw:
        return None
    out: dict[str, float] = {}
    for item in raw:
        if "=" not in item:
            raise ValueError(
                f"Invalid --model-confidences item '{item}'. Expected model=score (e.g. ocr_hunyuang=0.9)"
            )
        k, v = item.split("=", 1)
        val = float(v)
        if val <= 0:
            raise ValueError(f"Confidence for '{k}' must be > 0, got {val}")
        out[k.strip()] = val
    return out


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_input = script_dir / "output"
    default_output = script_dir / "resolved_output"

    parser = argparse.ArgumentParser(
        description="Read disagreement JSON files and resolve conflicts via 3-tier rules."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=default_input,
        help=f"Directory containing *_disagreements.json files (default: {default_input})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help=f"Directory to write resolved texts and reports (default: {default_output})",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Optional substring filter to process only matching disagreement file(s).",
    )
    parser.add_argument(
        "--model-confidences",
        nargs="+",
        default=None,
        help="Optional explicit model confidence scores, e.g. ocr_hunyuang=0.92 ocr_paddle=0.88 ocr_api=0.75. "
        "Defaults to the model weights stored in each disagreement JSON.",
    )
    parser.add_argument(
        "--pair-threshold",
        type=float,
        default=0.5,
        help="Minimum combined confidence for 2-agreeing models (default: 0.5).",
    )
    parser.add_argument(
        "--review-threshold",
        type=float,
        default=0.5,
        help="Winning confidence below this flags a conflict as needs_review (default: 0.5).",
    )
    parser.add_argument(
        "--only-review-md",
        action="store_true",
        help="Include only conflicts flagged needs_review in the Markdown report.",
    )
    parser.add_argument(
        "--max-md-entries",
        type=int,
        default=0,
        help="Maximum rows in Markdown report (0 = all; JSON always has all).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    explicit_confs = _parse_confidences(args.model_confidences)

    json_files = sorted(args.input_dir.glob("*_disagreements.json"))
    if args.file:
        filt = args.file.lower()
        json_files = [p for p in json_files if filt in p.name.lower()]

    if not json_files:
        print(f"No *_disagreements.json files found in {args.input_dir}")
        return

    print(f"Resolving conflicts for {len(json_files)} file(s) from {args.input_dir}\n")
    summaries: list[dict] = []

    for jp in json_files:
        s = resolve_disagreement_file(
            disagreement_json_path=jp,
            output_dir=args.output_dir,
            explicit_confidences=explicit_confs,
            pair_confidence_threshold=args.pair_threshold,
            review_threshold=args.review_threshold,
            only_review_md=args.only_review_md,
            max_md_entries=args.max_md_entries,
        )
        summaries.append(s)
        print(
            f"Resolved: {s['doc_name']}\n"
            f"  -> Conflicts: {s['total_conflicts']:,} | "
            f"two_agree: {s['two_agree']:,} | "
            f"all_disagree: {s['all_disagree']:,} | "
            f"changed: {s['changed_from_voted']:,} | "
            f"needs_review: {s['needs_review']:,}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "resolution_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    md_lines = [
        "# Conflict Resolution Overall Summary",
        "",
        "| Document | Total Conflicts | `all_agree` | `two_agree` | `all_disagree` | Changed | Needs Review |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for s in summaries:
        md_lines.append(
            f"| {s['doc_name']} | {s['total_conflicts']:,} | {s['all_agree']:,} | "
            f"{s['two_agree']:,} | {s['all_disagree']:,} | {s['changed_from_voted']:,} | "
            f"{s['needs_review']:,} |"
        )
    (args.output_dir / "resolution_summary.md").write_text(
        "\n".join(md_lines) + "\n", encoding="utf-8"
    )
    print(f"\nDone! Resolved outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
