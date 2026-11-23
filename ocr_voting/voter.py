"""Step 4 & 5: Weighted column voting and confidence-score disagreement resolution."""

from __future__ import annotations

from dataclasses import dataclass

from aligner import align_blocks, align_to_columns
from normalizer import normalize_document_blocks


@dataclass
class Disagreement:
    """A location where the OCR models did not unanimously agree."""

    index: int
    block_index: int
    block_start: int
    block_end: int
    global_start: int
    global_end: int
    resolution: str  # "high_confidence" or "low_confidence"
    confidence: float  # winning confidence score in [0, 1] (minimum across span if grouped)
    margin: float  # confidence gap between top-1 and top-2 candidates
    voted_text: str
    votes: dict[str, str]
    weighted_votes: dict[str, float]
    candidate_confidences: dict[str, float]
    context_before: str
    context_after: str

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "block_index": self.block_index,
            "block_char_range": [self.block_start, self.block_end],
            "global_char_range": [self.global_start, self.global_end],
            "resolution": self.resolution,
            "confidence": round(self.confidence, 4),
            "margin": round(self.margin, 4),
            "voted_text": self.voted_text,
            "votes": self.votes,
            "weighted_votes": {k: round(v, 4) for k, v in self.weighted_votes.items()},
            "candidate_confidences": {
                k: round(v, 4) for k, v in self.candidate_confidences.items()
            },
            "context": {
                "before": self.context_before,
                "voted": self.voted_text,
                "after": self.context_after,
            },
        }


@dataclass
class DocumentVoteResult:
    """Complete voting output and disagreement breakdown for a single document."""

    doc_name: str
    models: list[str]
    model_weights: dict[str, float]
    confidence_threshold: float
    total_blocks: int
    total_columns: int
    unanimous_columns: int
    high_confidence_disagreements: int
    low_confidence_disagreements: int
    avg_confidence: float
    voted_text: str
    disagreements: list[Disagreement]

    @property
    def agreement_rate(self) -> float:
        return (
            self.unanimous_columns / self.total_columns
            if self.total_columns > 0
            else 1.0
        )

    def to_summary_dict(self) -> dict:
        return {
            "doc_name": self.doc_name,
            "models": self.models,
            "model_weights": {k: round(v, 4) for k, v in self.model_weights.items()},
            "confidence_threshold": self.confidence_threshold,
            "total_blocks": self.total_blocks,
            "total_columns": self.total_columns,
            "unanimous_columns": self.unanimous_columns,
            "high_confidence_disagreement_columns": self.high_confidence_disagreements,
            "low_confidence_disagreement_columns": self.low_confidence_disagreements,
            "agreement_rate": round(self.agreement_rate, 6),
            "avg_confidence": round(self.avg_confidence, 6),
            "total_disagreement_entries": len(self.disagreements),
            "high_confidence_entries": sum(
                1 for d in self.disagreements if d.resolution == "high_confidence"
            ),
            "low_confidence_entries": sum(
                1 for d in self.disagreements if d.resolution == "low_confidence"
            ),
        }


def estimate_model_weights(
    models: list[str],
    all_block_columns: list[list[list[str]]],
) -> dict[str, float]:
    """Automatically estimate reliability weights for each model from aligned columns.

    Each model's reliability score is its average pairwise character agreement with the
    other models across all aligned columns. This naturally down-weights models with
    severe omissions or noise while keeping comparably accurate models near-equal.
    """
    n_models = len(models)
    total_cols = sum(len(cols) for cols in all_block_columns)
    if total_cols == 0 or n_models < 2:
        return {m: 1.0 for m in models}

    pair_matches = [[0] * n_models for _ in range(n_models)]
    for columns in all_block_columns:
        for col in columns:
            for i in range(n_models):
                ci = col[i]
                for j in range(i + 1, n_models):
                    if ci == col[j]:
                        pair_matches[i][j] += 1
                        pair_matches[j][i] += 1

    raw_scores: dict[str, float] = {}
    for i, m in enumerate(models):
        avg_agree = sum(pair_matches[i][j] for j in range(n_models) if j != i) / (
            (n_models - 1) * total_cols
        )
        # Keep weights strictly positive; add small floor so every model retains a vote.
        raw_scores[m] = max(avg_agree, 0.05)

    mean_score = sum(raw_scores.values()) / n_models
    return {m: raw_scores[m] / mean_score for m in models}


def _score_column(
    col: list[str],
    models: list[str],
    weights: dict[str, float],
    total_weight: float,
) -> tuple[str, float, float, dict[str, float], dict[str, float]]:
    """Compute weighted votes and confidence scores for a single aligned column.

    Returns:
        ``(winner, confidence, margin, weighted_votes, candidate_confidences)``
    """
    weighted_votes: dict[str, float] = {}
    for m_idx, ch in enumerate(col):
        w = weights[models[m_idx]]
        weighted_votes[ch] = weighted_votes.get(ch, 0.0) + w

    # Sort candidates by descending weighted vote; deterministic tie-break prefers
    # non-empty character over gap, then lexicographical order.
    ranked = sorted(
        weighted_votes.items(),
        key=lambda kv: (-kv[1], kv[0] == "", kv[0]),
    )
    winner, top_w = ranked[0]
    second_w = ranked[1][1] if len(ranked) > 1 else 0.0

    confidence = top_w / total_weight if total_weight > 0 else 1.0
    margin = (top_w - second_w) / total_weight if total_weight > 0 else 1.0
    candidate_confidences = {
        ch: w / total_weight for ch, w in ranked
    }
    ordered_weighted = dict(ranked)
    return winner, confidence, margin, ordered_weighted, candidate_confidences


def vote_document(
    doc_name: str,
    model_texts_full: dict[str, str],
    model_weights: dict[str, float] | None = None,
    confidence_threshold: float = 0.5,
    ignore_whitespace: bool = True,
    normalize_punctuation: bool = True,
    group_adjacent: bool = True,
    context_window: int = 15,
) -> DocumentVoteResult:
    """Execute the full 5-step OCR voting pipeline on one document.

    1. Normalize the OCR outputs.
    2. Align character sequences using exact Levenshtein alignment.
    3. Group corresponding characters into columns.
    4. Calculate weighted votes for each column.
    5. Resolve disagreements using confidence scores.
    """
    models = sorted(model_texts_full.keys())

    # Step 1: Normalize
    model_blocks = {
        m: normalize_document_blocks(
            model_texts_full[m],
            ignore_whitespace=ignore_whitespace,
            normalize_punctuation=normalize_punctuation,
        )
        for m in models
    }

    # Step 2 & 3: Align blocks, then Levenshtein-align characters into columns
    aligned_blocks = align_blocks(model_blocks)
    all_block_columns: list[list[list[str]]] = []
    for block_map in aligned_blocks:
        _, columns = align_to_columns(block_map)
        all_block_columns.append(columns)

    # Step 4: Determine model weights (explicit or auto-estimated)
    if model_weights:
        weights = {m: float(model_weights.get(m, 1.0)) for m in models}
    else:
        weights = estimate_model_weights(models, all_block_columns)
    total_weight = sum(weights.values())

    total_columns = 0
    unanimous_columns = 0
    high_conf_cols = 0
    low_conf_cols = 0
    confidence_sum = 0.0

    voted_blocks: list[str] = []
    disagreements: list[Disagreement] = []
    global_offset = 0

    # Step 4 & 5: Calculate weighted votes and resolve disagreements via confidence scores
    for block_idx, columns in enumerate(all_block_columns, start=1):
        col_winners: list[str] = []
        col_confidences: list[float] = []
        col_margins: list[float] = []
        col_unanimous: list[bool] = []

        for col in columns:
            total_columns += 1
            winner, conf, margin, _, _ = _score_column(
                col, models, weights, total_weight
            )
            is_unanimous = len(set(col)) == 1
            confidence_sum += conf

            if is_unanimous:
                unanimous_columns += 1
            elif conf > confidence_threshold:
                high_conf_cols += 1
            else:
                low_conf_cols += 1

            col_winners.append(winner)
            col_confidences.append(conf)
            col_margins.append(margin)
            col_unanimous.append(is_unanimous)

        # Assemble consensus block text and track each column's start offset.
        block_chars: list[str] = []
        col_char_start: list[int] = []
        cur_pos = 0
        for ch in col_winners:
            col_char_start.append(cur_pos)
            if ch:
                block_chars.append(ch)
                cur_pos += len(ch)
        block_text = "".join(block_chars)
        voted_blocks.append(block_text)

        # Extract disagreements (either per-column or grouped contiguous spans).
        col_i = 0
        n_cols = len(columns)
        while col_i < n_cols:
            if col_unanimous[col_i]:
                col_i += 1
                continue

            span_start = col_i
            if group_adjacent:
                while col_i < n_cols and not col_unanimous[col_i]:
                    col_i += 1
            else:
                col_i += 1
            span_end = col_i

            span_voted = "".join(col_winners[span_start:span_end])
            block_start = col_char_start[span_start]
            block_end = block_start + len(span_voted)

            # Aggregate each model's vote across the span.
            votes: dict[str, str] = {}
            span_weighted: dict[str, float] = {}
            for m_idx, m_name in enumerate(models):
                m_text = "".join(
                    columns[c][m_idx] for c in range(span_start, span_end)
                )
                votes[m_name] = m_text
                span_weighted[m_text] = span_weighted.get(m_text, 0.0) + weights[m_name]

            span_weighted = dict(
                sorted(span_weighted.items(), key=lambda kv: (-kv[1], kv[0] == "", kv[0]))
            )
            span_confs = {
                cand: w / total_weight for cand, w in span_weighted.items()
            }

            min_conf = min(col_confidences[span_start:span_end])
            min_margin = min(col_margins[span_start:span_end])
            resolution = (
                "high_confidence" if min_conf > confidence_threshold else "low_confidence"
            )

            ctx_before = block_text[max(0, block_start - context_window) : block_start]
            ctx_after = block_text[block_end : block_end + context_window]

            disagreements.append(
                Disagreement(
                    index=len(disagreements) + 1,
                    block_index=block_idx,
                    block_start=block_start,
                    block_end=block_end,
                    global_start=global_offset + block_start,
                    global_end=global_offset + block_end,
                    resolution=resolution,
                    confidence=min_conf,
                    margin=min_margin,
                    voted_text=span_voted,
                    votes=votes,
                    weighted_votes=span_weighted,
                    candidate_confidences=span_confs,
                    context_before=ctx_before,
                    context_after=ctx_after,
                )
            )

        global_offset += len(block_text) + 2

    voted_text = "\n\n".join(voted_blocks)
    avg_confidence = confidence_sum / total_columns if total_columns > 0 else 1.0

    return DocumentVoteResult(
        doc_name=doc_name,
        models=models,
        model_weights=weights,
        confidence_threshold=confidence_threshold,
        total_blocks=len(aligned_blocks),
        total_columns=total_columns,
        unanimous_columns=unanimous_columns,
        high_confidence_disagreements=high_conf_cols,
        low_confidence_disagreements=low_conf_cols,
        avg_confidence=avg_confidence,
        voted_text=voted_text,
        disagreements=disagreements,
    )
