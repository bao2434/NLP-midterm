# OCR Voting Tool (Levenshtein Alignment + Weighted Confidence Voting)

Aligns OCR results from multiple models for the same source documents without shifting when words/characters are missing, computes weighted column votes, and resolves disagreements using confidence scores.

## 5-Step Pipeline

1. **Normalize** ([normalizer.py](normalizer.py)) — Unicode NFKC normalization, invisible character removal, punctuation unification, and intra-block whitespace normalization.
2. **Levenshtein Alignment** ([aligner.py](aligner.py)) — Exact minimum-edit-distance Levenshtein alignment (accelerated with Ukkonen's exact banded DP) so missing words or characters insert explicit gaps `""` and never shift subsequent text.
3. **Column Grouping** ([aligner.py](aligner.py)) — Groups corresponding characters across all 3 models into aligned columns via Center-Star Multi-Sequence Alignment.
4. **Weighted Column Voting** ([voter.py](voter.py)) — Computes weighted votes and normalized confidence scores (`weight_for_candidate / total_weight`) for each column. Model weights are either user-supplied (`--weights`) or automatically estimated from pairwise agreement.
5. **Confidence-Based Disagreement Resolution** ([voter.py](voter.py)) — Selects the highest-confidence candidate per column, flags each disagreement as `high_confidence` or `low_confidence` (against `--confidence-threshold`), and records exact block/character positions, context, confidence score, and each model's vote.

## Usage

```powershell
# Run on all matched files in res/ (auto-estimated model weights)
python ocr_voting/main.py

# Run on a single document
python ocr_voting/main.py --file q5

# Supply explicit model weights
python ocr_voting/main.py --weights ocr_hunyuang=1.2 ocr_paddle=1.1 ocr_api=0.8

# Report every single disagreeing character column separately (ungrouped)
python ocr_voting/main.py --file q5 --per-char
```

## Outputs (`ocr_voting/output/`)

For each document `<name>`:
- `<name>_voted.txt` — Final consensus OCR text.
- `<name>_disagreements.json` — Complete JSON with summary, model weights, and every disagreement (location, `resolution`, `confidence`, `margin`, `voted_text`, per-model `votes`, `weighted_votes`, `candidate_confidences`, and `context`).
- `<name>_disagreements.md` — Human-readable Markdown table of disagreements.

Overall summaries are saved to `summary.json` and `summary.md`.

## Conflict Resolution (`resolver.py`)

Reads the generated `*_disagreements.json` (and `*_voted.txt`) files and resolves every conflict with explicit 3-tier rules:
1. **All 3 models agree (`all_agree`)** — accept the result (`confidence = 1.0`).
2. **Two models agree (`two_agree`)** — calculate their combined confidence `(w1 + w2) / (w1 + w2 + w3)` and compare against the dissenting model and `--pair-threshold`.
3. **All 3 models disagree (`all_disagree`)** — resolve using confidence-weighted voting and flag `needs_review` if below `--review-threshold`.

```powershell
# Resolve all disagreement files using stored model weights
python ocr_voting/resolver.py

# Resolve with custom per-model confidence scores
python ocr_voting/resolver.py --model-confidences ocr_hunyuang=0.92 ocr_paddle=0.88 ocr_api=0.75

# Resolve a single document and show only needs_review rows in Markdown
python ocr_voting/resolver.py --file q5 --only-review-md
```

Outputs are written to `ocr_voting/resolved_output/` (`*_resolved.txt`, `*_resolved_conflicts.json`, `*_resolved_conflicts.md`, and `resolution_summary.md`).
