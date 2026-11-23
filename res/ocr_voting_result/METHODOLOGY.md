# Multi-Model OCR Voting & Conflict Resolution Methodology

This document explains the end-to-end process used in `ocr_voting/` to align OCR outputs from three different models (`ocr_api`, `ocr_hunyuang`, `ocr_paddle`), perform weighted character-level voting, identify disagreements, and resolve conflicts to produce fixed output files without overwriting originals.

---

## 1. Motivation & Core Challenges

Comparing OCR outputs from different models on the same scanned books presents four main challenges:

1. **Inconsistent filenames**: Filenames may differ in Unicode normalization (NFC vs. NFD in Vietnamese titles) or trailing underscores (e.g., `..._q5.txt` vs. `..._q5_.txt`).
2. **Different line wrapping**: Each OCR engine breaks lines at different positions within a page, so line-by-line comparison fails immediately.
3. **Missing words causing text shift**: If one model drops a word or phrase, direct character-index comparison shifts the entire remainder of the text out of alignment.
4. **Unequal model reliability**: On certain documents (e.g., `phuong-te-hoc...`), one model may only capture partial fragments of each page while the other two capture full pages. Unweighted voting cannot distinguish reliable models from noisy ones during 3-way ties.

To solve these issues, the system is organized into **two phases**:
- **Phase 1 (`main.py`)**: Normalize → Levenshtein Align → Group into Columns → Weighted Column Voting → Export Voted Text & Disagreements.
- **Phase 2 (`resolver.py`)**: Read Disagreements → Apply 3-Tier Conflict Resolution Rules → Write Fixed Files to a Separate Folder.

---

## 2. Phase 1: Alignment, Weighted Voting & Disagreement Detection

### Step 0: File Discovery & Matching (`file_matcher.py`)
Subfolders under `res/` are treated as distinct OCR models. Filenames are normalized via:
- Unicode **NFC** normalization,
- Stripping trailing underscores (`_`) before `.txt`.

Files sharing the same normalized key are grouped together as outputs of the same source document.

### Step 1: Normalization (`normalizer.py`)
Each raw OCR file is split into **page/paragraph blocks** separated by blank lines (`\r?\n\s*\r?\n`). Within each block:
1. Apply Unicode **NFKC** normalization (standardizes full-width/half-width and compatibility characters).
2. Remove invisible/zero-width control characters (`\u200b`–`\u200f`, `\ufeff`, etc.).
3. Unify common punctuation variants (e.g., `,` → `，`, `(` → `（`, `:` → `：`).
4. Remove spurious intra-block whitespace and line breaks so line-wrapping differences do not create false disagreements.

### Step 2: Block & Exact Levenshtein Character Alignment (`aligner.py`)

#### 2a. Block-Level Alignment
- If all three models produce the exact same number of blocks (true for 5 of the 7 documents), blocks are paired 1-to-1.
- If a model missed or split a page (e.g., `q8` has `112 / 112 / 111` blocks), **Needleman-Wunsch global alignment** using character-bigram Jaccard similarity aligns the block sequences so pages never drift out of sync.

#### 2b. Character-Level Exact Levenshtein Alignment
Within each aligned block, pairs of character sequences $(A, B)$ are aligned using **exact minimum-edit-distance Levenshtein dynamic programming** (match cost `0`; substitution, insertion, and deletion cost `1`), accelerated with **Ukkonen's exact banded DP** (which guarantees mathematically optimal global edit distance).

Because Levenshtein alignment inserts explicit gap tokens (`""`) wherever characters are omitted or inserted, missing words never shift subsequent text:

```text
Model A: 夫 汗 者 血 之 異 名 內 則 為 津 液
Model B: 夫 汗 者  -  - 異 名 內 則 為 津 液
```

### Step 3: Grouping Characters into Columns (`aligner.py`)
To align all three models simultaneously, we use **Center-Star Multi-Sequence Alignment (MSA)**:
1. Select the **pivot (center) model** with the highest total bigram similarity to the other two models (automatically avoiding severely truncated outputs as the pivot).
2. Run exact Levenshtein alignment between the pivot and each other model.
3. Map matched/substituted characters onto pivot positions and collect insertions between pivot characters into inter-pivot slots (recursively aligning multi-model insertions in the same slot).
4. Emit a sequence of **aligned columns**, where column $j$ contains one character (or gap `""`) per model:

```text
Column:      0   1   2   3   4   5
ocr_api:     新  刋  南  桑  神  效
ocr_hunyuang:新  判  南  桑  神  效
ocr_paddle:  新  刊  南  棄  神  效
```

### Step 4: Weighted Column Voting (`voter.py`)
Each model $m$ is assigned a positive reliability weight $w_m$:
- **Automatic weighting (default)**: Computed from each model's average pairwise character agreement with the other models across all aligned columns in the document, normalized to mean `1.0`. This naturally down-weights models with heavy omissions (e.g., `ocr_api=0.17` vs. `ocr_hunyuang=1.42, ocr_paddle=1.42` on `phuong-te-hoc...`) while keeping comparably accurate models near `1.0`.
- **Manual weighting (`--weights`)**: Users can override weights explicitly (e.g., `--weights ocr_hunyuang=1.2 ocr_paddle=1.1 ocr_api=0.8`).

For each aligned column $j$, let $W = \sum_m w_m$. Each distinct candidate $c$ (character or `""`) receives:

$$V(c) = \sum_{m : \text{col}_j[m] = c} w_m, \qquad \text{conf}(c) = \frac{V(c)}{W} \in [0, 1]$$

The column winner is $c^* = \arg\max_c \text{conf}(c)$, and the confidence margin is $\text{conf}(c_1) - \text{conf}(c_2)$.

### Step 5: Disagreement Recording (`voter.py`, `main.py`)
Columns where models do not unanimously agree are recorded (grouped into contiguous spans by default, or per-column with `--per-char`) with:
- Exact location: `block_index`, `block_char_range`, `global_char_range`
- Surrounding consensus context (`context_before`, `voted`, `context_after`)
- Per-model votes (`votes`), `weighted_votes`, `candidate_confidences`, `confidence`, and `margin`

Outputs are written to `ocr_voting/output/`:
- `<doc>_voted.txt`
- `<doc>_disagreements.json`
- `<doc>_disagreements.md`
- `summary.json` / `summary.md`

---

## 3. Phase 2: 3-Tier Conflict Resolution & Fixed File Output (`resolver.py`)

`resolver.py` reads `*_disagreements.json` and `*_voted.txt`, applies explicit 3-tier conflict resolution rules to every disagreement entry, and writes the **fixed files** to a separate directory (`ocr_voting/resolved_output/`) so original files are never overwritten.

### The 3 Resolution Rules

Let $w_1, w_2, w_3$ be the confidence weights of the three models (loaded from `summary.model_weights` in the disagreement JSON or supplied via `--model-confidences`), with total weight $W = w_1 + w_2 + w_3$.

1. **Rule 1 — All Three Models Agree (`all_agree`)**
   - Condition: Only 1 distinct text across all 3 models.
   - Action: Accept the text directly with `winning_confidence = 1.0` and `needs_review = False`.

2. **Rule 2 — Two Models Agree (`two_agree`)**
   - Condition: Two models $(m_a, m_b)$ produce the same text $t_{\text{pair}}$ and the third model $m_c$ produces $t_{\text{single}} \neq t_{\text{pair}}$.
   - Action: Calculate the **combined confidence** of the two agreeing models:
     $$C_{\text{pair}} = \frac{w_a + w_b}{W}, \qquad C_{\text{single}} = \frac{w_c}{W}$$
     - If $C_{\text{pair}} \ge C_{\text{single}}$: accept $t_{\text{pair}}$. Flag `needs_review = True` only if $C_{\text{pair}} < \text{pair\_threshold}$ (default `0.5`).
     - If $C_{\text{single}} > C_{\text{pair}}$ (rare case where a single high-reliability model outweighs two weak models combined): select $t_{\text{single}}$ and set `needs_review = True`.

3. **Rule 3 — All Three Models Disagree (`all_disagree`)**
   - Condition: All 3 models produce distinct texts ($t_1, t_2, t_3$).
   - Action: Perform **confidence-weighted voting**:
     $$S(t_m) = \frac{w_m}{W}, \qquad t^* = \arg\max_m S(t_m)$$
     Select $t^*$ and flag `needs_review = True` if $S(t^*) < \text{review\_threshold}$ (default `0.5`).

### Non-Destructive File Generation
To produce the fixed text `<doc>_resolved.txt`:
1. Load the baseline `<doc>_voted.txt` split into blocks (`\n\n`).
2. For each block, apply all changed conflict resolutions in **reverse character-offset order** (right-to-left) so earlier character indices remain valid.
3. Write the fixed files to `ocr_voting/resolved_output/`:
   - `<doc>_resolved.txt` — Final fixed document text.
   - `<doc>_resolved_conflicts.json` — Complete resolution log with rules applied, combined/weighted confidence scores, and review flags.
   - `<doc>_resolved_conflicts.md` — Human-readable resolution table.
   - `resolution_summary.json` / `resolution_summary.md` — Overall summary across all documents.

---

## 4. Worked Example (from `Tân_san...q5.txt`)

Auto-estimated model weights on `q5`: `ocr_api = 0.973`, `ocr_hunyuang = 1.029`, `ocr_paddle = 0.998` ($W = 3.000$).

| Case | Location | `ocr_api` | `ocr_hunyuang` | `ocr_paddle` | Rule Applied | Confidence Calculation | Resolved | Review? |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Unanimous | Block 1, Pos 0 | `新` | `新` | `新` | `all_agree` | $3.000 / 3.000 = 1.00$ | `新` | No |
| 2 vs 1 | Block 1, Pos 3-4 | `桑` | `桑` | `棄` | `two_agree` | Combined pair (`api`+`hunyuang`): $(0.973 + 1.029) / 3.000 = 0.67$ | `桑` | No |
| 3-way tie | Block 1, Pos 1-2 | `刋` | `判` | `刊` | `all_disagree` | Weighted scores: `判`=0.34, `刊`=0.33, `刋`=0.32 | `判` | Yes |

---

## 5. Empirical Summary on `res/`

| Document | Blocks | Aligned Cols | Unanimous % | Avg Conf % | Total Conflicts | `two_agree` | `all_disagree` | Needs Review |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Linh khởi điểm...12 đường khóa.txt` | 197 | 139,856 | 32.99% | 86.32% | 7,463 | 7,295 | 168 | 168 |
| `phuong-te-hoc...dung-hua.txt` | 266 | 303,967 | 5.13% | 93.14% | 6,784 | 4,938 | 1,846 | 1,846 |
| `Tân_san...q10.txt` | 131 | 22,728 | 72.98% | 88.94% | 4,393 | 2,699 | 1,694 | 1,694 |
| `Tân_san...q5.txt` | 51 | 7,826 | 70.50% | 87.76% | 1,625 | 974 | 651 | 651 |
| `Tân_san...q6.txt` | 150 | 22,777 | 72.61% | 88.87% | 4,621 | 2,951 | 1,670 | 1,670 |
| `Tân_san...q8.txt` | 112 | 19,783 | 69.39% | 87.13% | 4,008 | 2,273 | 1,735 | 1,735 |
| `Tân_san...q9.txt` | 130 | 21,203 | 71.03% | 88.07% | 4,295 | 2,586 | 1,709 | 1,709 |

---

## 6. Reproducing the Pipeline

```powershell
# Phase 1: Normalize, Levenshtein-align, vote, and generate disagreement reports
python ocr_voting/main.py

# Phase 2: Read disagreements, apply 3-tier conflict resolution, and write fixed files
python ocr_voting/resolver.py
```

Optional overrides:
```powershell
# Custom model weights for Phase 1
python ocr_voting/main.py --weights ocr_hunyuang=1.2 ocr_paddle=1.1 ocr_api=0.8

# Custom model confidence scores and output folder for Phase 2
python ocr_voting/resolver.py --model-confidences ocr_hunyuang=0.92 ocr_paddle=0.88 ocr_api=0.75 --output-dir ocr_voting/resolved_output
```
