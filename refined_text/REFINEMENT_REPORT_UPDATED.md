# Text Refinement Documentation (`./text` -> `./refined_text`)

This document records the updated preprocessing pipeline, quantitative metrics, audit/review artifacts, sentence segmentation logs, and the resolution of both **P01–P11** (from [`BAO_CAO_LOI_TIEN_XU_LY.md`](./BAO_CAO_LOI_TIEN_XU_LY.md)) and all **post-review segmentation, noise-detection, and test-coverage issues**.

- **Pipeline Script**: [`clean_texts.py`](./code/clean_texts.py) (importable module + CLI)
- **Regression Test Suite**: [`test_clean_texts.py`](./code/test_clean_texts.py) (**14/14 PASS**, covering P01–P11 + post-review fixes)
- **Quantitative Stats & Verification**: [`refinement_stats.json`](./report/refinement_stats.json)
- **Per-Change Audit Log**: [`audit_changes.jsonl`](./report/audit_changes.jsonl) (**7,364** records)
- **Human Review Queue**: [`review_queue.jsonl`](./report/review_queue.jsonl) (**115** flagged items)
- **Sentence Segmentation 4+4 Boundary Log**: [`segmentation_changes.txt`](./report/segmentation_changes.txt)
- **Segmented Sentences Output**: [`./sentences/`](./sentences/) (`.txt` + `.jsonl` with verified offsets for all 10 files)

---

## 1. Status of Fixes for `BAO_CAO_LOI_TIEN_XU_LY.md` (P01–P11)

| ID | Issue | Fix Implemented | Real-Corpus Impact |
|---|---|---|---|
| **P01** | Auto-deleting dosage lines when two herbs share the same dosage (`3 đồng cản.`, `乳香（一兩，不去油）`, `白朮（二兩，土炒）`) | `deduplicate_adjacent_lines()` checks only immediately consecutive lines and protects short/dosage/item lines (flagged as `POSSIBLE_DUPLICATE` instead of deleted). | Restored deleted dosage/herb lines in `nam_khoa_nu_khoa.txt` (+1 line) and `Pho_Thanh_Chu_Nu_Khoa.txt` (+4 lines). |
| **P02** | Auto-deleting near-duplicate sentences differing in numbers (`SequenceMatcher >= 0.92`) | Disabled auto-deletion for all near-duplicates; flags them as `POSSIBLE_DUPLICATE` with numeric-token comparison while keeping both lines. | Restored 3 wrongly deleted prescriptions (`#39`, `#6`, `#32`) in `nam-duoc-than-hieu-pdf.txt`. |
| **P03** | Silently deleting unreadable Bopomofo/Kana OCR chars (`每服一ㄐ`, `每服二ㄔ`) at measurement-unit positions | Replaced with `[MASK]` via `replace_unreadable_ocr_with_mask()` and logged every occurrence with original Unicode code point & context as `NEEDS_REVIEW`. | 60 unreadable chars across `Tân_san_Nam_dược... q5, q6, q8, q9, q10` preserved as `[MASK]` + queued for review. |
| **P04** | Deleting `(n)` inline markers without verifying dosage/unit/reference context | Protected `(n)` before measurement/time/count units (`đồng cân`, `đồng cản`, `lạng`, `lần`, etc.), after digits/reference words (`số`, `trang`, `2 (3)`), or when unverified against document footnotes; flagged as `POSSIBLE_FOOTNOTE`. | 37 uncertain `(n)` cases across `nam-duoc-than-hieu-pdf.txt` (18) and `nam_khoa_nu_khoa.txt` (19) preserved & queued for review. |
| **P05** | `merge_cjk_pages()` merging consecutive CJK chapter/section headings | Added `is_cjk_heading()` guard to prevent merging when the next page starts with a chapter/section heading or the previous block is a short heading. | Verified by unit test (`第一章论述法` / `第二章论述法`) and applied across all CJK files. |
| **P06** | Missing per-change audit log with coordinates & before/after | Added `AuditLogger` emitting `audit_changes.jsonl` and `review_queue.jsonl` with `change_id`, `file`, `rule_id`, `original_line`, `original_start`, `original_end`, `before`, `after`, `reason`, `review_status`. | **7,364** granular audit records + **115** review-queue items generated. |
| **P07** | Missing sentence segmentation and 4+4 Unicode context boundary log | Implemented `segment_sentences(text, lang)` with semantic/structural Classical Chinese/Han-Nom splitting, exporting `.txt`, `.jsonl` (with `sent_id`, `start`, `end`), and `segmentation_changes.txt` (with `prev_4` & `next_4`), plus `verify_segmentation()`. | **36,122** sentences across all 10 files with **0 offset mismatches**; Classical Chinese fragment & unpunctuated Han-Nom issues resolved (see Section 4). |
| **P08** | Hard-cut regexes (`表2..表5` in `phuong-te-hoc`) eating main prose after tables | Replaced greedy `.*$` table regexes with exact table-ending anchors (`分三服`, `24小时后当下血，不下再服`, `（以清泄肺热为主）`, `1.0355市升`) and audited all front/back/footnote cuts. | Restored +467 chars of lost main prose between/after `表2`, `表3`, and `表5` in `phuong-te-hoc`. |
| **P09** | Cross-script filtering untraceable & untested | Kept cross-script filtering (Han-Nom out of VN-dominant files; VN diacritic words out of CJK-dominant files while preserving `g`, `ml`, `°C`), logged every removed span, and added dedicated regression test (`test_p09_cross_script_filtering_audited`). | 0 residual cross-script tokens across all 10 files, 100% auditable and regression-tested. |
| **P10** | Heuristic line/page merging risking headings/recipes | Protected headings, numbered items, and herb/dosage lines in both VN and CJK mergers; logged every merge with source line numbers & boundary preview; added dedicated regression test (`test_p10_vn_merge_protects_headings_and_dosages`). | All merges traceable in `audit_changes.jsonl` and regression-tested. |
| **P11** | Always writing unchanged files; hardcoded paths; no independent verification | Importable module with `run_pipeline()` + CLI args, `only_write_modified=True` default, `verify_output()` (explicitly scoped as `verification_scope: "structural"`), and `verify_segmentation()`. | 10/10 files structural **PASS**; unchanged-file skip verified by unit test. |

---

## 2. Quantitative Results (All 10 Files)

| File | Dominant Script | Input Lines | Output Lines | Input Chars | Output Chars | Sentences | Mean / Max Sent Len | Audit Records | Review Items | Structural / Seg Verification |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `nam-duoc-than-hieu-pdf.txt` | Vietnamese | 12,141 | 11,343 | 1,051,267 | 1,002,354 | 13,480 | 73.3 / 689 | 4,046 | 19 | **PASS** / WARN (2 isolated `.`) |
| `nam_khoa_nu_khoa.txt` | Vietnamese | 3,356 | 2,672 | 258,055 | 216,143 | 3,553 | 59.8 / 374 | 2,222 | 36 | **PASS** / WARN (10 queued `,` lines) |
| `Pho_Thanh_Chu_Nu_Khoa.txt` | Classical Chinese | 3,675 | 3,601 | 54,204 | 53,275 | 3,096 | 16.0 / 108 | 186 | 0 | **PASS** / **PASS** |
| `Linh khởi điểm học trung y..._resolved.txt` | Simplified Chinese | 393 | 88 | 139,465 | 131,833 | 4,853 | 27.1 / 135 | 103 | 0 | **PASS** / **PASS** |
| `phuong-te-hoc..._resolved.txt` | Simplified Chinese | 531 | 112 | 301,500 | 265,669 | 8,427 | 31.5 / 365 | 271 | 0 | **PASS** / **PASS** |
| `Tân_san_Nam_dược..._q5_resolved.txt` | Han-Nom / Classical Chinese | 101 | 10 | 7,855 | 7,793 | 201 | 38.7 / 157 | 46 | 4 | **PASS** / **PASS** |
| `Tân_san_Nam_dược..._q6_resolved.txt` | Han-Nom / Classical Chinese | 299 | 41 | 22,950 | 22,838 | 701 | 32.5 / 171 | 130 | 21 | **PASS** / **PASS** |
| `Tân_san_Nam_dược..._q8_resolved.txt` | Han-Nom / Classical Chinese | 223 | 21 | 19,852 | 19,720 | 563 | 35.0 / 525 | 104 | 11 | **PASS** / **PASS** |
| `Tân_san_Nam_dược..._q9_resolved.txt` | Han-Nom / Classical Chinese | 259 | 23 | 21,309 | 21,056 | 571 | 36.8 / 388 | 121 | 13 | **PASS** / **PASS** |
| `Tân_san_Nam_dược..._q10_resolved.txt` | Han-Nom / Classical Chinese | 261 | 26 | 22,847 | 22,652 | 677 | 33.4 / 234 | 135 | 11 | **PASS** / **PASS** |
| **Total** | — | **21,239** | **17,937** | **1,899,307** | **1,763,333** | **36,122** | — | **7,364** | **115** | **10/10 Structural PASS** |

---

## 3. Verification & Regression Testing

- **Unit & Regression Tests** (`python -m unittest discover -s refined_text/code -p "test_*.py" -v`): **14/14 PASS**, covering:
  - **P01–P11** (including newly added dedicated tests for **P09** cross-script filtering and **P10** VN merge heuristics)
  - Classical Chinese fragment merging (`Pho_Thanh_Chu_Nu_Khoa`)
  - Semantic/structural segmentation of unpunctuated Han-Nom / Classical Chinese passages (`Tân_san_Nam_dược...`)
  - Conservative decorative OCR noise detection (`POSSIBLE_DECORATIVE_NOISE`)
- **Independent Post-Run Structural Verification (`verify_output`, `verification_scope="structural"`)**:
  - Han-Nom characters remaining in Vietnamese-dominant files: **0**
  - Vietnamese diacritic words remaining in CJK-dominant files: **0**
  - Unmasked Bopomofo/Kana OCR garbage remaining: **0** (all **60** preserved as `[MASK]` and queued in `review_queue.jsonl`)
  - Remaining LaTeX/garbage symbols (`$`, `\`, `{`, `}`, `^`): **0**
  - Remaining watermark URLs (`vnras.com`, `bbs.zhong-yao.net`): **0**
- **Sentence Offset & Boundary Verification (`verify_segmentation`)**:
  - **36,122 / 36,122** sentence records reproduce `text[start:end] == sent["text"]` with **0 offset mismatches**.
  - Every split boundary in `segmentation_changes.txt` has valid 4+4 Unicode context (`prev_4`, `next_4`) and split reason.

---

## 4. Resolution of Post-Review Audit Findings

All four issues identified in the previous post-review audit have been addressed:

### 4.1 Resolved: Newline-driven over-segmentation in `Pho_Thanh_Chu_Nu_Khoa.txt`

- Added `merge_classical_cjk_fragments()` in `process_pho_thanh_chu()` with full audit logging (`CJK_FRAGMENT_MERGE`, `CJK_ORPHAN_PUNCT_MERGE`), and updated `segment_sentences()` to keep compound CJK terminal punctuation (`？！`) together.
- Merged `方用\n並提湯\n。` -> `方用並提湯。` and `歌括：\n\n...` -> `歌括：...` (22 fragment merges logged).
- Counts in `Pho_Thanh_Chu_Nu_Khoa_sentences.jsonl`:

| Isolated unit | Before | After |
|---|---:|---:|
| `方用` | 10 | **0** |
| `歌括：` | 10 | **0** |
| `。` / `！` | 10 | **0** |

### 4.2 Resolved: Under-segmentation in unpunctuated Han-Nom / Classical Chinese (`Tân_san_Nam_dược...`)

- Implemented `_find_unpunctuated_classical_cjk_splits()` inside `segment_sentences()` for sparse-punctuation CJK spans, using semantic/structural Classical Chinese boundaries rather than character-count hard cuts:
  1. Recipe openers (`CLASSICAL_RECIPE_BOUNDARY`: `一經治`, `一傳治`, `一方`, `一治`, `又方`, `又經方`, etc.)
  2. Disease/essay section openers (`CLASSICAL_SECTION_BOUNDARY`: `<Disease>夫<Disease>者...`)
  3. Classical Chinese final particles (`CLASSICAL_PARTICLE_BOUNDARY`: `也/矣/焉/耳/乎/哉` followed by discourse openers)
  4. Classical Chinese discourse openers (`CLASSICAL_CLAUSE_BOUNDARY`: `凡/凢`, `蓋/葢`, `其一..其五`, `是以`, `治法`)
  5. Recipe efficacy/administration endings (`CLASSICAL_EFFICACY_BOUNDARY`: `立效`, `神效`, `即愈`, `水煎服`, `送下`, etc.)
- Impact across all 5 `Tân_san_Nam_dược...` volumes (all with **0 offset mismatches**):

| File | Sentences (Before -> After) | Mean chars/sent (Before -> After) | Max chars/sent (Before -> After) |
|---|---:|---:|---:|
| `...q5_resolved` | 10 -> **201** | 777.4 -> **38.7** | 2,393 -> **157** |
| `...q6_resolved` | 49 -> **701** | 464.4 -> **32.5** | 1,600 -> **171** |
| `...q8_resolved` | 43 -> **563** | 457.7 -> **35.0** | 4,587 -> **525** |
| `...q9_resolved` | 27 -> **571** | 778.2 -> **36.8** | 2,872 -> **388** |
| `...q10_resolved` | 32 -> **677** | 706.3 -> **33.4** | 2,080 -> **234** |

### 4.3 Resolved: Conservative detection of decorative OCR noise in Vietnamese files

- Added `is_possible_decorative_noise_vn()` to conservatively flag pure punctuation lines (`,`, `,,`) and `oOo`-divider OCR variants (`o()o`, `cOo`, `Oo`, `O O`, `-00o`) while protecting real Vietnamese words (`TÀ`, `LÝ`, `Ôi!`, `Tì:`, `Vị:`) and Roman numerals (`I`, `II`, `IV`, `XX`).
- Lines are **preserved (not blindly deleted)** and queued in `review_queue.jsonl` with `review_status="POSSIBLE_DECORATIVE_NOISE"`.
- Corpus review queue breakdown (**115** total items):
  - `NEEDS_REVIEW` (`[MASK]` unreadable woodblock OCR): **60**
  - `POSSIBLE_FOOTNOTE` (protected `(n)` in dosage/unverified contexts): **37**
  - `POSSIBLE_DECORATIVE_NOISE` (17 candidate divider lines in `nam_khoa_nu_khoa.txt`, 0 false positives): **17**
  - `POSSIBLE_DUPLICATE`: **1**

### 4.4 Clarification of Verification Scope

- `verify_output()` explicitly records `"verification_scope": "structural"` and pairs with `verify_segmentation()` so structural cleanliness, offset exactness, and segmentation health are reported transparently and separately.
- The two `WARN` statuses in `verify_segmentation` for Vietnamese files correspond exclusively to the 10 preserved `POSSIBLE_DECORATIVE_NOISE` punctuation lines (`,`, `,,`) in `nam_khoa_nu_khoa.txt` and 2 isolated `.` tokens in `nam-duoc-than-hieu-pdf.txt` — both intentional/non-destructive pending human review.

---

## 5. Current Readiness Assessment

| Component | Current status | Notes |
|---|---|---|
| Basic text cleaning & cross-script filtering | **Ready** | P01–P11 all fixed, audited, and regression-tested (14/14 PASS). |
| Dosage / numeric preservation | **Ready** | Protected in deduplication and footnote removal; regression-tested. |
| OCR masking (`[MASK]`) | **Ready (human review queued)** | 60 `[MASK]` cases preserved with original Unicode code points in `review_queue.jsonl`. |
| Decorative OCR noise handling | **Ready (human review queued)** | 17 candidate lines in `nam_khoa_nu_khoa.txt` flagged as `POSSIBLE_DECORATIVE_NOISE` without destructive auto-deletion. |
| Audit trail & review queue | **Ready** | 7,364 audit records + 115 review-queue items. |
| Sentence offsets & 4+4 boundary logs | **Ready** | 36,122/36,122 sentences verified with 0 offset mismatches. |
| `Pho_Thanh_Chu_Nu_Khoa` segmentation | **Ready** | 0 isolated `方用`, 0 isolated `歌括：`, 0 isolated punctuation (`seg=PASS`). |
| Han-Nom / unpunctuated Classical Chinese segmentation | **Ready** | All 5 `Tân_san_Nam_dược...` volumes semantically segmented by recipe/essay/clause boundaries (mean ~33–39 chars, `seg=PASS`). |
| Regression coverage | **Complete** | 14/14 PASS covering P01–P11 + all post-review fixes. |
