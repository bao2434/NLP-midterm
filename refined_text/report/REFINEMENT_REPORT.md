# Text Refinement Documentation (`./text` -> `./refined_text`)

This document records the updated preprocessing pipeline, quantitative metrics, audit/review artifacts, sentence segmentation logs, and fixes applied for all issues **P01–P11** reported in [`BAO_CAO_LOI_TIEN_XU_LY.md`](../BAO_CAO_LOI_TIEN_XU_LY.md).

- **Pipeline Script**: [`clean_texts.py`](../code/clean_texts.py) (importable module + CLI)
- **Regression Test Suite**: [`test_clean_texts.py`](../code/test_clean_texts.py) (9/9 PASS)
- **Quantitative Stats & Verification**: [`refinement_stats.json`](./refinement_stats.json)
- **Per-Change Audit Log**: [`audit_changes.jsonl`](./audit_changes.jsonl) (7,325 records)
- **Human Review Queue**: [`review_queue.jsonl`](./review_queue.jsonl) (98 flagged items)
- **Sentence Segmentation 4+4 Boundary Log**: [`segmentation_changes.txt`](./segmentation_changes.txt)
- **Segmented Sentences Output**: [`../sentences/`](../sentences/) (`.txt` + `.jsonl` with offsets for all 10 files)

---

## 1. Summary of Fixes for `BAO_CAO_LOI_TIEN_XU_LY.md` (P01–P11)

| ID | Issue | Fix Implemented | Real-Corpus Impact |
|---|---|---|---|
| **P01** | Auto-deleting dosage lines when two herbs share the same dosage (`3 đồng cản.`, `乳香（一兩，不去油）`, `白朮（二兩，土炒）`) | `deduplicate_adjacent_lines()` now checks only immediately consecutive lines (never across intervening lines) and protects short/dosage/item lines (flagged as `POSSIBLE_DUPLICATE` instead of deleted). | Restored deleted dosage/herb lines in `nam_khoa_nu_khoa.txt` (+1 line) and `Pho_Thanh_Chu_Nu_Khoa.txt` (+4 lines). |
| **P02** | Auto-deleting near-duplicate sentences differing in numbers (`SequenceMatcher >= 0.92`) | Disabled auto-deletion for all near-duplicates; now flags them as `POSSIBLE_DUPLICATE` with numeric-token comparison while keeping both lines. | Restored 3 wrongly deleted prescriptions (`#39`, `#6`, `#32`) in `nam-duoc-than-hieu-pdf.txt`. |
| **P03** | Silently deleting unreadable Bopomofo/Kana OCR chars (`每服一ㄐ`, `每服二ㄔ`) at measurement-unit positions | Replaced with `[MASK]` via `replace_unreadable_ocr_with_mask()` and logged every occurrence with original Unicode code point & context as `NEEDS_REVIEW`. | 60 unreadable chars across `Tân_san_Nam_dược... q5, q6, q8, q9, q10` preserved as `[MASK]` + queued for review. |
| **P04** | Deleting `(n)` inline markers without verifying dosage/unit/reference context | Protected `(n)` before measurement/time/count units (`đồng cân`, `đồng cản`, `lạng`, `lần`, etc.), after digits/reference words (`số`, `trang`, `2 (3)`), or when unverified against document footnotes; flagged as `POSSIBLE_FOOTNOTE`. | 37 uncertain `(n)` cases across `nam-duoc-than-hieu-pdf.txt` (18) and `nam_khoa_nu_khoa.txt` (19) preserved & queued for review. |
| **P05** | `merge_cjk_pages()` merging consecutive CJK chapter/section headings | Added `is_cjk_heading()` guard to prevent merging when the next page starts with a chapter/section heading or the previous block is a short heading. | Verified by unit test (`第一章论述法` / `第二章论述法`) and applied across all CJK files. |
| **P06** | Missing per-change audit log with coordinates & before/after | Added `AuditLogger` emitting `audit_changes.jsonl` and `review_queue.jsonl` with `change_id`, `file`, `rule_id`, `original_line`, `original_start`, `original_end`, `before`, `after`, `reason`, `review_status`. | 7,325 granular audit records + 98 review-queue items generated. |
| **P07** | Missing sentence segmentation and 4+4 Unicode context boundary log | Implemented `segment_sentences(text, lang)` (protecting decimals, abbreviations, list indices) exporting `.txt`, `.jsonl` (with `sent_id`, `start`, `end`), and `segmentation_changes.txt` (with `prev_4` & `next_4`). | All 10 files segmented with full offset & 4+4 boundary logs. |
| **P08** | Hard-cut regexes (`表2..表5` in `phuong-te-hoc`) eating main prose after tables | Replaced greedy `.*$` table regexes with exact table-ending anchors (`分三服`, `24小时后当下血，不下再服`, `（以清泄肺热为主）`, `1.0355市升`) and audited all front/back/footnote cuts. | Restored +467 chars of lost main prose between/after `表2`, `表3`, and `表5` in `phuong-te-hoc`. |
| **P09** | Cross-script filtering untraceable | Kept user-requested cross-script filtering (Han-Nom out of VN-dominant files; VN diacritic words out of CJK-dominant files) and logged every removed span in `audit_changes.jsonl`. | 0 residual cross-script tokens across all 10 files, 100% auditable. |
| **P10** | Heuristic line/page merging risking headings/recipes | Protected headings, numbered items, and herb/dosage lines in both VN and CJK mergers; logged every merge with source line numbers & boundary preview. | All merges traceable in `audit_changes.jsonl`. |
| **P11** | Always writing unchanged files; hardcoded paths; no independent verification | Refactored into importable module with `run_pipeline()` + CLI args, `only_write_modified=True` default, and `verify_output()` independent verification against raw input. | 10/10 files verified `PASS`; unchanged-file skip verified by unit test. |

---

## 2. Quantitative Results (All 10 Files)

| File | Dominant Script | Input Lines | Output Lines | Input Chars | Output Chars | Audit Records | Review Items | Verification |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `nam-duoc-than-hieu-pdf.txt` | Vietnamese | 12,141 | 11,343 | 1,051,267 | 1,002,354 | 4,046 | 19 | **PASS** |
| `nam_khoa_nu_khoa.txt` | Vietnamese | 3,356 | 2,672 | 258,055 | 216,143 | 2,205 | 19 | **PASS** |
| `Pho_Thanh_Chu_Nu_Khoa.txt` | Classical Chinese | 3,675 | 3,644 | 54,204 | 53,318 | 164 | 0 | **PASS** |
| `Linh khởi điểm học trung y..._resolved.txt` | Simplified Chinese | 393 | 88 | 139,465 | 131,833 | 103 | 0 | **PASS** |
| `phuong-te-hoc..._resolved.txt` | Simplified Chinese | 531 | 112 | 301,500 | 265,669 | 271 | 0 | **PASS** |
| `Tân_san_Nam_dược..._q5_resolved.txt` | Han-Nom / Classical Chinese | 101 | 10 | 7,855 | 7,793 | 46 | 4 | **PASS** |
| `Tân_san_Nam_dược..._q6_resolved.txt` | Han-Nom / Classical Chinese | 299 | 41 | 22,950 | 22,838 | 130 | 21 | **PASS** |
| `Tân_san_Nam_dược..._q8_resolved.txt` | Han-Nom / Classical Chinese | 223 | 21 | 19,852 | 19,720 | 104 | 11 | **PASS** |
| `Tân_san_Nam_dược..._q9_resolved.txt` | Han-Nom / Classical Chinese | 259 | 23 | 21,309 | 21,056 | 121 | 13 | **PASS** |
| `Tân_san_Nam_dược..._q10_resolved.txt` | Han-Nom / Classical Chinese | 261 | 26 | 22,847 | 22,652 | 135 | 11 | **PASS** |
| **Total** | — | **21,239** | **17,980** | **1,899,307** | **1,763,376** | **7,325** | **98** | **10/10 PASS** |

---

## 3. Verification & Regression Testing

- **Unit & Regression Tests** (`python -m unittest discover -s refined_text/code -p "test_*.py" -v`): **9/9 PASS** covering P01, P02, P03, P04, P05, P06, P07, P08, and P11.
- **Independent Post-Run Verification (`verify_output`)**:
  - Han-Nom characters remaining in Vietnamese-dominant files: **0**
  - Vietnamese diacritic words remaining in CJK-dominant files: **0**
  - Unmasked Bopomofo/Kana OCR garbage remaining: **0** (all **60** preserved as `[MASK]` and queued in `review_queue.jsonl`)
  - Remaining LaTeX/garbage symbols (`$`, `\`, `{`, `}`, `^`): **0**
  - Remaining watermark URLs (`vnras.com`, `bbs.zhong-yao.net`): **0**
