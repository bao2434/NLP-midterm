#!/usr/bin/env python3
"""
Text Refinement & Sentence Segmentation Pipeline for Han-Nom / Vietnamese TCM Corpus.

Fixes all issues P01-P11 identified in BAO_CAO_LOI_TIEN_XU_LY.md while fulfilling all
preprocessing requirements:
  1. Remove headers, footers, page numbers, watermarks, and non-main-content pages.
  2. Merge words/sentences broken across page breaks while protecting headings & recipes.
  3. Safe adjacent-line deduplication (never auto-deletes dosage lines, short lines, or
     near-duplicates differing in numbers/units; flags candidates to review queue).
  4. Remove footnotes outside main content & unambiguous inline footnote markers while
     protecting numbers/units in prescriptions (flagging uncertain cases).
  5. Fix garbage characters & abnormal whitespace; replace unreadable woodblock OCR
     Bopomofo/Kana characters at meaningful positions with [MASK] + NEEDS_REVIEW log.
  6. Normalize punctuation without changing meaning.
  7. Preserve proper nouns, era names, numbers, and measurement units.
  8. Skip tables and image/figure plates with exact boundaries (no prose loss).
  9. Filter Han-Nom text in Vietnamese-dominant files and Vietnamese text in CJK-dominant
     files, with full per-change audit logging.
 10. Sentence segmentation with 4+4 Unicode context boundary logs and offsets.
 11. Importable module with configurable paths, only-write-modified behavior, and
     independent output verification.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import unicodedata
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "text"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "refined_text" / "res"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "refined_text" / "report"
DEFAULT_SENTENCES_DIR = PROJECT_ROOT / "refined_text" / "sentences"


# ---------------------------------------------------------------------------
# Audit Logging (P06, P09, P10)
# ---------------------------------------------------------------------------

@dataclass
class AuditRecord:
    change_id: int
    file: str
    rule_id: str
    original_line: int
    original_start: int
    original_end: int
    before: str
    after: str
    reason: str
    review_status: str = "AUTO_APPLIED"  # AUTO_APPLIED | NEEDS_REVIEW | POSSIBLE_DUPLICATE | POSSIBLE_FOOTNOTE | AMBIGUOUS_MERGE


class AuditLogger:
    """Collects per-change audit records for full traceability and human review."""

    def __init__(self, filename: str = ""):
        self.filename = filename
        self.records: list[AuditRecord] = []
        self._next_id = 1

    def log(
        self,
        rule_id: str,
        original_line: int,
        before: str,
        after: str,
        reason: str,
        original_start: int = 0,
        original_end: int = 0,
        review_status: str = "AUTO_APPLIED",
        filename: str | None = None,
    ) -> AuditRecord:
        rec = AuditRecord(
            change_id=self._next_id,
            file=filename or self.filename,
            rule_id=rule_id,
            original_line=original_line,
            original_start=original_start,
            original_end=original_end,
            before=before,
            after=after,
            reason=reason,
            review_status=review_status,
        )
        self._next_id += 1
        self.records.append(rec)
        return rec

    def review_items(self) -> list[AuditRecord]:
        return [r for r in self.records if r.review_status != "AUTO_APPLIED"]

    def to_dicts(self) -> list[dict]:
        return [asdict(r) for r in self.records]


# ---------------------------------------------------------------------------
# Common regexes & utility helpers
# ---------------------------------------------------------------------------

# Unicode ranges for Han-Nom / CJK ideographs & CJK-specific symbols
HAN_NOM_RE = re.compile(
    r"[\u2e80-\u2fdf\u3005-\u3007\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0003134f]+"
)

# Vietnamese-specific characters (diacritics + đ/Đ)
VN_DIAC_WORD_RE = re.compile(
    r"\b[A-Za-z]*[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ"
    r"ÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ]"
    r"[A-Za-zàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ"
    r"ÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ]*\b"
)

# Bopomofo / Kana OCR misreads in woodblock texts
BOPOMOFO_KANA_RE = re.compile(r"[\u3100-\u312f\u3040-\u30ff]")

# Numbers & measurement units protection (P01, P02, P04)
VN_UNITS_PATTERN = (
    r"(?:đồng\s*cân|đồng\s*cản|đ/c|lạng|phân|chỉ|ly|cân|thăng|hợp|đấu|chén|bát|thìa|"
    r"giọt|hạt|miếng|lát|tấc|thước|g|kg|mg|ml|cc|ngày|tháng|năm|lần|thang|tuần|giờ|phút|tuổi|độ)"
)
CJK_UNITS_PATTERN = r"[兩両錢分厘毫斤升合斗碗盞杯盃枚粒片丸個个條只日次服歲月年寸尺]"

VN_DOSAGE_LINE_RE = re.compile(
    rf"(?:\b\d+(?:[.,]\d+)?\s*{VN_UNITS_PATTERN}\b|^\s*[-•*]\s*.+\d|\bđồng\s*c[âả]n\b)",
    re.I,
)
CJK_DOSAGE_LINE_RE = re.compile(
    rf"[一二三四五六七八九十百半\d]+{CJK_UNITS_PATTERN}"
)

NUMBER_TOKEN_RE = re.compile(r"\d+(?:[.,/]\d+)?|[一二三四五六七八九十百千万廿卅半]+")


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def _extract_number_tokens(s: str) -> list[str]:
    return NUMBER_TOKEN_RE.findall(s)


def _is_dosage_or_item_line(s: str) -> bool:
    """Return True if a line looks like an herb/dosage line or numbered entry (must not auto-dedup)."""
    st = s.strip()
    if not st:
        return False
    if re.match(r"^(?:\d+[.)]\s|[-–—•*]\s|[IVX]+\.\s)", st):
        return True
    if len(st) < 40 and (VN_DOSAGE_LINE_RE.search(st) or CJK_DOSAGE_LINE_RE.search(st)):
        return True
    return False


# ---------------------------------------------------------------------------
# Cross-script filtering (P09: audited & clean)
# ---------------------------------------------------------------------------

def strip_han_nom_in_vn(
    text: str,
    stats: dict,
    audit: AuditLogger | None = None,
    line_no: int = 0,
) -> str:
    """Remove Han-Nom / CJK characters from Vietnamese-dominant text with full audit logging."""
    matches = list(HAN_NOM_RE.finditer(text))
    if not matches:
        return text

    orig = text
    removed_spans = [m.group(0) for m in matches]
    text, n = HAN_NOM_RE.subn("", text)
    stats["cross_script_tokens_removed"] += n

    # Clean empty parentheses/brackets or stray punctuation left behind
    text = re.sub(r"\(\s*\)", "", text)
    text = re.sub(r"\[\s*\]", "", text)
    text = re.sub(r"[ \t]{2,}", " ", text).strip()
    if re.fullmatch(r"[\s\-–—•·.,;:!?()\[\]\"']*", text):
        text = ""

    if audit is not None:
        audit.log(
            rule_id="CROSS_SCRIPT_HAN_IN_VN",
            original_line=line_no,
            original_start=matches[0].start(),
            original_end=matches[-1].end(),
            before=orig,
            after=text,
            reason=f"Removed {n} Han-Nom span(s) in Vietnamese-dominant file: {', '.join(removed_spans[:5])}",
            review_status="AUTO_APPLIED",
        )
    return text


def strip_vietnamese_in_cjk(
    text: str,
    stats: dict,
    audit: AuditLogger | None = None,
    line_no: int = 0,
) -> str:
    """Remove Vietnamese words from Han-Nom / Chinese-dominant text (preserving g, ml, kg, °C)."""
    matches = list(VN_DIAC_WORD_RE.finditer(text))
    if not matches:
        return text

    orig = text
    removed_words = [m.group(0) for m in matches]
    text, n = VN_DIAC_WORD_RE.subn("", text)
    stats["cross_script_tokens_removed"] += n
    text = re.sub(r"[ \t]{2,}", " ", text).strip()

    if audit is not None:
        audit.log(
            rule_id="CROSS_SCRIPT_VN_IN_CJK",
            original_line=line_no,
            original_start=matches[0].start(),
            original_end=matches[-1].end(),
            before=orig,
            after=text,
            reason=f"Removed {n} Vietnamese word(s) in CJK-dominant file: {', '.join(removed_words[:5])}",
            review_status="AUTO_APPLIED",
        )
    return text


# ---------------------------------------------------------------------------
# Vietnamese cleaning helpers (P04 fixed + audited)
# ---------------------------------------------------------------------------

def clean_latex_and_markdown_vn(
    text: str,
    stats: dict,
    audit: AuditLogger | None = None,
    line_no: int = 0,
) -> str:
    """Clean LaTeX / Markdown OCR artifacts in Vietnamese texts."""
    orig = text

    def _sub_sup(m):
        stats["inline_footnote_markers_removed"] += 1
        if audit is not None:
            audit.log(
                rule_id="LATEX_SUPERSCRIPT_FOOTNOTE",
                original_line=line_no,
                original_start=m.start(),
                original_end=m.end(),
                before=m.group(0),
                after="",
                reason="Removed LaTeX superscript footnote marker",
                review_status="AUTO_APPLIED",
            )
        return ""

    text = re.sub(r"\s*\$\s*\^\{\s*\(?\s*\d{1,3}\s*\)?\s*\}\s*\$", _sub_sup, text)

    # Remove stray LaTeX superscript dots/backticks: $ ^{·} $, $ ^{｀} $
    text, n = re.subn(r"\s*\$\s*\^\{\s*[·｀`\'*]\s*\}\s*\$", "", text)
    stats["garbage_tokens_cleaned"] += n

    # Unwrap LaTeX text wrappers: $ \textit{...} $, $ \textbf{...} $, $ \text{...} $
    text, n = re.subn(r"\$\s*\\text(?:it|bf|rm|sf|tt)?\{([^}]*)\}\s*\$", r"\1", text)
    stats["garbage_tokens_cleaned"] += n

    # Unwrap unclosed $ \text{...
    text, n = re.subn(r"\$\s*\\text(?:it|bf|rm)?\{", "", text)
    stats["garbage_tokens_cleaned"] += n

    # Remove leading backslash on lines
    text, n = re.subn(r"^\\+\*?", "", text)
    stats["garbage_tokens_cleaned"] += n

    # Remove markdown bold/italic wrappers (*...* or **...**)
    text, n = re.subn(r"\*\*([^*]+)\*\*", r"\1", text)
    stats["garbage_tokens_cleaned"] += n
    text, n = re.subn(r"\*([^*\n]+)\*", r"\1", text)
    stats["garbage_tokens_cleaned"] += n

    # Remove remaining stray markdown asterisks, braces, backslashes, carets, $
    text, n = re.subn(r"(?<!\d)\*(?!\d)", "", text)
    stats["garbage_tokens_cleaned"] += n
    text, n = re.subn(r"[\\{}^]", "", text)
    stats["garbage_tokens_cleaned"] += n
    text, n = re.subn(r"\$\s*(?![\d])", "", text)
    stats["garbage_tokens_cleaned"] += n

    # Replace OCR table pipes in herb lines
    if "|" in text:
        text, n = re.subn(r"\s*\|\s*", " ", text)
        stats["garbage_tokens_cleaned"] += n

    if audit is not None and text != orig:
        audit.log(
            rule_id="CLEAN_LATEX_MARKDOWN_VN",
            original_line=line_no,
            original_start=0,
            original_end=len(orig),
            before=orig,
            after=text,
            reason="Cleaned LaTeX/Markdown OCR artifacts",
            review_status="AUTO_APPLIED",
        )
    return text


def normalize_punctuation_vn(
    text: str,
    stats: dict,
    audit: AuditLogger | None = None,
    line_no: int = 0,
) -> str:
    """Normalize punctuation and whitespace in Vietnamese text without altering meaning."""
    orig = text

    replacements = [
        ("«", '"'), ("»", '"'),
        ("《", '"'), ("》", '"'),
        ("“", '"'), ("”", '"'),
        ("‘", "'"), ("’", "'"),
        ("，", ", "), ("：", ": "), ("；", "; "),
        ("（", "("), ("）", ")"),
        ("？", "?"), ("！", "!"),
        ("︰", ": "),
    ]
    for old, new in replacements:
        if old in text:
            text = text.replace(old, new)

    text = re.sub(r"\.{4,}", "...", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\s+([,.;:!?)\]])", r"\1", text)
    text = re.sub(r"([(\[])\s+", r"\1", text)
    text = re.sub(r"([,;:])([A-Za-zÀ-ỹĐđ])", r"\1 \2", text)
    text = re.sub(r"(?<!\.)\.([A-ZÀ-ỸĐ][a-zà-ỹđ])", r". \1", text)

    text = text.strip()
    if text != orig:
        stats["punctuation_ws_normalized_lines"] += 1
        if audit is not None:
            audit.log(
                rule_id="NORMALIZE_PUNCT_WS_VN",
                original_line=line_no,
                original_start=0,
                original_end=len(orig),
                before=orig,
                after=text,
                reason="Normalized punctuation and whitespace in Vietnamese line",
                review_status="AUTO_APPLIED",
            )
    return text


# Protected context patterns for P04 (never auto-delete (n) in these contexts)
_PAREN_PROTECTED_AFTER_RE = re.compile(
    rf"^\s*(?:{VN_UNITS_PATTERN}\b|đồng(?:\s*[.,;:!?\)]|\s*$))",
    re.I,
)
_PAREN_PROTECTED_BEFORE_RE = re.compile(
    r"(?:số|trang|mục|bài|quyển|chương|phần|tập|điều|khoản|hạng|loại|thứ|cấp|dùng|uống|lấy|mỗi|khoảng|chừng|\d)\s*$",
    re.I,
)


def remove_inline_footnote_markers_vn(
    text: str,
    stats: dict,
    audit: AuditLogger | None = None,
    line_no: int = 0,
    known_footnotes: set[int] | None = None,
) -> str:
    """
    Remove unambiguous inline footnote index markers like [1], (1), (12) in Vietnamese prose.
    P04 fix: Protects (n) when followed by measurement/dosage/time units (e.g. 'Phục linh (3) đồng cân')
    or preceded by digits/reference words (e.g. 'dùng 2 (3) lần', 'số (1)'), logging uncertain cases
    as POSSIBLE_FOOTNOTE instead of deleting them.
    """
    # 1. Remove bracketed [1]..[99] (unambiguous footnote markers; never touch [MASK])
    def _sub_bracket(m):
        stats["inline_footnote_markers_removed"] += 1
        if audit is not None:
            audit.log(
                rule_id="INLINE_FOOTNOTE_BRACKET",
                original_line=line_no,
                original_start=m.start(),
                original_end=m.end(),
                before=text[max(0, m.start() - 15):min(len(text), m.end() + 15)],
                after=(
                    text[max(0, m.start() - 15):m.start()]
                    + text[m.end():min(len(text), m.end() + 15)]
                ),
                reason=f"Removed bracketed inline footnote marker {m.group(0).strip()}",
                review_status="AUTO_APPLIED",
            )
        return ""

    text = re.sub(r"\s*\[\d{1,2}\]", _sub_bracket, text)

    # 2. Context-checked removal of (1)..(99)
    out_parts: list[str] = []
    last_end = 0
    for m in re.finditer(r"(\s*)\((\d{1,2})\)", text):
        full_start, full_end = m.start(), m.end()
        num = int(m.group(2))
        before_ctx = text[:full_start]
        after_ctx = text[full_end:]
        snippet = text[max(0, full_start - 20):min(len(text), full_end + 20)]

        # Never delete at very start of line (list/section marker)
        if full_start == 0:
            continue

        # Protect dosage / unit / numeric / reference contexts (P04)
        is_protected_after = bool(_PAREN_PROTECTED_AFTER_RE.match(after_ctx))
        is_protected_before = bool(_PAREN_PROTECTED_BEFORE_RE.search(before_ctx))
        if is_protected_after or is_protected_before:
            stats["possible_footnotes_flagged"] += 1
            if audit is not None:
                audit.log(
                    rule_id="P04_PROTECTED_PAREN_NUMBER",
                    original_line=line_no,
                    original_start=full_start,
                    original_end=full_end,
                    before=snippet,
                    after=snippet,
                    reason=f"Preserved '({num})' due to protected numeric/unit/reference context",
                    review_status="POSSIBLE_FOOTNOTE",
                )
            continue

        # If known_footnotes is supplied and num is not in it, keep & flag for review
        if known_footnotes is not None and num not in known_footnotes:
            stats["possible_footnotes_flagged"] += 1
            if audit is not None:
                audit.log(
                    rule_id="P04_UNVERIFIED_PAREN_NUMBER",
                    original_line=line_no,
                    original_start=full_start,
                    original_end=full_end,
                    before=snippet,
                    after=snippet,
                    reason=f"Preserved '({num})' because footnote #{num} was not verified in document notes",
                    review_status="POSSIBLE_FOOTNOTE",
                )
            continue

        # Verified/safe inline footnote marker -> remove & log
        out_parts.append(text[last_end:full_start])
        last_end = full_end
        stats["inline_footnote_markers_removed"] += 1
        if audit is not None:
            after_snippet = (
                text[max(0, full_start - 20):full_start]
                + text[full_end:min(len(text), full_end + 20)]
            )
            audit.log(
                rule_id="INLINE_FOOTNOTE_PAREN",
                original_line=line_no,
                original_start=full_start,
                original_end=full_end,
                before=snippet,
                after=after_snippet,
                reason=f"Removed inline footnote marker ({num})",
                review_status="AUTO_APPLIED",
            )

    if last_end == 0:
        return text
    out_parts.append(text[last_end:])
    return "".join(out_parts)


# ---------------------------------------------------------------------------
# Safe Deduplication (P01 & P02 fixed)
# ---------------------------------------------------------------------------

def deduplicate_adjacent_lines(
    lines: list[str] | list[tuple[int, str]],
    stats: dict,
    threshold: float = 0.92,
    audit: AuditLogger | None = None,
) -> list[str] | list[tuple[int, str]]:
    """
    Safe adjacent-line deduplication fixing P01 and P02:
      - Accepts either list[str] or list[tuple[int, str]] (with source line numbers).
      - NEVER compares across intervening non-empty lines (fixes P01: two herbs with same dosage).
      - NEVER auto-deletes short lines (< 25 chars) or dosage/numbered item lines.
      - NEVER auto-deletes near-duplicates (ratio >= threshold when s != ps); instead flags them
        as POSSIBLE_DUPLICATE in the audit/review queue and keeps BOTH lines (fixes P02).
      - Only auto-deletes exact immediately consecutive identical prose lines (len >= 25).
    """
    is_indexed = bool(lines) and isinstance(lines[0], tuple)
    indexed_lines: list[tuple[int, str]] = (
        lines if is_indexed else [(i + 1, l) for i, l in enumerate(lines)]  # type: ignore[assignment]
    )

    out: list[tuple[int, str]] = []
    for line_no, line in indexed_lines:
        s = line.strip()
        if not s:
            out.append((line_no, ""))
            continue

        # Only inspect the IMMEDIATELY previous element in `out` (no skipping across other lines!)
        if out and out[-1][1].strip():
            prev_line_no, prev_raw = out[-1]
            ps = prev_raw.strip()

            # Case 1: Exact immediately consecutive match
            if s == ps:
                if len(s) >= 25 and not _is_dosage_or_item_line(s):
                    stats["duplicate_lines_removed"] += 1
                    if audit is not None:
                        audit.log(
                            rule_id="EXACT_CONSECUTIVE_DUPLICATE",
                            original_line=line_no,
                            original_start=0,
                            original_end=len(s),
                            before=s,
                            after="",
                            reason=f"Removed exact consecutive OCR duplicate of line {prev_line_no}",
                            review_status="AUTO_APPLIED",
                        )
                    continue
                else:
                    # Short or dosage/item line: keep & flag for review
                    stats["possible_duplicates_flagged"] += 1
                    if audit is not None:
                        audit.log(
                            rule_id="P01_PROTECTED_EXACT_DUPLICATE",
                            original_line=line_no,
                            original_start=0,
                            original_end=len(s),
                            before=s,
                            after=s,
                            reason=f"Kept short/dosage exact repeat of line {prev_line_no} (protected)",
                            review_status="POSSIBLE_DUPLICATE",
                        )

            # Case 2: Near-duplicate -> NEVER auto-delete (P02), only flag for review
            elif len(s) >= 25 and len(ps) >= 25:
                ratio = difflib.SequenceMatcher(None, s, ps).ratio()
                if ratio >= threshold:
                    nums_cur = _extract_number_tokens(s)
                    nums_prev = _extract_number_tokens(ps)
                    num_note = (
                        f"numbers differ ({nums_prev} vs {nums_cur})"
                        if nums_cur != nums_prev
                        else "numbers match"
                    )
                    stats["possible_duplicates_flagged"] += 1
                    if audit is not None:
                        audit.log(
                            rule_id="P02_NEAR_DUPLICATE_FLAGGED",
                            original_line=line_no,
                            original_start=0,
                            original_end=len(s),
                            before=f"PREV(L{prev_line_no}): {ps} || CURR(L{line_no}): {s}",
                            after=s,
                            reason=f"Kept near-duplicate (ratio={ratio:.3f}, {num_note}); requires human review",
                            review_status="POSSIBLE_DUPLICATE",
                        )

        out.append((line_no, line))

    if is_indexed:
        return out
    return [l for _, l in out]


def collapse_blank_lines(lines: list[str]) -> list[str]:
    out = []
    prev_blank = True
    for l in lines:
        if not l.strip():
            if not prev_blank:
                out.append("")
            prev_blank = True
        else:
            out.append(l)
            prev_blank = False
    while out and not out[-1].strip():
        out.pop()
    return out


# ---------------------------------------------------------------------------
# 1. Processor: nam-duoc-than-hieu-pdf.txt
# ---------------------------------------------------------------------------

def is_ndth_footnote_line(s: str) -> bool:
    """Detect bottom-of-page footnote lines in Nam Duoc Than Hieu."""
    if re.match(r"^[\\*]*\(\d{1,2}\)\s*\S", s):
        return True
    if re.match(r"^\d{1,2}\.\s+", s):
        editorial_markers = [
            "Nguyên bản", "nguyên bản", "BTM", "Bản thảo", "chữ nôm", "chữ Nôm",
            "định chính", "định chỉnh", "dùng thay", "dung thay", "cùng họ",
            "Đổi tên chữ Hán", "Thco thản phá", "Quốc sử di biên",
            "Thập tam phương gia giảm và", "Xem bản dịch", "Bản in ", "bản in ",
            "Lý Thời Trân", "Lý Thơi Trần", "Piper ", "Artemisia ", "Adenosma ",
            "Jatropha ", "Croton ", "Hồng Nghĩa giác tư",
        ]
        if any(k in s for k in editorial_markers):
            return True
    return False


def is_vn_heading_or_item(s: str) -> bool:
    """Check if a Vietnamese line should NOT be merged onto the previous line."""
    if not s:
        return True
    if len(s) < 70 and s == s.upper() and re.search(r"[A-ZÀ-ỸĐ]{2,}", s):
        return True
    if re.match(r"^(?:QUYỂN|Quyển|CHƯƠNG|PHẦN|MỤC|BÀI|HẾT|I+V?X?\b|\d+[.)]\s|-\s|—|\*)", s):
        return True
    if _is_dosage_or_item_line(s):
        return True
    return False


def process_nam_duoc_than_hieu(
    text: str,
    audit: AuditLogger | None = None,
) -> tuple[str, dict]:
    stats: dict = defaultdict(int)
    text = nfc(text)
    raw_lines = text.splitlines()
    stats["input_lines"] = len(raw_lines)
    stats["input_chars"] = len(text)

    # 1. Trim front matter -> main content starts at "QUYỂN ĐẦU" (keep 2 title lines before it)
    start_idx = 0
    for i, l in enumerate(raw_lines):
        if l.strip() == "QUYỂN ĐẦU":
            start_idx = max(0, i - 2)
            break
    stats["front_matter_lines_removed"] = start_idx
    if audit is not None and start_idx > 0:
        audit.log(
            rule_id="FRONT_MATTER_TRIM",
            original_line=1,
            original_start=0,
            original_end=start_idx,
            before=f"Lines 1..{start_idx}: {raw_lines[0][:60]} ... {raw_lines[start_idx - 1][:60]}",
            after="",
            reason="Removed front matter before main content ('QUYỂN ĐẦU')",
            review_status="AUTO_APPLIED",
        )

    # 2. Trim back matter ("MỤC LỤC" to EOF)
    end_idx = len(raw_lines)
    for i in range(len(raw_lines) - 1, start_idx, -1):
        if raw_lines[i].strip() == "MỤC LỤC":
            end_idx = i
            break
    stats["back_matter_lines_removed"] = len(raw_lines) - end_idx
    if audit is not None and end_idx < len(raw_lines):
        audit.log(
            rule_id="BACK_MATTER_TRIM",
            original_line=end_idx + 1,
            original_start=end_idx,
            original_end=len(raw_lines),
            before=f"Lines {end_idx + 1}..{len(raw_lines)}: {raw_lines[end_idx][:60]} ...",
            after="",
            reason="Removed back matter starting at 'MỤC LỤC'",
            review_status="AUTO_APPLIED",
        )

    # Collect known footnote numbers from detected footnote lines for P04 verification
    known_footnotes: set[int] = set()
    for l in raw_lines[start_idx:end_idx]:
        s = l.strip()
        m_fn = re.match(r"^[\\*]*\((\d{1,2})\)\s*\S", s)
        if m_fn:
            known_footnotes.add(int(m_fn.group(1)))

    # 3. Filter internal TOC, herb plates, URLs, standalone page numbers, footnotes
    filtered: list[tuple[int, str]] = []
    skip_plate = False
    skip_internal_toc = False

    for offset, l in enumerate(raw_lines[start_idx:end_idx]):
        line_no = start_idx + offset + 1
        s = l.strip()

        if s == "THƯƠNG HÀN CÁCH PHÁP TRỊ LỆ MỤC LỤC":
            skip_internal_toc = True
            stats["non_main_content_lines_removed"] += 1
            if audit is not None:
                audit.log("INTERNAL_TOC_REMOVE", line_no, s, "", "Removed internal TOC header")
            continue
        if skip_internal_toc:
            if re.match(r"^(?:I\b|QUYỂN|THƯƠNG HÀN CÁCH PHÁP TRỊ LỆ$|Bài\s+1\b|1\)\s)", s) and "MỤC LỤC" not in s:
                skip_internal_toc = False
            elif re.match(r"^37\.\s", s):
                skip_internal_toc = False
                stats["non_main_content_lines_removed"] += 1
                continue
            else:
                stats["non_main_content_lines_removed"] += 1
                continue

        if re.search(r"https?://vnras\.com", s, re.I):
            stats["headers_footers_removed"] += 1
            if audit is not None:
                audit.log("WATERMARK_URL_REMOVE", line_no, s, "", "Removed vnras.com watermark URL")
            continue

        if re.match(r"^Bảng\s+[IVX]+$", s):
            skip_plate = True
            stats["tables_figures_lines_removed"] += 1
            if audit is not None:
                audit.log("PLATE_BLOCK_REMOVE", line_no, s, "", "Removed herb illustration plate header")
            continue
        if skip_plate:
            if not s:
                skip_plate = False
                continue
            if len(s) < 45 and not re.match(r"^(NAM DƯỢC|QUYỂN|Mười khoa|Muời khoa)", s):
                stats["tables_figures_lines_removed"] += 1
                continue
            else:
                skip_plate = False

        if re.match(r"^\d{1,4}$", s):
            stats["headers_footers_removed"] += 1
            if audit is not None:
                audit.log("PAGE_NUMBER_REMOVE", line_no, s, "", "Removed standalone page number")
            continue

        if is_ndth_footnote_line(s):
            stats["footnote_lines_removed"] += 1
            if audit is not None:
                audit.log("FOOTNOTE_LINE_REMOVE", line_no, s, "", "Removed bottom-of-page footnote line")
            continue

        filtered.append((line_no, l))

    # 4. Clean LaTeX/Markdown, inline footnotes, Han-Nom cross-script, punctuation
    cleaned: list[tuple[int, str]] = []
    for line_no, l in filtered:
        s = l.strip()
        if not s:
            cleaned.append((line_no, ""))
            continue
        s = clean_latex_and_markdown_vn(s, stats, audit=audit, line_no=line_no)
        s = remove_inline_footnote_markers_vn(
            s, stats, audit=audit, line_no=line_no,
            known_footnotes=known_footnotes if known_footnotes else None,
        )
        s = strip_han_nom_in_vn(s, stats, audit=audit, line_no=line_no)
        if not s:
            continue
        s = normalize_punctuation_vn(s, stats, audit=audit, line_no=line_no)
        if s:
            cleaned.append((line_no, s))

    # 5. Safe deduplication (P01 & P02 fixed)
    cleaned = deduplicate_adjacent_lines(cleaned, stats, audit=audit)  # type: ignore[assignment]

    # 6. Merge prose lines broken across page/line breaks (P10: audited & protected)
    merged: list[str] = []
    i = 0
    while i < len(cleaned):
        cur_ln, cur = cleaned[i]
        if not cur:
            merged.append("")
            i += 1
            continue
        while i + 1 < len(cleaned):
            nxt_idx = i + 1
            if nxt_idx < len(cleaned) and not cleaned[nxt_idx][1] and nxt_idx + 1 < len(cleaned):
                cand_ln, cand = cleaned[nxt_idx + 1]
                if (
                    len(cur) >= 45
                    and not is_vn_heading_or_item(cur)
                    and cur[-1] not in ".!?:;)\""
                    and cand
                    and cand[0].islower()
                    and not is_vn_heading_or_item(cand)
                ):
                    if audit is not None:
                        audit.log(
                            "VN_PAGE_BREAK_MERGE",
                            cur_ln,
                            f"{cur[-30:]} [L{cur_ln}..L{cand_ln}] {cand[:30]}",
                            f"{cur[-30:]} {cand[:30]}",
                            f"Merged prose across blank line (L{cur_ln} + L{cand_ln})",
                        )
                    cur = cur + " " + cand
                    stats["page_break_merges"] += 1
                    i = nxt_idx + 1
                    continue
                else:
                    break
            elif nxt_idx < len(cleaned) and cleaned[nxt_idx][1]:
                cand_ln, cand = cleaned[nxt_idx]
                if (
                    len(cur) >= 45
                    and not is_vn_heading_or_item(cur)
                    and cur[-1] not in ".!?:;)\""
                    and cand[0].islower()
                    and not is_vn_heading_or_item(cand)
                ):
                    if audit is not None:
                        audit.log(
                            "VN_LINE_WRAP_MERGE",
                            cur_ln,
                            f"{cur[-30:]} [L{cur_ln}+L{cand_ln}] {cand[:30]}",
                            f"{cur[-30:]} {cand[:30]}",
                            f"Merged wrapped prose lines (L{cur_ln} + L{cand_ln})",
                        )
                    cur = cur + " " + cand
                    stats["page_break_merges"] += 1
                    i = nxt_idx
                    continue
                else:
                    break
            else:
                break
        merged.append(cur)
        i += 1

    merged = collapse_blank_lines(merged)
    out = "\n".join(merged) + "\n"
    stats["output_lines"] = len(merged)
    stats["output_chars"] = len(out)
    return out, dict(stats)


# ---------------------------------------------------------------------------
# 2. Processor: nam_khoa_nu_khoa.txt
# ---------------------------------------------------------------------------

def process_nam_khoa_nu_khoa(
    text: str,
    audit: AuditLogger | None = None,
) -> tuple[str, dict]:
    stats: dict = defaultdict(int)
    text = nfc(text)
    raw_lines = text.splitlines()
    stats["input_lines"] = len(raw_lines)
    stats["input_chars"] = len(text)

    # 1. Remove translator credit header line 1
    start_idx = 0
    if raw_lines and re.match(r"^DỊCH GI[ẢÁ]:", raw_lines[0].strip()):
        start_idx = 1
        stats["front_matter_lines_removed"] = 1
        if audit is not None:
            audit.log("FRONT_MATTER_TRIM", 1, raw_lines[0], "", "Removed translator credit header")

    # 2. Remove endnotes section ("附註女科" / "PHỤ CHÚ về NỮ KHOA") & back matter
    end_idx = len(raw_lines)
    for i, l in enumerate(raw_lines):
        if l.strip() == "附註女科":
            end_idx = i
            break
    # Collect verified endnote numbers from the endnotes section if present
    known_footnotes: set[int] | None = None
    if end_idx < len(raw_lines):
        found_notes: set[int] = set()
        for l in raw_lines[end_idx:]:
            m_en = re.match(r"^\s*\(?(\d{1,2})[.)]\s+", l)
            if m_en:
                found_notes.add(int(m_en.group(1)))
        if found_notes:
            known_footnotes = found_notes

    while end_idx > start_idx and raw_lines[end_idx - 1].strip() in ("", "HÉT", "HẾT", "oOo"):
        end_idx -= 1

    stats["back_matter_and_endnotes_lines_removed"] = len(raw_lines) - end_idx
    if audit is not None and end_idx < len(raw_lines):
        audit.log(
            "BACK_MATTER_ENDNOTES_TRIM",
            end_idx + 1,
            f"Lines {end_idx + 1}..{len(raw_lines)}: {raw_lines[end_idx][:60]} ...",
            "",
            "Removed translator endnotes ('附註女科') and back matter TOC/colophon",
            original_start=end_idx,
            original_end=len(raw_lines),
        )

    # 3. Clean lines
    cleaned: list[tuple[int, str]] = []
    for offset, l in enumerate(raw_lines[start_idx:end_idx]):
        line_no = start_idx + offset + 1
        s = l.strip()
        if not s:
            cleaned.append((line_no, ""))
            continue
        if re.fullmatch(r"(?:[oO0]\s*[oO0]\s*[oO0]|-{3,}|\*{3,})", s):
            stats["headers_footers_removed"] += 1
            if audit is not None:
                audit.log("DECORATIVE_DIVIDER_REMOVE", line_no, s, "", "Removed decorative divider")
            continue

        orig_s = s
        s, n = re.subn(r"\s*\(Xem phụ chú[^)]*\)", "", s, flags=re.I)
        if n:
            stats["inline_footnote_markers_removed"] += n
            if audit is not None:
                audit.log("INLINE_FOOTNOTE_XREF", line_no, orig_s, s, "Removed '(Xem phụ chú ...)' cross-reference")

        s = clean_latex_and_markdown_vn(s, stats, audit=audit, line_no=line_no)
        s = remove_inline_footnote_markers_vn(
            s, stats, audit=audit, line_no=line_no, known_footnotes=known_footnotes
        )
        s = strip_han_nom_in_vn(s, stats, audit=audit, line_no=line_no)
        if not s:
            continue
        s = normalize_punctuation_vn(s, stats, audit=audit, line_no=line_no)
        if s:
            cleaned.append((line_no, s))

    # 4. Safe deduplication (P01 & P02 fixed)
    cleaned = deduplicate_adjacent_lines(cleaned, stats, audit=audit)  # type: ignore[assignment]

    # 5. Merge sentences broken across lines/pages (protect headings & herb dosage lines)
    merged: list[str] = []
    i = 0
    while i < len(cleaned):
        cur_ln, cur = cleaned[i]
        if not cur:
            merged.append("")
            i += 1
            continue
        while i + 1 < len(cleaned):
            cand_ln, cand = cleaned[i + 1]
            if (
                cand
                and len(cur) >= 50
                and not is_vn_heading_or_item(cur)
                and cur[-1] not in ".!?:;)\""
                and cand[0].islower()
                and not is_vn_heading_or_item(cand)
                and not re.match(r"^[\u4e00-\u9fff]", cur)
                and not re.match(r"^[\u4e00-\u9fff]", cand)
            ):
                if audit is not None:
                    audit.log(
                        "VN_LINE_WRAP_MERGE",
                        cur_ln,
                        f"{cur[-30:]} [L{cur_ln}+L{cand_ln}] {cand[:30]}",
                        f"{cur[-30:]} {cand[:30]}",
                        f"Merged wrapped prose lines (L{cur_ln} + L{cand_ln})",
                    )
                cur = cur + " " + cand
                stats["page_break_merges"] += 1
                i += 1
            else:
                break
        merged.append(cur)
        i += 1

    merged = collapse_blank_lines(merged)
    out = "\n".join(merged) + "\n"
    stats["output_lines"] = len(merged)
    stats["output_chars"] = len(out)
    return out, dict(stats)


# ---------------------------------------------------------------------------
# 3. Processor: Pho_Thanh_Chu_Nu_Khoa.txt (Classical Chinese, Traditional)
# ---------------------------------------------------------------------------

def normalize_cjk_punct_and_ws(
    text: str,
    stats: dict,
    audit: AuditLogger | None = None,
    line_no: int = 0,
) -> str:
    orig = text
    text = strip_vietnamese_in_cjk(text, stats, audit=audit, line_no=line_no)
    text = text.replace("︰", "：").replace("﹗", "！").replace("‧", "·")
    text = re.sub(r"\.{4,}", "……", text)
    text = re.sub(
        r"(?<=[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef])\s+(?=[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef])",
        "",
        text,
    )
    text = re.sub(r"[ \t]{2,}", " ", text).strip()
    if text != orig:
        stats["punctuation_ws_normalized_lines"] += 1
        if audit is not None:
            audit.log(
                "NORMALIZE_PUNCT_WS_CJK",
                line_no,
                orig[:120],
                text[:120],
                "Normalized CJK punctuation and whitespace",
            )
    return text


def process_pho_thanh_chu(
    text: str,
    audit: AuditLogger | None = None,
) -> tuple[str, dict]:
    stats: dict = defaultdict(int)
    text = nfc(text)
    raw_lines = text.splitlines()
    stats["input_lines"] = len(raw_lines)
    stats["input_chars"] = len(text)

    cleaned: list[tuple[int, str]] = []
    skip_meipi = False
    for line_no, l in enumerate(raw_lines, 1):
        s = l.strip()
        if not s:
            if skip_meipi:
                skip_meipi = False
            cleaned.append((line_no, ""))
            continue

        # Remove marginalia commentary blocks (眉批︰ ... up to blank line or 歌括)
        if s.startswith("眉批"):
            skip_meipi = True
            stats["footnote_lines_removed"] += 1
            if audit is not None:
                audit.log("MEIPI_MARGINALIA_REMOVE", line_no, s, "", "Removed marginalia header '眉批'")
            continue
        if skip_meipi:
            if s.startswith("歌括"):
                skip_meipi = False
            else:
                stats["footnote_lines_removed"] += 1
                if audit is not None:
                    audit.log("MEIPI_MARGINALIA_REMOVE", line_no, s, "", "Removed marginalia commentary line")
                continue

        # Remove editorial index numbers at end of titles like 身瘦不孕（29）
        orig_s = s
        s, n = re.subn(r"（\d{1,3}）$", "", s)
        if n:
            stats["inline_footnote_markers_removed"] += n
            if audit is not None:
                audit.log("EDITORIAL_TITLE_INDEX_REMOVE", line_no, orig_s, s, "Removed trailing editorial index")

        # Remove decorative bullet symbol ◎ at start of lines
        if s.startswith("◎"):
            orig_s = s
            s = s.lstrip("◎").strip()
            stats["garbage_tokens_cleaned"] += 1
            if audit is not None:
                audit.log("DECORATIVE_BULLET_REMOVE", line_no, orig_s, s, "Removed leading decorative '◎'")

        s = normalize_cjk_punct_and_ws(s, stats, audit=audit, line_no=line_no)
        if s:
            cleaned.append((line_no, s))

    cleaned = deduplicate_adjacent_lines(cleaned, stats, audit=audit)  # type: ignore[assignment]
    final_lines = collapse_blank_lines([l for _, l in cleaned])
    out = "\n".join(final_lines) + "\n"
    stats["output_lines"] = len(final_lines)
    stats["output_chars"] = len(out)
    return out, dict(stats)


# ---------------------------------------------------------------------------
# 4. CJK LaTeX cleaning & Heading-Aware Page Merging (P05 & P10 fixed)
# ---------------------------------------------------------------------------

def clean_latex_cjk(
    text: str,
    stats: dict,
    audit: AuditLogger | None = None,
    line_no: int = 0,
) -> str:
    """Clean LaTeX OCR artifacts in Simplified Chinese textbooks."""
    orig = text
    text, n = re.subn(r"\$\s*\\underset\{\\cdot\}\{([^}]+)\}\s*\$", r"\1", text)
    stats["garbage_tokens_cleaned"] += n

    def _unwrap_underset(m):
        inner = re.sub(r"\\underset\{\\cdot\}\{([^}]+)\}", r"\1", m.group(1))
        stats["garbage_tokens_cleaned"] += 1
        return inner

    text = re.sub(r"\$([^$]*\\underset[^$]*)\$", _unwrap_underset, text)
    text, n = re.subn(r"\$?\s*\^?\{?\s*\\circ\s*\}?\s*(?:\\text|\\mathrm)?\{?\s*C\s*\}?\s*\$?", "°C", text)
    stats["garbage_tokens_cleaned"] += n
    text, n = re.subn(r"\$(\d+(?:\.\d+)?°C)", r"\1", text)
    stats["garbage_tokens_cleaned"] += n
    text, n = re.subn(r"\$?\s*\\sim\s*\$?", "~", text)
    stats["garbage_tokens_cleaned"] += n
    text, n = re.subn(r"\$\\underline\{\\text\{([^}]+)\}\}\$", r"\1", text)
    stats["garbage_tokens_cleaned"] += n
    text, n = re.subn(r"\\n", "", text)
    stats["garbage_tokens_cleaned"] += n

    if audit is not None and text != orig:
        audit.log(
            "CLEAN_LATEX_CJK",
            line_no,
            orig[:120],
            text[:120],
            "Cleaned LaTeX OCR artifacts in CJK text",
        )
    return text


_CJK_HEADING_PREFIX_RE = re.compile(
    r"^(?:"
    r"第[一二三四五六七八九十百千万\d]+[章节篇回课堂讲部卷节]"
    r"|上篇|中篇|下篇|总论|各论|绪言|前言|引言|导论|小结|附录[A-Z\d]?|附[：:]|思考[：:]|练习案例"
    r"|【[^】]{1,20}】"
    r"|[一二三四五六七八九十]+、"
    r"|（[一二三四五六七八九十\d]+）"
    r"|\d+\.\d+(?:\.\d+)?\s*[\u4e00-\u9fff]"
    r")"
)


def is_cjk_heading(s: str) -> bool:
    """Detect whether a CJK string starts with (or is) a chapter/section/lesson heading (P05)."""
    st = s.strip()
    if not st:
        return False
    if _CJK_HEADING_PREFIX_RE.match(st):
        return True
    # Short title-only line without clause/terminal punctuation
    if len(st) <= 25 and not re.search(r"[，、。！？；：,.!?;:]", st):
        return True
    return False


def merge_cjk_pages(
    pages: list[str],
    stats: dict,
    audit: AuditLogger | None = None,
) -> list[str]:
    """
    Merge sentences split across page breaks in CJK page-per-line OCR files.
    P05 fix: Never merges when the next page starts with a chapter/section heading OR
    when the previous chunk is itself a standalone short heading.
    """
    if not pages:
        return []
    merged = [pages[0]]
    term_punct = set("。！？；：”’）】》.!?;:")

    for idx, p in enumerate(pages[1:], 2):
        prev = merged[-1]
        prev_ends_punct = bool(prev) and prev[-1] in term_punct
        next_is_heading = is_cjk_heading(p)
        prev_is_short_heading = len(prev) <= 40 and is_cjk_heading(prev)

        if prev and not prev_ends_punct and not next_is_heading and not prev_is_short_heading:
            if audit is not None:
                audit.log(
                    "CJK_PAGE_BREAK_MERGE",
                    idx,
                    f"{prev[-20:]} | {p[:20]}",
                    f"{prev[-20:]}{p[:20]}",
                    f"Merged CJK page {idx} into preceding incomplete sentence",
                )
            merged[-1] = prev + p
            stats["page_break_merges"] += 1
        else:
            merged.append(p)
    return merged


# ---------------------------------------------------------------------------
# 5. Processor: Linh khởi điểm học trung y... (Simplified Chinese)
# ---------------------------------------------------------------------------

def process_linh_khoi_diem(
    text: str,
    audit: AuditLogger | None = None,
) -> tuple[str, dict]:
    stats: dict = defaultdict(int)
    text = nfc(text)
    raw_lines = text.splitlines()
    stats["input_lines"] = len(raw_lines)
    stats["input_chars"] = len(text)

    non_empty = [(i + 1, l.strip()) for i, l in enumerate(raw_lines) if l.strip()]

    start_idx = 0
    for i, (_, p) in enumerate(non_empty):
        if p.startswith("体验——中医入门之道"):
            start_idx = i
            break
    stats["front_matter_pages_removed"] = start_idx
    if audit is not None and start_idx > 0:
        audit.log(
            "FRONT_MATTER_TRIM",
            non_empty[0][0],
            f"{start_idx} front-matter pages before '体验——中医入门之道'",
            "",
            "Removed covers, CIP, prefaces, and TOC",
        )

    end_idx = len(non_empty)
    for i, (_, p) in enumerate(non_empty):
        if p.startswith("附录C"):
            end_idx = i
            break
    stats["back_matter_pages_removed"] = len(non_empty) - end_idx
    if audit is not None and end_idx < len(non_empty):
        audit.log(
            "BACK_MATTER_TRIM",
            non_empty[end_idx][0],
            f"{len(non_empty) - end_idx} back-matter pages starting at '附录C'",
            "",
            "Removed reader testimonials (附录C) and postscript (后记)",
        )

    pages = non_empty[start_idx:end_idx]

    cleaned_pages: list[tuple[int, str]] = []
    for line_no, p in pages:
        if re.fullmatch(r"更多精彩欢迎访问[：:]中医中药论坛[：:]https?[：:]//\S+", p):
            stats["headers_footers_removed"] += 1
            if audit is not None:
                audit.log("WATERMARK_PAGE_REMOVE", line_no, p, "", "Removed pure forum watermark page")
            continue

        orig_p = p
        p, n = re.subn(r"更多精彩欢迎访问[：:]中医中药论坛[：:]https?[：:]//\S+", "", p)
        stats["headers_footers_removed"] += n
        p, n2 = re.subn(r"^——带你体验中医的12堂课", "", p)
        stats["headers_footers_removed"] += n2
        if audit is not None and (n or n2):
            audit.log("RUNNING_HEADER_WATERMARK_REMOVE", line_no, orig_p[:80], p[:80], "Removed forum watermark / running header")

        p = clean_latex_cjk(p, stats, audit=audit, line_no=line_no)

        p, n = re.subn(r"思考§", "思考：", p)
        stats["garbage_tokens_cleaned"] += n
        p, n = re.subn(r"恩考§", "思考：", p)
        stats["garbage_tokens_cleaned"] += n
        p, n = re.subn(r"§", "", p)
        stats["garbage_tokens_cleaned"] += n
        p, n = re.subn(r"[\$\\{}^]", "", p)
        stats["garbage_tokens_cleaned"] += n

        p = normalize_cjk_punct_and_ws(p, stats, audit=audit, line_no=line_no)
        if p:
            cleaned_pages.append((line_no, p))

    cleaned_pages = deduplicate_adjacent_lines(cleaned_pages, stats, audit=audit)  # type: ignore[assignment]
    merged = merge_cjk_pages([p for _, p in cleaned_pages], stats, audit=audit)

    out = "\n\n".join(merged) + "\n"
    stats["output_lines"] = len(merged)
    stats["output_chars"] = len(out)
    return out, dict(stats)


# ---------------------------------------------------------------------------
# 6. Processor: phuong-te-hoc... (Simplified Chinese, P08 table fix!)
# ---------------------------------------------------------------------------

def process_phuong_te_hoc(
    text: str,
    audit: AuditLogger | None = None,
) -> tuple[str, dict]:
    stats: dict = defaultdict(int)
    text = nfc(text)
    raw_lines = text.splitlines()
    stats["input_lines"] = len(raw_lines)
    stats["input_chars"] = len(text)

    non_empty = [(i + 1, l.strip()) for i, l in enumerate(raw_lines) if l.strip()]

    kept_pages: list[tuple[int, str]] = []
    for i, (line_no, p) in enumerate(non_empty):
        if p.startswith("附录1.") or p.startswith("附录1．"):
            stats["back_matter_and_table_appendix_pages_removed"] = len(non_empty) - i
            if audit is not None:
                audit.log(
                    "BACK_MATTER_APPENDIX_TRIM",
                    line_no,
                    f"{len(non_empty) - i} pages starting at '{p[:40]}'",
                    "",
                    "Removed table appendix (附录1), mnemonic songs (附录2), and stroke index",
                )
            break
        if i <= 2 and (p.startswith("高等医药院校教材") or p.startswith("前言") or p.startswith("编写说明")):
            stats["front_matter_pages_removed"] += 1
            if audit is not None:
                audit.log("FRONT_MATTER_TRIM", line_no, p[:80], "", "Removed cover / preface / editorial notes")
            continue
        if p.startswith("目录") or (i <= 10 and p.count(".....") >= 3):
            stats["front_matter_pages_removed"] += 1
            if audit is not None:
                audit.log("TOC_PAGE_REMOVE", line_no, p[:80], "", "Removed TOC page")
            continue
        kept_pages.append((line_no, p))

    # Exact table span patterns (P08 fix: NEVER eat prose after 表2, 表3, 表5!)
    table_patterns = [
        ("表1", r"表1四逆汤和通脉四逆汤鉴别表.*$"),
        ("表2", r"表2小承气汤与厚朴三物汤鉴别表.*?分三服"),
        ("表3", r"表3抵当汤和抵当丸鉴别表.*?24小时后当下血，不下再服"),
        ("表4", r"表4麻黄杏仁甘草石膏汤与麻黄汤鉴别表.*?（以清泄肺热为主）"),
        ("表5", r"表5历代衡量与辩的对照表.*?1\.0355市升"),
    ]

    cleaned_pages: list[tuple[int, str]] = []
    for line_no, p in kept_pages:
        for tname, pat in table_patterns:
            m = re.search(pat, p)
            if m:
                removed = m.group(0)
                p = p[:m.start()] + p[m.end():]
                stats["tables_figures_lines_removed"] += 1
                if audit is not None:
                    audit.log(
                        "EMBEDDED_TABLE_REMOVE",
                        line_no,
                        removed[:100] + ("..." if len(removed) > 100 else ""),
                        "",
                        f"Removed embedded comparison table {tname} while preserving surrounding prose",
                        original_start=m.start(),
                        original_end=m.end(),
                    )

        p = clean_latex_cjk(p, stats, audit=audit, line_no=line_no)

        if any(k in p for k in ("臣药：有两种意义", "佐药：有三种意义", "使药：有两种意义")):
            sup_map = {"1": "①", "2": "②", "3": "③"}
            p = re.sub(r"\$\^\{([123])\}\$", lambda m: sup_map[m.group(1)], p)

        def _sub_sup(m):
            stats["inline_footnote_markers_removed"] += 1
            if audit is not None:
                audit.log("INLINE_CITATION_SUP_REMOVE", line_no, m.group(0), "", "Removed inline LaTeX superscript citation")
            return ""

        p = re.sub(r"\$\s*\^\{\s*\d{1,2}\s*\}\s*\$", _sub_sup, p)
        p, n = re.subn(r"[\$\\{}^]", "", p)
        stats["garbage_tokens_cleaned"] += n

        p = normalize_cjk_punct_and_ws(p, stats, audit=audit, line_no=line_no)
        if p:
            cleaned_pages.append((line_no, p))

    cleaned_pages = deduplicate_adjacent_lines(cleaned_pages, stats, audit=audit)  # type: ignore[assignment]
    merged = merge_cjk_pages([p for _, p in cleaned_pages], stats, audit=audit)

    out = "\n\n".join(merged) + "\n"
    stats["output_lines"] = len(merged)
    stats["output_chars"] = len(out)
    return out, dict(stats)


# ---------------------------------------------------------------------------
# 7. Processor: Tân_san_Nam_dược_thần_hiệu_thập_khoa_ứng_trị_q{5,6,8,9,10}
#    (P03 fixed: replace Bopomofo/Kana with [MASK] + NEEDS_REVIEW audit log!)
# ---------------------------------------------------------------------------

def _cjk_to_int(s: str) -> int | None:
    if not s:
        return None
    if s.isdigit():
        return int(s)
    digits = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    if len(s) == 1 and s in digits:
        return digits[s]
    if s == "十":
        return 10
    if len(s) == 2:
        if s[0] == "十" and s[1] in digits:
            return 10 + digits[s[1]]
        if s[0] in digits and s[1] == "十":
            return digits[s[0]] * 10
        if s[0] in digits and s[1] in digits:
            return digits[s[0]] * 10 + digits[s[1]]
    if len(s) == 3 and s[1] == "十" and s[0] in digits and s[2] in digits:
        return digits[s[0]] * 10 + digits[s[2]]
    if re.fullmatch(r"[一二三四五六七八九十]\d{1,2}", s):
        return 999
    return None


def strip_woodblock_folio_prefix(
    page: str,
    page_idx: int,
    stats: dict,
    audit: AuditLogger | None = None,
    line_no: int = 0,
) -> str:
    if re.match(r"^(?:一[經经傳传方治法小肾径仗佈]|三開|五疸|四肢|九阴)", page):
        return page

    m_mix = re.match(r"^([一二三四五六七八九十]\d{1,2})", page)
    if m_mix:
        stats["headers_footers_removed"] += 1
        if audit is not None:
            audit.log(
                "WOODBLOCK_FOLIO_REMOVE",
                line_no,
                page[:20],
                page[m_mix.end():20 + m_mix.end()],
                f"Removed corrupted woodblock folio prefix '{m_mix.group(1)}'",
                original_start=0,
                original_end=m_mix.end(),
            )
        return page[m_mix.end():]

    m = re.match(r"^([一二三四五六七八九十]{1,3})", page)
    if m:
        num_str = m.group(1)
        val = _cjk_to_int(num_str)
        expected_folio = (page_idx + 1) // 2
        if val is not None and abs(val - expected_folio) <= 4:
            rest = page[m.end():]
            if not re.match(r"^[兩両錢分斤升合碗盞杯盃枚粒片丸個个日次服歲月年]", rest):
                stats["headers_footers_removed"] += 1
                if audit is not None:
                    audit.log(
                        "WOODBLOCK_FOLIO_REMOVE",
                        line_no,
                        page[:20],
                        rest[:20],
                        f"Removed woodblock folio prefix '{num_str}' (expected ~{expected_folio})",
                        original_start=0,
                        original_end=m.end(),
                    )
                return rest
    return page


def replace_unreadable_ocr_with_mask(
    page: str,
    stats: dict,
    audit: AuditLogger | None = None,
    line_no: int = 0,
) -> str:
    """
    P03 fix: Replace unreadable Bopomofo/Kana OCR characters (e.g. '每服一ㄐ', '每服二ㄔ')
    with '[MASK]' instead of silently deleting them, and record each original char in audit
    with review_status='NEEDS_REVIEW' for verification against the woodblock scan.
    """
    matches = list(BOPOMOFO_KANA_RE.finditer(page))
    if not matches:
        return page

    for m in matches:
        ch = m.group(0)
        ctx_before = page[max(0, m.start() - 8):min(len(page), m.end() + 8)]
        ctx_after = (
            page[max(0, m.start() - 8):m.start()]
            + "[MASK]"
            + page[m.end():min(len(page), m.end() + 8)]
        )
        stats["unreadable_ocr_masked"] += 1
        if audit is not None:
            audit.log(
                rule_id="P03_OCR_UNREADABLE_MASK",
                original_line=line_no,
                original_start=m.start(),
                original_end=m.end(),
                before=ctx_before,
                after=ctx_after,
                reason=f"Replaced unreadable OCR char '{ch}' (U+{ord(ch):04X}) with [MASK]; possible measurement unit",
                review_status="NEEDS_REVIEW",
            )

    return BOPOMOFO_KANA_RE.sub("[MASK]", page)


def process_tan_san_nam_duoc(
    text: str,
    filename: str = "",
    audit: AuditLogger | None = None,
) -> tuple[str, dict]:
    stats: dict = defaultdict(int)
    text = nfc(text)
    raw_lines = text.splitlines()
    stats["input_lines"] = len(raw_lines)
    stats["input_chars"] = len(text)

    non_empty = [(i + 1, l.strip()) for i, l in enumerate(raw_lines) if l.strip()]

    cleaned_pages: list[tuple[int, str]] = []
    for idx, (line_no, p) in enumerate(non_empty):
        if idx == 0 and re.fullmatch(r"卷[一二三四五六七八九十拾捌玖伍陸]", p):
            stats["front_matter_pages_removed"] += 1
            if audit is not None:
                audit.log("FRONT_MATTER_COVER_REMOVE", line_no, p, "", "Removed cover-only volume title page")
            continue
        if idx <= 2 and p.startswith("卷九小兒科") and "新刊" not in p:
            stats["front_matter_pages_removed"] += 1
            if audit is not None:
                audit.log("TOC_PAGE_REMOVE", line_no, p[:80], "", "Removed q9 disease-list TOC page")
            continue

        p = strip_woodblock_folio_prefix(p, idx, stats, audit=audit, line_no=line_no)
        # P03 fix: mask unreadable Bopomofo/Kana instead of deleting
        p = replace_unreadable_ocr_with_mask(p, stats, audit=audit, line_no=line_no)
        p = normalize_cjk_punct_and_ws(p, stats, audit=audit, line_no=line_no)
        if p:
            cleaned_pages.append((line_no, p))

    cleaned_pages = deduplicate_adjacent_lines(cleaned_pages, stats, audit=audit)  # type: ignore[assignment]

    merged_text = ""
    for i, (line_no, p) in enumerate(cleaned_pages):
        if i == 0:
            merged_text = p
        else:
            if (
                re.match(r"^(?:一[經经傳传方治法]|又[經经傳传]?方)", p)
                and merged_text[-1] in "效愈止消散妙安瘥良平。"
            ):
                merged_text += "\n\n" + p
            else:
                if audit is not None:
                    audit.log(
                        "WOODBLOCK_PAGE_BREAK_MERGE",
                        line_no,
                        f"{merged_text[-15:]} | {p[:15]}",
                        f"{merged_text[-15:]}{p[:15]}",
                        f"Merged woodblock page (L{line_no}) across page break",
                    )
                merged_text += p
                stats["page_break_merges"] += 1

    out = merged_text.strip() + "\n"
    stats["output_lines"] = len([l for l in out.splitlines() if l.strip()])
    stats["output_chars"] = len(out)
    return out, dict(stats)


# ---------------------------------------------------------------------------
# Sentence Segmentation with 4+4 Unicode Context Logging (P07)
# ---------------------------------------------------------------------------

_VN_ABBREV_RE = re.compile(
    r"(?:\b(?:Tp|Tr|Nxb|NXB|Gs|GS|Ts|TS|Bs|BS|Ths|ThS|Mr|Ms|Dr|St|v\.v|etc|đ/c|P|Q|H|X|T)|"
    r"\b[IVXLCDM]+|"
    r"(?:^|\s)\d{1,3}|"
    r"(?:^|\s)[A-Za-zÀ-ỹĐđ])\.$"
)


def segment_sentences(
    text: str,
    lang: str,
) -> tuple[list[dict], list[dict]]:
    """
    Segment cleaned text into sentences and record 4+4 Unicode context at every split boundary (P07).
    Returns:
      - sentences: list of {"sent_id": int, "start": int, "end": int, "text": str}
      - boundaries: list of {"split_id": int, "offset": int, "prev_4": str, "next_4": str, "reason": str}
    """
    sentences: list[dict] = []
    boundaries: list[dict] = []
    n = len(text)
    if n == 0:
        return sentences, boundaries

    split_points: list[tuple[int, str]] = []  # (offset_after_prev_sent, reason)

    if lang == "cjk":
        cjk_end = set("。！？；")
        closers = set("”’』」）】》\"'")
        i = 0
        while i < n:
            ch = text[i]
            if ch == "\n":
                # Paragraph break
                j = i
                while j < n and text[j] in "\r\n":
                    j += 1
                split_points.append((i, "PARAGRAPH_BREAK"))
                i = j
                continue
            if ch in cjk_end:
                end_pos = i + 1
                while end_pos < n and text[end_pos] in closers:
                    end_pos += 1
                # Only record split if there is more non-whitespace text ahead
                if end_pos < n and text[end_pos] not in "\r\n":
                    split_points.append((end_pos, f"CJK_TERMINAL_{ch}"))
                i = end_pos
                continue
            i += 1
    else:
        # Vietnamese sentence segmentation
        closers = set("\"'”’)]")
        i = 0
        while i < n:
            ch = text[i]
            if ch == "\n":
                j = i
                while j < n and text[j] in "\r\n":
                    j += 1
                split_points.append((i, "LINE_OR_PARAGRAPH_BREAK"))
                i = j
                continue
            if ch in ".?!;":
                # Protect decimals/version numbers: digit.digit
                if ch == "." and i > 0 and i + 1 < n and text[i - 1].isdigit() and text[i + 1].isdigit():
                    i += 1
                    continue
                # Protect ellipsis (...)
                if ch == "." and ((i + 1 < n and text[i + 1] == ".") or (i > 0 and text[i - 1] == ".")):
                    i += 1
                    continue
                # Protect abbreviations & numbered list markers like "1.", "I.", "Tp."
                if ch == ".":
                    prefix = text[max(0, i - 12):i + 1]
                    if _VN_ABBREV_RE.search(prefix):
                        i += 1
                        continue
                end_pos = i + 1
                while end_pos < n and text[end_pos] in closers:
                    end_pos += 1
                # Require whitespace before next sentence (avoid breaking URLs/abbrevs)
                if end_pos < n and text[end_pos] in " \t":
                    next_non_ws = end_pos
                    while next_non_ws < n and text[next_non_ws] in " \t":
                        next_non_ws += 1
                    if next_non_ws < n and text[next_non_ws] not in "\r\n":
                        # For ';', only split if followed by a substantial clause (>= 8 chars)
                        if ch != ";" or (n - next_non_ws >= 8):
                            split_points.append((end_pos, f"VN_TERMINAL_{ch}"))
                i = end_pos
                continue
            i += 1

    # Build sentences & 4+4 boundary records
    prev_start = 0
    sent_id = 1
    split_id = 1
    for offset, reason in split_points:
        raw_chunk = text[prev_start:offset]
        stripped = raw_chunk.strip()
        if stripped:
            lead_ws = len(raw_chunk) - len(raw_chunk.lstrip())
            s_start = prev_start + lead_ws
            s_end = s_start + len(stripped)
            sentences.append({
                "sent_id": sent_id,
                "start": s_start,
                "end": s_end,
                "text": stripped,
            })
            sent_id += 1

            # Compute 4 Unicode characters immediately before & after split (skipping newlines/spaces at boundary)
            left_clean = text[:offset].rstrip(" \t\r\n")
            right_clean = text[offset:].lstrip(" \t\r\n")
            if left_clean and right_clean:
                boundaries.append({
                    "split_id": split_id,
                    "offset": offset,
                    "prev_4": left_clean[-4:],
                    "next_4": right_clean[:4],
                    "reason": reason,
                })
                split_id += 1
        prev_start = offset

    # Final trailing sentence
    tail = text[prev_start:]
    stripped_tail = tail.strip()
    if stripped_tail:
        lead_ws = len(tail) - len(tail.lstrip())
        s_start = prev_start + lead_ws
        sentences.append({
            "sent_id": sent_id,
            "start": s_start,
            "end": s_start + len(stripped_tail),
            "text": stripped_tail,
        })

    return sentences, boundaries


# ---------------------------------------------------------------------------
# Independent Output Verification (P11)
# ---------------------------------------------------------------------------

def verify_output(raw: str, refined: str, lang: str, filename: str) -> dict:
    """Run independent verification checks comparing refined output against raw input."""
    checks: dict = {
        "file": filename,
        "lang": lang,
        "changed": refined != raw,
        "retention_ratio": round(len(refined) / max(1, len(raw)), 4),
        "issues": [],
    }
    if lang == "vi":
        han_left = len(HAN_NOM_RE.findall(refined))
        checks["han_nom_remaining"] = han_left
        if han_left > 0:
            checks["issues"].append(f"Found {han_left} remaining Han-Nom spans in VN file")
    else:
        vn_left = len(VN_DIAC_WORD_RE.findall(refined))
        checks["vn_words_remaining"] = vn_left
        if vn_left > 0:
            checks["issues"].append(f"Found {vn_left} remaining VN diacritic words in CJK file")

    bopo_raw = len(BOPOMOFO_KANA_RE.findall(nfc(raw)))
    bopo_ref = len(BOPOMOFO_KANA_RE.findall(refined))
    mask_count = refined.count("[MASK]")
    checks["bopomofo_kana_raw"] = bopo_raw
    checks["bopomofo_kana_remaining"] = bopo_ref
    checks["mask_tokens"] = mask_count
    if bopo_ref > 0:
        checks["issues"].append(f"Found {bopo_ref} unmasked Bopomofo/Kana chars")
    if bopo_raw > 0 and mask_count < bopo_raw:
        checks["issues"].append(f"Expected >= {bopo_raw} [MASK] tokens, got {mask_count}")

    latex_left = len(re.findall(r"[\$\\{}^]", refined))
    checks["latex_symbols_remaining"] = latex_left
    if latex_left > 0:
        checks["issues"].append(f"Found {latex_left} remaining LaTeX/garbage symbols")

    urls_left = len(re.findall(r"vnras\.com|bbs\.zhong-yao\.net", refined, re.I))
    checks["watermark_urls_remaining"] = urls_left
    if urls_left > 0:
        checks["issues"].append(f"Found {urls_left} remaining watermark URLs")

    checks["status"] = "PASS" if not checks["issues"] else "WARN"
    return checks


# ---------------------------------------------------------------------------
# Dispatch & Importable Pipeline Runner (P11)
# ---------------------------------------------------------------------------

def detect_file_lang(fname: str) -> str:
    if fname in ("nam-duoc-than-hieu-pdf.txt", "nam_khoa_nu_khoa.txt"):
        return "vi"
    return "cjk"


def process_file_content(
    fname: str,
    raw: str,
    audit: AuditLogger | None = None,
) -> tuple[str, dict]:
    if fname == "nam-duoc-than-hieu-pdf.txt":
        return process_nam_duoc_than_hieu(raw, audit=audit)
    elif fname == "nam_khoa_nu_khoa.txt":
        return process_nam_khoa_nu_khoa(raw, audit=audit)
    elif fname == "Pho_Thanh_Chu_Nu_Khoa.txt":
        return process_pho_thanh_chu(raw, audit=audit)
    elif fname.startswith("Linh khởi điểm"):
        return process_linh_khoi_diem(raw, audit=audit)
    elif fname.startswith("phuong-te-hoc"):
        return process_phuong_te_hoc(raw, audit=audit)
    elif fname.startswith("Tân_san_Nam_dược"):
        return process_tan_san_nam_duoc(raw, fname, audit=audit)
    else:
        raise ValueError(f"Unhandled file: {fname}")


def run_pipeline(
    input_dir: str | Path = DEFAULT_INPUT_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    report_dir: str | Path = DEFAULT_REPORT_DIR,
    sentences_dir: str | Path | None = DEFAULT_SENTENCES_DIR,
    only_write_modified: bool = True,
) -> dict:
    """
    Run the full cleaning, auditing, verification, and sentence segmentation pipeline.
    Importable for local or Kaggle execution (P11).
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    report_path = Path(report_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    report_path.mkdir(parents=True, exist_ok=True)
    if sentences_dir is not None:
        Path(sentences_dir).mkdir(parents=True, exist_ok=True)

    files = sorted(f for f in os.listdir(input_path) if f.endswith(".txt"))
    report: dict = {}
    verification: dict = {}
    all_audit_records: list[dict] = []
    all_review_records: list[dict] = []
    seg_log_lines: list[str] = []
    written_files: list[str] = []
    skipped_files: list[str] = []

    for fname in files:
        in_file = input_path / fname
        out_file = output_path / fname
        raw = in_file.read_text(encoding="utf-8")

        audit = AuditLogger(filename=fname)
        refined, stats = process_file_content(fname, raw, audit=audit)
        lang = detect_file_lang(fname)

        # P11: Only write files that actually changed
        changed = refined != raw
        stats["file_modified"] = changed
        stats["audit_records"] = len(audit.records)
        stats["review_items"] = len(audit.review_items())

        if changed or not only_write_modified:
            out_file.write_text(refined, encoding="utf-8", newline="\n")
            written_files.append(fname)
        else:
            skipped_files.append(fname)

        # Independent verification
        ver = verify_output(raw, refined, lang, fname)
        verification[fname] = ver

        # P07: Sentence segmentation + 4+4 context log
        if sentences_dir is not None:
            sents, bounds = segment_sentences(refined, lang)
            stats["sentences_count"] = len(sents)
            stats["split_boundaries_count"] = len(bounds)
            stem = Path(fname).stem
            sent_txt_path = Path(sentences_dir) / f"{stem}_sentences.txt"
            sent_jsonl_path = Path(sentences_dir) / f"{stem}_sentences.jsonl"
            sent_txt_path.write_text(
                "\n".join(s["text"] for s in sents) + ("\n" if sents else ""),
                encoding="utf-8",
                newline="\n",
            )
            with sent_jsonl_path.open("w", encoding="utf-8", newline="\n") as jf:
                for s in sents:
                    jf.write(json.dumps(s, ensure_ascii=False) + "\n")

            seg_log_lines.append(f"=== {fname} ({lang}, {len(sents)} sentences, {len(bounds)} splits) ===")
            for b in bounds:
                seg_log_lines.append(
                    f"  #{b['split_id']} offset={b['offset']} [{b['reason']}] "
                    f"prev4={b['prev_4']!r} | next4={b['next_4']!r}"
                )

        all_audit_records.extend(audit.to_dicts())
        all_review_records.extend(asdict(r) for r in audit.review_items())
        report[fname] = stats

    # Write JSON stats + verification
    (report_path / "refinement_stats.json").write_text(
        json.dumps(
            {
                "files": report,
                "verification": verification,
                "written_files": written_files,
                "skipped_unchanged_files": skipped_files,
                "total_audit_records": len(all_audit_records),
                "total_review_items": len(all_review_records),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
        newline="\n",
    )

    # Write per-change audit log (JSONL) & review queue (JSONL)
    with (report_path / "audit_changes.jsonl").open("w", encoding="utf-8", newline="\n") as af:
        for rec in all_audit_records:
            af.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with (report_path / "review_queue.jsonl").open("w", encoding="utf-8", newline="\n") as rf:
        for rec in all_review_records:
            rf.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Write segmentation 4+4 boundary log
    if sentences_dir is not None:
        (report_path / "segmentation_changes.txt").write_text(
            "\n".join(seg_log_lines) + ("\n" if seg_log_lines else ""),
            encoding="utf-8",
            newline="\n",
        )

    return {
        "files": report,
        "verification": verification,
        "written_files": written_files,
        "skipped_unchanged_files": skipped_files,
        "total_audit_records": len(all_audit_records),
        "total_review_items": len(all_review_records),
    }


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Clean & segment Han-Nom / Vietnamese TCM texts.")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR), help="Input folder containing raw .txt files")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output folder for cleaned .txt files")
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR), help="Output folder for stats & audit logs")
    parser.add_argument("--sentences-dir", default=str(DEFAULT_SENTENCES_DIR), help="Output folder for segmented sentences")
    parser.add_argument("--no-sentences", action="store_true", help="Disable sentence segmentation output")
    parser.add_argument("--write-unchanged", action="store_true", help="Write output even if file is unchanged")
    args = parser.parse_args(argv)

    res = run_pipeline(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        report_dir=args.report_dir,
        sentences_dir=None if args.no_sentences else args.sentences_dir,
        only_write_modified=not args.write_unchanged,
    )
    for fname, st in res["files"].items():
        ver = res["verification"][fname]["status"]
        print(
            f"[{ver}] {fname}: {st['input_chars']:,} -> {st['output_chars']:,} chars | "
            f"{st['input_lines']} -> {st['output_lines']} lines | "
            f"audit={st['audit_records']}, review={st['review_items']}"
        )
    print(
        f"\nWritten: {len(res['written_files'])}, Skipped unchanged: {len(res['skipped_unchanged_files'])}, "
        f"Total audit records: {res['total_audit_records']:,}, Review items: {res['total_review_items']}"
    )


if __name__ == "__main__":
    main()
