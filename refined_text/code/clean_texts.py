#!/usr/bin/env python3
"""
Text Refinement Pipeline for ./text -> ./refined_text

Implements all required NLP preprocessing & text-cleaning steps while preserving
the integrity of the main content, proper nouns, era names, numbers, and units:
  1. Remove headers, footers, page numbers, watermarks, and non-main-content pages
     (covers, CIP/copyright, prefaces/colophons, tables of contents, indices).
  2. Merge words/sentences broken across page breaks and line wraps.
  3. Remove duplicate lines caused by OCR overlapping regions.
  4. Remove footnotes/marginalia outside main content and inline footnote markers
     such as [1], (1), $^{1}$, $^{(1)}$, etc.
  5. Fix garbage characters (LaTeX/Markdown OCR artifacts, stray symbols, Unicode
     decomposed accents) and abnormal whitespace.
  6. Normalize punctuation without changing meaning.
  7. Preserve proper nouns, era names (niên hiệu), numbers, and measurement units.
  8. Skip tables and image/figure plates.
"""

import difflib
import json
import os
import re
import unicodedata
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "text")
OUTPUT_DIR = os.path.join(BASE_DIR, "refined_text")


# ---------------------------------------------------------------------------
# Common utility helpers
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


def strip_han_nom_in_vn(text: str, stats: dict) -> str:
    """Remove Han-Nom / CJK characters from Vietnamese-dominant text."""
    text, n = HAN_NOM_RE.subn("", text)
    if n:
        stats["cross_script_tokens_removed"] += n
        # Clean empty parentheses/brackets or stray punctuation left behind
        text = re.sub(r"\(\s*\)", "", text)
        text = re.sub(r"\[\s*\]", "", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = text.strip()
        # If line became only punctuation/symbols (e.g. "—" or ":" or "."), drop it
        if re.fullmatch(r"[\s\-–—•·.,;:!?()\[\]\"']*", text):
            return ""
    return text


def strip_vietnamese_in_cjk(text: str, stats: dict) -> str:
    """Remove Vietnamese words from Han-Nom / Chinese-dominant text (preserving units like g, ml, kg, °C)."""
    text, n = VN_DIAC_WORD_RE.subn("", text)
    if n:
        stats["cross_script_tokens_removed"] += n
        text = re.sub(r"[ \t]{2,}", " ", text).strip()
    return text


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def clean_latex_and_markdown_vn(text: str, stats: dict) -> str:
    """Clean LaTeX / Markdown OCR artifacts in Vietnamese texts."""
    # Remove LaTeX superscript footnote markers: $ ^{3} $, $ ^{(1)} $, $^{2}$, etc.
    def _sub_sup(m):
        stats["inline_footnote_markers_removed"] += 1
        return ""

    text = re.sub(r"\s*\$\s*\^\{\s*\(?\s*\d{1,3}\s*\)?\s*\}\s*\$", _sub_sup, text)

    # Remove stray LaTeX superscript dots/backticks: $ ^{·} $, $ ^{｀} $
    text, n = re.subn(r"\s*\$\s*\^\{\s*[·｀`\'*]\s*\}\s*\$", "", text)
    stats["garbage_tokens_cleaned"] += n

    # Unwrap LaTeX text wrappers: $ \textit{...} $, $ \textbf{...} $, $ \text{...} $
    text, n = re.subn(r"\$\s*\\text(?:it|bf|rm|sf|tt)?\{([^}]*)\}\s*\$", r"\1", text)
    stats["garbage_tokens_cleaned"] += n

    # Unwrap unclosed $ \text{... (e.g. L1122 in NDTH)
    text, n = re.subn(r"\$\s*\\text(?:it|bf|rm)?\{", "", text)
    stats["garbage_tokens_cleaned"] += n

    # Remove leading backslash on lines (e.g. \Dùng gừng..., \*Đất lòng bếp...*)
    text, n = re.subn(r"^\\+\*?", "", text)
    stats["garbage_tokens_cleaned"] += n

    # Remove markdown bold/italic wrappers (*...* or **...**)
    text, n = re.subn(r"\*\*([^*]+)\*\*", r"\1", text)
    stats["garbage_tokens_cleaned"] += n
    text, n = re.subn(r"\*([^*\n]+)\*", r"\1", text)
    stats["garbage_tokens_cleaned"] += n

    # Remove remaining stray markdown asterisks, braces, backslashes, carets
    # Preserve $ only if part of currency like 1.450 $ (though colophon is removed anyway)
    text, n = re.subn(r"(?<!\d)\*(?!\d)", "", text)
    stats["garbage_tokens_cleaned"] += n
    text, n = re.subn(r"[\\{}^]", "", text)
    stats["garbage_tokens_cleaned"] += n
    text, n = re.subn(r"\$\s*(?![\d])", "", text)
    stats["garbage_tokens_cleaned"] += n

    # Replace OCR table pipes in herb lines: |富歸|ĐƯƠNG QUY|2 dòng -> 富歸 ĐƯƠNG QUY 2 dòng
    if "|" in text:
        text, n = re.subn(r"\s*\|\s*", " ", text)
        stats["garbage_tokens_cleaned"] += n

    return text


def normalize_punctuation_vn(text: str, stats: dict) -> str:
    """Normalize punctuation and whitespace in Vietnamese text without altering meaning."""
    orig = text

    # Normalize guillemets and CJK brackets/punctuation inside Vietnamese prose
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

    # Normalize multiple dots (4+ dots -> ...)
    text = re.sub(r"\.{4,}", "...", text)

    # Collapse multiple spaces/tabs
    text = re.sub(r"[ \t]{2,}", " ", text)

    # Fix spacing around punctuation: no space before , . ; : ! ? ) ]
    text = re.sub(r"\s+([,.;:!?)\]])", r"\1", text)
    # No space right after ( [
    text = re.sub(r"([(\[])\s+", r"\1", text)

    # Ensure space after , ; : ! ? when followed by a letter (not digit, to preserve 1,5 / 2:3)
    text = re.sub(r"([,;:])([A-Za-zÀ-ỹĐđ])", r"\1 \2", text)
    # Ensure space after period when followed by an uppercase letter (not inside acronyms/numbers)
    text = re.sub(r"(?<!\.)\.([A-ZÀ-ỸĐ][a-zà-ỹđ])", r". \1", text)

    text = text.strip()
    if text != orig:
        stats["punctuation_ws_normalized_lines"] += 1
    return text


def remove_inline_footnote_markers_vn(text: str, stats: dict) -> str:
    """Remove inline footnote index markers like (1), [1], (12) in Vietnamese prose."""
    def _sub(m):
        stats["inline_footnote_markers_removed"] += 1
        return ""

    # Remove [1]..[99] anywhere
    text = re.sub(r"\s*\[\d{1,2}\]", _sub, text)

    # Remove (1)..(99) when it appears inline (NOT at the very start of a line,
    # and NOT preceded by 'số ' or 'trang ' or 'mục ', and NOT followed by ' ngày'/' lần'/etc.)
    text = re.sub(
        r"(?<!^)(?<!số )(?<!Số )(?<!trang )\s*\(\d{1,2}\)(?!\s*(?:ngày|tháng|năm|lần|thang|chén|bát))",
        _sub,
        text,
    )
    return text


def deduplicate_adjacent_lines(lines: list[str], stats: dict, threshold: float = 0.92) -> list[str]:
    """Remove exact or near-duplicate adjacent lines caused by OCR overlap."""
    out = []
    for line in lines:
        s = line.strip()
        if not s:
            out.append("")
            continue
        # Compare against last non-empty line within recent window
        is_dup = False
        for prev in reversed(out[-3:]):
            ps = prev.strip()
            if not ps:
                continue
            if s == ps and len(s) >= 6:
                is_dup = True
                break
            if len(s) >= 25 and len(ps) >= 25:
                ratio = difflib.SequenceMatcher(None, s, ps).ratio()
                if ratio >= threshold:
                    is_dup = True
                    break
        if is_dup:
            stats["duplicate_lines_removed"] += 1
        else:
            out.append(line)
    return out


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
    # Pattern 1: starts with (1), (2), ... (99) (allowing optional leading * or \)
    if re.match(r"^[\\*]*\(\d{1,2}\)\s*\S", s):
        return True
    # Pattern 2: editorial footnotes starting with '1. ', '2. ' (distinguished from
    # numbered herbs/prescriptions by editorial markers)
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
    return False


def process_nam_duoc_than_hieu(text: str) -> tuple[str, dict]:
    stats = defaultdict(int)
    text = nfc(text)
    raw_lines = text.splitlines()
    stats["input_lines"] = len(raw_lines)
    stats["input_chars"] = len(text)

    # 1. Trim front matter (L1..L75: cover, publisher, URL, temple photo, LỜI NÓI ĐẦU,
    #    TIỂU SỬ, LỜI TỰA, PHÀM LỆ) -> main content starts at L76 ("NAM DƯỢC THẦN HIỆU" / "QUYỂN ĐẦU")
    start_idx = 0
    for i, l in enumerate(raw_lines):
        if l.strip() == "QUYỂN ĐẦU":
            start_idx = max(0, i - 2)  # keep "NAM DƯỢC THẦN HIỆU" / "Muời khoa chữa bệnh"
            break
    stats["front_matter_lines_removed"] = start_idx

    # 2. Trim back matter (MỤC LỤC at L11760 to end of file, including colophon & back cover)
    end_idx = len(raw_lines)
    for i in range(len(raw_lines) - 1, start_idx, -1):
        if raw_lines[i].strip() == "MỤC LỤC":
            end_idx = i
            break
    stats["back_matter_lines_removed"] = len(raw_lines) - end_idx

    lines = raw_lines[start_idx:end_idx]

    # 3. Filter out internal TOC ("THƯƠNG HÀN CÁCH PHÁP TRỊ LỆ MỤC LỤC"),
    #    herb illustration plate blocks ("Bảng II/IV/VI/VII"), URLs, page numbers, footnotes
    filtered = []
    skip_plate = False
    skip_internal_toc = False

    for l in lines:
        s = l.strip()

        # Internal TOC: THƯƠNG HÀN CÁCH PHÁP TRỊ LỆ MỤC LỤC (37 numbered titles)
        if s == "THƯƠNG HÀN CÁCH PHÁP TRỊ LỆ MỤC LỤC":
            skip_internal_toc = True
            stats["non_main_content_lines_removed"] += 1
            continue
        if skip_internal_toc:
            # Ends when actual prescription I / bài 1 starts (or heading after 37.)
            if re.match(r"^(?:I\b|QUYỂN|THƯƠNG HÀN CÁCH PHÁP TRỊ LỆ$|Bài\s+1\b|1\)\s)", s) and "MỤC LỤC" not in s:
                skip_internal_toc = False
            elif re.match(r"^37\.\s", s):
                # Last item of internal TOC
                skip_internal_toc = False
                stats["non_main_content_lines_removed"] += 1
                continue
            else:
                stats["non_main_content_lines_removed"] += 1
                continue

        # Watermark URLs
        if re.search(r"https?://vnras\.com", s, re.I):
            stats["headers_footers_removed"] += 1
            continue

        # Image plate blocks: "Bảng II", "Bảng IV", "Bảng VI", "Bảng VII" + short herb captions until blank line
        if re.match(r"^Bảng\s+[IVX]+$", s):
            skip_plate = True
            stats["tables_figures_lines_removed"] += 1
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

        # Standalone page numbers
        if re.match(r"^\d{1,4}$", s):
            stats["headers_footers_removed"] += 1
            continue

        # Footnote lines at page bottoms
        if is_ndth_footnote_line(s):
            stats["footnote_lines_removed"] += 1
            continue

        filtered.append(l)

    # 4. Clean LaTeX/Markdown artifacts, inline footnote markers, punctuation & whitespace
    cleaned = []
    for l in filtered:
        s = l.strip()
        if not s:
            cleaned.append("")
            continue
        s = clean_latex_and_markdown_vn(s, stats)
        s = remove_inline_footnote_markers_vn(s, stats)
        s = strip_han_nom_in_vn(s, stats)
        if not s:
            continue
        s = normalize_punctuation_vn(s, stats)
        if s:
            cleaned.append(s)

    # 5. Deduplicate adjacent OCR duplicate lines
    cleaned = deduplicate_adjacent_lines(cleaned, stats)

    # 6. Merge sentences broken across page breaks / line breaks
    #    Only merge when previous line does NOT end with terminal punctuation,
    #    is long enough to be prose (>= 45 chars), and next line starts with a lowercase letter
    #    (preserving poems, headings, and numbered items).
    merged = []
    i = 0
    while i < len(cleaned):
        cur = cleaned[i]
        if not cur:
            merged.append("")
            i += 1
            continue
        while i + 1 < len(cleaned):
            # Peek next non-empty line if separated by at most 1 blank line (page break artifact)
            nxt_idx = i + 1
            if nxt_idx < len(cleaned) and not cleaned[nxt_idx] and nxt_idx + 1 < len(cleaned):
                cand = cleaned[nxt_idx + 1]
                # Only bridge across a blank line if cur ends mid-sentence (no terminal punct)
                # AND cand starts with lowercase Vietnamese letter
                if (
                    len(cur) >= 45
                    and not is_vn_heading_or_item(cur)
                    and cur[-1] not in ".!?:;)\""
                    and cand
                    and cand[0].islower()
                    and not is_vn_heading_or_item(cand)
                ):
                    cur = cur + " " + cand
                    stats["page_break_merges"] += 1
                    i = nxt_idx + 1
                    continue
                else:
                    break
            elif nxt_idx < len(cleaned) and cleaned[nxt_idx]:
                cand = cleaned[nxt_idx]
                if (
                    len(cur) >= 45
                    and not is_vn_heading_or_item(cur)
                    and cur[-1] not in ".!?:;)\""
                    and cand[0].islower()
                    and not is_vn_heading_or_item(cand)
                ):
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

def process_nam_khoa_nu_khoa(text: str) -> tuple[str, dict]:
    stats = defaultdict(int)
    text = nfc(text)
    raw_lines = text.splitlines()
    stats["input_lines"] = len(raw_lines)
    stats["input_chars"] = len(text)

    # 1. Remove translator credit header line 1 ("DỊCH GIÁ: ĐỊNH NINH - LỄ ĐỨC THIẾP")
    start_idx = 0
    if raw_lines and re.match(r"^DỊCH GI[ẢÁ]:", raw_lines[0].strip()):
        start_idx = 1
        stats["front_matter_lines_removed"] = 1

    # 2. Remove endnotes section ("PHỤ CHÚ về NỮ KHOA của DỊCH GIẢ", L2872..L3218)
    #    and back matter ("MỤC LỤC NỮ KHOA" + publisher colophon, L3219..L3356)
    end_idx = len(raw_lines)
    for i, l in enumerate(raw_lines):
        if l.strip() == "附註女科":
            end_idx = i
            break
    # Also strip trailing "HÉT" right before endnotes
    while end_idx > start_idx and raw_lines[end_idx - 1].strip() in ("", "HÉT", "HẾT", "oOo"):
        end_idx -= 1

    stats["back_matter_and_endnotes_lines_removed"] = len(raw_lines) - end_idx

    lines = raw_lines[start_idx:end_idx]

    # 3. Clean lines
    cleaned = []
    for l in lines:
        s = l.strip()
        if not s:
            cleaned.append("")
            continue
        # Remove decorative dividers (oOo, O Oo, o0o, ---, ***)
        if re.fullmatch(r"(?:[oO0]\s*[oO0]\s*[oO0]|-{3,}|\*{3,})", s):
            stats["headers_footers_removed"] += 1
            continue
        # Remove inline cross-reference to footnotes: (Xem phụ chú Nam Khoa)
        s, n = re.subn(r"\s*\(Xem phụ chú[^)]*\)", "", s, flags=re.I)
        stats["inline_footnote_markers_removed"] += n

        s = clean_latex_and_markdown_vn(s, stats)
        s = remove_inline_footnote_markers_vn(s, stats)
        s = strip_han_nom_in_vn(s, stats)
        if not s:
            continue
        s = normalize_punctuation_vn(s, stats)
        if s:
            cleaned.append(s)

    # 4. Deduplicate adjacent lines
    cleaned = deduplicate_adjacent_lines(cleaned, stats)

    # 5. Merge sentences broken across lines/pages (only prose lines, not Han headings or herb dosage lines)
    merged = []
    i = 0
    while i < len(cleaned):
        cur = cleaned[i]
        if not cur:
            merged.append("")
            i += 1
            continue
        while i + 1 < len(cleaned):
            cand = cleaned[i + 1]
            if (
                cand
                and len(cur) >= 50
                and cur[-1] not in ".!?:;)\""
                and cand[0].islower()
                and not re.match(r"^[\u4e00-\u9fff]", cur)
                and not re.match(r"^[\u4e00-\u9fff]", cand)
            ):
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

def normalize_cjk_punct_and_ws(text: str, stats: dict) -> str:
    orig = text
    # Remove any Vietnamese words in CJK-dominant text
    text = strip_vietnamese_in_cjk(text, stats)
    # Normalize variant CJK punctuation
    text = text.replace("︰", "：").replace("﹗", "！").replace("‧", "·")
    text = re.sub(r"\.{4,}", "……", text)
    # Remove spaces between CJK characters
    text = re.sub(r"(?<=[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef])\s+(?=[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef])", "", text)
    # Collapse remaining multiple spaces
    text = re.sub(r"[ \t]{2,}", " ", text).strip()
    if text != orig:
        stats["punctuation_ws_normalized_lines"] += 1
    return text


def process_pho_thanh_chu(text: str) -> tuple[str, dict]:
    stats = defaultdict(int)
    text = nfc(text)
    raw_lines = text.splitlines()
    stats["input_lines"] = len(raw_lines)
    stats["input_chars"] = len(text)

    cleaned = []
    skip_meipi = False
    for l in raw_lines:
        s = l.strip()
        if not s:
            if skip_meipi:
                skip_meipi = False
            cleaned.append("")
            continue

        # Remove marginalia commentary blocks (眉批︰ ... up to blank line or 歌括)
        if s.startswith("眉批"):
            skip_meipi = True
            stats["footnote_lines_removed"] += 1
            continue
        if skip_meipi:
            if s.startswith("歌括"):
                skip_meipi = False
            else:
                stats["footnote_lines_removed"] += 1
                continue

        # Remove editorial index numbers at end of titles like 身瘦不孕（29）
        s, n = re.subn(r"（\d{1,3}）$", "", s)
        stats["inline_footnote_markers_removed"] += n

        # Remove decorative bullet symbol ◎ at start of lines
        if s.startswith("◎"):
            s = s.lstrip("◎").strip()
            stats["garbage_tokens_cleaned"] += 1

        s = normalize_cjk_punct_and_ws(s, stats)
        if s:
            cleaned.append(s)

    cleaned = deduplicate_adjacent_lines(cleaned, stats)
    cleaned = collapse_blank_lines(cleaned)
    out = "\n".join(cleaned) + "\n"
    stats["output_lines"] = len(cleaned)
    stats["output_chars"] = len(out)
    return out, dict(stats)


# ---------------------------------------------------------------------------
# 4. Processor: Linh khởi điểm học trung y... (Simplified Chinese, 1 page/line)
# ---------------------------------------------------------------------------

def clean_latex_cjk(text: str, stats: dict) -> str:
    """Clean LaTeX OCR artifacts in Simplified Chinese textbooks."""
    # Emphasis dots: $\underset{\cdot}{及}$ -> 及
    text, n = re.subn(r"\$\s*\\underset\{\\cdot\}\{([^}]+)\}\s*\$", r"\1", text)
    stats["garbage_tokens_cleaned"] += n
    # Chained emphasis dots inside one $...$: $\underset{\cdot}{影}\underset{\cdot}{响}$
    def _unwrap_underset(m):
        inner = re.sub(r"\\underset\{\\cdot\}\{([^}]+)\}", r"\1", m.group(1))
        stats["garbage_tokens_cleaned"] += 1
        return inner
    text = re.sub(r"\$([^$]*\\underset[^$]*)\$", _unwrap_underset, text)

    # Temperature: $37^{\circ}C$ -> 37°C, $40°C -> 40°C, 37\circC -> 37°C, $^\circ C$ -> °C
    text, n = re.subn(r"\$?\s*\^?\{?\s*\\circ\s*\}?\s*(?:\\text|\\mathrm)?\{?\s*C\s*\}?\s*\$?", "°C", text)
    stats["garbage_tokens_cleaned"] += n
    text, n = re.subn(r"\$(\d+(?:\.\d+)?°C)", r"\1", text)
    stats["garbage_tokens_cleaned"] += n
    # \sim -> ~
    text, n = re.subn(r"\$?\s*\\sim\s*\$?", "~", text)
    stats["garbage_tokens_cleaned"] += n

    # \underline{\text{...}} -> ...
    text, n = re.subn(r"\$\\underline\{\\text\{([^}]+)\}\}\$", r"\1", text)
    stats["garbage_tokens_cleaned"] += n

    # Literal \n inside OCR strings -> space
    text, n = re.subn(r"\\n", "", text)
    stats["garbage_tokens_cleaned"] += n

    return text


def merge_cjk_pages(pages: list[str], stats: dict) -> list[str]:
    """Merge sentences split across page breaks in CJK page-per-line OCR files."""
    if not pages:
        return []
    merged = [pages[0]]
    term_punct = set("。！？；：”’）】》.!?;:")
    for p in pages[1:]:
        prev = merged[-1]
        # If previous page does not end with sentence-ending punctuation, merge seamlessly
        if prev and prev[-1] not in term_punct:
            merged[-1] = prev + p
            stats["page_break_merges"] += 1
        else:
            merged.append(p)
    return merged


def process_linh_khoi_diem(text: str) -> tuple[str, dict]:
    stats = defaultdict(int)
    text = nfc(text)
    raw_lines = text.splitlines()
    stats["input_lines"] = len(raw_lines)
    stats["input_chars"] = len(text)

    non_empty = [l.strip() for l in raw_lines if l.strip()]

    # 1. Remove front matter (covers, back cover, title, CIP, prefaces, TOC)
    #    Main content starts at "体验——中医入门之道"
    start_idx = 0
    for i, p in enumerate(non_empty):
        if p.startswith("体验——中医入门之道"):
            start_idx = i
            break
    stats["front_matter_pages_removed"] = start_idx

    # 2. Remove back matter (附录C网友学习感言 & 后记)
    end_idx = len(non_empty)
    for i, p in enumerate(non_empty):
        if p.startswith("附录C"):
            end_idx = i
            break
    stats["back_matter_pages_removed"] = len(non_empty) - end_idx

    pages = non_empty[start_idx:end_idx]

    # 3. Clean page-level watermarks, running headers, LaTeX, garbage symbols, whitespace
    cleaned_pages = []
    for p in pages:
        # Skip pure watermark page (e.g., "更多精彩欢迎访问：中医中药论坛：http：//bbs.zhong-yao.net")
        if re.fullmatch(r"更多精彩欢迎访问[：:]中医中药论坛[：:]https?[：:]//\S+", p):
            stats["headers_footers_removed"] += 1
            continue

        # Strip embedded forum watermark & running book header
        p, n = re.subn(r"更多精彩欢迎访问[：:]中医中药论坛[：:]https?[：:]//\S+", "", p)
        stats["headers_footers_removed"] += n
        p, n = re.subn(r"^——带你体验中医的12堂课", "", p)
        stats["headers_footers_removed"] += n

        # Clean LaTeX artifacts
        p = clean_latex_cjk(p, stats)

        # Replace OCR bullet symbol § (used in "思考§..." questions) with clean newline/space or bullet
        # Replace § with a clean fullwidth space or nothing after 思考
        p, n = re.subn(r"思考§", "思考：", p)
        stats["garbage_tokens_cleaned"] += n
        p, n = re.subn(r"恩考§", "思考：", p)
        stats["garbage_tokens_cleaned"] += n
        p, n = re.subn(r"§", "", p)
        stats["garbage_tokens_cleaned"] += n

        # Normalize stray LaTeX chars
        p, n = re.subn(r"[\$\\{}^]", "", p)
        stats["garbage_tokens_cleaned"] += n

        # Normalize CJK punctuation & whitespace
        p = normalize_cjk_punct_and_ws(p, stats)
        if p:
            cleaned_pages.append(p)

    # 4. Deduplicate any repeated pages & merge page-break split sentences
    cleaned_pages = deduplicate_adjacent_lines(cleaned_pages, stats)
    merged = merge_cjk_pages(cleaned_pages, stats)

    out = "\n\n".join(merged) + "\n"
    stats["output_lines"] = len(merged)
    stats["output_chars"] = len(out)
    return out, dict(stats)


# ---------------------------------------------------------------------------
# 5. Processor: phuong-te-hoc... (Simplified Chinese, 1 page/line)
# ---------------------------------------------------------------------------

def process_phuong_te_hoc(text: str) -> tuple[str, dict]:
    stats = defaultdict(int)
    text = nfc(text)
    raw_lines = text.splitlines()
    stats["input_lines"] = len(raw_lines)
    stats["input_chars"] = len(text)

    non_empty = [l.strip() for l in raw_lines if l.strip()]

    # 1. Remove front matter: cover (0), 前言 (1), 编写说明 (2), and 目录 (5..9)
    #    Keep 绪言 (3..4) and 上篇总论 (10..) up to end of main text (before 附录1.常用中成药简表)
    kept_pages = []
    for i, p in enumerate(non_empty):
        # Stop at 附录1 (which starts the tables appendix, followed by 附录2 方歌 and stroke index)
        if p.startswith("附录1.") or p.startswith("附录1．"):
            stats["back_matter_and_table_appendix_pages_removed"] = len(non_empty) - i
            break
        # Skip cover, 前言, 编写说明
        if i <= 2 and (p.startswith("高等医药院校教材") or p.startswith("前言") or p.startswith("编写说明")):
            stats["front_matter_pages_removed"] += 1
            continue
        # Skip TOC pages (目录... and continuation pages ending with page numbers like .....48 / .....226)
        if p.startswith("目录") or (i <= 10 and p.count(".....") >= 3):
            stats["front_matter_pages_removed"] += 1
            continue
        kept_pages.append(p)

    # 2. Clean embedded tables, LaTeX, superscript footnote markers, garbage chars
    cleaned_pages = []
    for p in kept_pages:
        # Remove embedded comparison tables:
        #   - 表1四逆汤和通脉四逆汤鉴别表... (to end of that page)
        if "表1四逆汤和通脉四逆汤鉴别表" in p:
            idx = p.index("表1四逆汤和通脉四逆汤鉴别表")
            # Also trim broken fragment right before it if needed
            p = p[:idx].rstrip("、")
            stats["tables_figures_lines_removed"] += 1
        #   - Page starting with 表2小承气汤与厚朴三物汤鉴别表 ... 表3 ... 表4 ...
        if p.startswith("表2小承气汤与厚朴三物汤鉴别表"):
            # Keep only the prose paragraph between 表2 and 表3 (around "3.2.3" / "（3）剂型更换的变化") if present
            # Let's extract any non-table prose sentences or skip the table portions cleanly:
            # Remove 表2..up to next prose, 表3..up to next prose, 表4..to end
            p = re.sub(r"表2小承气汤与厚朴三物汤鉴别表.*?(?=（[23]）|剂型更换|$)", "", p)
            p = re.sub(r"表3抵当汤和抵当丸鉴别表.*?(?=（[34]）|药量增减|药味加减|$)", "", p)
            p = re.sub(r"表4麻黄杏仁甘草石膏汤与麻黄汤鉴别表.*$", "", p)
            stats["tables_figures_lines_removed"] += 3
        #   - 表5历代衡量与辩的对照表...
        if "表5历代衡量" in p:
            p = re.sub(r"表5历代衡量.*$", "", p)
            stats["tables_figures_lines_removed"] += 1

        # Clean LaTeX emphasis dots, temperatures, \n
        p = clean_latex_cjk(p, stats)

        # Handle LaTeX superscripts $^{1}$, $^{2}$, $^{3}$:
        # - Inside 臣药/佐药/使药 definitions ("有两种意义。$^{1}$辅助..."), convert to ①②③ to keep meaning
        # - After citations/quotes (e.g. ”。$^{1}$), remove as inline footnote markers
        if any(k in p for k in ("臣药：有两种意义", "佐药：有三种意义", "使药：有两种意义")):
            sup_map = {"1": "①", "2": "②", "3": "③"}
            p = re.sub(r"\$\^\{([123])\}\$", lambda m: sup_map[m.group(1)], p)
        def _sub_sup(m):
            stats["inline_footnote_markers_removed"] += 1
            return ""
        p = re.sub(r"\$\s*\^\{\s*\d{1,2}\s*\}\s*\$", _sub_sup, p)

        # Remove remaining stray LaTeX symbols
        p, n = re.subn(r"[\$\\{}^]", "", p)
        stats["garbage_tokens_cleaned"] += n

        # Normalize punctuation & whitespace
        p = normalize_cjk_punct_and_ws(p, stats)
        if p:
            cleaned_pages.append(p)

    cleaned_pages = deduplicate_adjacent_lines(cleaned_pages, stats)
    merged = merge_cjk_pages(cleaned_pages, stats)

    out = "\n\n".join(merged) + "\n"
    stats["output_lines"] = len(merged)
    stats["output_chars"] = len(out)
    return out, dict(stats)


# ---------------------------------------------------------------------------
# 6. Processor: Tân_san_Nam_dược_thần_hiệu_thập_khoa_ứng_trị_q{5,6,8,9,10}
# ---------------------------------------------------------------------------

CJK_NUM = r"[一二三四五六七八九十廿卅百\d]+"

def _cjk_to_int(s: str) -> int | None:
    """Parse a woodblock folio number like 九, 十一, 六四, 四13 into an int if valid."""
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
            # Woodblock shorthand: 六四 = 64, 四三 = 43
            return digits[s[0]] * 10 + digits[s[1]]
    if len(s) == 3 and s[1] == "十" and s[0] in digits and s[2] in digits:
        return digits[s[0]] * 10 + digits[s[2]]
    # Mixed OCR like 四13 -> folio 63 / 43
    if re.fullmatch(r"[一二三四五六七八九十]\d{1,2}", s):
        return 999  # clearly a corrupted folio number
    return None


def strip_woodblock_folio_prefix(page: str, page_idx: int, stats: dict) -> str:
    """
    Remove woodblock centerfold folio numbers at the start of pages (e.g. '六四末酒調服...',
    '十一一治石瘧...', '十二又方...', '四13無漿...').
    Only strips when the leading numeral matches the expected folio range (~page_idx//2)
    and is NOT part of main text like '一經治', '一方', '一傳治', '三開', '五疸'.
    """
    # Never strip recipe/section openers
    if re.match(r"^(?:一[經经傳传方治法法小肾径仗佈]|三開|五疸|四肢|九阴)", page):
        return page

    # Mixed CJK+ASCII digit folio error like 四13
    m_mix = re.match(r"^([一二三四五六七八九十]\d{1,2})", page)
    if m_mix:
        stats["headers_footers_removed"] += 1
        return page[m_mix.end():]

    # Check 1-3 char CJK numeral prefix
    m = re.match(r"^([一二三四五六七八九十]{1,3})", page)
    if m:
        num_str = m.group(1)
        val = _cjk_to_int(num_str)
        expected_folio = (page_idx + 1) // 2
        if val is not None and abs(val - expected_folio) <= 4:
            # Make sure it's not a dosage/count like "三兩" or "十九空心" (19 pills)
            rest = page[m.end():]
            if not re.match(r"^[兩両錢分斤升合碗盞杯盃枚粒片丸個个日次服歲月年]", rest):
                stats["headers_footers_removed"] += 1
                return rest
    return page


def process_tan_san_nam_duoc(text: str, filename: str) -> tuple[str, dict]:
    stats = defaultdict(int)
    text = nfc(text)
    raw_lines = text.splitlines()
    stats["input_lines"] = len(raw_lines)
    stats["input_chars"] = len(text)

    non_empty = [l.strip() for l in raw_lines if l.strip()]

    # 1. Remove cover-only first page (e.g., "卷捌", "卷玖", "卷拾") and q9 TOC page
    cleaned_pages = []
    for idx, p in enumerate(non_empty):
        # Cover-only title page (2 chars: 卷捌 / 卷玖 / 卷拾)
        if idx == 0 and re.fullmatch(r"卷[一二三四五六七八九十拾捌玖伍陸]", p):
            stats["front_matter_pages_removed"] += 1
            continue
        # q9 TOC page: starts with "卷九小兒科息驚慢驚..." (list of disease names before "新刊南菓神效十科應治卷...")
        if idx <= 2 and p.startswith("卷九小兒科") and "新刊" not in p:
            stats["front_matter_pages_removed"] += 1
            continue

        # Strip woodblock folio page number at start of page
        p = strip_woodblock_folio_prefix(p, idx, stats)

        # Clean Bopomofo / Kana OCR garbage characters that were misread from cursive woodblock chars:
        # In woodblock Nam Duoc Than Hieu, cursive 錢 (tiền/đồng cân) is consistently OCR'd as
        # Bopomofo/Kana (ㄐ, ㄔ, ㄇ, ㄟ, ㄢ, ㄘ, ㄗ, ㄡ, ㄏ, り) right after numerals (e.g. 每服一ㄐ, 各三ㄇ).
        # Wait: removing pure garbage Bopomofo/Kana symbols cleans non-CJK OCR noise.
        # Better yet: only strip stray Bopomofo/Kana if not a valid Han char? Wait - replacing with 錢
        # could be risky if one was 寸/分, whereas leaving Bopomofo leaves garbage characters!
        # Let's check: in "每服一ㄐ", "每服二ㄔ", "每服半ㄏ", removing the Bopomofo symbol removes the garbage char.
        # Wait: let's check requirement: "Sửa ký tự rác và khoảng trắng bất thường" AND
        # "Giữ tên riêng, niên hiệu, con số và đơn vị đo theo nguồn".
        # Removing stray Bopomofo/Kana garbage chars [\u3100-\u312f\u3040-\u30ff] fixes OCR garbage without inventing text.
        p, n = re.subn(r"[\u3100-\u312f\u3040-\u30ff]", "", p)
        stats["garbage_tokens_cleaned"] += n

        # Normalize whitespace & stray symbols
        p = normalize_cjk_punct_and_ws(p, stats)
        if p:
            cleaned_pages.append(p)

    # 2. Deduplicate adjacent identical/overlapping pages
    cleaned_pages = deduplicate_adjacent_lines(cleaned_pages, stats)

    # 3. Merge pages into continuous prose (woodblock pages have almost no terminal punctuation,
    #    so every page break splits a sentence/recipe mid-way!).
    #    Join pages directly, inserting paragraph breaks before major recipe/section starters if the
    #    previous page ended cleanly, or merging seamlessly when broken mid-sentence.
    merged_text = ""
    for i, p in enumerate(cleaned_pages):
        if i == 0:
            merged_text = p
        else:
            # If the new page starts with a new recipe marker ("一經治", "一傳治", "又方", "又經方", "一方")
            # and previous page ended with a common recipe ending ("效", "愈", "止", "消", "散", "妙", "安", "瘥", "良", "。"),
            # keep a paragraph break; otherwise merge seamlessly across page break!
            if (
                re.match(r"^(?:一[經经傳传方治法]|又[經经傳传]?方)", p)
                and merged_text[-1] in "效愈止消散妙安瘥良平。"
            ):
                merged_text += "\n\n" + p
            else:
                merged_text += p
                stats["page_break_merges"] += 1

    out = merged_text.strip() + "\n"
    stats["output_lines"] = len([l for l in out.splitlines() if l.strip()])
    stats["output_chars"] = len(out)
    return out, dict(stats)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report = {}

    files = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".txt"))
    print(f"Found {len(files)} text files in {INPUT_DIR}")

    for fname in files:
        in_path = os.path.join(INPUT_DIR, fname)
        out_path = os.path.join(OUTPUT_DIR, fname)

        with open(in_path, "r", encoding="utf-8") as f:
            raw = f.read()

        if fname == "nam-duoc-than-hieu-pdf.txt":
            refined, stats = process_nam_duoc_than_hieu(raw)
        elif fname == "nam_khoa_nu_khoa.txt":
            refined, stats = process_nam_khoa_nu_khoa(raw)
        elif fname == "Pho_Thanh_Chu_Nu_Khoa.txt":
            refined, stats = process_pho_thanh_chu(raw)
        elif fname.startswith("Linh khởi điểm"):
            refined, stats = process_linh_khoi_diem(raw)
        elif fname.startswith("phuong-te-hoc"):
            refined, stats = process_phuong_te_hoc(raw)
        elif fname.startswith("Tân_san_Nam_dược"):
            refined, stats = process_tan_san_nam_duoc(raw, fname)
        else:
            raise ValueError(f"Unhandled file: {fname}")

        with open(out_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(refined)

        report[fname] = stats
        print(f"[OK] {fname}: {stats['input_chars']:,} -> {stats['output_chars']:,} chars | {stats['input_lines']} -> {stats['output_lines']} lines")

    summary_path = os.path.join(BASE_DIR, "refinement_stats.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nSaved stats to {summary_path}")


if __name__ == "__main__":
    main()
