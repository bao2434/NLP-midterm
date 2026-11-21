"""Auditable, conservative Hán–Việt text preprocessing before sentence alignment.

Project: NLP-midterm / NLP-module/alignment_tool/HanVietPreprocessor.py

Rule references (user-supplied):
- Quy_định_đồ_án.pdf: Quy định về dữ liệu văn bản; Tách câu.
- Hướng Dẫn Dóng Hàng câu Dịch nghĩa Hán - Việt.pdf: §5.2, §5.3.

No translation, no automatic semantic corrections, no guessing chapter matches.
Keeps original files untouched. Writes ONE combined plain-text audit log containing
ALL automatic/manual content edits, any section extraction, all sentence boundaries,
and rendering differences between clean text and one-line sentence outputs.
Each split records exactly up to 4 raw code points on each side, plus a readable
4-visible-character context on each side for situations involving whitespace.

Python stdlib only. Works locally or in Kaggle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


HAN_END = "。！？；"
VI_END = ".?!;:"
CLOSERS = '”’»」』）)]}'
ABBREVIATIONS = {"ts", "ths", "pgs", "gs", "bs", "tp", "q", "p", "tr", "mr", "mrs", "dr", "prof", "stt", "v.v", "vv", "vd", "tp.hcm"}


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _sha(value: str | bytes) -> str:
    return hashlib.sha256(value.encode("utf-8") if isinstance(value, str) else value).hexdigest()


def _raw4(text: str, at: int, side: str) -> str:
    return text[max(0, at - 4):at] if side == "left" else text[at:at + 4]


def _visible4(text: str, at: int, side: str) -> str:
    """Four non-whitespace code points, ordered as in source; O(4+spaces)."""
    output: list[str] = []
    if side == "left":
        cursor = at - 1
        while cursor >= 0 and len(output) < 4:
            if not text[cursor].isspace():
                output.append(text[cursor])
            cursor -= 1
        return "".join(reversed(output))
    cursor = at
    while cursor < len(text) and len(output) < 4:
        if not text[cursor].isspace():
            output.append(text[cursor])
        cursor += 1
    return "".join(output)


def _location(text: str, pos: int) -> str:
    return f"line={text.count(chr(10), 0, pos) + 1}, col={pos - (text.rfind(chr(10), 0, pos) + 1) + 1}"


class Audit:
    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.sequence = 0
        self.counts: dict[str, int] = {}
        self._out = filepath.open("w", encoding="utf-8", newline="\n")
        self._out.write("HAN–VIET PREPROCESSING AUDIT — original files are NOT modified\n")
        self._out.write("Offsets: zero-based Unicode code-point positions in the text BEFORE each operation.\n")
        self._out.write("All before/after excerpts are JSON-escaped: \\n is a newline, \\t is a tab.\n")
        self._out.write("For SPLIT, raw_left4/raw_right4 are 4 literal code points each;\n")
        self._out.write("visible_left4/visible_right4 skip whitespace for easier inspection.\n\n")

    def close(self):
        self._out.flush()
        self._out.close()

    def write(self, kind: str, lang: str, description: str, **fields: Any) -> None:
        self.sequence += 1
        self.counts[kind] = self.counts.get(kind, 0) + 1
        self._out.write(f"[{self.sequence:06d}] [{lang.upper()}] [{kind}] {description}\n")
        for name, val in fields.items():
            self._out.write(f"  {name}: {_json(val)}\n")
        self._out.write("\n")

    def edit(self, lang: str, reason: str, prior: str, start: int, end: int, new: str, *, source: str):
        old = prior[start:end]
        if old == new:
            return
        self.write("EDIT", lang, reason, source=source, at=start, end_before=end,
                   location_before=_location(prior, start), old=old, new=new,
                   before_left4=_raw4(prior, start, "left"),
                   before_right4=_raw4(prior, end, "right"))

    def split(self, lang: str, text: str, pos: int, reason: str):
        self.write("SPLIT", lang, reason, at=pos, location=_location(text, pos),
                   raw_left4=_raw4(text, pos, "left"), raw_right4=_raw4(text, pos, "right"),
                   visible_left4=_visible4(text, pos, "left"),
                   visible_right4=_visible4(text, pos, "right"))


def _replace_matches(text: str, regex: re.Pattern[str], transform: Callable[[re.Match[str]], str],
                     audit: Audit, lang: str, reason: str, source: str) -> str:
    matches = list(regex.finditer(text))
    if not matches:
        return text
    chunks: list[str] = []
    prev = 0
    for match in matches:
        new = transform(match)
        chunks.append(text[prev:match.start()])
        chunks.append(new)
        audit.edit(lang, reason, text, match.start(), match.end(), new, source=source)
        prev = match.end()
    chunks.append(text[prev:])
    return "".join(chunks)


def _replace_one(text: str, start: int, end: int, replacement: str, audit: Audit, lang: str,
                 reason: str, source: str) -> str:
    audit.edit(lang, reason, text, start, end, replacement, source=source)
    return text[:start] + replacement + text[end:]


def _normalise(text: str, audit: Audit, lang: str) -> str:
    # Order matters; all replacements (even BOM and CRLF) have explicit EDIT events.
    if text.startswith("\ufeff"):
        text = _replace_one(text, 0, 1, "", audit, lang, "Remove leading UTF-8 BOM", "automatic")
    text = _replace_matches(text, re.compile(r"\r\n|\r"), lambda _: "\n", audit, lang,
                            "Normalize line endings to LF", "automatic")
    text = _replace_matches(text, re.compile(r"[\u00a0\u202f\u3000]"), lambda _: " ", audit, lang,
                            "Replace nonbreaking/fullwidth space with ordinary space", "automatic")
    # Tabs inside a raw TXT are layout characters, not tab-separated parallel data.
    text = _replace_matches(text, re.compile(r"\t"), lambda _: " ", audit, lang,
                            "Convert tab to ordinary space in raw book text", "automatic")
    # Zero-width space in OCR dumps is usually formatting noise, but log every removal.
    text = _replace_matches(text, re.compile("\u200b"), lambda _: "", audit, lang,
                            "Remove zero-width space U+200B", "automatic")
    text = _replace_matches(text, re.compile(r"[ \t]+(?=\n|\Z)"), lambda _: "", audit, lang,
                            "Remove trailing spaces/tabs at physical line end", "automatic")
    # NFC, line by line: audit full before/after of each affected physical line.
    text = _replace_matches(text, re.compile(r"[^\n]+"),
                            lambda m: unicodedata.normalize("NFC", m.group()), audit, lang,
                            "Unicode NFC normalization (affected line)", "automatic")
    text = _replace_matches(text, re.compile(r"\n{3,}"), lambda _: "\n\n", audit, lang,
                            "Reduce excessive blank lines to one empty line", "automatic")
    return text


def _line_ranges(text: str) -> list[tuple[int, int]]:
    """Physical line ranges, each including its trailing LF when present."""
    out, start = [], 0
    for i, char in enumerate(text):
        if char == "\n":
            out.append((start, i + 1))
            start = i + 1
    if start < len(text):
        out.append((start, len(text)))
    return out


def _manual(text: str, cfg: dict[str, Any], audit: Audit, lang: str) -> str:
    """Operator-approved edits only. Line numbers reference post-normalization text.

    Supports remove_lines (one-based), join_after_lines (one-based) and replacements.
    Every edit includes actual old and new text in the single audit file.
    """
    remove = cfg.get("remove_lines", [])
    join = cfg.get("join_after_lines", [])
    for seq, name in [(remove, "remove_lines"), (join, "join_after_lines")]:
        if not isinstance(seq, list) or any(not isinstance(n, int) or isinstance(n, bool) or n < 1 for n in seq):
            raise ValueError(f"{lang}.{name} must be a list of positive line numbers")
        if len(set(seq)) != len(seq):
            raise ValueError(f"{lang}.{name} contains repeated line numbers")
    if set(remove) & set(join):
        raise ValueError(f"{lang}: cannot both remove and join after the same physical line")
    original_lines = _line_ranges(text)
    if any(n > len(original_lines) for n in [*remove, *join]):
        raise ValueError(f"{lang}: manual line number exceeds {len(original_lines)}")
    actions = [(n, "remove") for n in remove] + [(n, "join") for n in join]
    for line_no, action in sorted(actions, reverse=True):
        begin, end = original_lines[line_no - 1]
        if action == "remove":
            text = _replace_one(text, begin, end, "", audit, lang,
                                f"Manual: remove reviewed line {line_no}", "human-config")
        else:
            newline = end - 1
            if text[newline:newline + 1] != "\n":
                raise ValueError(f"{lang}: cannot join after final line {line_no}")
            run = re.match(r"\n(?:[ \t]*\n)*", text[newline:])
            assert run
            right_start = newline + run.end()
            if not text[begin:newline].strip() or not text[right_start:].strip():
                raise ValueError(f"{lang}: refusing join near empty content at line {line_no}")
            glue = "" if lang == "han" else " "
            text = _replace_one(text, newline, right_start, glue, audit, lang,
                                f"Manual: merge OCR/page-break after line {line_no}", "human-config")
    for row in cfg.get("replacements", []):
        old, new = row.get("find"), row.get("replace")
        if not isinstance(old, str) or not old or not isinstance(new, str):
            raise ValueError(f"{lang}: replacement requires nonempty 'find' and string 'replace'")
        expected = row.get("count", 1)
        if not isinstance(expected, int) or isinstance(expected, bool) or expected < 1:
            raise ValueError(f"{lang}: replacement count must be a positive integer")
        found = text.count(old)
        if found != expected:
            raise ValueError(f"{lang}: {_json(old)} appears {found} times, expected exactly {expected}")
        reason = str(row.get("reason", "operator-verified substitution"))
        # Replace left->right, without relying on regex metacharacters.
        index = 0
        for _ in range(expected):
            pos = text.find(old, index)
            text = _replace_one(text, pos, pos + len(old), new, audit, lang,
                                "Manual: " + reason, "human-config")
            index = pos + len(new)
    return text


def _select_section(text: str, start_marker: str | None, end_marker: str | None,
                    audit: Audit, lang: str) -> str:
    """Choose exact whole-line heading anchors (not occurrences inside body text)."""
    if not start_marker and not end_marker:
        return text

    def heading_indices(anchor: str) -> list[int]:
        pattern = re.compile(r"(?m)^[ \t]*" + re.escape(anchor) + r"[ \t]*(?:\r?$)")
        return [m.start() for m in pattern.finditer(text)]

    if start_marker:
        hits = heading_indices(start_marker)
        if len(hits) != 1:
            raise ValueError(f"{lang}: start heading {start_marker!r} occurs {len(hits)} times as a separate line; provide an exact unique heading")
        start = hits[0]
    else:
        start = 0
    if end_marker:
        hits = [i for i in heading_indices(end_marker) if i > start]
        if len(hits) != 1:
            raise ValueError(f"{lang}: end heading {end_marker!r} occurs {len(hits)} times after start; provide an exact unique heading")
        end = hits[0]
    else:
        end = len(text)
    if end <= start:
        raise ValueError(f"{lang}: empty section")
    audit.write("SECTION", lang, "Non-destructive extraction of exact whole-line section",
                start_marker=start_marker, end_marker=end_marker, start_offset=start, end_offset=end,
                omitted_prefix=text[:start], omitted_suffix=text[end:],
                selected_sha256=_sha(text[start:end]))
    return text[start:end]


def _candidate_reviews(text: str, lang: str) -> list[dict[str, Any]]:
    lines = text.splitlines()
    reviews: list[dict[str, Any]] = []
    prev_nonblank = ""
    for n, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped == prev_nonblank and len(stripped) >= 6:
            reviews.append({"language": lang, "line": n, "type": "possible_OCR_duplicate",
                            "text": line, "action": "review_only"})
        prev_nonblank = stripped
        if re.search(r"\[\d+\]|\(\d+\)|（\d+）", line):
            reviews.append({"language": lang, "line": n, "type": "possible_footnote_or_valid_number",
                            "text": line, "action": "review_only; DO NOT auto-delete formula numbering"})
        if re.fullmatch(r"\s*(?:\d{1,4}|[—–-]\s*\d{1,4}\s*[—–-])\s*", line):
            reviews.append({"language": lang, "line": n, "type": "possible_page_number",
                            "text": line, "action": "review_only"})
    return reviews


def _vi_boundary(text: str, i: int) -> bool:
    char = text[i]
    if char not in VI_END:
        return False
    if char in ".:":
        if i and i + 1 < len(text) and text[i - 1].isdigit() and text[i + 1].isdigit():
            return False
        if char == ":" and re.search(r"(?:https?|ftp)$", text[max(0, i - 6):i], re.I):
            return False
    # Dot after chapter marker: "1. ĐIỀU KINH" is NOT a sentence boundary.
    if char == ".":
        prefix = text[:i]
        token = re.search(r"([^\s]+)$", prefix)
        if token:
            word = token.group(1)
            if re.fullmatch(r"\d+", word):
                return False
            if word.casefold() in ABBREVIATIONS:
                return False
            if len(word) == 1 and word.isalpha():
                return False
    j = i + 1
    while j < len(text) and text[j] in CLOSERS:
        j += 1
    return j == len(text) or text[j].isspace()


def _boundaries(block: str, lang: str) -> list[tuple[int, str]]:
    result: list[tuple[int, str]] = []
    for i, char in enumerate(block):
        if (char in HAN_END if lang == "han" else _vi_boundary(block, i)):
            # Keep runs such as "！？" attached to one unit.
            if lang == "han" and i + 1 < len(block) and block[i + 1] in HAN_END:
                continue
            after = i + 1
            while after < len(block) and block[after] in CLOSERS:
                after += 1
            if after < len(block) and block[after:].strip():
                result.append((after, "Hán punctuation" if lang == "han" else "Vietnamese punctuation; verify semantics"))
    # Avoid repeated indices due to a closing quote or punctuation run.
    seen: set[int] = set()
    return [(pos, reason) for pos, reason in result if not (pos in seen or seen.add(pos))]


def _render_unit(raw: str, lang: str) -> str:
    raw = raw.strip()
    if lang == "han":
        # Only remove physical line breaks inside a unit, never every ordinary Han space.
        return re.sub(r"[ \t]*\n[ \t]*", "", raw)
    return re.sub(r"\s+", " ", raw)


def _heading_split_points(block: str, lang: str) -> list[int]:
    """Conservative heading detection. A hint only, not a semantic claim."""
    positions: list[int] = []
    cursor = 0
    for line in block.splitlines(keepends=True):
        trimmed = line.strip()
        next_cursor = cursor + len(line)
        if line.endswith("\n") and trimmed and block[next_cursor:].strip():
            if lang == "han":
                heading = (bool(re.fullmatch(r"[\u3400-\u9fff]{2,12}", trimmed)) or
                           (len(trimmed) <= 50 and bool(re.fullmatch(
                               r"[\u3400-\u9fff]{1,15}[（(][^()（）\n]{1,30}[）)]", trimmed))))
            else:
                letters = "".join(c for c in trimmed if c.isalpha())
                heading = (bool(re.fullmatch(r"[\u3400-\u9fff]{2,12}", trimmed)) or
                           (len(trimmed) <= 70 and len(letters) >= 2 and letters == letters.upper()
                            and not trimmed.endswith(tuple(VI_END))))
            if heading:
                positions.append(next_cursor)
        cursor = next_cursor
    return positions


def _sentences(text: str, lang: str, audit: Audit) -> list[dict[str, Any]]:
    """Split punctuation + visually explicit short headings; no arbitrary OCR line splitting.

    Cleaned text remains unchanged. Output sentence texts are a view with one
    sentence per line; any whitespace rendering changes are audited.
    """
    units: list[dict[str, Any]] = []
    separators = list(re.finditer(r"\n(?:[ \t]*\n)+", text))
    blocks: list[tuple[int, int]] = []
    left = 0
    for match in separators:
        if left < match.start() and text[left:match.start()].strip():
            blocks.append((left, match.start()))
        if text[:match.start()].strip() and text[match.end():].strip():
            audit.split(lang, text, match.start(), "paragraph boundary (not a semantic sentence decision)")
        left = match.end()
    if left < len(text) and text[left:].strip():
        blocks.append((left, len(text)))

    def emit(start: int, stop: int, kind: str):
        raw = text[start:stop]
        output = _render_unit(raw, lang)
        if not output:
            return
        if raw != output:
            audit.write("RENDER_UNIT", lang, "Change only in 1-sentence-per-line view; clean text unchanged",
                        source_start=start, source_end=stop, raw=raw, emitted=output)
        needs_review = (kind != "punctuation_hint")
        units.append({"id": len(units), "language": lang, "text": output,
                      "source_start": start, "source_end": stop, "split_basis": kind,
                      "needs_review": needs_review})

    for a, b in blocks:
        block = text[a:b]
        candidates = [(pos, reason, "punctuation_hint") for pos, reason in _boundaries(block, lang)]
        candidates += [(pos, "short standalone heading line (review)", "heading_hint")
                       for pos in _heading_split_points(block, lang)]
        # Equal boundary positions: punctuation takes priority.
        candidates.sort(key=lambda rec: (rec[0], rec[2] != "punctuation_hint"))
        chunk_start = a
        for pos, why, kind in candidates:
            global_pos = a + pos
            if global_pos <= chunk_start or not text[chunk_start:global_pos].strip():
                continue
            emit(chunk_start, global_pos, kind)
            audit.split(lang, text, global_pos, why)
            chunk_start = global_pos
        if chunk_start < b:
            segment = text[chunk_start:b]
            end_chars = HAN_END if lang == "han" else VI_END
            terminal_punct = bool(re.search(r"[" + re.escape(end_chars) + r"][" +
                                            re.escape(CLOSERS) + r"]*\s*$", segment))
            reason = "punctuation_hint" if terminal_punct else "no_explicit_terminal_punctuation"
            emit(chunk_start, b, reason)
    return units


@dataclass
class PreprocessResult:
    output_dir: Path
    audit_path: Path
    summary: dict[str, Any]
    units: dict[str, list[dict[str, Any]]]


class HanVietPreprocessor:
    def preprocess(self, han_file: str | Path, vi_file: str | Path,
                   output_dir: str | Path, *, config_path: str | Path | None = None,
                   han_start: str | None = None, han_end: str | None = None,
                   vi_start: str | None = None, vi_end: str | None = None) -> PreprocessResult:
        """Preprocess TWO books or matched chapters with a unified audit text file.

        config_path optional JSON:
          {"han":{"remove_lines":[1], "join_after_lines":[7],
                  "replacements":[{"find":"...","replace":"...","count":1,"reason":"..."}]},
           "vi":{...}}
        Config line numbers refer to the selected, automatically normalized text,
        BEFORE manual edits. Lines are processed from bottom to top.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        cfg = json.loads(Path(config_path).read_text(encoding="utf-8")) if config_path else {}
        if not isinstance(cfg, dict):
            raise ValueError("Configuration root must be a JSON object")
        audit = Audit(output_dir / "preprocess_changes.txt")
        unit_sets: dict[str, list[dict[str, Any]]] = {}
        summary: dict[str, Any] = {"inputs": {}, "audit": "preprocess_changes.txt",
                                   "rules": "Quy_định_đồ_án.pdf; Hướng Dẫn Dóng Hàng câu Dịch nghĩa Hán - Việt.pdf"}
        try:
            for lang, src, start_marker, end_marker in (
                ("han", Path(han_file), han_start, han_end),
                ("vi", Path(vi_file), vi_start, vi_end),
            ):
                source_bytes = src.read_bytes()
                # Do not use utf-8-sig: BOM deletion must be logged explicitly.
                source = source_bytes.decode("utf-8")
                audit.write("INPUT", lang, "Read UTF-8 without modifying original file",
                            path=str(src), size_bytes=len(source_bytes), original_sha256=_sha(source_bytes))
                chosen = _select_section(source, start_marker, end_marker, audit, lang)
                cleaned = _normalise(chosen, audit, lang)
                cleaned = _manual(cleaned, cfg.get(lang, {}), audit, lang)
                review = _candidate_reviews(cleaned, lang)
                for c in review:
                    audit.write("REVIEW", lang, c["type"], line=c["line"], text=c["text"],
                                action=c["action"])
                units = _sentences(cleaned, lang, audit)
                unit_sets[lang] = units
                (output_dir / f"{lang}_clean.txt").write_text(cleaned, encoding="utf-8", newline="\n")
                (output_dir / f"{lang}_sentences.txt").write_text(
                    "".join(item["text"] + "\n" for item in units), encoding="utf-8", newline="\n")
                with (output_dir / f"{lang}_sentences.jsonl").open("w", encoding="utf-8", newline="\n") as f:
                    for item in units:
                        f.write(_json(item) + "\n")
                with (output_dir / f"{lang}_review_candidates.jsonl").open("w", encoding="utf-8", newline="\n") as f:
                    for item in review:
                        f.write(_json(item) + "\n")
                summary["inputs"][lang] = {"path": str(src), "original_sha256": _sha(source_bytes),
                    "clean_sha256": _sha(cleaned), "section_selected": bool(start_marker or end_marker),
                    "characters_cleaned": len(cleaned), "units": len(units),
                    "ambiguous_units": sum(row["needs_review"] for row in units),
                    "review_candidates": len(review)}
                audit.write("OUTPUT", lang, "Clean text + sentence view created",
                            original_sha256=_sha(source_bytes), clean_sha256=_sha(cleaned),
                            sentence_count=len(units), review_candidates=len(review))
            summary["audit_event_counts"] = dict(audit.counts)
            (output_dir / "preprocess_summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            return PreprocessResult(output_dir, audit.filepath, summary, unit_sets)
        except Exception as exc:
            audit.write("ERROR", "system", "Processing aborted, inspect partial outputs",
                        error_type=type(exc).__name__, message=str(exc))
            raise
        finally:
            audit.close()


def _cli():
    parser = argparse.ArgumentParser(description="Audited conservative Hán–Việt text preprocessing")
    parser.add_argument("--han", required=True, help="Hán TXT file (UTF-8)")
    parser.add_argument("--vi", required=True, help="Vietnamese translation TXT file (UTF-8)")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--config", help="Optional reviewed manual-edits JSON")
    parser.add_argument("--han-start", help="Unique start anchor (included)")
    parser.add_argument("--han-end", help="Next heading anchor (excluded)")
    parser.add_argument("--vi-start", help="Unique start anchor (included)")
    parser.add_argument("--vi-end", help="Next heading anchor (excluded)")
    args = parser.parse_args()
    result = HanVietPreprocessor().preprocess(
        args.han, args.vi, args.output, config_path=args.config,
        han_start=args.han_start, han_end=args.han_end,
        vi_start=args.vi_start, vi_end=args.vi_end)
    print(json.dumps(result.summary, ensure_ascii=False, indent=2))
    print(f"Audit trail: {result.audit_path}")


if __name__ == "__main__":
    _cli()
