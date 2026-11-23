"""Step 1: Normalize OCR outputs prior to Levenshtein alignment and voting."""

from __future__ import annotations

import re
import unicodedata

# Map common ASCII / half-width / variant punctuation to canonical CJK forms.
_PUNCTUATION_MAP = str.maketrans(
    {
        ",": "，",
        ";": "；",
        ":": "：",
        "?": "？",
        "!": "！",
        "(": "（",
        ")": "）",
        "[": "【",
        "]": "】",
        "•": "·",
        "・": "·",
        "ㆍ": "·",
    }
)

# Zero-width and invisible formatting characters that occasionally appear in OCR output.
_INVISIBLE_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\ufeff]")


def split_blocks(text: str) -> list[str]:
    """Split an OCR document into page/paragraph blocks separated by blank lines."""
    stripped = text.strip()
    if not stripped:
        return []
    return re.split(r"\r?\n\s*\r?\n", stripped)


def normalize_text(
    text: str,
    ignore_whitespace: bool = True,
    normalize_punctuation: bool = True,
) -> str:
    """Normalize an OCR text block for accurate character-level alignment.

    Operations applied:
    1. Unicode NFKC normalization (canonicalizes full/half-width & compatibility forms).
    2. Strip invisible / zero-width control characters.
    3. Optionally unify punctuation variants (e.g. ``,`` -> ``，``, ``(`` -> ``（``).
    4. Optionally remove spurious OCR line breaks and whitespace within a block.
    """
    out = unicodedata.normalize("NFKC", text)
    out = _INVISIBLE_RE.sub("", out)
    if normalize_punctuation:
        out = out.translate(_PUNCTUATION_MAP)
    if ignore_whitespace:
        out = re.sub(r"\s+", "", out)
    else:
        out = out.replace("\r\n", "\n").replace("\r", "\n").strip()
    return out


def normalize_document_blocks(
    raw_text: str,
    ignore_whitespace: bool = True,
    normalize_punctuation: bool = True,
) -> list[str]:
    """Split a raw OCR document into blocks and normalize each block."""
    return [
        normalize_text(
            block,
            ignore_whitespace=ignore_whitespace,
            normalize_punctuation=normalize_punctuation,
        )
        for block in split_blocks(raw_text)
    ]
