"""Utilities for the Kim Han-Nom Sentence Segmentation API module.

Provides text file discovery, line-based chunk splitting, sentence splitting helpers,
and result text extraction and concatenation.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Sequence

# Sentence-ending punctuation for /sentence-punctuation fallback splitting
_SENT_END_RE = re.compile(r"(?<=[。？！；])")


def split_cjk_sentences(text: str) -> list[str]:
    """Split an already-punctuated text into sentences on CJK sentence-ending marks."""
    sentences: list[str] = []
    for line in text.replace("\r\n", "\n").replace("\r", "\n").splitlines():
        line = line.strip()
        if not line:
            continue
        for part in _SENT_END_RE.split(line):
            part = part.strip()
            if part:
                sentences.append(part)
    return sentences


def split_text_into_line_chunks(text: str) -> list[str]:
    """Divide text into chunks where each chunk is a non-empty line before a newline."""
    chunks: list[str] = []
    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        cleaned = raw_line.strip()
        if cleaned:
            chunks.append(cleaned)
    return chunks


def concat_chunk_results(results: Sequence[str]) -> str:
    """Concatenate segmented text outputs from multiple chunk API calls for a file.

    Ensures a trailing newline exists on each non-empty chunk result before
    concatenating the next one so two sentences never end up on the same line
    at the concatenation point.
    """
    parts: list[str] = []
    for res in results:
        if not res:
            continue
        cleaned = res.replace("\r\n", "\n").replace("\r", "\n").strip()
        if not cleaned:
            continue
        if not cleaned.endswith("\n"):
            cleaned += "\n"
        parts.append(cleaned)
    return "".join(parts)


def discover_text_files(
    roots: str | Path | list[str | Path],
    extensions: Sequence[str] = (".txt",),
    recursive: bool = True,
    skip_macosx: bool = True,
) -> dict[str, str]:
    """Find all text files under ``roots`` and return ``{doc_name: source_path}``."""
    if isinstance(roots, (str, Path)):
        roots = [roots]

    normalized_exts = {
        ext.lower() if ext.startswith(".") else f".{ext.lower()}"
        for ext in extensions
        if ext.strip()
    }
    if not normalized_exts:
        normalized_exts = {".txt"}

    paths: list[Path] = []
    for root in roots:
        candidate = Path(root)
        if candidate.is_file() and candidate.suffix.lower() in normalized_exts:
            paths.append(candidate)
        elif candidate.is_dir():
            iterator = candidate.rglob("*") if recursive else candidate.iterdir()
            for item in iterator:
                if item.is_file() and item.suffix.lower() in normalized_exts:
                    paths.append(item)

    docs: dict[str, str] = {}
    for path in sorted(set(paths), key=lambda p: str(p)):
        path_str = str(path)
        if skip_macosx and "__MACOSX" in path_str:
            continue
        name = re.sub(r"[^\w\-]+", "_", path.stem).strip("_") or "document"
        while name in docs:
            name += "_"
        docs[name] = path_str
    return docs


def extract_result_text(result: dict[str, Any]) -> str:
    """Extract the segmented text content from an API response dictionary."""
    text = result.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()
    sentences = result.get("sentences")
    if isinstance(sentences, list) and sentences:
        cleaned = [str(s).strip() for s in sentences if str(s).strip()]
        return "\n".join(cleaned)
    return ""
