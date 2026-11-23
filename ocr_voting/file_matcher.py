"""Discover model folders and match OCR files representing the same content."""

from __future__ import annotations

import os
import unicodedata
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DocumentGroup:
    """A set of OCR output files from different models for the same source document."""

    doc_key: str
    stem: str
    model_files: dict[str, Path]


def normalize_filename(filename: str) -> str:
    """Normalize a filename so slight naming variants match the same document.

    Handles:
    - Unicode NFC vs NFD differences (common in Vietnamese filenames).
    - Trailing underscores before the file extension (e.g. ``q5_.txt`` vs ``q5.txt``).
    """
    nfc = unicodedata.normalize("NFC", filename)
    stem, ext = os.path.splitext(nfc)
    normalized_stem = stem.rstrip("_")
    return f"{normalized_stem}{ext.lower()}"


def discover_document_groups(
    input_dir: Path,
    file_filter: str | None = None,
) -> tuple[list[str], list[DocumentGroup]]:
    """Scan ``input_dir`` for model subfolders and group matching ``.txt`` files.

    Args:
        input_dir: Path to the directory containing one subfolder per OCR model.
        file_filter: Optional substring to filter document names (case-insensitive).

    Returns:
        A tuple of ``(model_names, document_groups)``.
    """
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    model_dirs = sorted(
        [p for p in input_dir.iterdir() if p.is_dir() and not p.name.startswith(".")],
        key=lambda p: p.name,
    )
    if len(model_dirs) < 2:
        raise ValueError(
            f"Expected at least 2 model subfolders in {input_dir}, found {len(model_dirs)}"
        )

    model_names = [d.name for d in model_dirs]
    grouped: dict[str, dict[str, Path]] = {}

    for model_dir in model_dirs:
        for file_path in sorted(model_dir.iterdir(), key=lambda p: p.name):
            if not file_path.is_file() or file_path.suffix.lower() != ".txt":
                continue
            key = normalize_filename(file_path.name)
            grouped.setdefault(key, {})[model_dir.name] = file_path

    filter_norm = (
        unicodedata.normalize("NFC", file_filter).lower() if file_filter else None
    )

    groups: list[DocumentGroup] = []
    for key in sorted(grouped.keys()):
        if filter_norm and filter_norm not in key.lower():
            continue
        stem = os.path.splitext(key)[0]
        groups.append(DocumentGroup(doc_key=key, stem=stem, model_files=grouped[key]))

    return model_names, groups
