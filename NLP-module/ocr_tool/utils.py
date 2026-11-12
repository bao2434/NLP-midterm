"""Utilities used by PaddleOCR_Tool.

This module contains the reusable parts of the original notebook:
PDF discovery/rendering, page selection, OCR result persistence, text cleanup,
reading-order reconstruction, and GPU/task helpers.
"""
from __future__ import annotations

import glob
import json
import re
import subprocess
from pathlib import Path
from typing import Any

HUNYUAN_PROMPT = (
    "提取文档图片中正文的所有信息用markdown格式表示，其中页眉、页脚部分忽略，"
    "表格用html格式表达，文档中公式用latex格式表示，按照阅读顺序组织进行解析。"
)

PADDLE_IGNORE = {
    "header", "header_image", "footer", "footer_image", "number",
    "footnote", "aside_text", "image", "chart", "seal",
}

COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#00a0b0", "#f032e6", "#9a6324", "#808000", "#000075",
]

# Cache PDF handles across page renders.
_DOCS: dict[str, Any] = {}


def clean_degenerate(
    text: str,
    max_unit: int = 40,
    min_chars: int = 60,
    min_cycles: int = 4,
) -> str:
    """Trim a repeated trailing cycle caused by a VLM generation loop."""
    stripped = text.rstrip()
    for unit_length in range(1, max_unit + 1):
        unit = stripped[-unit_length:]
        if len(stripped) < min_chars or not unit.strip():
            continue
        repetitions, index = 0, len(stripped)
        while (
            index >= unit_length
            and stripped[index - unit_length:index] == unit
        ):
            repetitions += 1
            index -= unit_length
        if repetitions * unit_length >= min_chars and repetitions >= min_cycles:
            return stripped[:index].rstrip()
    return text


def md_to_lines(markdown: str) -> list[str]:
    """Convert OCR Markdown/HTML to plain-text lines."""
    markdown = re.sub(r"</t[dh]>", " ", markdown)
    markdown = re.sub(r"</tr>|<br\s*/?>", "\n", markdown)
    markdown = re.sub(r"<[^>]+>", "", markdown)
    markdown = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", markdown)

    output: list[str] = []
    for line in markdown.splitlines():
        line = re.sub(r"^\s{0,3}#{1,6}\s*", "", line)
        line = re.sub(r"^\s*[-*+]\s+(?=\S)", "", line)
        # Remove Hunyuan image coordinates at the end of a line.
        line = re.sub(
            r"(?:[ \t,，]*\(\d{1,4},\d{1,4}\),"
            r"\(\d{1,4},\d{1,4}\))+\s*$",
            "",
            line,
        )
        line = line.replace("**", "").replace("__", "").strip()
        if line.startswith("|") and line.endswith("|"):
            line = " ".join(
                cell.strip()
                for cell in line.strip("|").split("|")
                if cell.strip()
            )
        if line and not re.fullmatch(r"[-|:\s]+", line):
            output.append(line)
    return output


def page_lines(record: dict[str, Any]) -> list[str]:
    """Convert one page record to text lines in estimated reading order."""
    if "blocks" not in record:
        return md_to_lines(record.get("text", ""))

    blocks = record["blocks"]
    ordered = sorted(
        (block for block in blocks if block.get("order") is not None),
        key=lambda block: block["order"],
    )

    def center(block: dict[str, Any]) -> tuple[float, float]:
        x1, y1, x2, y2 = block["box"]
        return (x1 + x2) / 2, (y1 + y2) / 2

    # Paddle does not assign reading order to some tables/captions.
    for block in (b for b in blocks if b.get("order") is None):
        if not ordered:
            ordered.append(block)
            continue
        cx, cy = center(block)
        nearest = min(
            range(len(ordered)),
            key=lambda i: (
                (center(ordered[i])[0] - cx) ** 2
                + (center(ordered[i])[1] - cy) ** 2
            ),
        )
        ordered.insert(
            nearest + int(cy > center(ordered[nearest])[1]),
            block,
        )
    return [
        line
        for block in ordered
        for line in md_to_lines(block.get("text", ""))
    ]


def parse_pages(spec: str, page_count: int) -> list[int]:
    """Parse 1-based page selection, e.g. '1-10,15'; return zero-based pages."""
    if not spec.strip():
        return list(range(page_count))

    selected: list[int] = []
    for part in spec.split(","):
        start, separator, end = part.strip().partition("-")
        first = int(start)
        last = int(end) if separator else first
        if first < 1 or last < first:
            raise ValueError(f"Invalid page range: {part!r}")
        selected.extend(range(first - 1, min(last, page_count)))
    return sorted(set(selected))


def render_pdf_page(
    source: str | Path,
    page: int,
    dpi: int = 200,
    max_side: int = 1600,
):
    """Render a zero-based PDF page as an RGB PIL image."""
    import pypdfium2 as pdfium

    source = str(Path(source).resolve())
    if source not in _DOCS:
        _DOCS[source] = pdfium.PdfDocument(source)
    document = _DOCS[source]
    if page < 0 or page >= len(document):
        raise IndexError(
            f"Page {page} is outside the PDF ({len(document)} pages)."
        )

    image = document[page].render(scale=dpi / 72).to_pil().convert("RGB")
    if max(image.size) > max_side:
        scale = max_side / max(image.size)
        image = image.resize(
            (int(image.width * scale), int(image.height * scale))
        )
    return image


def close_pdf_documents() -> None:
    """Close all cached PDF handles."""
    for document in _DOCS.values():
        document.close()
    _DOCS.clear()


def discover_pdfs(
    roots: str | Path | list[str | Path],
    page_spec: str = "",
    recursive: bool = True,
    skip_macosx: bool = True,
) -> dict[str, tuple[str, list[int]]]:
    """Find PDFs and return {book_name: (source_path, selected_zero_based_pages)}."""
    if isinstance(roots, (str, Path)):
        roots = [roots]

    paths: list[str] = []
    for root in roots:
        root = str(root)
        candidate = Path(root)
        if candidate.is_file() and candidate.suffix.lower() == ".pdf":
            paths.append(str(candidate))
        elif candidate.is_dir():
            pattern = "**/*.pdf" if recursive else "*.pdf"
            paths.extend(glob.glob(str(candidate / pattern), recursive=recursive))

    import pypdfium2 as pdfium

    books: dict[str, tuple[str, list[int]]] = {}
    for path in sorted(set(paths)):
        if skip_macosx and "__MACOSX" in path:
            continue
        name = re.sub(r"[^\w\-]+", "_", Path(path).stem)
        while name in books:
            name += "_"
        document = pdfium.PdfDocument(path)
        try:
            pages = parse_pages(page_spec, len(document))
        finally:
            document.close()
        books[name] = (path, pages)
    return books


def make_tasks(books: dict[str, tuple[str, list[int]]]) -> list[dict[str, Any]]:
    """Expand discovered books into one task per selected page."""
    return [
        {"book": book, "src": source, "page": page}
        for book, (source, selected) in books.items()
        for page in selected
    ]


def write_tasks(tasks: list[dict[str, Any]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for task in tasks:
            stream.write(json.dumps(task, ensure_ascii=False) + "\n")
    return path


def read_tasks(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def result_file(output: str | Path, book: str, name: str, shard: int) -> Path:
    return Path(output) / book / f"{name}_shard{shard}.jsonl"


def load_results(output: str | Path, book: str, name: str) -> dict[int, dict]:
    """Load and merge all shard and previous-run result files for one book."""
    folder = Path(output) / book
    records: dict[int, dict] = {}
    if not folder.exists():
        return records
    for file in sorted(folder.glob(f"{name}_shard*.jsonl")):
        with file.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    record = json.loads(line)
                    records[record["page"]] = record
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue
    for file in sorted(folder.glob(f"{name}_shard*_prev.jsonl")):
        with file.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    record = json.loads(line)
                    records.setdefault(record["page"], record)
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue
    return records


def load_skipped(output: str | Path, book: str, name: str) -> set[int]:
    """Read skipped page indices from skip logs."""
    skipped: set[int] = set()
    folder = Path(output) / book
    if not folder.exists():
        return skipped
    for file in folder.glob(f"{name}_skip*.log"):
        for value in file.read_text(encoding="utf-8").split():
            try:
                skipped.add(int(value))
            except ValueError:
                pass
    return skipped


def gpu_count() -> int:
    """Return visible NVIDIA GPU count; returns zero if nvidia-smi is unavailable."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            check=False,
        )
        return sum(
            line.startswith("GPU ") for line in result.stdout.splitlines()
        )
    except FileNotFoundError:
        return 0


def draw_layout_boxes(
    image,
    boxes: list[dict[str, Any]],
    output: str | Path | None = None,
):
    """Draw labeled layout boxes on a rendered page; return the PIL image."""
    from PIL import ImageDraw, ImageFont

    image = image.copy()
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=max(14, image.width // 60))
    labels = sorted({box.get("label", "unknown") for box in boxes})

    for index, box_data in enumerate(boxes, 1):
        label = box_data.get("label", "unknown")
        coordinates = box_data.get("box", [])
        if len(coordinates) != 4:
            continue
        color = COLORS[labels.index(label) % len(COLORS)]
        draw.rectangle(coordinates, outline=color, width=3)
        draw.text(
            (coordinates[0] + 4, coordinates[1] + 2),
            f"{index}.{label}",
            fill=color,
            font=font,
        )

    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        image.save(output)
    return image


def install_paddle_environment(
    venv: str = "/tmp/venv_paddle",
    python: str | None = None,
    paddle_version: str = "3.2.1",
    cuda_index: str = "https://www.paddlepaddle.org.cn/packages/stable/cu126/",
) -> str:
    """Reproduce the notebook's isolated Python 3.11 Paddle GPU environment.

    Requires uv and a compatible NVIDIA/CUDA runtime. Returns the venv Python path.
    """
    import sys

    host_python = python or sys.executable

    def run(command: list[str]) -> None:
        subprocess.run(command, check=True)

    # The notebook installs these in the host environment before creating
    # the isolated Paddle environment.
    run([host_python, "-m", "pip", "install", "-q", "pypdfium2", "uv"])

    venv_path = Path(venv)
    if not venv_path.exists():
        run([
            host_python, "-m", "uv", "venv", "--seed",
            "--python", "3.11", str(venv_path),
        ])

    venv_python = str(venv_path / ("Scripts/python.exe" if __import__("os").name == "nt" else "bin/python"))
    run([
        venv_python, "-m", "pip", "install", "-q",
        f"paddlepaddle-gpu=={paddle_version}",
        "-i", cuda_index, "--extra-index-url", "https://pypi.org/simple",
    ])
    run([
        venv_python, "-m", "pip", "install", "-q",
        "paddleocr[doc-parser]", "pypdfium2", "pillow",
    ])
    run([
        venv_python, "-c", "import paddle; paddle.utils.run_check()",
    ])
    return venv_python
