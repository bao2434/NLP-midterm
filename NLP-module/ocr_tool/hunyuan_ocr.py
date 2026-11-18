"""
Reusable PDF/image OCR module extracted from the HunyuanOCR notebook.

Dependencies:
    pip install "transformers>=4.57" accelerate torch pypdfium2 pillow

Optional Paddle backend:
    pip install "paddleocr"

Quick use:
    from hunyuan_ocr import HunyuanOCR, OCRConfig

    ocr = HunyuanOCR(OCRConfig(dtype="fp16"))
    result = ocr.process_pdf("book.pdf", output_dir="ocr_output")
    ocr.export_txt(result, "book.txt")
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

DEFAULT_PROMPT = (
    "提取文档图片中正文的所有信息用markdown格式表示，其中页眉、页脚部分忽略，"
    "表格用html格式表达，文档中公式用latex格式表示，按照阅读顺序组织进行解析。"
)
PADDLE_IGNORE = {
    "header", "header_image", "footer", "footer_image", "number",
    "footnote", "aside_text", "image", "chart", "seal",
}


@dataclass
class OCRConfig:
    engine: str = "hunyuan"             # "hunyuan" or "paddle"
    model: str = "tencent/HunyuanOCR"
    prompt: str = DEFAULT_PROMPT
    dtype: str = "fp32"                 # fp32, fp16, bf16
    dpi: int = 200
    max_side: int = 1600
    max_new_tokens: int = 6000
    repetition_penalty: float = 1.08
    device: Optional[str] = None        # None = auto (CUDA if available)
    paddle_version: str = "v1.6"
    paddle_args: Optional[dict] = None


def clean_degenerate(text: str, max_unit: int = 40,
                     min_chars: int = 60, min_cycles: int = 4) -> str:
    """Trim a short repeated tail sometimes produced by vision-language models."""
    st = text.rstrip()
    for unit_len in range(1, max_unit + 1):
        unit = st[-unit_len:]
        if len(st) < min_chars or not unit.strip():
            continue
        count, pos = 0, len(st)
        while pos >= unit_len and st[pos-unit_len:pos] == unit:
            count, pos = count + 1, pos - unit_len
        if count * unit_len >= min_chars and count >= min_cycles:
            return st[:pos].rstrip()
    return text


def markdown_to_lines(markdown: str) -> list[str]:
    """Convert OCR Markdown/HTML to readable plain-text lines."""
    markdown = re.sub(r"</t[dh]>", " ", markdown)
    markdown = re.sub(r"</tr>|<br\s*/?>", "\n", markdown)
    markdown = re.sub(r"<[^>]+>", "", markdown)
    markdown = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", markdown)
    lines = []
    for line in markdown.splitlines():
        line = re.sub(r"^\s{0,3}#{1,6}\s*", "", line)
        line = re.sub(r"^\s*[-*+]\s+(?=\S)", "", line)
        line = re.sub(
            r"(?:[ \t,，]*\(\d{1,4},\d{1,4}\),\(\d{1,4},\d{1,4}\))+\s*$",
            "", line
        )
        line = line.replace("**", "").replace("__", "").strip()
        if line.startswith("|") and line.endswith("|"):
            line = " ".join(
                cell.strip() for cell in line.strip("|").split("|") if cell.strip()
            )
        if line and not re.fullmatch(r"[-|:\s]+", line):
            lines.append(line)
    return lines


def page_lines(record: dict) -> list[str]:
    """Return a page's OCR text as lines, respecting Paddle reading order."""
    if "blocks" not in record:
        return markdown_to_lines(record.get("text", ""))
    ordered = sorted(
        (b for b in record["blocks"] if b.get("order") is not None),
        key=lambda b: b["order"]
    )

    def center(block):
        box = block["box"]
        return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

    for block in (b for b in record["blocks"] if b.get("order") is None):
        if not ordered:
            ordered.append(block)
            continue
        cx, cy = center(block)
        nearest = min(
            range(len(ordered)),
            key=lambda i: (center(ordered[i])[0] - cx) ** 2
                         + (center(ordered[i])[1] - cy) ** 2
        )
        ordered.insert(nearest + (cy > center(ordered[nearest])[1]), block)
    return [line for block in ordered for line in markdown_to_lines(block["text"])]


class _HunyuanBackend:
    def __init__(self, config: OCRConfig):
        import torch
        from transformers import AutoProcessor
        try:
            from transformers import HunYuanVLForConditionalGeneration as Model
        except ImportError:
            from transformers import AutoModelForImageTextToText as Model

        self.torch = torch
        self.config = config
        self.device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype_map = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }
        if config.dtype not in dtype_map:
            raise ValueError("dtype must be one of: fp32, fp16, bf16")
        self.processor = AutoProcessor.from_pretrained(
            config.model, trust_remote_code=True, use_fast=False
        )
        self.model = Model.from_pretrained(
            config.model, dtype=dtype_map[config.dtype], trust_remote_code=True
        ).to(self.device).eval()

    def predict(self, image):
        torch = self.torch
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": self.config.prompt},
            ],
        }]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(self.device)
        for key, value in list(inputs.items()):
            if torch.is_tensor(value) and value.is_floating_point():
                inputs[key] = value.to(self.model.dtype)
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
                repetition_penalty=self.config.repetition_penalty,
            )
        generated = output[:, inputs["input_ids"].shape[1]:]
        text = self.processor.batch_decode(generated, skip_special_tokens=True)[0]
        return clean_degenerate(text), {
            "hit_max_tokens": generated.shape[1] >= self.config.max_new_tokens
        }


class _PaddleBackend:
    def __init__(self, config: OCRConfig):
        from paddleocr import PaddleOCRVL
        self.pipeline = PaddleOCRVL(
            pipeline_version=config.paddle_version,
            **(config.paddle_args or {})
        )
        self.max_new_tokens = config.max_new_tokens

    def predict(self, image):
        try:
            result = list(self.pipeline.predict(str(image), max_new_tokens=self.max_new_tokens))[0]
        except TypeError:
            result = list(self.pipeline.predict(str(image)))[0]
        return self.parse(result.json)

    @staticmethod
    def parse(data):
        data = data.get("res", data) if isinstance(data, dict) else {}
        blocks = [
            {
                "label": block.get("block_label"),
                "box": [round(x) for x in block.get("block_bbox", [0, 0, 0, 0])],
                "order": block.get("block_order"),
                "text": clean_degenerate(block.get("block_content") or ""),
            }
            for block in data.get("parsing_res_list", [])
            if block.get("block_label") not in PADDLE_IGNORE
        ]
        blocks = [block for block in blocks if block["text"]]
        record = {"blocks": blocks}
        text = "\n".join(page_lines(record))
        boxes = [
            {
                "label": block["label"],
                "box": [round(x) for x in block["coordinate"]],
                "order": block.get("order"),
                "score": round(block.get("score", 0), 2),
            }
            for block in (data.get("layout_det_res") or {}).get("boxes", [])
        ]
        return text, {"blocks": blocks, "boxes": boxes}


class HunyuanOCR:
    """High-level OCR interface for scanned PDFs and individual images.

    Results are stored as JSONL so interrupted runs can resume without repeating
    pages already completed.
    """

    def __init__(self, config: Optional[OCRConfig] = None):
        self.config = config or OCRConfig()
        if self.config.engine not in {"hunyuan", "paddle"}:
            raise ValueError("engine must be 'hunyuan' or 'paddle'")
        self._backend = None
        self._pdf_cache = {}

    def _load_backend(self):
        if self._backend is None:
            self._backend = (
                _HunyuanBackend(self.config)
                if self.config.engine == "hunyuan"
                else _PaddleBackend(self.config)
            )
        return self._backend

    def render_page(self, pdf_path: str | Path, page_index: int):
        """Render a zero-based PDF page to a resized RGB PIL image."""
        import pypdfium2 as pdfium
        from PIL import Image
        path = str(Path(pdf_path).resolve())
        if path not in self._pdf_cache:
            self._pdf_cache[path] = pdfium.PdfDocument(path)
        document = self._pdf_cache[path]
        if page_index < 0 or page_index >= len(document):
            raise IndexError(f"Page index {page_index} outside PDF ({len(document)} pages)")
        image = document[page_index].render(
            scale=self.config.dpi / 72
        ).to_pil().convert("RGB")
        if max(image.size) > self.config.max_side:
            scale = self.config.max_side / max(image.size)
            image = image.resize(
                (int(image.width * scale), int(image.height * scale)),
                Image.Resampling.LANCZOS,
            )
        return image

    def process_image(self, image_path: str | Path) -> dict:
        """OCR one image file; returns text, metadata and elapsed seconds."""
        start = time.time()
        text, extra = self._load_backend().predict(Path(image_path))
        return {
            "text": text,
            "lines": markdown_to_lines(text) if "blocks" not in extra else page_lines(
                {"blocks": extra["blocks"]}
            ),
            "seconds": round(time.time() - start, 2),
            **extra,
        }

    def process_pdf(
        self,
        pdf_path: str | Path,
        output_dir: str | Path = "ocr_output",
        pages: Optional[list[int]] = None,
        resume: bool = True,
        progress: bool = True,
    ) -> dict[int, dict]:
        """OCR selected zero-based pages. Existing JSONL results are reused if resume=True."""
        import pypdfium2 as pdfium
        pdf_path = Path(pdf_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        document = pdfium.PdfDocument(str(pdf_path))
        page_count = len(document)
        document.close()
        selected = list(range(page_count)) if pages is None else sorted(set(pages))
        if any(page < 0 or page >= page_count for page in selected):
            raise ValueError(f"Page indices must be between 0 and {page_count - 1}")

        safe_name = re.sub(r"[^\w.-]+", "_", pdf_path.stem)
        result_path = output_dir / f"{safe_name}_{self.config.engine}.jsonl"
        records = {}
        if resume and result_path.exists():
            with result_path.open(encoding="utf-8") as file:
                for line in file:
                    try:
                        record = json.loads(line)
                        records[int(record["page"])] = record
                    except (ValueError, KeyError, json.JSONDecodeError):
                        continue

        todo = [page for page in selected if page not in records]
        for count, page_index in enumerate(todo, 1):
            started = time.time()
            try:
                image = self.render_page(pdf_path, page_index)
                text, extra = self._load_backend().predict(image)
                record = {
                    "page": page_index,
                    "text": text,
                    "width": image.width,
                    "height": image.height,
                    "seconds": round(time.time() - started, 2),
                    **extra,
                }
                with result_path.open("a", encoding="utf-8") as file:
                    file.write(json.dumps(record, ensure_ascii=False) + "\n")
                records[page_index] = record
                if progress:
                    print(
                        f"[{count}/{len(todo)}] page {page_index + 1}/{page_count}: "
                        f"{record['seconds']}s, {len(text)} chars"
                    )
            except Exception as error:
                if progress:
                    print(f"[ERROR] page {page_index + 1}: {type(error).__name__}: {error}")
        return {page: records[page] for page in selected if page in records}

    def export_txt(
        self,
        records: dict[int, dict],
        output_path: str | Path,
        page_order: Optional[list[int]] = None,
    ) -> Path:
        """Export OCR records as plain text, ordered by original PDF page."""
        order = page_order if page_order is not None else sorted(records)
        chunks = [
            "\n".join(page_lines(records[page]))
            for page in order if page in records
        ]
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n\n".join(chunk for chunk in chunks if chunk) + "\n",
                        encoding="utf-8")
        return path

    def close(self):
        """Release cached PDF handles and model memory."""
        for document in self._pdf_cache.values():
            document.close()
        self._pdf_cache.clear()
        self._backend = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def parse_page_spec(spec: str, page_count: int) -> list[int]:
    """Parse one-based page notation like '1-10,15' into zero-based indices."""
    if not spec.strip():
        return list(range(page_count))
    selected = set()
    for part in spec.split(","):
        start, separator, end = part.strip().partition("-")
        first = int(start)
        last = int(end) if separator else first
        if first < 1 or last < first:
            raise ValueError(f"Invalid page range: {part}")
        selected.update(range(first - 1, min(last, page_count)))
    return sorted(selected)
