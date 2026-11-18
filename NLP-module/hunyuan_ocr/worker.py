"""Single-GPU PDF page OCR worker (HunyuanOCR or PaddleOCR-VL) with automatic resume."""
from __future__ import annotations

import argparse
import json
import re
import signal
import tempfile
import time
from pathlib import Path
from typing import Any

from .config import DEFAULT_HUNYUAN_PROMPT

PADDLE_IGNORE = {
    "header", "header_image", "footer", "footer_image", "number",
    "footnote", "aside_text", "image", "chart", "seal"
}

_DOCS: dict[str, Any] = {}


def clean_degenerate(s: str, max_unit: int = 40, min_chars: int = 60, min_cycles: int = 4) -> str:
    """Trim infinite repetitive loops caused by VLM token generation degeneration."""
    st = s.rstrip()
    for u in range(1, max_unit + 1):
        unit = st[-u:]
        if len(st) < min_chars or not unit.strip():
            continue
        n, i = 0, len(st)
        while i >= u and st[i - u:i] == unit:
            n, i = n + 1, i - u
        if n * u >= min_chars and n >= min_cycles:
            return st[:i].rstrip()
    return s


def md_to_lines(md: str) -> list[str]:
    """Convert OCR Markdown/HTML output to clean plain-text lines."""
    md = re.sub(r"</t[dh]>", " ", md)
    md = re.sub(r"</tr>|<br\s*/?>", "\n", md)
    md = re.sub(r"<[^>]+>", "", md)
    md = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", md)
    out = []
    for ln in md.splitlines():
        ln = re.sub(r"^\s{0,3}#{1,6}\s*", "", ln)
        ln = re.sub(r"^\s*[-*+]\s+(?=\S)", "", ln)
        ln = re.sub(r"(?:[ \t,，]*\(\d{1,4},\d{1,4}\),\(\d{1,4},\d{1,4}\))+\s*$", "", ln)
        ln = ln.replace("**", "").replace("__", "").strip()
        if ln.startswith("|") and ln.endswith("|"):
            ln = " ".join(c.strip() for c in ln.strip("|").split("|") if c.strip())
        if ln and not re.fullmatch(r"[-|:\s]+", ln):
            out.append(ln)
    return out


def page_lines(rec: dict) -> list[str]:
    """Extract ordered text lines from a page OCR record."""
    if "blocks" not in rec:
        return md_to_lines(rec["text"])
    ordered = sorted((b for b in rec["blocks"] if b["order"] is not None), key=lambda b: b["order"])
    center = lambda b: ((b["box"][0] + b["box"][2]) / 2, (b["box"][1] + b["box"][3]) / 2)
    for b in (b for b in rec["blocks"] if b["order"] is None):
        if not ordered:
            ordered.append(b)
            continue
        cx, cy = center(b)
        k = min(range(len(ordered)), key=lambda i: (center(ordered[i])[0] - cx) ** 2 + (center(ordered[i])[1] - cy) ** 2)
        ordered.insert(k + (cy > center(ordered[k])[1]), b)
    return [ln for b in ordered for ln in md_to_lines(b["text"])]


def render_page(src: str | Path, page: int, dpi: int = 200, max_side: int = 1600):
    """Render a 0-indexed PDF page to a PIL RGB Image capped at max_side."""
    import pypdfium2 as pdfium
    src_str = str(src)
    if src_str not in _DOCS:
        _DOCS[src_str] = pdfium.PdfDocument(src_str)
    img = _DOCS[src_str][page].render(scale=dpi / 72).to_pil().convert("RGB")
    if max(img.size) > max_side:
        s = max_side / max(img.size)
        img = img.resize((int(img.width * s), int(img.height * s)))
    return img


def close_cached_docs() -> None:
    for d in _DOCS.values():
        d.close()
    _DOCS.clear()


class HunyuanEngine:
    def __init__(self, args: argparse.Namespace):
        import torch
        from transformers import AutoProcessor
        try:
            from transformers import HunYuanVLForConditionalGeneration as ModelCls
        except ImportError:
            from transformers import AutoModelForImageTextToText as ModelCls

        self.torch = torch
        self.max_new = args.max_new
        self.rep = args.rep_penalty
        self.prompt = args.prompt or DEFAULT_HUNYUAN_PROMPT
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]

        self.proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True, use_fast=False)
        self.model = ModelCls.from_pretrained(args.model, dtype=dtype, trust_remote_code=True).to(self.dev).eval()

    def run(self, img_path: str) -> tuple[str, dict]:
        torch = self.torch
        msgs = [{"role": "user", "content": [{"type": "image", "image": img_path}, {"type": "text", "text": self.prompt}]}]
        inputs = self.proc.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(self.dev)
        for k, v in list(inputs.items()):
            if torch.is_tensor(v) and v.is_floating_point():
                inputs[k] = v.to(self.model.dtype)
        with torch.inference_mode():
            out = self.model.generate(
                **inputs, max_new_tokens=self.max_new, do_sample=False, repetition_penalty=self.rep
            )
        gen = out[:, inputs["input_ids"].shape[1]:]
        text = self.proc.batch_decode(gen, skip_special_tokens=True)[0]
        return clean_degenerate(text), {"hit_max": int(gen.shape[1]) >= self.max_new}


class PaddleEngine:
    def __init__(self, args: argparse.Namespace):
        from paddleocr import PaddleOCRVL
        self.pipe = PaddleOCRVL(pipeline_version=args.paddle_version, **json.loads(args.paddle_args))
        self.max_new = args.max_new

    def run(self, img_path: str) -> tuple[str, dict]:
        try:
            res = list(self.pipe.predict(img_path, max_new_tokens=self.max_new))[0]
        except TypeError:
            res = list(self.pipe.predict(img_path))[0]
        return self.parse(res.json)

    @staticmethod
    def parse(j: dict) -> tuple[str, dict]:
        j = j.get("res", j) if isinstance(j, dict) else {}
        blocks = [
            {
                "label": b.get("block_label"),
                "box": [round(x) for x in b.get("block_bbox", [0, 0, 0, 0])],
                "order": b.get("block_order"),
                "text": clean_degenerate(b.get("block_content") or ""),
            }
            for b in j.get("parsing_res_list", [])
            if b.get("block_label") not in PADDLE_IGNORE
        ]
        blocks = [b for b in blocks if b["text"]]
        text = "\n".join(page_lines({"blocks": blocks}))
        boxes = [
            {
                "label": b["label"],
                "box": [round(x) for x in b["coordinate"]],
                "order": b.get("order"),
                "score": round(b.get("score", 0), 2),
            }
            for b in (j.get("layout_det_res") or {}).get("boxes", [])
        ]
        return text, {"blocks": blocks, "boxes": boxes}


ENGINES = {"hunyuan": HunyuanEngine, "paddle": PaddleEngine}


def result_file(out: Path, book: str, name: str, shard: int) -> Path:
    return out / book / f"{name}_shard{shard}.jsonl"


def load_results(out: Path, book: str, name: str) -> dict[int, dict]:
    """Load and merge all shard JSONL records for a given book."""
    recs = {}
    book_dir = out / book
    if not book_dir.exists():
        return recs
    for f in sorted(book_dir.glob(f"{name}_shard*.jsonl")):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                    recs[r["page"]] = r
                except (json.JSONDecodeError, KeyError):
                    pass
    return recs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=list(ENGINES))
    ap.add_argument("--name", default=None)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--max-side", type=int, default=1600)
    ap.add_argument("--max-new", type=int, default=6000)
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--model", default="tencent/HunyuanOCR")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--rep-penalty", type=float, default=1.08)
    ap.add_argument("--paddle-version", default="v1.6")
    ap.add_argument("--paddle-args", default="{}")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--page-timeout", type=int, default=300)
    a = ap.parse_args()

    name, out = a.name or a.engine, Path(a.out)
    with open(a.tasks, encoding="utf-8") as fh:
        tasks = [json.loads(l) for l in fh if l.strip()]

    mine = [t for i, t in enumerate(tasks) if i % a.nshard == a.shard]
    done, skip = set(), set()
    for book in {t["book"] for t in mine}:
        done |= {(book, p) for p in load_results(out, book, name)}
        for f in (out / book).glob(f"{name}_skip*.log"):
            skip |= {(book, int(p)) for p in f.read_text().split()}

    inflight = out / f"_inflight_{name}_{a.shard}"
    if inflight.exists():
        book, _, p = inflight.read_text().rpartition("|")
        if book and (book, int(p)) not in done:
            (out / book).mkdir(parents=True, exist_ok=True)
            with open(out / book / f"{name}_skip.log", "a", encoding="utf-8") as f:
                f.write(p + "\n")
            skip.add((book, int(p)))
            print(f"[{name}#{a.shard}] {book} page {int(p) + 1}: hung previously -> skipping", flush=True)
        inflight.unlink()

    todo = [t for t in mine if (t["book"], t["page"]) not in (done | skip)][:a.limit]
    print(f"[{name}#{a.shard}] {len(mine)} assigned, {len(mine) - len(todo)} done/skipped, {len(todo)} todo", flush=True)
    if not todo:
        return

    engine = ENGINES[a.engine](a)
    tmp = Path(tempfile.mkdtemp()) / "page.png"

    has_alarm = hasattr(signal, "SIGALRM")
    if has_alarm:
        def on_timeout(signum, frame):
            raise TimeoutError(f"exceeded {a.page_timeout}s")
        signal.signal(signal.SIGALRM, on_timeout)

    t_start = time.time()
    for n, t in enumerate(todo, 1):
        book, page, t0 = t["book"], t["page"], time.time()
        (out / book).mkdir(parents=True, exist_ok=True)
        inflight.write_text(f"{book}|{page}", encoding="utf-8")
        if has_alarm:
            signal.alarm(a.page_timeout)
        try:
            img = render_page(t["src"], page, a.dpi, a.max_side)
            img.save(tmp)
            text, extra = engine.run(str(tmp))
            if has_alarm:
                signal.alarm(0)
            rec = {
                "page": page, "text": text, "w": img.width, "h": img.height,
                "sec": round(time.time() - t0, 1), **extra
            }
            with open(result_file(out, book, name, a.shard), "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            eta = (time.time() - t_start) / n * (len(todo) - n) / 60
            print(f"[{name}#{a.shard}] {n}/{len(todo)} {book} p.{page + 1}: {rec['sec']}s, "
                  f"{len(text)} chars | ETA ~{eta:.0f}m", flush=True)
        except Exception as e:  # noqa: BLE001
            if has_alarm:
                signal.alarm(0)
            if isinstance(e, TimeoutError):
                with open(out / book / f"{name}_skip.log", "a", encoding="utf-8") as f:
                    f.write(f"{page}\n")
            print(f"[{name}#{a.shard}] {book} p.{page + 1}: ERROR {type(e).__name__}: {str(e)[:200]}", flush=True)

    inflight.unlink(missing_ok=True)
    close_cached_docs()


if __name__ == "__main__":
    main()