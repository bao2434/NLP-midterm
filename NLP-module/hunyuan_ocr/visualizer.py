from __future__ import annotations

import json
import re
from pathlib import Path
from PIL import ImageDraw, ImageFont

from .config import DEFAULT_LAYOUT_PROMPT, OCRConfig
from .worker import load_results, render_page

COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#00a0b0", "#f032e6", "#9a6324", "#808000", "#000075"
]


def sample_demo_pages(books: dict[str, tuple[str, list[int]]], n: int) -> list[dict]:
    """Select n middle pages across books for layout inspection."""
    out = []
    for frac in (1 / 3, 1 / 2, 2 / 3, 1 / 4, 3 / 4):
        for b, (src, sel) in books.items():
            t = {"book": b, "src": src, "page": sel[int(len(sel) * frac)]} if sel else None
            if t and t not in out and len(out) < n:
                out.append(t)
    return out


def parse_layout(text: str, w: int, h: int) -> list[tuple[str, list[float]]]:
    """Parse normalized (0-1000) bbox coordinates into pixel coordinates."""
    out = []
    for obj in re.findall(r"\{[^{}]*\}", text):
        label = re.search(r'"layout_type"\s*:\s*"([^"]*)"', obj)
        pts = [(int(x), int(y)) for x, y in re.findall(r"\((\d+),\s*(\d+)\)", obj)]
        if pts:
            xs = [x * w / 1000 for x, _ in pts]
            ys = [y * h / 1000 for _, y in pts]
            out.append((label.group(1) if label else "box", [min(xs), min(ys), max(xs), max(ys)]))
    return out


def draw_boxes(task: dict, boxes: list[tuple[str, list[float]]], cfg: OCRConfig) -> Path:
    """Draw reading-order bounding boxes onto the rendered page and save to layout_dir."""
    img = render_page(task["src"], task["page"], cfg.dpi, cfg.max_side)
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=max(14, img.width // 60))
    labels = sorted({l for l, _ in boxes})

    for i, (label, box) in enumerate(boxes, 1):
        c = COLORS[labels.index(label) % len(COLORS)]
        d.rectangle(box, outline=c, width=3)
        d.text((box[0] + 4, box[1] + 2), f"{i}.{label}", fill=c, font=font)

    cfg.layout_dir.mkdir(parents=True, exist_ok=True)
    out_path = cfg.layout_dir / f"{cfg.engine}_{task['book']}_p{task['page'] + 1}.png"
    img.save(out_path)
    summary = ", ".join(f"{i}.{l}" for i, (l, _) in enumerate(boxes, 1))
    print(f"{task['book']} — p.{task['page'] + 1}: {len(boxes)} boxes: {summary}")
    return out_path


def run_layout_preview(pipeline, books: dict[str, tuple[str, list[int]]]) -> list[Path]:
    """Run layout detection on sample pages and save annotated images."""
    cfg = pipeline.cfg
    demo = sample_demo_pages(books, cfg.n_layout_samples)
    if not demo:
        return []

    demo_tasks_path = cfg.work_dir / "demo.jsonl"
    with open(demo_tasks_path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(t, ensure_ascii=False) + "\n" for t in demo)

    layout_name = f"{cfg.engine}_layout"
    pipeline.run_workers(
        tasks_file=demo_tasks_path,
        name=layout_name,
        budget_hours=1.0,
        extra_args=["--prompt", DEFAULT_LAYOUT_PROMPT, "--max-new", "3000"],
    )

    saved_images = []
    for t in demo:
        rec = load_results(cfg.raw_dir, t["book"], layout_name).get(t["page"])
        if rec:
            boxes = parse_layout(rec["text"], rec["w"], rec["h"])
            saved_images.append(draw_boxes(t, boxes, cfg))
    return saved_images