"""Bertalign + Vecalign + CLC API (3-way voting) for an existing Chinese/Vietnamese translation.

Put this file in NLP-module/alignment_tool/ beside HanVietAligner.py.
The *same* sentence lists are passed to every engine; results are mapped by integer indices.
Only EXACT same (source-index tuple, target-index tuple) from >=2 of the 3
SUCCESSFULLY executed methods are accepted. Never substitutes one engine for another.
No text is translated or silently corrected. See 3_dong_hang_voting_nu_khoa.ipynb.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from typing import Any

METHODS = ("bertalign", "vecalign", "api")
SENT_HAN = re.compile(r"(?<=[。！？；])")
SENT_VI = re.compile(r"(?<=[.!?;])\s+(?=\S)|(?<=[。！？；])")
VEC_RE = re.compile(r"^\[(.*?)\]:\[(.*?)\]:(-?[\d.eE+]+)$")


@dataclass
class EnsembleResult:
    output_dir: Path
    methods_ok: dict[str, bool]
    method_errors: dict[str, str]
    stats: dict[str, Any]
    accepted: list[dict[str, Any]] = field(default_factory=list)
    review: list[dict[str, Any]] = field(default_factory=list)
    rows: list[dict[str, Any]] = field(default_factory=list)
    disagreements: list[dict[str, Any]] = field(default_factory=list)


def _format_model_proposals(proposals: list[dict[str, Any]]) -> str:
    if not proposals:
        return "(no alignment)"
    parts = []
    for item in proposals:
        h_label = ",".join(str(i) for i in item["h_ids"])
        v_label = ",".join(str(j) for j in item["v_ids"])
        parts.append(f"H[{h_label}] <-> V[{v_label}]: {_plain(item['han'])} ||| {_plain(item['viet'])}")
    return "; ".join(parts)


def split_parallel_texts_by_lines(
    han_text: str,
    vi_text: str,
    max_chars: int = 5000,
    *,
    max_units: int | None = 80,
    split_mode: str = "sentences",
) -> list[tuple[str, str]]:
    """Split a (han, vi) text pair into smaller pairs at newline boundaries.

    Guarantees:
    - Cuts only at line boundaries (never mid-line).
    - Every returned pair has len(han_part) + len(vi_part) <= max_chars
      (unless a single Han line + a single Vi line alone already exceed max_chars,
      in which case lines are kept intact rather than cut mid-line).
    - When max_units is set (default 80, matching CLC API's max_units_per_request),
      also keeps each side's unit count <= max_units (unless a single line alone
      exceeds max_units).
    - Every returned pair is non-empty on both sides.
    """
    if max_chars < 1:
        raise ValueError("max_chars must be >= 1")
    if max_units is not None and max_units < 1:
        raise ValueError("max_units must be >= 1 or None")

    han_lines = [line for line in han_text.replace("\r\n", "\n").replace("\r", "\n").splitlines() if line.strip()]
    vi_lines = [line for line in vi_text.replace("\r\n", "\n").replace("\r", "\n").splitlines() if line.strip()]
    if not han_lines or not vi_lines:
        raise ValueError("Han or Vietnamese text is empty after stripping blank lines")

    def _line_unit_count(line: str, lang: str) -> int:
        if split_mode == "lines":
            return 1
        return max(1, len(split_sentences(line, lang)))

    han_units = [_line_unit_count(line, "han") for line in han_lines]
    vi_units = [_line_unit_count(line, "vi") for line in vi_lines]

    han_joined = "\n".join(han_lines)
    vi_joined = "\n".join(vi_lines)
    units_ok_whole = (
        max_units is None
        or (sum(han_units) <= max_units and sum(vi_units) <= max_units)
    )
    if len(han_joined) + len(vi_joined) <= max_chars and units_ok_whole:
        return [(han_joined, vi_joined)]

    def _fits(h_len: int, v_len: int, h_u: int, v_u: int) -> bool:
        if h_len + v_len > max_chars:
            return False
        if max_units is not None and (h_u > max_units or v_u > max_units):
            return False
        return True

    parts: list[tuple[str, str]] = []

    # Case 1: Same line count -> lockstep line-by-line grouping.
    if len(han_lines) == len(vi_lines):
        cur_h: list[str] = []
        cur_v: list[str] = []
        cur_h_len = cur_v_len = 0
        cur_h_u = cur_v_u = 0
        for hl, vl, hu, vu in zip(han_lines, vi_lines, han_units, vi_units):
            add_h = (1 if cur_h else 0) + len(hl)
            add_v = (1 if cur_v else 0) + len(vl)
            if cur_h and not _fits(cur_h_len + add_h, cur_v_len + add_v, cur_h_u + hu, cur_v_u + vu):
                parts.append(("\n".join(cur_h), "\n".join(cur_v)))
                cur_h, cur_v = [hl], [vl]
                cur_h_len, cur_v_len = len(hl), len(vl)
                cur_h_u, cur_v_u = hu, vu
            else:
                cur_h.append(hl)
                cur_v.append(vl)
                cur_h_len += add_h
                cur_v_len += add_v
                cur_h_u += hu
                cur_v_u += vu
        if cur_h or cur_v:
            parts.append(("\n".join(cur_h), "\n".join(cur_v)))
        return parts

    # Case 2: Different line counts -> advance proportionally by cumulative char progress,
    # cutting strictly at newline boundaries and keeping both sides non-empty per part.
    nh, nv = len(han_lines), len(vi_lines)
    han_total = sum(len(line) + 1 for line in han_lines)
    vi_total = sum(len(line) + 1 for line in vi_lines)
    han_cum = [0.0]
    for line in han_lines:
        han_cum.append(han_cum[-1] + (len(line) + 1) / han_total)
    vi_cum = [0.0]
    for line in vi_lines:
        vi_cum.append(vi_cum[-1] + (len(line) + 1) / vi_total)

    hi = vi_i = 0
    while hi < nh or vi_i < nv:
        cur_h = []
        cur_v = []
        cur_h_len = cur_v_len = 0
        cur_h_u = cur_v_u = 0

        # Ensure at least one line from each side that still has remaining lines.
        if hi < nh:
            cur_h.append(han_lines[hi])
            cur_h_len = len(han_lines[hi])
            cur_h_u = han_units[hi]
            hi += 1
        if vi_i < nv:
            cur_v.append(vi_lines[vi_i])
            cur_v_len = len(vi_lines[vi_i])
            cur_v_u = vi_units[vi_i]
            vi_i += 1

        while hi < nh or vi_i < nv:
            rem_h_len = sum(len(line) + 1 for line in han_lines[hi:]) if hi < nh else 0
            rem_v_len = sum(len(line) + 1 for line in vi_lines[vi_i:]) if vi_i < nv else 0
            rem_h_u = sum(han_units[hi:]) if hi < nh else 0
            rem_v_u = sum(vi_units[vi_i:]) if vi_i < nv else 0
            if _fits(cur_h_len + rem_h_len, cur_v_len + rem_v_len, cur_h_u + rem_h_u, cur_v_u + rem_v_u):
                if hi < nh:
                    cur_h.extend(han_lines[hi:])
                    cur_h_len += rem_h_len
                    cur_h_u += rem_h_u
                    hi = nh
                if vi_i < nv:
                    cur_v.extend(vi_lines[vi_i:])
                    cur_v_len += rem_v_len
                    cur_v_u += rem_v_u
                    vi_i = nv
                break

            can_h = hi < nh and not (hi == nh - 1 and vi_i < nv)
            can_v = vi_i < nv and not (vi_i == nv - 1 and hi < nh)

            if can_h and can_v:
                side = "han" if han_cum[hi] <= vi_cum[vi_i] else "vi"
            elif can_h:
                side = "han"
            elif can_v:
                side = "vi"
            else:
                break

            if side == "han":
                add = (1 if cur_h else 0) + len(han_lines[hi])
                if _fits(cur_h_len + add, cur_v_len, cur_h_u + han_units[hi], cur_v_u):
                    cur_h.append(han_lines[hi])
                    cur_h_len += add
                    cur_h_u += han_units[hi]
                    hi += 1
                else:
                    break
            else:
                add = (1 if cur_v else 0) + len(vi_lines[vi_i])
                if _fits(cur_h_len, cur_v_len + add, cur_h_u, cur_v_u + vi_units[vi_i]):
                    cur_v.append(vi_lines[vi_i])
                    cur_v_len += add
                    cur_v_u += vi_units[vi_i]
                    vi_i += 1
                else:
                    break

        parts.append(("\n".join(cur_h), "\n".join(cur_v)))

    return parts


def write_parallel_parts(
    han_text: str,
    vi_text: str,
    parts_dir: str | Path,
    *,
    stem: str = "part",
    max_chars: int = 5000,
    max_units: int | None = 80,
    split_mode: str = "sentences",
) -> list[tuple[Path, Path]]:
    """Split a (han, vi) text pair at newline boundaries and write part .txt files."""
    parts_dir = Path(parts_dir)
    parts_dir.mkdir(parents=True, exist_ok=True)
    # Remove stale part files from previous runs with a different split count.
    for old in parts_dir.glob(f"{stem}_*_part_*.txt"):
        if old.is_file():
            old.unlink()

    pairs = split_parallel_texts_by_lines(
        han_text, vi_text, max_chars=max_chars, max_units=max_units, split_mode=split_mode
    )
    written: list[tuple[Path, Path]] = []
    for idx, (han_part, vi_part) in enumerate(pairs, start=1):
        h_path = parts_dir / f"{stem}_han_part_{idx:03d}.txt"
        v_path = parts_dir / f"{stem}_vi_part_{idx:03d}.txt"
        h_path.write_text(han_part, encoding="utf-8")
        v_path.write_text(vi_part, encoding="utf-8")
        written.append((h_path, v_path))
    return written


def write_alignment_outputs(
    out_dir: str | Path,
    output_rows: list[dict[str, Any]],
    disagreements: list[dict[str, Any]],
    *,
    all_ok: bool,
    output_name: str | None = None,
) -> tuple[Path, Path, Path]:
    """Write TSV, XLSX (DONG_HANG), and note XLSX (NOTES + DISAGREEMENTS) to out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_stem = output_name if output_name else "nu_khoa_parallel"
    stem = base_stem if all_ok else f"{base_stem}.PARTIAL"
    tsv_path = out_dir / (stem + ".tsv")
    xlsx_path = out_dir / (stem + ".xlsx")

    old_stem = f"{base_stem}.PARTIAL" if all_ok else base_stem
    for extension in (".tsv", ".xlsx"):
        old = out_dir / (old_stem + extension)
        if old.exists():
            old.unlink()

    with tsv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        for row in output_rows:
            writer.writerow([_plain(row["han"]), _plain(row["viet"])])

    from openpyxl import Workbook

    wb = Workbook()
    sheet = wb.active
    sheet.title = "DONG_HANG"
    sheet.append(["Hán", "Việt"])
    for row in output_rows:
        sheet.append([_plain(row["han"]), _plain(row["viet"])])
    sheet.column_dimensions["A"].width = 65
    sheet.column_dimensions["B"].width = 85
    wb.save(xlsx_path)

    notes_path = out_dir / (f"{output_name}_note.xlsx" if output_name else "notes.xlsx")
    notes = Workbook()
    notes_sheet = notes.active
    notes_sheet.title = "NOTES"
    notes_sheet.append(["Tên tác phẩm", "Câu Hán", "Câu Việt", "Lỗi (mô tả ngắn gọn)", "Chỉnh sửa (ghi chú đề xuất)"])
    notes_sheet.column_dimensions["A"].width = 32
    notes_sheet.column_dimensions["B"].width = 70
    notes_sheet.column_dimensions["C"].width = 70
    notes_sheet.column_dimensions["D"].width = 55
    notes_sheet.column_dimensions["E"].width = 85

    disagree_sheet = notes.create_sheet(title="DISAGREEMENTS")
    disagree_sheet.append(["ID", "Hán IDs", "Việt IDs", "Câu Hán", "Câu Việt", *METHODS])
    disagree_sheet.column_dimensions["A"].width = 16
    disagree_sheet.column_dimensions["B"].width = 14
    disagree_sheet.column_dimensions["C"].width = 14
    disagree_sheet.column_dimensions["D"].width = 55
    disagree_sheet.column_dimensions["E"].width = 55
    disagree_sheet.column_dimensions["F"].width = 65
    disagree_sheet.column_dimensions["G"].width = 65
    disagree_sheet.column_dimensions["H"].width = 65

    for item in disagreements:
        part_prefix = f"Part {item['part']} | " if item.get("part") is not None else ""
        model_summary = "\n".join(
            f"{method}: {item['proposals_text'][method]}" for method in METHODS
        )
        notes_sheet.append([
            item["id"],
            item["han"],
            item["viet"],
            f"[{item['id']}] {part_prefix}3-way disagreement (H={item['h_ids']}, V={item['v_ids']})",
            model_summary,
        ])
        disagree_sheet.append([
            item["id"],
            ",".join(str(i) for i in item["h_ids"]),
            ",".join(str(j) for j in item["v_ids"]),
            item["han"],
            item["viet"],
            *(item["proposals_text"][method] for method in METHODS),
        ])
    notes.save(notes_path)
    return tsv_path, xlsx_path, notes_path


def _plain(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def cut_section(text: str, start_marker: str, end_marker: str | None = None,
                body_marker: str | None = None) -> str:
    """Extract one COMMON section; fail closed when an anchor is missing.

    body_marker is an additional start marker AFTER start_marker, e.g. 'Đàn bà'.
    It is INCLUDED in the output. end_marker is EXCLUDED.
    """
    start = text.find(start_marker)
    if start < 0:
        raise ValueError(f"Không tìm thấy mốc BẮT ĐẦU {start_marker!r}. Kiểm tra TXT/encoding.")
    start += len(start_marker)
    end = len(text)
    if end_marker:
        end = text.find(end_marker, start)
        if end < 0:
            raise ValueError(f"Không tìm thấy mốc KẾT THÚC {end_marker!r}. "
                             "Hãy kiểm tra tên mục kế tiếp, không chia sách bằng tỉ lệ ký tự.")
    if body_marker:
        position = text.find(body_marker, start, end)
        if position < 0:
            raise ValueError(f"Không thấy body_marker={body_marker!r} trong mục đã chọn")
        start = position
    section = text[start:end].strip()
    if not section:
        raise ValueError("Mục trích xuất rỗng")
    return section


def split_sentences(text: str, language: str) -> list[str]:
    """Heuristic splitter. One physical row is not necessarily one sentence.

    Intended for text from the same common chapter, not entire unmatched books.
    """
    if language not in {"han", "vi"}:
        raise ValueError("language = han hoặc vi")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    segments: list[str] = []
    splitter = SENT_HAN if language == "han" else SENT_VI
    for para in re.split(r"\n\s*\n", text):
        lines = [line.strip() for line in para.splitlines() if line.strip()]
        if not lines:
            continue
        # Preserve paragraph boundaries. OCR line wraps are not sentence boundaries.
        merged = ("" if language == "han" else " ").join(lines)
        for item in splitter.split(merged):
            item = _plain(item)
            if item:
                segments.append(item)
    return segments


def _as_indices(x: Any, n: int, what: str) -> tuple[int, ...]:
    if x is None:
        return ()
    if isinstance(x, int):
        x = [x]
    if not isinstance(x, (list, tuple, range)):
        raise ValueError(f"{what}: danh sách index không hợp lệ {x!r}")
    out = tuple(int(k) for k in x)
    if out and (sorted(set(out)) != list(out) or out != tuple(range(out[0], out[-1] + 1))):
        raise ValueError(f"{what}: index không liên tiếp, trùng hoặc ngược thứ tự: {out}")
    if any(k < 0 or k >= n for k in out):
        raise ValueError(f"{what}: index ngoài phạm vi: {out}")
    return out


def _bead(h: Any, v: Any, nh: int, nv: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    return _as_indices(h, nh, "Hán"), _as_indices(v, nv, "Việt")


def validate_beads(beads: list[tuple[tuple[int, ...], tuple[int, ...]]], nh: int, nv: int):
    used_h, used_v = set(), set()
    prev_h = prev_v = -1
    for h, v in beads:
        _bead(h, v, nh, nv)
        if not h or not v:  # Vecalign allows insertions/deletions, record gaps separately.
            continue
        if h[0] <= prev_h or v[0] <= prev_v:
            raise ValueError("Beads bị đảo thứ tự/chéo nhau")
        if used_h.intersection(h) or used_v.intersection(v):
            raise ValueError("Beads dùng trùng câu")
        prev_h, prev_v = h[-1], v[-1]
        used_h.update(h)
        used_v.update(v)


class EnsembleHanVietAligner:
    def __init__(self, *, vec_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                 overlap_n: int = 4, batch_size: int = 16,
                 api_token: str | None = None, api_interval_s: float = 3.0):
        self.vec_model = vec_model
        self.overlap_n = overlap_n
        self.batch_size = batch_size
        self.api_token = api_token
        self.api_interval_s = api_interval_s
        if overlap_n < 1 or batch_size < 1:
            raise ValueError("overlap_n và batch_size phải > 0")

    def run_bertalign(self, han: list[str], vi: list[str]):
        import faiss
        import numpy as np
        from bertalign import Bertalign
        import bertalign.aligner as bert_module
        import bertalign.corelib as bert_corelib
        # Official Bertalign's auto-detector supports 25 languages and does not include
        # Vietnamese. Upstream also uses a live Google Translate detection call.
        # Both texts' languages are KNOWN, so supply zh/vi for this *one* constructor.
        # is_split=True means Vietnamese never goes through the unsupported splitter.
        original_detector = bert_module.detect_lang
        known_languages = iter(("zh", "vi"))
        try:
            bert_module.detect_lang = lambda _text: next(known_languages)
            aligner = Bertalign("\n".join(han), "\n".join(vi), is_split=True)
        finally:
            bert_module.detect_lang = original_detector

        # When running on a GPU machine (e.g. Kaggle GPU) with faiss-cpu installed,
        # Bertalign's find_top_k_sents calls faiss.StandardGpuResources() and crashes.
        # Fall back to exact CPU faiss.IndexFlatIP while keeping GPU embedding encoding.
        patched_faiss_cpu = not hasattr(faiss, "StandardGpuResources")
        orig_core_topk = getattr(bert_corelib, "find_top_k_sents", None)
        orig_aligner_topk = getattr(bert_module, "find_top_k_sents", None)
        if patched_faiss_cpu:
            def _cpu_find_top_k_sents(src_vecs, tgt_vecs, k=3):
                src = np.ascontiguousarray(src_vecs, dtype=np.float32)
                tgt = np.ascontiguousarray(tgt_vecs, dtype=np.float32)
                index = faiss.IndexFlatIP(src.shape[1])
                index.add(tgt)
                return index.search(src, k)

            bert_corelib.find_top_k_sents = _cpu_find_top_k_sents
            if orig_aligner_topk is not None:
                bert_module.find_top_k_sents = _cpu_find_top_k_sents
        try:
            aligner.align_sents()
        finally:
            if patched_faiss_cpu:
                if orig_core_topk is not None:
                    bert_corelib.find_top_k_sents = orig_core_topk
                if orig_aligner_topk is not None:
                    bert_module.find_top_k_sents = orig_aligner_topk

        src_seen = getattr(aligner, "src_sents", None)
        tgt_seen = getattr(aligner, "tgt_sents", None)
        if src_seen is not None and [_plain(s) for s in src_seen] != [_plain(s) for s in han]:
            raise ValueError("Bertalign đã làm thay đổi ranh giới câu Hán; không thể voting theo index")
        if tgt_seen is not None and [_plain(s) for s in tgt_seen] != [_plain(s) for s in vi]:
            raise ValueError("Bertalign đã làm thay đổi ranh giới câu Việt; không thể voting theo index")
        beads = [_bead(x[0], x[1], len(han), len(vi)) for x in aligner.result]
        validate_beads(beads, len(han), len(vi))
        return [b for b in beads if b[0] and b[1]]

    def run_vecalign(self, han: list[str], vi: list[str], work_dir: Path):
        import numpy as np
        from sentence_transformers import SentenceTransformer
        work_dir.mkdir(parents=True, exist_ok=True)
        model = SentenceTransformer(self.vec_model)
        paths = {}
        for label, sentences in (("han", han), ("viet", vi)):
            source = work_dir / f"{label}.sentences.txt"
            overlap = work_dir / f"{label}.overlaps.txt"
            emb = work_dir / f"{label}.overlaps.emb"
            source.write_text("\n".join(sentences) + "\n", encoding="utf-8")
            # Use actual Vecalign overlap generator (not an approximation).
            cmd = [sys.executable, "-m", "vecalign.overlap", "-i", str(source),
                   "-o", str(overlap), "-n", str(self.overlap_n)]
            subprocess.run(cmd, check=True, text=True, capture_output=True)
            overlap_lines = overlap.read_text(encoding="utf-8").splitlines()
            if not overlap_lines:
                raise ValueError(f"Vecalign không tạo được overlaps cho {label}")
            vectors = model.encode(overlap_lines, batch_size=self.batch_size,
                                   show_progress_bar=False, convert_to_numpy=True)
            np.asarray(vectors, dtype=np.float32).tofile(emb)
            paths[label] = (source, overlap, emb)
        hs, ho, he = paths["han"]
        vs, vo, ve = paths["viet"]
        cmd = [sys.executable, "-m", "vecalign.vecalign", "--alignment_max_size", "8",
               "--src", str(hs), "--tgt", str(vs),
               "--src_embed", str(ho), str(he),
               "--tgt_embed", str(vo), str(ve)]
        proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
        (work_dir / "vecalign_raw.txt").write_text(proc.stdout, encoding="utf-8")
        beads = []
        for line in proc.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            match = VEC_RE.fullmatch(line)
            if not match:
                raise ValueError(f"Không nhận diện được dòng Vecalign: {line!r}")
            def nums(s):
                return [int(part.strip()) for part in s.split(",") if part.strip()]
            beads.append(_bead(nums(match.group(1)), nums(match.group(2)), len(han), len(vi)))
        validate_beads(beads, len(han), len(vi))
        return [b for b in beads if b[0] and b[1]]

    def run_api(self, han: list[str], vi: list[str], *, api_response: dict | None = None):
        """Dóng bằng CLC và ÁNH XẠ về đúng unit đầu vào.

        api_response=None: gọi API thật; dict: phát lại response đã lưu,
        không gọi mạng, kiểm tra nguyên văn text để chống dùng nhầm dataset.
        """
        try:
            from .HanVietAligner import HanVietAligner, AlignmentError, _parse_api_result
        except ImportError:
            from HanVietAligner import HanVietAligner, AlignmentError, _parse_api_result

        if api_response is None:
            client = HanVietAligner(token=self.api_token, split_mode="lines",
                                   min_interval_s=self.api_interval_s)
            rows = client.align_texts("\n".join(han), "\n".join(vi))
            self._last_api_trace = client.last_trace
            self._last_api_mode = "LIVE"
        else:
            if not isinstance(api_response, dict):
                raise ValueError("api_response phải là dict JSON")
            if api_response.get("is_success") is not True or api_response.get("code") != "000000":
                raise AlignmentError("API response replay không thành công; không dùng để voting")
            data = api_response.get("data")
            if not isinstance(data, dict) or not isinstance(data.get("result"), list):
                raise AlignmentError("API response replay thiếu data.result (list)")
            rows, trace = _parse_api_result(data["result"], han, vi)
            self._last_api_trace = trace
            self._last_api_mode = "REPLAY_VERIFIED"
        self._last_api_rows = rows
        beads = [_bead(range(item["han_start"], item["han_end"] + 1),
                       range(item["vi_start"], item["vi_end"] + 1), len(han), len(vi))
                 for item in rows]
        validate_beads(beads, len(han), len(vi))
        return beads

    @staticmethod
    def build_output_rows(
        han: list[str],
        vi: list[str],
        outputs: dict[str, list],
        accepted: list[dict[str, Any]],
        all_ok: bool,
        *,
        disagreement_start_index: int = 1,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Interleave accepted (>=2/3) pairs with in-place 3-way disagreement placeholders.

        When all 3 engines succeeded, any gap between consecutive accepted beads is a region
        where all three models failed to reach >=2/3 agreement. Each gap is assigned a unique
        ID (DISAGREE_001, ...), placed in-order in `rows`, and recorded in `disagreements`
        with what each model proposed for that region.
        """
        if not all_ok:
            return [], []

        rows: list[dict[str, Any]] = []
        disagreements: list[dict[str, Any]] = []
        prev_h = -1
        prev_v = -1

        def append_gap(h_start: int, h_end: int, v_start: int, v_end: int) -> None:
            h_gap = list(range(h_start, h_end))
            v_gap = list(range(v_start, v_end))
            if not h_gap and not v_gap:
                return
            disagreement_id = f"DISAGREE_{disagreement_start_index + len(disagreements):03d}"
            h_gap_set = set(h_gap)
            v_gap_set = set(v_gap)
            proposals: dict[str, list[dict[str, Any]]] = {}
            proposals_text: dict[str, str] = {}
            for method in METHODS:
                matched = []
                for bh, bv in outputs.get(method, []):
                    if (h_gap_set & set(bh)) or (v_gap_set & set(bv)):
                        matched.append({
                            "h_ids": list(bh),
                            "v_ids": list(bv),
                            "han": " ".join(han[i] for i in bh),
                            "viet": " ".join(vi[j] for j in bv),
                        })
                proposals[method] = matched
                proposals_text[method] = _format_model_proposals(matched)

            han_text = _plain(" ".join(han[i] for i in h_gap)) if h_gap else ""
            vi_text = _plain(" ".join(vi[j] for j in v_gap)) if v_gap else ""
            han_placeholder = (
                f"[{disagreement_id}] DISAGREEMENT: {han_text}"
                if han_text else f"[{disagreement_id}] DISAGREEMENT (empty)"
            )
            vi_placeholder = (
                f"[{disagreement_id}] DISAGREEMENT: {vi_text}"
                if vi_text else f"[{disagreement_id}] DISAGREEMENT (empty)"
            )

            entry = {
                "id": disagreement_id,
                "kind": "THREE_WAY_DISAGREEMENT",
                "h_ids": h_gap,
                "v_ids": v_gap,
                "han": han_text,
                "viet": vi_text,
                "proposals": proposals,
                "proposals_text": proposals_text,
            }
            disagreements.append(entry)
            rows.append({
                "kind": "DISAGREEMENT",
                "disagreement_id": disagreement_id,
                "h_ids": h_gap,
                "v_ids": v_gap,
                "han": han_placeholder,
                "viet": vi_placeholder,
                "votes": 0,
                "voters": [],
            })

        for item in accepted:
            append_gap(prev_h + 1, item["h_ids"][0], prev_v + 1, item["v_ids"][0])
            rows.append({**item, "kind": "ACCEPTED", "disagreement_id": None})
            prev_h = item["h_ids"][-1]
            prev_v = item["v_ids"][-1]

        append_gap(prev_h + 1, len(han), prev_v + 1, len(vi))
        return rows, disagreements

    @staticmethod
    def vote(han: list[str], vi: list[str], outputs: dict[str, list], errors: dict[str, str]):
        """Conservative exact-bead majority. 3/3 must have RUN, else nothing accepted."""
        counts: dict[tuple, set[str]] = defaultdict(set)
        for name, beads in outputs.items():
            for item in beads:
                if item[0] and item[1]:
                    counts[item].add(name)
        all_ok = all(method in outputs for method in METHODS)
        candidates = sorted((k for k, voters in counts.items() if len(voters) >= 2),
                            key=lambda b: (b[0][0], b[1][0]))
        accepted, taken_h, taken_v, last_h, last_v = [], set(), set(), -1, -1
        for h, v in candidates:
            if set(h) & taken_h or set(v) & taken_v or h[0] <= last_h or v[0] <= last_v:
                continue  # conflicting candidate needs manual review
            if not all_ok:
                continue  # NEVER label 2/2 agreement as 2/3 voting
            accepted.append({"h_ids": list(h), "v_ids": list(v),
                             "han": " ".join(han[i] for i in h),
                             "viet": " ".join(vi[j] for j in v),
                             "votes": len(counts[(h, v)]), "voters": sorted(counts[(h, v)])})
            taken_h.update(h); taken_v.update(v)
            last_h, last_v = h[-1], v[-1]
        _, disagreements = EnsembleHanVietAligner.build_output_rows(han, vi, outputs, accepted, all_ok)
        accepted_keys = {(tuple(row['h_ids']), tuple(row['v_ids'])) for row in accepted}
        reviews: list[dict[str, Any]] = list(disagreements)
        for (h, v), voters in sorted(counts.items(), key=lambda it:(it[0][0][0], it[0][1][0])):
            if (h, v) not in accepted_keys:
                reviews.append({"kind": "DISAGREEMENT" if all_ok else "INCOMPLETE_ENGINES",
                                "h_ids": list(h), "v_ids": list(v),
                                "han": " ".join(han[i] for i in h),
                                "viet": " ".join(vi[j] for j in v),
                                "votes": len(voters), "voters": sorted(voters)})
        for i, sentence in enumerate(han):
            if i not in taken_h:
                reviews.append({"kind": "UNACCEPTED_HAN", "h_ids": [i], "han": sentence})
        for j, sentence in enumerate(vi):
            if j not in taken_v:
                reviews.append({"kind": "UNACCEPTED_VI", "v_ids": [j], "viet": sentence})
        stats = {"status": "VOTING_READY_REVIEW" if all_ok else "INCOMPLETE_METHODS",
                 "engines_ok": sorted(outputs), "engines_failed": errors,
                 "han_sentences": len(han), "vi_sentences": len(vi),
                 "accepted_pairs": len(accepted),
                 "disagreement_blocks": len(disagreements),
                 "review_items": len(reviews),
                 "accepted_han_coverage": round(len(taken_h) / len(han), 4) if han else 0,
                 "accepted_vi_coverage": round(len(taken_v) / len(vi), 4) if vi else 0,
                 "note": "Voting >=2/3 is NOT semantic accuracy >=95%. Manually review and compute gold evaluation."}
        return accepted, reviews, stats

    def align_section(self, han: list[str], vi: list[str], out_dir: str | Path,
                      *, enable_api: bool = False, enable_bertalign: bool = True,
                      enable_vecalign: bool = True,
                      api_response: dict | str | Path | None = None,
                      output_name: str | None = None,
                      disagreement_start_index: int = 1) -> EnsembleResult:
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        meta_dir = (out_dir / "_details" / output_name) if output_name else out_dir
        meta_dir.mkdir(parents=True, exist_ok=True)
        # Không giữ nhầm đề xuất / mapping từ lần chạy trước nếu engine mới lỗi.
        for filename in [*(f"{method}_proposals.json" for method in METHODS), "api_mapping.json"]:
            candidate = meta_dir / filename
            if candidate.is_file():
                candidate.unlink()
        if not han or not vi:
            raise ValueError("Không đủ câu Hán và Việt")
        if any("\n" in item or "\t" in item for item in han + vi):
            raise ValueError("Mỗi đơn vị phải là một dòng vật lý, không chứa tab")
        input_hash = hashlib.sha256(json.dumps({"han": han, "vi": vi},
                             ensure_ascii=False).encode("utf-8")).hexdigest()
        (meta_dir / "input_sentences.json").write_text(json.dumps({
            "han": han, "viet": vi, "sha256": input_hash}, ensure_ascii=False, indent=2), encoding="utf-8")
        # Cho phép kiểm thử offline bằng response API đã lưu, KHÔNG gọi API/quota.
        if api_response is not None:
            if not enable_api:
                raise ValueError("api_response được cung cấp nhưng enable_api=False")
            if isinstance(api_response, (str, Path)):
                api_response = json.loads(Path(api_response).read_text(encoding="utf-8"))
            if not isinstance(api_response, dict):
                raise ValueError("api_response phải là dict hoặc đường dẫn JSON")
        outputs = {}; errors = {}
        opts = {"bertalign": enable_bertalign, "vecalign": enable_vecalign, "api": enable_api}
        for name in METHODS:
            if not opts[name]:
                errors[name] = "DISABLED"; continue
            try:
                if name == "bertalign":
                    beads = self.run_bertalign(han, vi)
                elif name == "vecalign":
                    beads = self.run_vecalign(han, vi, meta_dir / "vecalign_files")
                else:
                    beads = self.run_api(han, vi, api_response=api_response)
                validate_beads(beads, len(han), len(vi))
                outputs[name] = beads
                if name == "api":
                    # Bằng chứng cho việc quy đổi native index -> index chung.
                    (meta_dir / "api_mapping.json").write_text(json.dumps({
                        "mode": self._last_api_mode,
                        "trace": self._last_api_trace,
                        "mapped_rows": self._last_api_rows,
                    }, ensure_ascii=False, indent=2), encoding="utf-8")
                (meta_dir / f"{name}_proposals.json").write_text(json.dumps([
                    {"h_ids": list(h), "v_ids": list(v)} for h, v in beads
                ], ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception as exc:
                errors[name] = f"{type(exc).__name__}: {exc}"
                print(f"[{name}] LỖI: {errors[name]}")
        accepted, review, stats = self.vote(han, vi, outputs, errors)
        all_ok = len(outputs) == len(METHODS)
        output_rows, disagreements = self.build_output_rows(
            han, vi, outputs, accepted, all_ok,
            disagreement_start_index=disagreement_start_index,
        )
        write_alignment_outputs(
            out_dir, output_rows, disagreements,
            all_ok=all_ok, output_name=output_name,
        )

        (meta_dir / "review.jsonl").write_text("".join(json.dumps(item, ensure_ascii=False)+"\n"
                                                   for item in review), encoding="utf-8")
        (meta_dir / "vote_details.json").write_text(json.dumps(output_rows, ensure_ascii=False, indent=2), encoding="utf-8")
        (meta_dir / "report.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        return EnsembleResult(
            out_dir,
            {k: k in outputs for k in METHODS},
            errors,
            stats,
            accepted,
            review,
            output_rows,
            disagreements,
        )
