"""Dóng hàng câu/đoạn Hán – Việt *theo nghĩa* bằng API Kim Hán Nôm (CLC).

File độc lập, thiết kế để gọi tương tự HanVietTranslator.py; không cần GPU.

Cài đặt trên Kaggle / local::

    pip install requests openpyxl

Sử dụng với 2 thư mục chứa các cặp file cùng tên::

    from HanVietAligner import HanVietAligner

    aligner = HanVietAligner()  # hoặc HanVietAligner(token="...")
    result = aligner.align(
        han_dir="/kaggle/input/han-txt",
        viet_dir="/kaggle/input/viet-txt",
        work_dir="/kaggle/working",
        limit=2,                        # thử 2 lệnh gọi API chưa có checkpoint
    )
    print(result.output_dir, result.zip_path, result.book_stats)
    result = aligner.align("/kaggle/input/han-txt", "/kaggle/input/viet-txt",
                           "/kaggle/working")

Kaggle: chạy tiếp bằng cách Add Input Output phiên trước và truyền
resume_from="/kaggle/input/<ten-output-notebook>" (thư mục chứa align_raw).

Nếu tiếng Việt được sinh bởi HanVietTranslator.py, ưu tiên sử dụng checkpoint:

    result = aligner.align_from_translator(
        translation_work_dir="/kaggle/input/translation-output",
        work_dir="/kaggle/working",
    )

Thư mục translation_work_dir cần có chunks.jsonl và dich_raw/dich_shard*.jsonl;
ID của từng chunk cho biết cặp Hán/dịch tương ứng, nên không cắt đồng thời
hai cuốn sách theo vị trí tương đối (cách này dễ dóng hàng sai).

Đầu ra:
- dong_hang/<sach>_parallel.tsv: KHÔNG có header, chính xác 2 cột, phân cách TAB;
- dong_hang/<sach>_parallel.xlsx: sheet DONG_HANG, 2 cột Hán, Việt;
- dong_hang/<sach>_report.json: thống kê, độ bao phủ, các vấn đề cần kiểm tra;
- dong_hang/<sach>_review.jsonl: cặp điểm thấp, đơn vị bị bỏ sót, bất thường;
- align_raw/: checkpoint từng lượt gọi API để chạy tiếp mà không gọi lại;
- dong_hang.zip: gói kết quả.

Nếu giới hạn limit khiến chưa xử lý hết, file xuất có tên *.PARTIAL.tsv/xlsx.
Đọc trường status trong report trước khi dùng làm dữ liệu chính thức.

Theo tài liệu hướng dẫn dóng hàng Hán–Việt:
- Chấp nhận 1–1, 1–n, n–1, n–m; index kết thúc của API là INCLUSIVE.
- Không chuyển sang phiên âm / tính Levenshtein: đây là dóng hàng DỊCH NGHĨA.
- Không tự sửa câu Hán, câu Việt hoặc bịa cặp; giữ [MASK] khi OCR không rõ.
- Score chỉ dùng để ưu tiên kiểm tra, không phải xác suất cặp đúng.
- Nên rà soát thủ công những cặp điểm thấp hoặc đơn vị chưa được dóng.

Nguồn contract: TaiLieuAPI_HanNom_CLC.docx (sentence-alignment),
Hướng Dẫn Dóng Hàng câu Dịch nghĩa Hán - Việt.pdf, Quy_định_đồ_án.pdf.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import zipfile
import unicodedata
from collections import defaultdict
from shutil import copy2
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import requests


DEFAULT_BASE_URL = "https://kimhannom.fit.hcmus.edu.vn"
DEFAULT_ENDPOINT = "/api/web/clc-sinonom/sentence-alignment"
API_RESEGMENTATION_MAPPING_VERSION = 1  # Notebook kiểm tra tránh dùng module Kaggle cũ


class AlignmentError(RuntimeError):
    """Lỗi API hoặc dữ liệu dóng hàng không hợp lệ."""


class QuotaExceeded(AlignmentError):
    """Số lần gọi ghi nhận trong workspace đạt giới hạn của tài liệu API."""


@dataclass
class AlignmentConfig:
    base_url: str = DEFAULT_BASE_URL
    endpoint: str = DEFAULT_ENDPOINT
    token: str | None = None
    timeout_s: float = 90.0
    max_retries: int = 2
    min_interval_s: float = 3.0
    max_units_per_request: int = 80
    max_chars_per_request: int = 12000
    review_score_below: float = 0.45
    split_mode: Literal["sentences", "lines"] = "sentences"
    block_mode: Literal["auto", "whole", "paired_paragraphs"] = "whole"
    enforce_local_quotas: bool = True

    def validate(self) -> None:
        if not self.base_url.startswith(("https://", "http://")):
            raise ValueError("base_url phải có http:// hoặc https://")
        if not self.endpoint.startswith("/"):
            raise ValueError("endpoint phải bắt đầu bằng /")
        if self.timeout_s <= 0 or self.max_retries < 0 or self.min_interval_s < 0:
            raise ValueError("timeout_s, max_retries, min_interval_s không hợp lệ")
        if self.max_units_per_request < 1 or self.max_chars_per_request < 1:
            raise ValueError("max_units_per_request và max_chars_per_request phải > 0")
        if self.split_mode not in ("sentences", "lines"):
            raise ValueError("split_mode phải là sentences hoặc lines")
        if self.block_mode not in ("auto", "whole", "paired_paragraphs"):
            raise ValueError("block_mode không hợp lệ")


@dataclass
class AlignmentResult:
    output_dir: Path
    zip_path: Path
    book_stats: dict[str, dict[str, Any]] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


@dataclass
class _Block:
    han_units: list[str]
    vi_units: list[str]
    origin: str
    hit_max: bool = False


class HanVietAligner:
    """Wrapper API sentence-alignment có cache, kiểm tra phủ và xuất TSV/XLSX.

    Config có thể truyền trực tiếp bằng keyword, tương tự HanVietTranslator::

        HanVietAligner(min_interval_s=3, review_score_below=0.45)
        HanVietAligner(config=AlignmentConfig(...))

    API này dùng tài khoản/quota Lab (nếu server yêu cầu); không phải API
    miễn phí offline. Nếu server yêu cầu xác thực, đặt biến CLC_API_TOKEN
    hoặc truyền token=...; token không được ghi vào checkpoint.
    """

    def __init__(self, config: AlignmentConfig | None = None, **options: Any):
        if config is not None and options:
            raise ValueError("Chọn config=AlignmentConfig(...) HOẶC các tham số trực tiếp")
        self.config = config if config is not None else AlignmentConfig(**options)
        self.config.validate()
        self.session = requests.Session()
        self._last_request_monotonic: float | None = None
        self._quota_path: Path | None = None
        self.last_response: dict[str, Any] | None = None
        self.last_trace: dict[str, Any] | None = None

    @property
    def api_url(self) -> str:
        return self.config.base_url.rstrip("/") + self.config.endpoint

    def align_texts(self, han: str, vi: str) -> list[dict[str, Any]]:
        """Dóng hàng 1 khối văn bản (không checkpoint), trả về cặp và score.

        han, vi: văn bản gồm các câu; split_mode='lines' nếu đã tách sẵn
        mỗi câu trên một dòng. Những câu không tìm thấy cặp KHÔNG được bịa ra.
        """
        han_units = _split_units(han, self.config.split_mode, language="han")
        vi_units = _split_units(vi, self.config.split_mode, language="vi")
        self._check_size(han_units, vi_units)
        rows, _ = self._request_and_parse(han_units, vi_units)
        return rows

    def align(
        self,
        han_dir: str | Path,
        viet_dir: str | Path,
        work_dir: str | Path = "/kaggle/working",
        *,
        limit: int | None = None,
        resume_from: str | Path | None = None,
    ) -> AlignmentResult:
        """Dóng hàng *.txt từ hai thư mục, cặp sách ghép theo TÊN FILE.

        Với sách quá lớn mà không có ranh giới đoạn tương ứng, hàm sẽ từ
        chối tự chia ngẫu nhiên. Hãy cung cấp 2 bản chia đoạn cùng thứ tự,
        hoặc dùng align_from_translator() để tận dụng chunk ID gốc.

        limit là số lệnh gọi API CHƯA có trong checkpoint (không phải số sách).
        """
        if limit is not None and limit < 1:
            raise ValueError("limit phải >= 1 hoặc None")
        han_dir = Path(han_dir)
        viet_dir = Path(viet_dir)
        pairs = _match_text_files(han_dir, viet_dir)
        jobs: dict[str, list[_Block]] = {}
        for stem, (han_path, vi_path) in pairs.items():
            han = han_path.read_text(encoding="utf-8-sig")
            vi = vi_path.read_text(encoding="utf-8-sig")
            jobs[stem] = self._make_blocks(han, vi)
        return self._execute(jobs, Path(work_dir), limit, resume_from=resume_from)

    def align_book(
        self,
        han_file: str | Path,
        viet_file: str | Path,
        work_dir: str | Path = "/kaggle/working",
        *,
        book_name: str | None = None,
        limit: int | None = None,
        resume_from: str | Path | None = None,
    ) -> AlignmentResult:
        """Dóng hàng trực tiếp 1 cặp file .txt không nhất thiết cùng tên."""
        han_file, viet_file = Path(han_file), Path(viet_file)
        if not han_file.is_file() or not viet_file.is_file():
            raise FileNotFoundError("Không tìm thấy đủ file Hán và Việt")
        if limit is not None and limit < 1:
            raise ValueError("limit phải >= 1 hoặc None")
        han = han_file.read_text(encoding="utf-8-sig")
        vi = viet_file.read_text(encoding="utf-8-sig")
        name = book_name or han_file.stem
        return self._execute({name: self._make_blocks(han, vi)}, Path(work_dir),
                             limit, resume_from=resume_from)

    def align_from_translator(
        self,
        translation_work_dir: str | Path,
        work_dir: str | Path = "/kaggle/working",
        *,
        limit: int | None = None,
        resume_from: str | Path | None = None,
    ) -> AlignmentResult:
        """Dóng hàng từ checkpoint của HanVietTranslator (an toàn theo chunk ID).

        Đọc chunks.jsonl [{id, zh}] và dich_raw/dich_shard*.jsonl [{id, vi}].
        Không gửi đoạn chưa dịch hoặc có marker [chưa dịch] đến API.
        Tạo báo cáo nhắc người kiểm tra trường hit_max.
        """
        if limit is not None and limit < 1:
            raise ValueError("limit phải >= 1 hoặc None")
        root = Path(translation_work_dir)
        chunks_file = root / "chunks.jsonl"
        raw_dir = root / "dich_raw"
        if not chunks_file.is_file() or not raw_dir.is_dir():
            raise FileNotFoundError("Cần chunks.jsonl và dich_raw/ từ HanVietTranslator")
        translated: dict[str, dict[str, Any]] = {}
        for path in sorted(raw_dir.glob("dich_shard*.jsonl")):
            for item in _read_jsonl(path):
                if isinstance(item.get("id"), str) and isinstance(item.get("vi"), str):
                    translated[item["id"]] = item
        if not translated:
            raise AlignmentError("Không thấy bản dịch trong dich_raw/dich_shard*.jsonl")
        jobs: dict[str, list[_Block]] = defaultdict(list)
        for item in _read_jsonl(chunks_file):
            key = item.get("id", "")
            if not isinstance(key, str) or len(key.rsplit("|", 2)) != 3:
                raise AlignmentError(f"ID chunk không đúng format: {key!r}")
            name = key.rsplit("|", 2)[0]
            result = translated.get(key)
            zh = item.get("zh")
            if not isinstance(zh, str) or not zh.strip():
                continue
            if result is None or not result["vi"].strip() or "[chưa dịch]" in result["vi"].lower():
                jobs[name].append(_Block(
                    _split_units(zh, self.config.split_mode, "han"), [],
                    origin=f"{key}:untranslated"))
                continue
            vi = result["vi"]
            jobs[name].append(_Block(
                _split_units(zh, self.config.split_mode, "han"),
                _split_units(vi, self.config.split_mode, "vi"),
                origin=key, hit_max=bool(result.get("hit_max", False)),
            ))
        if not jobs:
            raise AlignmentError("Không thấy chunk nào hợp lệ trong chunks.jsonl")
        return self._execute(dict(jobs), Path(work_dir), limit,
                             resume_from=resume_from)

    def _make_blocks(self, han: str, vi: str) -> list[_Block]:
        hc = _paragraphs(han)
        vc = _paragraphs(vi)
        if not hc or not vc:
            raise AlignmentError("Bản Hán hoặc bản Việt rỗng")
        use_paras = self.config.block_mode == "paired_paragraphs" or (
            self.config.block_mode == "auto" and len(hc) > 1 and len(hc) == len(vc)
        )
        if use_paras:
            if len(hc) != len(vc):
                raise AlignmentError(
                    "Số đoạn Hán và Việt khác nhau; không thể ghép theo vị trí đoạn. "
                    "Hãy chia đúng đoạn tương ứng hoặc dùng align_from_translator()."
                )
            return [_Block(
                _split_units(h, self.config.split_mode, "han"),
                _split_units(v, self.config.split_mode, "vi"),
                origin=f"paragraph:{i}",
            ) for i, (h, v) in enumerate(zip(hc, vc))]
        # 1 lần gọi nếu cả hai văn bản đủ nhỏ; không cắt 2 ngôn ngữ bằng tỉ lệ ký tự
        block = _Block(_split_units(han, self.config.split_mode, "han"),
                       _split_units(vi, self.config.split_mode, "vi"), origin="whole_text")
        self._check_size(block.han_units, block.vi_units)
        return [block]

    def _check_size(self, han_units: list[str], vi_units: list[str]) -> None:
        if not han_units or not vi_units:
            raise AlignmentError("Một vế không có câu/đoạn hợp lệ để dóng hàng")
        if max(len(han_units), len(vi_units)) > self.config.max_units_per_request or (
            sum(map(len, han_units)) + sum(map(len, vi_units)) > self.config.max_chars_per_request
        ):
            raise AlignmentError(
                f"Khối quá lớn cho một lần gọi (Hán {len(han_units)} đơn vị, "
                f"Việt {len(vi_units)} đơn vị). Cần chia theo cặp đoạn/chương tương ứng "
                "hoặc dùng align_from_translator(); không tự cắt theo tỉ lệ ký tự."
            )

    def _headers(self) -> dict[str, str]:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; HanVietAligner/1.0; academic research)",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        token = self.config.token or os.environ.get("CLC_API_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def _request_and_parse(
        self, han_units: list[str], vi_units: list[str]
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        self._check_size(han_units, vi_units)
        payload = {"han": "\n".join(han_units), "vi": "\n".join(vi_units)}
        response_obj: dict[str, Any] | None = None
        for attempt in range(self.config.max_retries + 1):
            if self._last_request_monotonic is not None:
                elapsed = time.monotonic() - self._last_request_monotonic
                if elapsed < self.config.min_interval_s:
                    time.sleep(self.config.min_interval_s - elapsed)
            self._reserve_quota()
            self._last_request_monotonic = time.monotonic()
            try:
                response = self.session.post(
                    self.api_url, json=payload, headers=self._headers(),
                    timeout=(min(self.config.timeout_s, 15.0), self.config.timeout_s),
                )
                if response.status_code in (401, 403):
                    raise AlignmentError(
                        f"API HTTP {response.status_code}: cần quyền truy cập/token hợp lệ "
                        "(CLC_API_TOKEN hoặc token=...)."
                    )
                if response.status_code == 429 or 500 <= response.status_code <= 599:
                    if attempt < self.config.max_retries:
                        retry_after = response.headers.get("Retry-After")
                        try:
                            delay = min(float(retry_after), 60.0) if retry_after else 2 ** attempt * 3
                        except ValueError:
                            delay = 2 ** attempt * 3
                        time.sleep(max(1.0, delay))
                        continue
                response.raise_for_status()
                response_obj = response.json()
                break
            except (requests.Timeout, requests.ConnectionError) as exc:
                if attempt >= self.config.max_retries:
                    raise AlignmentError(f"Không kết nối được API sau retries: {exc}") from exc
                time.sleep(2 ** attempt * 3)
            except (requests.HTTPError, ValueError) as exc:
                raise AlignmentError(f"API lỗi HTTP/JSON: {exc}") from exc
        if not isinstance(response_obj, dict):
            raise AlignmentError("Không có response JSON hợp lệ")
        if response_obj.get("is_success") is not True or response_obj.get("code") != "000000":
            raise AlignmentError(
                f"API báo lỗi: code={response_obj.get('code')!r}, "
                f"message={str(response_obj.get('message'))[:300]}"
            )
        data = response_obj.get("data")
        if not isinstance(data, dict) or not isinstance(data.get("result"), list):
            raise AlignmentError("API response thiếu data.result (list)")
        rows, trace = _parse_api_result(data["result"], han_units, vi_units)
        self.last_response = response_obj
        self.last_trace = trace
        return rows, trace

    def _reserve_quota(self) -> None:
        """Tự giới hạn những call trong workspace; quota thực tế áp dụng cả ngoài workspace."""
        if not self.config.enforce_local_quotas or self._quota_path is None:
            return
        path = self._quota_path
        path.parent.mkdir(parents=True, exist_ok=True)
        now = time.time()
        try:
            history = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        except (json.JSONDecodeError, OSError):
            raise AlignmentError("File quota history hỏng; kiểm tra thủ công trước khi chạy tiếp")
        history = [float(t) for t in history if isinstance(t, (float, int)) and now - t < 604800]
        # Mốc quota theo tài liệu API (phút/giờ/ngày/tuần): 25/50/200/600.
        for seconds, cap, label in [(60, 25, "phút"), (3600, 50, "giờ"),
                                    (86400, 200, "ngày"), (604800, 600, "tuần")]:
            if sum(now - t < seconds for t in history) >= cap:
                raise QuotaExceeded(
                    f"Đã chạm giới hạn nội bộ {cap} lần/{label}. "
                    "Giữ checkpoint, chạy tiếp khi đủ quota (hoặc xác minh quota với Lab)."
                )
        history.append(now)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(history), encoding="utf-8")
        tmp.replace(path)

    @staticmethod
    def _restore_checkpoints(previous_work_dir: Path, raw_dir: Path) -> None:
        """Nạp lại align_raw từ Output phiên Kaggle trước (Add Input -> Notebook)."""
        previous = previous_work_dir / "align_raw"
        if not previous.is_dir():
            raise FileNotFoundError(f"resume_from phải chứa align_raw/: {previous}")
        count = 0
        for source in previous.rglob("*.json"):
            destination = raw_dir / source.relative_to(previous)
            if source.resolve() == destination.resolve() or destination.exists():
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            copy2(source, destination)
            count += 1
        print(f"Đã nạp {count} checkpoint/quota từ {previous}")

    def _execute(
        self, jobs: dict[str, list[_Block]], work_dir: Path, limit: int | None,
        *, resume_from: str | Path | None = None,
    ) -> AlignmentResult:
        work_dir.mkdir(parents=True, exist_ok=True)
        raw_dir = work_dir / "align_raw"
        output_dir = work_dir / "dong_hang"
        output_dir.mkdir(parents=True, exist_ok=True)
        raw_dir.mkdir(parents=True, exist_ok=True)
        if resume_from is not None:
            self._restore_checkpoints(Path(resume_from), raw_dir)
        self._quota_path = raw_dir / "_api_quota_history.json"
        budget = limit
        errors: list[str] = []
        stats: dict[str, dict[str, Any]] = {}
        for name, blocks in jobs.items():
            safe_name = _safe_name(name)
            book_raw_dir = raw_dir / safe_name
            book_raw_dir.mkdir(parents=True, exist_ok=True)
            rows: list[dict[str, Any]] = []
            reviews: list[dict[str, Any]] = []
            n_han = sum(len(b.han_units) for b in blocks)
            n_vi = sum(len(b.vi_units) for b in blocks)
            covered_han = covered_vi = 0
            pending = failed = low_score = 0
            for i, block in enumerate(blocks):
                if not block.han_units or not block.vi_units:
                    reviews.append({"block": i, "origin": block.origin, "kind": "untranslated_or_empty",
                                    "han_units": block.han_units, "vi_units": block.vi_units})
                    failed += 1
                    continue
                if block.hit_max:
                    reviews.append({"block": i, "origin": block.origin, "kind": "translation_hit_max_tokens",
                                    "message": "Bản dịch LLM có thể bị cắt cụt"})
                fingerprint = hashlib.sha256(json.dumps({
                    "han": block.han_units, "vi": block.vi_units, "url": self.api_url,
                    "split_mode": self.config.split_mode,
                }, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:20]
                checkpoint = book_raw_dir / f"block_{i:06d}_{fingerprint}.json"
                try:
                    if checkpoint.exists():
                        cached = json.loads(checkpoint.read_text(encoding="utf-8"))
                        api_rows = cached["rows"]
                        trace = cached["trace"]
                        print(f"[{name}] block {i+1}/{len(blocks)}: dùng checkpoint")
                    elif budget is not None and budget <= 0:
                        pending += 1
                        continue
                    else:
                        api_rows, trace = self._request_and_parse(block.han_units, block.vi_units)
                        cached = {"rows": api_rows, "trace": trace, "origin": block.origin}
                        tmp = checkpoint.with_suffix(".tmp")
                        tmp.write_text(json.dumps(cached, ensure_ascii=False, indent=2), encoding="utf-8")
                        tmp.replace(checkpoint)
                        if budget is not None:
                            budget -= 1
                        print(f"[{name}] block {i+1}/{len(blocks)}: {len(api_rows)} cặp")
                except QuotaExceeded as exc:
                    # Dừng thay vì thử tiếp hàng nghìn block và lặp cùng lỗi quota.
                    pending += len(blocks) - i
                    message = f"{name}: {exc}"
                    errors.append(message)
                    reviews.append({"block": i, "origin": block.origin,
                                    "kind": "quota_exceeded", "message": str(exc)})
                    print("DỪNG DO QUOTA:", message)
                    break
                except (AlignmentError, OSError, KeyError, ValueError) as exc:
                    failed += 1
                    message = f"{name} block {i} ({block.origin}): {exc}"
                    errors.append(message)
                    reviews.append({"block": i, "origin": block.origin, "kind": "api_or_cache_error",
                                    "message": str(exc)})
                    print("LỖI:", message)
                    continue
                for index, row in enumerate(api_rows):
                    record = {**row, "block": i, "origin": block.origin}
                    rows.append(record)
                    if row["score"] is None or row["score"] < self.config.review_score_below:
                        low_score += 1
                        reviews.append({"kind": "low_score", "row": record, "block": i, "index": index})
                covered_han += len(block.han_units) - len(trace["unmatched_han_indices"])
                covered_vi += len(block.vi_units) - len(trace["unmatched_vi_indices"])
                if trace["unmatched_han_indices"] or trace["unmatched_vi_indices"]:
                    reviews.append({"kind": "unmatched_units", "block": i, "origin": block.origin,
                                    "han": [block.han_units[j] for j in trace["unmatched_han_indices"]],
                                    "vi": [block.vi_units[j] for j in trace["unmatched_vi_indices"]]})
            has_pending = pending > 0 or failed > 0
            suffix = "_parallel.PARTIAL" if has_pending else "_parallel"
            tsv = output_dir / f"{safe_name}{suffix}.tsv"
            xlsx = output_dir / f"{safe_name}{suffix}.xlsx"
            # Tránh giữ lại ấn phẩm hoàn chỉnh cũ khi nguồn/chunk đã thay đổi.
            stale_suffix = "_parallel" if has_pending else "_parallel.PARTIAL"
            for ext in (".tsv", ".xlsx"):
                stale_file = output_dir / f"{safe_name}{stale_suffix}{ext}"
                if stale_file.exists():
                    stale_file.unlink()
            _export_tsv(tsv, rows)
            _export_xlsx(xlsx, rows)
            report = {
                "book": name, "status": (
                    "incomplete" if has_pending else
                    "needs_review" if reviews else "complete"
                ),
                "han_units": n_han, "vi_units": n_vi,
                "paired_rows": len(rows),
                "covered_han_units": covered_han, "covered_vi_units": covered_vi,
                "han_coverage": round(covered_han / n_han, 4) if n_han else 0,
                "vi_coverage": round(covered_vi / n_vi, 4) if n_vi else 0,
                "low_score_pairs": low_score,
                "review_items": len(reviews),
                "pending_blocks": pending, "failed_blocks": failed,
                "total_blocks": len(blocks),
                "tsv": str(tsv), "xlsx": str(xlsx),
                "notes": (
                    "Score API chỉ là chỉ dấu cần rà soát; không chứng minh độ chính xác 95%. "
                    "Không tự sửa nội dung bản dịch."
                ),
            }
            (output_dir / f"{safe_name}_report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            _write_jsonl(output_dir / f"{safe_name}_review.jsonl", reviews)
            stats[name] = report
        zip_path = work_dir / "dong_hang.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for item in sorted(output_dir.iterdir()):
                if item.is_file():
                    zf.write(item, arcname=item.name)
        return AlignmentResult(output_dir=output_dir, zip_path=zip_path,
                               book_stats=stats, errors=errors)


def _paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text.replace("\r\n", "\n")) if p.strip()]


def _split_units(text: str, mode: str, language: str) -> list[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []
    if mode == "lines":
        return [re.sub(r"\s+", " ", row).strip() for row in text.splitlines() if row.strip()]
    result: list[str] = []
    for para in _paragraphs(text):
        # Nối các dòng OCR bị xuống dòng cơ học; giữ nguyên mọi ký tự có nội dung.
        # Với chữ Hán, không chèn dấu cách vào giữa 2 chữ; với Việt, chèn 1 space.
        chunks = [s.strip() for s in para.splitlines() if s.strip()]
        joined = ("" if language == "han" else " ").join(chunks)
        if language == "han":
            parts = re.split(r"(?<=[。！？；])|(?<=[!?;])\s+", joined)
        else:
            # Chỉ ngắt sau dấu câu khi có khoảng trắng, giảm nhầm số thập phân.
            parts = re.split(r"(?<=[.!?;])\s+(?=\S)|(?<=[。！？；])", joined)
        result.extend(s.strip() for s in parts if s.strip())
    return result


def _compact_alignment_text(text: str) -> str:
    """Chuẩn hoá để ĐỐI CHIẾU vị trí, không thay đổi dữ liệu xuất.

    API có thể chia lại câu và thêm/bớt khoảng trắng giữa câu; chỉ bỏ
    whitespace và NFC, TUYỆT ĐỐI không bỏ dấu câu hoặc tự sửa chữ.
    """
    return "".join(unicodedata.normalize("NFC", text).split())


def _api_score(value: Any, index: int) -> float | None:
    if value is None:
        return None
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise AlignmentError(f"result[{index}].score không phải số") from exc
    return score if math.isfinite(score) else None


def _api_text_index_mapping(
    items: list[dict[str, Any]], units: list[str], field: str, language: str
) -> tuple[list[tuple[int, ...]], dict[str, Any]]:
    """Ánh xạ API text segments -> chỉ số unit gốc theo span ký tự liên tục.

    API CLC tự tách lại câu ngay cả khi nhận input mỗi unit một dòng.
    Do đó các index của API là index của cách TÁCH RIÊNG đó và KHÔNG thể
    trực tiếp dùng làm index của Bertalign / Vecalign.

    Bắt buộc so khớp nguyên văn (sau khi bỏ whitespace); nếu có thay đổi
    dấu câu/chữ, thất bại rõ ràng thay vì đoán bằng fuzzy matching.
    """
    normalized_units = [_compact_alignment_text(text) for text in units]
    if any(not text for text in normalized_units):
        raise AlignmentError(f"{language}: có unit đầu vào chỉ gồm khoảng trắng")

    offsets: list[tuple[int, int]] = []
    position = 0
    for text in normalized_units:
        offsets.append((position, position + len(text)))
        position += len(text)
    source = "".join(normalized_units)
    cursor = 0
    matches: list[tuple[int, ...]] = []
    gaps: list[dict[str, int]] = []

    for i, row in enumerate(items):
        segment = row.get(field)
        if not isinstance(segment, str) or not segment.strip():
            raise AlignmentError(f"result[{i}].{field} thiếu văn bản để ánh xạ về {language}")
        needle = _compact_alignment_text(segment)
        start = source.find(needle, cursor)
        if start < 0:
            raise AlignmentError(
                f"result[{i}].{field} KHÔNG khớp liên tục với {language} đầu vào "
                f"sau vị trí {cursor}; không tự đoán/fuzzy. "
                f"Đoạn: {segment[:70]!r}"
            )
        end = start + len(needle)
        indices = tuple(
            j for j, (lo, hi) in enumerate(offsets)
            if lo < end and hi > start
        )
        if not indices:
            raise AlignmentError(f"result[{i}].{field} không trùng unit {language} nào")
        if start > cursor:
            gaps.append({"start": cursor, "end": start})
        matches.append(indices)
        cursor = end

    if cursor < len(source):
        gaps.append({"start": cursor, "end": len(source)})
    return matches, {
        "covered_chars": sum(len(_compact_alignment_text(item[field])) for item in items),
        "total_chars": len(source),
        "unmatched_character_ranges": gaps,
    }


def _parse_api_result(
    items: list[Any], han_units: list[str], vi_units: list[str]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Quy đổi API về ĐÚNG các unit đã cung cấp cho ba phương pháp.

    Ưu tiên so khớp trường han_text / vietnamese_text (API tự phân đoạn lại);
    chỉ dùng trực tiếp native index khi response KHÔNG có nội dung và index
    thực sự nằm trong giới hạn input. Những kết quả bị mất/đổi chữ sẽ báo lỗi.
    """
    checked: list[dict[str, Any]] = []
    prev_h = prev_v = -1
    for i, row in enumerate(items):
        if not isinstance(row, dict):
            raise AlignmentError(f"result[{i}] không phải object")
        keys = ("han_start", "han_end", "vi_start", "vi_end")
        try:
            hs, he, vs, ve = (row[k] for k in keys)
        except KeyError as exc:
            raise AlignmentError(f"result[{i}] thiếu index {exc}") from exc
        if not all(type(value) is int for value in (hs, he, vs, ve)):
            raise AlignmentError(f"result[{i}] có index không phải số nguyên")
        if not (0 <= hs <= he and 0 <= vs <= ve):
            raise AlignmentError(f"result[{i}] có index rỗng hoặc đảo chiều")
        if hs <= prev_h or vs <= prev_v:
            raise AlignmentError(f"result[{i}] native index đảo thứ tự hoặc trùng; cần rà soát")
        prev_h, prev_v = he, ve
        checked.append(row)

    if not checked:
        return [], {"mapping_mode": "empty", "unmatched_han_indices": list(range(len(han_units))),
                    "unmatched_vi_indices": list(range(len(vi_units)))}

    text_present = any("han_text" in row or "vietnamese_text" in row for row in checked)
    if text_present:
        if not all("han_text" in row and "vietnamese_text" in row for row in checked):
            raise AlignmentError("API trả text không đồng nhất giữa các hàng; không thể ánh xạ an toàn")
        h_maps, h_info = _api_text_index_mapping(checked, han_units, "han_text", "han")
        v_maps, v_info = _api_text_index_mapping(checked, vi_units, "vietnamese_text", "vi")
        mapping_mode = "verified_text_offsets"
    else:
        # Fallback chỉ khi API không cung cấp text để đối chiếu; giữ contract cũ.
        for i, row in enumerate(checked):
            if row["han_end"] >= len(han_units) or row["vi_end"] >= len(vi_units):
                raise AlignmentError(
                    f"result[{i}] vượt phạm vi và không có text để ánh xạ; dừng an toàn"
                )
        h_maps = [tuple(range(row["han_start"], row["han_end"] + 1)) for row in checked]
        v_maps = [tuple(range(row["vi_start"], row["vi_end"] + 1)) for row in checked]
        h_info = v_info = None
        mapping_mode = "native_indices_without_text_verification"

    # Khi API chia nhỏ 1 unit thuốc thành nhiều hàng, các bead trùng/overlap.
    # Gộp từng chuỗi kề nhau có chung H hoặc V thành một bead n-m.
    # Không gộp 2 bead rời rạc, vì làm vậy sẽ tạo ra cặp không có bằng chứng.
    grouped: list[dict[str, Any]] = []
    for i, (h_ids, v_ids) in enumerate(zip(h_maps, v_maps)):
        hset, vset = set(h_ids), set(v_ids)
        score = _api_score(checked[i].get("score"), i)
        if grouped and (
            hset.intersection(grouped[-1]["h_ids"])
            or vset.intersection(grouped[-1]["v_ids"])
        ):
            current = grouped[-1]
            current["h_ids"] = sorted(set(current["h_ids"]) | hset)
            current["v_ids"] = sorted(set(current["v_ids"]) | vset)
            current["source_rows"].append(i)
            current["scores"].append(score)
        else:
            grouped.append({"h_ids": list(h_ids), "v_ids": list(v_ids),
                            "source_rows": [i], "scores": [score]})

    used_h: set[int] = set()
    used_v: set[int] = set()
    prev_h = prev_v = -1
    output: list[dict[str, Any]] = []
    for group in grouped:
        hs, he = group["h_ids"][0], group["h_ids"][-1]
        vs, ve = group["v_ids"][0], group["v_ids"][-1]
        # Fail closed với nhóm chéo nhau/không liền nhau sau khi quy đổi.
        if (group["h_ids"] != list(range(hs, he + 1))
            or group["v_ids"] != list(range(vs, ve + 1))
            or hs <= prev_h or vs <= prev_v
            or used_h.intersection(group["h_ids"])
            or used_v.intersection(group["v_ids"])):
            raise AlignmentError(
                f"API sau quy đổi có cặp trùng/chéo/không liên tục: "
                f"H={group['h_ids']} V={group['v_ids']}; cần kiểm tra thủ công"
            )
        used_h.update(group["h_ids"])
        used_v.update(group["v_ids"])
        prev_h, prev_v = he, ve
        scores = [score for score in group["scores"] if score is not None]
        output.append({
            "han": " ".join(han_units[hs:he + 1]),
            "vi": " ".join(vi_units[vs:ve + 1]),
            "han_start": hs, "han_end": he,
            "vi_start": vs, "vi_end": ve,
            # Giữ score thấp nhất, ưu tiên đánh dấu rà soát khi nhóm có điểm thấp.
            "score": min(scores) if scores else None,
            "api_source_rows": group["source_rows"],
            "alignment_type": f"{he - hs + 1}-{ve - vs + 1}",
        })

    trace = {
        "mapping_mode": mapping_mode,
        "api_result_count": len(checked),
        "mapped_pairs": len(output),
        "merged_api_rows": len(checked) - len(output),
        "unmatched_han_indices": sorted(set(range(len(han_units))) - used_h),
        "unmatched_vi_indices": sorted(set(range(len(vi_units))) - used_v),
        "han_text_matching": h_info,
        "vi_text_matching": v_info,
    }
    return output, trace


def _match_text_files(han_dir: Path, viet_dir: Path) -> dict[str, tuple[Path, Path]]:
    if not han_dir.is_dir() or not viet_dir.is_dir():
        raise FileNotFoundError("han_dir và viet_dir phải là 2 thư mục có file *.txt")

    def collect(folder: Path) -> dict[str, Path]:
        found: dict[str, Path] = {}
        for p in sorted(folder.rglob("*.txt")):
            if p.stem in found:
                raise AlignmentError(
                    f"Trùng tên sách {p.stem!r} trong {folder}; đổi tên để ghép đúng cặp"
                )
            found[p.stem] = p
        return found

    h = collect(han_dir)
    v = collect(viet_dir)
    missing_h = sorted(set(v) - set(h))
    missing_v = sorted(set(h) - set(v))
    if missing_h or missing_v:
        raise AlignmentError(
            f"Thiếu file ghép cặp: thiếu bản Hán {missing_h[:10]}, "
            f"thiếu bản Việt {missing_v[:10]}"
        )
    if not h:
        raise AlignmentError("Không tìm thấy cặp *.txt nào")
    return {name: (h[name], v[name]) for name in sorted(h)}


def _safe_name(name: str) -> str:
    cleaned = re.sub(r"[^\w.\-]+", "_", name, flags=re.UNICODE).strip("._")
    if not cleaned:
        raise AlignmentError("Tên sách rỗng sau khi chuẩn hoá")
    return cleaned


def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as fp:
        all_lines = fp.readlines()
    for line_no, line in enumerate(all_lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if isinstance(row, dict):
                yield row
        except json.JSONDecodeError:
            # Dòng cuối đang ghi dở do timeout notebook; cho phép resume.
            if line_no == len(all_lines):
                continue
            raise AlignmentError(f"JSONL lỗi ở {path}:{line_no}")


def _one_line(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _export_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    # TSV không header theo hướng dẫn: <CÂU_HÁN>\t<CÂU_VIỆT>
    with path.open("w", encoding="utf-8", newline="") as fp:
        for row in rows:
            h = _one_line(row["han"])
            v = _one_line(row["vi"])
            if not h or not v:
                raise AlignmentError("Không thể xuất cặp thiếu một vế")
            fp.write(f"{h}\t{v}\n")


def _export_xlsx(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Font, PatternFill
    except ImportError as exc:
        raise RuntimeError("Thiếu openpyxl. Chạy: pip install openpyxl") from exc
    wb = Workbook()
    sh = wb.active
    sh.title = "DONG_HANG"
    sh.append(["Hán", "Việt"])
    for row in rows:
        han, vi = _one_line(row["han"]), _one_line(row["vi"])
        # Buộc hai ô giữ kiểu chuỗi, không tạo công thức và không sửa nội dung.
        sh.append([han, vi])
        sh.cell(sh.max_row, 1).data_type = "s"
        sh.cell(sh.max_row, 2).data_type = "s"
    for c in sh[1]:
        c.font = Font(bold=True, color="FFFFFF")
        c.fill = PatternFill("solid", fgColor="344054")
    sh.column_dimensions["A"].width = 55
    sh.column_dimensions["B"].width = 75
    sh.freeze_panes = "A2"
    for row in sh.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical="top")
    wb.save(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")


__all__ = ["HanVietAligner", "AlignmentConfig", "AlignmentResult", "AlignmentError"]
