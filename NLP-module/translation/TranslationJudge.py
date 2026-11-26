"""LLM-as-a-Judge cho bản dịch Hán -> Việt từ HanVietTranslator.

Module chạy độc lập với việc dịch. Nó ĐỌC output đã có:
    <translator_output>/chunks.jsonl
    <translator_output>/dich_raw/dich_shard*.jsonl
và GHI kết quả vào <work_dir>/translation_quality/; không sửa checkpoint dịch.

Giao diện:
    from TranslationJudge import TranslationJudge
    tool = TranslationJudge(
        judge_models=["model-a", "model-b", "model-c", "model-d"],
        repair_model="model-e", load_4bit=True,
    )
    result = tool.evaluate("/kaggle/input/translation-output", "/kaggle/working", limit=3)
    print(result.stats)

Để tiếp tục lần Save Version sau, truyền previous_result_dir trỏ đến output lần trước.

- 95% trong code nghĩa là >=95% SỐ JUDGE đánh giá PASS cho MỖI chunk.
  Với 3-4 judge: phải đồng thuận hoàn toàn để chấp nhận; đây là quy ước
  triển khai có thể điều chỉnh, KHÔNG tự nhận là công thức của giảng viên.
- Mỗi judge phải là một checkpoint model khác nhau; repair_model phải khác.
- Các đánh giá từ LLM chỉ là đánh giá tự động, không thay thế thẩm định người.
- Số chunk được chấp nhận KHÔNG phải số cặp câu dóng hàng hợp lệ.
- Tìm vị trí bằng dich_accepted_annotated/<book>.annotated.txt (Ctrl+F CHUNK 000006)
  hoặc needs_review_locator.tsv. Giữ TXT sạch không có nhãn CHUNK.
"""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

try:
    from .utils import attach_source_locations, render_book_with_index, _safe_book_filename
except ImportError:
    # Hỗ trợ cả import trực tiếp khi thư mục translation nằm trong sys.path.
    from utils import attach_source_locations, render_book_with_index, _safe_book_filename


RUBRIC_FIELDS = ("faithful", "complete", "terminology", "numbers_units", "no_added_content")
JUDGE_SYSTEM = (
    "Bạn là người kiểm định bản dịch văn ngôn và bạch thoại Hán -> Việt trong sách Y học cổ truyền. "
    "Đánh giá theo văn bản gốc, không dựa vào độ trôi chảy đơn thuần. "
    "Đặc biệt kiểm tra tên huyệt, bài thuốc, vị thuốc, số lượng, đơn vị, chỉ định, phủ định "
    "và ý bị lược bỏ hay thêm vào. Nếu không chắc, đánh dấu uncertain=true. "
    "Chỉ trả về đúng một JSON object; không có Markdown hay giải thích ngoài JSON."
)
JUDGE_USER = """VĂN BẢN HÁN:
{zh}

BẢN DỊCH VIỆT:
{vi}

Trả về JSON với các khóa boolean bắt buộc:
"faithful" (không dịch sai ý), "complete" (không bỏ ý),
"terminology" (thuật ngữ YHCT phù hợp), "numbers_units" (số/liều lượng đúng),
"no_added_content" (không bịa/thêm ý), "uncertain" (không đủ căn cứ),
"issues" (danh sách chuỗi mô tả lỗi; [] nếu không có),
"explanation" (1 câu ngắn giải thích đánh giá).
Đánh dấu false nếu có lỗi, không vì bản dịch nghe tự nhiên mà cho qua."""
REPAIR_SYSTEM = (
    "Bạn là dịch giả và hiệu đính sách Y học cổ truyền Hán -> Việt. "
    "Hãy so sánh với Hán gốc, sửa mọi sai sót có căn cứ (thiếu ý, thừa ý, "
    "tên vị thuốc, liều lượng), giữ nguyên thứ tự, không bịa nội dung. "
    "Trả về JSON object chỉ có khóa revised_vi chứa bản dịch tiếng Việt đã sửa, "
    "không sao chép chữ Hán hoặc lời dẫn. Nếu chữ Hán không rõ, giữ [MASK]."
)
REPAIR_USER = """VĂN BẢN HÁN:
{zh}

BẢN DỊCH HIỆN TẠI:
{vi}

CÁC NGHI VẤN DO JUDGE NÊU:
{issues}

Hãy dịch/hiệu đính lại sát Hán gốc và chỉ xuất JSON: {{"revised_vi": "..."}}."""


def _digest(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
        except (ValueError, TypeError):
            continue
        if isinstance(item, dict):
            result.append(item)
    return result


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def _get_object(raw: str) -> dict[str, Any]:
    clean = raw.strip()
    if clean.startswith("```"):
        clean = re.sub(r"^```(?:json)?\s*", "", clean, flags=re.I)
        clean = re.sub(r"\s*```\s*$", "", clean)
    try:
        data = json.loads(clean)
    except ValueError:
        start, end = clean.find("{"), clean.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Không tìm được JSON object trong phản hồi LLM")
        data = json.loads(clean[start:end + 1])
    if not isinstance(data, dict):
        raise ValueError("LLM phải trả về JSON object")
    return data


def _judge_result(raw: str) -> dict[str, Any]:
    data = _get_object(raw)
    for field in (*RUBRIC_FIELDS, "uncertain"):
        if type(data.get(field)) is not bool:
            raise ValueError(f"Judge trả sai/thiếu boolean {field}")
    issues = data.get("issues", [])
    if not isinstance(issues, list) or not all(isinstance(x, str) for x in issues):
        raise ValueError("Judge trả issues không phải list[str]")
    return {
        **{k: data[k] for k in RUBRIC_FIELDS},
        "uncertain": data["uncertain"],
        "issues": issues,
        "explanation": str(data.get("explanation", ""))[:2000],
        "pass": all(data[k] for k in RUBRIC_FIELDS) and not data["uncertain"],
    }


def _repair_result(raw: str) -> str:
    obj = _get_object(raw)
    vi = obj.get("revised_vi")
    if not isinstance(vi, str) or not vi.strip():
        raise ValueError("Repair không trả revised_vi hợp lệ")
    vi = vi.strip()
    if vi in ("[chưa dịch]", "[MASK]"):
        raise ValueError("Repair chỉ trả placeholder")
    return vi


class GenerationEngine(Protocol):
    def load(self, model_name: str) -> None: ...
    def ask(self, messages: list[dict[str, str]], max_new_tokens: int) -> str: ...
    def unload(self) -> None: ...


class LocalHFEngine:
    """Chạy tuần tự từng checkpoint model, không dùng API trả phí."""

    def __init__(self, load_4bit: bool = True, max_input_tokens: int = 4096):
        self.load_4bit = load_4bit
        self.max_input_tokens = max_input_tokens
        self.model = None
        self.tokenizer = None

    def load(self, model_name: str) -> None:
        self.unload()
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        opts: dict[str, Any] = {}
        if torch.cuda.is_available():
            opts.update(device_map="auto", dtype=torch.float16)
            if self.load_4bit:
                from transformers import BitsAndBytesConfig
                opts["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                )
        elif self.load_4bit:
            raise RuntimeError("4-bit cần GPU NVIDIA; với CPU hãy dùng load_4bit=False")
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **opts).eval()

    def ask(self, messages: list[dict[str, str]], max_new_tokens: int) -> str:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Chưa gọi load(model_name)")
        tok = self.tokenizer
        try:
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
        except (TypeError, ValueError):
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        tokens = tok(prompt, return_tensors="pt", add_special_tokens=False)
        if tokens["input_ids"].shape[1] > self.max_input_tokens:
            raise ValueError("Prompt vượt giới hạn input, cần giảm độ dài chunk; không tự cắt")
        tokens = tokens.to(self.model.device)
        import torch
        with torch.inference_mode():
            output = self.model.generate(
                **tokens, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        return tok.decode(output[0][tokens["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def unload(self) -> None:
        self.model = None
        self.tokenizer = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


@dataclass
class QualityResult:
    output_dir: Path
    stats_path: Path
    audit_path: Path
    review_path: Path
    accepted_dir: Path
    stats: dict[str, Any]
    annotated_dir: Path | None = None
    locator_dir: Path | None = None
    review_locator_path: Path | None = None


class TranslationJudge:
    def __init__(
        self,
        judge_models: list[str],
        repair_model: str,
        *,
        min_pass_fraction: float = .95,
        max_revisions: int = 1,
        load_4bit: bool = True,
        max_judge_tokens: int = 350,
        max_repair_tokens: int = 1100,
        max_json_retries: int = 1,
        backend: GenerationEngine | None = None,
    ) -> None:
        if not (3 <= len(judge_models) <= 4):
            raise ValueError("Phải cấu hình 3 hoặc 4 LLM judge")
        if any(not m or not m.strip() for m in judge_models) or len(set(judge_models)) != len(judge_models):
            raise ValueError("Judge phải là 3-4 model checkpoint khác nhau")
        if not repair_model or repair_model in judge_models:
            raise ValueError("Model repair phải khác tất cả judge")
        if not (0 < min_pass_fraction <= 1):
            raise ValueError("min_pass_fraction phải nằm trong (0,1]")
        if max_revisions not in (0, 1):
            raise ValueError("Bản đầu hỗ trợ 0 hoặc 1 lượt sửa; chưa triển khai nhiều vòng")
        self.judge_models = tuple(judge_models)
        self.repair_model = repair_model
        self.min_pass_fraction = min_pass_fraction
        self.max_revisions = max_revisions
        self.max_judge_tokens = max_judge_tokens
        self.max_repair_tokens = max_repair_tokens
        self.max_json_retries = max_json_retries
        self.backend = backend or LocalHFEngine(load_4bit=load_4bit)
        self.prompt_hash = _digest(JUDGE_SYSTEM + JUDGE_USER + REPAIR_SYSTEM + REPAIR_USER)

    def _load_source(self, source_dir: Path, limit: int | None,
                     source_text_dir: Path | None = None) -> tuple[list[dict[str, Any]], int]:
        chunk_file = source_dir / "chunks.jsonl"
        if not chunk_file.is_file():
            raise FileNotFoundError(f"Không tìm thấy {chunk_file}")
        chunks = _jsonl(chunk_file)
        if source_text_dir is not None and any(
            r.get("source_start") is None or r.get("source_line_start") is None
            for r in chunks
        ):
            # Hồi phục tọa độ cho checkpoint được tạo bởi bản Translator cũ.
            chunks = attach_source_locations(chunks, source_text_dir)
        if len({x.get('id') for x in chunks}) != len(chunks):
            raise ValueError("chunks.jsonl có ID trùng")
        all_total = len(chunks)
        if limit is not None:
            if limit < 1:
                raise ValueError("limit phải >= 1")
            chunks = chunks[:limit]
        translation_files = sorted((source_dir / "dich_raw").glob("dich_shard*.jsonl"),
                                   key=lambda p: ('_prev' not in p.stem, p.name))
        if not translation_files:
            raise FileNotFoundError(f"Không có checkpoint dịch trong {source_dir / 'dich_raw'}")
        translated: dict[str, dict[str, Any]] = {}
        for path in translation_files:
            for r in _jsonl(path):
                if isinstance(r.get("id"), str) and isinstance(r.get("vi"), str):
                    translated[r['id']] = r
        records = []
        for chunk in chunks:
            ident, zh = chunk.get("id"), chunk.get("zh")
            if not isinstance(ident, str) or not isinstance(zh, str) or not zh.strip():
                raise ValueError("chunks.jsonl chứa bản ghi thiếu id hoặc zh")
            tr = translated.get(ident, {})
            initial = tr.get("vi", "")
            records.append({
                "id": ident,
                "book": ident.rsplit('|', 2)[0],
                "zh": zh,
                "chunk_index": chunk.get("chunk_index", int(ident.rsplit('|', 2)[-2])),
                "chunk_number": chunk.get("chunk_number", int(ident.rsplit('|', 2)[-2]) + 1),
                "source_file": chunk.get("source_file"),
                "source_sha256": chunk.get("source_sha256"),
                "source_start": chunk.get("source_start"),
                "source_end": chunk.get("source_end"),
                "source_line_start": chunk.get("source_line_start"),
                "source_line_end": chunk.get("source_line_end"),
                "source_spans": chunk.get("source_spans"),
                "initial_vi": initial.strip() if isinstance(initial, str) else "",
                "hit_max": bool(tr.get("hit_max", False))
            })
        return records, all_total

    def _cache(self, root: Path, previous: Path | None) -> dict[tuple[str, str, str, str], dict[str, Any]]:
        cache: dict[tuple[str, str, str, str], dict[str, Any]] = {}
        dirs = [root / "checkpoints"]
        if previous is not None:
            if (previous / "translation_quality" / "checkpoints").is_dir():
                dirs.insert(0, previous / "translation_quality" / "checkpoints")
            elif (previous / "checkpoints").is_dir():
                dirs.insert(0, previous / "checkpoints")
        for folder in dirs:
            if not folder.is_dir():
                continue
            for path in sorted(folder.glob('*.jsonl')):
                for row in _jsonl(path):
                    key = (row.get('stage'), row.get('id'), row.get('model'), row.get('candidate_sha'))
                    if (len(key) == 4 and all(isinstance(k, str) for k in key)
                            and row.get("prompt_hash") == self.prompt_hash
                            and row.get("ok") is True):
                        cache[key] = row
        return cache

    def _ask_json(self, messages: list[dict[str, str]], max_tokens: int, parse):
        error: Exception | None = None
        for trial in range(self.max_json_retries + 1):
            try:
                resp = self.backend.ask(messages, max_new_tokens=max_tokens)
                return parse(resp)
            except (ValueError, TypeError, KeyError) as exc:
                error = exc
                if trial < self.max_json_retries:
                    messages = [*messages, {"role": "user", "content": (
                        "Phản hồi trước không đúng JSON schema. Hãy chỉ trả đúng một JSON object hợp lệ."
                    )}]
        raise ValueError(f"LLM không trả JSON hợp lệ sau retry: {error}")

    def _run_stage(
        self, root: Path, cache: dict, stage: str, rows: list[dict[str, Any]],
        model: str, parse, prompt_fn, max_tokens: int,
    ) -> dict[str, dict[str, Any]]:
        results = {}
        pending = []
        for row in rows:
            key = (stage, row['id'], model, _digest(row['vi']))
            if key in cache:
                results[row['id']] = cache[key]['result']
            else:
                pending.append((row, key))
        if not pending:
            return results
        self.backend.load(model)
        try:
            checkpoint = root / 'checkpoints' / (stage + '__' + _digest(model)[:12] + '.jsonl')
            for row, key in pending:
                try:
                    result = self._ask_json(prompt_fn(row), max_tokens, parse)
                    packed = {"stage": stage, "id": row['id'], "model": model,
                              "candidate_sha": key[3], "prompt_hash": self.prompt_hash,
                              "ok": True, "result": result}
                    _append_jsonl(checkpoint, packed)
                    cache[key] = packed
                    results[row['id']] = result
                except Exception as exc:
                    results[row['id']] = {"_error": f"{type(exc).__name__}: {exc}"}
        finally:
            self.backend.unload()
        return results

    def _judge_all(self, root: Path, cache: dict, stage: str, rows: list[dict[str, Any]]):
        by_model = {}
        def prompt_fn(row):
            return [{"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": JUDGE_USER.format(zh=row['zh'], vi=row['vi'])}]
        for model in self.judge_models:
            by_model[model] = self._run_stage(root, cache, stage, rows, model,
                                               _judge_result, prompt_fn, self.max_judge_tokens)
        per_id = {}
        for row in rows:
            judgments = {m: by_model[m][row['id']] for m in self.judge_models}
            valid = [r for r in judgments.values() if '_error' not in r]
            votes = sum(r['pass'] for r in valid)
            total = len(self.judge_models)
            per_id[row['id']] = {
                "judgments": judgments,
                "valid_judges": len(valid),
                "passes": votes,
                "total_judges": total,
                "pass_fraction": votes / total,
                "agreement_fraction": (max(votes, total - votes) / total) if len(valid) == total else None,
                "accepted": len(valid) == total and votes / total >= self.min_pass_fraction,
            }
        return per_id

    def evaluate(
        self,
        translation_output_dir: str | Path,
        work_dir: str | Path,
        *,
        limit: int | None = None,
        previous_result_dir: str | Path | None = None,
        source_text_dir: str | Path | None = None,
    ) -> QualityResult:
        """Đánh giá chunk đã dịch, sửa thất bại 1 lần, đánh giá lại.

        `previous_result_dir` có thể là root Kaggle notebook Output cũ hoặc
        thư mục translation_quality cũ; chỉ ĐỌC checkpoint, không sao chép Input.
        `source_text_dir` chỉ cần cho chunks.jsonl cũ chưa có source_start/end:
        đặt thư mục TXT Hán ban đầu để khôi phục vị trí ký tự/dòng gốc.
        """
        source_dir = Path(translation_output_dir)
        root = Path(work_dir) / 'translation_quality'
        root.mkdir(parents=True, exist_ok=True)
        previous = Path(previous_result_dir) if previous_result_dir else None
        rows, all_total = self._load_source(
            source_dir, limit,
            Path(source_text_dir) if source_text_dir is not None else None,
        )
        cached = self._cache(root, previous)
        available = [dict(r, vi=r['initial_vi']) for r in rows
                     if r['initial_vi'] and r['initial_vi'] != '[chưa dịch]']
        first = self._judge_all(root, cached, 'initial', available) if available else {}
        repaired: dict[str, str] = {}
        repair_errors: dict[str, str] = {}
        failed = []
        for row in available:
            eval0 = first[row['id']]
            if not eval0['accepted'] or row['hit_max']:
                if eval0['valid_judges'] != len(self.judge_models):
                    repair_errors[row['id']] = 'Thiếu kết quả judge; không tự sửa'
                else:
                    failed.append(row)

        if failed and self.max_revisions:
            def repair_prompt(row):
                responses = first[row['id']]['judgments'].values()
                issues = [i for judge in responses for i in judge.get('issues', [])]
                if row['hit_max']:
                    issues.append('Bản dịch cũ chạm max_new_tokens, có nguy cơ cụt')
                return [{"role": "system", "content": REPAIR_SYSTEM},
                        {"role": "user", "content": REPAIR_USER.format(
                            zh=row['zh'], vi=row['vi'],
                            issues=json.dumps(issues[:15], ensure_ascii=False))}]
            fixes = self._run_stage(root, cached, 'repair', failed, self.repair_model,
                                    _repair_result, repair_prompt, self.max_repair_tokens)
            for row in failed:
                fix = fixes[row['id']]
                if isinstance(fix, str) and fix.strip() and fix.strip() != row['vi']:
                    repaired[row['id']] = fix.strip()
                else:
                    repair_errors[row['id']] = (
                        fix.get('_error') if isinstance(fix, dict) else 'LLM repair không thay đổi bản dịch')

        revised_rows = [dict(r, vi=repaired[r['id']]) for r in available if r['id'] in repaired]
        second = self._judge_all(root, cached, 'revised', revised_rows) if revised_rows else {}
        reports = []
        accepted_texts = {}
        book_counts = {}
        for row in rows:
            book = row['book']
            stats = book_counts.setdefault(book, {"total_chunks": 0, "translated_chunks": 0,
                                                   "accepted_chunks": 0, "review_chunks": 0,
                                                   "revised_accepted_chunks": 0})
            stats['total_chunks'] += 1
            initial = row['initial_vi']
            valid_trans = bool(initial and initial != '[chưa dịch]')
            if valid_trans:
                stats['translated_chunks'] += 1
            zero = first.get(row['id'])
            one = second.get(row['id'])
            revision = repaired.get(row['id'])
            final_pass = bool(one and one['accepted']) if revision else bool(
                zero and zero['accepted'] and not row['hit_max'])
            final_vi = revision if revision is not None else initial
            status = ('ACCEPTED_REVISED' if revision and final_pass else
                      'ACCEPTED' if final_pass else
                      'MISSING_TRANSLATION' if not valid_trans else 'NEEDS_REVIEW')
            if final_pass:
                stats['accepted_chunks'] += 1
                if revision:
                    stats['revised_accepted_chunks'] += 1
                accepted_texts[row['id']] = final_vi
            else:
                stats['review_chunks'] += 1
            issues: list[str] = []
            if not valid_trans:
                issues.append("Chưa có bản dịch")
            if row['hit_max']:
                issues.append("Bản dịch chạm max_new_tokens; có nguy cơ thiếu nội dung")
            last_evaluation = one if revision else zero
            if last_evaluation:
                for model_name, evaluation in last_evaluation['judgments'].items():
                    if evaluation.get('_error'):
                        issues.append(f"{model_name}: lỗi judge: {evaluation['_error']}")
                    elif not evaluation.get('pass', False):
                        for issue in evaluation.get('issues') or [evaluation.get('explanation') or 'Không đạt rubric']:
                            label = f"{model_name}: {issue}"
                            if label not in issues:
                                issues.append(label)
            if repair_errors.get(row['id']):
                issues.append(f"Sửa bản dịch: {repair_errors[row['id']]}")
            reports.append({
                "id": row['id'], "book": book, "han": row['zh'],
                "chunk_index": row['chunk_index'], "chunk_number": row['chunk_number'],
                "search_tag": f"CHUNK {row['chunk_number']:06d}",
                "source_file": row['source_file'], "source_sha256": row['source_sha256'],
                "source_start": row['source_start'], "source_end": row['source_end'],
                "source_line_start": row['source_line_start'],
                "source_line_end": row['source_line_end'],
                "source_spans": row['source_spans'],
                "initial_vi": initial, "final_vi": final_vi,
                "hit_max": row['hit_max'], "status": status,
                "reasons": issues,
                "initial_evaluation": zero, "revised_evaluation": one,
                "repair_error": repair_errors.get(row['id']),
            })

        audit = root / 'evaluation_audit.jsonl'
        review = root / 'needs_review.jsonl'
        audit.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in reports), encoding='utf-8')
        review.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in reports
                                  if r['status'] not in ('ACCEPTED', 'ACCEPTED_REVISED')), encoding='utf-8')

        accepted_dir = root / 'dich_accepted'
        annotated_dir = root / 'dich_accepted_annotated'
        locator_dir = root / 'dich_accepted_index'
        accepted_dir.mkdir(exist_ok=True)
        by_book: dict[str, list[dict[str, Any]]] = {}
        for row, report in zip(rows, reports):
            # Văn bản TXT sạch chỉ gồm bản dịch được chấp nhận hoặc placeholder.
            # Chỉ bản có nhãn CHUNK mới hiển thị text chưa đạt, tránh lẫn vào corpus.
            candidate = accepted_texts.get(row['id'])
            content = candidate if candidate is not None else (
                '[chưa đạt đánh giá]' if row['initial_vi'] else '[chưa dịch]'
            )
            by_book.setdefault(row['book'], []).append({
                **row, "text": content, "status": report['status'],
                "candidate_vi": report['final_vi'],
                "reasons": report['reasons'],
            })
        index_by_id: dict[str, dict[str, Any]] = {}
        for book, book_rows in by_book.items():
            safe = _safe_book_filename(book)
            index_rows = render_book_with_index(
                book, book_rows,
                accepted_dir / f'{safe}.txt',
                annotated_dir / f'{safe}.annotated.txt',
                locator_dir / f'{safe}.tsv',
            )
            index_by_id.update({item['chunk_id']: item for item in index_rows})

        # Một TSV gọn cho toàn bộ chunk chưa đạt; có thể mở Excel và Ctrl+F tag.
        review_locator = root / 'needs_review_locator.tsv'
        review_fields = [
            'book', 'chunk_number', 'search_tag', 'chunk_id', 'status', 'reasons',
            'source_file', 'source_start', 'source_end', 'source_line_start',
            'source_line_end', 'translation_start', 'translation_end',
            'translation_line_start', 'translation_line_end',
            'source_preview', 'candidate_vi_preview',
        ]
        with review_locator.open('w', encoding='utf-8-sig', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=review_fields, delimiter='\t')
            writer.writeheader()
            for report in reports:
                if report['status'] in ('ACCEPTED', 'ACCEPTED_REVISED'):
                    continue
                entry = index_by_id[report['id']]
                writer.writerow({
                    **{k: entry.get(k) for k in review_fields if k in entry},
                    'reasons': '; '.join(report['reasons']),
                    'candidate_vi_preview': (report['final_vi'] or '')[:250],
                })
        stats = {
            "source_total_chunks": all_total,
            "sampled_chunks": len(rows),
            "translated_chunks": len(available),
            "accepted_chunks": len(accepted_texts),
            "revised_accepted_chunks": sum(r['status'] == 'ACCEPTED_REVISED' for r in reports),
            "needs_review_chunks": sum(r['status'] not in ('ACCEPTED', 'ACCEPTED_REVISED') for r in reports),
            "judge_models": list(self.judge_models),
            "repair_model": self.repair_model,
            "min_pass_fraction": self.min_pass_fraction,
            "meaning_of_95pct": "Tỷ lệ judge PASS cho từng chunk, không phải độ chính xác bản dịch",
            "note": "Đây là số CHUNK. Không được báo cáo là số cặp câu đã dóng hàng.",
            "by_book": book_counts,
        }
        stats_path = root / 'quality_stats.json'
        stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        return QualityResult(
            root, stats_path, audit, review, accepted_dir, stats,
            annotated_dir, locator_dir, review_locator,
        )
