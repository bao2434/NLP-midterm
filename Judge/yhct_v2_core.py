"""Shared IO, inference, context and checkpoint helpers for YHCT V2.

Import in both Kaggle Judge and Repair notebooks. Do not modify V1 modules.
"""
from __future__ import annotations
import csv, gc, hashlib, io, json, os, re, statistics, time, zipfile
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone

VERSION = 'yhct-v2-context-evidence-score-2026-11-27'
HAN_RE = re.compile(r'[\u3400-\u9fff\uf900-\ufaff\U00020000-\U0002EBEF]')
UID_RE = re.compile(r'^([LP])-S(\d{6})$')


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stable_hash(obj) -> str:
    return sha_bytes(json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8'))


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def valid_shard(s: str) -> str:
    if not isinstance(s, str) or not re.fullmatch(r'[a-zA-Z0-9_.-]{1,110}', s):
        raise ValueError(f'SHARD_ID không hợp lệ: {s!r}')
    return s


def read_tsv(path: Path) -> list[dict]:
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f, delimiter='\t'))


def dump_tsv(path: Path, rows: list[dict], columns: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=columns, delimiter='\t', lineterminator='\n', extrasaction='ignore')
    writer.writeheader()
    for row in rows:
        writer.writerow({col: row.get(col, '') for col in columns})
    atomic_string(path, stream.getvalue())


def atomic_string(path: Path, contents: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf-8', newline='') as f:
        f.write(contents)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def write_json(path: Path, obj):
    atomic_string(path, json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def append_jsonl(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(obj, ensure_ascii=False) + '\n')
        f.flush(); os.fsync(f.fileno())


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists(): return []
    ret = []
    with path.open('r', encoding='utf-8-sig') as f:
        for i, line in enumerate(f, 1):
            if not line.strip(): continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict): raise TypeError('not object')
                ret.append(row)
            except (ValueError, TypeError) as ex:
                # One truncated tail could occur on a crashed session: only skip the FINAL line.
                if not f.read().strip():
                    print(f'[WARN] Bỏ dòng JSONL cuối hỏng: {path.name}:{i}: {ex}')
                    break
                raise ValueError(f'JSONL lỗi ở {path}:{i}: {ex}') from ex
    return ret


def norm_input_row(row: dict, *, require_vi=True):
    uid = str(row.get('unit_id', '')).strip()
    if not UID_RE.fullmatch(uid): raise ValueError(f'unit_id không đúng: {uid!r}')
    han = str(row.get('han', row.get('zh', ''))).strip()
    vi = str(row.get('viet', row.get('vi', ''))).strip()
    if not han: raise ValueError(f'{uid}: Hán rỗng')
    if require_vi and not vi: raise ValueError(f'{uid}: Việt rỗng')
    return {'unit_id': uid, 'han': han, 'viet': vi, 'source_line': str(row.get('source_line', '')),
            'section': str(row.get('section', '')), 'source_model': str(row.get('model', row.get('source_model','')))}


def chinese_leftover(text: str, allow_phrases=()) -> int:
    copy = text or ''
    for phrase in allow_phrases:
        if phrase:
            copy = copy.replace(phrase, '')
    return len(HAN_RE.findall(copy))


def build_context(target: list[dict], extras: list[dict], *, window=1,
                  same_section=True, max_side_chars=650) -> list[dict]:
    if window not in (0, 1, 2): raise ValueError('CONTEXT_WINDOW chỉ nhận 0,1,2')
    by_id = {}
    for row in extras + target:
        rr = norm_input_row(row, require_vi=False)
        old = by_id.get(rr['unit_id'])
        if old and old['han'] != rr['han']:
            raise ValueError(f'Mâu thuẫn Hán giữa file input tại {rr["unit_id"]}')
        by_id[rr['unit_id']] = rr
    result = []
    for row in target:
        rr = norm_input_row(row)
        m = UID_RE.fullmatch(rr['unit_id'])
        prefix, index = m.group(1), int(m.group(2))
        before = []; after = []
        for delta, receiver in [(-d, before) for d in range(window,0,-1)] + [(d, after) for d in range(1,window+1)]:
            neighbor = by_id.get(f'{prefix}-S{index+delta:06d}')
            if not neighbor: continue
            if same_section and rr['section'] and neighbor['section'] and rr['section'] != neighbor['section']:
                continue
            receiver.append(neighbor['han'])
        rr['ctx_before_han'] = '\n'.join(before)[-max_side_chars:]
        rr['ctx_after_han'] = '\n'.join(after)[:max_side_chars]
        result.append(rr)
    return result


def parse_json_object(s: str) -> dict:
    s = (s or '').strip()
    try:
        obj = json.loads(s)
        if isinstance(obj,dict): return obj
    except ValueError: pass
    decoder = json.JSONDecoder()
    for idx, ch in enumerate(s):
        if ch != '{': continue
        try: obj, _ = decoder.raw_decode(s[idx:])
        except ValueError: continue
        if isinstance(obj,dict): return obj
    raise ValueError('Không tìm thấy JSON object')


class CausalModel:
    """Load one HF model at a time. Same interface in Judge and Repair notebooks."""
    def __init__(self, cfg: dict, cache_root: Path, max_input_tokens: int):
        self.cfg = cfg
        self.cache_root = Path(cache_root)
        self.max_input_tokens = max_input_tokens
        self.model = None
        self.tokenizer = None

    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        name = self.cfg['name']
        args = {'cache_dir': str(self.cache_root / 'hub')}
        trust = bool(self.cfg.get('trust_remote_code', name.startswith('tencent/Hy-MT2-')))
        self.tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=trust, **args)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.cfg.get('quant4') and not torch.cuda.is_available():
            raise RuntimeError('4-bit BitsAndBytes cần GPU CUDA')
        opts = {'low_cpu_mem_usage': True, 'trust_remote_code': trust}
        if torch.cuda.is_available():
            opts.update({'device_map':'auto','dtype':torch.float16})
            if self.cfg.get('quant4'):
                opts['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True,
                    bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.float16)
        else:
            opts['dtype'] = torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(name, **opts, **args).eval()

    def infer(self, messages: list[dict], max_new_tokens: int) -> tuple[str, dict]:
        import torch
        tok = self.tokenizer
        template_opts = {}
        if self.cfg['name'].startswith('Qwen/Qwen3'):
            template_opts['enable_thinking'] = False
        try:
            encoded = tok.apply_chat_template(messages, add_generation_prompt=True,
                 tokenize=True, return_tensors='pt', return_dict=True, **template_opts)
            if isinstance(encoded, torch.Tensor): encoded = {'input_ids':encoded}
        except (TypeError,ValueError):
            try:
                rendered=tok.apply_chat_template(messages,add_generation_prompt=True,
                       tokenize=False,**template_opts)
            except (TypeError,ValueError):
                rendered='\n'.join(f"{m['role'].upper()}: {m['content']}" for m in messages)+'\nASSISTANT:'
            encoded=tok(rendered,return_tensors='pt',add_special_tokens=False)
        n = encoded['input_ids'].shape[-1]
        if n > self.max_input_tokens:
            raise ValueError(f'Prompt {n} tokens > MAX_INPUT_TOKENS={self.max_input_tokens}; giảm CONTEXT_WINDOW')
        # For sharded or quantized models, the embedding device is reliable for inputs.
        device = self.model.get_input_embeddings().weight.device
        encoded = {k:v.to(device) for k,v in encoded.items()}
        with torch.inference_mode():
            out = self.model.generate(**encoded,max_new_tokens=max_new_tokens,do_sample=False,
                            pad_token_id=tok.pad_token_id or tok.eos_token_id)
        ids = out[0,n:]
        decoded = tok.decode(ids,skip_special_tokens=True).strip()
        return decoded, {'prompt_tokens':n,'generated_tokens':len(ids),
                         'hit_token_limit':len(ids)>=max_new_tokens}

    def unload(self):
        self.model=None; self.tokenizer=None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        except ImportError: pass


def find_output_manifests(input_root: Path, manifest_name: str,
                          signature: str, shard_id: str, stage: str='') -> list[Path]:
    found=[]
    if not input_root.exists(): return found
    for manifest in input_root.rglob(manifest_name):
        try: payload=json.loads(manifest.read_text(encoding='utf-8'))
        except (ValueError, OSError): continue
        if (payload.get('signature')==signature and payload.get('shard_id')==shard_id
                and (not stage or payload.get('stage')==stage)):
            found.append(manifest.parent)
    return sorted(set(found))


def restore_checkpoints(output_dir: Path, older_dirs: list[Path], filename: str, signature: str,
                        keys: set) -> dict:
    ret={}
    for directory in [*older_dirs, output_dir]:
        for row in load_jsonl(directory / filename):
            key=row.get('key')
            if row.get('ok') is True and row.get('signature')==signature and key in keys:
                if key in ret and row.get('text',row.get('result')) != ret[key].get('text',ret[key].get('result')):
                    raise ValueError(f'Checkpoint mâu thuẫn ở {key}')
                ret[key]=row
    return ret
