"""Experimental compact sentence-level judge for YHCT V2.

Fast path: short evidence-based prompt, per-batch local integer indices, compact
JSON. Output is validated and normalized to the original V2 judge result schema.
This is a new scoring prompt; do not mix checkpoints with earlier V2 prompts.
"""
import json
import re
from yhct_judge_v2 import (
    parse_judge,
    aggregate_judgments as _v2_aggregate_judgments,
    arbitrate_two_judges as _v2_arbitrate_two_judges,
)

FAST_JUDGE_VERSION = 'yhct-fast5-balanced-context-guard-v3-2026-11-28'
MAX_COMPACT_BATCH = 12
SHORT_OUTPUT_MIN_SCORE = 97.0

# Keep the substantive V2 evaluation instructions and terminology guidance.
# Shorten OUTPUT rather than stripping away the MATERIAL used to judge.
FAST_SYSTEM_EN = '''You are a FAIR, EVIDENCE-BASED reviewer of translations from Chinese into Vietnamese
in Traditional Chinese Medicine (TCM). Grade EVERY TARGET sentence independently.
The Chinese sentences immediately before/after each target are CONTEXT ONLY:
use them to resolve references and technical meanings, never evaluate them,
copy them as error evidence, or require them to appear in the Vietnamese target.
Different cases in a batch are independent, never each other's context.
In EACH numbered case, CTX_BEFORE and CTX_AFTER are Chinese REFERENCE ONLY;
TARGET_HAN is the ONLY Chinese sentence to evaluate against TARGET_VI.
Do not swap, concatenate, or attribute content from neighboring numbered cases.
Every returned ID must refer to the SAME numbered case, never the previous/next one.

REVIEW PROCEDURE (INTERNAL; do not print these steps):
1. Compare the TARGET Chinese and TARGET Vietnamese for semantic fidelity,
   important facts, herb/formula identity, measurements, negation, conditions,
   treatment roles, and additions/omissions. Read the context only if needed.
2. Verify that each alleged error reflects a REAL source-target meaning difference.
   Do not speculate or invent a specialized interpretation without evidence.
3. Classify every VERIFIED error by category and severity, THEN assign a
   REQUIRED score from 0 to 100. Never choose a high score merely to save tokens.

FAIRNESS AND TERMINOLOGY:
- ACCEPT faithful paraphrases, ordinary Vietnamese synonyms, correct Sino-Vietnamese
  terminology, concise headings, and word-order/style differences. Do NOT demand
  literal word-for-word translation, Pinyin or English medical names.
- Correct anchors: 黄芪=Hoàng kỳ; 肉桂=Nhục quế; 酸枣仁=Toan táo nhân;
  柏子仁=Bá tử nhân/Bách tử nhân; 远志=Viễn chí; 人参=Nhân sâm;
  白术=Bạch truật; 甘草=Cam thảo; 炙甘草=Chích cam thảo;
  半夏=Bán hạ; 陈皮=Trần bì; 升麻=Thăng ma; 柴胡=Sài hồ.
- 君药/为君 = quân dược / vị quân / thuốc chủ; 臣药/为臣 = thần dược /
  vị thần / thuốc phụ trợ (depending on context); 佐药/使药=tá dược/sứ dược.
- 四君子汤=Tứ Quân Tử Thang, NOT 'Thập tứ quân tử thang';
  六君子汤=Lục Quân Tử Thang; 朱砂安神丸=Chu sa an thần hoàn.
- 方解=Giải phương/Phương giải; 思考之二=Suy nghĩ thứ hai;
  思考之三=Suy nghĩ thứ ba. These are acceptable headings: do not invent
  a hidden medical meaning or replace 思考 with 'phương số'.
- 今且从俗论之 may correctly include 'tạm';
  治标 means treat manifestations/root-vs-branch, NOT 治表 (exterior).
- A claimed omission is not real if that information is ALREADY present
  in Vietnamese. A claimed addition is not real if Chinese actually supports it.
  Never mark a correct medicine name as erroneous simply for its spelling style.

REPORT ONLY REAL ERRORS:
- addition: unsupported herbs, doses, symptoms, diagnoses, instructions, facts;
- omission: material source fact, herb, amount, restriction, negation or relationship missing;
- mistranslation: changed meaning, treatment role, amount, polarity or relationship;
- terminology: demonstrably WRONG herb, formula or medical referent;
- untranslated: unapproved Chinese characters remaining in Vietnamese;
- fluency: only if broken wording obscures what the source actually means.
Severity: critical = clinically dangerous distortion (dosage/toxicity/
contraindication/critical negation); major = verified material change, missing
or invented meaning; minor = real, non-material issue. Style differences are
NOT errors. Chinese OCR uncertainty is NOT automatically a translation error.
If the source/target relationship cannot be established, mark the appropriate
uncertainty flag instead of inventing a certain error.

SCORING AFTER ERROR CHECK (subjective rubric, NOT percentage accuracy):
98-100 = no verified errors; 95-97 = at most minor real issues;
60-89 = verified major error; 0-59 = verified critical or largely unusable.
Do not assign <95 solely for a stylistic preference. Do not anchor to any
fixed example score. EVERY case MUST have a numeric score.

CONDITIONAL, MINIMAL JSON OUTPUT (one item per local integer ID; no Markdown):
A. ONLY if score >= 97 AND errors are EMPTY AND both uncertainty flags are 0,
   output a SHORT result: [ID, SCORE]. This means no error has been claimed.
B. OTHERWISE output FULL result:
   [ID, ERRORS, SCORE, NEEDS_CONTEXT, SOURCE_UNCERTAIN].
   Flags are 0 or 1. ERRORS is [] or a list of
   [CATEGORY, SEVERITY, SOURCE_SPAN, TARGET_SPAN, REASON, SUGGESTED_VI].
   Copy spans EXACTLY from the same case's TARGET Chinese/Vietnamese; never
   from context. For omission, empty TARGET_SPAN is allowed; for addition or
   fluency, empty SOURCE_SPAN is allowed. Keep REASON short but specific and
   SUGGESTED_VI in Vietnamese, not English or Pinyin.
C. A real minor error at score 97 STILL requires full output so it is not lost.
   Any major/critical claim, Chinese left untranslated, or uncertainty also
   requires FULL output even if the model assigns a high score.
D. Return EVERY supplied ID once, in order, in ONE JSON array, and nothing else.
'''

FAST_SYSTEM_VI = '''Bạn là giám khảo CÔNG BẰNG, DỰA TRÊN BẰNG CHỨNG khi đánh giá dịch Hán → Việt
Y học cổ truyền. Chỉ chấm câu Hán–Việt ĐÍCH của MỖI CASE. Câu Hán trước/sau
CHỈ là ngữ cảnh để hiểu nghĩa/đại từ/thuật ngữ; không chấm, không lấy làm
bằng chứng lỗi hoặc bắt câu Việt phải chứa nội dung của câu lân cận.
Các CASE là độc lập với nhau.
Trong từng CASE đánh số, CTX_BEFORE/CTX_AFTER là Hán NGỮ CẢNH CHỈ ĐỂ THAM KHẢO;
TARGET_HAN là câu Hán DUY NHẤT đem đối chiếu với TARGET_VI của CÙNG CASE.
TUYỆT ĐỐI không ghép nội dung từ câu trước/sau hoặc từ CASE khác, không lấy
vị thuốc/số lượng/mệnh đề ở ngữ cảnh để kết luận bản dịch câu đích thiếu ý.
ID kết quả phải thuộc CHÍNH CASE được chấm, không lẫn với ID sát bên.

THỨ TỰ ĐÁNH GIÁ (không cần in các bước):
1. Đối chiếu nghĩa câu đích, tên vị thuốc/bài thuốc, liều lượng, phủ định,
   điều kiện, vai trò quân–thần–tá–sứ, quan hệ nhân quả và thông tin thêm/bớt.
2. Chỉ ghi lỗi nếu CHỨNG MINH được khác biệt ý nghĩa thật. Không tự diễn giải
   từ phổ thông thành thuật ngữ chuyên ngành khi nguồn/ngữ cảnh không chứng minh.
3. Phân loại lỗi và mức độ, SAU ĐÓ BẮT BUỘC chấm điểm 0–100 từng câu.
   Không cho điểm cao chỉ để được trả JSON ngắn.

CHẤM CÔNG BẰNG:
- Chấp nhận diễn đạt đồng nghĩa, thuật ngữ Hán–Việt đúng, thứ tự từ, tiêu đề,
  và khác biệt văn phong không làm đổi nghĩa. Không bắt dùng Pinyin/tiếng Anh.
- Thuật ngữ đúng: 黄芪=Hoàng kỳ; 肉桂=Nhục quế; 酸枣仁=Toan táo nhân;
  柏子仁=Bá tử nhân/Bách tử nhân; 远志=Viễn chí; 人参=Nhân sâm;
  白术=Bạch truật; 甘草=Cam thảo; 炙甘草=Chích cam thảo;
  半夏=Bán hạ; 陈皮=Trần bì; 升麻=Thăng ma; 柴胡=Sài hồ.
- 君药/为君=quân dược/vị quân/thuốc chủ; 臣药/为臣=thần dược/vị thần/
  thuốc phụ trợ tùy ngữ cảnh; 佐药/使药=tá dược/sứ dược.
- 四君子汤=Tứ Quân Tử Thang, KHÔNG phải Thập tứ quân tử thang;
  六君子汤=Lục Quân Tử Thang; 朱砂安神丸=Chu sa an thần hoàn.
- 方解=Giải phương/Phương giải; 思考之二=Suy nghĩ thứ hai;
  思考之三=Suy nghĩ thứ ba (đừng bịa thành phương số ba).
- 今且从俗论之 có thể dịch có chữ 'tạm'; 治标 (trị ngọn) khác 治表 (trị biểu).
- Trước khi báo bỏ sót, kiểm tra câu Việt đã có ý ấy chưa. Trước khi báo
  bịa thêm, kiểm tra câu Hán có hỗ trợ ý ấy không.

CHỈ BÁO LỖI THỰC:
addition = thêm dữ kiện không có nguồn; omission = bỏ ý quan trọng;
mistranslation = đổi nghĩa/vai trò/phủ định/số lượng; terminology = sai hẳn
vị thuốc/bài thuốc/khái niệm; untranslated = Hán tự còn sót không được duyệt;
fluency = câu Việt sai đến mức gây hiểu nhầm nghĩa, KHÔNG phạt văn phong thường.
critical = sai lệch nguy hiểm về liều, độc tính, chống chỉ định, phủ định;
major = sai/thêm/bỏ ý có ý nghĩa; minor = lỗi nhỏ có thật, không đổi ý chính.
Nếu nguồn OCR không chắc, đánh dấu cờ nghi ngờ; không coi lỗi OCR là lỗi dịch.

THANG ĐIỂM (sau khi xác định lỗi; điểm chủ quan, KHÔNG là % chính xác):
98–100: không có lỗi xác thực; 95–97: nhiều nhất lỗi minor;
60–89: lỗi major thật; 0–59: lỗi critical hoặc sai phần lớn.
Không chấm dưới 95 chỉ vì cách diễn đạt khác sở thích.

JSON ĐẦU RA CÓ ĐIỀU KIỆN, ÍT TOKEN:
A. CHỈ KHI score >= 97, errors=[] và hai cờ nghi ngờ đều 0:
   trả kết quả NGẮN: [ID, SCORE].
B. MỌI trường hợp khác: [ID, ERRORS, SCORE, NEEDS_CONTEXT, SOURCE_UNCERTAIN].
   Cờ chỉ nhận 0 hoặc 1. Mỗi lỗi trong ERRORS gồm:
   [CATEGORY, SEVERITY, SOURCE_SPAN, TARGET_SPAN, REASON, SUGGESTED_VI].
   Trích SOURCE_SPAN/TARGET_SPAN nguyên văn từ đúng câu đích, không từ context.
   Omission cho phép TARGET_SPAN rỗng, addition/fluency cho phép SOURCE_SPAN rỗng.
   Lý do ngắn nhưng cụ thể; gợi ý sửa bằng tiếng Việt.
C. Điểm 97 mà có lỗi minor vẫn phải trả đầy đủ để giữ lỗi.
   Lỗi major/critical, Hán tự còn sót hoặc cờ nghi ngờ đều phải trả đầy đủ.
D. Mỗi ID số cục bộ phải có đúng một kết quả theo thứ tự, trả MỘT mảng JSON,
   không thêm chữ nào khác.'''


def _messages(rows, lang):
    if lang not in ('en','vi'):
        raise ValueError('FAST Judge supports en/vi')
    if not 1 <= len(rows) <= MAX_COMPACT_BATCH:
        raise ValueError('Batch size must be 1–12')
    chunks=[]
    for i,row in enumerate(rows,1):
        # Cap CONTEXT characters (not target text) to reduce prefill cost.
        before=str(row.get('ctx_before_han') or '')
        after=str(row.get('ctx_after_han') or '')
        chunks.append(f'#{i}\nCTX_BEFORE:{before or "-"}\nTARGET_HAN:{row["han"]}\nTARGET_VI:{row["viet"]}\nCTX_AFTER:{after or "-"}')
    return [
        {'role':'system','content':FAST_SYSTEM_EN if lang=='en' else FAST_SYSTEM_VI},
        {'role':'user','content':'\n\n'.join(chunks) + '\nJSON ONLY.'},
    ]


def _complete_json_array_prefix(txt):
    """Salvage ONLY complete JSON elements at the start of a truncated array.

    This never invents a missing item or closes a partially generated item.
    Callers must still verify each local ID and evidence against its OWN row.
    """
    raw=txt.lstrip()
    if raw.startswith('['):
        pos=0
    else:
        # Support a truncated {"results":[... array as emitted by some models.
        match=re.match(r'\{\s*"results"\s*:\s*\[', raw)
        if not match:
            return None
        pos=match.end()-1
    pos+=1
    decoder=json.JSONDecoder()
    valid=[]
    while True:
        while pos<len(raw) and raw[pos].isspace():pos+=1
        if pos>=len(raw):return valid or None
        if raw[pos]==']':return valid
        try:
            obj,end=decoder.raw_decode(raw,pos)
        except (ValueError,TypeError):
            return valid or None
        pos=end
        while pos<len(raw) and raw[pos].isspace():pos+=1
        if pos<len(raw) and raw[pos] not in ',]':
            return valid or None
        # Only accept a complete item that is properly separated by comma/].
        valid.append(obj)
        if pos<len(raw) and raw[pos]==']':return valid
        if pos>=len(raw):return valid
        pos+=1


def _json_array(raw):
    txt=(raw or '').strip()
    if txt.startswith('```'):
        txt=re.sub(r'^```(?:json)?\s*','',txt,flags=re.I).rsplit('```',1)[0].strip()
    try:
        obj=json.loads(txt)
        if isinstance(obj,list): return obj
        if isinstance(obj,dict) and isinstance(obj.get('results'),list): return obj['results']
    except (ValueError,TypeError): pass
    # IMPORTANT: a truncated outer list may contain complete inner lists.
    # Recover the OUTER prefix first; never mistake [ID,SCORE] inside it
    # for the entire batch response.
    if txt.lstrip().startswith('[') or re.match(r'^\s*\{\s*"results"\s*:',txt):
        partial=_complete_json_array_prefix(txt)
        if partial is not None:return partial
    decoder=json.JSONDecoder()
    for i,c in enumerate(txt):
        if c!='[': continue
        try:
            obj,_=decoder.raw_decode(txt[i:])
            if isinstance(obj,list): return obj
        except (ValueError,TypeError):continue
    # Last resort, only when the *beginning* is recognizably a JSON result
    # array. Entirely malformed outputs still require retry/fallback.
    partial=_complete_json_array_prefix(txt)
    if partial is not None:return partial
    raise ValueError('No valid complete JSON array/items in model output')


def _single_item(raw):
    """Accept [[1,99]] and [1,99] only when there is exactly one result."""
    items=_json_array(raw)
    if len(items) in (2,5) and not isinstance(items[0],list):
        return items
    if len(items)==1 and isinstance(items[0],list):
        return items[0]
    raise ValueError('Single-case output must contain exactly one local result')


def _normalize_local_id(value):
    """Accept an integer ID or its exact decimal JSON string equivalent.

    Some Qwen models emit "1" instead of 1 despite an integer-ID prompt.
    Reject booleans, fractional IDs, other representations, and invalid strings;
    the caller must still match the ID to its own TARGET row.
    """
    if type(value) is int and value >= 1:
        return value
    if isinstance(value, str) and re.fullmatch(r'[1-9][0-9]*', value):
        return int(value)
    raise ValueError(f'Invalid local ID: {value!r}')


def _parse_one(item, row, expected_id):
    """Normalize conditional short/full arrays to the ORIGINAL V2 judgment schema.

    Short [ID,SCORE] is accepted ONLY for a high-scoring, unflagged, no-error case.
    Full [ID,ERRORS,SCORE,NEEDS_CONTEXT,SOURCE_UNCERTAIN] preserves all evidence.
    """
    if not isinstance(item, list) or len(item) not in (2, 5):
        raise ValueError('Each result must have 2 or 5 fields')
    if len(item) == 2:
        i, score = item
        if isinstance(score,str) and re.fullmatch(r'(?:\d+(?:\.\d*)?|\.\d+)',score.strip()):
            score=float(score.strip())
        if type(score) not in (int,float) or not SHORT_OUTPUT_MIN_SCORE <= score <= 100:
            raise ValueError(f'Short result requires numeric score >= {SHORT_OUTPUT_MIN_SCORE:g}')
        errors, needs_context, source_uncertain = [], 0, 0
        tier = 'short_high'
    else:
        i, errors, score, needs_context, source_uncertain = item
        tier = 'detailed'
    if _normalize_local_id(i) != expected_id:
        raise ValueError(f'Expected local ID {expected_id}, got {i!r}')
    malformed_errors=[]
    if not isinstance(errors,list):
        # Preserve uncertain content and force review rather than treating
        # null/string/object as a clean empty error list.
        malformed_errors.append({'raw':errors, 'validation_error':'ERRORS is not a list'})
        errors=[]
    if len(errors)>15:
        malformed_errors.append({'raw':errors[15:], 'validation_error':'Too many errors; tail omitted'})
        errors=errors[:15]
    def flag(v):
        if type(v) is bool:return v
        if type(v) is int and v in (0,1):return bool(v)
        if isinstance(v,str) and v.strip().lower() in ('0','1','true','false'):
            return v.strip().lower() in ('1','true')
        raise ValueError('Flags must be 0/1 or unambiguous boolean strings')

    flags={'needs_context':flag(needs_context),
           'source_uncertain':flag(source_uncertain)}
    objs=[]
    unverified=list(malformed_errors)
    validated_fixes=[]
    for idx,e in enumerate(errors):
        if not isinstance(e,list) or len(e)!=6:
            unverified.append({'raw':e,'validation_error':'Error item does not have 6 fields'})
            continue
        claim=dict(zip(('category','severity','source_span','target_span',
                        'reason','suggested_translation'),e))
        # Validate EACH alleged error in isolation. A malformed/hallucinated
        # span must not cause loss of the other, independently valid claims.
        try:
            single={'errors':[claim], 'score':score,
                    'needs_context':flags['needs_context'],
                    'source_uncertain':flags['source_uncertain'],
                    'score_explanation':''}
            checked=parse_judge(json.dumps(single,ensure_ascii=False),row)
            objs.extend(checked['errors'])
            validated_fixes.extend(checked.get('format_fixes',[]))
        except (TypeError,ValueError) as exc:
            # This includes source_span/target_span quoted from CONTEXT or
            # invented by the model, invalid severity/category, and empty
            # required spans. Never manufacture evidence or silently PASS.
            claim['validation_error']=str(exc)[:250]
            claim['error_index']=idx
            unverified.append(claim)
    obj={'errors':objs,'score':score,
         'needs_context':flags['needs_context'],
         'source_uncertain':flags['source_uncertain'],
         'score_explanation':''}
    result=parse_judge(json.dumps(obj,ensure_ascii=False),row)
    result['fast_output_tier']='detailed_unverified' if unverified else tier
    if validated_fixes:
        result['format_fixes'].extend(
            'verified_error_normalized:'+fix for fix in validated_fixes
        )
    if unverified:
        # Preserve raw claims, including the potentially important but
        # non-verifiable assertions. Triage wrappers enforce REVIEW_REQUIRED
        # when the row has all necessary Judge results.
        result['requires_manual_review']=True
        result['unverified_errors']=unverified
        result['format_fixes'].append('unverified_error_claim_quarantined_for_review')
    return result


def _parse_batch(raw,rows):
    items=_json_array(raw)
    valid={}; failures={}; seen=set(); duplicates=set()
    for item in items:
        if not isinstance(item,list) or not item: continue
        try:
            i=_normalize_local_id(item[0])
        except ValueError:
            continue
        if not (1 <= i <= len(rows)): continue
        cid=str(rows[i-1]['candidate_id'])
        if cid in seen:
            duplicates.add(cid)
            valid.pop(cid,None);failures[cid]='Duplicate ID'
            continue
        seen.add(cid)
        try: valid[cid]=_parse_one(item,rows[i-1],i)
        except (TypeError,ValueError) as e: failures[cid]=str(e)
    for cid in duplicates:
        valid.pop(cid,None);failures[cid]='Duplicate ID'
    for row in rows:
        cid=str(row['candidate_id'])
        if cid not in seen: failures[cid]='Missing result ID'
    return valid,failures


def judge_batch_with_fallback(model,rows,lang,max_new_tokens,*,max_attempts=2,model_name=''):
    """Interface-compatible replacement for yhct_judge_v2.judge_batch_with_fallback.

    max_new_tokens is per-case budget, compact mode suggested 150-200.
    On partial output, retain valid cases and recursively split only failures.
    """
    if not 1<=len(rows)<=MAX_COMPACT_BATCH:raise ValueError('Batch size 1–12')
    cids=[str(r['candidate_id']) for r in rows]
    if len(cids)!=len(set(cids)):raise ValueError('Duplicate candidate IDs')
    completed={};failures={};calls=0
    def walk(group):
        nonlocal calls
        if not group:return
        if len(group)==1:
            row=group[0];cid=str(row['candidate_id'])
            for attempt in range(1,max_attempts+1):
                try:
                    calls+=1
                    raw,meta=model.infer(_messages(group,lang),max_new_tokens*2)
                    parsed=_parse_one(_single_item(raw),row,1)
                    m=dict(meta);m.update(shared_batch_size=1,shared_batch_tokens=False,
                                          inference_method='single' if len(rows)==1 else 'single_fallback')
                    completed[cid]=(parsed,m,attempt)
                    return
                except Exception as exc:
                    failures[cid]=f'{type(exc).__name__}: {str(exc)[:250]}'
                    if 'out of memory' in str(exc).lower():
                        try:
                            import torch
                            torch.cuda.empty_cache()
                        except Exception:pass
            return
        try:
            calls+=1
            raw,meta=model.infer(_messages(group,lang),max_new_tokens*len(group))
            # Even when token-limited, retain fully generated JSON items.
            # Only incomplete/invalid items are retried separately.
            valid,_=_parse_batch(raw,group)
            for row in group:
                cid=str(row['candidate_id'])
                if cid in valid:
                    m=dict(meta);m.update(shared_batch_size=len(group),shared_batch_tokens=True,inference_method='batch')
                    completed[cid]=(valid[cid],m,1)
            missing=[r for r in group if str(r['candidate_id']) not in completed]
        except Exception as exc:
            missing=list(group)
            if 'out of memory' in str(exc).lower():
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:pass
        if missing:
            if len(missing)==1 or len(missing)<len(group):
                # Partial batch success: retry only the unresolved subset
                # together before bisecting again. This avoids needlessly
                # doubling the number of costly 14B generate() calls.
                walk(missing)
            else:
                half=len(missing)//2
                walk(missing[:half]);walk(missing[half:])
    walk(rows)
    return completed,failures,calls


def migrate_legacy_judge_checkpoints(input_root, rows_full, shard, stage, new_signature,
                                     model_configs, stable_hash, utc_now):
    """Explicitly tag/import *individually validated* checkpoints from V2 prompt.

    Only old V2 judgments with an identical row ID, source/target text, context,
    model configuration and a matching original checkpoint hash may be reused.
    This intentionally mixes PROMPT versions and is labeled in each record.
    Returns a map keyed by the NEW prompt checkpoint key.
    """
    from pathlib import Path
    from yhct_v2_core import load_jsonl
    source_by_id={r['candidate_id']:r for r in rows_full}
    models={c['name']:c for c in model_configs}
    result={}
    for manifest_path in Path(input_root).rglob('judge_manifest.json'):
        try:
            manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
        except (OSError,ValueError):continue
        old_signature=manifest.get('signature')
        if (manifest.get('shard_id')!=shard or manifest.get('stage')!=stage or
            manifest.get('fast_judge_version') or not isinstance(old_signature,str) or
            old_signature==new_signature):
            continue
        old_models={str(m.get('name')):m for m in manifest.get('models',[]) if isinstance(m,dict)}
        for rec in load_jsonl(manifest_path.parent/'judge_checkpoint.jsonl'):
            cid=rec.get('candidate_id');name=rec.get('judge_model')
            row=source_by_id.get(cid);cfg=models.get(name)
            if (not row or not cfg or old_models.get(name)!=cfg or rec.get('signature')!=old_signature
                or rec.get('ok') is not True or not isinstance(rec.get('result'),dict)):
                continue
            original_key=stable_hash([old_signature,shard,cfg,cid,row['han'],row['viet'],
                                      row.get('ctx_before_han',''),row.get('ctx_after_han','')])
            if rec.get('key')!=original_key:continue
            new_key=stable_hash([new_signature,shard,cfg,cid,row['han'],row['viet'],
                                 row.get('ctx_before_han',''),row.get('ctx_after_han','')])
            new_rec=dict(rec)
            new_rec.update(signature=new_signature,key=new_key,
                           migrated_from_signature=old_signature,
                           migrated_from_prompt='v2-long-prompt',
                           migrated_at=utc_now(),
                           judge_output_format='v2-legacy')
            if new_key in result and result[new_key]['result'] != new_rec['result']:
                raise ValueError('Conflicting legacy judgments '+str(cid))
            result[new_key]=new_rec
    return result



def migrate_prior_fast5_checkpoints(input_root, rows_full, shard, stage, new_signature,
                                    model_configs, stable_hash, utc_now,
                                    source_sha=None):
    """Opt-in import from earlier FAST5 *prompt* signatures.

    Each old record must match its own content-derived key, model config,
    source/target pair and context. Output is labeled as a cross-prompt import.
    Conflicting older FAST5 votes are discarded (not silently selected).
    Prompt language may differ; migrated records retain provenance. The user
    should not use cross-prompt migration for a controlled A/B benchmark.
    """
    from pathlib import Path
    from yhct_v2_core import load_jsonl
    rows_by_id = {str(r['candidate_id']): r for r in rows_full}
    model_by_name = {cfg['name']: cfg for cfg in model_configs}
    merged = {}
    conflicts = set()
    skipped = {'malformed': 0, 'migrated_v2': 0, 'context_or_model_mismatch': 0}
    manifests_found = 0
    for manifest_path in Path(input_root).rglob('judge_manifest.json'):
        try:
            manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            continue
        prev_sig=manifest.get('signature')
        if (manifest.get('shard_id') != shard or manifest.get('stage') != stage
                or not manifest.get('fast_judge_version')
                or not isinstance(prev_sig,str) or prev_sig == new_signature):
            continue
        if (source_sha is not None and manifest.get('source_sha256')
                and manifest['source_sha256'] != source_sha):
            continue
        manifests_found+=1
        prev_models={c.get('name'):c for c in manifest.get('models',[])
                     if isinstance(c,dict)}
        for rec in load_jsonl(manifest_path.parent/'judge_checkpoint.jsonl'):
            # This record was originally a V2 result already migrated into an
            # old FAST5 output. Let the separate V2 importer handle it directly.
            if rec.get('migrated_from_prompt') == 'v2-long-prompt':
                skipped['migrated_v2']+=1
                continue
            cid=str(rec.get('candidate_id',''))
            name=rec.get('judge_model')
            row=rows_by_id.get(cid)
            cfg=model_by_name.get(name)
            result=rec.get('result')
            if (rec.get('ok') is not True or not isinstance(result,dict)
                    or not isinstance(result.get('errors'),list)
                    or isinstance(result.get('score'),bool)):
                skipped['malformed']+=1
                continue
            try:
                score=float(result['score'])
                if not 0 <= score <= 100: raise ValueError('invalid score')
            except (KeyError,ValueError,TypeError,OverflowError):
                skipped['malformed']+=1
                continue
            if (row is None or cfg is None or prev_models.get(name) != cfg
                    or rec.get('signature') != prev_sig
                    or str(rec.get('unit_id')) != str(row['unit_id'])):
                skipped['context_or_model_mismatch']+=1
                continue
            old_key=stable_hash([prev_sig,shard,cfg,row['candidate_id'],
                                row['han'],row['viet'],
                                row.get('ctx_before_han',''),row.get('ctx_after_han','')])
            if rec.get('key') != old_key:
                skipped['context_or_model_mismatch']+=1
                continue
            new_key=stable_hash([new_signature,shard,cfg,row['candidate_id'],
                                row['han'],row['viet'],
                                row.get('ctx_before_han',''),row.get('ctx_after_han','')])
            if new_key in conflicts:
                continue
            new_rec=dict(rec)
            new_rec.update(key=new_key, signature=new_signature,
                           migrated_from_signature=prev_sig,
                           migrated_from_prompt='fast5-prior-signature',
                           migrated_from_language=manifest.get('prompt_language'),
                           migrated_from_fast_judge_version=manifest.get('fast_judge_version'),
                           migrated_at=utc_now())
            if new_key in merged and merged[new_key]['result'] != new_rec['result']:
                del merged[new_key]
                conflicts.add(new_key)
            else:
                merged[new_key]=new_rec
    return merged, {'manifests_found': manifests_found,
                    'safe_records': len(merged),
                    'conflicting_keys_skipped':len(conflicts),
                    'skipped':skipped}

def aggregate_judgments(row, model_names, scores_by_model, han_allowlist=(), score_threshold=95.0):
    """Original V2 majority vote, with one safety gate for malformed omission evidence.

    An omission claim with an empty source span has not been verified. It is
    saved as a completed model result but can NEVER be auto-accepted/repaired.
    Postprocessing only: does not change prompts, signatures or checkpoint keys.
    """
    result = _v2_aggregate_judgments(row, model_names, scores_by_model,
                                      han_allowlist, score_threshold)
    unresolved = [
        {'judge': name, 'claims': scores_by_model[name].get('unverified_errors', [])}
        for name in model_names if name in scores_by_model and
        scores_by_model[name].get('requires_manual_review')
    ]
    result['unverified_claim_count'] = sum(len(x['claims']) for x in unresolved)
    result['unverified_claims_json'] = json.dumps(unresolved, ensure_ascii=False)
    if unresolved and not result['missing_models']:
        result['status'] = 'REVIEW_REQUIRED'
    return result


def arbitrate_two_judges(row, base_model_names, scores_by_model,
                          han_allowlist=(), score_threshold=95.0):
    """V2 pair arbitration; request 14B if a base judge left unverified evidence."""
    result = _v2_arbitrate_two_judges(row, base_model_names, scores_by_model,
                                       han_allowlist, score_threshold)
    if result['pair_ready'] and any(
        scores_by_model[m].get('requires_manual_review') for m in base_model_names
    ) and not result['pair_summary']['han_leftover_count']:
        result = dict(result, needs_arbiter=True,
                      reason='base_unverified_claim_requires_arbiter')
    return result
