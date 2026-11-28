"""Evidence-based sentence-level Judge and Re-judge prompts, validation and voting."""
import json
import re
import statistics
import unicodedata
from yhct_v2_core import parse_json_object, chinese_leftover

JUDGE_SYSTEM_EN = '''You are a CONSERVATIVE, EVIDENCE-BASED evaluator of Chinese-to-Vietnamese
translation in Traditional Chinese Medicine (TCM). Evaluate EXACTLY ONE TARGET
sentence, not the surrounding paragraph. Your job is to identify REAL semantic
or factual translation defects, NOT to improve style or force one preferred wording.

STRICT CONTEXT BOUNDARY:
- Previous/next Chinese sentences are CONTEXT ONLY. Use them solely to resolve
  references, meanings, or terminology in the TARGET Chinese sentence.
- Never evaluate, quote as error evidence, or demand translation of surrounding
  sentences. Errors must concern the target Chinese/target Vietnamese pair only.

EVALUATION PRINCIPLES — BE FAIR AND LENIENT:
- PRESUME that a plausible translation is acceptable until you can demonstrate a
  material source-target meaning difference. Burden of proof is on the judge.
- Accept correct Sino-Vietnamese / Vietnamese TCM names, conventional technical
  expressions, paraphrases, and synonyms. Do NOT require Pinyin, English, literal
  word-by-word wording, or your preferred alternative.
- Do NOT flag idiomatic wording, abbreviated headings, word order, punctuation,
  capitalization, stylistic variation, or a different but faithful explanation.
- Do NOT mark a word as mistranslated merely because it can have a different TCM
  meaning. Only use a specialized reading when SUPPORTED by source/context.
- Do NOT call an error "major" when your suggested phrase is semantically
  equivalent to the given Vietnamese. If unsure, set needs_context=true and
  DO NOT invent or force a confident error.
- Do not correct presumed Chinese OCR errors; set source_uncertain=true where needed.

REPORT ONLY THESE SUBSTANTIATED ERRORS:
1. addition: important information introduced without Chinese-source support,
   especially fabricated herb names, doses, symptoms, diagnoses, treatment advice,
   causal relationships, or formula ingredients.
2. omission: a meaningful source fact, herb, dosage, restriction, condition,
   negation or relationship is absent from the Vietnamese target.
3. mistranslation: a CLEAR changed factual meaning, agent/role, polarity,
   temporal order, amount, medical concept, or relationship.
4. terminology: a demonstrably wrong herb/formula/technical concept that changes
   the referent; do not penalize valid standard variants or Hán-Việt terms.
5. untranslated: source Chinese text left in the Vietnamese translation without
   an explicitly justified/approved notation (do not confuse Vietnamese
   Sino-Vietnamese words written in Latin letters with Chinese characters).
6. fluency: only if the Vietnamese is so broken that the intended meaning is
   genuinely unclear. Never flag merely awkward but comprehensible phrasing.

ERROR SEVERITY:
- critical: a clearly demonstrated and potentially dangerous change to clinical
  instructions, dosage, toxicity, contraindications, critical negation, etc.
- major: a verifiable material change/omission/invention of meaning (including
  actual wrong herb or formula identity, fabricated medical fact or amount).
- minor: a real but non-material issue; ordinarily does NOT justify Repair.
  Simple style differences are NOT errors and must NOT appear in errors[].

MANDATORY ORDER OF OUTPUT (not a written chain of thought):
1. Compare ONLY target Chinese and target Vietnamese, using context as reference.
2. List each PROVEN error in errors[]; if none, errors=[]. For each error, copy
   source_span and target_span VERBATIM from TARGET sentences, never from context.
   For an omission, target_span may be empty; for additions/fluency, source_span
   may be empty. Explain the precise difference, not a vague preference.
3. Assign category and severity conservatively.
4. AFTER errors, assign a REQUIRED numeric score 0-100, consistent with findings:
   98-100: no verified errors; meaning preserved, including acceptable synonyms;
   95-97: at most a proven non-material MINOR issue; still generally acceptable;
   60-89: PROVEN major error, calibrated to its actual effect;
   0-59: PROVEN critical or broadly unusable translation.
   Scores 90-94 should not be used to penalize merely acceptable wording.
   Never score below 95 solely because an alternative wording sounds better.
   Do NOT automatically give any fixed score merely because errors is nonempty.
   This number is a subjective quality rating, NOT a measured percentage.

TCM TERMINOLOGY CALIBRATION (illustrative, not an exhaustive or rigid glossary):
- 黄芪 -> "Hoàng kỳ"; 肉桂 -> "Nhục quế"; 酸枣仁 -> "Toan táo nhân";
  人参 -> "Nhân sâm"; 白术 -> "Bạch truật"; 甘草 -> "Cam thảo".
  These are CORRECT conventional Sino-Vietnamese names: never demand Pinyin.
- 君药 / 为君 -> "quân dược", "vị quân", "làm quân", "thuốc chủ";
  臣药 / 为臣 -> "thần dược", "vị thần", "làm thần", "thuốc phụ trợ"
  (depending on context); 佐药 / 使药 -> "tá dược" / "sứ dược".
- 四君子汤 -> "Tứ Quân Tử Thang" (NOT "Thập tứ quân tử thang");
  六君子汤 -> "Lục Quân Tử Thang"; 半夏 -> "Bán hạ";
  陈皮 -> "Trần bì"; 升麻 -> "Thăng ma"; 柴胡 -> "Sài hồ".
- 方解 -> "Giải phương" / "Phương giải"; 思考之二 -> "Suy nghĩ thứ hai";
  思考之三 -> "Suy nghĩ thứ ba". Do NOT invent a specialized meaning
  of 思考 or replace it with "phương số ..." without source evidence.
- 今且从俗论之 -> a rendering such as "Bây giờ tạm theo tục mà bàn về điều ấy"
  is acceptable; 且 can express "tạm / for now" in context.
- Distinguish 标 (ngọn, triệu chứng; e.g., 治标) from 表 (biểu); do not
  conflate medically distinct concepts merely because both sound similar.

CALIBRATION CASES (only guidance; do not evaluate these examples as target):
A. 黄芪 -> "Hoàng kỳ" or 肉桂 -> "Nhục quế": ACCEPT, not a terminology error.
B. 【方解】思考之三。 -> "【Giải phương】Suy nghĩ thứ ba.": ACCEPT;
   this is an ordinary heading; do not fabricate a hidden medical meaning.
C. 四君子汤 -> "Thập tứ quân tử thang": FLAG a major terminology error:
   "thập tứ" names a different formula from "tứ".
D. Source names three herbs but target introduces an unsupported "mỗi vị 10g":
   FLAG addition, typically major (critical if a dangerous clinical instruction).
E. Source says 人参、白术、甘草 but target substitutes different herbs:
   FLAG a major factual/terminology error, not merely a style issue.
F. 炙甘草 -> "Chích cam thảo" is CORRECT (炙 is present in source).
   五味子 -> "Ngũ vị tử" is CORRECT. Do not call these inventions.
G. 《脾胃论》 -> "Sách Tỳ vị luận" is CORRECT; a book reference is
   not omitted simply because Vietnamese adds "Sách" or omits 《》.
H. 心为君主之官 -> "tâm là quan quân chủ" is acceptable TCM usage.
   补气血 -> "Bổ khí huyết" is acceptable. Do not require English
   translations or unnecessarily literal alternatives.

BEFORE FLAGGING OMITTED OR INVENTED CONTENT, literally verify the Vietnamese
TARGET does not already contain its meaning. In particular, do not claim a
herb/dose/book title is missing when it plainly appears. If you cannot give
a meaningfully different Vietnamese correction, do NOT assign a major error.
Suggested translations MUST be Vietnamese, never English or Pinyin.

Output exactly ONE valid JSON object, NO Markdown. Fields in this order:
needs_context (boolean), source_uncertain (boolean), errors (array), score (number),
score_explanation (brief string). Error fields: category, severity, source_span,
target_span, reason, suggested_translation. NO example numeric score is supplied:
choose the score based on YOUR validated observations.'''

JUDGE_SYSTEM_VI = '''Bạn là giám khảo đánh giá bản dịch Hán -> Việt Y học cổ truyền,
đánh giá theo hướng THẬN TRỌNG, CÔNG BẰNG và dựa vào BẰNG CHỨNG.
CHỈ CHẤM MỘT CÂU ĐÍCH (TARGET), không chấm đoạn văn hay câu trước/sau.

RANH GIỚI NGỮ CẢNH:
- Hán trước/sau CHỈ để giải nghĩa, tham chiếu, xác định thuật ngữ của câu đích.
- KHÔNG xem nội dung câu lân cận là phần mà bản Việt câu đích đã bỏ sót.
- Mọi source_span/target_span phải thuộc đúng câu Hán/Việt đích, không thuộc context.

NGUYÊN TẮC CHẤM LỎNG NHƯNG KHÔNG BỎ SÓT LỖI THẬT:
- Mặc định một cách diễn đạt hợp lý có thể đúng; chỉ kết luận sai nếu chứng minh
  được sự khác biệt NỘI DUNG CÓ Ý NGHĨA giữa Hán gốc và Việt.
- Chấp nhận thuật ngữ Hán–Việt chuẩn, tên thuốc thường dùng, diễn đạt đồng nghĩa,
  cách viết tắt, đảo thứ tự từ, cách dịch tiêu đề, dấu câu, văn phong khác nhau.
- Không đòi phiên âm Pinyin, tiếng Anh hoặc diễn đạt đúng theo sở thích của model.
- Nếu hai cách diễn đạt giữ nguyên nghĩa, KHÔNG ghi lỗi, KHÔNG trừ điểm.
- Không tự bịa nghĩa chuyên ngành cho từ phổ thông khi Hán gốc/ngữ cảnh không
  chứng minh. Nếu thực sự thiếu ngữ cảnh: needs_context=true, không phán bừa.
- Không coi lỗi OCR của nguồn Hán là lỗi người dịch; đánh dấu source_uncertain=true.

CHỈ GHI LỖI CÓ BẰNG CHỨNG:
- addition: bịa/thêm dữ kiện không có nguồn, đặc biệt vị thuốc, liều dùng, bệnh,
  chẩn đoán, triệu chứng, công dụng hay quan hệ nhân quả.
- omission: thiếu một dữ kiện quan trọng, tên vị thuốc, phủ định, liều lượng,
  điều kiện, quan hệ hoặc ý nghĩa chính.
- mistranslation: đổi hẳn nghĩa, phủ định, tên gọi, vai trò, số lượng hoặc quan hệ.
- terminology: gọi SAI vị thuốc, phương thuốc hay khái niệm thực sự; không phạt
  biến thể Hán–Việt đúng hoặc cách nói đồng nghĩa.
- untranslated: còn Hán tự chưa dịch trong câu Việt, không thuộc ngoại lệ đã duyệt.
- fluency: chỉ khi diễn đạt tệ đến mức hiểu sai/không hiểu nghĩa, không phạt văn vụng.

MỨC ĐỘ:
- critical: lỗi có căn cứ, có thể gây nguy hiểm về liều, độc tính, chống chỉ định,
  hướng dẫn điều trị hoặc phủ định quan trọng.
- major: sai/thêm/bỏ ý có ý nghĩa thực sự, bịa dữ kiện hoặc sai tên thuốc/bài thuốc.
- minor: lỗi nhỏ có thật nhưng không làm đổi ý chính, thường KHÔNG cần Repair.
  Khác biệt văn phong đơn thuần KHÔNG phải là lỗi.

TRÌNH TỰ BẮT BUỘC CỦA KẾT QUẢ:
(1) đối chiếu câu đích; (2) GHI LỖI, trích đúng source_span và target_span nguyên văn;
(3) phân loại category và severity; (4) SAU CÙNG cho score số 0–100.
Lỗi omission được để target_span rỗng; addition/fluency được phép để source_span rỗng.
Nếu không có lỗi chứng minh được: errors=[]. KHÔNG bịa lỗi chỉ để trừ điểm.

THANG ĐIỂM:
98–100: không có lỗi có bằng chứng, nghĩa được giữ đúng, kể cả từ đồng nghĩa;
95–97: nhiều nhất lỗi minor có thật, không đổi nghĩa chính, vẫn cơ bản chấp nhận;
60–89: lỗi major đã chứng minh, tùy độ ảnh hưởng;
0–59: lỗi critical đã chứng minh hoặc bản dịch sai phần lớn.
Không dùng vùng 90–94 để phạt các cách diễn đạt đúng nhưng khác sở thích.
Không cho điểm dưới 95 chỉ vì có thể diễn đạt hay hơn. Không cố định một điểm
cho tất cả câu có lỗi. Đây là điểm quy ước, KHÔNG là % chính xác thực đo.

VÍ DỤ THUẬT NGỮ ĐÚNG (không phạt):
黄芪 = Hoàng kỳ; 肉桂 = Nhục quế; 酸枣仁 = Toan táo nhân;
人参 = Nhân sâm; 白术 = Bạch truật; 甘草 = Cam thảo;
君药/为君 = quân dược/vị quân/làm quân/thuốc chủ;
臣药/为臣 = thần dược/vị thần/làm thần/thuốc phụ trợ (tùy ngữ cảnh);
佐药 = tá dược; 使药 = sứ dược;
四君子汤 = Tứ Quân Tử Thang; 六君子汤 = Lục Quân Tử Thang;
半夏 = Bán hạ; 陈皮 = Trần bì; 升麻 = Thăng ma; 柴胡 = Sài hồ;
方解 = Giải phương/Phương giải; 思考之二 = Suy nghĩ thứ hai;
思考之三 = Suy nghĩ thứ ba.
Ví dụ 今且从俗论之 -> "Bây giờ tạm theo tục mà bàn về điều ấy" là hợp lý.
PHÂN BIỆT 标 (ngọn, trị ngọn như 治标) và 表 (biểu).

VÍ DỤ LỖI THẬT:
四君子汤 = "Thập tứ quân tử thang" là lỗi MAJOR vì sai tên bài thuốc.
Nếu bản Hán không có liều dùng mà bản Việt tự thêm "mỗi vị 10g", đây là
ADDITION MAJOR (hoặc CRITICAL nếu tạo chỉ dẫn nguy hiểm).
Nếu nguồn nêu 人参、白术、甘草 nhưng bản Việt đổi thành các vị thuốc khác, là MAJOR.
炙甘草 = Chích cam thảo; 五味子 = Ngũ vị tử; 补气血 = Bổ khí huyết;
心为君主之官 = tâm là quan quân chủ; 《脾胃论》 = Sách Tỳ vị luận.
Đây là các cách diễn đạt ĐÚNG, không ghi lỗi bịa thêm/thiếu ý.
Trước khi báo omission hoặc addition, kiểm tra đúng nghĩa tương ứng đã
có trong câu Việt chưa. Nếu không thể nêu bản sửa khác biệt về nghĩa,
KHÔNG gắn lỗi major. suggested_translation phải bằng TIẾNG VIỆT.

Chỉ xuất MỘT JSON hợp lệ, không Markdown. Thứ tự: needs_context,
source_uncertain, errors, score, score_explanation. Mỗi error có category,
severity, source_span, target_span, reason, suggested_translation.
Không sao chép ví dụ để chấm câu đích.'''

JUDGE_USER_EN = '''CONTEXT ONLY — PREVIOUS CHINESE SENTENCE(S) [DO NOT EVALUATE]:
{before}

====== TARGET CHINESE SENTENCE [EVALUATE ONLY THIS] ======
{han}
====== END TARGET CHINESE ======

TARGET VIETNAMESE TRANSLATION [EVALUATE ONLY THIS]:
{viet}

CONTEXT ONLY — NEXT CHINESE SENTENCE(S) [DO NOT EVALUATE]:
{after}

Return exactly one JSON object containing keys in this order:
needs_context, source_uncertain, errors, score, score_explanation.
needs_context and source_uncertain MUST be JSON booleans.
errors MUST be an array (possibly empty). Each error MUST have:
category (mistranslation/omission/addition/terminology/untranslated/fluency),
severity (minor/major/critical), source_span (exact substring of target Chinese),
target_span (exact substring of target Vietnamese), reason (specific meaning
mismatch), suggested_translation (optional phrase, or empty string).
score MUST be a number from 0 to 100, assigned AFTER examining the errors.
Do not reproduce any template error or default score. If no proven errors,
return errors=[] and a high score. Copy spans EXACTLY, including punctuation;
do NOT quote content from context-only sentences.'''

JUDGE_USER_VI = '''NGỮ CẢNH HÁN TRƯỚC (CHỈ THAM KHẢO, KHÔNG CHẤM):
{before}

====== HÁN GỐC CÂU ĐÍCH — CHỈ CHẤM CÂU NÀY ======
{han}
====== KẾT THÚC CÂU ĐÍCH ======

BẢN VIỆT CÂU ĐÍCH — CHỈ CHẤM CÂU NÀY:
{viet}

NGỮ CẢNH HÁN SAU (CHỈ THAM KHẢO, KHÔNG CHẤM):
{after}

Trả MỘT JSON hợp lệ, các khóa đúng thứ tự: needs_context,
source_uncertain, errors, score, score_explanation.
Hai trường đầu là boolean; errors là danh sách (có thể rỗng).
Mỗi lỗi gồm category, severity, source_span, target_span, reason,
suggested_translation. Mỗi span phải trích NGUYÊN VĂN từ đúng câu đích,
kể cả dấu câu; không trích ngữ cảnh. Lỗi omission được phép target_span rỗng,
addition/fluency được phép source_span rỗng.
score bắt buộc là số 0–100, CHỈ cho sau khi đã xác định lỗi.
Nếu không có lỗi được chứng minh: errors=[] và điểm cao; không sao chép một
lỗi mẫu hoặc điểm mặc định.'''

# GLM calibration is deliberately model-specific.  Do not change the rubric,
# category schema, or score ranges; only address demonstrated false positives.
JUDGE_GLM_CALIBRATION_EN = """GLM-SPECIFIC QUALITY CONTROL (read before judging):
Do not use English paraphrases as a correction to a Vietnamese translation.
Sino-Vietnamese medical names are often already correct: 肉桂=Nhục quế,
酸枣仁=Toan táo nhân, 远志=Viễn chí, 炙甘草=Chích cam thảo,
五味子=Ngũ vị tử, 补气血=Bổ khí huyết, and 《脾胃论》=Sách Tỳ vị luận.
An alternative style or word order is not a meaning error. Re-read the exact
TARGET Vietnamese to check whether an allegedly omitted herb, amount or title
is already there. If the proposed corrected Vietnamese preserves the exact
meaning of the current Vietnamese, remove the error and score accordingly.
Do not call a translation major while giving it an acceptable 95+ score.
Never evaluate context sentences as target sentences.
"""
JUDGE_GLM_CALIBRATION_VI = """GHI CHÚ CHO GLM: Chỉ đối chiếu câu đích, không lấy ngữ cảnh làm lỗi.
Chấp nhận Nhục quế, Toan táo nhân, Viễn chí, Chích cam thảo, Ngũ vị tử,
Bổ khí huyết, Sách Tỳ vị luận là cách dịch Hán–Việt hợp lý.
Đừng đề xuất câu tiếng Anh làm bản sửa tiếng Việt. Nếu chỉ khác văn phong,
không báo major. Kiểm tra sự tồn tại thật của thông tin trước khi báo bỏ sót.
"""

# Curated EXACT source→target terminology anchors. The validator uses these
# narrowly: it can reject a claim against an already-correct local term, but
# cannot assert an entire sentence is correct merely from this glossary.
APPROVED_TCM_TERMS = {
    '黄芪': ('Hoàng kỳ',), '肉桂': ('Nhục quế',),
    '酸枣仁': ('Toan táo nhân',), '远志': ('Viễn chí',),
    '柏子仁': ('Bá tử nhân', 'Bách tử nhân'),
    '人参': ('Nhân sâm',), '白术': ('Bạch truật',),
    '甘草': ('Cam thảo',), '炙甘草': ('Chích cam thảo',),
    '五味子': ('Ngũ vị tử',),
    '朱砂安神丸': ('Chu sa an thần hoàn', 'Hoàn Chu sa an thần'),
    '四君子汤': ('Tứ quân tử thang', 'Tứ quân tử'),
    '六君子汤': ('Lục quân tử thang', 'Lục quân tử'),
    '半夏': ('Bán hạ',), '陈皮': ('Trần bì',),
    '升麻': ('Thăng ma',), '柴胡': ('Sài hồ',),
    '补气血': ('Bổ khí huyết',),
    '脾胃论': ('Tỳ vị luận', 'Sách Tỳ vị luận'),
    '思考之二': ('Suy nghĩ thứ hai',),
    '思考之三': ('Suy nghĩ thứ ba',),
    '思考之四': ('Suy nghĩ thứ tư',),
}

TRIAGE_VERSION = 'judge-triage-v3-majority-safe-2026-11-27'

ERROR_CATEGORIES = {'mistranslation','omission','addition','terminology','untranslated','fluency'}
SEVERITIES = {'critical','major','minor'}


def prompt_messages(row: dict, lang='en', model_name: str = '') -> list[dict]:
    if lang not in ('en','vi'): raise ValueError('JUDGE_PROMPT_LANGUAGE chỉ dùng en hoặc vi')
    system = JUDGE_SYSTEM_EN if lang=='en' else JUDGE_SYSTEM_VI
    if 'glm' in model_name.lower():
        system += '\n' + (JUDGE_GLM_CALIBRATION_EN if lang=='en' else JUDGE_GLM_CALIBRATION_VI)
    user = (JUDGE_USER_EN if lang=='en' else JUDGE_USER_VI).format(
        before=row.get('ctx_before_han') or '(not available / không có)',
        han=row['han'],viet=row['viet'],
        after=row.get('ctx_after_han') or '(not available / không có)')
    return [{'role':'system','content':system},{'role':'user','content':user}]



# Only apply deterministic, meaning-preserving formatting normalization.  Never
# auto-correct a paraphrased/imagined error span or a source/context mismatch.
_PUNCT_ALIASES = str.maketrans({
    '。':'.','．':'.','｡':'.','，':',','、':',',
    '：':':','；':';','？':'?','！':'!',
    '（':'(', '）':')', '［':'[', '］':']',
    '“':'"', '”':'"', '‘':"'", '’':"'",
})


def _span_match_key(span: str, *, with_positions=False):
    chars=[]; positions=[]
    for i, ch in enumerate(span):
        if ch.isspace():
            continue
        chars.append(ch.translate(_PUNCT_ALIASES))
        positions.append(i)
    return (''.join(chars), positions) if with_positions else ''.join(chars)


def _exact_target_span(span: str, target: str, field: str):
    """Return EXACT source/target text; repair punctuation/whitespace only if unique."""
    if not span:
        return '', False
    if span in target:
        return span, False
    # Sometimes models wrap a quoted span in quotation marks not present in text.
    candidates=[span]
    unquoted=span.strip().strip('\"“”‘’\'`')
    if unquoted and unquoted != span:
        candidates.append(unquoted)
    normalized_target, indices=_span_match_key(target, with_positions=True)
    for candidate in candidates:
        if candidate in target:
            return candidate, True
        key=_span_match_key(candidate)
        if not key:
            continue
        first=normalized_target.find(key)
        if first<0 or normalized_target.find(key, first+1)>=0:
            continue  # ambiguous match: never guess which occurrence is intended
        actual=target[indices[first]:indices[first+len(key)-1]+1]
        return actual, True
    raise ValueError(f'{field} không thuộc TARGET {"Hán" if field=="source_span" else "Việt"}: {span[:85]!r}')


def _parse_explicit_boolean(value, field):
    if type(value) is bool:
        return value, False
    if isinstance(value, str) and value.strip().lower() in ('true','false'):
        return value.strip().lower() == 'true', True
    raise ValueError(f'Sai/missing boolean: {field}')


def parse_judge(raw: str, row: dict) -> dict:
    obj=parse_json_object(raw)
    fixes=[]
    bools={}
    for key in ('needs_context','source_uncertain'):
        bools[key], fixed=_parse_explicit_boolean(obj.get(key), key)
        if fixed: fixes.append(key + '_string_to_boolean')
    score=obj.get('score')
    if isinstance(score,str) and re.fullmatch(r'(?:\d+(?:\.\d*)?|\.\d+)',score.strip()):
        score=float(score.strip()); fixes.append('score_numeric_string_to_number')
    if type(score) not in (int,float) or not 0 <= score <= 100:
        raise ValueError('Judge phải trả score số từ 0 đến 100')
    items=obj.get('errors')
    if not isinstance(items,list) or len(items)>15:
        raise ValueError('errors phải là list tối đa 15 lỗi')
    errors=[]
    for idx,it in enumerate(items):
        if not isinstance(it,dict): raise ValueError('error phải là object')
        c=it.get('category'); sev=it.get('severity')
        if c not in ERROR_CATEGORIES or sev not in SEVERITIES:
            raise ValueError(f'category/severity không hợp lệ: {c}/{sev}')
        source=it.get('source_span'); target=it.get('target_span')
        if not isinstance(source,str) or not isinstance(target,str):
            raise ValueError('source_span, target_span bắt buộc là string')
        if c not in ('addition','fluency') and not source:
            raise ValueError(f'category={c} phải chỉ rõ source_span')
        if c != 'omission' and not target:
            raise ValueError(f'category={c} phải chỉ rõ target_span')
        source, source_fixed=_exact_target_span(source,row['han'],'source_span')
        target, target_fixed=_exact_target_span(target,row['viet'],'target_span')
        if source_fixed: fixes.append(f'errors[{idx}].source_span_format')
        if target_fixed: fixes.append(f'errors[{idx}].target_span_format')
        reason=it.get('reason')
        if not isinstance(reason,str) or len(reason.strip())<6:
            raise ValueError('Mỗi lỗi phải có reason cụ thể')
        errors.append({'category':c,'severity':sev,'source_span':source,
            'target_span':target,'reason':reason.strip()[:650],
            'suggested_translation':str(it.get('suggested_translation') or '')[:250]})
    warning=[]
    max_sev = ('critical' if any(e['severity']=='critical' for e in errors) else
               'major' if any(e['severity']=='major' for e in errors) else
               'minor' if errors else 'none')
    if max_sev=='none' and score<95: warning.append('score_low_without_errors')
    if max_sev in ('critical','major') and score>=95: warning.append('score_high_despite_serious_error')
    if max_sev=='critical' and score>59: warning.append('critical_score_not_in_rubric')
    if max_sev=='major' and score>89: warning.append('major_score_not_in_rubric')
    if max_sev=='minor' and score<95: warning.append('minor_score_not_in_rubric')
    return {'errors':errors,'score':float(score),'score_explanation':str(obj.get('score_explanation') or '')[:600],
            'needs_context':bools['needs_context'],'source_uncertain':bools['source_uncertain'],
            'max_severity':max_sev,'score_warnings':warning,'format_fixes':fixes}


def judge_with_retries(model, row: dict, lang: str, max_new_tokens: int, *, max_attempts=3, model_name: str = ''):
    """Generate, validate and request a targeted correction (not a fresh blind guess)."""
    if not isinstance(max_attempts,int) or max_attempts<1:
        raise ValueError('max_attempts phải >=1')
    base=prompt_messages(row,lang,model_name=model_name)
    retry_history=[]
    for attempt in range(max_attempts):
        messages=base + retry_history
        raw,meta=model.infer(messages,max_new_tokens)
        try:
            if meta.get('hit_token_limit'):
                raise ValueError('Generation bị cắt ở MAX_NEW_TOKENS')
            parsed=parse_judge(raw,row)
            return parsed,meta,attempt+1
        except (ValueError,TypeError) as exc:
            if attempt == max_attempts-1:
                raise ValueError(f'{max_attempts} attempts failed: {exc}') from exc
            # The old notebook omitted the model's INVALID JSON in the retry,
            # making 'correct your last answer' impossible. Keep it for targeted retry.
            retry_history=[
                {'role':'assistant','content':raw[:3200] if isinstance(raw,str) else str(raw)[:3200]},
                {'role':'user','content':(
                    'VALIDATION FAILED: '+str(exc)[:350]+'\n'
                    'Revise the previous JSON. Quote source_span ONLY from TARGET CHINESE '
                    'and target_span ONLY from TARGET VIETNAMESE, copying EXACT characters. '
                    'If the supposed error cannot be supported by these exact sentences, '
                    'REMOVE that error and update the score. Do NOT quote a context sentence. '
                    'needs_context and source_uncertain must be true or false (JSON booleans). '
                    'Keep score a numeric value 0-100, and return ONLY one complete JSON object.'
                )}
            ]
    raise AssertionError('unreachable')


def prompt_messages_batch(rows: list[dict], lang='en', model_name: str = '') -> list[dict]:
    """One inference for 2-12 independent sentence-level judgments.

    Each case has its OWN context. Never compare one case to another or judge
    its neighboring case as context. The original single-case rubric is kept.
    """
    if not 2 <= len(rows) <= 12:
        raise ValueError('Batch Judge chỉ nhận 2-12 câu')
    ids = [str(row['candidate_id']) for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('candidate_id trùng trong batch')
    base = prompt_messages(rows[0], lang, model_name)
    if lang == 'en':
        extra_system = (
            '\nBATCH EVALUATION: You now receive MULTIPLE INDEPENDENT TARGET cases. '
            'Apply the exact SAME rubric to each case SEPARATELY. Each case has '
            'its OWN Chinese context, Chinese target, and Vietnamese target. '
            'Do NOT transfer judgments, factual content, scores, or errors across '
            'cases. In every error, source_span and target_span must be copied '
            'verbatim from the target sentence WITHIN THAT SAME CASE. '
            'Do not downgrade quality because another case has an error. '
            'Return only ONE JSON object with a results array, one result per '
            'candidate_id. Every result must contain needs_context (boolean), '
            'source_uncertain (boolean), errors (array), score (0-100 number), '
            'and score_explanation (string). Do not omit any case.'
        )
        header = (
            'Judge each independent case strictly within its CASE_START/CASE_END '
            'boundaries. Context applies ONLY within its own case.\n'
            'OUTPUT: {"results":[{"candidate_id":"<exact id>", '
            '"needs_context":false, "source_uncertain":false, '
            '"errors":[], "score":<number>, "score_explanation":"..."}, ...]}.\n'
            'Do not copy example values. One entry per supplied candidate_id.\n'
        )
        def format_case(row, index):
            return (f'\n=== CASE_START {index} / candidate_id={json.dumps(str(row["candidate_id"]), ensure_ascii=False)} ===\n'
                f'CONTEXT BEFORE (CHINESE, REFERENCE ONLY, NOT A TARGET):\n{row.get("ctx_before_han") or "(none)"}\n'
                f'TARGET CHINESE (EVALUATE ONLY THIS):\n{row["han"]}\n'
                f'TARGET VIETNAMESE (EVALUATE ONLY THIS):\n{row["viet"]}\n'
                f'CONTEXT AFTER (CHINESE, REFERENCE ONLY, NOT A TARGET):\n{row.get("ctx_after_han") or "(none)"}\n'
                f'=== CASE_END {index} ===\n')
    elif lang == 'vi':
        extra_system = (
            '\nCHẤM THEO NHÓM: Đây là nhiều câu ĐỘC LẬP. Với mỗi CASE, chỉ chấm '
            'cặp Hán–Việt ĐÍCH của CASE đó, dùng ngữ cảnh riêng chỉ để hiểu nghĩa. '
            'Không lấy câu ở CASE khác làm ngữ cảnh hay bằng chứng lỗi. '
            'Mỗi lỗi phải trích nguyên văn từ đúng câu đích trong CASE đó. '
            'Mỗi CASE cần errors, score và các trường đúng schema như chấm đơn lẻ. '
            'Trả MỘT object JSON với mảng results, đủ mọi candidate_id.'
        )
        header = (
            'Chấm RIÊNG từng CASE, không dùng dữ liệu của CASE khác.\n'
            'ĐẦU RA: {"results":[{"candidate_id":"<ID chính xác>", '
            '"needs_context":false, "source_uncertain":false, '
            '"errors":[], "score":<số>, "score_explanation":"..."}, ...]}.\n'
            'Không sao chép các giá trị ví dụ. Mỗi ID chỉ xuất hiện một lần.\n'
        )
        def format_case(row, index):
            return (f'\n=== CASE_START {index} / candidate_id={json.dumps(str(row["candidate_id"]), ensure_ascii=False)} ===\n'
                f'NGỮ CẢNH HÁN TRƯỚC (KHÔNG CHẤM):\n{row.get("ctx_before_han") or "(không có)"}\n'
                f'CÂU HÁN ĐÍCH (CHỈ CHẤM CÂU NÀY):\n{row["han"]}\n'
                f'CÂU VIỆT ĐÍCH (CHỈ CHẤM CÂU NÀY):\n{row["viet"]}\n'
                f'NGỮ CẢNH HÁN SAU (KHÔNG CHẤM):\n{row.get("ctx_after_han") or "(không có)"}\n'
                f'=== CASE_END {index} ===\n')
    else:
        raise ValueError('Ngôn ngữ prompt chỉ en/vi')
    return [{'role':'system','content':base[0]['content']+extra_system},
            {'role':'user','content':header+''.join(format_case(r,i) for i,r in enumerate(rows,1))}]


def parse_judge_batch(raw: str, rows: list[dict]) -> tuple[dict, dict]:
    """Validate each returned case independently; never infer missing cases.

    A bad item or absent ID is NOT counted as a completed judgment. Valid
    sibling items are retained and only failed cases require single retry.
    """
    obj = parse_json_object(raw)
    items = obj.get('results')
    if not isinstance(items, list) or len(items)>len(rows)+3:
        raise ValueError('Batch JSON phải chứa results là mảng hợp lý')
    expected = {str(row['candidate_id']): row for row in rows}
    valid = {}
    invalid = {}
    duplicates = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        cid = item.get('candidate_id')
        if not isinstance(cid, str) or cid not in expected:
            continue  # never apply an unknown result to another case
        if cid in valid or cid in invalid:
            duplicates.add(cid)
            valid.pop(cid,None)
            invalid[cid]='candidate_id bị lặp trong kết quả batch'
            continue
        try:
            valid[cid] = parse_judge(json.dumps(item, ensure_ascii=False), expected[cid])
        except (ValueError, TypeError) as exc:
            invalid[cid] = str(exc)
    for cid in duplicates:
        valid.pop(cid,None)
        invalid[cid] = 'candidate_id bị lặp trong kết quả batch'
    for cid in expected:
        if cid not in valid and cid not in invalid:
            invalid[cid] = 'Model không trả mục tương ứng với candidate_id'
    return valid,invalid


def judge_batch_with_fallback(model, rows: list[dict], lang: str,
        max_new_tokens: int, *, max_attempts: int = 3, model_name: str = ''):
    """Judge 1-12 cases; adaptively split failed/oversized batches.

    A multi-case prompt contains only independent target cases (no tensor batch).
    If a group is too large for the model context/VRAM, or has invalid/truncated
    JSON, bisect the group and retry smaller chunks. If only some outputs are
    valid, keep them and process the unresolved IDs only. Final fallback uses
    the existing JSON-repair retry on individual cases.

    Returns completed[candidate_id] = (result, token_meta, attempts),
            failures[candidate_id] = reason, total number of model calls.
    """
    if not 1 <= len(rows) <= 12:
        raise ValueError('Batch chỉ hỗ trợ 1 đến 12 câu')
    ids = [str(r['candidate_id']) for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('candidate_id trùng trong batch')

    completed = {}
    failures = {}
    calls = 0

    def process(group):
        nonlocal calls
        if not group:
            return
        if len(group) == 1:
            row = group[0]
            cid = str(row['candidate_id'])
            if cid in completed:
                return
            try:
                result, meta, attempts = judge_with_retries(
                    model, row, lang, max_new_tokens,
                    max_attempts=max_attempts, model_name=model_name)
                calls += attempts
                per_meta = dict(meta)
                per_meta.update({
                    'shared_batch_size': 1,
                    'shared_batch_tokens': False,
                    'inference_method': 'single' if len(rows) == 1 else 'single_fallback',
                })
                completed[cid] = (result, per_meta, attempts)
            except Exception as exc:
                # The underlying routine uses up to max_attempts calls; its
                # exception does not expose the exact count on failure.
                calls += max_attempts
                failures[cid] = f'{type(exc).__name__}: {str(exc)[:300]}'
            return

        try:
            calls += 1
            raw, meta = model.infer(
                prompt_messages_batch(group, lang, model_name),
                max_new_tokens * len(group))
            if meta.get('hit_token_limit'):
                raise ValueError('Batch JSON có nguy cơ bị cắt do max_new_tokens')
            valid, _ = parse_judge_batch(raw, group)
            for row in group:
                cid = str(row['candidate_id'])
                if cid in valid:
                    per_meta = dict(meta)
                    per_meta.update({
                        'shared_batch_size': len(group),
                        'shared_batch_tokens': True,
                        'inference_method': 'batch',
                    })
                    completed[cid] = (valid[cid], per_meta, 1)
            unresolved = [r for r in group if str(r['candidate_id']) not in completed]
            if not unresolved:
                return
        except Exception as exc:
            unresolved = list(group)
            # A CUDA OOM can leave cached allocations; release inactive cache
            # before the smaller retry. Do not discard successful checkpoints.
            if 'out of memory' in str(exc).lower() or 'cuda oom' in str(exc).lower():
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

        if len(unresolved) == 1:
            process(unresolved)
        else:
            mid = len(unresolved) // 2
            process(unresolved[:mid])
            process(unresolved[mid:])

    process(list(rows))
    return completed, failures, calls


def _clean_phrase(text: str) -> str:
    """Conservative normalization: case, unicode width, whitespace, punctuation.

    This intentionally DOES NOT strip Vietnamese diacritics, numbers, or words.
    """
    t=unicodedata.normalize('NFKC',str(text or '')).casefold()
    return re.sub(r'[\W_]+','',t,flags=re.UNICODE)


def _span_overlap(a: str, b: str) -> bool:
    x, y = _clean_phrase(a), _clean_phrase(b)
    return bool(x and y and (x == y or (min(len(x),len(y))>=4 and (x in y or y in x))))


def _approved_term_claim(row: dict, issue: dict) -> str | None:
    """Reject ONLY narrow, demonstrably erroneous criticisms of known-correct terms.

    For a broad source span containing multiple facts, do not guess that the
    glossary validates the whole sentence. No semantic 'verification' is claimed.
    """
    src = issue.get('source_span','').strip('《》【】「」。，：； ')
    tgt = _clean_phrase(issue.get('target_span',''))
    if src not in APPROVED_TCM_TERMS: return None
    approved = tuple(_clean_phrase(x) for x in APPROVED_TCM_TERMS[src])
    cat = issue.get('category')
    if cat in ('terminology','mistranslation') and tgt and tgt in approved:
        return 'approved_exact_tcm_term'
    if cat=='omission' and not issue.get('target_span'):
        if any(term and term in _clean_phrase(row['viet']) for term in approved):
            return 'approved_term_already_in_target'
    if cat=='addition' and tgt in approved and src in row.get('han',''):
        return 'approved_term_is_source_supported'
    return None


def _screen_issue(row: dict, issue: dict) -> str | None:
    """Return an objective screen reason, or None if semantic truth is unresolved.

    A passing screen is NOT independent proof of an LLM's semantic interpretation.
    """
    by_glossary = _approved_term_claim(row,issue)
    if by_glossary:return by_glossary
    if issue['category'] in ('mistranslation','terminology','fluency'):
        proposed = _clean_phrase(issue.get('suggested_translation',''))
        original = _clean_phrase(issue.get('target_span',''))
        if proposed and original and proposed==original:
            return 'proposed_correction_identical_to_target'
    return None


def same_issue(a: dict, b: dict) -> bool:
    """Strict consensus suitable for sending to automatic Repair.

    Shared spans alone are not enough. Require the SAME error category, the
    SAME target/source region, and a compatible actual Vietnamese correction.
    Unresolved/underspecified claims should go to REVIEW, not auto Repair.
    """
    if a.get('category')!=b.get('category'):
        return False
    category=a.get('category')
    src_a,src_b=a.get('source_span',''),b.get('source_span','')
    tgt_a,tgt_b=a.get('target_span',''),b.get('target_span','')
    # Omissions have no target span, additions may have no source span.
    if category=='omission':
        if not _span_overlap(src_a,src_b):return False
        if tgt_a and tgt_b and not _span_overlap(tgt_a,tgt_b):return False
    elif category in ('addition','fluency') and not src_a and not src_b:
        if not _span_overlap(tgt_a,tgt_b):return False
    else:
        if not (_span_overlap(src_a,src_b) and _span_overlap(tgt_a,tgt_b)):
            return False
    sa = _clean_phrase(a.get('suggested_translation',''))
    sb = _clean_phrase(b.get('suggested_translation',''))
    if not sa or not sb or not _span_overlap(sa,sb):
        return False
    # Must propose an actual change, not just copy the flagged target phrase.
    original_a, original_b = _clean_phrase(tgt_a),_clean_phrase(tgt_b)
    if (original_a and sa==original_a) or (original_b and sb==original_b):
        return False
    return True


def _high_risk_claim(row: dict, issue: dict) -> bool:
    """Conservative risk routing, NOT a proof that the reported error exists."""
    if issue.get('severity')=='critical':return True
    cat = issue.get('category')
    src = str(issue.get('source_span') or '')
    tgt = str(issue.get('target_span') or '')
    if cat=='untranslated':return True
    if cat=='addition':
        # Invented clinical information must be reviewed even from one judge.
        return bool(re.search(r'\d|克|钱|两|g\b|mg\b|ml\b|liều|thuốc|vị|bệnh',tgt,re.I))
    if re.search(r'\d|[零一二三四五六七八九十百千万][钱两克]|\b(?:mg|ml|g)\b|\b(?:\d+\s*(?:g|mg|ml|tiền))\b',src+' '+tgt,re.I):
        return True
    if any(word in src for word in ('禁','不可','不得','不宜','禁忌','毒','无毒','治标','治本','剂量','配伍禁忌')):
        return True
    # A claim about the *identity* of a particular herb/formula is high risk;
    # broad claims about an entire sentence containing some herb are not.
    exact_src=src.strip('《》【】「」。，：； ')
    if exact_src in APPROVED_TCM_TERMS:
        # A correct canonical translation would already have been screened.
        return True
    if cat in ('terminology','mistranslation') and ('汤' in exact_src or '丸' in exact_src):
        if len(exact_src)<=12:return True
    return False


def aggregate_judgments(row: dict, model_names: list[str], scores_by_model: dict,
                        han_allowlist=(), score_threshold=95.0) -> dict:
    """Triage based on verified formatting and cautious consensus, never pure score.

    ACCEPTED_* means automatic preliminary triage, NOT human-verified truth.
    Preserve every raw Judge claim in feedback_json for audits. Automatic Repair
    receives ONLY consensus_issues_json. A lone risky error -> REVIEW_REQUIRED.
    """
    done={m:scores_by_model[m] for m in model_names if m in scores_by_model}
    missing=[m for m in model_names if m not in done]
    feedback=[]; screened=[]; serious=[]; minority=[]
    clean_models=[]
    for model, val in done.items():
        raw_errors=val.get('errors') or []
        filtered=[]
        for e in raw_errors:
            reason=_screen_issue(row,e)
            if reason:
                screened.append({'judge':model,'screen_reason':reason,'issue':e})
            else:
                filtered.append(e)
                if e.get('severity') in ('major','critical'):
                    serious.append((model,e))
        if val.get('score',0)>=score_threshold and not any(e.get('severity') in ('major','critical') for e in filtered):
            clean_models.append(model)
        feedback.append({'model':model,'score':val.get('score'),
            'needs_context':val.get('needs_context',False),
            'source_uncertain':val.get('source_uncertain',False),
            'errors':raw_errors,'screened_issue_count':len(raw_errors)-len(filtered),
            'score_warnings':val.get('score_warnings',[]),
            'score_explanation':val.get('score_explanation','')})
    median=statistics.median([s['score'] for s in done.values()]) if done else None
    han_left=chinese_leftover(row['viet'],han_allowlist)
    score_conflicts=sum(bool(x.get('score_warnings')) for x in done.values())
    consensus=[]; matched=set(); consensus_indices=set()
    for i,(model,e) in enumerate(serious):
        peers=[(m2,e2,j) for j,(m2,e2) in enumerate(serious)
               if j!=i and m2!=model and same_issue(e,e2)]
        agree_models={model}|{m for m,_,_ in peers}
        if len(agree_models)<2:continue
        # Avoid duplicate claims; each consensus set is keyed by all member indexes.
        match_set=frozenset([i]+[j for _,_,j in peers])
        if match_set in matched:continue
        matched.add(match_set)
        consensus_indices.update(match_set)
        consensus.append({'category':e['category'],'severity':e['severity'],
            'source_span':e.get('source_span',''),'target_span':e.get('target_span',''),
            'suggested_translation':e.get('suggested_translation',''),
            'judge_models':sorted(agree_models),'judge_count':len(agree_models),
            'judge_reasons':[{'model':m,'reason':v.get('reason','')} for m,v in [(model,e)]+[(m,ep) for m,ep,_ in peers]]})
    minority=[{'judge':m,'risk':'high' if _high_risk_claim(row,e) else 'uncorroborated',
               'issue':e} for i,(m,e) in enumerate(serious) if i not in consensus_indices]
    high_risk=any(e['risk']=='high' for e in minority)
    context_votes=sum(bool(v.get('needs_context') or v.get('source_uncertain')) for v in done.values())
    threshold_votes=sum(v.get('score',0)>=score_threshold for v in done.values())
    status='PENDING'
    if not missing:
        if han_left:status='REVIEW_REQUIRED'
        elif context_votes>=2:status='CONTEXT_REVIEW'
        elif any(c['severity']=='critical' for c in consensus):status='REVIEW_REQUIRED'
        elif consensus:status='REPAIR_REQUIRED'
        elif high_risk:status='REVIEW_REQUIRED'
        elif threshold_votes>=2 and len(clean_models)>=2:
            status='ACCEPTED_WITH_NOTES' if (minority or screened or
                 any(v.get('errors') for v in done.values()) or context_votes or score_conflicts) else 'ACCEPTED_PROVISIONAL'
        else:
            status='REVIEW_REQUIRED'
    return {'status':status,'median_score':median,'judge_count':len(done),'judge_expected':len(model_names),
        'han_leftover_count':han_left,'score_conflict_votes':score_conflicts,
        'clean_judge_count':len(clean_models),'high_score_votes':threshold_votes,
        'raw_serious_claims':len(serious),'screened_claim_count':len(screened),
        'consensus_issue_count':len(consensus),'minority_high_risk_count':sum(x['risk']=='high' for x in minority),
        'triage_version':TRIAGE_VERSION,
        'consensus_issues_json':json.dumps(consensus,ensure_ascii=False),
        'review_issues_json':json.dumps(minority,ensure_ascii=False),
        'screened_issues_json':json.dumps(screened,ensure_ascii=False),
        'feedback_json':json.dumps(feedback,ensure_ascii=False),
        'individual_scores_json':json.dumps({m:v['score'] for m,v in done.items()},ensure_ascii=False),
        'missing_models':','.join(missing)}


def arbitrate_two_judges(row: dict, base_model_names: list[str], scores_by_model: dict,
                           han_allowlist=(), score_threshold=95.0) -> dict:
    """Decision from two judges, without any change to model/prompt/checkpoint keys.

    If both independent judges agree on a preliminary accept or an actionable
    consensus issue, do not call the 14B arbiter. Disagreements and ambiguous
    scores require the arbiter. A deterministic Chinese-character gate needs no
    extra model: send that sentence to review. Context-uncertain decisions shared
    by both judges likewise need outside context, not a third opinion.
    """
    if len(base_model_names) != 2:
        raise ValueError('base_model_names phải có đúng hai Judge')
    if any(name not in scores_by_model for name in base_model_names):
        return {'needs_arbiter': False, 'pair_ready': False, 'reason': 'pending_base', 'pair_summary': None}
    summary=aggregate_judgments(row,base_model_names,scores_by_model,han_allowlist,score_threshold)
    status=summary['status']
    if summary['han_leftover_count']:
        reason='han_hard_gate'
    elif status == 'CONTEXT_REVIEW' and all(
            scores_by_model[m].get('needs_context') or scores_by_model[m].get('source_uncertain')
            for m in base_model_names):
        reason='both_need_context'
    elif status in ('ACCEPTED_PROVISIONAL','ACCEPTED_WITH_NOTES','REPAIR_REQUIRED'):
        reason='pair_agrees'
    else:
        reason='disagreement'
    return {'needs_arbiter': reason=='disagreement', 'pair_ready': True,
            'reason': reason, 'pair_summary':summary}
