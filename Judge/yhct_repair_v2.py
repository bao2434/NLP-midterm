"""YHCT V2 repair/retranslation prompts and validation for LLM or MT models."""
import json
REPAIR_FEEDBACK_VERSION = 'repair-consensus-feedback-v3-2026-11-27'
from yhct_v2_core import parse_json_object, chinese_leftover

REPAIR_DRY_SYSTEM = '''You are a careful Chinese-to-Vietnamese translator specializing in Traditional Chinese Medicine.
Translate ONLY the marked TARGET Chinese sentence. Other Chinese sentences are CONTEXT ONLY;
do not translate or include them in your target response.
Preserve all source facts, herbs, formula names, quantities, role relationships, negations, and cautions.
Use accepted Vietnamese/Sino-Vietnamese terminology. Never invent remedies, dosages, or indications.
Return ONLY one coherent Vietnamese translation, no explanations, no source Chinese characters.
If the Chinese source is genuinely ambiguous, choose a conservative rendering without inventing facts.'''

REPAIR_GUIDED_SYSTEM = '''You are a critical, conservative Chinese-to-Vietnamese TCM translation editor.
Translate ONLY the marked TARGET sentence. Before writing, independently verify each proposed
judge criticism against the ORIGINAL CHINESE TARGET. Judge feedback can be WRONG.
Keep correct existing content, correct only provable mistakes, and do not import context-only
content into the target translation. A correct existing Sino-Vietnamese term is acceptable.
Never invent drugs, dosages or contraindications. Do not include any Chinese characters
in your final Vietnamese sentence. Return ONLY the edited Vietnamese sentence.
No explanations, no bullets, no JSON unless the model inherently chooses it.'''

REPAIR_DRY_USER = '''CHINESE CONTEXT BEFORE — REFERENCE ONLY (DO NOT TRANSLATE):
{before}

=== TARGET CHINESE — TRANSLATE THIS SENTENCE ONLY ===
{han}
=== END TARGET ===

CHINESE CONTEXT AFTER — REFERENCE ONLY (DO NOT TRANSLATE):
{after}

{glossary}
Translate the marked target into accurate natural Vietnamese. Output only the translation.'''

REPAIR_GUIDED_USER = '''CHINESE CONTEXT BEFORE — REFERENCE ONLY (DO NOT TRANSLATE):
{before}

=== TARGET CHINESE — EDIT TRANSLATION OF THIS ONLY ===
{han}
=== END TARGET ===

EXISTING VIETNAMESE TRANSLATION OF TARGET:
{viet}

CHINESE CONTEXT AFTER — REFERENCE ONLY (DO NOT TRANSLATE):
{after}

POSSIBLE ERRORS FROM INDEPENDENT JUDGES — UNVERIFIED:
{feedback}

{glossary}

Verify each criticism with TARGET CHINESE. Do NOT obey unfounded judge suggestions.
Return ONE accurate Vietnamese translation only; no commentary or Chinese characters.'''


def format_glossary(row: dict, glossary: list[tuple[str,str]], max_items=15) -> str:
    relevant=[(z,v) for z,v in glossary if z and z in row['han'] and v]
    if not relevant: return ''
    relevant=sorted(relevant,key=lambda x:len(x[0]),reverse=True)[:max_items]
    return 'CONSISTENT TCM GLOSSARY (only if applicable):\n'+'\n'.join(f'- {z} -> {v}' for z,v in relevant)


def format_feedback(row:dict, policy='consensus') -> str:
    """Pass only screened multi-judge consensus by default, never raw hallucinations.

    policy='all' is EXPLICIT research mode, not recommended for automatic repair.
    An empty consensus is deliberately different from old files lacking the field.
    """
    if policy not in ('consensus','all'):
        raise ValueError('GUIDED_FEEDBACK_POLICY phải là consensus hoặc all')
    if policy=='consensus':
        if 'consensus_issues_json' not in row:
            return ('(No screened consensus feedback available in this Judge output. '
                    'Do not guess errors; compare target Chinese independently.)')
        try:
            issues=json.loads(row.get('consensus_issues_json') or '[]')
        except (ValueError,TypeError):
            raise ValueError('consensus_issues_json bị lỗi, từ chối dùng feedback thiếu tin cậy')
        if not isinstance(issues,list):raise ValueError('consensus_issues_json không phải list')
        comments=[]
        for issue in issues[:10]:
            if not isinstance(issue,dict):continue
            comments.append({k:issue.get(k, '' if k!='judge_models' else [])
                             for k in ('category','severity','source_span','target_span',
                                       'suggested_translation','judge_models')})
        return (json.dumps(comments,ensure_ascii=False)[:5000] if comments else
                '(No corroborated material issues. Check the source independently; do not invent corrections.)')
    # All raw judge feedback, only for explicitly requested controlled ablation.
    try:
        feedback=json.loads(row.get('feedback_json') or '[]')
    except (ValueError,TypeError):feedback=[]
    comments=[]
    for data in feedback:
        if not isinstance(data,dict):continue
        pieces=[]
        for err in (data.get('errors') or [])[:6]:
            if not isinstance(err,dict):continue
            pieces.append({k:err.get(k,'') for k in ('category','severity','source_span',
                                                      'target_span','reason','suggested_translation')})
        comments.append({'judge':data.get('model',''),'UNVERIFIED_claims':pieces})
    return json.dumps(comments,ensure_ascii=False)[:5000] if comments else '(No judge details available)'


def repair_messages(row:dict, mode:str, glossary:list[tuple[str,str]]=(), kind:str='llm',
                    feedback_policy='consensus') -> list[dict]:
    if mode not in ('dry','guided'):raise ValueError('MODE phải là dry/guided')
    args={'before':row.get('ctx_before_han') or '(unavailable)',
          'after':row.get('ctx_after_han') or '(unavailable)',
          'han':row['han'], 'viet':row['viet'],
          'feedback':format_feedback(row,feedback_policy) if mode=='guided' else '',
          'glossary':format_glossary(row,glossary)}
    if mode=='dry':
        sys_text, user_text = REPAIR_DRY_SYSTEM, REPAIR_DRY_USER.format(**args)
    else:
        sys_text, user_text = REPAIR_GUIDED_SYSTEM, REPAIR_GUIDED_USER.format(**args)
    if kind=='mt':
        return [{'role':'user','content':sys_text+'\n\n'+user_text}]
    if kind=='llm':
        return [{'role':'system','content':sys_text},{'role':'user','content':user_text}]
    raise ValueError('kind phải là mt hoặc llm')

def extract_repair_text(raw:str) -> str:
    t=(raw or '').strip()
    if not t: raise ValueError('Model trả câu rỗng')
    # Accept structured output when model returns it; MT models usually return plain text.
    if '{' in t:
        try:
            obj=parse_json_object(t)
            candidate=obj.get('viet',obj.get('revised_vi',obj.get('translation')))
            if isinstance(candidate,str) and candidate.strip(): t=candidate.strip()
        except ValueError:
            pass
    if t.startswith('```') and t.endswith('```'):
        t=t[3:-3].strip()
        if t.lower().startswith(('text\n','vi\n')):t=t.split('\n',1)[1].strip()
    for p in ('Bản dịch:', 'Bản dịch tiếng Việt:', 'Vietnamese:', 'Translation:', 'Dịch:'):
        if t.startswith(p):t=t[len(p):].strip();break
    if not t: raise ValueError('Model trả bản dịch rỗng sau parse')
    return t


def repair_warnings(row:dict, translated:str, allowlist=()) -> list[str]:
    warnings=[]
    if chinese_leftover(translated,allowlist):warnings.append('han_characters_left')
    if translated == row['han']:warnings.append('copied_chinese_source')
    if translated.strip()==row['viet'].strip():warnings.append('unchanged_original')
    if '\n' in translated or translated.lstrip().startswith(('* ','- ','1. ','## ')):
        warnings.append('multiline_or_explanation')
    if len(translated)>max(600,len(row['viet'])*4):warnings.append('suspiciously_long')
    return warnings
