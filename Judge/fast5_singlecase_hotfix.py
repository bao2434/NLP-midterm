"""FAST5: conservative recovery + raw-output diagnostics for single-case JSON.

Compatible with the FAST5 Deferred Retry + adaptive batch runner. Install after
importing yhct_judge_fast5 (Cell 4) and after creating OUTPUT_DIR (Cell 6).
The prompt, generation parameters and checkpoint signature are NOT changed.

Only unambiguously identifiable ONE-case results can be recovered. Any missing
ID or missing flag forces requires_manual_review=True, so never auto-PASSes.
Wrong explicit IDs, duplicate case results and ambiguous outputs are rejected.
"""
from __future__ import annotations
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

PATCH_VERSION = 'fast5-singlecase-hotfix-v2-2026-11-28'


def install(module=None, *, diagnostics_path=None):
    if module is None:
        import yhct_judge_fast5 as module
    if getattr(module, '_singlecase_hotfix_v2_installed', False):
        return PATCH_VERSION + ': already installed'
    for attr in ('_single_item', '_parse_one', '_json_array', '_normalize_local_id', '_messages'):
        if not hasattr(module, attr):
            raise RuntimeError(f'FAST5 incompatible: missing {attr}')
    original_parse = module._parse_one
    original_messages = module._messages
    dest = Path(diagnostics_path) if diagnostics_path else None
    context = {'unit_id': None, 'candidate_id': None}

    class Recovered(list):
        def __init__(self, values, notes):
            super().__init__(values)
            self.recovery_notes = list(notes)

    def _norm_id(x):
        try:
            return module._normalize_local_id(x)
        except (ValueError, TypeError):
            return None

    def _score(x):
        if type(x) in (int, float):
            return True
        if isinstance(x, str):
            return bool(re.fullmatch(r'(?:\d+(?:\.\d*)?|\.\d+)', x.strip()))
        return False

    def _flag(x):
        if type(x) is bool or (type(x) is int and x in (0, 1)):
            return True
        return isinstance(x, str) and x.strip().lower() in ('0','1','true','false')

    def _one(value):
        """Return (normalized case, recovery notes) or None. Never remap wrong ID."""
        notes = []
        if isinstance(value, dict):
            # Model sometimes emits {result:{...}} or {judgment:{...}}.
            if len(value) == 1:
                for wrapper in ('result', 'judgment', 'data'):
                    if wrapper in value:
                        inner = _one(value[wrapper])
                        if inner is not None:
                            payload, info = inner
                            return payload, ['single_case_wrapper_'+wrapper, *info]
            aliases = ('id', 'local_id', 'case_id', 'index')
            supplied = [value[k] for k in aliases if k in value]
            if supplied and any(_norm_id(v) != 1 for v in supplied):
                return None
            if not supplied:
                notes.append('single_case_id_missing_inferred_1_requires_review')
            if 'score' not in value or 'errors' not in value:
                return None
            if not _score(value['score']) or not isinstance(value['errors'], list):
                return None
            nc = value.get('needs_context', 1)
            su = value.get('source_uncertain', 1)
            if not _flag(nc) or not _flag(su):
                return None
            if 'needs_context' not in value or 'source_uncertain' not in value:
                notes.append('single_case_missing_flags_set_uncertain')
            return [1, value['errors'], value['score'], nc, su], notes
        if not isinstance(value, list):
            return None
        if len(value) == 2 and _norm_id(value[0]) == 1 and _score(value[1]):
            return list(value), notes
        if len(value) in (3,4,5) and _norm_id(value[0]) == 1 and isinstance(value[1], list) and _score(value[2]):
            if len(value) == 5:
                return (list(value), notes) if _flag(value[3]) and _flag(value[4]) else None
            # Never silently assume missing uncertainty flags are false.
            if len(value) == 4 and not _flag(value[3]):
                return None
            notes.append('single_case_missing_flags_set_uncertain')
            if len(value) == 3:
                return [1, value[1], value[2], 1, 1], notes
            return [1, value[1], value[2], value[3], 1], notes
        # No explicit ID, but an entire 4-field full verdict for exactly one case.
        if (len(value) == 4 and isinstance(value[0], list) and _score(value[1])
                and _flag(value[2]) and _flag(value[3])):
            return [1, value[0], value[1], value[2], value[3]], [
                'single_case_id_missing_inferred_1_requires_review']
        return None

    def _decoded(raw):
        txt = (raw or '').strip()
        if txt.startswith('```'):
            txt = re.sub(r'^```(?:json)?\s*', '', txt, flags=re.I)
            if '```' in txt:
                txt = txt.split('```', 1)[0].strip()
        try:
            return json.loads(txt)
        except (ValueError, TypeError):
            # FAST5 can salvage complete items of truncated outer JSON arrays.
            return module._json_array(raw)

    def _diagnose(raw, message, parsed=None):
        if dest is None:
            return
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            raw = str(raw or '')
            item = {
                'at_utc': datetime.now(timezone.utc).isoformat(),
                'unit_id': context['unit_id'],
                'candidate_id': context['candidate_id'],
                'error': str(message)[:400],
                'raw_length': len(raw),
                'raw_sha256': hashlib.sha256(raw.encode('utf-8')).hexdigest(),
                'raw_preview': raw[:1600],
                'json_shape': (type(parsed).__name__ if parsed is not None else None),
            }
            with dest.open('a', encoding='utf-8') as f:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        except OSError:
            pass  # Diagnostics must not block real checkpointing.

    def _single_item_safe(raw):
        try:
            data = _decoded(raw)
        except (ValueError, TypeError) as exc:
            _diagnose(raw, f'Unparseable JSON: {type(exc).__name__}: {exc}')
            raise ValueError('Single-case output is not valid/salvageable JSON') from exc

        # dict containing array vs one dict: allow only an unambiguous output.
        if isinstance(data, dict) and isinstance(data.get('results'), list):
            data = data['results']
        direct = _one(data)
        if direct is not None:
            values, notes = direct
            if notes:
                return Recovered(values, notes)
            return values
        if not isinstance(data, list):
            data = [data]
        candidates = [x for x in (_one(item) for item in data) if x is not None]
        # Multiple matching items with ID=1 are *ambiguous*: always reject.
        if len(candidates) == 1:
            values, notes = candidates[0]
            if len(data) != 1:
                notes = [*notes, 'single_case_extra_results_dropped_for_review']
            # Wrap to force REVIEW if anything was omitted or extra results exist.
            return Recovered(values, notes) if notes else values
        message = (f'Single-case output: expected exactly one result with local ID=1; '
                   f'got array length={len(data)}, matching_items={len(candidates)}')
        _diagnose(raw, message, data)
        raise ValueError(message)

    def _parse_one_safe(item, row, expected_id):
        parsed = original_parse(item, row, expected_id)
        if isinstance(item, Recovered):
            notes = item.recovery_notes
            parsed.setdefault('format_fixes', []).extend(notes)
            if notes:
                # IMPORTANT: when the answer was not exactly conforming, keep
                # it as an explicit manual review, not a silently clean PASS.
                parsed['requires_manual_review'] = True
                parsed.setdefault('unverified_errors', []).append({
                    'validation_error': 'nonstandard_singlecase_output',
                    'recovery_notes': notes,
                    'source_span': '', 'target_span': '',
                })
        return parsed

    def _messages_safe(rows, lang):
        if len(rows) == 1:
            context['unit_id'] = rows[0].get('unit_id')
            context['candidate_id'] = rows[0].get('candidate_id')
        else:
            context['unit_id'] = None
            context['candidate_id'] = None
        return original_messages(rows, lang)

    module._single_item = _single_item_safe
    module._parse_one = _parse_one_safe
    module._messages = _messages_safe
    module._singlecase_hotfix_v2_installed = True
    return f'{PATCH_VERSION}: installed; prompt/signature unchanged; diagnostics={dest}'
