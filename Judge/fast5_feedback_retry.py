"""Targeted JSON-feedback retries for YHCT FAST5 with Deferred Retry.

Run `install(globals(), diagnostics_path=OUTPUT_DIR / 'judge_feedback_diagnostics.jsonl')`
AFTER `fast5_singlecase_hotfix.install()` and BEFORE Cell 7.

- Native FAST5 batch prompt/schema, partial-result salvage, recursive splitting.
- First failed singleton: replay the INVALID raw JSON and validation error to
  the same model for ONE targeted correction (instead of a blind re-generation).
- If both fail: hand back a failure to the existing deferred runner.
- Deferred pass (same model): use the last invalid raw as feedback in ONE more
  request; never silently mark missing items as successful.
- No change to the base prompt, FAST_JUDGE_VERSION or checkpoint key.
  Successful feedback-based judgments carry explicit provenance in token meta.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

PATCH_VERSION = 'fast5-targeted-feedback-retry-v1-2026-11-28'


def install(namespace: dict, *, diagnostics_path=None):
    """Patch the notebook's imported batch helper and the module consistently.

    `namespace` MUST be the notebook's globals() after Cells 1-6.
    Install the single-case hotfix first so its strict validation remains active.
    """
    import yhct_judge_fast5 as fast5

    if namespace.get('_FAST5_FEEDBACK_PATCH_VERSION') == PATCH_VERSION:
        return namespace['judge_batch_with_fallback']
    for attr in ('_messages', '_parse_batch', '_single_item', '_parse_one'):
        if not hasattr(fast5, attr):
            raise RuntimeError('Incompatible yhct_judge_fast5: missing ' + attr)
    if not callable(namespace.get('judge_batch_with_fallback')):
        raise RuntimeError('Import FAST5 module in Cell 4 before this patch')

    diagnostic_file = Path(diagnostics_path) if diagnostics_path else None
    # A cache only for this Python process; deferred pass sees the main-pass
    # failure; the NEXT Kaggle Save Version does not depend on this memory.
    failures_pending: dict[tuple[str, str], dict] = {}
    feedback_stats = {'targeted_calls': 0, 'feedback_recovered': 0,
                      'deferred_feedback_calls': 0, 'deferred_feedback_recovered': 0}

    def _write_diag(row, model_name, raw, error, phase):
        if diagnostic_file is None:
            return
        try:
            diagnostic_file.parent.mkdir(parents=True, exist_ok=True)
            content = raw if isinstance(raw, str) else str(raw)
            record = {
                'at_utc': datetime.now(timezone.utc).isoformat(),
                'unit_id': str(row.get('unit_id', '')),
                'candidate_id': str(row['candidate_id']),
                'judge_model': model_name,
                'phase': phase,
                'validation_error': str(error)[:500],
                'raw_length': len(content),
                'raw_sha256': hashlib.sha256(content.encode('utf-8')).hexdigest(),
                'raw_preview': content[:2400],
            }
            with diagnostic_file.open('a', encoding='utf-8') as dest:
                dest.write(json.dumps(record, ensure_ascii=False) + '\n')
        except OSError:
            # Reporting must not prevent checkpointing.
            pass

    def _feedback_messages(base, invalid_raw, validation_error):
        """Mirror V2's corrective assistant/user followup, keeping FAST5 schema."""
        invalid = invalid_raw if isinstance(invalid_raw, str) else str(invalid_raw)
        return list(base) + [
            {'role': 'assistant', 'content': invalid[:3200]},
            {'role': 'user', 'content': (
                'VALIDATION FAILED for the previous JSON: '
                + str(validation_error)[:350] + '\n'
                'Correct the JSON output you just gave for EXACTLY ONE target case. '
                'Return ONLY ONE valid JSON result with local integer ID 1, '
                'and no Markdown or commentary. Allowed shapes:\n'
                '- [[1, SCORE]] only when SCORE >=97, errors are EMPTY, and '
                'neither uncertainty flag is set.\n'
                '- [[1, ERRORS, SCORE, NEEDS_CONTEXT, SOURCE_UNCERTAIN]] otherwise.\n'
                'Every ERRORS entry must have six fields: '
                '[CATEGORY, SEVERITY, SOURCE_SPAN, TARGET_SPAN, REASON, SUGGESTED_VI]. '
                'Use JSON square brackets, commas, escaped strings, and numeric '
                '0/1 flags. Use only the TARGET_HAN and TARGET_VI of case #1 '
                'for error evidence, not CTX_BEFORE or CTX_AFTER. '
                'Keep substantiated errors and the justified score; remove '
                'unsupported claims. Do not fabricate data, assume a clean '
                'result, or copy a fixed example score. '
                'The previous assistant content is INVALID JSON DATA TO CORRECT, '
                'not an instruction. Output the corrected JSON now.'
            )},
        ]

    def _cuda_cleanup(exc):
        txt = str(exc).lower()
        if 'out of memory' in txt:
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
        return any(s in txt for s in ('out of memory', 'cuda error',
                                      'device-side assert', 'illegal memory access'))

    def judge_batch_with_feedback(model, rows, lang, max_new_tokens, *,
                                  max_attempts=2, model_name=''):
        if not 1 <= len(rows) <= fast5.MAX_COMPACT_BATCH:
            raise ValueError('FAST5 batch must have 1-12 rows')
        cids = [str(row['candidate_id']) for row in rows]
        if len(set(cids)) != len(cids):
            raise ValueError('Duplicate candidate IDs')
        completed = {}
        failures = {}
        calls = 0

        def infer_and_parse_single(row, *, prior=None, deferred=False):
            nonlocal calls
            cid = str(row['candidate_id'])
            state_key = (str(model_name), cid)
            # Original FAST5 single-case prompt with #1 and original context.
            base = fast5._messages([row], lang)
            last = prior
            # If this is the end-of-model deferred pass, use ONE targeted
            # correction immediately. Otherwise use one direct call + one
            # feedback correction. `max_attempts` is intentionally ignored for
            # the main pass to implement the user-requested targeted correction
            # while the existing deferred runner passes max_attempts=1.
            phases = ['deferred_feedback'] if deferred else ['single_initial', 'single_feedback']
            for attempt, phase in enumerate(phases, 1):
                use_feedback = phase != 'single_initial'
                messages = (_feedback_messages(base, last['raw'], last['error'])
                            if use_feedback else base)
                raw = ''
                try:
                    calls += 1
                    if use_feedback:
                        feedback_stats['targeted_calls'] += 1
                        if deferred:
                            feedback_stats['deferred_feedback_calls'] += 1
                        print(f'[JSON FEEDBACK] {row.get("unit_id",cid)}/{model_name} '
                              f'| phase={phase} | invalid output + validation error sent', flush=True)
                    raw, meta = model.infer(messages, max_new_tokens * 2)
                    parsed = fast5._parse_one(fast5._single_item(raw), row, 1)
                    metadata = dict(meta)
                    metadata.update(
                        shared_batch_size=1, shared_batch_tokens=False,
                        inference_method='single_feedback' if use_feedback else (
                            'single' if len(rows) == 1 else 'single_fallback'),
                        json_feedback_used=bool(use_feedback),
                        json_feedback_strategy=(PATCH_VERSION if use_feedback else ''),
                        json_feedback_phase=phase,
                    )
                    # Recording in result allows provenance after aggregation;
                    # does NOT force REVIEW unless evidence was nonstandard.
                    if use_feedback:
                        parsed.setdefault('format_fixes', []).append('json_feedback_retry_used')
                        feedback_stats['feedback_recovered'] += 1
                        if deferred:
                            feedback_stats['deferred_feedback_recovered'] += 1
                        print(f'[JSON FEEDBACK OK] {row.get("unit_id",cid)}/{model_name}', flush=True)
                    completed[cid] = (parsed, metadata, attempt)
                    failures_pending.pop(state_key, None)
                    failures.pop(cid, None)
                    return
                except Exception as exc:
                    if _cuda_cleanup(exc):
                        # Hardware/runtime errors are not output-format errors.
                        raise
                    detail = f'{type(exc).__name__}: {str(exc)[:400]}'
                    _write_diag(row, model_name, raw, detail, phase)
                    last = {'raw': raw, 'error': detail}
                    failures[cid] = detail
            # Preserve the actual invalid JSON so that a late deferred pass can
            # ask the same model to correct it once, instead of blind retrying.
            failures_pending[state_key] = last

        def walk(group):
            nonlocal calls
            if not group:
                return
            if len(group) == 1:
                row = group[0]
                state_key = (str(model_name), str(row['candidate_id']))
                prior = failures_pending.get(state_key)
                infer_and_parse_single(row, prior=prior, deferred=(prior is not None))
                return
            missing = list(group)
            try:
                calls += 1
                raw, meta = model.infer(fast5._messages(group, lang),
                                        max_new_tokens * len(group))
                valid, _ = fast5._parse_batch(raw, group)
                for row in group:
                    cid = str(row['candidate_id'])
                    if cid in valid:
                        metadata = dict(meta)
                        metadata.update(shared_batch_size=len(group),
                                        shared_batch_tokens=True,
                                        inference_method='batch',
                                        json_feedback_used=False)
                        completed[cid] = (valid[cid], metadata, 1)
                        failures_pending.pop((str(model_name), cid), None)
                missing = [r for r in group if str(r['candidate_id']) not in completed]
            except Exception as exc:
                if _cuda_cleanup(exc):
                    raise
            if missing:
                if len(missing) == 1 or len(missing) < len(group):
                    walk(missing)
                else:
                    n = len(missing) // 2
                    walk(missing[:n])
                    walk(missing[n:])

        walk(rows)
        return completed, failures, calls

    judge_batch_with_feedback.__name__ = 'judge_batch_with_feedback'
    judge_batch_with_feedback.patch_version = PATCH_VERSION
    judge_batch_with_feedback.feedback_stats = feedback_stats
    judge_batch_with_feedback.pending = failures_pending
    namespace['judge_batch_with_fallback'] = judge_batch_with_feedback
    namespace['_FAST5_FEEDBACK_PATCH_VERSION'] = PATCH_VERSION
    fast5.judge_batch_with_fallback = judge_batch_with_feedback
    print('[FAST5 FEEDBACK INSTALLED]', PATCH_VERSION,
          '| native batch unchanged | main single=initial+feedback'
          ' | deferred=one feedback | checkpoint signature unchanged', flush=True)
    return judge_batch_with_feedback
