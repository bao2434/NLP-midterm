"""Deferred retry runner for the existing YHCT FAST5 Kaggle notebook.

Replace ONLY notebook Cell 7 with the short integration cell shipped alongside
this module. Cells 1–6 and 8 are kept. This module does not alter prompts,
signatures, row IDs, model configurations or voting.

Main pass: execute each model's outstanding tasks, allowing the FAST5 batch
parser to salvage independently valid items. Every remaining unsuccessful
sentence is QUEUED, never counted as a completed judgment. End-of-model pass:
try each queued sentence only ONCE, then continue to the next model even when
some attempts fail. Persist remaining tasks for the next Kaggle Save Version.
"""
from __future__ import annotations

from collections import Counter
import time
from pathlib import Path

DEFERRED_RETRY_RUNNER_VERSION = 'fast5-deferred-retry-adaptive-batch-v2-2026-11-28'

# Hard cap requested by the user: not even a mistakenly configured model may
# run batches larger than 8. Each model still starts at its own configured size
# if that size is smaller than 8 (e.g., Qwen3-14B defaults to 6).
HARD_MAX_BATCH_SIZE = 8
DEFAULT_GROW_AFTER_CLEAN_BATCHES = 3


def _runtime(g):
    """Initialize names required by the unmodified report/export Cell 8."""
    models = g['JUDGE_MODELS']
    if len(models) != 3 or len({m['name'] for m in models}) != 3:
        raise ValueError('FAST5 tiebreak runner requires three distinct Judge models')
    base = [m['name'] for m in models[:2]]
    arbiter = models[2]['name']
    keys = {(r['candidate_id'], m['name']): key for m, r, key in g['TASKS']}
    if len(keys) != len(g['TASKS']):
        raise ValueError('Duplicate TASKS candidate/model keys')
    completed = g['completed']

    def get_judgments_for_row(row):
        return {m['name']: completed[keys[(row['candidate_id'], m['name'])]]['result']
                for m in models if keys[(row['candidate_id'], m['name'])] in completed}

    def arbiter_required(row):
        votes = get_judgments_for_row(row)
        return g['arbitrate_two_judges'](
            row, base, votes, g['UNTRANSLATED_HAN_ALLOWLIST'],
            g['JUDGE_SCORE_THRESHOLD'])['needs_arbiter']

    def planned_keys():
        # The arbiter is required only if both base votes exist and disagree.
        return [key for m, row, key in g['TASKS']
                if m['name'] in base or g['JUDGE_EXECUTION_MODE'] == 'all3'
                or arbiter_required(row)]

    g.update(BASE_JUDGES=base, ARBITER_NAME=arbiter,
             keys_by_candidate=keys, get_judgments_for_row=get_judgments_for_row,
             arbiter_required=arbiter_required, planned_keys=planned_keys)
    return base, arbiter, planned_keys, arbiter_required


def run_judge_deferred(g: dict, model_factory=None) -> dict:
    """Run the whole FAST5 pipeline with one deferred retry per failed row.

    Args:
        g: notebook globals() after Cells 1–6; this function installs exactly
           the runtime names used by the existing Cell 8.
        model_factory: test-only injectable class compatible with CausalModel.
    """
    base, arbiter_name, planned_keys, arbiter_required = _runtime(g)
    if model_factory is None:
        model_factory = g['CausalModel']

    done = g['completed']
    tasks_all = g['TASKS']
    output_dir = Path(g['OUTPUT_DIR'])
    append_jsonl = g['append_jsonl']
    dump_tsv = g['dump_tsv']
    utc_now = g['utc_now']
    model_names = [m['name'] for m in g['JUDGE_MODELS']]
    capped_sizes = {}
    for name in model_names:
        requested = g['EFFECTIVE_JUDGE_BATCH_SIZES'][name]
        if type(requested) is not int or requested < 1:
            raise ValueError(f'Invalid configured batch size {name}={requested!r}')
        capped_sizes[name] = min(requested, HARD_MAX_BATCH_SIZE)
    grow_after = g.get('BATCH_GROW_AFTER_CLEAN', DEFAULT_GROW_AFTER_CLEAN_BATCHES)
    if type(grow_after) is not int or grow_after < 2:
        raise ValueError('BATCH_GROW_AFTER_CLEAN must be an integer >= 2')
    backoff_after = g['BATCH_BACKOFF_AFTER']
    if type(backoff_after) is not int or backoff_after < 1:
        raise ValueError('BATCH_BACKOFF_AFTER must be an integer >= 1')
    initial_missing = sum(key not in done for key in planned_keys())
    if g.get('CHECKPOINT_AUDIT_ONLY'):
        result = {
            'runner_version': DEFERRED_RETRY_RUNNER_VERSION,
            'execution_mode': g['JUDGE_EXECUTION_MODE'],
            'new_successes': 0, 'completed_all_model_checkpoints': len(done),
            'expected_required_now': len(planned_keys()),
            'remaining_required_now': initial_missing,
            'failed_tasks_this_run': 0,
            'batch_size_by_model': capped_sizes,
            'batch_hard_cap': HARD_MAX_BATCH_SIZE,
            'grow_after_clean_batches': grow_after,
            'new_batch_judgments': 0, 'new_single_judgments': 0,
            'model_calls_lower_bound': 0, 'model_load_seconds': 0.0,
            'inference_seconds': 0.0, 'wall_seconds': 0.0,
            'seconds_per_successful_judgment': None,
            'per_model_performance': {}, 'status': 'audit_only',
            'stop_reason': 'checkpoint_audit_only', 'errors': [],
            'main_pass_failures': 0, 'deferred_retries_attempted': 0,
            'deferred_retries_recovered': 0, 'pending_from_this_run': 0,
            'pending_next_run_by_model': {}, 'pending_next_run_file': None,
        }
        print('[AUDIT ONLY] Deferred Retry installed. NO model loaded.', flush=True)
        return result

    run_started = time.monotonic()
    deadline = (run_started + float(g['RUN_HOURS']) * 3600) if g.get('RUN_HOURS') else None
    max_new = g.get('MAX_NEW_INFERENCES')
    # Deliberately one single-case attempt in EACH phase (main and deferred).
    # The existing FAST5 helper can recursively split a malformed batch.
    single_attempts_main = 1
    single_attempts_deferred = 1

    successes = batch_results = single_results = 0
    model_calls = 0
    total_inference_seconds = total_load_seconds = 0.0
    stop_reason = None
    errors = []
    per_model = {}
    first_failure_keys = set()
    retry_attempted_keys = set()
    retry_recovered_keys = set()
    last_error_by_key = {}
    deferred_key_status = {}

    def limit_reason():
        if deadline is not None and time.monotonic() >= deadline:
            return 'deadline'
        if max_new is not None and successes >= max_new:
            return 'operation_limit'
        return None

    def fatal_error(exc):
        message = str(exc).lower()
        return isinstance(exc, (KeyboardInterrupt, SystemExit)) or any(
            x in message for x in ('out of memory', 'cuda error', 'device-side assert',
                                  'illegal memory access', 'cuda out of memory'))

    def save_success(row, key, model_name, payload, *, phase):
        nonlocal successes, batch_results, single_results
        if key in done:
            return False  # imported checkpoint must never be overwritten
        result, meta, attempts_used = payload
        meta = dict(meta)
        record = {
            'schema': g['VERSION'], 'stage': g['MODE'],
            'signature': g['SIGNATURE'], 'key': key,
            'candidate_id': row['candidate_id'], 'unit_id': row['unit_id'],
            'judge_model': model_name, 'ok': True, 'result': result,
            'tokens': meta, 'attempts_used': attempts_used,
            'batch_size': meta.get('shared_batch_size', 1),
            'inference_method': meta.get('inference_method', 'single'),
            'completed_utc': utc_now(), 'run_phase': phase,
        }
        append_jsonl(output_dir / 'judge_checkpoint.jsonl', record)
        done[key] = record
        successes += 1
        if meta.get('inference_method') == 'batch':
            batch_results += 1
        else:
            single_results += 1
        if successes % 20 == 0:
            print('[CHECKPOINT]', successes, 'new judgments', flush=True)
        last_error_by_key.pop(key, None)
        deferred_key_status.pop(key, None)
        return True

    def save_failure(row, key, model_name, message, *, phase):
        desc = f'{row["unit_id"]}/{model_name}: {message}'
        last_error_by_key[key] = desc
        errors.append(desc)
        status = 'queued_for_end_of_model' if phase == 'main' else 'failed_after_deferred_retry'
        deferred_key_status[key] = status
        append_jsonl(output_dir / 'judge_errors.jsonl', {
            'at': utc_now(), 'stage': g['MODE'], 'signature': g['SIGNATURE'],
            'candidate_id': row['candidate_id'], 'unit_id': row['unit_id'],
            'judge_model': model_name, 'key': key, 'phase': phase,
            'status': status, 'error': desc,
            'retry_in_next_run': phase == 'deferred_retry',
        })
        label = '[DEFER]' if phase == 'main' else '[NEXT RUN]'
        print(f'{label} {desc}', flush=True)

    for cfg in g['JUDGE_MODELS']:
        name = cfg['name']
        outstanding = [(r, key) for m, r, key in tasks_all
                       if m['name'] == name and key not in done and
                       (name in base or g['JUDGE_EXECUTION_MODE'] == 'all3'
                        or arbiter_required(r))]
        if not outstanding:
            print(f'[MODEL SKIP] {name}: no required missing checkpoints', flush=True)
            continue
        stop_reason = limit_reason()
        if stop_reason:
            break

        requested_size = g['EFFECTIVE_JUDGE_BATCH_SIZES'][name]
        configured_size = capped_sizes[name]
        current_size = configured_size
        fallback_streak = fallback_groups = backoffs = 0
        clean_batch_streak = 0
        batch_growths = 0
        min_batch_seen = current_size
        batch_size_transitions = []
        if requested_size > HARD_MAX_BATCH_SIZE:
            print(f'[BATCH CAP] {name}: requested={requested_size} -> '
                  f'initial={configured_size}; HARD MAX={HARD_MAX_BATCH_SIZE}', flush=True)
        model = model_factory(cfg, g['CACHE'], g['MAX_INPUT_TOKENS'])
        load_at = time.monotonic()
        print(f'LOAD: {name} | tasks: {len(outstanding)} | batch: {current_size}', flush=True)
        try:
            model.load()
            total_load_seconds += time.monotonic() - load_at
        except Exception as exc:
            message = f'{name} MODEL_LOAD {type(exc).__name__}: {str(exc)[:300]}'
            append_jsonl(output_dir / 'judge_errors.jsonl',
                         {'at': utc_now(), 'phase': 'load', 'judge_model': name, 'error': message})
            print('[MODEL LOAD ERROR]', message, flush=True)
            errors.append(message)
            try:
                model.unload()
            except Exception:
                pass
            if fatal_error(exc):
                stop_reason = 'fatal_model_load_error'
                break
            # A model that cannot be loaded must not prevent the other model from running.
            per_model[name] = {'new_successes': 0, 'model_load_failed': True,
                               'tasks_skipped_for_next_run': len(outstanding)}
            continue

        per_seconds = 0.0
        per_calls = 0
        per_successes = 0
        queued = {}  # insertion order; keyed by the checkpoint key
        retry_attempted = 0
        retry_recovered = 0
        offset = 0
        try:
            # Main pass: don't get stuck on any single problematic case.
            while offset < len(outstanding):
                stop_reason = limit_reason()
                if stop_reason:
                    break
                size = current_size
                if max_new is not None:
                    size = min(size, max_new - successes)
                if size < 1:
                    stop_reason = 'operation_limit'
                    break
                group = outstanding[offset:offset + size]
                offset += len(group)
                started = time.monotonic()
                try:
                    batch_ok, batch_bad, calls = g['judge_batch_with_fallback'](
                        model, [r for r, _ in group], g['JUDGE_PROMPT_LANGUAGE'],
                        g['MAX_NEW_TOKENS'], max_attempts=single_attempts_main,
                        model_name=name)
                except Exception as exc:
                    if fatal_error(exc):
                        stop_reason = 'fatal_inference_error'
                        print('[FATAL MODEL ERROR]', name, type(exc).__name__, str(exc)[:250], flush=True)
                        # Schedule all unprocessed rows for next session. No false successes.
                        batch_ok, batch_bad, calls = {}, {}, 0
                        for row, key in group:
                            batch_bad[str(row['candidate_id'])] = f'{type(exc).__name__}: {str(exc)[:250]}'
                    else:
                        batch_ok, batch_bad, calls = {}, {}, 0
                        for row, key in group:
                            batch_bad[str(row['candidate_id'])] = f'{type(exc).__name__}: {str(exc)[:250]}'
                elapsed = time.monotonic() - started
                per_seconds += elapsed
                total_inference_seconds += elapsed
                per_calls += calls
                model_calls += calls

                # Adaptive batch AIMD-like policy:
                # - back off after repeated batches that needed fallback,
                # - grow only after consecutive FULL-SIZE clean native batches,
                # - never exceed min(model-configured initial batch, hard cap 8).
                # A partial LAST group is not evidence of stability or failure.
                if (len(group) == current_size and len(group) > 1
                        and stop_reason is None):
                    whole_batch_success = all(
                        (str(row['candidate_id']) in batch_ok and
                         batch_ok[str(row['candidate_id'])][1].get('shared_batch_size') == len(group) and
                         batch_ok[str(row['candidate_id'])][1].get('inference_method') == 'batch')
                        for row, _ in group)
                    if whole_batch_success:
                        fallback_streak = 0
                        clean_batch_streak += 1
                        if clean_batch_streak >= grow_after and current_size < configured_size:
                            old = current_size
                            current_size = min(configured_size, HARD_MAX_BATCH_SIZE, old * 2)
                            batch_growths += 1
                            batch_size_transitions.append({'event':'grow','from':old,'to':current_size})
                            clean_batch_streak = 0
                            print(f'[BATCH RECOVERY] {name}: {old}->{current_size} '
                                  f'after {grow_after} clean full batches '
                                  f'(cap={configured_size}, hard_max={HARD_MAX_BATCH_SIZE})', flush=True)
                    else:
                        clean_batch_streak = 0
                        fallback_groups += 1
                        fallback_streak += 1
                        if fallback_streak >= backoff_after and current_size > 2:
                            old = current_size
                            current_size = max(2, (current_size + 1) // 2)
                            backoffs += 1
                            batch_size_transitions.append({'event':'backoff','from':old,'to':current_size})
                            min_batch_seen = min(min_batch_seen, current_size)
                            fallback_streak = 0
                            print(f'[BATCH BACKOFF] {name}: {old}->{current_size}; '
                                  f'grow again after {grow_after} consecutive clean batches', flush=True)

                for row, key in group:
                    cid = str(row['candidate_id'])
                    if cid in batch_ok:
                        if save_success(row, key, name, batch_ok[cid], phase='main'):
                            per_successes += 1
                    else:
                        message = str(batch_bad.get(cid, 'Unknown batch failure'))
                        save_failure(row, key, name, message, phase='main')
                        queued[key] = row
                        first_failure_keys.add(key)
                if stop_reason:
                    break

            print(f'[DEFERRED RETRY] {name}: initial failures {len(queued)}; '
                  f'processed main {offset}/{len(outstanding)}; retry max 1 attempt/case', flush=True)
            # Deferred pass: one standalone attempt per queued sentence while
            # this exact model is still loaded; do not fail the entire pipeline.
            for key, row in queued.items():
                if key in done:
                    continue
                if stop_reason is None:
                    stop_reason = limit_reason()
                if stop_reason:
                    print(f'[DEFERRED RETRY SKIPPED] {name}: {stop_reason}', flush=True)
                    break
                started = time.monotonic()
                retry_attempted += 1
                retry_attempted_keys.add(key)
                cid = str(row['candidate_id'])
                try:
                    retry_ok, retry_bad, calls = g['judge_batch_with_fallback'](
                        model, [row], g['JUDGE_PROMPT_LANGUAGE'],
                        g['MAX_NEW_TOKENS'], max_attempts=single_attempts_deferred,
                        model_name=name)
                except Exception as exc:
                    retry_ok = {}
                    retry_bad = {cid: f'{type(exc).__name__}: {str(exc)[:250]}'}
                    calls = 0
                    if fatal_error(exc):
                        stop_reason = 'fatal_inference_error'
                elapsed = time.monotonic() - started
                per_seconds += elapsed
                total_inference_seconds += elapsed
                per_calls += calls
                model_calls += calls
                if cid in retry_ok:
                    if save_success(row, key, name, retry_ok[cid], phase='deferred_retry'):
                        per_successes += 1
                        retry_recovered += 1
                        retry_recovered_keys.add(key)
                        print(f'[RETRY RECOVERED] {row["unit_id"]}/{name}', flush=True)
                else:
                    save_failure(row, key, name,
                                 str(retry_bad.get(cid, 'Unknown deferred retry failure')),
                                 phase='deferred_retry')
                if stop_reason:
                    break
        finally:
            try:
                model.unload()
            except Exception as exc:
                print('[WARN] unload:', name, type(exc).__name__, str(exc)[:160], flush=True)

        still_queued = sum(key not in done for key in queued)
        per_model[name] = {
            'new_successes': per_successes,
            'configured_batch_size': requested_size,
            'capped_initial_batch_size': configured_size,
            'hard_max_batch_size': HARD_MAX_BATCH_SIZE,
            'final_batch_size': current_size,
            'min_batch_size_seen': min_batch_seen,
            'fallback_groups': fallback_groups,
            'batch_backoffs': backoffs,
            'batch_growths': batch_growths,
            'batch_size_transitions': batch_size_transitions,
            'grow_after_clean_batches': grow_after,
            'model_calls': per_calls,
            'inference_seconds': round(per_seconds, 2),
            'seconds_per_successful_judgment': round(per_seconds / per_successes, 2)
            if per_successes else None,
            'main_pass_failed_rows': len(queued),
            'deferred_retries_attempted': retry_attempted,
            'deferred_retries_recovered': retry_recovered,
            'still_missing_after_retry': still_queued,
        }
        print(f'[MODEL SUMMARY] {name}: success={per_successes} '
              f'deferred={len(queued)} retried={retry_attempted} '
              f'recovered={retry_recovered} next_run={still_queued} '
              f'| batch initial={configured_size} min={min_batch_seen} '
              f'final={current_size} backoffs={backoffs} growths={batch_growths}', flush=True)
        if stop_reason:
            break

    keys_required = set(planned_keys())
    pending = []
    for cfg, row, key in tasks_all:
        if key not in keys_required or key in done:
            continue
        phase = deferred_key_status.get(key, 'not_attempted_this_run')
        pending.append({
            'unit_id': row['unit_id'], 'candidate_id': row['candidate_id'],
            'judge_model': cfg['name'], 'status': phase,
            'last_error': last_error_by_key.get(key, ''), 'checkpoint_key': key,
        })
    pending_by_model = Counter(row['judge_model'] for row in pending)
    pending_by_status = Counter(row['status'] for row in pending)
    pending_file = output_dir / 'judge_pending_next_run.tsv'
    dump_tsv(pending_file, pending,
             ['unit_id', 'candidate_id', 'judge_model', 'status',
              'last_error', 'checkpoint_key'])
    remaining = len(pending)
    if remaining and stop_reason is None:
        stop_reason = 'deferred_errors_for_next_run' if any(
            r['status'] == 'failed_after_deferred_retry' for r in pending) else 'incomplete'

    print('========== FINAL JUDGE / NEXT SAVE VERSION ==========', flush=True)
    print(f'[FINAL] New successes: {successes}; total checkpoint restored+new: {len(done)}', flush=True)
    print(f'[FINAL] Deferred initially: {len(first_failure_keys)} '
          f'| retried: {len(retry_attempted_keys)} '
          f'| recovered: {len(retry_recovered_keys)} '
          f'| still failing: {len([r for r in pending if r["status"] == "failed_after_deferred_retry"])}',
          flush=True)
    print(f'[FINAL] Required pending: {remaining} | by model: {dict(pending_by_model)}', flush=True)
    print('[FINAL] Stop reason:', stop_reason or 'complete', flush=True)
    print('[FINAL] Pending TSV:', pending_file, flush=True)
    if remaining:
        print('[NEXT SAVE VERSION] Keep judge_manifest.json + judge_checkpoint.jsonl '
              'together in your Kaggle Input; unresolved rows will be retried '
              'automatically. judge_errors.jsonl is diagnostic only.', flush=True)
    print('=====================================================', flush=True)

    return {
        'runner_version': DEFERRED_RETRY_RUNNER_VERSION,
        'execution_mode': g['JUDGE_EXECUTION_MODE'],
        'new_successes': successes,
        'completed_all_model_checkpoints': len(done),
        'expected_required_now': len(keys_required),
        'remaining_required_now': remaining,
        'failed_tasks_this_run': sum(r['status'] == 'failed_after_deferred_retry' for r in pending),
        'batch_size_by_model': capped_sizes,
        'batch_hard_cap': HARD_MAX_BATCH_SIZE,
        'grow_after_clean_batches': grow_after,
        'new_batch_judgments': batch_results,
        'new_single_judgments': single_results,
        'model_calls_lower_bound': model_calls,
        'model_load_seconds': round(total_load_seconds, 2),
        'inference_seconds': round(total_inference_seconds, 2),
        'wall_seconds': round(time.monotonic() - run_started, 2),
        'seconds_per_successful_judgment': round(total_inference_seconds / successes, 2)
        if successes else None,
        'per_model_performance': per_model,
        'main_pass_failures': len(first_failure_keys),
        'deferred_retries_attempted': len(retry_attempted_keys),
        'deferred_retries_recovered': len(retry_recovered_keys),
        'pending_from_this_run': sum(key not in done for key in first_failure_keys),
        'pending_next_run_by_model': dict(pending_by_model),
        'pending_next_run_by_status': dict(pending_by_status),
        'pending_next_run_file': str(pending_file),
        'status': 'complete' if remaining == 0 else 'incomplete',
        'stop_reason': stop_reason,
        'errors': errors[-12:],
    }
