"""Step 2 & 3: Exact Levenshtein character alignment and column grouping (MSA)."""

from __future__ import annotations

from collections import Counter


# ---------------------------------------------------------------------------
# Block / page alignment (handles occasional missing or extra pages)
# ---------------------------------------------------------------------------

def _char_bigrams(text: str) -> Counter[str]:
    if len(text) < 2:
        return Counter(text)
    return Counter(text[i : i + 2] for i in range(len(text) - 1))


def _block_similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    bg_a = _char_bigrams(a)
    bg_b = _char_bigrams(b)
    inter = sum((bg_a & bg_b).values())
    union = sum((bg_a | bg_b).values())
    return inter / union if union else 0.0


def _align_two_block_lists(
    ref: list[str],
    other: list[str],
    gap_penalty: float = 0.08,
) -> list[tuple[int | None, int | None]]:
    """Needleman-Wunsch global alignment of two lists of normalized blocks."""
    n, m = len(ref), len(other)
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    bt = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = -i * gap_penalty
        bt[i][0] = 1
    for j in range(1, m + 1):
        dp[0][j] = -j * gap_penalty
        bt[0][j] = 2

    for i in range(1, n + 1):
        ref_i = ref[i - 1]
        dp_i = dp[i]
        dp_prev = dp[i - 1]
        bt_i = bt[i]
        for j in range(1, m + 1):
            sim = _block_similarity(ref_i, other[j - 1])
            s_diag = dp_prev[j - 1] + sim
            s_up = dp_prev[j] - gap_penalty
            s_left = dp_i[j - 1] - gap_penalty
            if s_diag >= s_up and s_diag >= s_left:
                dp_i[j] = s_diag
                bt_i[j] = 0
            elif s_up >= s_left:
                dp_i[j] = s_up
                bt_i[j] = 1
            else:
                dp_i[j] = s_left
                bt_i[j] = 2

    pairs: list[tuple[int | None, int | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        move = bt[i][j]
        if move == 0:
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif move == 1:
            pairs.append((i - 1, None))
            i -= 1
        else:
            pairs.append((None, j - 1))
            j -= 1
    pairs.reverse()
    return pairs


def align_blocks(model_blocks: dict[str, list[str]]) -> list[dict[str, str]]:
    """Align normalized page/paragraph blocks across all models."""
    models = list(model_blocks.keys())
    counts = {len(v) for v in model_blocks.values()}
    if len(counts) == 1:
        num_blocks = next(iter(counts))
        return [{m: model_blocks[m][i] for m in models} for i in range(num_blocks)]

    pivot = max(
        models,
        key=lambda m: (len(model_blocks[m]), sum(len(b) for b in model_blocks[m])),
    )
    pivot_list = model_blocks[pivot]
    n_pivot = len(pivot_list)

    aligned_on_pivot: dict[str, list[str]] = {pivot: list(pivot_list)}
    insertions: dict[int, dict[str, list[str]]] = {}

    for m in models:
        if m == pivot:
            continue
        other_list = model_blocks[m]
        pairs = _align_two_block_lists(pivot_list, other_list)
        mapped = [""] * n_pivot
        last_pivot_idx = -1
        for ref_idx, other_idx in pairs:
            if ref_idx is not None:
                last_pivot_idx = ref_idx
                mapped[ref_idx] = other_list[other_idx] if other_idx is not None else ""
            elif other_idx is not None:
                slot = last_pivot_idx + 1
                insertions.setdefault(slot, {}).setdefault(m, []).append(
                    other_list[other_idx]
                )
        aligned_on_pivot[m] = mapped

    result: list[dict[str, str]] = []
    for slot in range(n_pivot + 1):
        if slot in insertions:
            max_ins = max(len(v) for v in insertions[slot].values())
            for k in range(max_ins):
                result.append(
                    {
                        m: (
                            insertions[slot][m][k]
                            if m in insertions[slot] and k < len(insertions[slot][m])
                            else ""
                        )
                        for m in models
                    }
                )
        if slot < n_pivot:
            result.append({m: aligned_on_pivot[m][slot] for m in models})
    return result


# ---------------------------------------------------------------------------
# Step 2: Exact Levenshtein character alignment (Ukkonen-banded exact DP)
# ---------------------------------------------------------------------------

def _banded_levenshtein_align(
    a: str,
    b: str,
    band: int,
) -> tuple[int, list[tuple[str, str]]] | None:
    """Compute Levenshtein alignment restricted to diagonal band ``|i - j| <= band``.

    By Ukkonen's theorem, if the resulting distance ``d <= band``, it is guaranteed
    to be the exact global minimum Levenshtein alignment. Returns ``None`` if the
    true distance may exceed ``band`` so the caller can widen the band.
    """
    n, m = len(a), len(b)
    if abs(n - m) > band:
        return None

    inf = n + m + 1
    # Store rows sparsely as (j_start, dp_row, bt_row) for memory & speed.
    # bt codes: 0 = diag (match/sub), 1 = up (delete from a), 2 = left (insert from b)
    rows_bt: list[tuple[int, bytearray]] = []

    j_end0 = min(m, band)
    prev_start = 0
    prev_dp = list(range(j_end0 + 1))
    bt0 = bytearray([2] * (j_end0 + 1))
    bt0[0] = 0
    rows_bt.append((0, bt0))

    for i in range(1, n + 1):
        j_start = max(0, i - band)
        j_end = min(m, i + band)
        width = j_end - j_start + 1
        cur_dp = [inf] * width
        cur_bt = bytearray(width)
        ch_a = a[i - 1]
        prev_len = len(prev_dp)

        for idx in range(width):
            j = j_start + idx

            # Up: (i - 1, j) -> delete a[i - 1]
            p_up = j - prev_start
            c_up = prev_dp[p_up] + 1 if 0 <= p_up < prev_len else inf

            if j == 0:
                cur_dp[idx] = c_up
                cur_bt[idx] = 1
                continue

            # Diag: (i - 1, j - 1) -> match or substitute
            p_diag = j - 1 - prev_start
            if 0 <= p_diag < prev_len:
                c_diag = prev_dp[p_diag] + (0 if ch_a == b[j - 1] else 1)
            else:
                c_diag = inf

            # Left: (i, j - 1) -> insert b[j - 1]
            c_left = cur_dp[idx - 1] + 1 if idx > 0 else inf

            if c_diag <= c_up and c_diag <= c_left:
                cur_dp[idx] = c_diag
                cur_bt[idx] = 0
            elif c_up <= c_left:
                cur_dp[idx] = c_up
                cur_bt[idx] = 1
            else:
                cur_dp[idx] = c_left
                cur_bt[idx] = 2

        rows_bt.append((j_start, cur_bt))
        prev_dp = cur_dp
        prev_start = j_start

    dist = prev_dp[m - prev_start]
    if dist > band:
        return None

    # Backtrace exact optimal alignment.
    aligned: list[tuple[str, str]] = []
    i, j = n, m
    while i > 0 or j > 0:
        j_start, bt_row = rows_bt[i]
        move = bt_row[j - j_start]
        if move == 0:
            aligned.append((a[i - 1], b[j - 1]))
            i -= 1
            j -= 1
        elif move == 1:
            aligned.append((a[i - 1], ""))
            i -= 1
        else:
            aligned.append(("", b[j - 1]))
            j -= 1
    aligned.reverse()
    return dist, aligned


def levenshtein_align(a: str, b: str) -> tuple[int, list[tuple[str, str]]]:
    """Compute exact minimum-edit-distance Levenshtein alignment between ``a`` and ``b``.

    Returns:
        ``(edit_distance, aligned_pairs)`` where each pair is ``(char_a_or_gap, char_b_or_gap)``.
    """
    n, m = len(a), len(b)
    if n == 0:
        return m, [("", ch) for ch in b]
    if m == 0:
        return n, [(ch, "") for ch in a]
    if a == b:
        return 0, [(ch, ch) for ch in a]

    # Fast-strip identical prefix and suffix (always optimal in Levenshtein alignment).
    prefix_len = 0
    min_len = min(n, m)
    while prefix_len < min_len and a[prefix_len] == b[prefix_len]:
        prefix_len += 1

    suffix_len = 0
    while (
        suffix_len < (min_len - prefix_len)
        and a[n - 1 - suffix_len] == b[m - 1 - suffix_len]
    ):
        suffix_len += 1

    a_mid = a[prefix_len : n - suffix_len]
    b_mid = b[prefix_len : m - suffix_len]

    if not a_mid and not b_mid:
        return 0, [(ch, ch) for ch in a]
    if not a_mid:
        mid_pairs = [("", ch) for ch in b_mid]
        dist = len(b_mid)
    elif not b_mid:
        mid_pairs = [(ch, "") for ch in a_mid]
        dist = len(a_mid)
    else:
        max_band = max(len(a_mid), len(b_mid))
        band = max(abs(len(a_mid) - len(b_mid)), 16)
        res = None
        while res is None:
            band = min(band, max_band)
            res = _banded_levenshtein_align(a_mid, b_mid, band)
            if res is None:
                if band == max_band:
                    raise RuntimeError("Levenshtein alignment failed to converge.")
                band *= 2
        dist, mid_pairs = res

    prefix_pairs = [(a[k], a[k]) for k in range(prefix_len)]
    suffix_pairs = [(a[n - suffix_len + k], a[n - suffix_len + k]) for k in range(suffix_len)]
    return dist, prefix_pairs + mid_pairs + suffix_pairs


# ---------------------------------------------------------------------------
# Step 3: Group corresponding characters into columns (Center-Star MSA)
# ---------------------------------------------------------------------------

def _choose_pivot(model_texts: dict[str, str]) -> str:
    """Select the center (pivot) sequence with highest similarity to all others."""
    models = list(model_texts.keys())
    if len(models) <= 2:
        return max(models, key=lambda m: len(model_texts[m]))

    scores: dict[str, float] = {m: 0.0 for m in models}
    for i, m1 in enumerate(models):
        t1 = model_texts[m1]
        for m2 in models[i + 1 :]:
            t2 = model_texts[m2]
            sim = _block_similarity(t1, t2)
            scores[m1] += sim
            scores[m2] += sim

    return max(models, key=lambda m: (scores[m], len(model_texts[m])))


def align_to_columns(
    model_texts: dict[str, str],
) -> tuple[list[str], list[list[str]]]:
    """Align multiple OCR sequences via Levenshtein alignment and group into columns.

    Args:
        model_texts: Mapping ``model_name -> normalized_text`` for one block.

    Returns:
        ``(models, columns)`` where each column is a list of strings (either a single
        character or ``""`` for a gap) ordered by ``models``.
    """
    models = list(model_texts.keys())
    non_empty = [m for m in models if model_texts[m]]
    if not non_empty:
        return models, []

    if len(non_empty) == 1:
        only_m = non_empty[0]
        return models, [
            [ch if m == only_m else "" for m in models] for ch in model_texts[only_m]
        ]

    pivot = _choose_pivot(model_texts)
    pivot_text = model_texts[pivot]
    n_pivot = len(pivot_text)

    aligned_chars: dict[str, list[str]] = {pivot: list(pivot_text)}
    # insertions[slot][model] holds characters inserted by `model` before pivot[slot].
    insertions: dict[int, dict[str, str]] = {}

    for m in models:
        if m == pivot:
            continue
        _, pairs = levenshtein_align(pivot_text, model_texts[m])
        mapped = [""] * n_pivot
        p_idx = 0
        for p_ch, m_ch in pairs:
            if p_ch != "":
                mapped[p_idx] = m_ch
                p_idx += 1
            else:
                slot_dict = insertions.setdefault(p_idx, {})
                slot_dict[m] = slot_dict.get(m, "") + m_ch
        aligned_chars[m] = mapped

    # Merge pivot positions and inter-pivot insertion slots into final columns.
    columns: list[list[str]] = []
    for slot in range(n_pivot + 1):
        if slot in insertions:
            slot_ins = insertions[slot]
            if len(slot_ins) == 1:
                only_m, ins_str = next(iter(slot_ins.items()))
                for ch in ins_str:
                    columns.append([ch if m == only_m else "" for m in models])
            else:
                sub_texts = {m: slot_ins.get(m, "") for m in models}
                _, sub_cols = align_to_columns(sub_texts)
                columns.extend(sub_cols)

        if slot < n_pivot:
            columns.append([aligned_chars[m][slot] for m in models])

    return models, columns
