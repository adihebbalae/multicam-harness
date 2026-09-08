# Ported from Wavy-Hec/CVBench bench/tests/test_segment_select_global.py
# @ a8e995dd0c069911c1c99acbc5854cc22337d34c.
# Deliberate delta vs source: the fork's coverage-reduce refusal is not
# exercised — this port carries no coverage reduce.
"""select_segments(): the global (cross-clip) selection rule, and proof that
per_clip still reproduces the inline top-K it replaced.

Pure CPU — no backend, no scorer model, no video. Runnable either way:

    python -m tests.test_segment_select_global
    pytest tests/test_segment_select_global.py
"""
import os
import random
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np                                                # noqa: E402

from inprocess.harnesses.segment_select import (                  # noqa: E402
    SEGMENT_SELECT_GLOBAL_PREFIX, SEGMENT_SELECT_PREFIX,
    SegmentSelectMethod, select_segments)


def _reduce_max(seg_opt):
    """seg_opt {video: {seg: [per-option]}} -> seg_scores, exactly as
    SegmentSelectMethod._prepare reduces it."""
    return {v: {sid: float(np.asarray(r).max()) for sid, r in d.items()}
            for v, d in seg_opt.items()}


def _old_inline_rule(seg_scores, keep_top):
    """ORACLE: the expression select_segments replaced, copied verbatim from
    segment_select.py before the split. Any per_clip divergence from this is a
    behaviour change."""
    return {v: sorted(sorted(d, key=lambda s: -d[s])[: keep_top])
            for v, d in seg_scores.items()}


# The toy: 2 options, 2 cameras, 3 chunks. Segment ids are 0-based, so
# vc1/vc2/vc3 are segments 0/1/2.
TOY = {1: {0: [0.2, 0.5], 1: [0.7, 0.3], 2: [0.9, 0.2]},
       2: {0: [0.2, 0.6], 1: [0.1, 0.1], 2: [0.3, 0.1]}}
TOY_K = 2


def test_toy_reduce():
    # max over options: c1 -> .5 .7 .9 ; c2 -> .6 .1 .3
    assert _reduce_max(TOY) == {1: {0: 0.5, 1: 0.7, 2: 0.9},
                                2: {0: 0.6, 1: 0.1, 2: 0.3}}


def test_toy_per_clip_top1():
    """Current rule, top-1 per camera: {c1: vc3, c2: vc1}."""
    got = select_segments(_reduce_max(TOY), "per_clip", 1, 1, TOY_K)
    assert got == {1: [2], 2: [0]}, got
    assert got == _old_inline_rule(_reduce_max(TOY), 1)


def test_toy_global_budget2_floor0():
    """Global top-2, no floor: the two best pairs are both camera 1's."""
    got = select_segments(_reduce_max(TOY), "global", 2, 0, TOY_K)
    assert got == {1: [1, 2], 2: []}, got


def test_toy_global_budget2_floor1():
    """Global top-2 with a floor of 1: each camera is seated first, which
    reproduces the per-camera top-1 set."""
    got = select_segments(_reduce_max(TOY), "global", 2, 1, TOY_K)
    assert got == {1: [2], 2: [0]}, got


def test_toy_global_budget3():
    """Global top-3: floor 0 and floor 1 agree here."""
    for floor in (0, 1):
        got = select_segments(_reduce_max(TOY), "global", 3, floor, TOY_K)
        assert got == {1: [1, 2], 2: [0]}, (floor, got)


def test_toy_floor_capped_below_one_per_clip():
    """Floor 1 with a 1-segment budget cannot seat both clips; the cap
    (keep_top // K = 0) drops it rather than overspending the budget."""
    got = select_segments(_reduce_max(TOY), "global", 1, 1, TOY_K)
    assert got == {1: [2], 2: []}, got


def test_per_clip_matches_old_inline_rule():
    """200 random score dicts, coarse-grained so TIES are common, with keys
    inserted in shuffled order so insertion-order tie-breaking is exercised
    (the inline rule leans on Python's stable sort over the dict)."""
    rng = random.Random(20260907)
    checked_ties = 0
    for _ in range(200):
        K = rng.randint(1, 6)
        seg_scores = {}
        for v in range(1, K + 1):
            n_seg = rng.randint(1, 10)
            # a tiny value ladder so equal scores show up constantly
            vals = {s: rng.choice([0.1, 0.2, 0.3, 0.4]) for s in range(n_seg)}
            keys = list(vals)
            rng.shuffle(keys)
            seg_scores[v] = {s: vals[s] for s in keys}
            if len(set(vals.values())) < len(vals):
                checked_ties += 1
        keep_top = rng.randint(1, 12)
        got = select_segments(seg_scores, "per_clip", keep_top, 1, K)
        want = _old_inline_rule(seg_scores, keep_top)
        assert got == want, (seg_scores, keep_top, got, want)
    assert checked_ties > 100, f"too few tied cases ({checked_ties})"


def test_global_floor1_starves_nobody_when_budget_allows():
    """With floor 1 and keep_top >= K, every clip keeps at least one segment
    (and the budget is never overspent)."""
    rng = random.Random(11)
    for _ in range(200):
        K = rng.randint(1, 6)
        seg_scores = {v: {s: rng.random() for s in range(rng.randint(1, 8))}
                      for v in range(1, K + 1)}
        keep_top = rng.randint(K, K + 10)
        got = select_segments(seg_scores, "global", keep_top, 1, K)
        assert set(got) == set(seg_scores)
        for v, segs in got.items():
            assert segs, (v, got, keep_top)
            assert segs == sorted(segs)
            assert set(segs) <= set(seg_scores[v])
        total = sum(len(s) for s in got.values())
        assert total <= keep_top, (total, keep_top)
        assert total == min(keep_top, sum(len(d) for d in seg_scores.values()))


def test_global_is_a_true_global_topn_without_floor():
    """floor 0: the kept set is exactly the N best (video, segment) pairs
    under the (-score, video, segment) tie-break."""
    rng = random.Random(5)
    for _ in range(200):
        K = rng.randint(1, 5)
        seg_scores = {v: {s: rng.choice([0.1, 0.2, 0.3])
                          for s in range(rng.randint(1, 6))}
                      for v in range(1, K + 1)}
        n = rng.randint(1, 15)
        got = select_segments(seg_scores, "global", n, 0, K)
        pairs = sorted(((v, s) for v, d in seg_scores.items() for s in d),
                       key=lambda p: (-seg_scores[p[0]][p[1]], p[0], p[1]))
        want = {v: [] for v in seg_scores}
        for v, s in pairs[:n]:
            want[v].append(s)
        want = {v: sorted(s) for v, s in want.items()}
        assert got == want, (seg_scores, n, got, want)


def test_value_errors():
    seg_scores = _reduce_max(TOY)
    for bad in ("per-clip", "GLOBAL", "", None, "clip"):
        try:
            select_segments(seg_scores, bad, 2, 1, TOY_K)
        except ValueError as e:
            assert "per_clip" in str(e) and "global" in str(e), e
        else:
            raise AssertionError(f"mode {bad!r} did not raise")
    try:
        select_segments(seg_scores, "global", 2, -1, TOY_K)
    except ValueError as e:
        assert "seg_floor" in str(e), e
    else:
        raise AssertionError("negative seg_floor did not raise")


def test_constructor_value_errors():
    """Config-level refusals — no backend attributes are touched on these
    paths, so a bare object() stands in for the model."""
    base = dict(name="segment_select", dedup_tau=1.0)

    try:
        SegmentSelectMethod(object(), seg_select="worldwide", **base)
    except ValueError as e:
        assert "seg_select" in str(e), e
    else:
        raise AssertionError("bad seg_select did not raise")

    try:
        SegmentSelectMethod(object(), seg_select="global", seg_floor=-2, **base)
    except ValueError as e:
        assert "seg_floor" in str(e), e
    else:
        raise AssertionError("negative seg_floor did not raise")

    # the defaults must still construct, and must still mean per_clip
    m = SegmentSelectMethod(object(), **base)
    assert m.seg_select == "per_clip" and m.seg_floor == 1


def test_defaults_are_unchanged():
    """The new kwargs default to the historic protocol."""
    m = SegmentSelectMethod(object(), name="segment_select", dedup_tau=1.0)
    assert m.seg_select == "per_clip"
    seg_scores = _reduce_max(TOY)
    assert (select_segments(seg_scores, m.seg_select, 2, m.seg_floor, TOY_K)
            == _old_inline_rule(seg_scores, 2))


def test_prompt_prefixes_differ_only_in_the_selection_sentences():
    a = SEGMENT_SELECT_PREFIX.split(". ")
    b = SEGMENT_SELECT_GLOBAL_PREFIX.split(". ")
    # sentence 0 (what the clips are) and the trailing sentences are identical
    assert a[0] == b[0]
    assert a[-1] == b[-1] == "Reason over the shown frames to answer."
    assert "across ALL clips" in SEGMENT_SELECT_GLOBAL_PREFIX
    assert "across ALL clips" not in SEGMENT_SELECT_PREFIX
    assert ("A clip may contribute several segments, one, or none"
            in SEGMENT_SELECT_GLOBAL_PREFIX)
    for shared in ("Frames are shown grouped by their source Video (ORIGINAL "
                   "numbering) and in temporal order within each Video",
                   "a banner '=== Video k ===' precedes each Video's frames",
                   "A clip whose frames were all removed as duplicates is "
                   "omitted.",
                   "near-duplicate frames were removed"):
        assert shared in SEGMENT_SELECT_PREFIX, shared
        assert shared in SEGMENT_SELECT_GLOBAL_PREFIX, shared
    # the ONLY textual deltas: the selection sentence and the added omission
    # clause. Rewriting one into the other must reproduce the per_clip prefix
    # byte for byte.
    rebuilt = SEGMENT_SELECT_GLOBAL_PREFIX.replace(
        "the most relevant segments across ALL clips (at most {top} in total) "
        "are represented",
        "its most relevant segments (at most {top}) are represented").replace(
        "A clip may contribute several segments, one, or none; a clip with no "
        "represented segment is omitted. ", "")
    assert rebuilt == SEGMENT_SELECT_PREFIX, rebuilt


def main():
    fails = 0
    for name, fn in sorted(list(globals().items())):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
        except Exception as e:                                # noqa: BLE001
            fails += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
        else:
            print(f"ok   {name}")
    print("FAILED" if fails else "ALL PASS")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
