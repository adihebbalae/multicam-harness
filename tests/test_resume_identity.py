# Ported from Wavy-Hec/MultiCam bench/tests/test_resume_identity.py
# @ ea900baed66e6604b5fd0f40ee75451b00f5ebd3.
# Deliberate delta vs source: the identity assertions go through this repo's
# mechanism — scan_output()'s per-knob value sets and main()'s stray-knob
# refusal — rather than the source's existing_identity()/seg_identity()
# helpers, which have no counterpart here. The source's stamp_seg_identity test
# has no analogue either: this runner builds ONE stamp dict from the knobs and
# applies it to every row it writes, error rows included, so no arm-specific
# stamp can go missing.
# Deliberate delta vs source: the fixtures use a SigLIP-tagged segment arm and
# a second segment arm in place of the fork's viclip arm and its relevance-free
# control (the viclip tag would trip the download precheck first, and this port
# carries no control arm), --budget 0 rather than an unset budget for the
# matched nframes x K, and --seg-floor 0 in place of the fork's launcher env.
"""The resume identity of a segment_select leg, and the submit-time refusals
that stand between a global leg and a silent no-op.

The selection MODE is a flag, not part of the method name: both modes write
`segment_select[_<scorer>][_opt]` and the resume key is
(dataset, id, method, backend, pass_idx), so nothing but the row's own
`seg_select`/`seg_floor` stamp tells a per_clip file from a global one. These
tests pin what a resume reads back from that stamp (`scan_output`, and the
stray-knob refusal in `main`) and the guards that refuse a leg whose global
ranking could not decide anything (`check_global_regimes`, and the flag-guard
order inside `main`).

Pure CPU — no backend, no scorer model, no video, no Slurm. Runnable either way:

    pytest tests/test_resume_identity.py
    python -m tests.test_resume_identity

Rows are synthetic on purpose: no global row exists on disk (the first campaign
has not run) and no segment_select row written so far is a prepare-error row,
so real results cannot exercise either path.
"""
import argparse
import contextlib
import io
import json
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from inprocess import run as run_module                          # noqa: E402
from inprocess.run import (IDENTITY_KNOB_DEFAULTS,               # noqa: E402
                           check_global_regimes, scan_output)

# what BOTH modes write as `method`; the siglip tag keeps the ViCLIP download
# precheck (which runs before these guards) out of the way. BACKEND is the HF
# id's basename, which is what rows record and what the file-identity guards
# compare — a fixture that disagreed would die with the wrong message.
MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
BACKEND = "Qwen2.5-VL-7B-Instruct"
METHOD = "segment_select_siglip_opt"
METHOD2 = "segment_select_siglip"
DATASET = "toy_subset"                   # = basename of the subset written below


def _args(**kw):
    """The runner's segment-selection defaults, overridden per test."""
    a = dict(seg_select="per_clip", seg_floor=1, budget=0, nframes=8)
    a.update(kw)
    return argparse.Namespace(**a)


def _stamped(seg_select, seg_floor=1):
    """The (seg_select, seg_floor) pair a run with these flags stamps on its
    rows and compares on a resume — the floor is None outside global, where it
    is inert."""
    return seg_select, (seg_floor if seg_select == "global" else None)


def _row(**kw):
    """A row shaped like the write loop's output for a segment_select leg: the
    run's stamp lands top-level, beside the media stamp."""
    sel, floor = _stamped("per_clip")
    r = {"id": 1, "method": METHOD, "backend": BACKEND, "dataset": DATASET,
         "pass_idx": 1, "correct": True, "media_remap": None,
         "seg_select": sel, "seg_floor": floor,
         "frame_alloc": {"mode": "segment_select", "K": 4,
                         "segment_select_mode": "per_clip", "seg_floor": None}}
    r.update(kw)
    return r


def _write(path, rows):
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return path


def _seg_seen(path):
    """(seg_select, seg_floor) value sets a resume would read out of `path`."""
    seen = scan_output(path)[2]
    return seen.get("seg_select", set()), seen.get("seg_floor", set())


def _stray(path, seg_select, seg_floor=1):
    """The seg knobs main() would refuse on for a run with these flags —
    main()'s own comparison, over the values scan_output reports."""
    want = dict(zip(("seg_select", "seg_floor"), _stamped(seg_select, seg_floor)))
    seen = scan_output(path)[2]
    return {k: sorted((v for v in seen.get(k, ()) if v != want[k]), key=str)
            for k in want
            if any(v != want[k] for v in seen.get(k, ()))}


def _rec(rid, K):
    """A question record with K clips — enough for num_videos()."""
    rec = {"id": rid, "task_type": "toy", "question": "what?", "answer": "A",
           "options": ["A. x", "B. y"]}
    rec.update({f"video_{i}": f"clip{i}.mp4" for i in range(1, K + 1)})
    return rec


def _subset(d, ks=(4, 4)):
    p = os.path.join(d, DATASET + ".json")
    with open(p, "w") as fh:
        json.dump([_rec(i, k) for i, k in enumerate(ks)], fh)
    return p


def _run_main(argv, video_root="."):
    """main() under `argv`, returning its SystemExit message and stdout.

    Every call below must die at a submit-time guard. Replace make_backend with
    a tripwire so a REGRESSED guard fails this test by name instead of pulling a
    7B checkpoint onto a CPU node.
    """
    def _tripwire(*a, **kw):
        raise AssertionError(
            f"reached make_backend: a submit-time guard did not fire for {argv}")

    argv = list(argv) + ["--model", MODEL, "--strict-answer-prompt", "1",
                         "--video-root", video_root]
    old_argv, old_make = sys.argv, run_module.make_backend
    sys.argv = ["run"] + argv
    run_module.make_backend = _tripwire
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            run_module.main()
    except SystemExit as e:
        return str(e), buf.getvalue()
    finally:
        sys.argv, run_module.make_backend = old_argv, old_make
    raise AssertionError(f"main() did not refuse: {argv}\n{buf.getvalue()}")


# --------------------------------------------------------------- identity ---

def test_per_clip_file_resumes_under_per_clip_flags():
    """(a) The common case: an existing per_clip file must stay resumable."""
    with tempfile.TemporaryDirectory() as d:
        p = _write(os.path.join(d, "per_clip.jsonl"),
                   [_row(id=i) for i in (1, 2, 3)])
        assert _seg_seen(p) == ({"per_clip"}, {None}), _seg_seen(p)
        assert _stray(p, "per_clip") == {}            # nothing stray -> resumes


def test_pre_feature_rows_compare_equal_to_per_clip():
    """(d) Rows written before the stamp existed carry NEITHER key (not a null
    value) and were per_clip by construction, so they must read as per_clip: they
    pool with new per_clip rows, and they do NOT silently accept a global leg,
    which is the whole point of keying identity on the mode."""
    with tempfile.TemporaryDirectory() as d:
        legacy = _row(id=1)
        legacy.pop("seg_select"), legacy.pop("seg_floor")
        assert "seg_select" not in legacy and "seg_floor" not in legacy
        p = _write(os.path.join(d, "legacy.jsonl"), [legacy, _row(id=2)])
        assert _seg_seen(p) == ({"per_clip"}, {None})   # ONE identity, not two
        assert _stray(p, "per_clip") == {}
        assert _stray(p, "global", 1) == {"seg_select": ["per_clip"],
                                          "seg_floor": [None]}
        assert IDENTITY_KNOB_DEFAULTS == {"seg_select": "per_clip",
                                          "seg_floor": None,
                                          "segment_seconds": 0.0}


def test_per_clip_file_is_refused_under_global():
    """(b) The silent-no-op case: same method name, same backend, same subset —
    only the flag differs, so without this the global leg finds every key done,
    appends nothing, and rewrites the summary under the new mode's label."""
    with tempfile.TemporaryDirectory() as d:
        p = _write(os.path.join(d, "per_clip.jsonl"), [_row(id=i) for i in (1, 2)])
        assert _stray(p, "global", 1) == {"seg_select": ["per_clip"],
                                          "seg_floor": [None]}

        msg, _ = _run_main(["--subset", _subset(d), "--methods", METHOD,
                            "--out", p, "--seg-select", "global",
                            "--budget", "96"], video_root=d)
        assert "refusing to append" in msg, msg
        assert "seg_select: file holds ['per_clip']" in msg, msg   # names BOTH
        assert "this run is 'global'" in msg, msg
        assert "own --out" in msg, msg                             # the remedy


def test_global_floors_do_not_pool():
    """(c) Two configured floors are two protocols; neither may resume the
    other's file, in either direction."""
    with tempfile.TemporaryDirectory() as d:
        for mine, theirs in ((1, 2), (2, 1)):
            p = _write(os.path.join(d, f"g{mine}.jsonl"),
                       [_row(id=1, seg_select="global", seg_floor=mine)])
            assert _seg_seen(p) == ({"global"}, {mine}), _seg_seen(p)
            assert _stray(p, "global", mine) == {}
            assert _stray(p, "global", theirs) == {"seg_floor": [mine]}


def test_capped_effective_floor_does_not_block_resume():
    """The defect this stamp exists to remove: select_segments caps the floor at
    N // K, so ONE floor-1 leg stamps frame_alloc.seg_floor 1 on its small-K
    records and 0 on its large-K ones. Identity reads the CONFIGURED value, so
    the leg still resumes itself."""
    with tempfile.TemporaryDirectory() as d:
        p = _write(os.path.join(d, "mixed_k.jsonl"), [
            _row(id=1, seg_select="global", seg_floor=1,
                 frame_alloc={"segment_select_mode": "global", "K": 2,
                              "seg_floor": 1}),
            _row(id=2, seg_select="global", seg_floor=1,
                 frame_alloc={"segment_select_mode": "global", "K": 13,
                              "seg_floor": 0}),      # capped away by the budget
        ])
        assert _seg_seen(p) == ({"global"}, {1}), _seg_seen(p)
        assert _stray(p, "global", 1) == {}


def test_error_rows_are_read_from_the_top_level_keys():
    """(e) A prepare failure returns a Result with NO frame_alloc, so the mode
    can only live top-level — which it does, because the run's stamp is one dict
    applied to every row it writes. Without that, one unreadable clip would make
    a global leg read back as per_clip and block every later resume."""
    err = _row(id=7, seg_select="global", seg_floor=1, frame_alloc=None,
               correct=False,
               error="prepare:FileNotFoundError: no candidate frames from any clip")
    with tempfile.TemporaryDirectory() as d:
        p = _write(os.path.join(d, "err.jsonl"), [err])
        assert _seg_seen(p) == ({"global"}, {1})
        counts, errored, _, _ = scan_output(p)
        assert errored == set(counts)            # it is an error row, and done


# ----------------------------------------------------------------- guards ---

def test_negative_floor_message_wins():
    """(f) A negative floor is also != 1, so the outside-global guard answers it
    first unless the negative check is ordered ahead: 'applies only to
    --seg-select global' sends the operator to set the mode instead of fixing
    the typo."""
    with tempfile.TemporaryDirectory() as d:
        s = _subset(d)
        msg, _ = _run_main(["--subset", s, "--methods", METHOD,
                            "--seg-floor", "-2"], video_root=d)
        assert "is negative" in msg, msg
        assert "applies only to" not in msg, msg
        # and it still fires under global, where the outside-global guard cannot
        msg, _ = _run_main(["--subset", s, "--methods", METHOD,
                            "--seg-select", "global", "--seg-floor", "-1"],
                           video_root=d)
        assert "is negative" in msg, msg


def test_budget_below_K_is_refused_at_submit():
    """(g) SegmentSelectMethod._prepare raises `budget < K views` — but only
    after make_backend has loaded the model, and then once per record per pass,
    so the leg pays for a GPU to write error rows."""
    try:
        check_global_regimes([METHOD], [_rec(0, 4)],
                             _args(seg_select="global", budget=3))
    except SystemExit as e:
        assert "one frame per clip" in str(e), e
        assert "frames_per_segment x (K x seg_floor + 1)" in str(e), e
    else:
        raise AssertionError("budget 3 < K 4 was not refused")

    # via main(), i.e. the call site really runs before make_backend
    with tempfile.TemporaryDirectory() as d:
        msg, _ = _run_main(["--subset", _subset(d, ks=(4,)), "--methods", METHOD,
                            "--seg-select", "global", "--budget", "3"],
                           video_root=d)
        assert "one frame per clip" in msg, msg


def test_inert_global_leg_is_refused_and_a_live_one_is_not():
    """N <= K x seg_floor is per_clip top-floor wearing a global tag. At four
    clips and 8-frame segments that is budget 32 — and the matched budget
    (nframes x K) too; 64 and 96 compete."""
    data = [_rec(0, 4), _rec(1, 4)]
    for budget in (0, 32):               # 0 = matched, i.e. 8 x 4 = 32
        try:
            check_global_regimes([METHOD], data,
                                 _args(seg_select="global", budget=budget))
        except SystemExit as e:
            assert "inert" in str(e), e
            assert ("Raise --budget to at least frames_per_segment x "
                    "(K x seg_floor + 1)") in str(e), e
        else:
            raise AssertionError(f"budget {budget} was not refused")
    for budget in (64, 96):
        got = check_global_regimes([METHOD], data,
                                   _args(seg_select="global", budget=budget))
        assert got == {"competes": 2}, (budget, got)
    # floor 0 never has a floor pass to cancel the ranking
    got = check_global_regimes([METHOD], data,
                               _args(seg_select="global", seg_floor=0, budget=32))
    assert got == {"pure global": 2}, got


def test_capped_floor_warns_even_when_every_record_is_live():
    """A leg whose budget cancels the floor on EVERY record is not refused — the
    ranking still decides — but the starvation guard the operator asked for is
    not in force anywhere, so it must not pass in silence."""
    data = [_rec(0, 13), _rec(1, 13)]        # N = 64 // 8 = 8 < K, so f -> 0
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        got = check_global_regimes([METHOD], data,
                                   _args(seg_select="global", budget=64))
    assert got == {"floor cancelled by the budget": 2}, got
    warns = [ln for ln in buf.getvalue().splitlines() if ln.startswith("WARNING")]
    assert len(warns) == 1, buf.getvalue()                 # not the mixed one
    assert "capped away on 2/2" in warns[0], warns
    assert "leave clips" in warns[0], warns
    # raising the budget past K x seg_floor puts the floor back and silences it
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        got = check_global_regimes([METHOD], data,
                                   _args(seg_select="global", budget=8 * 14))
    assert got == {"competes": 2}, got
    assert buf.getvalue() == "", buf.getvalue()


def test_mixed_regimes_warn_once_per_leg():
    """The scan reads only args and data, so a leg naming two segment arms must
    classify its records — and warn — ONCE, not once per arm."""
    data = [_rec(0, 4), _rec(1, 8)]          # at budget 64: competes / degenerate
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        got = check_global_regimes([METHOD], data,
                                   _args(seg_select="global", budget=64))
    assert got == {"competes": 1, "degenerate (== per_clip top-1)": 1}, got
    assert buf.getvalue().startswith("WARNING "), buf.getvalue()

    with tempfile.TemporaryDirectory() as d:
        p = _write(os.path.join(d, "per_clip.jsonl"), [_row(id=1)])
        msg, out = _run_main(["--subset", _subset(d, ks=(4, 8)),
                              "--methods", f"{METHOD},{METHOD2}",
                              "--out", p, "--seg-select", "global",
                              "--budget", "64"], video_root=d)
        assert "refusing to append" in msg, msg          # died at the resume guard
        warns = [ln for ln in out.splitlines() if ln.startswith("WARNING")]
        assert len(warns) == 1, out
        assert "mixes selection regimes" in warns[0], warns


def main():
    fails = 0
    for name, fn in sorted(list(globals().items())):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
        # SystemExit is NOT an Exception: every guard here raises it, so a
        # miswired test would otherwise abort this runner silently
        except (Exception, SystemExit) as e:                  # noqa: BLE001
            fails += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
        else:
            print(f"ok   {name}")
    print("FAILED" if fails else "ALL PASS")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
