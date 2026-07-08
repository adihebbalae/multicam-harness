#!/usr/bin/env python
"""Harness-equivalence drift detector: this repo vs the CVBench fork.

For a handful of real CVBench records, build every ported harness (Method) side
by side with its fork original — IDENTICAL constructor args, IDENTICAL
``answer(rec, video_root, seed=...)`` call — behind a FakeBackend that records
the full ``messages`` structure instead of running a model. Then assert, per
generate() call:

  * every text item is byte-identical,
  * every video item names the same file with the same ``nframes`` (and the
    frames a backend would decode from it hash pixel-identical, via decord on
    CPU at the InternVL ``get_index`` grid),
  * every image item (centralized montage) is pixel-identical (``tobytes()``),
  * generation kwargs (max_new_tokens / seed / temperature) match,

and per Result: ``frame_alloc`` dicts equal (where the method sets them), plus
method name / prediction / gold / abstained / num_model_calls, and no swallowed
``error``. No model is ever loaded; frame decoding runs on CPU via decord.

Pairs covered: temporal_weighted, temporal_even, cvbench_native,
centralized(stitch) with montage_kind='camera' AND 'video', per_stream with
stream_kind='camera' AND 'video'.

TODO(--slow): the clip_select pair (SummarySelectMethod / ClipScoreSelectMethod)
is SKIPPED here — it needs CLIP/SigLIP weights and the per-clip summary cache.
Add it behind a --slow flag once a tiny cached fixture exists.

Gating / usage (the fork checkout is READ-ONLY — never write bytecode into it):

  CVBENCH_FORK=/path/to/CVBench PYTHONDONTWRITEBYTECODE=1 \
      python tests/compare_prompts_vs_fork.py

Exits 0 with a skip message when CVBENCH_FORK is unset; exits 1 on any
divergence, printing the first divergent line/pixel with context.

sys.path hygiene (IMPORTANT — the ordering below is load-bearing):
  * The fork's package is ``bench.*`` and this repo's modules are top-level
    (``harnesses``/``models``/``dataloaders``/``evaluation``), so both trees can
    sit on sys.path without any import naming collision between them.
  * BUT importing ``bench.*`` triggers ``bench.reuse``, which does
    ``sys.path.insert(0, <fork>/Video-R1/src)`` — that directory jumps AHEAD of
    this repo for every later import. Video-R1/src contains a ``scripts/``
    directory (as does this repo) and top-level shims like ``eval_thinking``;
    to guarantee nothing from the fork ever shadows this repo's modules, this
    repo's path goes FIRST and ALL of its modules are imported BEFORE the fork
    path is added. (No actual collision was hit in this test — 'scripts' is the
    only shared name and neither side imports it — but the ordering keeps it
    that way.)
"""
import hashlib
import json
import os
import sys
from collections import defaultdict

sys.dont_write_bytecode = True  # fork is read-only; also run with PYTHONDONTWRITEBYTECODE=1

# ---------------------------------------------------------------------------
# Gate on CVBENCH_FORK
# ---------------------------------------------------------------------------
FORK = os.environ.get("CVBENCH_FORK")
if not FORK:
    print("SKIP: CVBENCH_FORK is not set (point it at the CVBench fork checkout, "
          "e.g. CVBENCH_FORK=/path/to/CVBench) — nothing compared.")
    sys.exit(0)
FORK = os.path.abspath(FORK)
if not os.path.isdir(os.path.join(FORK, "bench", "methods")):
    print(f"SKIP: CVBENCH_FORK={FORK} has no bench/methods — not a CVBench checkout.")
    sys.exit(0)

NEW_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- this repo FIRST, and fully imported BEFORE the fork path is added -------
if NEW_REPO in sys.path:
    sys.path.remove(NEW_REPO)
sys.path.insert(0, NEW_REPO)

from models import clients as new_clients                      # noqa: E402
from models.clients import Backend, GenOut                     # noqa: E402
from harnesses.uniform import (                                # noqa: E402
    TemporalWeightedMethod as NewTemporal,
    CVBenchNativeMethod as NewNative,
)
from harnesses.stitched import CentralizedMethod as NewCentralized      # noqa: E402
from harnesses.decentralized import PerStreamMethod as NewPerStream     # noqa: E402

# --- fork path LAST (appended, so this repo stays ahead of <fork> itself; the
# --- bench.reuse import below still front-inserts <fork>/Video-R1/src) -------
sys.path.append(FORK)

from bench.methods.temporal import TemporalWeightedMethod as ForkTemporal   # noqa: E402
from bench.methods.cvbench_native import CVBenchNativeMethod as ForkNative  # noqa: E402
from bench.methods.centralized import CentralizedMethod as ForkCentralized  # noqa: E402
from bench.methods.per_stream import PerStreamMethod as ForkPerStream       # noqa: E402


# ---------------------------------------------------------------------------
# FakeBackend: record messages, answer "(A)", zero token counts.
# Subclasses the NEW repo's Backend; the fork's methods only duck-type
# (.name + .generate) so the same class serves both sides. GenOut signature
# mirrors models.clients.GenOut exactly:
#   GenOut(text, input_tokens, video_tokens, output_tokens, latency_s)
# ---------------------------------------------------------------------------
class FakeBackend(Backend):
    name = "fake"

    def __init__(self):
        self.calls = []

    def generate(self, messages, max_new_tokens, *, seed=None, temperature=0.0) -> GenOut:
        self.calls.append(dict(messages=messages, max_new_tokens=max_new_tokens,
                               seed=seed, temperature=temperature))
        return GenOut(text="(A)", input_tokens=0, video_tokens=0,
                      output_tokens=0, latency_s=0.0)


# ---------------------------------------------------------------------------
# Comparators
# ---------------------------------------------------------------------------
class Divergence(Exception):
    pass


def first_diff_index(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n  # equal prefix; they differ by length


def cmp_text(where, ta, tb, stats):
    if not isinstance(ta, str) or not isinstance(tb, str):
        raise Divergence(f"{where}: non-str text payloads ({type(ta)} vs {type(tb)})")
    if ta != tb:
        i = first_diff_index(ta, tb)
        lo = max(0, i - 70)
        raise Divergence(
            f"{where}: text diverges at char {i} (fork len={len(ta)}, new len={len(tb)})\n"
            f"    fork[{lo}:{i+70}] = {ta[lo:i+70]!r}\n"
            f"    new [{lo}:{i+70}] = {tb[lo:i+70]!r}")
    stats["text_items"] += 1
    stats["text_bytes"] += len(ta.encode("utf-8"))


_VIDEO_HASHES = {}   # (path, nframes) -> (sha256 hex, raw bytes hashed, n frames)


def video_frames_hash(path, nframes):
    """Hash the exact pixels a backend would decode for this (path, nframes):
    decord CPU decode at the InternVL ``get_index`` uniform grid (imported from
    models.clients so the grid formula itself is the production one)."""
    key = (path, int(nframes))
    if key not in _VIDEO_HASHES:
        from decord import VideoReader, cpu
        vr = VideoReader(path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())
        idx = new_clients.get_index(None, fps, max_frame, first_idx=0,
                                    num_segments=int(nframes))
        h = hashlib.sha256()
        nbytes = 0
        for i in idx:
            arr = vr[int(min(max(0, int(i)), max_frame))].asnumpy()
            h.update(arr.tobytes())
            nbytes += arr.nbytes
        _VIDEO_HASHES[key] = (h.hexdigest(), nbytes, len(idx))
    return _VIDEO_HASHES[key]


def cmp_video(where, a, b, stats):
    # every scalar key must match exactly: video path, nframes, and any extras
    # a method may set (e.g. frame_indices)
    for k in sorted(set(a) | set(b)):
        if k == "type":
            continue
        if a.get(k) != b.get(k):
            raise Divergence(f"{where}: video item key {k!r} differs:\n"
                             f"    fork = {a.get(k)!r}\n    new  = {b.get(k)!r}")
    ha = video_frames_hash(a["video"], a["nframes"])
    hb = video_frames_hash(b["video"], b["nframes"])
    if ha[0] != hb[0]:
        raise Divergence(f"{where}: decoded frame pixels differ for {a['video']} "
                         f"@ nframes={a['nframes']} ({ha[0][:16]} vs {hb[0][:16]})")
    stats["video_items"] += 1
    stats["video_frames"] += ha[2]
    stats["video_bytes"] += ha[1]


def cmp_image(where, a, b, stats):
    ia, ib = a["image"], b["image"]
    if ia.mode != ib.mode or ia.size != ib.size:
        raise Divergence(f"{where}: montage mode/size differ: "
                         f"fork {ia.mode}{ia.size} vs new {ib.mode}{ib.size}")
    ba, bb = ia.tobytes(), ib.tobytes()
    if ba != bb:
        i = first_diff_index(ba, bb)
        bpp = len(ia.mode)  # RGB -> 3 bytes/pixel
        px = i // bpp
        x, y = px % ia.size[0], px // ia.size[0]
        raise Divergence(
            f"{where}: montage pixels diverge at byte {i} ≈ pixel ({x},{y}) of "
            f"{ia.size}: fork {ba[i:i+6].hex()} vs new {bb[i:i+6].hex()}")
    stats["image_items"] += 1
    stats["image_bytes"] += len(ba)


def cmp_item(where, a, b, stats):
    if a.get("type") != b.get("type"):
        raise Divergence(f"{where}: item type differs: fork={a.get('type')!r} "
                         f"new={b.get('type')!r}")
    t = a["type"]
    if t == "text":
        cmp_text(where, a["text"], b["text"], stats)
    elif t == "video":
        cmp_video(where, a, b, stats)
    elif t == "image":
        cmp_image(where, a, b, stats)
    else:
        raise Divergence(f"{where}: unknown content item type {t!r}")


def cmp_calls(calls_f, calls_n, stats):
    if len(calls_f) != len(calls_n):
        raise Divergence(f"generate() call counts differ: fork={len(calls_f)} "
                         f"new={len(calls_n)}")
    for ci, (cf, cn) in enumerate(zip(calls_f, calls_n)):
        where0 = f"call {ci}"
        for k in ("max_new_tokens", "seed", "temperature"):
            if cf[k] != cn[k]:
                raise Divergence(f"{where0}: kwarg {k} differs: fork={cf[k]!r} "
                                 f"new={cn[k]!r}")
        mf, mn = cf["messages"], cn["messages"]
        if len(mf) != len(mn):
            raise Divergence(f"{where0}: message counts differ: {len(mf)} vs {len(mn)}")
        for mi, (msgf, msgn) in enumerate(zip(mf, mn)):
            if msgf.get("role") != msgn.get("role"):
                raise Divergence(f"{where0} msg {mi}: role differs: "
                                 f"{msgf.get('role')!r} vs {msgn.get('role')!r}")
            cfl, cnl = msgf["content"], msgn["content"]
            if len(cfl) != len(cnl):
                types_f = [it.get("type") for it in cfl]
                types_n = [it.get("type") for it in cnl]
                raise Divergence(f"{where0} msg {mi}: content lengths differ: "
                                 f"fork {types_f} vs new {types_n}")
            for ii, (a, b) in enumerate(zip(cfl, cnl)):
                cmp_item(f"{where0} msg {mi} item {ii} ({a.get('type')})", a, b, stats)
        stats["calls"] += 1


def cmp_results(where, rf, rn, stats):
    """Result-level equivalence: frame_alloc (where set) + scoring surface."""
    if rf.error is not None or rn.error is not None:
        raise Divergence(f"{where}: harness swallowed an error: "
                         f"fork={rf.error!r} new={rn.error!r}")
    for k in ("method", "prediction", "gold", "abstained", "num_model_calls"):
        vf, vn = getattr(rf, k), getattr(rn, k)
        if vf != vn:
            raise Divergence(f"{where}: Result.{k} differs: fork={vf!r} new={vn!r}")
    if rf.frame_alloc is not None or rn.frame_alloc is not None:
        if rf.frame_alloc != rn.frame_alloc:
            raise Divergence(f"{where}: frame_alloc differs:\n"
                             f"    fork = {rf.frame_alloc!r}\n"
                             f"    new  = {rn.frame_alloc!r}")
        stats["frame_alloc"] += 1


# ---------------------------------------------------------------------------
# Pairs (constructor args mirror bench/run_bench.py make_method + its argparse
# defaults: nframes=8, max_new_tokens=8192, temperature=0.7, budget=64,
# floor=2). SEED pins the answer(..., seed=) arg identically on both sides
# (frames are deterministic; the seed only reaches the recorded kwargs).
# ---------------------------------------------------------------------------
COMMON = dict(nframes=8, max_new_tokens=8192, temperature=0.7)
BUDGET, FLOOR = 64, 2
SEED = 1

# montage_kind='camera' is the make_method/argparse default; 'video' is the
# corrected CVBench framing used in production runs — both compared. Same for
# per_stream's stream_kind (the task's required camera AND video arms).
PAIRS = [
    ("temporal_weighted",
     lambda be: ForkTemporal(be, budget=BUDGET, floor=FLOOR, weighting="duration", **COMMON),
     lambda be: NewTemporal(be, budget=BUDGET, floor=FLOOR, weighting="duration", **COMMON)),
    ("temporal_even",
     lambda be: ForkTemporal(be, budget=BUDGET, floor=FLOOR, weighting="even", **COMMON),
     lambda be: NewTemporal(be, budget=BUDGET, floor=FLOOR, weighting="even", **COMMON)),
    ("cvbench_native",
     lambda be: ForkNative(be, **COMMON),
     lambda be: NewNative(be, **COMMON)),
    ("centralized[camera]",
     lambda be: ForkCentralized(be, montage_frames=0, cell_px=448, montage_kind="camera", **COMMON),
     lambda be: NewCentralized(be, montage_frames=0, cell_px=448, montage_kind="camera", **COMMON)),
    ("centralized[video]",
     lambda be: ForkCentralized(be, montage_frames=0, cell_px=448, montage_kind="video", **COMMON),
     lambda be: NewCentralized(be, montage_frames=0, cell_px=448, montage_kind="video", **COMMON)),
    ("per_stream[camera]",
     lambda be: ForkPerStream(be, stream_kind="camera", **COMMON),
     lambda be: NewPerStream(be, stream_kind="camera", **COMMON)),
    ("per_stream[video]",
     lambda be: ForkPerStream(be, stream_kind="video", **COMMON),
     lambda be: NewPerStream(be, stream_kind="video", **COMMON)),
    # clip_select (SummarySelectMethod / ClipScoreSelectMethod): SKIPPED — needs
    # CLIP/SigLIP weights + the per-clip summary cache. TODO: add behind --slow.
]


def pick_records(subset_path, video_root, n=3):
    data = json.load(open(subset_path))
    out = []
    for rec in data:
        vids = [rec.get(f"video_{i}") for i in range(1, 5)]
        vids = [v for v in vids if v]
        if vids and all(os.path.exists(os.path.join(video_root, v)) for v in vids):
            out.append(rec)
            if len(out) == n:
                break
    return out


def run_pair(name, fork_factory, new_factory, records, video_root):
    stats = defaultdict(int)
    be_f, be_n = FakeBackend(), FakeBackend()
    m_f, m_n = fork_factory(be_f), new_factory(be_n)
    for rec in records:
        rf = m_f.answer(rec, video_root, seed=SEED)
        rn = m_n.answer(rec, video_root, seed=SEED)
        cmp_results(f"[{name}] rec {rec['id']}", rf, rn, stats)
    # calls accumulate across records on each side, in the same record order
    try:
        cmp_calls(be_f.calls, be_n.calls, stats)
    except Divergence as e:
        raise Divergence(f"[{name}] {e}") from None
    return stats


def main():
    subset = os.path.join(NEW_REPO, "data", "subsets", "cvbench_full_k4_subset.json")
    video_root = os.path.join(FORK, "Video-R1", "src", "r1-v", "Evaluation", "CVBench")
    records = pick_records(subset, video_root, n=3)
    if len(records) < 3:
        print(f"SKIP: only {len(records)} record(s) of {subset} have all videos "
              f"under {video_root} — need 3.")
        sys.exit(0)
    ids = [r["id"] for r in records]
    print(f"fork      = {FORK} (pkg bench.*)")
    print(f"new repo  = {NEW_REPO} (top-level modules)")
    print(f"subset    = {subset}")
    print(f"video_root= {video_root}")
    print(f"records   = {ids}  seed={SEED}  args={COMMON} budget={BUDGET} floor={FLOOR}\n")

    failures = []
    for name, ff, nf in PAIRS:
        try:
            s = run_pair(name, ff, nf, records, video_root)
            print(f"PASS {name:22s} calls={s['calls']} "
                  f"text={s['text_items']} items/{s['text_bytes']:,}B "
                  f"video={s['video_items']} items/{s['video_frames']} frames/"
                  f"{s['video_bytes']:,}B "
                  f"image={s['image_items']} items/{s['image_bytes']:,}B "
                  f"frame_alloc={s['frame_alloc']}")
        except Divergence as e:
            failures.append((name, str(e)))
            print(f"FAIL {name}\n  {e}")
        except Exception as e:  # setup/import/decode errors are failures too
            failures.append((name, f"{type(e).__name__}: {e}"))
            print(f"FAIL {name} (unexpected {type(e).__name__}: {e})")

    print()
    if failures:
        print(f"{len(failures)}/{len(PAIRS)} pair(s) DIVERGED — the port has drifted "
              f"from the fork.")
        sys.exit(1)
    print(f"all {len(PAIRS)} pairs byte/pixel-identical across {len(records)} records "
          f"({ids}).")


if __name__ == "__main__":
    main()
