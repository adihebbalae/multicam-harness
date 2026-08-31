#!/usr/bin/env python3
# Ported from Wavy-Hec/CVBench hosting/remux_avi.py @ 4d8f5b2605e45d0860454886708fa8ff06e61840
# Deliberate delta vs source: --root defaults to this repo's release layout
# (data/crossview-release-annotations/...), i.e. where scripts/data/fetch_meva_videos.py
# writes the .avi clips, instead of the fork's own release path.
"""Remux the MEVA .avi clips to verified .mp4 siblings so the harness decodes
them correctly.

Why: decord (0.6.0 — the reader behind every arm on the InternVL backend, the
selection arms' candidate decode, and qwen_vl_utils whenever decord is
installed) returns the WRONG frame on random access into these AVI
containers. The AVI packets carry no pts (B-frame H.264, dts only), so every
seek lands on keyframe 0 and decodes forward: ``vr[i]`` returns sequential
frame ``i - <last keyframe before i>``, which on the release's usual 60-frame
keyframe grid is ``i mod 60`` (clips with an irregular grid are wrong in the
same way with different numbers). Verified on the release clips (a full
per-frame hash table vs random access): only the first seconds of every
5-minute clip are ever decoded, on every arm that looks at pixels — so any
run made against a bare .avi is invalid and must never be pooled with a
post-fix run. Rewriting the AVI index (``-c copy`` back to .avi) does NOT
help; a stream-copy remux to .mp4 does.

Guarantee of the .mp4 sibling (each point verified per clip below): the
H.264 elementary stream is byte-identical to the AVI's (no re-encode), so a
SEQUENTIAL decode is pixel-identical at every index. Random access is
correct to within the H.264 reorder depth: the AVI has no pts, so the
stream copy inherits pts = dts and decord's seek can land up to
``has_b_frames + 1`` (3) frames early on seeks after the first — 100 ms at
30 fps, immaterial for frames sampled seconds apart, but not "frame-exact"
(``-fflags +genpts`` only halves it).

What this does: for every ``*.avi`` under --root (default: the release's
``videos/meva``), write ``<same name>.mp4`` next to it via ``ffmpeg -c copy``,
then VERIFY it against the .avi before keeping it:
  * same frame count and fps (+-0.01);
  * the raw H.264 elementary streams of the two containers hash identically
    (whole file, no decode — rules out a foreign or partial copy);
  * sequential decode of the .mp4 == sequential decode of the .avi at several
    indices in the first SEQ_UPTO frames (exact hash match);
  * harness-style random access (one fresh reader, ``get_batch`` of spread
    indices past the first keyframe interval) lands within TOL frames of the
    asked index — this is what fails on the broken .avi (it returns an index
    < 60) and what bounds the reorder jitter on the .mp4.
An existing sibling is re-verified by default (``--skip-existing`` trusts
it); one that fails is moved aside to ``<name>.mp4.bad`` and regenerated. A
clip that cannot be verified keeps no .mp4 and is listed at the end with a
non-zero exit, so ``video_paths`` refuses it loudly instead of falling back
to the broken .avi. Originals are never modified or deleted.

Cost: the tree doubles (the .mp4 is ~99.7% of the .avi size), and each clip
takes seconds to tens of seconds on a network filesystem (verification
dominates); budget hours, not minutes, for the full release at --workers 6.
Run one instance at a time per root.

``video_paths`` (inprocess/dataloaders/qa_json.py) prefers the .mp4 sibling
automatically and refuses a bare .avi. Run ``--check`` (exit 0) before
submitting any MEVA leg.

Usage (the cvbench env has ffmpeg + decord), from the repo root:
  conda run -n cvbench python scripts/data/remux_avi.py            # remux + verify all
  conda run -n cvbench python scripts/data/remux_avi.py --check    # verify only, write nothing
  conda run -n cvbench python scripts/data/remux_avi.py --workers 8 --report remux.jsonl
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DEFAULT_ROOT = os.path.join(REPO, "data", "crossview-release-annotations",
                            "crossview-release", "videos", "meva")
# Verification window: sequential decode of both containers up to SEQ_UPTO
# frames (~20 s at 30 fps; well past the 60-frame keyframe grid, cheap to
# decode), exact-hash comparisons at SEQ_PROBES, and random-access probes at
# RA_PROBES with a tolerance of TOL frames (H.264 reorder depth 2, plus one).
SEQ_UPTO = 600
SEQ_PROBES = (0, 61, 200, 400, 600)
RA_PROBES = (150, 450, 590)
TOL = 3
STALE_TMP_AGE_S = 3600          # a live worker rewrites its tmp continuously
_PTR_RE = re.compile(r"^\[\w+ @ 0x[0-9a-f]+\] ")


def ffmpeg_bin():
    """The env's own ffmpeg first (documented invocation), then PATH."""
    for c in (os.path.join(sys.prefix, "bin", "ffmpeg"), shutil.which("ffmpeg")):
        if c and os.path.exists(c):
            return c
    raise SystemExit("ffmpeg not found - run under the cvbench conda env")


def _es_md5(ffmpeg, path):
    """md5 of the raw H.264 elementary stream (Annex B) — container-free."""
    cmd = [ffmpeg, "-v", "error", "-i", path, "-map", "0:v:0", "-c", "copy"]
    if not path.lower().endswith(".avi"):
        cmd += ["-bsf:v", "h264_mp4toannexb"]
    cmd += ["-f", "h264", "-"]
    p = subprocess.run(cmd, capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg es dump failed: {p.stderr.decode(errors='replace')[:200]}")
    return hashlib.md5(p.stdout).hexdigest()


def _seq_hashes(path, upto):
    """md5 of each sequentially decoded frame 0..upto (inclusive)."""
    from decord import VideoReader, cpu
    vr = VideoReader(path, ctx=cpu(0), num_threads=1)
    out = []
    for _ in range(min(upto, len(vr) - 1) + 1):
        out.append(hashlib.md5(vr.next().asnumpy().tobytes()).hexdigest())
    return out


def verify(avi, mp4, ffmpeg):
    """(reason, max_offset): reason is None when ``mp4`` is a verified copy of
    ``avi``; max_offset is the largest |random-access index error| seen."""
    from decord import VideoReader, cpu
    va = VideoReader(avi, ctx=cpu(0), num_threads=1)
    vm = VideoReader(mp4, ctx=cpu(0), num_threads=1)
    n = len(vm)
    if n != len(va):
        return f"frame count {n} != avi {len(va)}", None
    if abs(vm.get_avg_fps() - va.get_avg_fps()) > 0.01:
        return f"fps {vm.get_avg_fps():.3f} != avi {va.get_avg_fps():.3f}", None
    if n < 2:
        return "fewer than 2 frames", None
    if _es_md5(ffmpeg, avi) != _es_md5(ffmpeg, mp4):
        return "H.264 elementary stream differs from the avi (foreign or partial copy)", None
    upto = min(SEQ_UPTO, n - 1)
    ha = _seq_hashes(avi, upto)
    hm = _seq_hashes(mp4, upto)
    if len(ha) != len(hm):
        return f"sequential decode lengths differ ({len(ha)} avi vs {len(hm)} mp4)", None
    for i in SEQ_PROBES:
        i = min(i, upto)
        if ha[i] != hm[i]:
            return f"sequential frame {i} differs between avi and mp4", None
    # harness pattern: one fresh reader, several seeks in one get_batch
    where = {h: i for i, h in enumerate(hm)}
    idx = sorted({min(i, upto) for i in RA_PROBES})
    frames = VideoReader(mp4, ctx=cpu(0), num_threads=1).get_batch(idx).asnumpy()
    worst = 0
    for k, i in enumerate(idx):
        j = where.get(hashlib.md5(frames[k].tobytes()).hexdigest())
        if j is None:
            return f"random-access frame {i} is not among sequential frames 0..{upto}", None
        worst = max(worst, abs(j - i))
        if abs(j - i) > TOL:
            return f"random-access frame {i} decoded as frame {j} (wrap/seek defect)", None
    return None, worst


def _remux(avi, mp4, ffmpeg):
    """(status, info) after writing + verifying a fresh sibling."""
    tmp = f"{mp4}.{os.getpid()}.tmp.mp4"   # .mp4 suffix so ffmpeg picks the container
    cmd = [ffmpeg, "-v", "warning", "-y", "-i", avi, "-c", "copy", tmp]
    p = subprocess.run(cmd, capture_output=True, text=True)
    # the expected diagnostic: the AVI packets have no pts, so the muxer
    # synthesizes pts = dts (the source of the bounded seek jitter)
    warn = [_PTR_RE.sub("", ln.strip()) for ln in p.stderr.splitlines() if ln.strip()]
    info = {"ffmpeg_warnings": len(warn), "first_warning": warn[0][:120] if warn else None}
    try:
        if p.returncode != 0:
            return "ffmpeg-failed: " + p.stderr.strip()[:200], info
        try:
            r, off = verify(avi, tmp, ffmpeg)
        except Exception as e:  # noqa: BLE001 — report, never crash the pool
            r, off = f"verify raised {type(e).__name__}: {e}", None
        if r is not None:
            return "verify-failed: " + r, info
        os.replace(tmp, mp4)
        info["max_offset"] = off
        return "ok", info
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def process(avi, ffmpeg, check_only=False, force=False, skip_existing=False):
    """(avi, status, info). status: ok | ok-existing | exists | missing |
    requarantined+ok | ffmpeg-failed: .. | verify-failed: .. | bad-existing: .."""
    mp4 = avi[:-4] + ".mp4"
    try:
        if os.path.exists(mp4) and (check_only or not force):
            if skip_existing and not check_only:
                return avi, "exists", {}
            try:
                r, off = verify(avi, mp4, ffmpeg)
            except Exception as e:  # noqa: BLE001
                r, off = f"verify raised {type(e).__name__}: {e}", None
            if r is None:
                return avi, ("ok" if check_only else "ok-existing"), {"max_offset": off}
            if check_only:
                return avi, "bad-existing: " + r + " (video_paths WOULD decode it)", {}
            # a wrong sibling must not stay where video_paths will pick it up
            os.replace(mp4, mp4 + ".bad")
            status, info = _remux(avi, mp4, ffmpeg)
            info["quarantined"] = "bad-existing: " + r
            return avi, ("requarantined+ok" if status == "ok" else status), info
        if check_only:
            return avi, "missing", {}
        status, info = _remux(avi, mp4, ffmpeg)
        return avi, status, info
    except Exception as e:  # noqa: BLE001 — never take the pool down
        return avi, f"error: {type(e).__name__}: {e}", {}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--check", action="store_true",
                    help="verify existing .mp4 siblings only; write nothing "
                         "(--force/--skip-existing are ignored)")
    ap.add_argument("--force", action="store_true",
                    help="re-remux even when the .mp4 sibling exists")
    ap.add_argument("--skip-existing", action="store_true",
                    help="trust existing siblings instead of re-verifying them")
    ap.add_argument("--limit", type=int, default=0, help="first N clips only")
    ap.add_argument("--subset", default=None, metavar="SUBSET_JSON",
                    help="only the .avi clips referenced by this eval subset "
                         "(video_1..video_13 relative paths under the release root)")
    ap.add_argument("--report", default=None,
                    help="write one {avi, status, ...} json line per clip here")
    args = ap.parse_args()
    if args.force and args.skip_existing:
        ap.error("--force and --skip-existing are mutually exclusive")

    now = time.time()
    avis, stale = [], []
    for d, _, fs in os.walk(args.root):
        for f in fs:
            if f.lower().endswith(".avi"):
                avis.append(os.path.join(d, f))
            elif f.endswith(".tmp.mp4"):
                p = os.path.join(d, f)
                if now - os.path.getmtime(p) > STALE_TMP_AGE_S:
                    stale.append(p)          # an interrupted run's leftover
    avis.sort()
    if args.subset:
        recs = json.load(open(args.subset))
        want = {os.path.basename(v) for r in recs for i in range(1, 14)
                for v in [r.get(f"video_{i}")] if v and v.lower().endswith(".avi")}
        avis = [a for a in avis if os.path.basename(a) in want]
        print(f"--subset {args.subset}: {len(want)} referenced .avi, {len(avis)} found under root",
              flush=True)
    if args.limit:
        avis = avis[:args.limit]
    if not avis:
        raise SystemExit(f"no .avi under {args.root}")
    if stale and not args.check:
        for s in stale:
            os.remove(s)
    ffmpeg = ffmpeg_bin()
    print(f"{len(avis)} .avi under {args.root} "
          f"({'check' if args.check else 'remux+verify'}, workers={args.workers}, "
          f"ffmpeg={ffmpeg}, stale tmp removed={len(stale) if not args.check else 0})",
          flush=True)

    counts = {}
    bad = []
    worst = 0
    rep = open(args.report, "w") if args.report else None
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process, a, ffmpeg, args.check, args.force, args.skip_existing): a
                for a in avis}
        for k, fut in enumerate(as_completed(futs), 1):
            try:
                avi, status, info = fut.result()
            except Exception as e:  # noqa: BLE001
                avi, status, info = futs[fut], f"error: {type(e).__name__}: {e}", {}
            key = status.split(":")[0]
            counts[key] = counts.get(key, 0) + 1
            if key not in ("ok", "ok-existing", "exists", "requarantined+ok"):
                bad.append((avi, status))
            if info.get("max_offset") is not None:
                worst = max(worst, info["max_offset"])
            if rep:
                rep.write(json.dumps({"avi": os.path.relpath(avi, args.root),
                                      "status": status, **info}) + "\n")
                rep.flush()
            if k % 100 == 0 or k == len(avis):
                print(f"  {k}/{len(avis)} {counts}", flush=True)
    if rep:
        rep.close()
    print(f"done: {counts}; worst random-access offset seen: {worst} frame(s) (tol {TOL})")
    if bad:
        print(f"{len(bad)} clip(s) without a verified .mp4 - video_paths will "
              "refuse a missing sibling and DECODE a bad-existing one:")
        for avi, status in bad[:50]:
            print(f"  {os.path.relpath(avi, args.root)}: {status}")
        sys.exit(1)


if __name__ == "__main__":
    main()
