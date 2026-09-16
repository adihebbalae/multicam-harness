#!/usr/bin/env python3
"""Demo figures for the four-camera MEVA demo: what the selector looked at, what
the model answered, and how the demo set sits against the meva1033 ladder.

For every question in ``analysis/demo_meva4cam_subset.json`` this writes, into
``analysis/figs_demo/`` (gitignored):

  q<id>_selection.png   one row per camera slot: a clip-duration timeline with
                        the 8 scored segments as a heat strip (SigLIP or ViCLIP,
                        whichever leg --selection-method names; the scorer and
                        the query mode are read off that row's frame_alloc and
                        printed, never inferred from the flag), the kept
                        segments outlined in bold (a second dashed outline for a
                        ``--seg-select global`` leg when one is present), the
                        selected frame times as ticks, the release's annotated
                        event windows for that camera as labelled bars, and the
                        kept frames themselves as thumbnails at their times.
                        A global leg also gets its own tick row and its own
                        per-camera frame count, because its whole point is an
                        unequal budget across cameras.
  q<id>_answer.png      the question, its options with the gold one marked, a
                        per-arm table (prediction, correct/wrong, latency,
                        tokens, protocol) and a trace excerpt taken from a
                        SIGHTED arm only - the blind arm's chain of thought is
                        never passed off as reasoning over the frames.
  summary.png           the demo set as arms x questions (each cell carries the
                        pass-1 prediction and the passes-correct fraction
                        behind it), beside the meva1033 reference ladder.
  manifest.json         everything drawn, so the narration and the slides can
                        cite it instead of re-deriving it.

Read-only: it never loads a model, never submits anything and never touches
``bench/``. Run from the repo root under the ``cvbench`` conda env (matplotlib +
PIL + decord):

  python3 analysis/render_demo.py
  python3 analysis/render_demo.py --rows 'bench/results/bench_demo_*_internvl_demo_*.jsonl'
  python3 analysis/render_demo.py --ids 860 251 --thumbs-per-cam 8
  python3 analysis/render_demo.py --selection-method segment_select_viclip_opt \
      --rows 'bench/results/bench_demo_*_internvl_demo_*direct.jsonl' \
      --out analysis/figs_demo_viclip

MEDIA. MEVA records spell their clips ``.avi`` and an ``.avi`` must never reach
a decoder (decord returns the wrong frame on random access into those
containers). Resolution goes through ``bench.reuse.video_paths`` when that
imports, and through a byte-identical local replica otherwise; either way a
missing ``.mp4`` sibling is a hard error naming the file, never a silent
fallback to the ``.avi``.

ARMS are identified per result row, not per file: the method field picks the
arm - including which scorer a segment_select row used, so a SigLIP and a
ViCLIP leg are two arms and not one - the tag parsed out of the filename
distinguishes legs of the same arm (per-clip vs --seg-select global), and
the protocol (reasoning vs direct answer) is read from the rows themselves
(a non-empty ``think``, or ``<think>`` in ``response_text``), because both modes
share the ``<answer>`` parser and neither backend has a thinking toggle.
"""
from __future__ import annotations

import argparse
import glob as _glob
import json
import os
import re
import sys
import textwrap

# No model is loaded here, but bench.reuse pulls torch in transitively. Mask the
# GPUs and pin HF offline before any of that can import, exactly as
# bench/make_input_examples.py does.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import Normalize, to_rgb  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RELEASE = os.path.join(REPO, "crossview-release-annotations", "crossview-release")
ANN_DIR = os.path.join(RELEASE, "annotations", "multi-cam-dataset", "meva")

DEFAULT_SUBSET = "analysis/demo_meva4cam_subset.json"
DEFAULT_OUT = "analysis/figs_demo"

# Which segment_select leg the selection figure, the manifest's cameras[] block
# and its selection_leg/global_leg entries are drawn from. Rows of both scorers
# can be passed at once: the other one still shows up as an arm in the answer
# panel and the summary grid, it just does not drive the timelines.
DEFAULT_SELECTION_METHOD = "segment_select_siglip"
VICLIP_SELECTION_METHOD = "segment_select_viclip_opt"

# The demo run's own rows, once the interactive leg has landed.
DEFAULT_ROWS = [
    "bench/results/bench_demo_meva4cam_subset_internvl_demo_reason.jsonl",
    "bench/results/bench_demo_meva4cam_subset_internvl_demo_direct.jsonl",
    # the global leg, widened to a glob: it has been tagged both demo_global and
    # demo_global_direct, and the arm is identified from the row's
    # frame_alloc.segment_select_mode either way
    "bench/results/bench_demo_meva4cam_subset_internvl_demo_global*.jsonl",
]
# Until then: the media-valid InternVL3-8B legs on the meva1033 pool, which
# carry all five demo ids (4 passes each).
FALLBACK_ROWS = [
    "bench/results/rescored/bench_crossview_meva1033_subset_internvl_mp4sg96_shard*.jsonl",
    "bench/results/rescored/bench_crossview_meva1033_subset_internvl_mp4fs96_shard*.jsonl",
    "bench/results/rescored/bench_crossview_meva1033_subset_internvl_mp4bd_shard*.jsonl",
]

# meva1033 reference ladder (docs/plan_2026-09-16_demo_feasibility.md, and the
# bounds/ladder numbers behind it). label, value, kind.
LADDER = [
    ("Chance (4 options)", 25.00, "floor"),
    ("Model, no video (reasoning, T=0.7)", 34.75, "arm"),
    ("All cameras, 32 frames", 42.88, "arm"),
    ("All cameras, 96 frames", 43.85, "arm"),
    ("SigLIP segments, 96 frames", 49.10, "arm"),
    ("Text-only letter floor", 54.70, "floor"),
]
# The ViCLIP-opt rungs of the same bounds table (raw accuracy, meva1033,
# InternVL3-8B; docs/performance_bounds_2026-09-10.md section 4). They join the
# ladder only when a ViCLIP arm is actually on the figure, so a SigLIP-only
# render keeps the six-rung ladder - and the manifest block - it had before a
# second scorer existed.
LADDER_VICLIP = [
    ("ViCLIP-opt segments, 32 frames", 43.56, "arm"),
    ("ViCLIP-opt segments, 64 frames", 47.97, "arm"),
    ("ViCLIP-opt segments, 96 frames", 48.31, "arm"),
]
CAVEAT = ("Demo questions were chosen where selection helps or fails visibly; "
          "the ladder is the full 1,033-question pool.")


def ladder_for(questions):
    """The reference ladder the summary draws, sorted by value: the base rungs,
    plus the ViCLIP-opt rungs when a ViCLIP arm was rendered."""
    has_viclip = any(g["arm"] in ("viclip_perclip", "viclip_global")
                     for q in questions for g in q["groups"])
    if not has_viclip:
        return list(LADDER)
    return sorted(LADDER + LADDER_VICLIP, key=lambda r: r[1])


# short column headers: "event_ordering" is wider than a summary cell
TYPE_SHORT = {"event_ordering": "ordering", "temporal": "temporal",
              "spatial": "spatial", "counting": "counting"}

QA_FILES = {
    "temporal": "qa_temporal.json",
    "event_ordering": "qa_event_ordering.json",
    "spatial": "qa_spatial.json",
}

# Okabe-Ito accents: readable next to viridis and safe for every common CVD.
C_KEPT = "#111111"      # kept-segment outline (per-clip)
C_GLOBAL = "#0072B2"    # kept-segment outline (--seg-select global)
C_EVENT = "#D55E00"     # annotated event window
C_TICK = "#222222"      # selected frame time
C_GOLD = "#F0E442"      # gold option highlight
C_OK = "#009E73"
C_BAD = "#CC3311"
C_INK = "#1a1a1a"
C_MUTE = "#5a5a5a"
C_PANEL = "#f2f2f0"

FS_TINY = 11
# the selection page is the widest/tallest one and gets downscaled hardest for
# the recording, so its smallest type is a notch bigger than everywhere else
FS_TINY_SEL = 13
FS_BODY = 12
FS_ROW = 13
FS_HEAD = 15
FS_TITLE = 19

MARK_OK = "✓"       # check
MARK_BAD = "✗"      # ballot x
MARK_STAR = "★"     # in-window thumbnail
ARROW = "←"

NOTES: list[str] = []


def note(msg: str) -> None:
    """A degradation the figures should own up to: printed and put in the manifest."""
    line = f"[note] {msg}"
    print(line)
    NOTES.append(msg)


def rel(path: str) -> str:
    """Resolve a path against the repo root so the script runs from anywhere."""
    return path if os.path.isabs(path) else os.path.join(REPO, path)


# --------------------------------------------------------------------------
# media resolution (.avi -> verified .mp4 sibling)
# --------------------------------------------------------------------------

MAX_SLOTS_FALLBACK = 16


def _local_video_paths(rec, video_root):
    """Replica of bench.reuse.video_paths' resolution: a record's .avi resolves
    to the remuxed .mp4 sibling, and a missing sibling is a hard error. Used
    only when bench.reuse cannot be imported."""
    out = []
    for i in range(1, MAX_SLOTS_FALLBACK + 1):
        v = rec.get(f"video_{i}")
        if not v:
            continue
        p = os.path.normpath(os.path.join(video_root, v))
        if p.lower().endswith(".avi"):
            mp4 = p[:-4] + ".mp4"
            if not os.path.exists(mp4):
                raise FileNotFoundError(
                    f"{p}: refusing to decode an .avi - decord returns wrong "
                    f"frames on random access into these containers. The "
                    f"remuxed sibling is missing: {mp4} "
                    f"(hosting/remux_avi.py writes it).")
            p = mp4
        out.append(p)
    return out


def get_video_paths_fn(use_reuse=True):
    if use_reuse:
        try:
            if REPO not in sys.path:
                sys.path.insert(0, REPO)
            from bench.reuse import video_paths as _vp  # noqa: WPS433
            return _vp, "bench.reuse.video_paths"
        except Exception as exc:  # pragma: no cover - env dependent
            note(f"bench.reuse did not import ({exc.__class__.__name__}: {exc}); "
                 f"using the local .avi->.mp4 replica")
    return _local_video_paths, "local .avi->.mp4 replica"


def cam_id_of(path: str) -> str:
    """Camera id = third-from-last dot field of the filename
    (...school.G336.r13.mp4 -> G336)."""
    parts = os.path.basename(path).split(".")
    return parts[-3] if len(parts) >= 3 else "?"


# --------------------------------------------------------------------------
# release annotations
# --------------------------------------------------------------------------

_ANN_CACHE: dict[str, list] = {}


def load_annotations(qtype):
    fn = QA_FILES.get(qtype)
    if not fn:
        return None
    if qtype not in _ANN_CACHE:
        path = os.path.join(ANN_DIR, fn)
        if not os.path.exists(path):
            note(f"release annotations missing: {path}")
            _ANN_CACHE[qtype] = []
        else:
            with open(path) as fh:
                _ANN_CACHE[qtype] = json.load(fh)
    return _ANN_CACHE[qtype]


def release_item(rec):
    """The release QA item a harness record came from: qa_<type>.json indexed by
    the integer after '#' in orig_id."""
    qtype = rec.get("question_type")
    orig = rec.get("orig_id") or ""
    if "#" not in orig:
        note(f"id {rec.get('id')}: no orig_id index; no annotated windows")
        return None
    slot, _, idx = orig.partition("#")
    items = load_annotations(qtype)
    if not items:
        return None
    try:
        idx = int(idx)
    except ValueError:
        note(f"id {rec.get('id')}: orig_id '{orig}' has no integer index")
        return None
    if not (0 <= idx < len(items)):
        note(f"id {rec.get('id')}: orig_id index {idx} out of range for {qtype}")
        return None
    item = items[idx]
    got = (item.get("metadata") or {}).get("slot")
    if got and slot and got != slot:
        note(f"id {rec.get('id')}: orig_id slot '{slot}' != annotation slot '{got}'")
    return item


_TS_RE = re.compile(r"([0-9.]+)\s*-\s*([0-9.]+)\s*s")


def _dbg_end(dbg):
    if not isinstance(dbg, dict):
        return None
    m = _TS_RE.search(str(dbg.get("timestamp") or ""))
    return float(m.group(2)) if m else None


def event_windows(item, qid):
    """Annotated evidence windows from metadata.verification, clip-relative to
    the camera that recorded them. event_a/event_b for temporal, ordered_events
    for event ordering. ordered_events carry no end_sec, so the end comes from
    debug_info's timestamp when it lines up and is estimated otherwise."""
    if not item:
        return []
    md = item.get("metadata") or {}
    ver = md.get("verification") or {}
    dbg = md.get("debug_info") or {}
    out = []

    def mk(ev, label, dbg_ev):
        if not isinstance(ev, dict) or ev.get("camera") is None:
            return None
        start = ev.get("start_sec")
        if start is None:
            return None
        end = ev.get("end_sec")
        est = False
        if end is None:
            end = _dbg_end(dbg_ev)
        if end is None:
            end = float(start) + 2.0
            est = True
        return {
            "label": label,
            "camera": str(ev.get("camera")),
            "start_sec": float(start),
            "end_sec": float(end),
            "end_estimated": est,
            "activity": ev.get("activity"),
            "description": ev.get("description") or ev.get("activity") or "",
        }

    if isinstance(ver.get("ordered_events"), list):
        dbg_events = dbg.get("events") if isinstance(dbg.get("events"), list) else []
        for i, ev in enumerate(ver["ordered_events"]):
            d = dbg_events[i] if i < len(dbg_events) else None
            if isinstance(d, dict) and ev.get("camera") and d.get("camera") != ev.get("camera"):
                d = None
            w = mk(ev, f"#{i + 1}", d)
            if w:
                out.append(w)
    else:
        for key, label in (("event_a", "A"), ("event_b", "B")):
            w = mk(ver.get(key), label, dbg.get(key))
            if w:
                out.append(w)
    if not out:
        note(f"id {qid}: no annotated event windows in the release verification block")
    return out


def requires_cameras(item):
    if not item:
        return None
    reqs = (item.get("metadata") or {}).get("requires_cameras")
    return [str(c) for c in reqs] if isinstance(reqs, list) else None


# --------------------------------------------------------------------------
# result rows
# --------------------------------------------------------------------------

_TAG_RE = re.compile(r"_(internvl|cvbench)(.*)$")


def tag_of(path):
    """Leg tag from the result filename: run_bench.sbatch builds
    bench_<subset>_<ENV><TAG>[_shardN].jsonl, so the tag is whatever follows the
    env key, with the shard suffix stripped."""
    stem = os.path.basename(path)
    for suf in (".jsonl", ".json"):
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
    stem = re.sub(r"_shard\d+$", "", stem)
    m = _TAG_RE.search(stem)
    tag = m.group(2) if m else stem
    return tag.lstrip("_") or (m.group(1) if m else stem)


def expand_rows(patterns):
    paths = []
    for pat in patterns:
        p = rel(pat)
        hits = sorted(_glob.glob(p))
        if hits:
            paths.extend(hits)
        elif os.path.exists(p):
            paths.append(p)
    seen, out = set(), []
    for p in paths:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def load_rows(paths, ids):
    rows = []
    for p in paths:
        n = 0
        with open(p) as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    note(f"{os.path.basename(p)}:{lineno}: malformed JSON line, skipped")
                    continue
                if ids and r.get("id") not in ids:
                    continue
                r["_tag"] = tag_of(p)
                r["_file"] = p
                r["_rescored"] = os.sep + "rescored" + os.sep in p
                rows.append(r)
                n += 1
        print(f"  {os.path.relpath(p, REPO)}: {n} matching rows (tag {tag_of(p)})")
    return rows


ARM_ORDER = ["blind", "native", "seg_perclip", "seg_global",
             "viclip_perclip", "viclip_global"]
ARM_LABEL = {
    "blind": "Blind (no video)",
    "native": "All cameras, 96 frames",
    "seg_perclip": "SigLIP segments 96 (per-clip)",
    "seg_global": "SigLIP segments 96 (global)",
    "viclip_perclip": "ViCLIP segments 96 (per-clip)",
    "viclip_global": "ViCLIP segments 96 (global)",
}
# the segment_select arms as one family, for the trace picker and the
# selection/global pickers
SEG_PERCLIP_ARMS = ("seg_perclip", "viclip_perclip")
SEG_GLOBAL_ARMS = ("seg_global", "viclip_global")
SEG_ARMS = SEG_PERCLIP_ARMS + SEG_GLOBAL_ARMS


def seg_arm_keys(method):
    """(per-clip key, global key) for a segment_select method. The scorer comes
    from the method name on the row, never from --selection-method: anything
    with 'viclip' in it is the ViCLIP family, and everything else keeps the
    keys the earlier legs and their manifests were written with."""
    return (("viclip_perclip", "viclip_global")
            if "viclip" in (method or "").lower()
            else ("seg_perclip", "seg_global"))


def scorer_display(raw):
    """'SigLIP' / 'ViCLIP' for the header and the colorbar, from the row's own
    frame_alloc.segment_scorer - not from the flag, and not from
    frame_alloc.clip_model, which the ViCLIP legs leave stamped with the CLIP
    default ('openai/clip-vit-base-patch32') while scoring with ViCLIP."""
    s = str(raw or "").lower()
    if "viclip" in s:
        return "ViCLIP"
    if "siglip" in s:
        return "SigLIP"
    if "clip" in s:
        return "CLIP"
    return str(raw or "").strip() or "segment"


_COUNT_WORD = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
               7: "seven", 8: "eight"}


def query_mode_phrase(fa):
    """What the segments were scored against, read off the row: the SigLIP legs
    score each segment against the question text, the ViCLIP-opt legs against
    every answer option and reduce over them."""
    if not fa:
        return ""
    mode = str(fa.get("query_mode") or "").lower()
    reduce_ = str(fa.get("segment_reduce") or "max")
    if mode == "options":
        try:
            k = int(fa.get("n_query_texts"))
        except (TypeError, ValueError):
            k = None
        target = ("the answer options" if k is None
                  else f"the {_COUNT_WORD.get(k, str(k))} answer options")
        return f"scored against {target}, {reduce_} over options"
    if mode == "question":
        return "scored against the question text"
    return f"query mode {mode}" if mode else ""


def arm_of(row):
    m = (row.get("method") or "").strip()
    fa = row.get("frame_alloc") or {}
    if m == "blind" or (m.startswith("blind")):
        return "blind"
    if m.startswith("segment_select"):
        mode = fa.get("segment_select_mode")
        if not mode:
            mode = "global" if "global" in (row.get("_tag") or "") else "per_clip"
        per_clip_key, global_key = seg_arm_keys(m)
        return global_key if mode == "global" else per_clip_key
    if m in ("cvbench_native", "native"):
        return "native"
    return f"other:{m or 'unknown'}"


def selection_latency_of(row):
    """The segment-scoring time (decode + scorer) stamped on a selection arm's
    row. None for arms that do no selection - the all-cameras and blind arms -
    whose own decode is not timed."""
    fa = row.get("frame_alloc") or {}
    v = fa.get("selection_latency_s")
    return v if isinstance(v, (int, float)) else None


def protocol_of(rows):
    """Reasoning is imposed by the prompt, not a model switch, so it is read off
    the rows: a non-empty think block, or <think> left in the raw response."""
    for r in rows:
        if (r.get("think") or "").strip():
            return "reasoning"
        if "<think>" in (r.get("response_text") or ""):
            return "reasoning"
    return "direct"


def group_rows(rows):
    """(arm, tag) -> group dict, with rows ordered by pass index."""
    groups = {}
    for r in rows:
        key = (arm_of(r), r.get("_tag") or "?")
        groups.setdefault(key, []).append(r)
    out = []
    for (arm, tag), rs in groups.items():
        rs.sort(key=lambda r: (r.get("pass_idx") if r.get("pass_idx") is not None else 0))
        n_ok = sum(1 for r in rs if r.get("correct"))
        out.append({
            "arm": arm,
            "tag": tag,
            "rows": rs,
            "first": rs[0],
            "method": rs[0].get("method"),
            "protocol": protocol_of(rs),
            "temperature": rs[0].get("temperature"),
            "n_pass": len(rs),
            "n_correct": n_ok,
            "rescored": bool(rs[0].get("_rescored")),
        })
    out.sort(key=lambda g: (ARM_ORDER.index(g["arm"]) if g["arm"] in ARM_ORDER else 99,
                            g["tag"]))
    return out


def _method_filter(cands, method):
    return [g for g in cands if (g["method"] or "") == method] if method else cands


def pick_selection_group(groups, method=None):
    """The per-clip segment_select group the selection figure is drawn from:
    the reasoning demo run if present, else the direct demo run, else the
    rescored mp4sg96 leg.

    ``method`` (--selection-method) restricts the candidates to one scorer's
    leg, so passing both scorers' rows still draws one timeline. When that
    scorer has no per-clip row in these files the pick falls back to whatever
    per-clip leg is here and says so: the header and the colorbar name the
    scorer off the row that was drawn, so a fallback is labelled honestly,
    whereas dropping the figure would hide the selection entirely."""
    have = [g for g in groups if g["arm"] in SEG_PERCLIP_ARMS
            and (g["first"].get("frame_alloc") or {}).get("segment_scores")]
    cands = _method_filter(have, method)
    fell_back = bool(method) and not cands and bool(have)
    if fell_back:
        cands = have
    if not cands:
        return None
    # demo run before a rescored leg, reasoning before direct, and within a
    # tier the bigger frame budget (a 32- and a 96-frame leg can both match)
    cands.sort(key=lambda g: (1 if g["rescored"] else 0,
                              0 if g["protocol"] == "reasoning" else 1,
                              -int((g["first"].get("frame_alloc") or {}).get("budget") or 0),
                              g["tag"]))
    if fell_back:
        note(f"--selection-method {method}: no per-clip row in these files; the "
             f"selection figure falls back to {cands[0]['method']} "
             f"(tag {cands[0]['tag']})")
    return cands[0]


def pick_global_group(groups, method=None):
    """The --seg-select global group drawn as the dashed outline and the blue
    tick row, from the same scorer as the per-clip pick."""
    have = [g for g in groups if g["arm"] in SEG_GLOBAL_ARMS
            and (g["first"].get("frame_alloc") or {}).get("segments_kept_per_video")]
    cands = _method_filter(have, method)
    fell_back = bool(method) and not cands and bool(have)
    if fell_back:
        cands = have
    cands.sort(key=lambda g: (1 if g["rescored"] else 0, g["tag"]))
    if not cands:
        return None
    if fell_back:
        note(f"--selection-method {method}: no --seg-select global row in these "
             f"files; the dashed outline falls back to {cands[0]['method']} "
             f"(tag {cands[0]['tag']})")
    return cands[0]


def trace_text(groups, limit=700, prefer=None):
    """The trace excerpt, taken from a SIGHTED arm only.

    The blind arm is never quoted as the model's reasoning: it saw no pixels, so
    its chain of thought is a language prior, and printed beside a selection
    figure it reads as reasoning over the selected frames. Order: the
    segment-select reasoning row, then any other sighted reasoning row, then a
    sighted direct row (a bare letter, which the caller labels as such). A blind
    row is reached only when no sighted row produced any text at all, and then
    the caller stamps the panel.

    ``prefer`` (the group the selection figure was drawn from) is tried first
    within the segment-select tier, so the ViCLIP page quotes the ViCLIP row and
    the SigLIP page the SigLIP one; either way the caption names the method and
    the tag it came from.

    Returns (text, group, full_len, is_reasoning, is_blind).
    """
    def body(g):
        r = g["first"]
        t = (r.get("think") or "").strip()
        if t:
            return t
        return (r.get("response_text") or "").strip()

    sighted = [g for g in groups if g["arm"] != "blind"]
    seg = [g for g in sighted if g["arm"] in SEG_ARMS]
    rest = [g for g in sighted if g["arm"] not in SEG_ARMS]
    if prefer is not None and any(g is prefer for g in seg):
        seg = [prefer] + [g for g in seg if g is not prefer]
    # the segment-select reasoning row first; then any other SIGHTED reasoning
    # row; then a sighted direct row, whose response_text is a bare letter
    order = ([g for g in seg if g["protocol"] == "reasoning"]
             + [g for g in rest if g["protocol"] == "reasoning"]
             + [g for g in seg if g["protocol"] != "reasoning"]
             + [g for g in rest if g["protocol"] != "reasoning"])
    blind = [g for g in groups if g["arm"] == "blind"]
    for g in order + blind:
        txt = body(g)
        if txt:
            clipped = txt[:limit]
            is_blind = g["arm"] == "blind"
            return (clipped + ("..." if len(txt) > limit else ""), g, len(txt),
                    g["protocol"] == "reasoning" and not is_blind, is_blind)
    return "", None, 0, False, False


# --------------------------------------------------------------------------
# video / thumbnails
# --------------------------------------------------------------------------

def open_reader(path):
    from decord import VideoReader, cpu  # imported late: heavy, and only here
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing demo media: {path}")
    if path.lower().endswith(".avi"):
        raise RuntimeError(f"refusing to decode an .avi: {path}")
    return VideoReader(path, ctx=cpu(0), num_threads=2)


def clip_meta(vr):
    n = len(vr)
    fps = float(vr.get_avg_fps() or 0) or 30.0
    return n, fps, n / fps


def thumbs_at(vr, times, fps, n_frames, width=160):
    idx = [max(0, min(n_frames - 1, int(round(t * fps)))) for t in times]
    if not idx:
        return []
    order = sorted(range(len(idx)), key=lambda i: idx[i])
    try:
        batch = vr.get_batch([idx[i] for i in order]).asnumpy()
    except Exception as exc:
        note(f"decode failed ({exc.__class__.__name__}) - falling back to single seeks")
        batch = np.stack([vr[idx[i]].asnumpy() for i in order])
    out = [None] * len(idx)
    for slot, i in enumerate(order):
        img = Image.fromarray(batch[slot])
        h = max(1, int(round(width * img.height / img.width)))
        out[i] = np.asarray(img.resize((width, h), Image.BILINEAR))
    return out


def even_pick(seq, k):
    if k <= 0 or not seq:
        return []
    if len(seq) <= k:
        return list(range(len(seq)))
    return sorted(set(int(round(x)) for x in np.linspace(0, len(seq) - 1, k)))


def pick_thumb_idx(times, k, windows, pad=3.0):
    """An even spread over the selected times, except that a frame the release
    actually annotates wins its slot: for each event window the nearest selected
    frame inside it replaces the even pick closest to it (at most one swap per
    window). Without this the even spread routinely steps over the two or three
    frames that carry the evidence, which is the whole point of the figure.

    The candidate is ranked by its distance to the window, not by membership of
    the ±pad band: holding a frame 2 s before the window is not the same as
    showing the annotated moment, and skipping the swap because some ±pad
    neighbour was already picked hid exactly the frame the label calls the
    nearest one (q251 camera 3: 39.8 s held, 44.5 s - 0.03 s off - never
    drawn)."""
    picks = even_pick(times, k)
    if not picks or not windows or len(times) <= k:
        return picks
    chosen = set(picks)

    def in_any(t):
        return any(w["start_sec"] - pad <= t <= w["end_sec"] + pad for w in windows)

    for w in windows:
        lo, hi = w["start_sec"] - pad, w["end_sec"] + pad
        centre = 0.5 * (w["start_sec"] + w["end_sec"])

        def gap(t, w=w):
            return max(w["start_sec"] - t, t - w["end_sec"], 0.0)

        cands = [i for i, t in enumerate(times) if lo <= t <= hi]
        if not cands:
            continue
        best = min(cands, key=lambda i: (gap(times[i]), abs(times[i] - centre)))
        # skip only when a frame at least as close to the window is already
        # held; a ±pad neighbour further out does not count as showing it
        if any(gap(times[i]) <= gap(times[best]) + 1e-9 for i in chosen):
            continue
        free = [i for i in chosen if not in_any(times[i])]
        if not free:
            # every held frame is near some window: give up the one that is
            # furthest from its own nearest window
            free = [i for i in chosen
                    if not any(v["start_sec"] <= times[i] <= v["end_sec"]
                               for v in windows)]
        if not free:
            continue
        drop = min(free, key=lambda i: abs(times[i] - times[best]))
        chosen.discard(drop)
        chosen.add(best)
    return sorted(chosen)


# --------------------------------------------------------------------------
# per-question assembly
# --------------------------------------------------------------------------

def window_gap(ev, times):
    """Seconds between an annotated window and the nearest selected frame; 0.0
    when a frame lands inside it, None when nothing was selected."""
    if not times:
        return None
    return round(min(max(ev["start_sec"] - t, t - ev["end_sec"], 0.0)
                     for t in times), 2)


def frame_count(per_video, slot, times):
    """A leg's own frame count for one camera: its stored per-video count when
    it has one, else however many times it recorded."""
    try:
        return int((per_video or {}).get(slot))
    except (TypeError, ValueError):
        return len(times)


def segment_bounds(n_total, fps, n_seg):
    """Same split segment_select uses: bounds = round(s * n / S)."""
    b = [round(s * n_total / n_seg) for s in range(n_seg + 1)]
    return [(b[s] / fps, b[s + 1] / fps) for s in range(n_seg)]


def build_question(rec, rows, vp_fn, video_root, thumbs_per_cam,
                   selection_method=None):
    qid = rec.get("id")
    item = release_item(rec)
    reqs = requires_cameras(item)
    events = event_windows(item, qid)
    groups = group_rows(rows)
    sel = pick_selection_group(groups, selection_method)
    glob_g = pick_global_group(groups, selection_method)

    if sel is None:
        note(f"id {qid}: no segment_select row with scores; the selection figure "
             f"falls back to an even sample and uncoloured segments")
    fa = (sel["first"].get("frame_alloc") or {}) if sel else {}
    scores_all = fa.get("segment_scores") or {}
    kept_all = fa.get("segments_kept_per_video") or {}
    times_all = fa.get("selected_times_s") or {}
    pvd = fa.get("per_video_decode") or []
    seg_per_video = int(fa.get("segments_per_video") or 8)
    keep_k = fa.get("segments_keep")

    gfa = (glob_g["first"].get("frame_alloc") or {}) if glob_g else {}
    gkept_all = gfa.get("segments_kept_per_video") or {}
    # the global leg's defining property is an UNEQUAL budget across cameras,
    # so its own frame times and per-camera counts are carried too - drawing
    # only its kept segments would hide exactly what it does differently
    gtimes_all = gfa.get("selected_times_s") or {}
    gper_all = gfa.get("selected_per_video") or {}
    per_all = fa.get("selected_per_video") or {}

    paths = vp_fn(rec, video_root)
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"missing demo media: {p}")

    cams = []
    for i, path in enumerate(paths, start=1):
        slot = str(i)
        vr = open_reader(path)
        n_mp4, fps_mp4, dur = clip_meta(vr)
        d = pvd[i - 1] if i - 1 < len(pvd) and isinstance(pvd[i - 1], dict) else {}
        n_total = int(d.get("n_total") or n_mp4)
        fps = float(d.get("fps") or fps_mp4)
        raw = scores_all.get(slot) or {}
        scores = {}
        for k, v in raw.items():
            try:
                scores[int(k)] = float(v)
            except (TypeError, ValueError):
                continue
        n_seg = int(d.get("n_segments") or 0) or (max(scores) + 1 if scores else seg_per_video)
        kept = sorted(int(x) for x in (kept_all.get(slot) or []))
        if not kept and scores:
            k = int(keep_k) if keep_k else 0
            if k > 0:
                kept = sorted(sorted(scores, key=lambda s: -scores[s])[:k])
                note(f"id {qid} slot {slot}: no segments_kept_per_video; "
                     f"showing the top-{k} of the stored scores")
        gkept = sorted(int(x) for x in (gkept_all.get(slot) or []))
        times = [float(t) for t in (times_all.get(slot) or [])]
        gtimes = [float(t) for t in (gtimes_all.get(slot) or [])]

        cam = cam_id_of(path)
        evidence = None if reqs is None else (cam in reqs)
        cam_events = [e for e in events if e["camera"] == cam]

        picks = pick_thumb_idx(times, thumbs_per_cam, cam_events)
        t_pick = [times[j] for j in picks]
        if not t_pick:
            t_pick = [dur * (j + 0.5) / max(1, thumbs_per_cam) for j in range(thumbs_per_cam)]
            sampled = True
        else:
            sampled = False
        imgs = thumbs_at(vr, t_pick, fps_mp4, n_mp4)
        hl = [any(e["start_sec"] - 3.0 <= t <= e["end_sec"] + 3.0 for e in cam_events)
              for t in t_pick]
        # the number the demo turns on: how close the selection got to each
        # annotated moment (0.0 = a frame inside the window)
        cam_events = [dict(e, nearest_frame_gap_s=window_gap(e, times))
                      for e in cam_events]
        del vr

        cams.append({
            "slot": i,
            "camera": cam,
            "path": os.path.relpath(path, REPO),
            "duration_s": round(dur, 2),
            "fps": fps_mp4,
            "n_frames_mp4": n_mp4,
            "n_frames_scored": n_total,
            "n_segments": n_seg,
            "segment_bounds_s": [[round(a, 2), round(b, 2)]
                                 for a, b in segment_bounds(n_total, fps, n_seg)],
            "segment_scores": {str(k): scores[k] for k in sorted(scores)},
            "max_score": max(scores.values()) if scores else None,
            "segments_kept": kept,
            "segments_kept_global": gkept,
            "selected_times_s": times,
            "n_frames_selected": frame_count(per_all, slot, times),
            "selected_times_global_s": gtimes,
            "n_frames_selected_global": (frame_count(gper_all, slot, gtimes)
                                        if glob_g else None),
            "evidence": evidence,
            "event_windows": cam_events,
            "thumb_times_s": [round(t, 2) for t in t_pick],
            "thumb_in_window": hl,
            "_thumbs": imgs,
            "_thumb_sampled": sampled,
        })

    return {
        "id": qid,
        "rec": rec,
        "item": item,
        "question_type": rec.get("question_type"),
        "task_type": rec.get("task_type"),
        "gold": rec.get("answer"),
        "requires_cameras": reqs,
        "events": events,
        "cams": cams,
        "groups": groups,
        "sel": sel,
        "global": glob_g,
        # segment_scorer first: it is the field the method actually scored with,
        # while clip_model stays at the CLIP default on the ViCLIP legs (both
        # fields agree on the SigLIP legs, so this changes nothing there)
        "seg_scorer": fa.get("segment_scorer") or fa.get("clip_model"),
        "scorer_name": scorer_display(fa.get("segment_scorer")
                                      or fa.get("clip_model")),
        "query_phrase": query_mode_phrase(fa),
        "budget": fa.get("budget"),
        "frames_per_segment": fa.get("frames_per_segment"),
        "segments_keep": keep_k,
    }


# --------------------------------------------------------------------------
# drawing helpers
# --------------------------------------------------------------------------

def ink_on(rgba):
    r, g, b = to_rgb(rgba)
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return "#ffffff" if lum < 0.55 else "#101010"


def shorten(text, n):
    text = " ".join(str(text or "").split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def lanes_for(events):
    """Pack overlapping windows into stacked lanes so labels never collide."""
    lanes = []
    out = {}
    for i, e in enumerate(sorted(range(len(events)), key=lambda j: events[j]["start_sec"])):
        ev = events[e]
        placed = False
        for li, end in enumerate(lanes):
            if ev["start_sec"] > end:
                lanes[li] = ev["end_sec"]
                out[e] = li
                placed = True
                break
        if not placed:
            lanes.append(ev["end_sec"])
            out[e] = len(lanes) - 1
    return out, max(1, len(lanes))


# --------------------------------------------------------------------------
# figure 1: selection
# --------------------------------------------------------------------------

def fig_selection(q, out_path, dpi):
    cams = q["cams"]
    n = len(cams)
    has_global = any(c["segments_kept_global"] or c.get("selected_times_global_s")
                     for c in cams)
    any_widened = any(_window_widened(c) for c in cams)

    all_scores = [v for c in cams for v in c["segment_scores"].values()]
    if all_scores:
        vmin, vmax = min(all_scores), max(all_scores)
        if vmax - vmin < 1e-9:
            vmin, vmax = vmin - 1e-3, vmax + 1e-3
    else:
        vmin, vmax = 0.0, 1.0
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("viridis")

    # inches
    W = 17.0
    M_L, LAB_W, GAP_L = 0.28, 1.78, 0.18
    TL_X, TL_W = M_L + LAB_W + GAP_L, 6.9
    TH_X = TL_X + TL_W + 0.55
    TH_W = W - TH_X - 0.28
    q_lines = textwrap.wrap(q["rec"].get("question", ""), 158)[:3]
    # + 0.26 for the scorer/query-mode line under the selection line
    HEAD_H = 0.52 + 0.250 * len(q_lines) + 0.26 + (0.78 if has_global else 0.54)
    ROW_H = 2.18 if has_global else 2.02   # a global leg adds a second tick row
    ROW_GAP = 0.16
    CB_H, M_B, M_T = 0.16, 0.26, 0.22
    n_leg = 6 if has_global else 4
    LEG_COLS = 3
    LEG_ROWS = -(-n_leg // LEG_COLS)
    LEG_H = 0.30 * LEG_ROWS
    # the colorbar carries tick labels AND its own label below the bar, so it
    # needs real clearance above the legend strip
    BOT = M_B + LEG_H + 0.62 + CB_H
    H = M_T + HEAD_H + n * ROW_H + (n - 1) * ROW_GAP + 0.34 + BOT

    fig = plt.figure(figsize=(W, H), dpi=dpi)
    fig.patch.set_facecolor("white")

    def fx(x):
        return x / W

    def fy(y):
        return y / H

    # ---- header -----------------------------------------------------------
    sel = q["sel"]
    y = H - M_T
    fig.text(fx(M_L), fy(y), f"Q{q['id']}  ·  {q['question_type']}  ·  "
             f"{n} camera streams  ·  gold {q['gold']}",
             fontsize=FS_TITLE, weight="bold", color=C_INK, va="top")
    y -= 0.40
    for line in q_lines:
        fig.text(fx(M_L), fy(y), line, fontsize=FS_ROW, color=C_INK, va="top")
        y -= 0.250
    y -= 0.05
    if sel:
        sub = (f"selection: {sel['method']}  (tag {sel['tag']}, {sel['protocol']}, "
               f"T={sel['temperature']}, pass {sel['first'].get('pass_idx')})"
               f"   budget {q['budget']} frames"
               f"   {q['frames_per_segment']} frames/segment")
        # the scorer and the query mode are read from this row's frame_alloc, so
        # the wrong --selection-method cannot caption a ViCLIP leg as SigLIP
        sub2 = f"scorer: {q['scorer_name']}  [{q['seg_scorer']}]"
        if q["query_phrase"]:
            sub2 += f"  \u00b7  {q['query_phrase']}"
    else:
        sub = "selection: no segment_select row found - segments shown uncoloured"
        sub2 = "scorer: unknown - no frame_alloc on any row to read it from"
    fig.text(fx(M_L), fy(y), sub, fontsize=FS_TINY_SEL, color=C_MUTE, va="top")
    y -= 0.26
    fig.text(fx(M_L), fy(y), sub2, fontsize=FS_TINY_SEL, color=C_INK, va="top")
    y -= 0.26
    if has_global:
        g = q["global"]
        fig.text(fx(M_L), fy(y), f"global leg: tag {g['tag']} "
                 f"(--seg-select global) - dashed segment outlines, blue ticks, "
                 f"and its own per-camera frame count "
                 f"(the global budget is unequal across cameras)",
                 fontsize=FS_TINY_SEL, color=C_GLOBAL, va="top")

    # ---- camera rows ------------------------------------------------------
    top = H - M_T - HEAD_H
    for r, c in enumerate(cams):
        row_top = top - r * (ROW_H + ROW_GAP)
        row_bot = row_top - ROW_H

        # label column
        ly = row_top - 0.10
        fig.text(fx(M_L), fy(ly), f"Camera {c['slot']}", fontsize=FS_ROW,
                 weight="bold", color=C_INK, va="top")
        ly -= 0.30
        fig.text(fx(M_L), fy(ly), c["camera"], fontsize=FS_HEAD, weight="bold",
                 color=C_INK, va="top", family="DejaVu Sans Mono")
        ly -= 0.32
        if c["evidence"] is True:
            fig.text(fx(M_L), fy(ly), "EVIDENCE", fontsize=FS_BODY, weight="bold",
                     color="white", va="top",
                     bbox=dict(boxstyle="square,pad=0.28", fc=C_EVENT, ec="none"))
        elif c["evidence"] is False:
            fig.text(fx(M_L), fy(ly), "padding", fontsize=FS_BODY, color=C_MUTE,
                     va="top", style="italic")
        else:
            fig.text(fx(M_L), fy(ly), "role unknown", fontsize=FS_BODY,
                     color=C_MUTE, va="top", style="italic")
        ly -= 0.34
        ms = c["max_score"]
        fig.text(fx(M_L), fy(ly),
                 f"max {ms:.4f}" if ms is not None else "max n/a",
                 fontsize=FS_BODY, color=C_INK, va="top")
        ly -= 0.26
        fig.text(fx(M_L), fy(ly), f"{c['duration_s']:.0f} s clip",
                 fontsize=FS_TINY_SEL, color=C_MUTE, va="top")

        # timeline
        tl_h = 1.50 if has_global else 1.34
        ax = fig.add_axes([fx(TL_X), fy(row_bot + 0.42), fx(TL_W), fy(tl_h)])
        # characters that fit across the strip at FS_TINY, for label placement
        chars = max(20, int(TL_W * 72.0 / (FS_TINY_SEL * 0.58)))
        draw_timeline(ax, c, norm, cmap, last=(r == n - 1), chars_across=chars)

        # thumbnails
        draw_thumbs(fig, c, TH_X, TH_W, row_bot, ROW_H, fx, fy)

    # ---- colorbar ---------------------------------------------------------
    cb_y = M_B + LEG_H + 0.62
    cax = fig.add_axes([fx(TL_X), fy(cb_y), fx(TL_W * 0.55), fy(CB_H)])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                      orientation="horizontal")
    cb.set_label(f"{q['scorer_name']} relevance", fontsize=FS_BODY, color=C_INK)
    cb.ax.tick_params(labelsize=FS_TINY_SEL)

    # ---- legend -----------------------------------------------------------
    ly = M_B
    ev_label = ("annotated event window (pale = widened)"
                if any_widened else "annotated event window (release verification)")
    items = [
        ("kept segment - bold outline + KEPT label", C_KEPT, "solid"),
        ("selected frame time (tick below the strip)", C_TICK, "tick"),
        (ev_label, C_EVENT, "fill"),
        (f"{MARK_STAR} frame inside an annotated window (±3 s)", C_EVENT, "star"),
    ]
    if has_global:
        items.insert(1, ("kept by --seg-select global - dashed outline", C_GLOBAL, "dashed"))
        items.insert(3, ("frame sent by the global leg (blue tick row)", C_GLOBAL, "tick"))
    col_w = (W - 2 * M_L) / LEG_COLS
    for i, (label, color, kind) in enumerate(items):
        x = M_L + (i % LEG_COLS) * col_w
        yy = ly + (LEG_ROWS - 1 - i // LEG_COLS) * 0.30
        if kind == "fill":
            fig.add_artist(Rectangle((fx(x), fy(yy + 0.02)), fx(0.30), fy(0.12),
                                     fc=color, ec="none",
                                     transform=fig.transFigure))
        elif kind == "tick":
            fig.add_artist(Rectangle((fx(x + 0.13), fy(yy - 0.02)), fx(0.035),
                                     fy(0.20), fc=color, ec="none",
                                     transform=fig.transFigure))
        elif kind == "star":
            fig.add_artist(Rectangle((fx(x), fy(yy - 0.02)), fx(0.30), fy(0.20),
                                     fc="none", ec=color, lw=2.6,
                                     transform=fig.transFigure))
        else:
            fig.add_artist(Rectangle(
                (fx(x), fy(yy - 0.02)), fx(0.30), fy(0.20), fc="none", ec=color,
                lw=2.6, ls="solid" if kind == "solid" else (0, (3, 2)),
                transform=fig.transFigure))
        fig.text(fx(x + 0.38), fy(yy + 0.08), shorten(label, 49),
                 fontsize=FS_TINY_SEL, color=C_INK, va="center")

    fig.savefig(out_path, dpi=dpi, facecolor="white")
    plt.close(fig)


# very short annotated windows are widened to this fraction of the clip so the
# bar stays visible at all; when that happens the bar is drawn pale with a
# full-opacity core at the true extent, and the legend says so
WIN_FLOOR_FRAC = 0.008


def _clip_span(c):
    return max(c["duration_s"],
               c["segment_bounds_s"][-1][1] if c["segment_bounds_s"] else 0)


def _window_widened(c):
    dur = _clip_span(c)
    return any((e["end_sec"] - e["start_sec"]) < dur * WIN_FLOOR_FRAC - 1e-9
               for e in c["event_windows"])


def draw_timeline(ax, c, norm, cmap, last, chars_across=82):
    dur = _clip_span(c)
    evs = c["event_windows"]
    lane_of, n_lane = lanes_for(evs) if evs else ({}, 0)
    LANE0, LANE_DY = 1.22, 0.74
    gtimes = c.get("selected_times_global_s") or []
    # a global leg needs a second tick row, and both counts share one caption
    # line so the strip itself does not have to shrink
    if gtimes:
        PC_LO, PC_HI, G_LO, G_HI, CAP_Y, Y_BOT = -0.42, -0.08, -0.86, -0.52, -1.18, -1.55
    else:
        PC_LO, PC_HI, G_LO, G_HI, CAP_Y, Y_BOT = -0.52, -0.08, None, None, -0.82, -1.15
    ax.set_xlim(0, dur)
    ax.set_ylim(Y_BOT, max(2.10, LANE0 + LANE_DY * max(0, n_lane - 1) + 0.86))
    ax.set_yticks([])
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#999999")
    ax.tick_params(axis="x", labelsize=FS_TINY_SEL, colors=C_INK, length=3)
    step = 30 if dur > 150 else (10 if dur > 50 else 5)
    ax.set_xticks(np.arange(0, dur + 1e-6, step))
    if last:
        ax.set_xlabel("seconds into this camera's clip", fontsize=FS_ROW, color=C_INK)
    else:
        ax.set_xticklabels([f"{int(t)}" for t in ax.get_xticks()])

    kept = set(c["segments_kept"])
    gkept = set(c["segments_kept_global"])
    for sid, (t0, t1) in enumerate(c["segment_bounds_s"]):
        sc = c["segment_scores"].get(str(sid))
        face = cmap(norm(sc)) if sc is not None else "#d9d9d9"
        ax.add_patch(Rectangle((t0, 0.0), t1 - t0, 1.0, facecolor=face,
                               edgecolor="white", lw=1.2, zorder=2))
        mid = 0.5 * (t0 + t1)
        if sc is not None:
            # 4 decimals: the top-K cut is routinely decided at the 4th, and at
            # 3 the same printed number appears both kept and dropped
            ax.text(mid, 0.66, f"{sc:.4f}", ha="center", va="center",
                    fontsize=FS_TINY_SEL, color=ink_on(face), zorder=4)
        if sid in kept:
            ax.add_patch(Rectangle((t0, 0.0), t1 - t0, 1.0, facecolor="none",
                                   edgecolor=C_KEPT, lw=3.2, zorder=5))
            ax.text(mid, 0.24, "KEPT", ha="center", va="center",
                    fontsize=FS_TINY_SEL, weight="bold", color=ink_on(face),
                    zorder=6)
        if sid in gkept:
            pad = (t1 - t0) * 0.06
            # no letter in the cell: the score owns its middle and the KEPT
            # label the row below, and the dashed outline, the blue tick row
            # and the legend already say "global"
            ax.add_patch(Rectangle((t0 + pad, 0.07), (t1 - t0) - 2 * pad, 0.86,
                                   facecolor="none", edgecolor=C_GLOBAL, lw=2.4,
                                   ls=(0, (4, 2)), zorder=7))

    for t in c["selected_times_s"]:
        ax.plot([t, t], [PC_LO, PC_HI], color=C_TICK, lw=1.1,
                solid_capstyle="butt", zorder=3)
    n_pc = c.get("n_frames_selected") or len(c["selected_times_s"])
    if c["selected_times_s"]:
        ax.text(0, CAP_Y,
                (f"per-clip: {n_pc} frames sent to the model" if gtimes
                 else f"{n_pc} frames sent to the model"),
                fontsize=FS_TINY_SEL, color=C_MUTE, va="center", ha="left")
    if gtimes:
        for t in gtimes:
            ax.plot([t, t], [G_LO, G_HI], color=C_GLOBAL, lw=1.1,
                    solid_capstyle="butt", zorder=3)
        n_g = c.get("n_frames_selected_global") or len(gtimes)
        # same line, right-aligned, in the tick row's own colour: the two counts
        # differ per camera and that difference is the point of the global arm
        ax.text(dur, CAP_Y, f"global: {n_g} frames sent to the model",
                fontsize=FS_TINY_SEL, color=C_GLOBAL, va="center", ha="right")

    if evs:
        for i, ev in enumerate(evs):
            lane = lane_of.get(i, 0)
            y0 = LANE0 + LANE_DY * (n_lane - 1 - lane)
            true_w = ev["end_sec"] - ev["start_sec"]
            w = max(true_w, dur * WIN_FLOOR_FRAC)
            widened = w > true_w + 1e-9
            # a 3-second window is a hairline against a 300-second clip, so drop
            # a guide from the bar down to the strip: where it lands is the point
            ax.plot([ev["start_sec"] + w / 2] * 2, [1.02, y0], color=C_EVENT,
                    lw=1.0, ls=(0, (2, 2)), zorder=5)
            ax.add_patch(Rectangle((ev["start_sec"], y0), w, 0.26,
                                   facecolor=C_EVENT, edgecolor="#7a3400", lw=1.0,
                                   alpha=0.42 if widened else 1.0, zorder=6))
            if widened:
                # the bar above is a visibility floor; this is the real extent
                ax.add_patch(Rectangle((ev["start_sec"], y0), true_w, 0.26,
                                       facecolor=C_EVENT, edgecolor="none",
                                       zorder=7))
                for edge in (ev["start_sec"], ev["end_sec"]):
                    ax.plot([edge, edge], [y0, y0 + 0.26], color="#7a3400",
                            lw=1.3, solid_capstyle="butt", zorder=8)
            gap = ev.get("nearest_frame_gap_s")
            if gap is None:
                gap_txt = ""
            elif gap <= 0.0:
                gap_txt = "  [frame inside]"
            elif gap < 1.0:
                # "0.0 s off" beside a legend that says "frame inside" reads as
                # a contradiction; below a second the second decimal is the point
                gap_txt = f"  [nearest frame {gap:.2f} s off]"
            else:
                gap_txt = f"  [nearest frame {gap:.1f} s off]"
            txt = (f"{ev['label']} {ev['start_sec']:.1f}-{ev['end_sec']:.1f}s  "
                   f"{shorten(ev['description'], 40)}"
                   + ("  (end est.)" if ev["end_estimated"] else "")
                   + ("  (bar widened)" if widened else "") + gap_txt)
            # the label sits ABOVE its bar, centred on it and clamped inside
            # the strip, so it can never sit on top of the bar or run off
            est = len(txt) * dur / float(chars_across)
            centre = ev["start_sec"] + w / 2
            lx = min(max(centre - est / 2, 0.0), max(dur - est, 0.0))
            ax.text(lx, y0 + 0.47, txt, fontsize=FS_TINY_SEL, color="#6b2f00",
                    va="center", ha="left", zorder=9)


def draw_thumbs(fig, c, x0, total_w, row_bot, row_h, fx, fy):
    imgs = c["_thumbs"]
    if not imgs:
        fig.text(fx(x0), fy(row_bot + row_h * 0.5), "no frames to show",
                 fontsize=FS_TINY_SEL, color=C_MUTE, va="center")
        return
    k = len(imgs)
    gap = 0.07
    w = (total_w - gap * (k - 1)) / k
    ar = imgs[0].shape[0] / imgs[0].shape[1]
    h = w * ar
    y = row_bot + (row_h - h) / 2.0 + 0.16
    for i, img in enumerate(imgs):
        x = x0 + i * (w + gap)
        ax = fig.add_axes([fx(x), fy(y), fx(w), fy(h)])
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        hot = c["thumb_in_window"][i]
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_color(C_EVENT if hot else "#b5b5b5")
            s.set_linewidth(3.4 if hot else 1.0)
        cap = f"{c['thumb_times_s'][i]:.1f} s"
        if hot:
            cap += f"  {MARK_STAR}"
        fig.text(fx(x + w / 2), fy(y - 0.055), cap, fontsize=FS_TINY_SEL,
                 ha="center", va="top",
                 color=C_EVENT if hot else C_INK,
                 weight="bold" if hot else "normal")
    if c["_thumb_sampled"]:
        fig.text(fx(x0), fy(y + h + 0.04), "no selected_times_s - even sample",
                 fontsize=FS_TINY_SEL, color=C_MUTE, va="bottom")


# --------------------------------------------------------------------------
# figure 2: answer
# --------------------------------------------------------------------------

TABLE_COLS = [
    ("Arm", 0.215, "left"),
    ("Pred", 0.050, "center"),
    ("Result", 0.185, "left"),
    ("Answer call", 0.085, "right"),
    ("Input tok", 0.090, "right"),
    ("Video tok", 0.090, "right"),
    ("Protocol", 0.285, "left"),
]
COL_PAD = 0.12  # inches, so a right-aligned number never touches the next column


def fig_answer(q, out_path, dpi):
    rec = q["rec"]
    gold = (q["gold"] or "").strip()
    options = rec.get("options") or []
    groups = q["groups"]
    trace, trace_src, trace_len, trace_reasons, trace_blind = trace_text(
        groups, prefer=q.get("sel"))

    # The latency column is the InternVL answer call only. Segment scoring
    # (decode + scorer) is a separate stamp on the selection arms' rows, and
    # the all-cameras arm's own decode is never timed at all.
    sel_lat = None
    if q.get("sel"):
        sel_lat = selection_latency_of(q["sel"]["first"])
    if sel_lat is None:
        for g in groups:
            if g["arm"] in SEG_ARMS:
                sel_lat = selection_latency_of(g["first"])
                if sel_lat is not None:
                    break
    foot = ("Answer call only, the same call for every arm. Segment selection "
            "is timed separately: "
            + (f"{sel_lat:.1f} s" if sel_lat is not None else "not stamped on these rows")
            + " for this arm (SigLIP or ViCLIP scoring, mostly frame "
              "decoding). The all-cameras arm's frame decode is not timed.")
    foot_lines = textwrap.wrap(foot, 158) or [""]

    W = 16.0
    M = 0.42
    qlines = textwrap.wrap(rec.get("question", ""), 132)
    opt_lines = []
    for o in options:
        opt_lines.append(textwrap.wrap(str(o), 124) or [""])
    trace_lines = []
    for para in (trace.splitlines() or [""]):
        trace_lines.extend(textwrap.wrap(para, 148) or [""])
    trace_lines = trace_lines[:26]

    h_head = 0.46
    h_q = 0.27 * len(qlines) + 0.16
    h_opts = sum(0.26 * len(ls) for ls in opt_lines) + 0.26
    h_tbl = (0.40 + 0.40 * max(1, len(groups)) + 0.34
             + 0.16 + 0.225 * len(foot_lines))
    h_trace = (0.34 + 0.225 * max(1, len(trace_lines)) + 0.30
               + (0.38 if trace_blind else 0.0))
    H = M + h_head + h_q + h_opts + h_tbl + h_trace + M

    fig = plt.figure(figsize=(W, H), dpi=dpi)
    fig.patch.set_facecolor("white")

    def fx(x):
        return x / W

    def fy(y):
        return y / H

    y = H - M
    fig.text(fx(M), fy(y), f"Q{q['id']}  ·  {q['question_type']}  ·  "
             f"{len(q['cams'])} cameras  ·  gold answer {gold}",
             fontsize=FS_TITLE, weight="bold", color=C_INK, va="top")
    y -= h_head

    for line in qlines:
        fig.text(fx(M), fy(y), line, fontsize=FS_HEAD, color=C_INK, va="top")
        y -= 0.27
    y -= 0.16

    for o, lines in zip(options, opt_lines):
        letter = str(o).strip()[:1]
        is_gold = letter.upper() == gold.upper()
        blk_h = 0.26 * len(lines)
        if is_gold:
            fig.add_artist(Rectangle(
                (fx(M - 0.10), fy(y - blk_h + 0.02)), fx(W - 2 * M + 0.20), fy(blk_h + 0.04),
                fc=C_GOLD, ec="#a89b00", lw=1.0, alpha=0.55,
                transform=fig.transFigure, zorder=0))
        for j, line in enumerate(lines):
            txt = line + (f"   {ARROW} gold" if (is_gold and j == len(lines) - 1) else "")
            fig.text(fx(M), fy(y), txt, fontsize=FS_BODY, va="top", zorder=2,
                     color=C_INK, weight="bold" if is_gold else "normal")
            y -= 0.26
    y -= 0.26

    # ---- arm table --------------------------------------------------------
    x_left = M
    tbl_w = W - 2 * M
    xs, acc = [], 0.0
    for name, frac, align in TABLE_COLS:
        xs.append((x_left + acc * tbl_w, frac * tbl_w, align, name))
        acc += frac

    hdr_h = 0.36
    fig.add_artist(Rectangle((fx(x_left - 0.08), fy(y - hdr_h)),
                                 fx(tbl_w + 0.16), fy(hdr_h),
                                 fc="#e3e3df", ec="none",
                                 transform=fig.transFigure, zorder=0))
    for x, w, align, name in xs:
        tx = (x + COL_PAD if align == "left"
              else (x + w - COL_PAD if align == "right" else x + w / 2))
        fig.text(fx(tx), fy(y - hdr_h / 2), name, fontsize=FS_BODY, weight="bold",
                 color=C_INK, va="center", ha=align, zorder=2)
    y -= hdr_h

    if not groups:
        fig.text(fx(x_left), fy(y - 0.2), "no result rows for this question",
                 fontsize=FS_BODY, color=C_BAD, va="center")
        y -= 0.4
    row_h = 0.40
    for i, g in enumerate(groups):
        r = g["first"]
        if i % 2 == 1:
            fig.add_artist(Rectangle((fx(x_left - 0.08), fy(y - row_h)),
                                         fx(tbl_w + 0.16), fy(row_h),
                                         fc="#f6f6f4", ec="none",
                                         transform=fig.transFigure, zorder=0))
        ok = bool(r.get("correct"))
        label = ARM_LABEL.get(g["arm"], g["arm"].replace("other:", ""))
        result = f"{MARK_OK} correct" if ok else f"{MARK_BAD} wrong"
        if g["n_pass"] > 1:
            result += f"   ({g['n_correct']}/{g['n_pass']} passes)"
        lat = r.get("latency_s")
        cells = [
            (f"{label}", C_INK, "bold"),
            (str(r.get("prediction") or "-"), C_OK if ok else C_BAD, "bold"),
            (result, C_OK if ok else C_BAD, "normal"),
            (f"{lat:.1f} s" if isinstance(lat, (int, float)) else "n/a", C_INK, "normal"),
            (f"{r.get('input_tokens'):,}" if isinstance(r.get("input_tokens"), int) else "n/a",
             C_INK, "normal"),
            (f"{r.get('video_tokens'):,}" if isinstance(r.get("video_tokens"), int) else "n/a",
             C_INK, "normal"),
            (f"{g['protocol']}, T={g['temperature']}  [{g['tag']}]", C_MUTE, "normal"),
        ]
        for (x, w, align, _), (txt, color, weight) in zip(xs, cells):
            tx = (x + COL_PAD if align == "left"
                  else (x + w - COL_PAD if align == "right" else x + w / 2))
            fig.text(fx(tx), fy(y - row_h / 2), txt, fontsize=FS_BODY, color=color,
                     va="center", ha=align, zorder=2,
                     weight="bold" if weight == "bold" else "normal")
        y -= row_h
    y -= 0.16
    for line in foot_lines:
        fig.text(fx(x_left), fy(y), line, fontsize=FS_TINY, color=C_MUTE,
                 va="top", zorder=2)
        y -= 0.225
    y -= 0.34

    # ---- trace ------------------------------------------------------------
    src_col = C_MUTE
    if trace_src is None:
        src = "no response_text on any arm for this question — nothing to quote"
    else:
        who = (f"{trace_src['method']} (tag {trace_src['tag']}, "
               f"{trace_src['protocol']}, T={trace_src['temperature']})")
        if trace_blind:
            src = (f"no sighted arm produced any text — falling back to the "
                   f"BLIND arm: {who}, first {len(trace)} of {trace_len} chars")
            src_col = C_BAD
        elif trace_reasons:
            src = (f"reasoning trace excerpt — {who}, "
                   f"first {len(trace)} of {trace_len} chars")
        else:
            src = (f"no sighted reasoning trace in these rows — this leg ran "
                   f"REASONING=0; showing the sighted arm's bare answer — {who}")
    fig.text(fx(M), fy(y), src, fontsize=FS_TINY, color=src_col, va="top")
    y -= 0.30
    box_h = (0.225 * max(1, len(trace_lines)) + 0.18
             + (0.38 if trace_blind else 0.0))
    fig.add_artist(Rectangle((fx(M - 0.12), fy(y - box_h)), fx(W - 2 * M + 0.24),
                                 fy(box_h), fc=C_PANEL, ec="#cccccc", lw=1.0,
                                 transform=fig.transFigure, zorder=0))
    ty = y - 0.16
    if trace_blind:
        # inside the panel, not only in the grey caption above it
        fig.text(fx(M), fy(ty), "BLIND ARM — NO VIDEO", fontsize=FS_HEAD,
                 weight="bold", color=C_BAD, va="top", zorder=3)
        ty -= 0.38
    for line in (trace_lines or ["(empty)"]):
        fig.text(fx(M), fy(ty), line, fontsize=FS_TINY, color=C_INK, va="top",
                 family="DejaVu Sans Mono", zorder=2)
        ty -= 0.225

    fig.savefig(out_path, dpi=dpi, facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------
# figure 3: summary
# --------------------------------------------------------------------------

def fig_summary(questions, out_path, dpi, ladder=None):
    ladder = list(ladder if ladder is not None else LADDER)
    arms = []
    for q in questions:
        for g in q["groups"]:
            key = (g["arm"], g["tag"])
            if key not in arms:
                arms.append(key)
    arms.sort(key=lambda k: (ARM_ORDER.index(k[0]) if k[0] in ARM_ORDER else 99, k[1]))

    W = 19.0
    M = 0.50
    LEFT_W = 9.4                 # the arms x questions block
    AX_L = M + LEFT_W + 3.3      # ladder axes start (the gap holds its y labels)
    AX_W = W - AX_L - M
    TOP_H = 1.05
    ROW_H0 = 0.60
    BOT_H = 1.30
    # tall enough for the arms block and for a readable ladder (six rungs for
    # one scorer, nine when the ViCLIP rungs join), and no taller: reserving
    # four arm rows left a blank band under a 3-arm block
    H = max(7.0, 3.55 + (len(arms) - 0.5) * ROW_H0,
            TOP_H + 0.55 + BOT_H + 0.52 * len(ladder))
    ROW_H = min(0.95, max(ROW_H0, (H - 3.55) / max(1, len(arms))))

    fig = plt.figure(figsize=(W, H), dpi=dpi)
    fig.patch.set_facecolor("white")

    def fx(x):
        return x / W

    def fy(y):
        return y / H

    fig.text(fx(M), fy(H - 0.34), "Four-camera MEVA demo set \u2014 arms \u00d7 "
             "questions, beside the meva1033 reference ladder",
             fontsize=FS_TITLE, weight="bold", color=C_INK, va="top")

    # ---- left: arms x questions ------------------------------------------
    lab_w = 3.15
    n_q = len(questions)
    tail_w = 1.40                    # the passes-correct column, plus "pass 1:"
    cell_w = (LEFT_W - lab_w - tail_w) / max(1, n_q)
    tail_x = M + lab_w + n_q * cell_w + tail_w / 2

    y = H - TOP_H
    fig.text(fx(M), fy(y), "pass-1 prediction and whether it matched the gold "
             "letter, with the arm's passes-correct fraction under it; "
             "\u2013 = arm not run on that question",
             fontsize=FS_TINY, color=C_MUTE, va="top")
    y -= 0.52

    fig.text(fx(M), fy(y), "Arm", fontsize=FS_BODY, weight="bold", color=C_INK,
             va="center")
    for j, q in enumerate(questions):
        x = M + lab_w + j * cell_w + cell_w / 2
        fig.text(fx(x), fy(y), f"Q{q['id']}", fontsize=FS_BODY, weight="bold",
                 color=C_INK, va="center", ha="center")
        qt = q["question_type"] or "?"
        fig.text(fx(x), fy(y - 0.25), shorten(TYPE_SHORT.get(qt, qt), 11),
                 fontsize=FS_TINY, color=C_MUTE, va="center", ha="center")
    fig.text(fx(tail_x), fy(y), "passes", fontsize=FS_BODY, weight="bold",
             color=C_INK, va="center", ha="center")
    fig.text(fx(tail_x), fy(y - 0.25), "correct", fontsize=FS_BODY, weight="bold",
             color=C_INK, va="center", ha="center")
    y -= 0.58

    for i, (arm, tag) in enumerate(arms):
        if i % 2 == 1:
            fig.add_artist(Rectangle((fx(M - 0.14), fy(y - ROW_H / 2)),
                                     fx(LEFT_W + 0.28), fy(ROW_H),
                                     fc="#f4f4f1", ec="none",
                                     transform=fig.transFigure, zorder=0))
        label = ARM_LABEL.get(arm, arm.replace("other:", ""))
        fig.text(fx(M), fy(y + 0.09), shorten(label, 34), fontsize=FS_BODY,
                 weight="bold", color=C_INK, va="center", zorder=2)
        fig.text(fx(M), fy(y - 0.15), f"tag {tag}", fontsize=FS_TINY, color=C_MUTE,
                 va="center", zorder=2)
        n_ok = n_seen = 0
        p_ok = p_tot = 0
        multi = False
        for j, q in enumerate(questions):
            g = next((g for g in q["groups"] if (g["arm"], g["tag"]) == (arm, tag)),
                     None)
            x = M + lab_w + j * cell_w + cell_w / 2
            if g is None:
                fig.text(fx(x), fy(y), "\u2013", fontsize=FS_HEAD, color="#999999",
                         va="center", ha="center", zorder=2)
                continue
            n_seen += 1
            ok = bool(g["first"].get("correct"))
            n_ok += 1 if ok else 0
            p_ok += g["n_correct"]
            p_tot += g["n_pass"]
            cy = y + (0.11 if g["n_pass"] > 1 else 0.0)
            fig.text(fx(x), fy(cy),
                     f"{MARK_OK if ok else MARK_BAD} {g['first'].get('prediction') or '?'}",
                     fontsize=FS_HEAD, weight="bold", color=C_OK if ok else C_BAD,
                     va="center", ha="center", zorder=2)
            if g["n_pass"] > 1:
                # a 1-in-4 fluke and a 4-of-4 must not read the same
                multi = True
                fig.text(fx(x), fy(y - 0.17),
                         f"{g['n_correct']}/{g['n_pass']}", fontsize=FS_TINY,
                         color=C_MUTE, va="center", ha="center", zorder=2)
        if p_tot:
            fig.text(fx(tail_x), fy(y + (0.11 if multi else 0.0)),
                     f"{p_ok}/{p_tot}", fontsize=FS_HEAD, weight="bold",
                     color=C_INK, va="center", ha="center", zorder=2)
            if multi:
                fig.text(fx(tail_x), fy(y - 0.17), f"pass 1: {n_ok}/{n_seen}",
                         fontsize=FS_TINY, color=C_MUTE, va="center",
                         ha="center", zorder=2)
        else:
            fig.text(fx(tail_x), fy(y), "0/0", fontsize=FS_HEAD, weight="bold",
                     color=C_INK, va="center", ha="center", zorder=2)
        y -= ROW_H

    # ---- right: the meva1033 ladder --------------------------------------
    ax_b = BOT_H
    ax_h = (H - TOP_H - 0.55) - ax_b
    ax = fig.add_axes([fx(AX_L), fy(ax_b), fx(AX_W), fy(ax_h)])
    labels = [l for l, _, _ in ladder]
    vals = [v for _, v, _ in ladder]
    kinds = [k for _, _, k in ladder]
    ypos = np.arange(len(ladder))
    bars = ax.barh(ypos, vals, height=0.62, linewidth=0.9, edgecolor="#333333",
                   color=["#BBBBBB" if k == "floor" else "#4477AA" for k in kinds])
    for b, k in zip(bars, kinds):
        if k == "floor":
            b.set_hatch("//")
    for yv, v in zip(ypos, vals):
        ax.text(v + 0.8, yv, f"{v:.2f}", va="center", ha="left", fontsize=FS_BODY,
                weight="bold", color=C_INK)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=FS_BODY, color=C_INK)
    ax.set_xlim(0, max(vals) * 1.25)
    ax.set_xlabel("accuracy on meva1033 (%), InternVL3-8B", fontsize=FS_BODY,
                  color=C_INK)
    ax.tick_params(axis="x", labelsize=FS_TINY)
    ax.set_title("meva1033 reference ladder (1,033 questions)\n"
                 "hatched = floors, not model scores",
                 fontsize=FS_HEAD, weight="bold", color=C_INK, loc="left", pad=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", color="#dddddd", lw=0.7)
    ax.set_axisbelow(True)

    # ---- caveat ----------------------------------------------------------
    fig.text(fx(M), fy(0.62), CAVEAT, fontsize=FS_HEAD, color=C_INK,
             va="center", weight="bold")
    tail = ("The text-only letter floor is a documented skew in the release's "
            "gold letters, not a model result.")
    if NOTES:
        tail += f"  {len(NOTES)} rendering note(s) in manifest.json."
    fig.text(fx(M), fy(0.30), tail, fontsize=FS_TINY, color=C_MUTE, va="center")

    fig.savefig(out_path, dpi=dpi, facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------
# manifest
# --------------------------------------------------------------------------

def manifest_entry(q):
    cams = []
    for c in q["cams"]:
        d = {k: v for k, v in c.items() if not k.startswith("_")}
        cams.append(d)
    arms = []
    for g in q["groups"]:
        r = g["first"]
        arms.append({
            "arm": g["arm"],
            "label": ARM_LABEL.get(g["arm"], g["arm"]),
            "method": g["method"],
            "tag": g["tag"],
            "file": os.path.relpath(g["first"].get("_file", ""), REPO),
            "protocol": g["protocol"],
            "temperature": g["temperature"],
            "n_passes": g["n_pass"],
            "n_correct": g["n_correct"],
            "prediction": r.get("prediction"),
            "gold": r.get("gold"),
            "correct": bool(r.get("correct")),
            "latency_s": r.get("latency_s"),
            "selection_latency_s": selection_latency_of(r),
            "input_tokens": r.get("input_tokens"),
            "video_tokens": r.get("video_tokens"),
            "predictions_all_passes": [x.get("prediction") for x in g["rows"]],
        })
    trace, src, tlen, t_reasons, t_blind = trace_text(q["groups"],
                                                      prefer=q.get("sel"))
    return {
        "id": q["id"],
        "question": q["rec"].get("question"),
        "options": q["rec"].get("options"),
        "gold": q["gold"],
        "question_type": q["question_type"],
        "task_type": q["task_type"],
        "orig_id": q["rec"].get("orig_id"),
        "requires_cameras": q["requires_cameras"],
        "selection_leg": None if not q["sel"] else {
            "method": q["sel"]["method"], "tag": q["sel"]["tag"],
            "protocol": q["sel"]["protocol"],
            "pass_idx": q["sel"]["first"].get("pass_idx"),
            "scorer": q["seg_scorer"], "budget": q["budget"],
            "frames_per_segment": q["frames_per_segment"],
            "segments_keep": q["segments_keep"],
        },
        "global_leg": None if not q["global"] else {
            "method": q["global"]["method"], "tag": q["global"]["tag"],
        },
        "event_windows": q["events"],
        "cameras": cams,
        "arms": arms,
        "trace_excerpt": trace,
        "trace_source": None if src is None else {
            "method": src["method"], "tag": src["tag"], "arm": src["arm"],
            "protocol": src["protocol"],
        },
        "trace_is_reasoning": t_reasons,
        "trace_is_blind": t_blind,
        "trace_full_chars": tlen,
    }


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Render the four-camera MEVA demo figures (selection, "
                    "answer, summary) from a demo subset and result rows.")
    ap.add_argument("--subset", default=DEFAULT_SUBSET,
                    help=f"demo subset JSON (default {DEFAULT_SUBSET})")
    ap.add_argument("--rows", nargs="+", default=None,
                    help="result JSONL paths or globs (default: the three demo "
                         "files, falling back to the rescored mp4sg96/mp4fs96/"
                         "mp4bd legs when none of those exist)")
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help=f"output directory (default {DEFAULT_OUT})")
    ap.add_argument("--ids", nargs="+", type=int, default=None,
                    help="render only these question ids")
    ap.add_argument("--selection-method", default=DEFAULT_SELECTION_METHOD,
                    help="the segment_select method whose rows drive the "
                         "selection figure, the manifest's cameras[] block and "
                         "its selection_leg/global_leg entries (default "
                         f"{DEFAULT_SELECTION_METHOD}; the ViCLIP legs are "
                         f"{VICLIP_SELECTION_METHOD}). Rows of the other "
                         "scorer, when passed, still appear as arms in the "
                         "answer panel and the summary grid")
    ap.add_argument("--thumbs-per-cam", type=int, default=6,
                    help="max kept-frame thumbnails per camera row (default 6)")
    ap.add_argument("--dpi", type=int, default=150, help="figure dpi (default 150)")
    ap.add_argument("--video-root", default=None,
                    help="release root the record video paths are relative to "
                         f"(default {os.path.relpath(RELEASE, REPO)})")
    ap.add_argument("--no-bench-reuse", action="store_true",
                    help="skip importing bench.reuse and use the local "
                         ".avi->.mp4 replica (faster: no torch import)")
    args = ap.parse_args()

    out_dir = rel(args.out)
    os.makedirs(out_dir, exist_ok=True)
    video_root = rel(args.video_root) if args.video_root else RELEASE

    subset_path = rel(args.subset)
    with open(subset_path) as fh:
        recs = json.load(fh)
    if args.ids:
        want = set(args.ids)
        missing = want - {r.get("id") for r in recs}
        if missing:
            note(f"--ids not in the subset, ignored: {sorted(missing)}")
        recs = [r for r in recs if r.get("id") in want]
    if not recs:
        print("nothing to render: the subset (after --ids) is empty")
        return 1
    ids = {r.get("id") for r in recs}

    patterns = args.rows
    if patterns is None:
        have = expand_rows(DEFAULT_ROWS)
        if have:
            patterns = DEFAULT_ROWS
        else:
            note("none of the demo result files exist yet; falling back to the "
                 "rescored meva1033 InternVL3-8B legs")
            patterns = FALLBACK_ROWS
    paths = expand_rows(patterns)
    if not paths:
        print(f"no result files matched: {patterns}")
        return 1

    print(f"subset      {os.path.relpath(subset_path, REPO)} "
          f"({len(recs)} question(s): {sorted(ids)})")
    print(f"selection   {args.selection_method}")
    print("rows")
    rows = load_rows(paths, ids)
    if not rows:
        note("no result rows matched the demo ids; figures will be mostly empty")

    vp_fn, vp_src = get_video_paths_fn(use_reuse=not args.no_bench_reuse)
    print(f"media       resolved through {vp_src}")
    print(f"out         {os.path.relpath(out_dir, REPO)}")

    by_id = {}
    for r in rows:
        by_id.setdefault(r.get("id"), []).append(r)

    written = []
    questions = []
    for rec in recs:
        qid = rec.get("id")
        qrows = by_id.get(qid, [])
        if not qrows:
            note(f"id {qid}: no result rows in any of the given files")
        q = build_question(rec, qrows, vp_fn, video_root, args.thumbs_per_cam,
                           args.selection_method)
        questions.append(q)

        p1 = os.path.join(out_dir, f"q{qid}_selection.png")
        fig_selection(q, p1, args.dpi)
        written.append(p1)
        p2 = os.path.join(out_dir, f"q{qid}_answer.png")
        fig_answer(q, p2, args.dpi)
        written.append(p2)
        print(f"  q{qid}: {os.path.basename(p1)}  {os.path.basename(p2)}")

    p3 = os.path.join(out_dir, "summary.png")
    ladder = ladder_for(questions)
    fig_summary(questions, p3, args.dpi, ladder=ladder)
    written.append(p3)

    manifest = {
        "generated_by": "analysis/render_demo.py",
        "subset": os.path.relpath(subset_path, REPO),
        "row_files": [os.path.relpath(p, REPO) for p in paths],
        "media_resolution": vp_src,
        "video_root": os.path.relpath(video_root, REPO),
        "thumbs_per_cam": args.thumbs_per_cam,
        "dpi": args.dpi,
        "ladder_meva1033": [{"label": l, "value": v, "kind": k} for l, v, k in ladder],
        "caveat": CAVEAT,
        "questions": [manifest_entry(q) for q in questions],
        "figures": [os.path.relpath(p, REPO) for p in written],
        "notes": NOTES,
    }
    p4 = os.path.join(out_dir, "manifest.json")
    with open(p4, "w") as fh:
        json.dump(manifest, fh, indent=1, default=str)
    written.append(p4)

    print()
    print(CAVEAT)
    print()
    for p in written:
        sz = os.path.getsize(p) / 1e6
        print(f"  {os.path.relpath(p, REPO)}  ({sz:.2f} MB)")
    if NOTES:
        print(f"\n{len(NOTES)} note(s) recorded in manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
