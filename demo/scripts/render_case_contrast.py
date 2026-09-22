#!/usr/bin/env python3
"""The two-case example for one CrossView-MEVA question, drawn from recorded rows only.

Both images are panels that sit under a slide title: 11.33 x 5.6 in at 200 dpi (2266 x 1120 px), no
headline of their own, nothing under 11 pt at that size.

  case1_q<ID>.png   "One camera at a time": one row per camera that carries an annotated
                    event - the exact 8 frames the single-camera run received (frames
                    inside the window outlined), a clip timeline with the event window,
                    and the model's verbatim answers over the 4 passes, marked
                    right/wrong. Cameras with no annotated event, and the no-video run,
                    are one text line each.
  case2_q<ID>.png   "All cameras together": for the winning run, the received frames that
                    fall inside an annotated window, enlarged, per event camera; a strip
                    of where every received frame falls on each camera's timeline (the
                    selection row's frame_alloc.selected_times_s, or the uniform grid for
                    an all-cameras arm); the answer and the gold.
  case_q<ID>.json   every value drawn on the two figures, plus the verbatim question and
                    options (the figures abbreviate them; SHORT below).
  --vet IDS         also/only write case_vetting.md: the same facts for several
                    candidate questions, ranked by how defensible the story is.

Nothing is inferred from a leg name. Answers are the rows' ``response_text``; the
frames are re-decoded at the indices the runner used:

  single camera / all cameras   InternVL get_index (MultiCam bench/backends/internvl.py):
                                seg = (n_frames - 1) / N,
                                index_i = int(seg / 2 + round(seg * i)), i = 0..N-1
  selection arms                round(selected_times_s * fps); the runner stores
                                frame_index / fps (bench/methods/segment_select.py)

``single_view<N>`` is delivered camera N: SingleViewMethod keeps content pair N of
build_messages, which lists video_1..video_K in slot order (bench/methods/single_view.py,
Video-R1/src/eval_thinking.py::video_paths).

TIME. Release event times are clip-relative, and the clips of one question do not
start together: the file name carries each clip's wall-clock start
(``2018-03-07.11-00-06.11-05-05.school.G336`` starts at 11:00:06). Event order is
therefore re-derived in wall-clock time, with the file names' 1-second resolution
as the error bar.

The disclosure printed on both figures is read from motivation.json (written by
evaluation/multicam_motivation.py): the any-of-5-runs "fixes" and "breaks" counts on the
multi-camera questions. The whole-pool count recomputed here from the legs must agree with
it, and a rendered question must be in its id list. The counter-evidence line (runs that
had frames inside every event window and were still wrong, runs that were right without
one) is built from the rows of the question.

CPU only (CUDA_VISIBLE_DEVICES is masked), never loads a model. Needs decord +
matplotlib: run under the cvbench env from the repo root:

  ~/anaconda3/envs/cvbench/bin/python demo/scripts/render_case_contrast.py \
      --release-root ~/MultiCam/crossview-release-annotations/crossview-release \
      --vet 750 29 144 964 954 990 819 823 885 --qid 954 990 --out docs/figs/demo_2026-09-22
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse  # noqa: E402
import gzip  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import textwrap  # noqa: E402
from collections import defaultdict  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from plotting.selection_by_task import INK, INK2, GRID, SURFACE, ARM_COLOR  # noqa: E402

POOL = "crossview_meva1033_subset"
LEG = os.path.join(REPO, "multicam_results", "legs", POOL + "_internvl_%s.jsonl.gz")
SUBSET = os.path.join(REPO, "data", "subsets", POOL + ".json")
QA_FILE = {"temporal": "qa_temporal.json", "event_ordering": "qa_event_ordering.json"}
SINGLE, BLIND = "mp4sv8d", "mp4bdd"
# the same single-camera run under the reasoning protocol (think, then answer; T = 0.7); only used for the
# counter-evidence line, and its temperature is read from the rows
SINGLE_REASONING = "mp4sv8"
# multi-camera arms, in the order a winning arm is looked for: leg tag, plain label, total frames.
# The all-cameras labels are for the four-camera questions this script renders; a question with
# another camera count gets its own label in build_case.
ARMS = [("mp4sg96", "SigLIP selection, 96 frames", 96),
        ("mp4sgva96", "ViCLIP selection, 96 frames", 96),
        ("mp4fs96", "all four cameras, 24 frames each", 96),
        ("mp4fs64", "all four cameras, 16 frames each", 64),
        ("mp4fs32", "all four cameras, 8 frames each", 32)]
NUMBER = {2: "two", 3: "three", 4: "four"}
SINGLE_FRAMES = 8
PASSES = 4
RIGHT, WRONG, WINDOW = "#12805a", "#c8372d", ARM_COLOR["siglip"]
WINDOW_INK = "#b0431a"      # the window orange, dark enough for text
MOTIVATION = os.path.join(REPO, "docs", "figs", "demo_2026-09-22", "motivation.json")
# Honest abbreviations for the slide; the verbatim question and options stay in case_q<ID>.json.
# "events" names each annotated event (gold order) in one word. A question without an entry
# falls back to the verbatim text.
SHORT = {
    954: {"question": "Which happened first: one person physically interacting with someone, or another person "
                      "(hat, bag) talking to someone?",
          "options": {"A": "the interaction first", "B": "the talking first", "C": "simultaneous",
                      "D": "cannot be determined"},
          "events": {"E1": "talking", "E2": "interaction"}},
    990: {"question": "Which happened first: a person exiting through a structure, or that same person reading a "
                      "document?",
          "options": {"A": "the exit first", "B": "the reading first", "C": "simultaneous",
                      "D": "cannot be determined"},
          "events": {"E1": "reading", "E2": "exit"}},
}
CLOCK_RES_S = 1.0          # file-name start times are whole seconds
PROTOCOL = "InternVL3-8B · direct answer, T = 0.1 · 4 passes"


# ----------------------------------------------------------------------------- rows
def read_leg(tag):
    with gzip.open(LEG % tag, "rt") as fh:
        return [json.loads(line) for line in fh]


def by_question(rows):
    """id -> method -> rows sorted by pass."""
    out = defaultdict(lambda: defaultdict(list))
    for r in rows:
        out[r["id"]][r["method"]].append(r)
    for methods in out.values():
        for rs in methods.values():
            rs.sort(key=lambda r: r["pass_idx"])
    return out


def load_legs():
    return {tag: by_question(read_leg(tag)) for tag in [SINGLE, SINGLE_REASONING, BLIND] + [t for t, _, _ in ARMS]}


def n_right(rows):
    assert len(rows) == PASSES, f"expected {PASSES} passes, got {len(rows)}"
    return sum(bool(r["correct"]) for r in rows)


def only(methods):
    (rows,) = methods.values()
    return rows


def behaviour_counts(legs, ids):
    """How often the chosen pattern, and its reverse, happen in the whole pool."""
    def views(i):
        return [n_right(rs) for _, rs in sorted(legs[SINGLE][i].items())]

    def arm(tag, i):
        return n_right(only(legs[tag][i]))

    strict = [i for i in ids if max(views(i)) == 0 and arm(BLIND, i) == 0
              and any(arm(t, i) == PASSES for t, _, _ in ARMS)]
    reverse = [i for i in ids if max(views(i)) == PASSES and arm("mp4fs96", i) == 0]
    return {"n": len(ids), "needs_all_cameras": len(strict), "needs_all_cameras_ids": strict,
            "one_camera_right_all_96_wrong": len(reverse)}


def motivation_counts(path, behaviour):
    """The any-of-5-runs counts on the multi-camera questions, from motivation.json, cross-checked
    against the whole-pool count recomputed from the legs here."""
    with open(path) as fh:
        den = json.load(fh)["behaviour"]["by_denominator"]
    assert den["all"]["any_of_5_runs"]["fixes"]["n"] == behaviour["needs_all_cameras"], "motivation.json is stale"
    assert sorted(den["all"]["any_of_5_runs"]["fixes"]["ids"]) == sorted(behaviour["needs_all_cameras_ids"])
    multi = den["multi"]
    return {"source": os.path.relpath(path, REPO), "n_multi_camera_questions": multi["n"], "n_runs": len(ARMS),
            "fixes": multi["any_of_5_runs"]["fixes"]["n"], "fixes_ids": multi["any_of_5_runs"]["fixes"]["ids"],
            "breaks": multi["any_of_5_runs"]["breaks"]["n"],
            "breaks_and_no_video_not_4of4": multi["any_of_5_runs"]["breaks_and_no_video_not_4of4"]["n"]}


def disclosure(m):
    assert m["breaks_and_no_video_not_4of4"] > m["fixes"], "the reverse flip is not more common"
    return ("Chosen example: %d of the %s multi-camera questions behave like this in at least one of %d runs; an "
            "always-right single-camera answer turning always-wrong is more common (%d)."
            % (m["fixes"], format(m["n_multi_camera_questions"], ","), m["n_runs"], m["breaks_and_no_video_not_4of4"]))


def counter_evidence(case):
    """Runs that do not fit 'right because it saw both events': built from this question's rows."""
    n_ev = len(case["events"])
    names = case["short"]["events"]
    every = "both event windows" if n_ev == 2 else "all %d event windows" % n_ev
    chosen = case["case2_arm"]
    had, without, neither = [], [], []
    for tag, a in case["arms"].items():
        if tag == chosen:
            continue
        missed = [names[k] for k, v in a["frames_in_window"].items() if v == 0]
        if a["n_right"] < PASSES and not missed:
            had.append("%s (%d of %d)" % (a["label"], a["n_right"], PASSES))
        elif a["n_right"] < PASSES and len(missed) == n_ev and a.get("scorer"):
            neither.append("%s (%d of %d)" % (a["label"], a["n_right"], PASSES))
        elif a["n_right"] == PASSES and missed:
            without.append("%s (no frame inside the %s event)" % (a["label"], " or ".join(missed)))
    parts = []
    if had:
        parts.append("frames inside %s, still not right: %s" % (every, "; ".join(had)))
    if neither:
        parts.append("no frame inside either window: " + "; ".join(neither))
    if without:
        parts.append("right %d of %d anyway: %s" % (PASSES, PASSES, "; ".join(without)))
    # a camera that is 0 of 4 here but 4 of 4 alone under the reasoning protocol: one camera can be enough
    for c in case["cameras"]:
        r = c["single_reasoning"]
        if r["n_right"] == PASSES and c["single"]["n_right"] < PASSES:
            parts.append("under the reasoning protocol (T = %s) camera %s alone is right %d of %d"
                         % (r["temperature"], c["camera"], r["n_right"], PASSES))
    if not parts:
        return "Counter-evidence: none among the other runs on this question."
    return "Counter-evidence, other runs on this question: " + ". ".join(p[0].upper() + p[1:] for p in parts) + "."


def short_text(case):
    """The slide's abbreviated question/options and one-word event names (SHORT), else the verbatim text."""
    s = SHORT.get(case["id"])
    if s is None:
        return {"abbreviated": False, "question": case["question"],
                "options": {o.split(".", 1)[0].strip(): o.split(".", 1)[1].strip() for o in case["options"]},
                "events": {e["label"]: e["activity"].replace("_", " ") for e in case["events"]}}
    assert sorted(s["options"]) == sorted(o.split(".", 1)[0].strip() for o in case["options"])
    assert sorted(s["events"]) == sorted(e["label"] for e in case["events"])
    return dict(s, abbreviated=True)


def reading(cam, case):
    """One sentence per event camera. Keeps 'the event is on another camera' apart from 'the event is on
    this camera but between the sampled frames'."""
    names = case["short"]["events"]
    parts = []
    for e in cam["events"]:
        hit = any(inside(t, e) for t in cam["single"]["times_s"])
        parts.append("the %s event is on this camera %s" % (
            names[e["label"]], "and inside a sampled frame" if hit else "but falls between the sampled frames"))
    for e in case["events"]:
        if e["camera"] != cam["camera"]:
            parts.append("the %s event is on another camera (%s)" % (names[e["label"]], e["camera"]))
    text = "; ".join(parts)
    return text[0].upper() + text[1:] + "."


# ----------------------------------------------------------------------------- annotation
def clip_start_s(path):
    """Wall-clock start of a MEVA clip, seconds since midnight, from its file name."""
    m = re.search(r"\d{4}-\d\d-\d\d\.(\d\d)-(\d\d)-(\d\d)\.\d\d-\d\d-\d\d\.", os.path.basename(path))
    if not m:
        raise ValueError("no start time in " + path)
    h, mi, s = map(int, m.groups())
    return 3600 * h + 60 * mi + s


def clock(sec):
    sec = float(sec)
    return "%02d:%02d:%04.1f" % (sec // 3600, sec % 3600 // 60, sec % 60)


def camera_of(path):
    return re.search(r"\.(G\d+)\.", os.path.basename(path)).group(1)


def release_item(rec, release_root):
    qt = rec["question_type"]
    if qt not in QA_FILE:
        raise ValueError(f"question {rec['id']} is '{qt}': the release annotates no event windows for it")
    path = os.path.join(release_root, "annotations", "multi-cam-dataset", "meva", QA_FILE[qt])
    with open(path) as fh:
        item = json.load(fh)[int(rec["orig_id"].rsplit("#", 1)[1])]
    assert item["question"].strip() == rec["question"].strip(), "release item does not match the pool question"
    return item


def annotated_events(item):
    """The events the gold is derived from, in the release's gold order."""
    meta = item["metadata"]
    if item["question_type"] == "temporal":
        raw = [meta["debug_info"]["event_a"], meta["debug_info"]["event_b"]]
        desc = [meta["verification"]["event_a"]["description"], meta["verification"]["event_b"]["description"]]
    else:
        raw = meta["debug_info"]["events"]
        desc = [e["description"] for e in meta["verification"]["ordered_events"]]
    out = []
    for e, d in zip(raw, desc):
        a, b = (float(x) for x in e["timestamp"].rstrip("s").split("-"))
        out.append({"camera": e["camera"], "activity": e["activity"], "description": d,
                    "clip_file": e["clip_file"], "start_s": a, "end_s": b})
    return out


def roman_labels(item, events):
    """Event-ordering questions name their events I..IV in the question text; map gold order -> numeral."""
    if item["question_type"] != "event_ordering":
        return ["E%d" % (k + 1) for k in range(len(events))]       # not A/B: those are option letters
    order = item["options"][item["answer"]].replace(" ", "").split("->")
    assert len(order) == len(events)
    return order


# ----------------------------------------------------------------------------- frames
def uniform_indices(n_frames, n):
    """InternVL get_index with no bound (MultiCam bench/backends/internvl.py)."""
    seg = float(n_frames - 1) / n
    return [int(seg / 2 + round(seg * i)) for i in range(n)]


def resolve_mp4(release_root, rel):
    """Pool records spell MEVA clips .avi; only the verified .mp4 sibling may reach a decoder."""
    p = os.path.join(release_root, os.path.splitext(rel)[0] + ".mp4")
    if not os.path.exists(p):
        raise SystemExit("missing .mp4 sibling: " + p)
    return p


class Clip:
    def __init__(self, path):
        from decord import VideoReader, cpu
        self.vr = VideoReader(path, ctx=cpu(0), num_threads=1)
        self.n = len(self.vr)
        self.fps = float(self.vr.get_avg_fps())

    def thumb(self, index, width):
        from PIL import Image
        index = min(max(0, int(index)), self.n - 1)
        im = Image.fromarray(self.vr[index].asnumpy()).convert("RGB")
        return im.resize((width, round(width * im.height / im.width)))


def inside(t, ev):
    return ev["start_s"] <= t <= ev["end_s"]


# ----------------------------------------------------------------------------- the case
def build_case(qid, release_root, legs, pool, arm=None, decode=True):
    rec = pool[qid]
    item = release_item(rec, release_root)
    events = annotated_events(item)
    for e, lab in zip(events, roman_labels(item, events)):
        e["label"] = lab
    slots = [rec[f"video_{k}"] for k in range(1, 14) if rec.get(f"video_{k}")]
    # frame count and fps as the run decoded them (recorded on every selection row)
    ran = only(legs[ARMS[0][0]][qid])[0]["frame_alloc"]["per_video_decode"]
    assert len(ran) == len(slots)
    cams = []
    for k, (rel, d) in enumerate(zip(slots, ran), 1):
        path = resolve_mp4(release_root, rel)
        clip = Clip(path) if decode else None
        if clip:
            assert (clip.n, clip.fps) == (d["n_total"], d["fps"]), "this decode differs from the run's: " + path
        cams.append({"slot": k, "camera": camera_of(rel), "file": os.path.basename(path), "path": path,
                     "clip_start_s": clip_start_s(rel), "clip_start": clock(clip_start_s(rel)),
                     "n_frames": d["n_total"], "fps": d["fps"], "_clip": clip})
    by_cam = {c["camera"]: c for c in cams}
    for e in events:
        c = by_cam.get(e["camera"])
        e["delivered"] = c is not None
        start = clip_start_s(e["clip_file"])
        if c:
            assert e["clip_file"] == c["file"], (e["clip_file"], c["file"])
        e["wall_start_s"], e["wall_end_s"] = start + e["start_s"], start + e["end_s"]
        e["wall_start"], e["wall_end"] = clock(e["wall_start_s"]), clock(e["wall_end_s"])

    # gold order re-derived in wall-clock time
    gaps = []
    for a, b in zip(events, events[1:]):
        gaps.append({"from": a["label"], "to": b["label"],
                     "clip_relative_start_gap_s": round(b["start_s"] - a["start_s"], 2),
                     "wall_start_gap_s": round(b["wall_start_s"] - a["wall_start_s"], 2),
                     "wall_gap_after_end_s": round(b["wall_start_s"] - a["wall_end_s"], 2)})
    min_gap = min(g["wall_start_gap_s"] for g in gaps)
    order = {"gaps": gaps, "min_wall_start_gap_s": min_gap,
             "holds_in_wall_clock": min_gap > 0,
             "robust": min_gap > 2 * CLOCK_RES_S,     # each of the two clip starts is +-1 s
             "overlap_in_wall_clock": any(g["wall_gap_after_end_s"] < 0 for g in gaps)}

    # single-camera runs
    sv = legs[SINGLE][qid]
    assert sorted(sv) == [f"single_view{k}" for k in range(1, len(cams) + 1)], sorted(sv)
    for c in cams:
        rows = sv[f"single_view{c['slot']}"]
        assert all(r["frame_alloc"]["view_idx"] == c["slot"] for r in rows)
        assert all(r["video_tokens"] == 256 * SINGLE_FRAMES for r in rows), "single-camera run is not 8 frames"
        c["events"] = [e for e in events if e["camera"] == c["camera"]]
        c["single"] = {"answers": [r["response_text"] for r in rows], "predictions": [r["prediction"] for r in rows],
                       "right": [bool(r["correct"]) for r in rows], "n_right": n_right(rows)}
        idx = uniform_indices(c["n_frames"], SINGLE_FRAMES)
        times = [round(i / c["fps"], 2) for i in idx]
        c["single"].update(frame_indices=idx, times_s=times,
                           in_window=[[e["label"] for e in c["events"] if inside(t, e)] for t in times])
        c["single"]["evidence"] = evidence_status(c["events"], times)
        rows = legs[SINGLE_REASONING][qid][f"single_view{c['slot']}"]
        assert all(r["frame_alloc"]["view_idx"] == c["slot"] and r["video_tokens"] == 256 * SINGLE_FRAMES
                   and (r["think"] or r["response_text"].lstrip().startswith("<think>")) and not r["error"]
                   for r in rows), "not an 8-frame reasoning run of this camera"
        (temperature,) = {r["temperature"] for r in rows}
        c["single_reasoning"] = {"leg": rows[0]["leg"], "temperature": temperature, "n_right": n_right(rows),
                                 "predictions": [r["prediction"] for r in rows]}

    blind = only(legs[BLIND][qid])
    arms = {}
    for tag, label, total in ARMS:
        rows = only(legs[tag][qid])
        arms[tag] = {"label": label, "answers": [r["response_text"] for r in rows],
                     "predictions": [r["prediction"] for r in rows], "n_right": n_right(rows)}
        alloc = rows[0]["frame_alloc"]
        if alloc.get("selected_times_s"):
            assert all(r["frame_alloc"]["selected_times_s"] == alloc["selected_times_s"] for r in rows), \
                "selection differs between passes"
            times = {int(k): v for k, v in alloc["selected_times_s"].items()}
            arms[tag]["scorer"] = alloc.get("segment_scorer")
            arms[tag]["frame_source"] = "frame_alloc.selected_times_s"
        else:
            per_view = alloc["per_view"]
            assert alloc["total_frames"] == total == sum(per_view) and len(set(per_view)) == 1, alloc
            arms[tag]["label"] = "all %s cameras, %d frames each" % (NUMBER.get(len(cams), len(cams)), per_view[0])
            times = {c["slot"]: [round(i / c["fps"], 2) for i in uniform_indices(c["n_frames"], n)]
                     for c, n in zip(cams, per_view)}
            arms[tag]["frame_source"] = "uniform grid (InternVL get_index)"
        arms[tag]["times_s"] = times
        arms[tag]["frames_in_window"] = {
            e["label"]: sum(inside(t, e) for t in times[by_cam[e["camera"]]["slot"]]) if e["delivered"] else 0
            for e in events}
        arms[tag]["windows_reached"] = sum(v > 0 for v in arms[tag]["frames_in_window"].values())
    winners = [t for t, _, _ in ARMS if arms[t]["n_right"] == PASSES]
    # case 2 shows the winning arm whose frames reach the most annotated windows (ties: ARMS order)
    chosen = arm or (max(winners, key=lambda t: arms[t]["windows_reached"]) if winners else None)

    return {"id": qid, "pool": POOL, "orig_id": rec["orig_id"], "task_type": rec["task_type"],
            "protocol": PROTOCOL, "question": rec["question"], "options": rec["options"], "gold": rec["answer"],
            "release_reasoning": item.get("reasoning"),
            "requires_cameras": item["metadata"]["requires_cameras"],
            "cameras": cams, "events": events, "gold_order_wall_clock": order,
            "blind": {"answers": [r["response_text"] for r in blind], "n_right": n_right(blind)},
            "arms": arms, "arms_right_on_all_passes": winners, "case2_arm": chosen}


def reading_short(cam, case):
    """The heading-row version of ``reading``: which event is on this camera and which is on another."""
    names = case["short"]["events"]
    here = ["%s event on this camera" % names[e["label"]] for e in cam["events"]]
    there = ["%s event on camera %s" % (names[e["label"]], e["camera"]) for e in case["events"]
             if e["camera"] != cam["camera"]]
    return "; ".join(here + there)


def evidence_status(cam_events, times):
    """Separates 'event not on this camera' from 'event on this camera but not in the sampled frames'."""
    if not cam_events:
        return "no annotated event on this camera"
    hit = [e["label"] for e in cam_events if any(inside(t, e) for t in times)]
    miss = [e["label"] for e in cam_events if e["label"] not in hit]
    parts = []
    if hit:
        parts.append("event %s on this camera and in the sampled frames" % ", ".join(hit))
    if miss:
        parts.append("event %s on this camera but not in the sampled frames" % ", ".join(miss))
    return "; ".join(parts)


def public(case):
    """The case without decoder handles or absolute paths."""
    out = json.loads(json.dumps(case, default=lambda o: None))
    for c in out["cameras"]:
        c.pop("_clip", None)
        c.pop("path", None)
        for e in c.get("events", []):
            e.pop("clip_file", None)
    return out


# ----------------------------------------------------------------------------- drawing
# The panel sits under a slide title, 11.33 in wide and at most 5.6 in tall, so the canvas is exactly that
# and point sizes are the sizes on the slide. Layout is in inches from the top-left corner. Nothing is
# smaller than F_MIN = 11 pt; the right/wrong counts are the largest type.
W_IN, H_IN, DPI = 11.33, 5.6, 200            # 2266 x 1120
F_NUM2, F_NUM, F_KEY, F_BODY, F_MIN = 24, 18, 15, 12, 11
M = 0.15                                     # side margin, inches
X0, X1 = M, W_IN - M
LINE = 1.22
PNG_BUDGET = 1_900_000
CAPTION = "outlined frame marks timing only – people are small in this view"


def save_png(fig, out, name):
    """Fixed canvas (no tight bbox). Full-colour PNG if it fits the budget; the frames are photographs, so
    otherwise an adaptive palette. No .svg twin for the same reason."""
    import io
    from PIL import Image
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, facecolor=SURFACE)
    plt.close(fig)
    im = Image.open(buf).convert("RGB")
    assert im.size == (round(W_IN * DPI), round(H_IN * DPI)), im.size
    path = os.path.join(out, name + ".png")
    im.save(path, optimize=True)
    colours = "all"
    for colours in (() if os.path.getsize(path) <= PNG_BUDGET else (256, 128, 64)):
        im.quantize(colours, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.FLOYDSTEINBERG).save(
            path, optimize=True)
        if os.path.getsize(path) <= PNG_BUDGET:
            break
    assert os.path.getsize(path) <= PNG_BUDGET, path
    print("%s: %d x %d, %.2f MB, %s colours" % (path, im.width, im.height, os.path.getsize(path) / 1e6, colours))


def canvas():
    return plt.figure(figsize=(W_IN, H_IN), dpi=DPI, facecolor=SURFACE)


def put(fig, x, y, s, size, colour=INK, weight="normal", ha="left", va="top", **kw):
    """Text at (x, y) inches from the top-left corner."""
    assert size >= F_MIN, "text under %d pt" % F_MIN
    return fig.text(x / W_IN, 1 - y / H_IN, s, fontsize=size, color=colour, fontweight=weight, ha=ha, va=va,
                    linespacing=LINE, **kw)


def width_in(fig, s, size, weight="normal"):
    t = fig.text(0, 0, s, fontsize=size, fontweight=weight)
    w = t.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi
    t.remove()
    return w


def fit(fig, s, size, width, weight="normal"):
    """Wrap ``s`` so that every line fits ``width`` inches at ``size`` pt, by measuring it."""
    for chars in range(len(s) + 1, 20, -2):
        wrapped = textwrap.fill(s, chars)
        if width_in(fig, wrapped, size, weight) <= width:
            return wrapped
    return wrapped


def height_in(s, size):
    return (s.count("\n") + 1) * size * LINE / 72


def para(fig, y, s, size, colour=INK, weight="normal", x=X0, width=None):
    """Wrapped text with its top at ``y`` inches; returns the y just under it."""
    s = fit(fig, s, size, width if width is not None else X1 - x, weight)
    put(fig, x, y, s, size, colour, weight)
    return y + height_in(s, size)


def rule(fig, y):
    fig.add_artist(plt.Line2D([X0 / W_IN, X1 / W_IN], [1 - y / H_IN] * 2, color=GRID, lw=0.9))


def axes(fig, x, y, w, h):
    """Axes with its top-left corner at (x, y) inches."""
    return fig.add_axes([x / W_IN, 1 - (y + h) / H_IN, w / W_IN, h / H_IN])


def runs(answers):
    """Verbatim answers over the passes, identical neighbours folded: ['A','D. x','D. x'] -> 'A · D. x ×2'."""
    out = []
    for a in (a.strip().replace("\n", " ") for a in answers):
        if out and out[-1][0] == a:
            out[-1][1] += 1
        else:
            out.append([a, 1])
    if all(len(a) <= 3 or (a[0] in "ABCD" and a[1] in ". )") for a, _ in out):
        # "D. Cannot be determined" is drawn as its letter; the verbatim text stays in the case JSON
        return "  ".join(a[0] if len(a) > 3 else a for a, n in out for _ in range(n))
    return " · ".join(a if n == 1 else "%s ×%d" % (a, n) for a, n in out)


def verdict(n):
    return "%s %d of %d" % ("✓" if n == PASSES else "✗", n, PASSES)


def verdict_colour(n):
    return RIGHT if n == PASSES else WRONG


def header(fig, case):
    """The abbreviated question and options (verbatim text is in case_q<ID>.json); the slide supplies the
    title. Returns the y under it."""
    y = para(fig, 0.07, case["short"]["question"], F_BODY + 0.5, INK)
    x = X0
    for letter, text in case["short"]["options"].items():
        gold = letter == case["gold"]
        s = "%s. %s%s" % (letter, text, "  ← annotated answer" if gold else "")
        put(fig, x, y + 0.02, s, F_BODY, RIGHT if gold else INK2, "semibold" if gold else "normal")
        x += width_in(fig, s, F_BODY, "semibold" if gold else "normal") + 0.22
    assert x - 0.22 <= X1, "options run off the panel"
    return y + 0.02 + height_in("", F_BODY)


def footer(fig, case, legend):
    """Counter-evidence (dark bold), legend and disclosure, the same on both images; stacked up from the
    bottom edge. Returns the y of the rule above it."""
    y = H_IN - 0.07
    for s, colour, weight in ((case["disclosure"], INK, "normal"), (legend, INK2, "normal"),
                              (case["counter_evidence"], INK, "bold")):
        s = fit(fig, s, F_MIN, X1 - X0, weight)
        put(fig, X0, y, s, F_MIN, colour, weight, va="bottom")
        y -= height_in(s, F_MIN) + 0.035
    y -= 0.02
    rule(fig, y)
    return y


def timeline(ax, cam, times, axis=True):
    """A clip timeline: the annotated window shaded, the frames the model received as ticks. The window's
    label is drawn by the caller, clear of the ticks."""
    dur = cam["n_frames"] / cam["fps"]
    ax.set_xlim(0, dur)
    ax.set_ylim(0, 1)
    ax.patch.set_alpha(0)
    ax.add_patch(Rectangle((0, 0.36), dur, 0.28, facecolor=GRID, edgecolor="none"))
    for e in cam["events"]:
        w = max(e["end_s"] - e["start_s"], dur / 300)      # a sub-second event stays visible
        ax.add_patch(Rectangle((e["start_s"], 0.02), w, 0.96, facecolor=WINDOW, alpha=0.45, edgecolor=WINDOW, lw=1.0))
    for t in times:
        hit = any(inside(t, e) for e in cam["events"])
        ax.plot([t, t], [0.06, 0.94], color=WINDOW_INK if hit else INK, lw=2.4 if hit else 1.2, solid_capstyle="butt")
    ax.set_yticks([])
    if axis:
        ticks = [x for x in (0, 60, 120, 180, 240, 300) if x <= dur + 1]
        ax.set_xticks(ticks)
        ax.set_xticklabels(["%d s" % x for x in ticks], fontsize=F_MIN, color=INK2)
        ax.tick_params(length=3, pad=2, colors=INK2)
    else:
        ax.set_xticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def show(ax, im, outlined, stamp=None):
    """A frame; ``stamp`` is its clip time, printed inside the bottom-left corner."""
    ax.imshow(im, aspect="auto")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(WINDOW if outlined else GRID)
        s.set_linewidth(4 if outlined else 0.8)
    if stamp:
        ax.text(0.035, 0.07, stamp, transform=ax.transAxes, fontsize=F_MIN, color="white", ha="left", va="bottom",
                fontweight="semibold", bbox=dict(facecolor=WINDOW_INK if outlined else "#1a1a1a", alpha=0.85,
                                                 edgecolor="none", pad=1.6))


def figure_case1(case, out):
    short = case["short"]["events"]
    fig = canvas()
    y = header(fig, case)
    event_cams = [c for c in case["cameras"] if c["events"]]
    other_cams = [c for c in case["cameras"] if not c["events"]]
    gap = 0.06
    fw = (X1 - X0 - gap * (SINGLE_FRAMES - 1)) / SINGLE_FRAMES
    y += 0.04
    for c in event_cams:
        s = c["single"]
        rule(fig, y)
        y += 0.04
        name = "Camera %s" % c["camera"]
        ans = "%s   %s right" % (runs(s["answers"]), verdict(s["n_right"]))
        put(fig, X0, y, name, F_KEY, INK, "semibold")
        put(fig, X1, y - 0.03, ans, F_NUM, verdict_colour(s["n_right"]), "semibold", ha="right")
        x_mid = X0 + width_in(fig, name, F_KEY, "semibold") + 0.2
        # one line: which event is here and which is elsewhere (whether a frame caught it is on the window label)
        mid = s["reading_short"]
        assert x_mid + width_in(fig, mid, F_BODY) <= X1 - width_in(fig, ans, F_NUM, "semibold") - 0.2, "heading row is full"
        put(fig, x_mid, y + 0.045, mid, F_BODY, INK)
        y += height_in("", F_NUM) - 0.03 + 0.04
        for j, (i, t) in enumerate(zip(s["frame_indices"], s["times_s"])):
            im = c["_clip"].thumb(i, 400)
            fh = fw * im.height / im.width
            show(axes(fig, X0 + j * (fw + gap), y, fw, fh), im, bool(s["in_window"][j]), "%.1f s" % t)
        y += fh + 0.05
        # the window's label sits in its own row above the strip, so it never covers a frame tick
        dur = c["n_frames"] / c["fps"]
        taken = []
        for e in c["events"]:
            lab = "%s event, %.0f–%.0f s%s" % (short[e["label"]], e["start_s"], e["end_s"],
                                               " · in a sampled frame" if any(inside(t, e) for t in s["times_s"]) else
                                               " · between the sampled frames")
            w = width_in(fig, lab, F_MIN, "semibold")
            cx = X0 + (X1 - X0) * (e["start_s"] + e["end_s"]) / 2 / dur
            left = min(max(cx - w / 2, X0), X1 - w)
            put(fig, left, y, lab, F_MIN, WINDOW_INK, "semibold")
            taken.append((left, left + w))
        if any(s["in_window"]):
            w = width_in(fig, CAPTION, F_MIN)
            if min(a for a, _ in taken) - X0 >= w + 0.2:
                put(fig, X0, y, CAPTION, F_MIN, INK2)
            else:
                assert X1 - max(b for _, b in taken) >= w + 0.2, "no room for the caption beside the window label"
                put(fig, X1, y, CAPTION, F_MIN, INK2, ha="right")
        y += height_in("", F_MIN) + 0.01
        # 8 evenly spaced frames: tick j of the timeline sits under the middle of thumbnail j
        timeline(axes(fig, X0, y, X1 - X0, 0.15), c, s["times_s"], axis=False)
        y += 0.15 + 0.06
    rule(fig, y)
    y += 0.05
    b = case["blind"]
    rest = ["%s  %s  %s" % (c["camera"], runs(c["single"]["answers"]), verdict(c["single"]["n_right"]))
            for c in other_cams] + ["no video  %s  %s" % (runs(b["answers"]), verdict(b["n_right"]))]
    lead = ("No annotated event on %s:   " % " or ".join(c["camera"] for c in other_cams)) if other_cams else ""
    all_wrong = b["n_right"] == 0 and all(c["single"]["n_right"] == 0 for c in other_cams)
    y = para(fig, y, lead + ";   ".join(rest), F_BODY, WRONG if all_wrong else INK, "semibold")
    top = footer(fig, case, "%s. Orange = annotated event window; ticks = the %d frames received. Wording abbreviated."
                 % (PROTOCOL, SINGLE_FRAMES))
    assert y <= top - 0.02, "case 1 rows run into the footer (%.2f > %.2f in)" % (y, top)
    save_png(fig, out, "case1_q%d" % case["id"])


def figure_case2(case, out):
    short = case["short"]["events"]
    arm = case["arms"][case["case2_arm"]]
    cams = case["cameras"]
    fig = canvas()
    y = header(fig, case) + 0.05
    event_cams = [c for c in cams if c["events"]]
    top = footer(fig, case, "%s. Orange = annotated event window; ticks = every frame received. Wording abbreviated."
                 % PROTOCOL)
    # one row: each event camera's in-window frames, enlarged, side by side, a header above each group
    gap, group_gap, slots = 0.09, 0.32, 4
    hits = {c["slot"]: [t for t in arm["times_s"][c["slot"]] if any(inside(t, e) for e in c["events"])]
            for c in event_cams}
    cap = slots if sum(map(len, hits.values())) <= slots else max(1, slots // len(event_cams))
    shown = {k: v[:cap] for k, v in hits.items()}
    n_shown = sum(map(len, shown.values()))
    rule(fig, y)
    y += 0.05
    how = (arm["label"] if "selected_times_s" in arm["frame_source"]
           else "%d evenly spaced frames from each" % len(arm["times_s"][cams[0]["slot"]]))
    y = para(fig, y, "All %s cameras together, %s – enlarged: the frames inside an event window"
             % (NUMBER.get(len(cams), len(cams)), how), F_BODY, INK, "semibold") + 0.04
    strips_h = 0.26 + len(cams) * 0.19 + 0.30
    notes = {}
    for c in event_cams:
        notes[c["slot"]] = ["%s event" % " and ".join(short[e["label"]] for e in c["events"]),
                            "%d of %d frames in the window" % (len(hits[c["slot"]]), len(arm["times_s"][c["slot"]]))
                            + (" (first %d shown)" % cap if len(hits[c["slot"]]) > cap else "")]
    ims = {(c["slot"], t): c["_clip"].thumb(round(t * c["fps"]), 640) for c in event_cams for t in shown[c["slot"]]}
    aspect = max(im.width / im.height for im in ims.values())
    n_col = max(n_shown, len(event_cams))
    # the note goes on one line (taller frames) when the groups still fit the panel that way, else on two
    for sep in (" · ", "\n"):
        notes = {k: sep.join(v) for k, v in notes.items()} if sep == " · " else {k: v.replace(" · ", "\n") for k, v in notes.items()}
        note_h = height_in("", F_KEY) + height_in(sep.strip(" ·"), F_MIN) + 0.05
        fh = top - 0.10 - strips_h - 0.10 - y - note_h          # the frames take the height that is left
        fw = min(fh * aspect, (X1 - X0 - group_gap * (len(event_cams) - 1) - gap * (n_col - len(event_cams))) / n_col)
        fh = fw / aspect
        need = sum(max(max(len(shown[c["slot"]]), 1) * (fw + gap) - gap, width_in(fig, notes[c["slot"]], F_MIN))
                   for c in event_cams) + group_gap * (len(event_cams) - 1)
        if need <= X1 - X0:
            break
    x = X0
    for c in event_cams:
        n = max(len(shown[c["slot"]]), 1)
        put(fig, x, y, "Camera %s" % c["camera"], F_KEY, INK, "semibold")
        note = notes[c["slot"]]
        put(fig, x, y + height_in("", F_KEY), note, F_MIN, INK)
        for j, t in enumerate(shown[c["slot"]]):
            im = ims[c["slot"], t]
            show(axes(fig, x + j * (fw + gap), y + note_h, fw, fw * im.height / im.width), im, True, "%.1f s" % t)
        x += max(n * fw + (n - 1) * gap, width_in(fig, note, F_MIN)) + group_gap
    assert x - group_gap <= X1 + 1e-6, "case 2 frame groups run off the panel"
    y += note_h + fh + 0.10
    rule(fig, y)
    y += 0.06
    # left: where every received frame falls; right: the answer, the largest type on the panel
    x_strip, x_lab, x_ans = X0 + 0.62, 6.45, 8.3
    put(fig, X0, y, "Where all %d received frames fall in each clip" % sum(len(v) for v in arm["times_s"].values()),
        F_MIN, INK, "semibold")
    for k, c in enumerate(cams):
        by = y + 0.26 + k * 0.19
        put(fig, X0, by + 0.075, c["camera"], F_MIN, INK, "semibold", va="center")
        timeline(axes(fig, x_strip, by, x_lab - 0.1 - x_strip, 0.15), c, arm["times_s"][c["slot"]],
                 axis=k == len(cams) - 1)
        lab = ("%s event" % " and ".join(short[e["label"]] for e in c["events"])) if c["events"] else "no annotated event"
        assert x_lab + width_in(fig, lab, F_MIN, "semibold") <= x_ans - 0.1, "strip label runs into the answer"
        put(fig, x_lab, by + 0.075, lab, F_MIN, WINDOW_INK if c["events"] else INK2,
            "semibold" if c["events"] else "normal", va="center")
    assert by + 0.15 + 0.25 <= top, "case 2 strips run into the footer"
    ok = arm["n_right"]
    ans = "answered  %s" % runs(arm["answers"])
    assert width_in(fig, ans, F_NUM, "semibold") <= X1 - x_ans, "answer does not fit its column"
    put(fig, x_ans, y - 0.01, ans, F_NUM, verdict_colour(ok), "semibold")
    ya = y - 0.01 + height_in("", F_NUM)
    put(fig, x_ans, ya, verdict(ok), F_NUM2, verdict_colour(ok), "bold")
    ya += height_in("", F_NUM2) + 0.02
    tail = "annotated answer: %s" % case["gold"]
    singles = sorted({c["single"]["n_right"] for c in cams})
    if len(singles) == 1:
        tail += "\neach camera alone: %d of %d\nno video: %d of %d" % (singles[0], PASSES, case["blind"]["n_right"], PASSES)
    put(fig, x_ans, ya, tail, F_BODY, INK, "semibold")
    ya += height_in(tail, F_BODY)
    assert ya <= top, "case 2 answer column runs into the footer"
    save_png(fig, out, "case2_q%d" % case["id"])


# ----------------------------------------------------------------------------- vetting
def letter_habit(legs, pool, case):
    """How far the annotated letter is the habit for this question type: its share of the
    answer key, and of what the single-camera and the case-2 runs answer, over the whole pool."""
    ids = [i for i, r in pool.items() if r["task_type"] == case["task_type"]]
    gold = case["gold"]

    def share(tag):
        preds = [r["prediction"] for i in ids for rows in legs[tag][i].values() for r in rows]
        return round(100 * sum(p == gold for p in preds) / len(preds), 1)

    return {"n_questions_of_type": len(ids),
            "answer_key_share": round(100 * sum(pool[i]["answer"] == gold for i in ids) / len(ids), 1),
            "single_camera_answer_share": share(SINGLE),
            "case2_arm_answer_share": share(case["case2_arm"]) if case["case2_arm"] else None}


# what a sceptic checks, and what each is worth; printed in case_vetting.md
WEIGHTS = {"order": 2.0, "windows": 4.0, "letter": 2.0, "one_camera": 1.0}


def score(case):
    """Defensibility. Higher is better."""
    o = case["gold_order_wall_clock"]
    order = 2 if (o["robust"] and not o["overlap_in_wall_clock"]) else 1 if o["holds_in_wall_clock"] else 0
    n_ev = len(case["events"])
    arm = case["arms"][case["case2_arm"]] if case["case2_arm"] else None
    windows = arm["windows_reached"] / n_ev if arm else 0.0
    most_on_one = max(len(c["events"]) for c in case["cameras"]) / n_ev
    sampled = sum(1 for c in case["cameras"] if any(c["single"]["in_window"]))
    habit = case["letter_habit"]["answer_key_share"] / 100
    return {"order": order, "windows_reached_by_case2_arm": "%d of %d" % (arm["windows_reached"] if arm else 0, n_ev),
            "largest_share_of_events_on_one_camera": round(most_on_one, 2),
            "single_cameras_that_sampled_an_event_frame": sampled,
            "every_event_camera_delivered": all(e["delivered"] for e in case["events"]),
            "total": round(WEIGHTS["order"] * order + WEIGHTS["windows"] * windows - WEIGHTS["letter"] * habit
                           - WEIGHTS["one_camera"] * (most_on_one - 1 / n_ev), 2)}


def order_verdict(case):
    o = case["gold_order_wall_clock"]
    if not o["holds_in_wall_clock"]:
        return "REVERSED"
    if o["overlap_in_wall_clock"]:
        return "start order holds by %.2f s, but the windows overlap" % o["min_wall_start_gap_s"]
    return "holds, robust" if o["robust"] else "holds, within clock error"


def vetting_markdown(cases, scanned, behaviour, counts):
    ranked = sorted(cases, key=lambda c: -c["vetting"]["total"])
    windowed = [c for c in scanned if "vetting" in c]
    full = [c for c in windowed if c["vetting"]["order"] == 2
            and c["arms"][c["case2_arm"]]["windows_reached"] == len(c["events"])]
    L = ["# Two-case example: candidate vetting", "",
         "Pool `%s`, %s. Recorded rows only (`multicam_results/legs`), release annotations for the event windows; "
         "written by `demo/scripts/render_case_contrast.py --vet`." % (POOL, PROTOCOL), "",
         "**" + disclosure(counts) + "** Counts from `%s`. On the whole pool the same set is %d of %s questions: every "
         "single camera is 0/4, no video is 0/4 and "
         "some multi-camera run is 4/4; all candidates below come from that set. Questions 750, 29, 144 and 964 were "
         "the ones named for vetting; the others were added because the scan of the whole set (last table) ranks them "
         "higher." % (counts["source"], behaviour["needs_all_cameras"], format(behaviour["n"], ",")), "",
         "## What the rows can and cannot support", "",
         "- **Most of the %d do not show the model the evidence.** Annotated events are short (often 0.5 to 7 s) and frames "
         "are sparse: 8 frames from a 300 s clip are 37.5 s apart, 24 are 12.5 s apart. Of the %d questions with event "
         "windows, only %d have a robust gold order and a 4/4 multi-camera run that received a frame from every window: "
         "%s. On the rest a run is right on all 4 passes without a frame from at least one window. The figures show "
         "this as it is (orange outline = frame inside a window); a slide must not say the model \"saw both events\" "
         "unless the outlined frames say so. An actor can be visible outside the annotated window; the rows cannot tell."
         % (behaviour["needs_all_cameras"], len(windowed), len(full), ", ".join(str(c["id"]) for c in full)),
         "- **Case 1 has two different readings and the figures keep them apart**: \"no annotated event on this camera\" "
         "and \"event on this camera but not in the sampled frames\". The strongest rows are single-camera runs that did "
         "receive a frame inside their own camera's event and still failed, because the other event is on another "
         "camera (column \"single cameras that sampled an event frame\").",
         "- **Answer-letter habit.** Event-ordering options are templated (questions 29 and 144 carry the same four "
         "option strings) and the key is lopsided; the share of each candidate's annotated letter is in the table. "
         "Temporal questions have a near-even key, and where the annotated letter is B the right answer goes against "
         "what the model usually says.",
         "- **Release `reasoning` strings are generated text and can be wrong**: question 954's says the opposite of its "
         "own key and timestamps. The key agrees with the timestamps; do not quote the reasoning on a slide.", "",
         "How frames are chosen (MultiCam `bench/backends/internvl.py::get_index`): for N frames from a clip of F frames, "
         "`seg = (F-1)/N`, `index_i = int(seg/2 + round(seg*i))`. A 300 s clip at 8 frames is sampled at about "
         "18.7, 56.2, 93.7, 131.2, 168.7, 206.2, 243.7 and 281.2 s. `single_viewN` keeps content pair N of "
         "`build_messages`, which lists `video_1..video_K` in order, so single view N is delivered camera N "
         "(the rows' `frame_alloc.view_idx` agrees, and the 2,048 video tokens are 8 frames). Selection arms: "
         "`selected_times_s` is `frame_index / fps`; frame counts and fps are the ones the run recorded "
         "(`per_video_decode`), and the renderer asserts its own decode agrees.", "",
         "Event times in the release are clip-relative; clips of one question start at different wall-clock seconds "
         "(file names, 1 s resolution). Order is re-derived below in wall-clock time; a start gap under 2 s cannot be "
         "told from the file names. The release's own `gap_sec` is computed on clip-relative times and is wrong "
         "whenever the two clips start at different seconds.", "",
         "## Ranking", "",
         "Score = %.0f x order (0 reversed, 1 fragile, 2 robust) + %.0f x share of event windows with a frame in the "
         "case-2 run - %.0f x share of the answer key that is the annotated letter - %.0f x (largest share of events "
         "on one camera - 1/events). The case-2 run is the 4/4 run that reaches the most windows."
         % (WEIGHTS["order"], WEIGHTS["windows"], WEIGHTS["letter"], WEIGHTS["one_camera"]), "",
         "| rank | question | type | gold order in wall clock | smallest start gap | single cameras / no video | "
         "case-2 run | event windows with a frame in it | single cameras that sampled an event frame | "
         "annotated letter: share of key / of single-camera answers / of case-2-run answers | score |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, c in enumerate(ranked, 1):
        s, o, h = c["vetting"], c["gold_order_wall_clock"], c["letter_habit"]
        L.append("| %d | %d | %s | %s | %.2f s | %s / %d of 4 | %s | %s | %d of %d | %s: %.1f%% / %.1f%% / %.1f%% | %.2f |" % (
            k, c["id"], c["task_type"].replace("CrossView-MEVA-", ""), order_verdict(c), o["min_wall_start_gap_s"],
            " ".join("%d" % x["single"]["n_right"] for x in c["cameras"]), c["blind"]["n_right"],
            c["arms"][c["case2_arm"]]["label"], s["windows_reached_by_case2_arm"],
            s["single_cameras_that_sampled_an_event_frame"], len(c["cameras"]), c["gold"], h["answer_key_share"],
            h["single_camera_answer_share"], h["case2_arm_answer_share"], s["total"]))
    L += ["", "Rendered: questions %s." % " and ".join(str(c["id"]) for c in ranked[:2])]

    for c in ranked:
        o = c["gold_order_wall_clock"]
        L += ["", "## Question %d (%s, `%s`)" % (c["id"], c["task_type"], c["orig_id"]), "",
              "> " + c["question"], ""]
        L += ["- %s%s" % (x, "  **(gold)**" if x.startswith(c["gold"] + ".") else "") for x in c["options"]]
        L += ["", "Release reasoning: " + str(c["release_reasoning"]), "",
              "| event | camera | delivered | clip time | wall clock | what |", "|---|---|---|---|---|---|"]
        for e in c["events"]:
            L.append("| %s | %s | %s | %.2f–%.2f s | %s–%s | %s |" % (
                e["label"], e["camera"], "yes" if e["delivered"] else "no", e["start_s"], e["end_s"], e["wall_start"],
                e["wall_end"], e["description"]))
        L += ["", "Gold order, consecutive events: " + "; ".join(
            "%s→%s start gap %.2f s clip-relative, **%.2f s wall clock** (next start minus previous end: %.2f s)"
            % (g["from"], g["to"], g["clip_relative_start_gap_s"], g["wall_start_gap_s"], g["wall_gap_after_end_s"])
            for g in o["gaps"]) + ".", "",
              "| camera | clip starts | annotated event | single-camera answers (4 passes) | right | "
              "sampled frames inside a window | reading |", "|---|---|---|---|---|---|---|"]
        for x in c["cameras"]:
            s = x["single"]
            hits = ["%.1f s (%s)" % (t, ",".join(w)) for t, w in zip(s["times_s"], s["in_window"]) if w]
            L.append("| %d: %s | %s | %s | %s | %d/4 | %s | %s |" % (
                x["slot"], x["camera"], x["clip_start"][:8],
                ", ".join("%s %.1f–%.1f s" % (e["label"], e["start_s"], e["end_s"]) for e in x["events"]) or "none",
                " ".join("`%s`" % a.strip() for a in s["answers"]), s["n_right"], ", ".join(hits) or "none", s["evidence"]))
        L += ["", "No video: " + " ".join("`%s`" % a.strip() for a in c["blind"]["answers"])
              + " (%d/4)." % c["blind"]["n_right"], "",
              "| multi-camera run | answers (4 passes) | right | frames inside each event window |", "|---|---|---|---|"]
        for t, a in c["arms"].items():
            L.append("| %s | %s | %d/4 | %s |" % (a["label"], " ".join("`%s`" % x.strip() for x in a["answers"]), a["n_right"],
                                                   ", ".join("%s: %d" % kv for kv in a["frames_in_window"].items())))

    L += ["", "## The whole set of %d, same checks" % behaviour["needs_all_cameras"], "",
          "So the candidates above are not a quiet pick from a better field. Sorted by score; spatial questions carry no event "
          "windows in the release and are listed without one.", "",
          "| question | type | gold order in wall clock | case-2 run | event windows with a frame in it | score |",
          "|---|---|---|---|---|---|"]
    for c in sorted(scanned, key=lambda c: -(c["vetting"]["total"] if "vetting" in c else -99)):
        if "vetting" not in c:
            L.append("| %d | %s | no annotated event windows | | | |" % (c["id"], c["task_type"].replace("CrossView-MEVA-", "")))
            continue
        L.append("| %d | %s | %s | %s | %s | %.2f |" % (
            c["id"], c["task_type"].replace("CrossView-MEVA-", ""), order_verdict(c), c["arms"][c["case2_arm"]]["label"],
            c["vetting"]["windows_reached_by_case2_arm"], c["vetting"]["total"]))
    return "\n".join(L) + "\n", [c["id"] for c in ranked]


# ----------------------------------------------------------------------------- main
def assemble(qid, root, legs, pool, behaviour, counts, arm=None, decode=True):
    case = build_case(qid, root, legs, pool, arm=arm, decode=decode)
    case["short"] = short_text(case)
    for c in case["cameras"]:
        c["single"]["reading"] = reading(c, case) if c["events"] else "No annotated event on this camera."
        c["single"]["reading_short"] = reading_short(c, case) if c["events"] else "no annotated event on this camera"
    case["counter_evidence"] = counter_evidence(case)
    case["disclosure"] = disclosure(counts)
    case["disclosure_counts"] = {k: v for k, v in counts.items() if not k.endswith("_ids")}
    case["in_disclosed_set"] = qid in counts["fixes_ids"]
    case["behaviour_counts"] = {k: v for k, v in behaviour.items() if not k.endswith("_ids")}
    case["in_needs_all_cameras_set"] = qid in behaviour["needs_all_cameras_ids"]
    case["letter_habit"] = letter_habit(legs, pool, case)
    case["vetting"] = score(case)
    return case


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--release-root", required=True, help="crossview-release (holds annotations/ and videos/)")
    ap.add_argument("--qid", type=int, nargs="*", default=[], help="question id(s) in %s to render" % POOL)
    ap.add_argument("--vet", type=int, nargs="*", default=[], help="question ids to vet into case_vetting.md")
    ap.add_argument("--arm", choices=[t for t, _, _ in ARMS], help="multi-camera arm for case 2 (default: the arm "
                    "right on all 4 passes with frames in the most event windows, selection arms first)")
    ap.add_argument("--motivation", default=MOTIVATION, help="motivation.json the disclosure counts are read from")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root = os.path.expanduser(args.release_root)
    os.makedirs(args.out, exist_ok=True)

    with open(SUBSET) as fh:
        pool = {r["id"]: r for r in json.load(fh)}
    legs = load_legs()
    behaviour = behaviour_counts(legs, sorted(pool))
    counts = motivation_counts(os.path.expanduser(args.motivation), behaviour)
    print(disclosure(counts))

    if args.vet:
        cases = [assemble(q, root, legs, pool, behaviour, counts, decode=False) for q in args.vet]
        scanned = []
        for q in behaviour["needs_all_cameras_ids"]:
            try:
                scanned.append(assemble(q, root, legs, pool, behaviour, counts, decode=False))
            except ValueError:
                scanned.append({"id": q, "task_type": pool[q]["task_type"]})
        md, ranked = vetting_markdown(cases, scanned, behaviour, counts)
        with open(os.path.join(args.out, "case_vetting.md"), "w") as fh:
            fh.write(md)
        print("vetted, best first:", ranked)

    for q in args.qid:
        case = assemble(q, root, legs, pool, behaviour, counts, arm=args.arm)
        if case["case2_arm"] is None:
            raise SystemExit(f"question {q}: no multi-camera arm is right on all 4 passes; name one with --arm")
        assert case["in_disclosed_set"], "question %d is not one of the %d the disclosure counts" % (q, counts["fixes"])
        figure_case1(case, args.out)
        figure_case2(case, args.out)
        with open(os.path.join(args.out, "case_q%d.json" % q), "w") as fh:
            json.dump(public(case), fh, indent=1)
        print("question %d: case 2 arm = %s" % (q, case["arms"][case["case2_arm"]]["label"]))


if __name__ == "__main__":
    main()
