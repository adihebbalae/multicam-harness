#!/usr/bin/env python3
# Ported from Wavy-Hec/CVBench analysis/make_question_bank.py @ bb372e450a25fc1b0e5aca6df8234768dc8e7abb
# Deltas from bb372e4: the annotations root, labels file and output path are
# argparse parameters; the note string names this repo's labeler path.
# theme_of, activities_of, entry and the selection rules are byte-identical.
"""Build the DOD-relevant question bank: real MEVA questions per evidence class.

The bank is a curated INPUT (tracked, like the question subsets): for each of
the five evidence-locality classes it collects real release questions — chosen
deterministically, never sampled — with their class assignment, operational
theme, and grounding metadata, plus the template roster for the classes the
release cannot fill yet. Class rules come from label_evidence_class.py (one
source of truth); a bank rebuilt after a rule change re-labels itself.

Status values: runnable (in the harness pool today), needs_conversion (exists
on disk but open-numeric/free-text — counting, summarization), pending_ruling
(camera/first-entrance: C1 positive-evidence vs C5 verification reading, plus
the camera-ID relabel trap), seed (C3-adjacent temporal pairs with ~1 s gaps).

Run from the repo root, after label_evidence_class.py:
  python3 scripts/data/make_question_bank.py
Writes data/subsets/dod_question_bank.json (tracked once committed).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import label_evidence_class as lec  # noqa: E402  (classify/load_release/event_spans)

DEFAULT_OUT = os.path.join(lec.REPO, "data", "subsets", "dod_question_bank.json")
N_PER = 3   # deterministic: first N of each source after a stable sort

THEME_BY_ACTIVITY = (
    ("perimeter / entry-exit monitoring", ("enters_scene", "exits_scene",
                                           "facility_door")),
    ("vehicle activity", ("vehicle", "trunk")),
    ("object transfer / carry", ("object", "carries", "loads", "unloads")),
    ("interaction / communications posture", ("talks", "texts", "phone",
                                              "embraces", "hand_interacts",
                                              "reads", "purchases")),
)


def theme_of(activities):
    for name, keys in THEME_BY_ACTIVITY:
        if any(k in a for a in activities for k in keys):
            return name
    return "general activity"


def activities_of(qt, item):
    v = (item.get("metadata") or {}).get("verification") or {}
    if qt == "temporal":
        return [e.get("activity") for e in (v.get("event_a"), v.get("event_b")) if e]
    if qt == "event_ordering":
        return [e.get("activity") for e in v.get("ordered_events") or []]
    if qt == "counting":
        return [v.get("activity")] if v.get("activity") else []
    if qt == "camera":
        return ["person_enters_scene_through_structure"]
    return []


def entry(qt, idx, item, status, cls=None, theme=None, note=None):
    md = item.get("metadata") or {}
    cls = cls or lec.classify(qt, item)[0]
    acts = [a for a in activities_of(qt, item) if a]
    return {k: v for k, v in {
        "evidence_class": cls,
        "status": status,
        "question_type": qt,
        "join_key": f"{qt}#{idx}",
        "dod_theme": theme or theme_of(acts),
        "question": item["question"],
        "options": item.get("options"),
        "answer": item.get("answer"),
        "slot": md.get("slot"),
        "requires_cameras": md.get("requires_cameras"),
        "activities": acts,
        "note": note,
    }.items() if v is not None}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--annotations", default=lec.ANN,
                    help="MEVA annotations dir holding the release qa_*.json")
    ap.add_argument("--labels", default=lec.DEFAULT_OUT,
                    help="labels written by label_evidence_class.py")
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    rel = lec.load_release(args.annotations)
    labels = json.load(open(args.labels))
    harness_ids = {row["join_key"]: row["ids"] for row in labels["labels"]}

    bank = []
    # C2 — runnable today (spatial) + a single-camera counting conversion seed
    for i, it in list(enumerate(rel["spatial"]))[:N_PER]:
        bank.append(entry("spatial", i, it, "runnable",
                          theme="single-sensor observation"))
    one_cam = [(i, it) for i, it in enumerate(rel["counting"])
               if len((it["metadata"].get("requires_cameras") or [])) == 1][:1]
    for i, it in one_cam:
        bank.append(entry("counting", i, it, "needs_conversion", cls="C2",
                          note="single-camera counting under the "
                               "positive-evidence reading; open-numeric today"))
    # C4 — runnable today
    for i, it in list(enumerate(rel["temporal"]))[:N_PER]:
        bank.append(entry("temporal", i, it, "runnable"))
    for i, it in list(enumerate(rel["event_ordering"]))[:N_PER]:
        bank.append(entry("event_ordering", i, it, "runnable",
                          theme="timeline reconstruction"))
    # C3 — seeds: near-simultaneous temporal pairs (smallest gap first)
    gaps = sorted(((it["metadata"]["verification"].get("gap_sec", 1e9), i, it)
                   for i, it in enumerate(rel["temporal"])),
                  key=lambda t: (t[0], t[1]))
    for g, i, it in gaps[:N_PER]:
        bank.append(entry("temporal", i, it, "seed", cls="C3-adjacent",
                          theme="concurrent cross-sensor coverage",
                          note=f"events {g:.1f}s apart with a 'simultaneously' "
                               "option — the nearest thing to a same-moment "
                               "cross-camera question in the release"))
    # C1/C3 pending ruling — first-entrance questions
    for i, it in list(enumerate(rel["camera"]))[:N_PER]:
        bank.append(entry("camera", i, it, "pending_ruling",
                          theme="perimeter / entry-exit monitoring",
                          note="positive evidence = one entrance moment in one "
                               "camera (C1); verification scans all feeds (C5); "
                               "options name physical camera IDs — the harness "
                               "relabel fix is prerequisite"))
    # C5 — needs conversion
    for i, it in list(enumerate(rel["counting"]))[:N_PER]:
        bank.append(entry("counting", i, it, "needs_conversion",
                          theme="network-wide activity census"))
    for i, it in list(enumerate(rel["summarization"]))[:1]:
        bank.append(entry("summarization", i, it, "needs_conversion",
                          theme="scene characterization / pattern of life",
                          note="free-text today; MCQ via scene_type/"
                               "top_activity distractor swap"))

    for e in bank:                       # harness ids where the question runs
        ids = harness_ids.get(e["join_key"])
        if ids:
            e["harness_ids"] = ids

    templates = {
        "C1": ["attribute snapshot at a named sensor + time",
               "presence check for a described subject on one sensor"],
        "C2": ["single-sensor activity recognition over a window",
               "vehicle maneuver sequence on one sensor",
               "single-camera activity count"],
        "C3": ["first-detection sensor for a described subject",
               "concurrent coverage: which sensors have eyes on the subject at t",
               "simultaneity check across the network"],
        "C4": ["cross-sensor precedence (which event happened first)",
               "incident timeline reconstruction across sensors",
               "re-ID handoff: where does the subject reappear, doing what"],
        "C5": ["network-wide activity census",
               "scene characterization across all feeds",
               "network-wide negative assertion (did ANY sensor record X)"],
    }
    out = {
        "note": "DOD-relevant question bank over the MEVA release: real "
                "questions per evidence-locality class, chosen "
                "deterministically. Classes and rules: "
                "scripts/data/label_evidence_class.py (one source of truth).",
        "themes_supported": [t for t, _ in THEME_BY_ACTIVITY]
        + ["activity counting", "cross-camera tracking / timeline reconstruction"],
        "themes_not_supported": [
            "weapons / threat / contraband", "abandoned objects",
            "loitering / dwell time", "crowd density",
            "identity (faces, plates, names)",
            "camera-to-camera geometry (no calibration in release)",
            "intent / anomaly detection"],
        "templates_by_class": templates,
        "counts": {"entries": len(bank)},
        "bank": bank,
    }
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
        fh.write("\n")
    from collections import Counter
    print(f"{len(bank)} entries -> {args.out}")
    print("by class/status:", dict(Counter((e['evidence_class'], e['status'])
                                           for e in bank)))


if __name__ == "__main__":
    main()
