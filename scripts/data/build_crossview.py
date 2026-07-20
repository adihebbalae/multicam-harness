#!/usr/bin/env python3
# Ported from Wavy-Hec/CVBench analysis/convert_crossview.py @ 480d6f41cddddc7efea9a09b79134811740ba17a
"""Convert UT Austin CrossView (Multi-Camera VQA) annotations into the CVBench
record schema so the existing eval harness can benchmark them side-by-side.

Scope: sources MEVA + ego-exo4d + agibot, MCQ only, question types
temporal / event_ordering / spatial. Each question is capped to <=4 cameras
because both eval legs hard-code video_1..video_4:
  * MEVA   - the answer-bearing cameras are listed in metadata.requires_cameras
             (len 1-4, verified), mapped to their video_paths file and placed
             first, then padded -> capping never drops an answer camera.
  * ego-exo4d - no per-camera grounding exists, so the ego/aria view goes first
             and the rest follow in listed order; dropped_cameras>0 is flagged
             (the cap is lossy for these and should be read as a lower bound).
  * agibot - robot rig has only <=3 cams (head_color + hand_right/left_color),
             so the cap never drops a view -> cap_answer_safe by geometry (no
             per-camera grounding, but nothing is dropped). nuscenes stays out:
             fixed at 6 cams with no grounding -> 100% lossy under the <=4 cap.

Outputs (under data/subsets/ by default):
  crossview_qa.json            - full converted pool
  crossview_subset.json        - balanced Stage-A subset (CVBench schema)
  crossview_subset_videos.txt  - deduped <=4 video paths referenced by the subset
  crossview_subset_fetch.json  - [{"video_paths":[...]}] for hosting/fetch_videos.py

Run (no GPU, no videos needed), from the repo root:
  python3 scripts/data/build_crossview.py --n 60
"""
import argparse
import json
import os
import re
from collections import Counter, defaultdict

# normalize() handles question/options/answer parsing + data-quality flags and is
# the single source of truth for "what is the question/answer" across CrossView.
from crossview_question_types import normalize

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
ANN = os.path.join(REPO, "data", "crossview-release-annotations", "crossview-release",
                   "annotations", "multi-cam-dataset")

# v1: which (source, file) pairs to read and which question types survive.
SOURCE_FILES = {
    "meva": ["qa_temporal.json", "qa_event_ordering.json", "qa_spatial.json"],
    "ego-exo4d": ["qa_temporal.json", "qa_event_ordering.json"],
    "agibot": ["qa_temporal.json", "qa_event_ordering.json"],
}
KEEP_TYPES = {"temporal", "event_ordering", "spatial"}
LETTERS = ["A", "B", "C", "D"]

# pretty task_type labels (kept distinct from CVBench's 15 types on purpose)
SOURCE_LABEL = {"meva": "MEVA", "ego-exo4d": "EgoExo4D", "agibot": "AgiBot"}
TYPE_LABEL = {"temporal": "Temporal", "event_ordering": "Event-Ordering",
              "spatial": "Spatial"}


def task_type_of(source, qtype):
    return f"CrossView-{SOURCE_LABEL.get(source, source)}-{TYPE_LABEL.get(qtype, qtype)}"


def is_aria(path):
    return "aria" in os.path.basename(path).lower()


def cap_cameras_meva(video_paths, metadata, stats):
    """Order = required cameras (from metadata.requires_cameras) first, then the
    rest in listed order; take <=4. Returns (chosen_paths, n_required_mapped)."""
    required = (metadata or {}).get("requires_cameras") or []
    chosen, seen = [], set()
    for cid in required:
        hit = next((vp for vp in video_paths
                    if vp not in seen and f".{cid}." in os.path.basename(vp)), None)
        if hit:
            chosen.append(hit)
            seen.add(hit)
        else:
            stats["unmapped_required"] += 1
    n_required_mapped = len(chosen)
    if n_required_mapped > 4:
        stats["skipped_required_gt4"] += 1
    for vp in video_paths:
        if vp not in seen:
            chosen.append(vp)
            seen.add(vp)
    return chosen[:4], n_required_mapped


def cap_cameras_ego(video_paths):
    """Generic non-MEVA cap: ego/aria view first if present, then the rest in
    listed order; take <=4. Serves ego-exo4d (aria-first) AND agibot (no aria, so
    just listed order head_color/hand_right/hand_left; <=3 cams -> nothing dropped)."""
    aria = [vp for vp in video_paths if is_aria(vp)]
    rest = [vp for vp in video_paths if not is_aria(vp)]
    return (aria + rest)[:4]


def convert(sources, meva_ext="mp4", require_local_root=None):
    stats = Counter()
    pool = []
    for source, files in SOURCE_FILES.items():
        if source not in sources:
            continue
        for fname in files:
            path = os.path.join(ANN, source, fname)
            items = json.load(open(path))
            for idx, item in enumerate(items):
                stats["read"] += 1
                rec = normalize(item, source)
                qtype = rec["question_type"]
                if qtype not in KEEP_TYPES:
                    stats["drop_type"] += 1
                    continue
                if not rec["is_mcq"]:
                    stats["drop_open_ended"] += 1
                    continue
                ans = str(rec["answer"]).strip().upper()
                if ans not in LETTERS:
                    stats["drop_bad_answer"] += 1
                    continue
                if {"missing_answer", "unparseable_question_json"} & set(rec["flags"]):
                    stats["drop_flagged"] += 1
                    continue
                video_paths = item.get("video_paths") or []
                if not video_paths:
                    stats["drop_no_video"] += 1
                    continue
                orig_num_cameras = len(video_paths)

                if source == "meva":
                    chosen, _ = cap_cameras_meva(video_paths, item.get("metadata"), stats)
                else:
                    chosen = cap_cameras_ego(video_paths)
                if not chosen:
                    stats["drop_no_video"] += 1
                    continue

                options = [f"{k}. {rec['options'][k]}" for k in LETTERS
                           if k in rec["options"]]
                if len(options) < 2:
                    stats["drop_few_options"] += 1
                    continue

                natural = (item.get("video_id") or item.get("slot")
                           or item.get("task_name") or "na")
                orig_id = f"{natural}#{idx}"

                out = {
                    # filled with a stable integer id after global sort
                    "id": None,
                    "task_type": task_type_of(source, qtype),
                    "video_1": None, "video_2": None, "video_3": None, "video_4": None,
                    "question": rec["question"],
                    "options": options,
                    "answer": ans,
                    # provenance / analysis extras (harness ignores unknown keys)
                    "source": source,
                    "question_type": qtype,
                    "orig_num_cameras": orig_num_cameras,
                    "dropped_cameras": max(0, orig_num_cameras - len(chosen)),
                    # MEVA keeps every requires_cameras view -> cap can't drop the
                    # answer; ego-exo4d has no grounding -> any drop may be lossy;
                    # agibot has <=3 cams so the cap never drops a view (safe by
                    # geometry, via the dropped==0 clause, not by grounding).
                    "cap_answer_safe": (source == "meva"
                                        or orig_num_cameras - len(chosen) == 0),
                    "orig_id": orig_id,
                }
                for i, vp in enumerate(chosen, 1):
                    # MEVA public source ships .avi; allow keeping that extension
                    # so videos can be placed without transcoding to .mp4.
                    if source == "meva" and meva_ext != "mp4" and vp.endswith(".mp4"):
                        vp = vp[:-4] + "." + meva_ext
                    out[f"video_{i}"] = vp
                # --require-local: keep only questions whose chosen videos are all
                # present under the release root (lets a partial/incremental video
                # fetch drive a runnable subset; the eval never sees a missing file).
                if require_local_root:
                    if any(not os.path.exists(os.path.join(require_local_root, out[f"video_{i}"]))
                           for i in range(1, 5) if out[f"video_{i}"]):
                        stats["drop_missing_local"] += 1
                        continue
                pool.append(out)
                stats["kept"] += 1

    # stable, reproducible id assignment
    pool.sort(key=lambda r: (r["source"], r["question_type"], r["orig_id"]))
    for i, r in enumerate(pool):
        r["id"] = i
    return pool, stats


def num_videos(rec):
    return sum(1 for i in range(1, 5) if rec.get(f"video_{i}"))


def cam_bucket(n):
    if n <= 1:
        return "1"
    if n <= 4:
        return "2-4"
    if n <= 7:
        return "5-7"
    return "8+"


def select(records, n, per_type_cap):
    """Balance within task_type over orig_num_cameras buckets, round-robin across
    task types (mirrors analysis/select_subset.py but keyed on #cameras buckets)."""
    by_type = defaultdict(list)
    for r in records:
        by_type[r["task_type"]].append(r)

    type_order = {}
    for tt, group in by_type.items():
        buckets = defaultdict(list)
        for r in group:
            buckets[cam_bucket(r["orig_num_cameras"])].append(r)
        for b in buckets:
            buckets[b].sort(key=lambda r: r["id"])
        interleaved, idx = [], {b: 0 for b in buckets}
        while any(idx[b] < len(buckets[b]) for b in sorted(buckets)):
            for b in sorted(buckets):
                if idx[b] < len(buckets[b]):
                    interleaved.append(buckets[b][idx[b]])
                    idx[b] += 1
        type_order[tt] = interleaved

    chosen, chosen_ids = [], set()

    def take(rec):
        if rec["id"] not in chosen_ids:
            chosen.append(rec)
            chosen_ids.add(rec["id"])

    for pass_i in range(per_type_cap):
        for tt in sorted(type_order):
            if n and len(chosen) >= n:
                break
            seq = type_order[tt]
            if pass_i < len(seq):
                take(seq[pass_i])
        if n and len(chosen) >= n:
            break

    # guarantee >=1 ego-exo4d question with a lossy cap, for the dropped-cameras split
    if not any(r["source"] == "ego-exo4d" and r["dropped_cameras"] > 0 for r in chosen):
        lossy = sorted((r for r in records
                        if r["source"] == "ego-exo4d" and r["dropped_cameras"] > 0),
                       key=lambda r: r["id"])
        if lossy:
            take(lossy[0])

    if n and len(chosen) < n:
        for r in sorted(records, key=lambda r: r["id"]):
            if len(chosen) >= n:
                break
            take(r)

    chosen.sort(key=lambda r: (r["task_type"], r["orig_num_cameras"], r["id"]))
    return chosen[:n] if n else chosen


def video_paths_of(rec):
    return [rec[f"video_{i}"] for i in range(1, 5) if rec.get(f"video_{i}")]


def report(tag, recs):
    by_type = Counter(r["task_type"] for r in recs)
    by_src = Counter(r["source"] for r in recs)
    by_nv = Counter(num_videos(r) for r in recs)
    by_orig = Counter(cam_bucket(r["orig_num_cameras"]) for r in recs)
    lossy = sum(1 for r in recs if r["dropped_cameras"] > 0)
    print(f"\n{tag}: {len(recs)} questions")
    print("  by source:", dict(by_src))
    print("  by task_type:")
    for k in sorted(by_type):
        print(f"    {by_type[k]:>4}  {k}")
    print("  by num_videos (model sees):", dict(sorted(by_nv.items())))
    print("  by orig_num_cameras bucket:", dict(sorted(by_orig.items())))
    print(f"  questions with dropped_cameras>0 (lossy cap): {lossy}")


def main():
    global ANN
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann-root", default=ANN,
                    help="CrossView release annotations root (multi-cam-dataset)")
    ap.add_argument("--n", type=int, default=60, help="Stage-A subset size")
    ap.add_argument("--sources", default="meva,ego-exo4d,agibot",
                    help="comma list of sources to include (e.g. 'meva' for a "
                         "MEVA-only run). meva videos are public; ego-exo4d and "
                         "agibot are video-gated (separate licenses); nuscenes is "
                         "not wired (structurally lossy under the <=4 cap).")
    ap.add_argument("--meva-video-ext", default="mp4", choices=["mp4", "avi"],
                    help="extension for MEVA video paths; 'avi' matches the public "
                         "MEVA source so no transcoding is needed (decord reads avi)")
    ap.add_argument("--require-local", default=None, metavar="RELEASE_ROOT",
                    help="drop questions whose videos are not present under this release "
                         "root; use to build a runnable subset from a partial/incremental "
                         "video fetch (e.g. AgiBot tars pulled so far)")
    ap.add_argument("--per-type-cap", type=int, default=20)
    ap.add_argument("--out-qa", default=os.path.join(REPO, "data", "subsets", "crossview_qa.json"))
    ap.add_argument("--out-subset", default=os.path.join(REPO, "data", "subsets", "crossview_subset.json"))
    ap.add_argument("--out-videos", default=os.path.join(REPO, "data", "subsets", "crossview_subset_videos.txt"))
    ap.add_argument("--out-fetch", default=os.path.join(REPO, "data", "subsets", "crossview_subset_fetch.json"))
    args = ap.parse_args()
    ANN = args.ann_root

    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    pool, stats = convert(sources, meva_ext=args.meva_video_ext,
                          require_local_root=args.require_local)
    json.dump(pool, open(args.out_qa, "w"), indent=2, ensure_ascii=False)

    subset = select(pool, args.n, args.per_type_cap)
    json.dump(subset, open(args.out_subset, "w"), indent=2, ensure_ascii=False)

    vids = sorted({v for r in subset for v in video_paths_of(r)})
    with open(args.out_videos, "w") as f:
        f.write("\n".join(vids) + "\n")
    json.dump([{"video_paths": video_paths_of(r)} for r in subset],
              open(args.out_fetch, "w"), indent=2, ensure_ascii=False)

    print("conversion stats:", dict(stats))
    assert stats["skipped_required_gt4"] == 0, "a MEVA question needs >4 required cameras!"
    report("FULL POOL", pool)
    report("SUBSET", subset)
    print(f"\nwrote:\n  {args.out_qa}\n  {args.out_subset}\n  {args.out_videos}\n  {args.out_fetch}")
    print(f"  unique subset videos: {len(vids)}")


if __name__ == "__main__":
    main()
