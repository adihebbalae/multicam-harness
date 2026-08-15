#!/usr/bin/env python3
# Ported from Wavy-Hec/CVBench analysis/convert_mvueval.py @ 8edaecf2811da5ad4e8c2fb8883a3cb01fbf059b
# Deliberate delta vs source: the fork hardcodes MAX_VIDEOS = 13 and its own
# LETTERS string. Here both come from the canonical definition in
# inprocess/dataloaders/qa_json.py (MAX_SLOTS / LETTERS_ALL) -- the loader that
# actually consumes these records via inprocess/run.py -- so emitted records
# cannot desync from the harness that reads them. Outputs default under
# data/subsets/ rather than the fork's analysis/ directory.
"""Convert MVU-Eval (multi-video QA, NeurIPS 2025 D&B) into the harness record
schema so inprocess/run.py can evaluate it with the centralized (stitch),
decentralized (per_stream), and native sequential arms.

Source: data/mvueval/MVU_Eval_QAs.json from the MVU-Eval-Team/MVU-Eval-Data HF
dataset (1,824 questions over ~4.9k mp4 clips; 2-13 videos per question,
letter MCQ with 2-11 options). Video filenames are kept as-is (some live under
subdirectories), so runs pass --video-root pointing at the clip directory.

The release ships the ground truth twice and the copies disagree: MVU_Eval_QAs.json
and mvu_eval_config.csv differ on 92 of the 1,824 questions (5.04%). This script
reads the JSON. Say which key you scored against when comparing to published
numbers.

Option lists are normalized to the harness "A. ..." convention (a handful of
source rows carry non-canonical prefixes); the ground-truth letter is asserted
to fall inside each record's option range.

Tasks (8): Counting, KIR (knowledge-intensive reasoning), TR (temporal
reasoning), OR (object recognition), SU (spatial understanding), ICL
(in-context learning), Comparison, RAG. The synchronized multi-camera content
is concentrated in SU (ScanNet / nuScenes multi-view); ``--tasks SU`` builds
that slice.

Outputs (under data/subsets/; the fork's `dev` naming is kept so the emitted
pool stays name-compatible with the committed mvueval_qa.json):
  mvueval_qa.json           - full converted pool
  mvueval_dev_subset.json   - task x clip-count stratified dev subset (~100)
  mvueval_dev_videos.txt    - deduped clip paths the dev subset needs
  mvueval_dev_fetch.json    - [{"video_paths": [...]}] per dev question

Run (no GPU, no videos needed), from the repo root:
  python3 scripts/data/build_mvueval.py --qa-json data/mvueval/MVU_Eval_QAs.json
"""
import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))

# This script is documented as a direct path run (`python3 scripts/data/...`), so
# sys.path[0] is scripts/data and the repo's top-level packages are NOT importable.
# Put the repo root on the path before importing them.
if REPO not in sys.path:
    sys.path.insert(0, REPO)

# How many video_i slots the harness reads, and the widest option-letter range it
# parses. Taken from the harness's own definitions so emitted records cannot
# desync from the loader that consumes them.
from inprocess.dataloaders.qa_json import LETTERS_ALL, MAX_SLOTS  # noqa: E402

DATA_ROOT = os.path.join(REPO, "data", "mvueval")

LETTERS = LETTERS_ALL
MAX_VIDEOS = MAX_SLOTS
_PREFIX_RE = re.compile(r"^\s*([A-M])\s*[.)>:：]\s*")


def normalize_options(raw_options, stats):
    """Force the harness convention: options[i] == '<letter i>. <text>'."""
    out = []
    for i, opt in enumerate(raw_options):
        want = f"{LETTERS[i]}. "
        opt = str(opt).strip()
        if opt.startswith(want):
            out.append(opt)
            continue
        stats["options_reprefixed"] += 1
        m = _PREFIX_RE.match(opt)
        body = opt[m.end():] if m else opt
        out.append(want + body)
    return out


def convert(qa_json, tasks=None):
    stats = Counter()
    data = json.load(open(qa_json, encoding="utf-8"))
    items = sorted(data.items(), key=lambda kv: int(kv[0]))
    pool = []
    for orig_id, item in items:
        stats["read"] += 1
        task = item["task"]
        if tasks and task not in tasks:
            stats["drop_task_filter"] += 1
            continue
        vids = item["video_paths"] or []
        if not 1 <= len(vids) <= MAX_VIDEOS:
            stats["drop_bad_videos"] += 1
            continue
        raw_opts = item["options"] or []
        if not 2 <= len(raw_opts) <= len(LETTERS):
            stats["drop_bad_options"] += 1
            continue
        ans = str(item["ground_truth"]).strip().upper()
        if ans not in LETTERS[:len(raw_opts)]:
            stats["drop_gt_out_of_range"] += 1
            continue
        out = {
            "id": None,  # stable int assigned after sort (source order is stable)
            "task_type": f"MVU-{task}",
            "question": item["question"],
            "options": normalize_options(raw_opts, stats),
            "answer": ans,
            # provenance / analysis extras (harness ignores unknown keys)
            "source": "mvueval",
            "question_type": task,
            "orig_num_cameras": len(vids),
            "cap_answer_safe": True,  # all clips are kept (no <=4 cap anymore)
            "orig_id": orig_id,
        }
        for i, vp in enumerate(vids, 1):
            out[f"video_{i}"] = vp
        pool.append(out)
        stats["kept"] += 1

    for i, r in enumerate(pool):
        r["id"] = i
    return pool, stats


def clip_bucket(n):
    if n <= 2:
        return "2"
    if n <= 4:
        return "3-4"
    if n <= 6:
        return "5-6"
    return "7+"


def select(records, n):
    """Dev subset: round-robin across task types, interleaving clip-count
    buckets within each task (mirrors build_crossview.select)."""
    by_task = defaultdict(list)
    for r in records:
        by_task[r["task_type"]].append(r)
    order = {}
    for tt, group in by_task.items():
        buckets = defaultdict(list)
        for r in group:
            buckets[clip_bucket(r["orig_num_cameras"])].append(r)
        for b in buckets:
            buckets[b].sort(key=lambda r: r["id"])
        interleaved, idx = [], {b: 0 for b in buckets}
        while any(idx[b] < len(buckets[b]) for b in sorted(buckets)):
            for b in sorted(buckets):
                if idx[b] < len(buckets[b]):
                    interleaved.append(buckets[b][idx[b]])
                    idx[b] += 1
        order[tt] = interleaved

    chosen, pass_i = [], 0
    while len(chosen) < n and any(pass_i < len(seq) for seq in order.values()):
        for tt in sorted(order):
            if len(chosen) >= n:
                break
            if pass_i < len(order[tt]):
                chosen.append(order[tt][pass_i])
        pass_i += 1
    chosen.sort(key=lambda r: r["id"])
    return chosen


def video_paths_of(rec):
    return [rec[f"video_{i}"] for i in range(1, MAX_VIDEOS + 1) if rec.get(f"video_{i}")]


def report(tag, recs):
    by_type = Counter(r["task_type"] for r in recs)
    by_nv = Counter(r["orig_num_cameras"] for r in recs)
    by_nopt = Counter(len(r["options"]) for r in recs)
    print(f"\n{tag}: {len(recs)} questions")
    print("  by task_type:")
    for k in sorted(by_type):
        print(f"    {by_type[k]:>4}  {k}")
    print("  by clip count:", dict(sorted(by_nv.items())))
    print("  by option count:", dict(sorted(by_nopt.items())))
    # per-question chance = 1/len(options); report per task + pooled
    per_task = defaultdict(list)
    for r in recs:
        per_task[r["task_type"]].append(1.0 / len(r["options"]))
    print("  chance floor per task (%):",
          {k: round(100 * sum(v) / len(v), 1) for k, v in sorted(per_task.items())})
    allv = [1.0 / len(r["options"]) for r in recs]
    print(f"  chance floor pooled: {100 * sum(allv) / len(allv):.2f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa-json", default=os.path.join(DATA_ROOT, "MVU_Eval_QAs.json"))
    ap.add_argument("--n", type=int, default=100, help="dev subset size")
    ap.add_argument("--tasks", default=None,
                    help="comma list of task codes to keep (e.g. 'SU' for the "
                         "synchronized multi-camera spatial slice); default all 8")
    ap.add_argument("--out-qa", default=os.path.join(REPO, "data", "subsets", "mvueval_qa.json"))
    ap.add_argument("--out-subset", default=os.path.join(REPO, "data", "subsets", "mvueval_dev_subset.json"))
    ap.add_argument("--out-videos", default=os.path.join(REPO, "data", "subsets", "mvueval_dev_videos.txt"))
    ap.add_argument("--out-fetch", default=os.path.join(REPO, "data", "subsets", "mvueval_dev_fetch.json"))
    args = ap.parse_args()

    tasks = {t.strip() for t in args.tasks.split(",")} if args.tasks else None
    pool, stats = convert(args.qa_json, tasks=tasks)
    json.dump(pool, open(args.out_qa, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    subset = select(pool, args.n)
    json.dump(subset, open(args.out_subset, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    vids = sorted({v for r in subset for v in video_paths_of(r)})
    with open(args.out_videos, "w", encoding="utf-8") as f:
        f.write("\n".join(vids) + "\n")
    json.dump([{"video_paths": video_paths_of(r)} for r in subset],
              open(args.out_fetch, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    print("conversion stats:", dict(stats))
    report("FULL POOL", pool)
    report("DEV SUBSET", subset)
    print(f"\nwrote:\n  {args.out_qa}\n  {args.out_subset}\n  {args.out_videos}\n  {args.out_fetch}")
    print(f"  unique dev videos to fetch: {len(vids)}")


if __name__ == "__main__":
    main()
