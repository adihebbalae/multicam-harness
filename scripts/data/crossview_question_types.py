#!/usr/bin/env python3
# Ported from Wavy-Hec/CVBench analysis/crossview_question_types.py @ 480d6f41cddddc7efea9a09b79134811740ba17a
"""Question-type analysis of the UT Austin CrossView (Multi-Camera) annotations.

Reads the 20 QA JSONs under the CrossView release annotations root (no videos
needed), normalizes the two schema styles, and writes a markdown report:
counts by source x question type, #cameras distributions, answer-format split,
verbatim examples, data-quality flags, and a CVBench-vs-CrossView comparison.

Also imported by build_crossview.py for normalize() — the single source of
truth for question/options/answer parsing across CrossView.

Run from the repo root:
  python3 scripts/data/crossview_question_types.py
"""
import argparse
import glob
import json
import os
import re
import statistics
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
ANN = os.path.join(REPO, "data", "crossview-release-annotations", "crossview-release",
                   "annotations", "multi-cam-dataset")
OUT_MD = os.path.join(REPO, "data", "crossview_question_types.md")

SOURCES = ["agibot", "ego-exo4d", "meva", "nuscenes"]
LEGACY_FILES = {"meva/qa_best_camera_pre_regenerate.json"}  # superseded by qa_best_camera.json


def normalize(item, source):
    """Return a unified record; flags carry data-quality issues."""
    flags = []
    q = item.get("question")
    answer = item.get("answer")
    reasoning = item.get("reasoning")
    # a handful of nuscenes items double-encode {"question":..,"options":..,
    # "correct_option":..} as a string, broken by unescaped inner quotes —
    # recover the fields by anchoring on the key names instead of json.loads
    if isinstance(q, str) and q.strip().startswith("{"):
        mq = re.search(r'"question":\s*"(.*?)",\s*\n\s*"options"', q, re.DOTALL)
        mo = re.findall(r'"([ABCD])":\s*"(.*?)"(?=,\n|\n\s*\})', q, re.DOTALL)
        ma = re.search(r'"(?:correct_option|answer)":\s*"([ABCD])"', q)
        if mq and mo and ma:
            if not isinstance(item.get("options"), dict):
                item = {**item, "options": dict(mo)}
            answer = answer if answer not in (None, "") else ma.group(1)
            q = mq.group(1)
            flags.append("double_encoded_question")
        else:
            flags.append("unparseable_question_json")
    if isinstance(answer, dict):  # {"options":.., "correct_option":..}
        answer = answer.get("correct_option") or answer.get("answer")
        flags.append("dict_answer")
    if answer in (None, ""):
        flags.append("missing_answer")
    options = item.get("options") if isinstance(item.get("options"), dict) else None
    return {
        "source": source,
        "question_type": item.get("question_type", "?"),
        "question": q if isinstance(q, str) else str(q),
        "options": options,
        "answer": answer,
        "reasoning": reasoning,
        "num_cameras": len(item.get("video_paths") or []),
        "camera_names": (item.get("metadata") or {}).get("camera_names"),
        "is_mcq": options is not None,
        "flags": flags,
    }


def load_all():
    recs, per_file = [], {}
    for source in SOURCES:
        for path in sorted(glob.glob(os.path.join(ANN, source, "qa_*.json"))):
            rel = os.path.relpath(path, ANN)
            items = json.load(open(path))
            per_file[rel] = len(items)
            if rel in LEGACY_FILES:
                continue
            for it in items:
                recs.append(normalize(it, source))
    return recs, per_file


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---:" if i else "---" for i in range(len(header))) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def cam_stats(vals):
    return (f"{min(vals)}–{max(vals)} (median {int(statistics.median(vals))})"
            if vals else "—")


def fmt_example(r, note=""):
    cams = f"{r['num_cameras']} camera" + ("s" if r["num_cameras"] != 1 else "")
    lines = [f"**{r['source']} / {r['question_type']}** — {cams}{note}"]
    q = r["question"].strip()
    if len(q) > 600:
        q = q[:600] + " …"
    lines.append("> " + q.replace("\n", "\n> "))
    if r["options"]:
        opts = " / ".join(f"{k}. {v}" for k, v in sorted(r["options"].items()))
        if len(opts) > 500:
            opts = opts[:500] + " …"
        lines.append("> " + opts)
    ans = str(r["answer"])
    lines.append(f"> **Answer: {ans[:200] + ' …' if len(ans) > 200 else ans}**")
    return "\n".join(lines)


def main():
    global ANN, OUT_MD
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann-root", default=ANN,
                    help="CrossView release annotations root (multi-cam-dataset)")
    ap.add_argument("--out-md", default=OUT_MD, help="markdown report output path")
    args = ap.parse_args()
    ANN = args.ann_root
    OUT_MD = args.out_md

    recs, per_file = load_all()
    n = len(recs)

    by_src_type = defaultdict(Counter)
    for r in recs:
        by_src_type[r["source"]][r["question_type"]] += 1
    all_types = sorted({t for c in by_src_type.values() for t in c})

    counts_rows = []
    for s in SOURCES:
        counts_rows.append([s] + [by_src_type[s].get(t, 0) or "" for t in all_types]
                           + [sum(by_src_type[s].values())])
    counts_rows.append(["**total**"] + [sum(by_src_type[s].get(t, 0) for s in SOURCES)
                                        for t in all_types] + [n])

    cams_by_src = defaultdict(list)
    for r in recs:
        cams_by_src[r["source"]].append(r["num_cameras"])
    cam_rows = []
    for s in SOURCES:
        vals = cams_by_src[s]
        hist = Counter(vals)
        hist_str = ", ".join(f"{k}cam×{v}" for k, v in sorted(hist.items()))
        cam_rows.append([s, cam_stats(vals), hist_str])

    # the #cameras axis inside one source+type (MEVA varies most)
    meva_axis = Counter(r["num_cameras"] for r in recs if r["source"] == "meva")

    mcq = sum(1 for r in recs if r["is_mcq"])
    free_by_type = Counter(f"{r['source']}/{r['question_type']}"
                           for r in recs if not r["is_mcq"])

    flag_counts = Counter(f for r in recs for f in r["flags"])
    flagged = [r for r in recs if r["flags"]]

    # deterministic, diverse examples
    def pick(source, qtype, idx=0, min_cams=2):
        sel = [r for r in recs if r["source"] == source and r["question_type"] == qtype
               and r["num_cameras"] >= min_cams] or \
              [r for r in recs if r["source"] == source and r["question_type"] == qtype]
        return sel[idx % len(sel)]

    examples = [
        fmt_example(pick("meva", "temporal", 2), " (surveillance, cross-camera)"),
        fmt_example(pick("meva", "camera", 0), " (best-camera selection)"),
        fmt_example(pick("ego-exo4d", "camera", 0), " (ego↔exo view selection)"),
        fmt_example(pick("agibot", "event_ordering", 0), " (robot manipulation)"),
        fmt_example(pick("nuscenes", "spatial", 1), " (driving, frame-of-reference)"),
        fmt_example(pick("nuscenes", "counting", 0), " (open-ended numeric)"),
        fmt_example(pick("meva", "spatial", 0), " (cross-camera geometry)"),
        fmt_example(pick("ego-exo4d", "summarization", 0), " (open-ended free text)"),
    ]

    report = f"""# UT Austin CrossView (Multi-Camera VQA): question-type analysis

**Source:** `crossview-release-annotations/crossview-release/` (annotations only, 36 MB).
Videos are *not* included — they come from MEVA, Ego-Exo4D, AgiBotWorld and nuScenes and
must be downloaded separately, then mapped via `video_manifest.csv` (7,760 videos).
Stats below exclude the legacy file `meva/qa_best_camera_pre_regenerate.json`
({per_file.get('meva/qa_best_camera_pre_regenerate.json', 0)} items superseded by `qa_best_camera.json`).

## 1. What this benchmark tests

Unlike CVBench (1–4 *related but separate* videos, association reasoning), every CrossView
question is grounded in **synchronized cameras observing the same scene at the same time**:
surveillance networks (MEVA), ego+exo rigs (Ego-Exo4D), robot head/hand cameras (AgiBotWorld),
and 6-camera surround view (nuScenes). This is true multi-camera understanding — the model
must fuse or select among simultaneous viewpoints, which is the VideoForest setting.

## 2. Question counts — source × type ({n:,} usable QAs)

{md_table(["source"] + all_types + ["total"], counts_rows)}

Question types: **camera** = which camera best captures X (view selection); **temporal** =
which event happened first across cameras; **event_ordering** = chronological order of
multiple events; **spatial** = relative position / frame-of-reference across views;
**spatio_temporal** = combined; **counting** = open-ended count across cameras;
**summarization** = open-ended scene/episode summary across all views.

## 3. Number of cameras per question

{md_table(["source", "cams (min–max, median)", "distribution"], cam_rows)}

**This is the accuracy-vs-#cameras axis the PI wants, and it is much wider than CVBench's
1–4:** MEVA alone spans {min(meva_axis)}–{max(meva_axis)} cameras *within the same question
types*, so per-#cameras accuracy curves can be computed without changing the task.
nuScenes is fixed at 6 (surround view), AgiBotWorld at 3 (head + both hands),
Ego-Exo4D at 5–7 (1 ego + 4–6 exo).

## 4. Answer formats

- **MCQ (A–D): {mcq:,} / {n:,}** — directly scorable like CVBench.
- **Open-ended: {n - mcq:,}** — all `summarization` ({sum(v for k, v in free_by_type.items() if 'summarization' in k):,})
  and all `counting` ({sum(v for k, v in free_by_type.items() if 'counting' in k):,}).
  Counting is numeric-exact-match scorable; summarization needs an LLM judge — neither fits
  the MCQ harness unchanged.

## 5. Example questions (verbatim)

{chr(10).join(e + chr(10) for e in examples)}
## 6. Data-quality flags

- **`question_format.md` is stale** (dated 2025-02-20): it claims agibot 934 / ego-exo4d 2,641 /
  nuscenes 1,258 questions, but the released JSONs contain 750 / 1,250 / 2,249. Treat the JSONs
  as ground truth; total {sum(per_file.values()):,} raw items across 20 files, {n:,} usable after
  dropping the legacy best-camera file.
- {flag_counts.get('double_encoded_question', 0)} nuscenes items double-encode the question as a JSON string with unescaped
  inner quotes (their top-level `answer` is also empty/null; the real answer sits inside the
  string — recovered automatically): {', '.join(sorted({r['source'] + '/' + r['question_type'] for r in flagged if 'double_encoded_question' in r['flags']}))}.
- {flag_counts.get('missing_answer', 0)} item(s) with an unrecoverable missing/null answer{' — drop or repair before eval' if flag_counts.get('missing_answer') else ' (after the recovery above)'}.
- `meva/qa_best_camera_pre_regenerate.json` is a pre-regeneration legacy duplicate — exclude it.
- A few AgiBotWorld/MEVA items reference only 1 camera ({Counter(r['num_cameras'] for r in recs)[1]} items with 1 video) —
  fine as a baseline bucket for the #cameras curve.

## 7. CVBench vs CrossView — comparison & recommendation

| | CVBench (running now) | CrossView (UT Austin) |
|---|---|---|
| QAs | 1,000 | {n:,} usable |
| Cameras/question | 1–4 *separate* videos | 1–16 *synchronized* cameras |
| Nature of task | cross-video association (link entities/events across unrelated clips) | true multicam fusion (same scene, multiple simultaneous views) |
| Domains | web video mix | surveillance, ego-exo activities, robot manipulation, autonomous driving |
| Answer format | MCQ + yes/no (all scorable) | {mcq:,} MCQ + {n - mcq:,} open-ended |
| Videos | one HF zip, downloaded ✓ | not included; 7,760 files from 4 licensed datasets |

**Recommendation.** Keep CVBench as the *now* benchmark (pipeline already running; failure
traces + accuracy-vs-#videos land this week). CrossView is the better instrument for the
PI's actual question — multicam scaling — because the #cameras axis reaches 16 and the
cameras are genuinely synchronized. For inference on CrossView, start with **MEVA only**:
it is the widest #cameras axis (1–16), publicly downloadable (mevadata.org, AWS open bucket,
no license gate like Ego-Exo4D/AgiBotWorld), and its MCQ types (temporal, spatial,
event_ordering, camera) reuse the CVBench scoring path as-is. nuScenes-mini covers too few
scenes to be useful; full nuScenes/Ego-Exo4D/AgiBotWorld need registration + large downloads.

---
*Generated by `scripts/data/crossview_question_types.py`.*
"""
    with open(OUT_MD, "w") as f:
        f.write(report)
    print(f"wrote {OUT_MD}")
    print(f"{n} usable QAs; MCQ {mcq}; flags: {dict(flag_counts)}")


if __name__ == "__main__":
    main()
