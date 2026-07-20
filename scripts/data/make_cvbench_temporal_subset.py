# Ported from Wavy-Hec/CVBench analysis/make_cvbench_temporal_subset.py @ 480d6f41cddddc7efea9a09b79134811740ba17a
"""Build the CVBench temporal-logic subset for the harness.

Takes the 1000-question CVBench eval set (data/CVBench.json), joins the
rule-based temporal levels from data/cvbench_temporal_logic_team.json,
keeps the level>=1 ("temporal logic") questions, normalizes them into the same
record shape the harness expects (the MEVA subsets), validates every referenced
video exists, and writes data/subsets/cvbench_temporal_subset.json.

video_root for these records is the CVBench video directory (pass it to
run_vqa.py via --video-root).

Run from repo root:  python scripts/data/make_cvbench_temporal_subset.py
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
CVBENCH = os.path.join(REPO, "data", "CVBench.json")
TEAM = os.path.join(REPO, "data", "cvbench_temporal_logic_team.json")
VIDEO_ROOT = os.path.join(REPO, "data", "CVBench")
OUT = os.path.join(REPO, "data", "subsets", "cvbench_temporal_subset.json")


def clean_video(v):
    """Harness video_paths() uses `if v:` — the literal string 'None' is truthy and
    would try to open a file named None. Map None/''/'None' -> falsy (None)."""
    if v is None:
        return None
    s = str(v).strip()
    return None if s in ("", "None", "null") else s


def temporal_level(team_rec):
    t = team_rec.get("temporal") if team_rec else None
    return (t or {}).get("level", 0) if isinstance(t, dict) else 0


def main():
    global CVBENCH, TEAM, VIDEO_ROOT, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--cvbench-json", default=CVBENCH,
                    help="full 1000-question CVBench eval set")
    ap.add_argument("--team-json", default=TEAM,
                    help="rule-based temporal-logic levels json")
    ap.add_argument("--video-root", default=VIDEO_ROOT,
                    help="CVBench video directory (existence validation)")
    ap.add_argument("--out", default=OUT, help="output subset json")
    args = ap.parse_args()
    CVBENCH, TEAM, VIDEO_ROOT, OUT = (args.cvbench_json, args.team_json,
                                      args.video_root, args.out)

    cv = json.load(open(CVBENCH))
    team = {str(q["id"]): q for q in json.load(open(TEAM))["questions"]}

    out, missing = [], []
    for r in cv:
        tid = str(r["id"])
        lvl = temporal_level(team.get(tid))
        if lvl < 1:
            continue
        vids = [clean_video(r.get(f"video_{i}")) for i in range(1, 5)]
        vids = [v for v in vids if v]
        # validate existence
        for v in vids:
            if not os.path.exists(os.path.join(VIDEO_ROOT, v)):
                missing.append((tid, v))
        rec = {
            "id": f"cvb-{tid}",
            "task_type": r["task_type"],
            "question": r["question"],
            "options": r["options"],            # already ["A. ...", ...] / ["Yes.","No."]
            "answer": r["answer"],              # letter (MC) or Yes/No — gt_choice handles both
            "source": "cvbench",
            "question_type": "temporal_logic",
            "temporal_level": lvl,              # 1 = reference, 2 = complex
            "orig_num_cameras": len(vids),      # CVBench: # of videos in the question (2-4)
            "cap_answer_safe": True,
            "orig_id": tid,
        }
        for i, v in enumerate(vids, 1):
            rec[f"video_{i}"] = v
        out.append(rec)

    json.dump(out, open(OUT, "w"), ensure_ascii=False)
    print(f"wrote {len(out)} temporal-logic questions -> {OUT}")
    import collections
    print("  video-count:", dict(sorted(collections.Counter(x["orig_num_cameras"] for x in out).items())))
    print("  temporal_level:", dict(sorted(collections.Counter(x["temporal_level"] for x in out).items())))
    yn = sum(1 for x in out if all(str(o).strip().strip(".").lower() in ("yes", "no") for o in x["options"]))
    print(f"  yes/no: {yn} | MC: {len(out)-yn}")
    if missing:
        print(f"  !! MISSING VIDEOS: {len(missing)} (e.g. {missing[:3]})")
    else:
        print(f"  all referenced videos exist under {VIDEO_ROOT}")


if __name__ == "__main__":
    main()
