# Ported from Wavy-Hec/CVBench bench/validate_scoring.py @ 480d6f41cddddc7efea9a09b79134811740ba17a
"""CPU-only gate: re-score an existing eval JSON through the harness parse+metrics
path and confirm it reproduces the stored accuracy. No GPU / no model load.

Usage (from repo root):
  python -m tests.validate_scoring [path/to/eval_*.json]
Default: tests/fixtures/scoring_fixture.json — the first 12 questions of the
canonical 60-Q CrossView Qwen run (outputs tail-truncated, parse-verified) with
a hard-asserted expected score of 5/12 (41.7%). Pointing it at the full dump in
the Wavy-Hec/CVBench fork also works and is the original gate:
  Video-R1/src/r1-v/eval_results/eval_crossview_subset_qwen3vl.json
  (expected 19/60 = 31.7%).
"""
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from dataloaders.qa_json import is_yesno  # noqa: E402
from evaluation import scoring as metrics  # noqa: E402
from evaluation.scoring import parse_choice, gt_choice  # noqa: E402

DEFAULT = os.path.join(REPO, "tests", "fixtures", "scoring_fixture.json")

# Recomputed by running the ported scoring functions over the fixture's 12
# records (see docs/PORTING.md); hard-asserted when the default fixture is scored.
EXPECTED_FIXTURE_SCORE = (5, 12)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    blob = json.load(open(path))
    res = blob.get("results", blob)
    rows, mismatch = [], 0
    for r in res:
        yn = is_yesno(r["options"])
        pred = parse_choice(r.get("output", ""), yn)
        gold = gt_choice(r["answer"], yn)
        if "prediction" in r and pred != r["prediction"]:
            mismatch += 1
        rows.append(dict(
            id=r["id"], task_type=r["task_type"], source=r.get("source"),
            orig_num_cameras=r.get("orig_num_cameras"), cap_answer_safe=r.get("cap_answer_safe"),
            num_videos=r.get("num_videos", 0), method="centralized", backend="(stored-output)",
            prediction=pred, gold=gold,
            correct=(pred.strip().upper() == gold.strip().upper()), abstained=(pred == "")))
    s = metrics.summarize(rows)
    print(f"file: {path}")
    print(f"re-scored overall: {s['overall']['correct']}/{s['overall']['total']} "
          f"= {s['overall']['acc']*100:.1f}%")
    print(f"prediction mismatches vs stored 'prediction': {mismatch}")
    stored = blob.get("summary", {}).get("overall_acc")
    if stored is not None:
        print(f"stored summary.overall_acc: {stored:.2f}%")
    print("by task_type:", {k: f"{v['correct']}/{v['total']}" for k, v in s["by_task_type"].items()})
    if os.path.abspath(path) == os.path.abspath(DEFAULT):
        got = (s["overall"]["correct"], s["overall"]["total"])
        assert got == EXPECTED_FIXTURE_SCORE, (
            f"fixture score drifted: got {got[0]}/{got[1]}, "
            f"expected {EXPECTED_FIXTURE_SCORE[0]}/{EXPECTED_FIXTURE_SCORE[1]}")
        print(f"fixture gate OK: {got[0]}/{got[1]} matches EXPECTED_FIXTURE_SCORE")


if __name__ == "__main__":
    main()
