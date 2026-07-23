# Ported from Wavy-Hec/CVBench bench/chance_level.py @ 883eb108a03e6ff153172887cdcb0be0da9eca3a
"""Random-guessing ("chance") accuracy for a QA subset, from its option lists.

CVBench is not a uniform 4-way multiple-choice set: 861 of the 1000 questions
have 4 options and 139 are yes/no.  So "what would a model score if it just
guessed?" is not one number -- it is the per-question expectation

    chance = mean over questions of 1 / (number of answer options)

i.e. the accuracy of a guesser that picks uniformly among the options it is
actually shown.  Pooled over a set this is the mean of the per-question chance
levels, so it differs by task category: categories that are all 4-option sit at
25.0%, while Cross-video Scene Recognition (122 of 149 questions are yes/no)
sits at 45.5%, and the 1000-question pool lands at 28.5%.

Source-data caveat: one row is malformed -- its four choices are concatenated
into a single string ("A. ... , B. ... , C. ... , D. ..."), so a naive
len(options) reads it as 1 option (chance 100%).  We detect embedded letter
markers and count it as 4.

Usage (from repo root):
  python -m evaluation.chance                       # per-category table
  python -m evaluation.chance --qa-json data/subsets/cvbench_temporal_subset.json
"""
import argparse
import json
import re
from collections import Counter, defaultdict

import yaml

DEFAULT_QA_JSON = "data/subsets/cvbench_full_runnable_subset.json"

# "A. " at the start of a string, or ", B. " / "; C. " inside one -- the marker
# pattern used to recover the option count from a concatenated options string.
_MARKER = re.compile(r"(?:^|[,;]\s*)([A-H])\.\s")


def default_qa_json(path="configs/datasets.yaml"):
    """The cvbench_full subset path from configs/datasets.yaml, else the
    relative fallback. Used only to fill an argparse default; the functions
    below take the path as an ordinary parameter."""
    try:
        with open(path) as fh:
            cfg = yaml.safe_load(fh) or {}
    except FileNotFoundError:
        return DEFAULT_QA_JSON
    return (cfg.get("subsets") or {}).get("cvbench_full", DEFAULT_QA_JSON)


def n_options(row):
    """Effective number of answer options for one QA row."""
    opts = row.get("options") or []
    if len(opts) == 1:
        markers = set(_MARKER.findall(opts[0]))
        if len(markers) >= 2:          # concatenated choices (see module docstring)
            return len(markers)
    return len(opts)


def load_rows(path):
    with open(path) as fh:
        return json.load(fh)


def chance_table(path):
    """-> (per_task, overall, detail).

    per_task : task_type -> chance in accuracy %
    overall  : chance in accuracy % over every question in the subset
    detail   : task_type -> {"n": int, "hist": {n_options: count}}
    """
    per_task_counts = defaultdict(list)
    for row in load_rows(path):
        per_task_counts[row["task_type"]].append(n_options(row))

    per_task, detail = {}, {}
    all_counts = []
    for task, counts in per_task_counts.items():
        per_task[task] = 100.0 * sum(1.0 / k for k in counts) / len(counts)
        detail[task] = {"n": len(counts), "hist": dict(sorted(Counter(counts).items()))}
        all_counts.extend(counts)
    overall = 100.0 * sum(1.0 / k for k in all_counts) / len(all_counts)
    return per_task, overall, detail


def pooled_detail(detail):
    """Sum the per-task option histograms into one pooled entry."""
    hist = defaultdict(int)
    for d in detail.values():
        for k, v in d["hist"].items():
            hist[k] += v
    return {"n": sum(d["n"] for d in detail.values()), "hist": dict(sorted(hist.items()))}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--qa-json", default=default_qa_json(),
                    help="QA subset json whose option lists define the chance level")
    args = ap.parse_args()

    per_task, overall, detail = chance_table(args.qa_json)
    print(f"\nRandom-guessing baseline   (source: {args.qa_json})")
    print("chance = mean over questions of 1 / (number of answer options)\n")
    print(f"{'task type':40s} {'n':>5s} {'option counts':>20s} {'chance %':>9s}")
    print("-" * 78)
    for task in sorted(per_task, key=lambda t: -per_task[t]):
        d = detail[task]
        print(f"{task:40s} {d['n']:5d} {str(d['hist']):>20s} {per_task[task]:9.2f}")
    print("-" * 78)
    print(f"{'OVERALL (pooled)':40s} {pooled_detail(detail)['n']:5d} "
          f"{'':>20s} {overall:9.2f}\n")


if __name__ == "__main__":
    main()
