#!/usr/bin/env python3
# Adapted from Wavy-Hec/CVBench analysis/evidence_class_accuracy.py @ 04165739e1049bd05be150b9b250cb7e4a7edbec
# Deltas from 0416573: the fork walks its own leg registry and result tree and
# cross-checks each leg's pooled accuracy against it; this tool takes the row
# files on the command line and groups them by (method, backend) itself.
# loo_floor and split are byte-identical to the fork.
"""Per-evidence-class accuracy for result rows, against per-class floors.

Joins the evidence-locality labels (scripts/data/label_evidence_class.py) to
the per-question rows written by ``inprocess/run.py`` and splits accuracy by
class: C2 (one camera, a time window — the spatial pool) vs C4 (windows across
cameras — temporal + event ordering). C1/C3/C5 are empty in the harness pool,
which is itself the finding that motivates the question-bank work.

Per class it reports the mean and std over passes (decoding variance, frames
fixed) plus the LEAVE-ONE-OUT task-conditioned modal-letter floor restricted
to that class — a class split is only readable against its own floor, because
the classes inherit different task mixes. A row whose id carries no label
stops the run: reading rows from a different pool must fail, not average.

Usage (from the repo root):
  python -m evaluation.evidence_class_split results/<leg>_shard*.jsonl
  python -m evaluation.evidence_class_split --out docs/splits.md results/*.jsonl
Options default to the meva1033 pool and its labels under data/subsets/.
"""
import argparse
import glob
import json
import os
from collections import Counter, defaultdict
from statistics import fmean, pstdev

DEFAULT_LABELS = "data/subsets/meva_evidence_labels.json"
DEFAULT_SUBSET = "data/subsets/crossview_meva1033_subset.json"
DEFAULT_ALIAS = "meva1033"
CLASSES = ("C2", "C4")


def class_map(labels, alias):
    """({record_id: evidence_class}, {record_id: task_type}) for one subset
    alias. The task_type map lets the join refuse a row whose id coincides
    with a label's id but which was not what that label was written for
    (rows from a foreign pool, e.g. All-Angles or MVU-Eval, that happen to
    share a bare integer id with a MEVA record)."""
    cmap = {}
    task_types = {}
    for row in labels["labels"]:
        rid = row["ids"].get(alias)
        if rid is not None:
            cmap[rid] = row["evidence_class"]
            task_types[rid] = row["task_type"]
    if not cmap:
        raise SystemExit(f"no ids for subset alias {alias!r} in the labels file")
    return cmap, task_types


def loo_floor(records, restrict_ids=None):
    """Leave-one-out task-conditioned modal-letter accuracy, optionally
    restricted to a set of record ids (the class): the guesser still learns
    per TASK over the whole pool — it cannot see the class — but is scored
    only on the restricted questions."""
    by_task = defaultdict(Counter)
    for r in records:
        by_task[r["task_type"]][r["answer"].strip().upper()] += 1
    hit = tot = 0
    for r in records:
        if restrict_ids is not None and r["id"] not in restrict_ids:
            continue
        c = by_task[r["task_type"]].copy()
        g = r["answer"].strip().upper()
        c[g] -= 1
        guess = max(sorted(c), key=lambda k: c[k])
        hit += (guess == g)
        tot += 1
    return 100.0 * hit / tot if tot else None


def read_rows(paths):
    rows = []
    for pattern in paths:
        files = sorted(glob.glob(pattern)) or [pattern]
        for f in files:
            if f.endswith("_summary.json"):
                continue
            with open(f) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        continue
    return rows


def split(rows, cmap, task_types):
    """{class: (mean_pct, std_pct, n_q)} over passes, plus pooled overall."""
    per = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # cls -> pass -> [hit, tot]
    pooled = [0, 0]
    for r in rows:
        if r.get("error"):
            continue
        rid = r.get("id")
        cls = cmap.get(rid)
        pi = r.get("pass_idx")
        if cls is None:
            raise SystemExit(f"row id={rid} has no label — labels stale?")
        label_task_type = task_types.get(rid)
        row_task_type = r.get("task_type")
        if row_task_type != label_task_type:
            raise SystemExit(
                f"row id={rid} has task_type {row_task_type!r}, but the "
                f"evidence-class labels for id={rid} were written for "
                f"task_type {label_task_type!r} (the MEVA pool) — this id "
                f"collides with a row from a different pool and cannot be "
                f"scored against MEVA floors"
            )
        per[cls][pi][1] += 1
        per[cls][pi][0] += bool(r.get("correct"))
        pooled[1] += 1
        pooled[0] += bool(r.get("correct"))
    out = {}
    for cls, passes in per.items():
        accs = [100.0 * h / t for h, t in passes.values() if t]
        nq = max(t for _, t in passes.values())
        out[cls] = (fmean(accs), pstdev(accs) if len(accs) > 1 else 0.0, nq)
    overall = 100.0 * pooled[0] / pooled[1] if pooled[1] else None
    return out, overall


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("rows", nargs="+", help="result row files (JSONL; globs ok)")
    ap.add_argument("--labels", default=DEFAULT_LABELS)
    ap.add_argument("--subset", default=DEFAULT_SUBSET,
                    help="the question pool the rows were run on (for the floors)")
    ap.add_argument("--alias", default=DEFAULT_ALIAS,
                    help="this pool's alias inside the labels file")
    ap.add_argument("--out", help="also write the table as markdown here")
    args = ap.parse_args()

    labels = json.load(open(args.labels))
    cmap, label_task_types = class_map(labels, args.alias)
    recs = json.load(open(args.subset))
    floors = {cls: loo_floor(recs, {i for i, c in cmap.items() if c == cls})
              for cls in CLASSES}
    floors["overall"] = loo_floor(recs)

    groups = defaultdict(list)
    for r in read_rows(args.rows):
        groups[(r.get("method"), r.get("backend"))].append(r)
    if not groups:
        raise SystemExit("no rows read")

    fmt = lambda c: f"{c[0]:.2f} ± {c[1]:.2f} (n={c[2]})" if c else "—"
    lines = [f"## {args.alias} pool",
             "floor (LOO): overall {:.2f} · C2 {:.2f} · C4 {:.2f}".format(
                 floors["overall"], floors["C2"], floors["C4"]), "",
             "| method | backend | C2 acc | C4 acc | overall | rows |",
             "|---|---|---|---|---|---|"]
    for (method, backend), rows in sorted(groups.items(), key=lambda kv: str(kv[0])):
        cls, overall = split(rows, cmap, label_task_types)
        lines.append(f"| {method} | {backend} | {fmt(cls.get('C2'))} "
                     f"| {fmt(cls.get('C4'))} | {overall:.2f} | {len(rows)} |")
    text = "\n".join(lines) + "\n"
    print(text, end="")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            fh.write(text)
        print(f"table -> {args.out}")


if __name__ == "__main__":
    main()
