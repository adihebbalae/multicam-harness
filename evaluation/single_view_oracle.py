#!/usr/bin/env python3
# Ported from Wavy-Hec/MultiCam analysis/single_view_oracle.py (written against
# bb372e450a25fc1b0e5aca6df8234768dc8e7abb). Deltas: row files come from the command line (plain or .gz JSONL, globs
# ok) instead of the fork's bench/results tree; class_map / loo_floor are this
# repo's evaluation.evidence_class_split. Estimators are byte-identical.
"""Upper bound on one pool and one model: the optimal-view oracle.

Reads the single_view1..K rows (one view attached, prompt otherwise identical
to the sequential arm) and asks what a system would score if it always picked
the right view. Four estimators, per evidence class and overall, because the
naive one is biased upward:

  raw best-of-K        mean over questions of max over views of the view's
                       pass-mean score — a max over K noisy estimates
  luck-best-of-K null  expected max of K draws with replacement from the
                       question's own per-view scores; raw - luck is the
                       headroom that survives the selection noise
  split-half CV        pick the best view on passes 1-2, score it on passes
                       3-4, and the mirror; unbiased under the 4-pass protocol
  worst / random view  min and mean over views — the rest of the curve

Completeness gate: a question counts when every DELIVERED view (video_i in
the record) has all its passes; 955 of 1,033 records are capped, so gating
on orig_num_cameras would drop them. Comparison rows (the sequential arm at
per-view parity, the blind floor) are split the same way; the blind leg is
reported on its complete questions only and flagged if partial.

    python -m evaluation.single_view_oracle \\
        --single-view multicam_results/legs/crossview_meva1033_subset_internvl_mp4sv8.jsonl.gz \\
        --sequential  multicam_results/legs/crossview_meva1033_subset_internvl_mp4fs32.jsonl.gz \\
        --blind       multicam_results/legs/crossview_meva1033_subset_internvl_mp4bd.jsonl.gz
"""
import argparse
import glob
import gzip
import json
import os
import sys
from collections import defaultdict
from statistics import fmean, pstdev

from evaluation.evidence_class_split import class_map, loo_floor, DEFAULT_LABELS, DEFAULT_SUBSET, DEFAULT_ALIAS

CLASSES = ("C2", "C4")


def rows_of(patterns):
    """Rows from plain or gzipped JSONL files; `patterns` is a list of globs."""
    out = []
    for pat in patterns or []:
        for f in sorted(glob.glob(pat)):
            opener = gzip.open if f.endswith(".gz") else open
            with opener(f, "rt") as fh:
                for line in fh:
                    if line.strip():
                        out.append(json.loads(line))
    return out


def luck_max(scores):
    """E[max of K draws with replacement] from the empirical scores."""
    x = sorted(scores)
    K = len(x)
    return sum(x[i - 1] * ((i / K) ** K - ((i - 1) / K) ** K) for i in range(1, K + 1))


def protocol(rows):
    """Decoding protocol actually used by a row set: temperatures seen and the
    share of rows carrying a reasoning trace. Printed so a comparison across
    legs cannot silently mix prompt templates or temperatures."""
    if not rows:
        return None
    temps = sorted({r.get("temperature") for r in rows if r.get("temperature") is not None})
    traced = sum(1 for r in rows if r.get("think")) / len(rows)
    return {"temperature": temps, "traced_share": round(traced, 3),
            "reasoning": "on" if traced > 0.5 else "off"}


def best_of(view_scores):
    """Indices of the best views (ties kept)."""
    m = max(view_scores.values())
    return [v for v, s in view_scores.items() if s == m]


def oracle(per_q, passes):
    """per_q: {qid: {view: {pass: 0/1}}} for complete questions only."""
    est = defaultdict(list)
    half_a = [p for p in passes[: len(passes) // 2]]
    half_b = [p for p in passes[len(passes) // 2:]]
    for q, views in per_q.items():
        s = {v: fmean(ps.values()) for v, ps in views.items()}
        est["raw_best"].append(max(s.values()))
        est["luck_best"].append(luck_max(list(s.values())))
        est["worst"].append(min(s.values()))
        est["random"].append(fmean(s.values()))
        # split-half: select on one half, score on the other, both directions
        cv = []
        for sel, sc in ((half_a, half_b), (half_b, half_a)):
            s_sel = {v: fmean(ps[p] for p in sel) for v, ps in views.items()}
            picks = best_of(s_sel)
            cv.append(fmean(fmean(views[v][p] for p in sc) for v in picks))
        est["split_half"].append(fmean(cv))
    return {k: 100.0 * fmean(v) for k, v in est.items()}, len(per_q)


def per_view_means(per_q, K):
    out = {}
    for v in range(1, K + 1):
        vals = [fmean(views[v].values()) for views in per_q.values() if v in views]
        out[v] = (100.0 * fmean(vals), len(vals)) if vals else (None, 0)
    return out


def arm_split(rows, method, cmap, ids_ok=None):
    """mean ± std over passes of one arm, per class and overall."""
    per = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    seen = defaultdict(set)
    for r in rows:
        if r.get("method") != method or r.get("error"):
            continue
        if ids_ok is not None and r["id"] not in ids_ok:
            continue
        cls = cmap[r["id"]]
        for c in (cls, "all"):
            per[c][r["pass_idx"]][1] += 1
            per[c][r["pass_idx"]][0] += bool(r["correct"])
        seen[cls].add(r["id"])
    out = {}
    for c, passes in per.items():
        accs = [100.0 * h / t for h, t in passes.values() if t]
        out[c] = (fmean(accs), pstdev(accs) if len(accs) > 1 else 0.0,
                  len(seen[c]) if c != "all" else sum(len(s) for s in seen.values()))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--subset", default=DEFAULT_SUBSET)
    ap.add_argument("--alias", default=DEFAULT_ALIAS)
    ap.add_argument("--labels", default=DEFAULT_LABELS)
    ap.add_argument("--single-view", nargs="+", required=True,
                    help="row files holding the single_view1..K arms (JSONL or .jsonl.gz; globs ok)")
    ap.add_argument("--sequential", nargs="*", default=[],
                    help="the sequential leg at per-view parity (8 frames/view = 32 total at K=4)")
    ap.add_argument("--blind", nargs="*", default=[], help="the blind leg's row files")
    ap.add_argument("--passes", type=int, default=4)
    ap.add_argument("--json")
    ap.add_argument("--md")
    args = ap.parse_args()

    records = json.load(open(args.subset))
    cmap = class_map(json.load(open(args.labels)), args.alias)
    delivered = {r["id"]: [i for i in range(1, 14) if r.get(f"video_{i}")] for r in records}
    K = max(len(v) for v in delivered.values())
    passes = list(range(1, args.passes + 1))

    # single-view rows -> per_q[qid][view][pass] = correct
    per_q = defaultdict(lambda: defaultdict(dict))
    sv_rows = rows_of(args.single_view)
    for r in sv_rows:
        if not r["method"].startswith("single_view") or r.get("error"):
            continue
        per_q[r["id"]][int(r["method"][len("single_view"):])][r["pass_idx"]] = int(bool(r["correct"]))
    complete = {q: v for q, v in per_q.items()
                if all(set(v.get(i, {})) == set(passes) for i in delivered[q])}
    dropped = len(per_q) - len(complete)

    by_class = {c: {q: v for q, v in complete.items() if cmap[q] == c} for c in CLASSES}
    by_class["all"] = complete
    result = {"pool": os.path.basename(args.subset), "K": K, "passes": args.passes,
              "questions_complete": len(complete), "questions_dropped_incomplete": dropped,
              "n_per_delivered_views": {},
              "protocol": {"single_view": protocol(sv_rows)},
              "oracle": {}, "per_view": {}, "floor_loo": {}, "sequential_parity": {}, "blind": {}}
    for k in range(1, K + 1):
        result["n_per_delivered_views"][k] = sum(1 for q in complete if len(delivered[q]) == k)
    for c, qs in by_class.items():
        if not qs:
            continue
        est, n = oracle(qs, passes)
        est["headroom_raw_minus_luck"] = est["raw_best"] - est["luck_best"]
        result["oracle"][c] = dict(est, n=n)
        result["per_view"][c] = per_view_means(qs, K)
        ids = None if c == "all" else set(qs)
        result["floor_loo"][c] = loo_floor(records, ids)

    seq = rows_of(args.sequential)
    if seq:
        result["protocol"]["sequential"] = protocol(seq)
        result["sequential_parity"] = {
            "leg": args.sequential,
            "cvbench_native": arm_split(seq, "cvbench_native", cmap, set(complete))}
    bl = rows_of(args.blind)
    if bl:
        result["protocol"]["blind"] = protocol(bl)
        have = defaultdict(set)
        for r in bl:
            if not r.get("error"):
                have[r["id"]].add(r["pass_idx"])
        bl_ok = {q for q, ps in have.items() if ps == set(passes)}
        result["blind"] = {"leg": args.blind, "questions_complete": len(bl_ok),
                           "questions_in_pool": len(records),
                           "partial": len(bl_ok) < len(records),
                           "blind": arm_split(bl, "blind", cmap, bl_ok)}

    md = render(result)
    print(md)
    if args.json:
        json.dump(result, open(args.json, "w"), indent=1)
    if args.md:
        open(args.md, "w").write(md)


def f(x, d=2):
    return "—" if x is None else f"{x:.{d}f}"


def render(R):
    L = []
    cols = [c for c in ("all",) + CLASSES if c in R["oracle"]]
    head = " | ".join(f"{c} (n={R['oracle'][c]['n']})" if c != "all" else f"overall (n={R['oracle'][c]['n']})" for c in cols)
    L.append(f"Pool {R['pool']}, K≤{R['K']}, {R['passes']} passes; {R['questions_complete']} complete questions, "
             f"{R['questions_dropped_incomplete']} dropped by the completeness gate; "
             f"questions by delivered views: {R['n_per_delivered_views']}.")
    L.append("Protocol per row source: " + "; ".join(
        f"{k}: T={v['temperature']}, reasoning {v['reasoning']} ({v['traced_share']:.0%} traced)"
        for k, v in R["protocol"].items() if v) + ".")
    protos = {(tuple(v["temperature"]), v["reasoning"]) for v in R["protocol"].values() if v}
    if len(protos) > 1:
        L.append("**Protocol mismatch across sources — read cross-source rows as confounded by prompt/temperature.**")
    L.append("")
    L.append(f"| Estimator | {head} |")
    L.append("|---|" + "---:|" * len(cols))
    L.append("| text-only floor (task-LOO) | " + " | ".join(f(R["floor_loo"][c]) for c in cols) + " |")
    if R["blind"]:
        b = R["blind"]["blind"]
        tag = " (PARTIAL: %d/%d q)" % (R["blind"]["questions_complete"], R["blind"]["questions_in_pool"]) if R["blind"]["partial"] else ""
        L.append(f"| blind arm{tag} | " + " | ".join(
            f"{f(b[c][0])} ± {f(b[c][1])}" if c in b else "—" for c in cols) + " |")
    if R["sequential_parity"]:
        s = R["sequential_parity"]["cvbench_native"]
        L.append("| sequential, all views, 8 frames/view (per-view parity) | " + " | ".join(
            f"{f(s[c][0])} ± {f(s[c][1])}" if c in s else "—" for c in cols) + " |")
    for v in range(1, R["K"] + 1):
        L.append(f"| single view {v} alone | " + " | ".join(
            f"{f(R['per_view'][c][v][0])} (n={R['per_view'][c][v][1]})" for c in cols) + " |")
    for key, name in (("worst", "worst view"), ("random", "random view (mean over views)"),
                      ("luck_best", "luck-best-of-K null"), ("raw_best", "raw best-of-K (inflated)"),
                      ("headroom_raw_minus_luck", "headroom: raw − luck"),
                      ("split_half", "**optimal view, split-half CV**")):
        L.append(f"| {name} | " + " | ".join(f(R["oracle"][c][key]) for c in cols) + " |")
    return "\n".join(L)


if __name__ == "__main__":
    main()
