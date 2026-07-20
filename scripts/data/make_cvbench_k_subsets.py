# Ported from Wavy-Hec/CVBench analysis/make_cvbench_k_subsets.py @ 480d6f41cddddc7efea9a09b79134811740ba17a
"""K-filtered views of the full-1000 CVBench runnable subset, for the D3
top-m selection arms. When top_m >= K the selection keeps ALL clips and the
prompt is byte-identical to temporal_weighted (whose full-1000 rows already
exist), so the top2 arm only needs K>=3 questions and the top3 arm only K=4 —
the remaining bins are composed from the temporal_weighted baseline at
analysis time.

Records are copied unchanged (same ids), so results pool/resume cleanly.

Usage:  python scripts/data/make_cvbench_k_subsets.py
Writes: data/subsets/cvbench_full_k34_subset.json  (K >= 3)
        data/subsets/cvbench_full_k4_subset.json   (K == 4)
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
SRC = os.path.join(REPO, "data", "subsets", "cvbench_full_runnable_subset.json")
OUT_DIR = os.path.join(REPO, "data", "subsets")


def num_videos(rec):
    return sum(1 for i in range(1, 5) if rec.get(f"video_{i}"))


def main():
    global SRC, OUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC,
                    help="full CVBench runnable subset json to filter")
    ap.add_argument("--out-dir", default=OUT_DIR,
                    help="directory the K-filtered subsets are written to")
    args = ap.parse_args()
    SRC, OUT_DIR = args.src, args.out_dir

    recs = json.load(open(SRC))
    for name, keep in (("cvbench_full_k34_subset.json", lambda k: k >= 3),
                       ("cvbench_full_k4_subset.json", lambda k: k == 4)):
        sub = [r for r in recs if keep(num_videos(r))]
        out = os.path.join(OUT_DIR, name)
        json.dump(sub, open(out, "w"), indent=1)
        ks = [num_videos(r) for r in sub]
        print(f"{out}: {len(sub)} records, K dist "
              f"{ {k: ks.count(k) for k in sorted(set(ks))} }")


if __name__ == "__main__":
    main()
