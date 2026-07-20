# Ported from Wavy-Hec/CVBench bench/report.py @ 480d6f41cddddc7efea9a09b79134811740ba17a
"""Render a bench results JSONL into a markdown + CSV report:
headline (method x backend) + accuracy by task_type + by orig_num_cameras.

Usage (from repo root):
  python -m evaluation.report --jsonl results/sweep_combined.jsonl
"""
import argparse
import csv
import json
import os
from collections import defaultdict

from evaluation import scoring as metrics


def _f(x, nd=1):
    return "—" if x is None else f"{x:.{nd}f}"


def _acc_str(a):
    return "—" if a["acc"] is None else f'{a["correct"]}/{a["total"]} ({a["acc"]*100:.1f}%)'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--out-md", default=None)
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.jsonl) if l.strip()]
    by = metrics.summarize_by_method_backend_passes(rows)
    out_md = args.out_md or args.jsonl.replace(".jsonl", "_report.md")
    out_csv = args.out_csv or args.jsonl.replace(".jsonl", "_report.csv")

    md = [f"# Multi-camera benchmark report", "",
          f"Source: `{args.jsonl}` — {len(rows)} rows, {len(by)} method×backend combos.", "",
          "## Headline (method × backend)", "",
          "| method / backend | n | accuracy | acc mean±std (%) | passes | abstain | errors | lat p50 (s) | lat mean (s) | in tok | out tok | calls |",
          "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    csv_rows = []
    for k, s in by.items():
        ov, lat, tk, op = s["overall"], s["latency_s"], s["tokens"], s["overall_passes"]
        ms = "—" if op["mean"] is None else f'{op["mean"]*100:.1f} ± {op["std"]*100:.1f}'
        md.append(f'| {k} | {ov["total"]} | {_acc_str(ov)} | {ms} | {op["n_passes"]} | {_f(s["abstain_rate"]*100 if s["abstain_rate"] is not None else None,0)}% '
                  f'| {s["errors"]} | {_f(lat["p50"])} | {_f(lat["mean"])} | {_f(tk["input_mean"],0)} | {_f(tk["output_mean"],0)} | {_f(tk["calls_mean"])} |')
        csv_rows.append({"method_backend": k, "n": ov["total"],
                         "acc": (ov["acc"] or 0), "acc_mean": op["mean"], "acc_std": op["std"],
                         "passes": op["n_passes"], "abstain": s["abstain_rate"],
                         "lat_p50": lat["p50"], "lat_mean": lat["mean"],
                         "in_tok": tk["input_mean"], "out_tok": tk["output_mean"],
                         "calls": tk["calls_mean"], "errors": s["errors"]})

    # accuracy by task_type (the centralized-vs-distributed-per-category view)
    md += ["", "## Accuracy by task_type", "",
           "| method / backend | " + " | ".join(_task_types(by)) + " |",
           "|---" * (len(_task_types(by)) + 1) + "|"]
    for k, s in by.items():
        cells = [k]
        for tt in _task_types(by):
            a = s["by_task_type"].get(tt)
            cells.append(_acc_str(a) if a else "—")
        md.append("| " + " | ".join(cells) + " |")

    # accuracy by original camera count (the "more cameras = harder" curve)
    md += ["", "## Accuracy by orig_num_cameras", ""]
    for k, s in by.items():
        md.append(f"- **{k}**: " + ", ".join(
            f'{cam}cam {_acc_str(a)}' for cam, a in s["by_orig_num_cameras"].items()))

    with open(out_md, "w") as f:
        f.write("\n".join(md) + "\n")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()) if csv_rows else ["method_backend"])
        w.writeheader(); w.writerows(csv_rows)
    print(f"wrote {out_md}\nwrote {out_csv}")
    print("\n".join(md[:20]))


def _task_types(by):
    tts = set()
    for s in by.values():
        tts.update(s["by_task_type"].keys())
    return sorted(tts)


if __name__ == "__main__":
    main()
