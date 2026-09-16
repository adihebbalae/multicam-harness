#!/usr/bin/env python3
"""Tail a bench result JSONL and print one line per answer as it lands, for the
recorded demo's terminal segment (the runner itself only prints per-arm summaries).

  python3 analysis/demo_live_tail.py bench/results/bench_demo_meva4cam_subset_internvl_demo_live.jsonl
  python3 analysis/demo_live_tail.py <file> --replay 1.5     # re-print an existing file, 1.5 s per row

Prints id, method, prediction vs gold, latency, tokens and, for segment_select rows,
the segments kept per camera. Exits when it sees the ``--expect`` row count (default 0 = never).
"""
import argparse, json, os, sys, time

def fmt(r):
    fa = r.get("frame_alloc") or {}
    kept = fa.get("segments_kept_per_video")
    sel = ""
    if kept:
        sel = "  kept segments/camera: " + " ".join(f"cam{k}:{len(v)}" for k, v in sorted(kept.items()))
    ok = "CORRECT" if r.get("correct") else ("ERROR  " if r.get("error") else "WRONG  ")
    return (f"q{r.get('id'):<5} {r.get('method'):<22} pred {str(r.get('prediction') or '-'):<2} "
            f"gold {r.get('gold')}  {ok}  {r.get('latency_s', 0):6.1f} s  "
            f"in {r.get('input_tokens', 0):>6} tok  video {r.get('video_tokens', 0):>6} tok{sel}")

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("path")
    ap.add_argument("--replay", type=float, default=0.0, help="seconds between rows when re-printing an existing file")
    ap.add_argument("--expect", type=int, default=0, help="stop after this many rows (0 = run until Ctrl-C)")
    a = ap.parse_args()
    seen = 0
    while not os.path.exists(a.path):
        time.sleep(0.5)
    with open(a.path) as f:
        while True:
            line = f.readline()
            if not line:
                if a.expect and seen >= a.expect:
                    return
                time.sleep(0.5); continue
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue  # torn line: the writer has not finished it yet
            seen += 1
            print(fmt(r), flush=True)
            if a.replay:
                time.sleep(a.replay)
            if a.expect and seen >= a.expect:
                return

if __name__ == "__main__":
    main()
