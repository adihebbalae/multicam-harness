# Ported from Wavy-Hec/CVBench bench/capture_trace.py @ 480d6f41cddddc7efea9a09b79134811740ba17a
"""Re-run a single question through InternVL3-8B for BOTH methods and capture the
full model output (the reasoning trace), which the sweep harness doesn't persist.

Usage (from repo root, internvl env, GPU):
  python -m scripts.capture_trace --id cvb-28 \
      --subset data/subsets/cvbench_temporal_subset.json
"""
import argparse
import json
import os

import yaml

from dataloaders.qa_json import build_messages, video_paths
from evaluation.scoring import parse_choice, gt_choice
from harnesses.stitched import build_montages, MONTAGE_PREFIX
from models.clients import InternVL3Backend

_CFG = yaml.safe_load(open("configs/datasets.yaml"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", default="cvb-28")
    ap.add_argument("--subset", default=_CFG["subsets"]["cvbench_temporal"])
    ap.add_argument("--video-root", default=_CFG["video_roots"]["cvbench"])
    ap.add_argument("--nframes", type=int, default=16)
    ap.add_argument("--max-tiles", type=int, default=4)
    ap.add_argument("--out", default="results/trace_cvb28.json")
    args = ap.parse_args()

    sub = {r["id"]: r for r in json.load(open(args.subset))}
    rec = sub[args.id]
    backend = InternVL3Backend("OpenGVLab/InternVL3-8B", num_frame=args.nframes, max_tiles=args.max_tiles)

    out = {"id": args.id, "question": rec["question"], "options": rec["options"], "gold": rec["answer"]}

    # --- native ---
    msgs, yn = build_messages(rec, args.video_root, args.nframes, no_video=False)
    g = backend.generate(msgs, max_new_tokens=8192, seed=1, temperature=0)
    out["native"] = {"prediction": parse_choice(g.text, yn), "gold": gt_choice(rec["answer"], yn),
                     "text": g.text, "output_tokens": g.output_tokens}

    # --- stitch (centralized) ---
    base_msgs, yn = build_messages(rec, args.video_root, args.nframes, no_video=True)
    scaffold = base_msgs[0]["content"][0]["text"]
    paths = video_paths(rec, args.video_root)
    montages = build_montages(paths, nframes=args.nframes, T=args.nframes, cell_px=448)
    k = len(paths)
    content = [{"type": "text", "text": MONTAGE_PREFIX.format(T=len(montages), k=k)}]
    content += [{"type": "image", "image": m} for m in montages]
    content += [{"type": "text", "text": scaffold}]
    g2 = backend.generate([{"role": "user", "content": content}], max_new_tokens=8192, seed=1, temperature=0)
    out["stitch"] = {"prediction": parse_choice(g2.text, yn), "gold": gt_choice(rec["answer"], yn),
                     "text": g2.text, "output_tokens": g2.output_tokens}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=2)
    print(f"\n{'='*70}\nQ: {rec['question']}\noptions: {rec['options']}\ngold: {rec['answer']}")
    for m in ("native", "stitch"):
        print(f"\n{'='*70}\n### {m.upper()}  pred={out[m]['prediction']}  gold={out[m]['gold']}  "
              f"{'CORRECT' if out[m]['prediction']==out[m]['gold'] else 'WRONG'}\n{'-'*70}")
        print(out[m]["text"])
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
