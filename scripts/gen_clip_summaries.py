# Ported from Wavy-Hec/CVBench bench/gen_clip_summaries.py @ 480d6f41cddddc7efea9a09b79134811740ba17a
"""Generate the QUESTION-AGNOSTIC per-clip text-summary cache used by the
``summary_select_*`` methods (harnesses/clip_select.py).

One VLM call per UNIQUE clip, written as one JSON row per line — the cost
amortizes over every question and every future method, unlike per_stream's
question-conditioned summaries (K calls per question, nothing cacheable). The
prompt/frame constants are imported from clip_select.py so cache rows and any
live fallback are guaranteed identical.

Usage (from repo root, `internvl` conda env):
  # clips of one subset (e.g. the 357 unique clips of the 130-question set):
  python -m scripts.gen_clip_summaries --subset data/subsets/cvbench_temporal_subset.json
  # a full clip manifest (one relative path per line), sharded 4 ways on Slurm:
  MANIFEST=<manifest.txt> CHUNK=4 sbatch --array=0-3 \
      scripts/gen_clip_summaries.sbatch
  # verify a subset is fully covered before a sharded eval:
  python -m scripts.gen_clip_summaries --subset <subset.json> --check

Resumable: rows already present in --out (and its _shard*.jsonl siblings) are
skipped; rows with a recorded ``error`` are retried only under --retry-errors.
"""
import argparse
import json
import os

import yaml

from harnesses.clip_select import (SUMMARY_PROMPT, SUMMARY_PROMPT_VER,
                                   SUMMARY_NFRAMES, SUMMARY_MAX_NEW_TOKENS,
                                   summary_cache_files, _clip_meta)

_CFG = yaml.safe_load(open("configs/datasets.yaml"))
DEFAULT_OUT = _CFG["summaries_cache"]


def clip_list(args):
    """Ordered unique relative clip paths from --subset (union of video_1..4)
    or --manifest (one path per line)."""
    seen, clips = set(), []

    def add(rel):
        if rel and rel not in seen:
            seen.add(rel)
            clips.append(rel)

    if args.subset:
        for rec in json.load(open(args.subset)):
            for i in range(1, 5):
                add(rec.get(f"video_{i}"))
    else:
        with open(args.manifest) as fh:
            for line in fh:
                add(line.strip())
    return clips


def existing_rows(out):
    """(ok_videos, errored_videos) across the SAME file set the eval-time
    loader reads (summary_cache_files: base JSONL + its _shard*.jsonl
    siblings), so 'resume skipped it' / '--check passed' implies
    'summary_select will see it'. A video is 'ok' only if some row has a
    non-empty summary, no error, and the CURRENT prompt_ver (a prompt bump
    regenerates); it is 'errored' if it only has error/stale rows."""
    base = out[:-6] if out.endswith(".jsonl") else out
    base = base.rsplit("_shard", 1)[0] + ".jsonl"
    ok, errored = set(), set()
    for p in summary_cache_files(base):
        with open(p) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                v = row.get("video")
                if (row.get("summary") and not row.get("error")
                        and row.get("prompt_ver", SUMMARY_PROMPT_VER) == SUMMARY_PROMPT_VER):
                    ok.add(v)
                else:
                    errored.add(v)
    return ok, errored - ok


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--subset", help="question subset JSON; summarizes the union "
                                      "of its video_1..4 clips")
    src.add_argument("--manifest", help="text file with one relative clip path per line")
    ap.add_argument("--video-root", default=_CFG["video_roots"]["cvbench"])
    ap.add_argument("--backend", default="internvl3")
    ap.add_argument("--nframes", type=int, default=SUMMARY_NFRAMES)
    ap.add_argument("--max-new-tokens", type=int, default=SUMMARY_MAX_NEW_TOKENS)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--chunk", type=int, default=0, help="number of shards (Slurm array)")
    ap.add_argument("--offset", type=int, default=0, help="this shard index in [0,chunk)")
    ap.add_argument("--limit", type=int, default=0, help="only first N clips (smoke test)")
    ap.add_argument("--retry-errors", action="store_true",
                    help="re-attempt clips whose cached row recorded an error")
    ap.add_argument("--check", action="store_true",
                    help="no GPU: report clips missing from the cache and exit "
                         "nonzero if any (run before a sharded eval)")
    args = ap.parse_args()

    clips = clip_list(args)
    out = args.out
    if args.chunk and args.chunk > 1:
        clips = clips[args.offset::args.chunk]
        base = out[:-6] if out.endswith(".jsonl") else out
        out = f"{base}_shard{args.offset}.jsonl"
    if args.limit:
        clips = clips[: args.limit]

    ok, errored = existing_rows(args.out)
    skip = ok if args.retry_errors else (ok | errored)
    todo = [c for c in clips if c not in skip]
    n_err_cached = sum(1 for c in clips if c in errored)
    print(f"clips={len(clips)} cached_ok={sum(1 for c in clips if c in ok)} "
          f"cached_error={n_err_cached} todo={len(todo)} out={out}")

    if args.check:   # a clip counts as covered only by a non-error summary
        missing = [c for c in clips if c not in ok]
        for c in missing[:20]:
            print(f"  missing: {c}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")
        raise SystemExit(1 if missing else 0)
    if not todo:
        print("cache already complete for this clip set")
        return

    from run_vqa import make_backend
    backend = make_backend(args.backend, nframes=args.nframes)

    from tqdm import tqdm
    os.makedirs(os.path.dirname(out), exist_ok=True)
    n_err = 0
    with open(out, "a") as fh:
        for rel in tqdm(todo, desc="summarize"):
            vp = os.path.normpath(os.path.join(args.video_root, rel))
            dur, ndec = _clip_meta(vp)
            row = {"video": rel, "summary": "", "model": backend.name,
                   "nframes": args.nframes,
                   "duration_s": round(dur, 2) if dur is not None else None,
                   "n_decoded": ndec, "max_new_tokens": args.max_new_tokens,
                   "prompt_ver": SUMMARY_PROMPT_VER, "latency_s": None,
                   "input_tokens": None, "output_tokens": None, "error": None}
            try:
                g = backend.generate(
                    [{"role": "user", "content": [
                        {"type": "video", "video": vp, "nframes": args.nframes},
                        {"type": "text", "text": SUMMARY_PROMPT}]}],
                    max_new_tokens=args.max_new_tokens, seed=None, temperature=0.0)
                row.update(summary=g.text.strip(), latency_s=round(g.latency_s, 3),
                           input_tokens=g.input_tokens, output_tokens=g.output_tokens)
            except Exception as e:   # record + move on; --retry-errors re-attempts
                row["error"] = f"{type(e).__name__}: {e}"
                n_err += 1
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
    print(f"done: {len(todo)} attempted, {n_err} errors -> {out}")


if __name__ == "__main__":
    main()
