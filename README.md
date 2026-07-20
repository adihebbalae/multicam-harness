# multicam-harness

Shared eval code for the multi-camera harness experiments: how does the **packaging** of
multi-camera video for a frozen VLM (the *harness*) change QA accuracy, at an equal frame
budget? This is the in-process HF-transformers sibling of the vLLM-serving variant of the
harness (no `serve_vllm.sh` here; models load in-process).

## Strategies

| Harness | What the model sees | Calls/question |
|---|---|---|
| `uniform` | All clips' frames in one sequential prompt; the frame budget is split across clips by duration (`temporal_weighted`) or evenly (`temporal_even`); `cvbench_native` = the original CVBench packaging | 1 |
| `stitched` | Time-synchronized frames stitched into labeled grid montages (centralized) | 1 |
| `decentralized` | Per-stream query-conditioned text descriptions → a text-only aggregation call; `--stream-kind camera\|video` | k+1 |
| `clip_select` | The budget is spent on question-relevant clips: cached summaries → an LLM router (`summary_select_*`), or CLIP/SigLIP thumbnail scoring (`clip_select[_<scorer>]_top<m>`) | 1–2 |

## Layout

```
run_vqa.py        # CLI entry: subset × methods × backends × passes → results JSONL
runner.py         # run loop: sharding, resume keys, per-pass seeding
dataloaders/      # qa_json.py (record → messages, vendored), video.py (frame sampling)
harnesses/        # base.py + uniform.py / stitched.py / decentralized.py / clip_select.py
models/           # clients.py — Qwen3-VL and InternVL3 backends (InternVL import stays lazy)
evaluation/       # scoring.py (parse_choice / gt_choice) + run metrics & summaries
plotting/         # plot_results.py — Table 1 + Plots 1–4 from results JSONL
configs/          # datasets.yaml — video roots, subset paths, summary-cache path
scripts/          # run_bench.sbatch, gen_clip_summaries.sbatch (SLURM)
scripts/data/     # download_videos.py, fetch_meva_videos.py, fetch_egoexo_videos.py, fetch_agibot_videos.py
data/subsets/     # committed question-subset JSONs (they define the benchmark)
docs/             # ported spec + analysis writeups (provenance: docs/PORTING.md)
envs/             # cvbench.yml, internvl.yml conda environments
tests/            # compare_prompts_vs_fork.py — prompt-equivalence gate vs the reference
```

Run everything from the repo root — modules are top-level packages
(`from harnesses.uniform import ...`).

## Setup

```bash
conda env create -f envs/cvbench.yml
conda env create -f envs/internvl.yml
```

> ⚠️ **The two-env split is mandatory.** Qwen legs run under `cvbench`; InternVL3 runs
> under `internvl` — a transformers version conflict breaks InternVL3's remote code
> under `cvbench`.

Models load from the local HF cache; pre-download them on a login node
(`HF_HUB_OFFLINE=1` works after that).

## Run

```bash
python run_vqa.py --subset data/subsets/cvbench_full_runnable_subset.json \
    --methods temporal_weighted --backends internvl3 \
    --video-root <your CVBench video dir> \
    --passes 4 --seeds 1,2,3,4 --temperature 0.7
```

The same run under SLURM, sharded 8 ways:

```bash
ENV=internvl SUBSET=data/subsets/cvbench_full_runnable_subset.json \
    METHODS=temporal_weighted BACKENDS=internvl3 \
    CHUNK=8 sbatch --array=0-7 scripts/run_bench.sbatch
```

One leg of the stitch frame-budget sweep — the `centralized` 2×2 arm at `NFRAMES` frames
per clip with nothing else changed (design + findings in `docs/stitch_frame_sweep.md`):

```bash
ENV=internvl SUBSET=data/subsets/cvbench_full_runnable_subset.json BACKENDS=internvl3 \
    METHODS=centralized MONTAGE_KIND=video NFRAMES=32 TAG=_fullstitch32 CHUNK=8 \
    VIDEO_ROOT=<your CVBench video dir> \
    sbatch --array=0-7 scripts/run_bench.sbatch
```

The sbatch partition defaults to `gpul40q`; override with `sbatch -p <partition>`.
Error-bar convention for anything reported: `--passes 4 --seeds 1,2,3,4 --temperature 0.7`.

## Results

Runs append to `results/<subset>_<...>.jsonl` — one row per question × method × backend ×
pass — and write a `*_summary.json` next to it. `results/` and `logs/` are gitignored —
outputs never go in git. `scripts/finalize_cvbench_full.sh` pools the full-1000 3-way
shards into one report; `scripts/pool_stitch_sweep.sh` does the same for the stitch
frame-budget sweep legs (renaming each leg's rows to `stitch<NN>_f<N>` by its TAG so the
budgets stay distinct arms). Caution: finalize's glob matches *every* full-1000 shard,
sweep legs included — since sweep rows record `method='centralized'`, running it with
sweep shards present folds all budgets into its centralized arm, so move sweep shards
aside (or use only `pool_stitch_sweep.sh`) when both run families coexist.

## Summarization / clip-selection scoring

Answers are scored by deterministic choice parsing (`evaluation/scoring.py`
`parse_choice`; an unparseable answer = abstain = wrong). `clip_select` needs the
per-clip summary cache: generate it with `scripts/gen_clip_summaries.sbatch` →
`results/clip_summaries_internvl3.jsonl`. An LLM judge is **not** implemented in v1
(the sibling repo has one; port it when needed — do not fabricate one).

## Plotting

```bash
python plotting/plot_results.py --jsonl results/<run>.jsonl --out-dir results/figs
```

`--jsonl` accepts one or more results files (shards are pooled); `--out-dir` defaults to
a `figs/` directory next to the first input. Writes Table 1 (`table1.md/.csv`) and
Plots 1–4.

## Ground rules

- **Inference-only** — no training; the harness is the variable.
- **Equal budget** — compare harnesses at the same frame budget (watch token parity too).
- **One change at a time** — only visual packaging differs between harnesses.
- **Inspect by hand** — watch the clips before trusting a number.
- **Never commit data, video, weights, or run outputs** (see `.gitignore`).
- **Single greedy passes are preliminary** — no error bars until `--passes 4`.
- **Ours:** the two-conda-env split is mandatory; harness equivalence claims are enforced
  by `tests/compare_prompts_vs_fork.py` (prompts must stay byte-identical to the
  reference implementation).

## Data

Subset JSONs are committed under `data/subsets/` (they define the benchmark); videos are
never committed. CVBench videos: `scripts/data/download_videos.py` (HF). CrossView: the
release annotations plus `scripts/data/fetch_meva_videos.py` (public S3),
`scripts/data/fetch_egoexo_videos.py` (license-gated), and
`scripts/data/fetch_agibot_videos.py` (gated HF). Set your video roots in
`configs/datasets.yaml` or pass `--video-root`.
