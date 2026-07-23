<!-- Results generated with the reference implementation (Wavy-Hec/CVBench bench/ — this
     repo's harnesses are prompt-equivalence-gated against it, tests/compare_prompts_vs_fork.py);
     pooling script ported @ 4fce134e79bed7400ae9ddb0c28f8744e65d22f3. Doc written 2026-07-20;
     chance baseline + figure scripts ported @ db18c06633ffd3183d008bf3fd8ac8138d273cd5,
     added 2026-07-22. -->
# Stitch frame-budget sweep — CVBench full-1000, InternVL3-8B

**Question.** The stitched (centralized 2×2 montage) harness loses to temporal sequencing
at the standard budgets. Is it *frame-starved* — would more frames per clip close the gap —
or is the grid packaging itself the ceiling?

**Design.** One arm, one knob: the `centralized` stitch harness re-run at per-clip frame
budgets N ∈ {8, 16, 32, 64, 128}, with **nothing else changed** between legs
(one-change-at-a-time). Protocol per leg:

- CVBench **full-1000** runnable subset (`data/subsets/cvbench_full_runnable_subset.json`),
  InternVL3-8B, `MONTAGE_KIND=video` (honest "Video i" cell labels).
- `--passes 4 --seeds 1,2,3,4 --temperature 0.7` (error-bar convention), 8-way sharded.
- InternVL tile budget fixed at 1 tile/montage across all legs, so each 2×2 montage costs
  256 visual tokens regardless of N and input tokens scale linearly with N (~224 px per
  camera cell inside the 448-px tile). Token parity across legs is therefore *deliberately
  broken* — the frame budget is the experiment variable, not a controlled nuisance; each
  leg compares to other harnesses only at its own budget.

**Leg dates:** f16 is the original full-run stitch leg (2026-06-28); f64 (2026-07-13)
doubles as the equal-token control from the temporal-vs-stitch comparison; f8/f32/f128
ran 2026-07-15 → 07-16.

## Headline (4 000 rows/leg = 1 000 Q × 4 passes; 0 errors; ≤ 9 abstains per leg)

| leg | frames/clip | accuracy (mean ± std %) | in tok (mean) | out tok (mean) | lat p50 (s) |
|---|--:|--:|--:|--:|--:|
| `stitch08_f8` | 8 | **57.7 ± 0.6** | 2 346 | 485 | 8.1 |
| `stitch16_f16` | 16 | 57.4 ± 1.4 | 4 419 | 407 | 7.6 |
| `stitch32_f32` | 32 | 57.5 ± 0.7 | 8 563 | 436 | 8.3 |
| `stitch64_f64` | 64 | 55.8 ± 0.5 | 16 851 | 382 | 10.0 |
| `stitch128_f128` | 128 | 54.9 ± 0.8 | 33 428 | 342 | 16.2 |

By camera count, every leg is weakest on 4-cam questions, and more frames never fix that
weakness (f8: 59.5/58.7/53.8; f128: 57.3/56.5/49.3 for 2/3/4 cams).

## Chance baseline

Every accuracy above should be read against what a model scores by guessing, and that
floor is **not** a flat 25%. CVBench mixes 861 four-option questions with 139 yes/no ones,
so the expected score of a guesser picking uniformly among the options it is shown is

    chance = mean over that set's questions of 1 / (number of answer options)

which differs by task category. `evaluation/chance.py` computes it from the subset's real
option lists (`python -m evaluation.chance` prints the full table):

| set | option mix | chance |
|---|---|--:|
| **full-1000 (pooled)** | 139 yes/no, 861 four-option | **28.5%** |
| Cross-video Scene Recognition | 122 yes/no, 27 four-option | 45.5% |
| Cross-video Entity Matching | 8 yes/no, 66 four-option | 27.7% |
| Joint-video Spatial Navigating | 2 yes/no, 40 four-option | 26.2% |
| Multi-video Temporal Reasoning | 2 yes/no, 73 four-option | 25.7% |
| all-four-option categories | — | 25.0% |

Two consequences for the reading below. First, raw accuracy flatters the categories with
high floors: Cross-video Scene Recognition ranks 4th of the fifteen on raw accuracy (68.0%
at f8) but most of its questions are binary, so it clears its own 45.5% floor by only
+22.5 — 13th of fifteen on margin, and second-lowest of all on accuracy relative to its
floor (1.49×). Ranking by margin rather than accuracy moves it most.
Second, and going the other way, Multi-video Temporal Reasoning at f64 sits only **+9.3 pts
above chance** with a ±3.3 pass std — at the larger budgets that category is not merely
degrading, it is approaching the guessing floor. The smallest margins overall belong to
Joint-video Counting (+13.8 averaged over the sweep) and, at f64 and f128, to Temporal
Reasoning; `python -m plotting.frame_budget_smallmultiples` prints the full ranking.

One source-data caveat the calculator handles: a single row's four choices are
concatenated into one options string, which a naive `len(options)` reads as one option
(chance 100%). Embedded `A. / B. / C. / D.` markers are detected and the row counts as 4.

## Reading

1. **Not frame starvation.** Accuracy is flat from 8 to 32 frames/clip and then *declines*
   (−2.9 pts at f128, well outside the stds). The stitched harness's deficit vs temporal
   sequencing is the packaging, not the frame count — consistent with the earlier
   equal-token f64 control.
2. **Where the extra frames hurt:** the biggest f8→f128 drops are Multi-video Temporal
   Reasoning (50.3 → 36.7), Cross-video Event Retrieval (56.1 → 44.4) and Cross-video
   Scene Recognition (68.0 → 60.6) — retrieval/temporal tasks drown in near-duplicate
   montages. Small gains appear on Joint-video Spatial Navigating (38.1 → 46.4) and
   Procedural Transfer (56.9 → 60.8), but they don't offset the losses.
3. **The model also reasons less as inputs grow:** mean output tokens trend down
   (485 at f8 → 342 at f128) while input grows 14×.
4. **f128 caveat:** at ~33 k input tokens this leg overruns InternVL3-8B's native context
   window and leans on the model's dynamic RoPE scaling — treat that arm as
   out-of-distribution for the model, not just for the harness.

## Reproduce

One SLURM leg per budget (only `NFRAMES`/`TAG` change; `VIDEO_ROOT` = your CVBench video dir):

```bash
ENV=internvl SUBSET=data/subsets/cvbench_full_runnable_subset.json BACKENDS=internvl3 \
  METHODS=centralized MONTAGE_KIND=video NFRAMES=32 TAG=_fullstitch32 CHUNK=8 \
  VIDEO_ROOT=<your CVBench video dir> \
  sbatch --array=0-7 scripts/run_bench.sbatch
```

Then pool the legs into one per-budget report (each leg records `method='centralized'`,
so pooling renames rows to `stitch<NN>_f<N>` keyed by the leg's TAG):

```bash
bash scripts/pool_stitch_sweep.sh
```

Result JSONLs are not committed (ground rules); the report regenerates from the shards.

### Figures

Both figure scripts recompute every number from the pooled JSONL (per-pass accuracy, then
mean and population std across passes) and draw each series' own chance line, so nothing
is hardcoded:

```bash
# four-series line chart: Overall + the three categories that move most
python -m plotting.frame_sweep_by_category --jsonl results/<pooled>.jsonl

# small multiples: one panel per task type, sorted by frame-budget gain
python -m plotting.frame_budget_smallmultiples --jsonl results/<pooled>.jsonl
```

Both take `--qa-json` for the option counts — it defaults to `subsets.cvbench_full` from
`configs/datasets.yaml` when that key is set, otherwise to
`data/subsets/cvbench_full_runnable_subset.json` — and `--out-dir` (defaults to
`figures/frame_sweep/`, gitignored).
Each prints the plotted arrays plus an above-chance row before writing png/pdf/svg.
