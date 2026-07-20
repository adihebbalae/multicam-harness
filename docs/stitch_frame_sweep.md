<!-- Results generated with the reference implementation (Wavy-Hec/CVBench bench/ — this
     repo's harnesses are prompt-equivalence-gated against it, tests/compare_prompts_vs_fork.py);
     pooling script ported @ 4fce134e79bed7400ae9ddb0c28f8744e65d22f3. Doc written 2026-07-20. -->
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
