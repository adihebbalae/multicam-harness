<!-- Ported from Wavy-Hec/CVBench analysis/cvbench_temporal_method.md @ 480d6f41cddddc7efea9a09b79134811740ba17a -->
# CVBench — Duration-Weighted Temporal Sequencing

*Mentor's "Next tasks" — implementation + experiment writeup.*
*Status (2026-06-29): code landed; **130-question 3-way run complete, reported & verified** (§7); **full-1000 3-way run complete, finalized & reported** (§7 — temporal_weighted 61.8% > even 60.8% > 2×2 stitch 57.4%). See the fork's `analysis/cvbench_temporal_STATUS.md` for the literal input format and finalize details.*

---

## 1. What was asked (the 6 sub-tasks) and how each is met

| # | Mentor's task | How it's implemented |
|---|---|---|
| 1 | **Arrange the videos in sequence** (Vid1 → Vid2 → Vid3) | The K clips of a question are fed to the model **sequentially** in their given `video_1..video_N` order, each preceded by a `=== Video k of K ===` banner. We do **not** reorder them — some CVBench questions *ask* for the chronological order (permutation answers), so the given index order is presented and the model reasons about ordering. |
| 2 | **64-frame budget weighted by duration** (2m/4m/2m → 16/32/16) | A single **total** budget (default 64) is split across the clips **in proportion to each clip's duration** via largest-remainder apportionment (`allocate_frames`). Reproduces the board's example exactly: `[120,240,120]s → [16,32,16]`. |
| 3 | **Sample frames uniformly within each clip** up to its allotment | Each clip is passed as a `{"type":"video","nframes":n_k}` block; the backend samples `n_k` frames **uniformly** across that clip's full duration (`load_video` in `models/clients.py` / Qwen `process_vision_info`). |
| 4 | **Enable reasoning mode** | The `<think>…</think> / <answer>…</answer>` scaffold (`QUESTION_TEMPLATE`) is in every prompt; `max_new_tokens=8192`. The full reasoning trace is now **persisted** to the results JSONL (`response_text`, `think`) so failures are interpretable. |
| 5 | **Prompts that disambiguate which frames belong to which video** | A prefix paragraph explains the sequential, duration-weighted layout; each clip gets a `=== Video k of K — n_k frames sampled uniformly across its full duration (~Ds) ===` banner; frames are labeled `Frame1, Frame2, …` within each clip. |
| 6 | **Confirm the "2×2 case" is the spatial stitching** | **Confirmed — yes.** See §4. |

---

## 2. The "130 set" and the full set

- **130 set** = `data/subsets/cvbench_temporal_subset.json` — the CVBench *temporal-logic* questions (rule-based `temporal_level ≥ 1`): **130 questions, 357 distinct clips, 2/3/4 clips per question = 48/23/59.** All 357 clips are present on disk and decode (the fork's stale `cvbench_temporal_missing.txt` predates the Jun-25 download — ignore it).
- **Full set** = the 1000-question `CVBench.json`. It references **1315 distinct clips**; 435 were on disk, **880 are being downloaded** from the public HF dataset (`Dongyh35/CVBench`) for the overnight run. The runnable subset is rebuilt by the fork's `analysis/make_cvbench_full_runnable_subset.py` as clips arrive.

---

## 3. The duration-weighted frame allocator

`harnesses/uniform.py :: allocate_frames(weights, budget=64, floor=2, caps=None)`

1. **Proportional split** of the whole budget by clip weight (duration), using **largest-remainder (Hamilton)** apportionment so the per-clip counts **sum exactly to the budget**. Pure proportional when nothing else binds → `[120,240,120] → [16,32,16]`.
2. **Caps**: a clip is never asked for more frames than it physically has (`len(VideoReader)`); any surplus is redistributed to the other clips.
3. **Floor** (default 2) is a **safety net only** — applied *after* the proportional split, it raises a clip whose share rounded below 2 up to 2 (borrowing from the clips most above their own floor). On the real 130 it never fires, so the split stays purely proportional.

**Worked examples (verified against the live module):**

| Durations (s) | Allocation (budget 64) | Note |
|---|---|---|
| `[120, 240, 120]` | `[16, 32, 16]` | the board's canonical example, reproduced exactly |
| `[146.6, 258.1]` (cvb-0, 2 clips) | `[23, 41]` | longer clip gets more |
| `[146.6, 52.1, 99.5]` | `[32, 11, 21]` | uneven 3-clip |
| `[52.1, 99.5, 58.3, 152.6]` (cvb-15, 4 clips) | `[9, 18, 10, 27]` | 4-clip, sums to 64 |
| `[130, 3]` | `[62, 2]` | floor net guarantees the tiny clip ≥ 2 |

All 130 questions verified: every allocation sums to 64 and respects per-clip frame caps.

---

## 4. The 2×2 question — answered

**Yes — the "2×2 case" is the spatial-stitching method we already use, named `centralized` in the harness** (`harnesses/stitched.py`).

- `grid_layout(k)` (`harnesses/stitched.py`) tiles the K synchronized frames into a grid: 1 clip → 1×1, 2 → 1×2, **3–4 clips → 2×2** (one empty cell for k=3). The grid is built with PIL, each cell labeled.
- For CVBench's *independent* clips it runs with `--montage-kind video` (cells labeled "Video i", with a preamble that says the clips are unrelated rather than synchronized).
- This is the method behind the earlier "stitching helps" results. So the temporal-sequencing method here is a **different** presentation (sequential frames, duration-weighted budget) that we now compare **head-to-head against** the 2×2 spatial stitch at an equal 64-frame budget.

---

## 5. The 3-way experiment (equal 64-frame budget, InternVL3-8B, 4 passes)

All three arms run on the **same** 130 questions, **same** backend (InternVL3-8B under the `internvl` conda env), **same** total frame budget (64), **4 sampled passes** (temp 0.7, seeds 1-4) for std error bars.

| Arm | Method | What it isolates |
|---|---|---|
| **A. duration-weighted** | `temporal_weighted` (`--weighting duration`) | the new method — frames split by clip duration |
| **B. even split (control)** | `temporal_even` (`--weighting even`) | identical sequencing/markers/total, but frames split **evenly** ⇒ isolates *"does duration weighting help?"* (this is the budget-matched `cvbench_native`) |
| **C. 2×2 spatial stitch** | `centralized --montage-kind video` | the spatial baseline (the "2×2 case") |

> Arms A and B use distinct result labels (`temporal_weighted` vs `temporal_even`) so they don't collide on the JSONL resume key.

**Reporting:** accuracy ± std by method, **and within each clip-count stratum (2/3/4 clips)** — the 48/23/59 mix means a pooled headline can hide a Simpson's-paradox effect, so per-stratum is the honest comparison.

---

## 6. How to run

```bash
# Arm A — duration-weighted, 4-pass, InternVL3, CVBench videos
ENV=internvl BACKENDS=internvl3 METHODS=temporal_weighted WEIGHTING=duration \
  SUBSET=data/subsets/cvbench_temporal_subset.json \
  VIDEO_ROOT=<your CVBench video dir> \
  BUDGET=64 FLOOR=2 PASSES=4 SEEDS=1,2,3,4 TEMPERATURE=0.7 MAX_NEW_TOKENS=8192 \
  OUT=results/bench_cvbench_temporal_internvl.jsonl \
  CHUNK=4 sbatch --array=0-3 scripts/run_bench.sbatch

# Arm B — even split: same command with WEIGHTING=even
# Arm C — 2x2 stitch: METHODS=centralized MONTAGE_KIND=video NFRAMES=16

# Report — the run writes a *_summary.json next to the JSONL; Table 1 + plots:
python plotting/plot_results.py --jsonl results/bench_cvbench_temporal_internvl.jsonl
```

For the overnight full set: after the download finishes, the fork's `analysis/make_cvbench_full_runnable_subset.py` rebuilds the runnable subset (`data/subsets/cvbench_full_runnable_subset.json` here), then the same three arms run with `SUBSET=data/subsets/cvbench_full_runnable_subset.json`.

---

## 7. Results

### 130-question set — InternVL3-8B, 64-frame budget, 4 passes (seeds 1–4, T=0.7)

All three arms share the **same 64-frame total budget**; the duration vs even arms also share the
*same* sequential framing and `=== Video k of K ===` banners — so the only thing that changes between
them is **how the 64 frames are split across clips**. This isolates the weighting effect cleanly.

| Method | Accuracy ± std | 2-clip | 3-clip | 4-clip | in-tok | abstain / err |
|---|---|---|---|---|---|---|
| **temporal_weighted (duration)** | **55.8 ± 0.7%** | 63.5% | 50.0% | **51.7%** | 17.2k | 0% / 0 |
| temporal_even (budget-matched control) | 51.5 ± 2.7% | 59.9% | 43.5% | 47.9% | 17.2k | 0% / 0 |
| centralized (2×2 spatial stitch) | 51.5 ± 2.6% | 62.5% | 51.1% | 42.8% | 4.4k | 0% / 0 |

**Headline:** duration-weighted sampling beats the budget-matched even split by **+4.3 points
(55.8% vs 51.5%) at an identical 64-frame budget and identical sequencing** — i.e. the gain comes
purely from *allocating frames in proportion to clip length*, not from any extra frames or framing.
It is also markedly more **stable across seeds** (±0.7 vs ±2.7), because long clips no longer get
under-sampled by a fixed-per-clip budget.

**Where the gain concentrates (by #clips).** The advantage grows with clip count — exactly where
duration disparity is largest. On 4-clip questions duration-weighting leads even-split by +3.8 and
the 2×2 stitch by **+8.9** points. The spatial stitch degrades monotonically as clips are added
(62.5% → 51.1% → 42.8%) because each added clip shrinks every cell's resolution; duration-weighting
holds roughly flat (63.5% → 50.0% → 51.7%) by spending frames where the content is. See
the fork's run-output figure `figs_temporal_130/acc_by_numclips.png`.

**By task type** (full breakdown in `*_report.md`). Duration-weighting tops the field on
Multi-video Key-Action Recognition (91.7% vs 58.3% stitch), Cross-video Event Retrieval
(80.6% vs 61.1%), Multi-view Scene Understanding (58.3% vs 41.7%) and Cross-video Counterfactual
Reasoning (78.1%). Its one clear regression is Cross-video **Object** Recognition (31.2% vs 75.0%
stitch) — static fine-grained object identity is the one place the high-resolution 2×2 tiling wins,
a known trade-off of temporal sampling.

Artifacts (in the fork's bench/results run data): `bench_cvbench_temporal_subset_internvl_ALL.jsonl`
(1,560 rows = 3 arms × 130 Q × 4 passes), `…_ALL_report.{md,csv}`, figures under
`figs_temporal_130/`.

### Full set (1000-Q runnable subset) — InternVL3-8B, 64-frame budget, 4 passes — COMPLETE

The same three arms ran over `data/subsets/cvbench_full_runnable_subset.json` (1000 runnable Q,
**0 blocked**) on InternVL3-8B, 4-pass, 64-frame budget (SLURM arrays 58246/58247/58248, 8 shards
each), finalized via the fork's `bench/finalize_cvbench_full.sh`.

| Method | Accuracy ± std | 2-clip | 3-clip | 4-clip | in-tok | abstain / err |
|---|---|---|---|---|---|---|
| **temporal_weighted (duration)** | **61.8 ± 0.7%** | 62.7% | 60.4% | **61.6%** | 17.2k | 0% / 0 |
| temporal_even (budget-matched control) | 60.8 ± 0.8% | 61.6% | 59.9% | 60.1% | 17.2k | 0% / 0 |
| centralized (2×2 spatial stitch) | 57.4 ± 1.4% | 59.2% | 59.6% | 52.3% | 4.4k | 0% / 0 |

**Headline:** duration-weighted wins on the full set too — **+1.0 pt over the budget-matched even
split** (61.8% vs 60.8%) and **+4.4 pts over the 2×2 stitch** (57.4%). The weighted-vs-even gap is
narrower than on the temporal-enriched 130 set (+4.3) because the full set is mostly non-temporal
questions, but the **4-clip advantage holds**: the stitch collapses to 52.3% at 4 clips (−6.9 from
its 2-clip 59.2%) while duration-weighting stays flat at 61.6% — a **+9.3 pt** lead exactly where
clip-length disparity is largest. By task type, temporal sampling dominates Cross-video Event
Retrieval (70.9% vs 52.6% stitch), Multi-view Scene Understanding (81.4% vs 72.7%) and Cross-video
Procedural Transfer (68.6% vs 60.3%); the 2×2 stitch's one clear win is fine-grained Multi-video
Attribute Recognition (75.0% vs 70.1%). The 130-set Object-Recognition regression did **not**
replicate at scale (weighted 57.0% now narrowly tops stitch 54.4%).

Artifacts (in the fork's bench/results run data): `bench_cvbench_full_runnable_subset_internvl_ALL.jsonl`
(12,000 rows = 3 arms × 1000 Q × 4 passes), `…_ALL_report.{md,csv}`, figures under
`figs_temporal_full/`.

---

## 8. Files

| File | Role |
|---|---|
| `harnesses/uniform.py` | `TemporalWeightedMethod` + `allocate_frames` / `largest_remainder` |
| `harnesses/base.py` | `Result` gains `response_text`, `think`, `frame_alloc` |
| `run_vqa.py` | registers method; `--budget` / `--floor` / `--weighting` |
| `scripts/run_bench.sbatch` | `BUDGET` / `FLOOR` / `WEIGHTING` env passthrough |
| the fork's `analysis/make_cvbench_temporal_subset.py` | builds the 130 temporal-logic subset (not ported) |
| the fork's `analysis/make_cvbench_full_runnable_subset.py` | builds the full runnable subset (grows with the download; not ported) |
| `data/subsets/cvbench_temporal_subset.json` | the 130 questions |
| the fork's `bench/cvbench_temporal_figs.py` | 3-way comparison figures (acc by method / #clips / task; not ported) |
| the fork's `bench/finalize_cvbench_full.sh` | one-command finalize for the full-1000 run (pool → report → figs; not ported) |
| the fork's `analysis/cvbench_temporal_STATUS.md` | live run status, literal input format, finalize/handoff (not ported) |

---

## 9. Appendix — rendering the literal prompt

The exact text fed to the model for any question can be reproduced offline (no GPU):

```python
import json
from harnesses.uniform import PREFIX, MARKER, SPLIT_DESC
from dataloaders.qa_json import build_messages, video_paths

VIDEO_ROOT = "<your CVBench video dir>"
rec = {r["id"]: r for r in json.load(open("data/subsets/cvbench_temporal_subset.json"))}["cvb-974"]
base, yn = build_messages(rec, VIDEO_ROOT, 8, no_video=True)   # scaffold text only
scaffold = base[0]["content"][0]["text"]
# durs / nframes come from Result.frame_alloc in the JSONL (or _clip_meta + allocate_frames):
durs, nframes, K, budget = [111.18, 116.98, 9.95, 193.69], [17, 17, 2, 28], 4, 64
print(PREFIX.format(K=K, budget=budget, split=SPLIT_DESC["duration"].format(K=K)))
for k, (n, d) in enumerate(zip(nframes, durs), 1):
    print(MARKER.format(k=k, K=K, n_k=n, dur_s=d), "<video block>")
print(scaffold)
```

A rendered example (cvb-974) and the per-clip duration→frames split are in
the fork's `analysis/cvbench_temporal_STATUS.md → "How we put the inputs into the model"`.
