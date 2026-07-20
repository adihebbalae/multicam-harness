<!-- Ported from Wavy-Hec/CVBench analysis/adaptive_frames_experiment.md @ 480d6f41cddddc7efea9a09b79134811740ba17a -->
# Multi-Clip Video QA — Clip Selection & Frame Sampling

CVBench temporal-sequencing eval, InternVL3-8B. This doc covers this line of work and its
action items only.

> **Meeting reframing (key takeaway).** The advisor's priority order is now explicit:
> **clip selection** — *which* clips (and *how many*) are relevant to a question — is the
> **PRIMARY** decision; **intra-clip frame sampling** is **SECONDARY**. Wrong clip selection
> directly causes wrong answers or context bloat, so it must be resolved **before** optimizing
> frame sampling. The frame-level experiment (§B) is closed — smarter frames didn't beat
> uniform — and the headline task is the **question-driven clip-selection** mechanism (§3).

---

## 1. Action items & deadlines

| # | Deliverable | Due | Status |
|---|---|---|---|
| **D1** | Slides / Google Doc clearly explaining the **clip-selection methodology** | Next meeting | to do |
| **D2** | Quantitative results: **stitching vs non-stitching**, **varying clip counts**, on CVBench (or UT dataset) | **Wed 2026-07-01** | data largely in hand → assemble (§2) |
| **D3** | A **principled, question-driven adaptive clip-selection** mechanism (not the current fixed sequential order) | ongoing research | design drafted (§3) |

Supporting context presented:
- Multi-clip benchmark on a subset of **temporally-correlated** questions; **thinking mode enabled**.
- **Duration-weighted** frame sampling: frames ∝ clip duration — e.g. ~17 frames for ~111–117 s
  clips, **~2 frames for a ~10 s clip** (this is literally cvb-974 → `[17,17,2,28]`, §A3).
- Preliminary: **even-split ≳ stitching**, and the gap widens as **#clips grows**. Hypothesis:
  many clips are **independently answerable**, so spatial stitching is counterproductive.
- **Update 2026-07-02:** the §B frame-selection ablation landed — motion 49.4% (−6.4),
  CLIP-frames 55.8% (tie) vs uniform 55.8%. Supports frame-sampling-is-secondary; the methods
  were retired from the harness and effort shifted to D3 clip selection (§3, v1 implemented).

---

## 2. D2 — stitching vs non-stitching × clip count (Wed deliverable) ✅

CVBench **full-1000**, InternVL3-8B, 4 passes, temp 0.7, 64-frame budget. Non-stitch = clips shown
sequentially; stitch = `centralized` 2×2 spatial montage.

**Accuracy by clip count K (mean ± std over 4 passes):**

| method | K=2 (n=487) | K=3 (n=236) | K=4 (n=277) | all |
|---|---|---|---|---|
| non-stitch (duration) | 62.7 ± 1.0 | 60.4 ± 2.6 | **61.6 ± 0.6** | **61.8** |
| non-stitch (even) | 61.6 ± 0.8 | 59.9 ± 1.2 | 60.1 ± 1.2 | **60.8** |
| stitch (2×2) | 59.2 ± 1.4 | 59.6 ± 1.7 | **52.3 ± 2.9** | **57.4** |

**Δ (best non-stitch − stitch), by K:  K=2 +3.4 · K=3 +0.7 · K=4 +9.2 pts**

*(Figure: accuracy by clip count — regenerate with the fork's `bench/cvbench_d2_clipcount_fig.py`.)*

**Finding (supports the hypothesis).** Stitching is roughly competitive at low K (K=2: −3.4; K=3:
≈tie) but **collapses at K=4 (52.3% vs 61.6%, −9.2 pts)** — i.e. as more clips are packed into one
2×2 montage, each view's effective resolution/temporal coverage degrades and the model can no longer
read them. This is direct evidence for *"many clips are independently answerable, so forcing them
into a shared stitched frame is counterproductive,"* and it gets worse exactly where clip selection
matters most (more clips ⇒ more chance some are irrelevant). Motivates **D3** (pick the right clips
rather than cramming all K together).

*Regenerate:* the fork's `bench/cvbench_d2_clipcount_fig.py` (no GPU; reads
the fork's pooled run JSONL, `bench_cvbench_full_runnable_subset_internvl_ALL.jsonl`).

---

## 3. D3 — principled adaptive CLIP **clip** selection (the primary problem)

In the baseline, clip selection is **fixed and sequential** — all K clips are fed in order,
budget split by duration; there is **no mechanism to choose which clips a question actually
needs**. The mechanism (v1 implemented in `harnesses/clip_select.py`, eval in flight):

1. **Score each clip's relevance to the question** — CLIP/SigLIP text-image similarity between the
   question and a cheap uniform thumbnail set per clip (or a lightweight VLM relevance pass).
2. **Select clips** — keep the top-`m` by relevance (or threshold), instead of always using all K;
   this is the "which clips / how many" decision the advisor flagged.
3. **Allocate the 64-frame budget by relevance** (not just duration) across the selected clips —
   a relevance-weighted generalization of `allocate_frames`.
4. **Ablate** vs the fixed-sequential baseline on the same subset: does question-driven selection
   beat "use everything," especially at larger K and when some clips are irrelevant?

This directly attacks the primary decision. Three v1 arms are implemented and running
(2026-07-02): `summary_select_route` (per-clip cached text summaries → one LLM selector call
that keeps the minimal clip set or ALL), `summary_select_top1` (forced single clip — the
hard-pruning diagnostic), and `clip_select_top1` (the CLIP-thumbnail scorer from step 1, no
LLM). Summaries come from `scripts/gen_clip_summaries.sbatch` →
`results/clip_summaries_internvl3.jsonl`; selection quality is audited by
the fork's `analysis/clip_selection_diagnostic.py`.

---

## 4. D1 — methodology to explain in the deck

The clip-selection + sampling pipeline, end to end (the parts to make legible in slides):
- **How many frames per clip** = `allocate_frames` (proportional, largest-remainder); **which
  frames within a clip** = the backend's uniform `get_index` (§A). Today **clip selection itself
  is the identity** (use all K) — that's the gap.
- The **primary vs secondary** framing, with cvb-974 as the worked example (a 10 s clip getting
  2 frames vs a 194 s clip getting 28).
- Why frame sampling looks *secondary* — the landed §B ablation: motion-based selection
  **−6.4 pts**, query-conditioned selection **tie**, vs the uniform baseline. Smarter *frames*
  bought nothing under this setup; the open upside is smarter *clips* (§3).

---

# Appendix A — how frames are chosen *today* (secondary decision, verified)

> A 5-agent adversarial review re-derived A1–A3 against the code + a clean-room reimplementation
> of `allocate_frames`; all confirmed (only correction = the `linspace`→`segment-midpoint` note).

### A1. Two functions: *how many* vs *which indices*

`allocate_frames` decides the per-clip **count**; the **within-clip index choice is delegated to
the backend** — the uniform harness (`harnesses/uniform.py`) never picks indices.

```python
# harnesses/uniform.py  (how many)
weights = durs if self.weighting == "duration" else [1.0] * K
nframes = allocate_frames(weights, self.budget, floor=self.floor, caps=ncaps)

# harnesses/uniform.py  (emits COUNT only — no indices)
content.append({"type": "video", "video": vp, "nframes": n_k})
```

```python
# models/clients.py (InternVL client)  (which indices — the uniform get_index grid;
# an explicit frame_indices override exists but no current method uses it)
if frame_indices is None:
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
for frame_index in frame_indices:
    frame_index = int(min(max(0, int(frame_index)), max_frame))   # clamp into range
    img = Image.fromarray(vr[frame_index].asnumpy()).convert("RGB")   # pixels read AFTER indices fixed
```

### A2. The within-clip sampler is uniform & blind to question/content

```python
# models/clients.py (InternVL client)
def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound: start, end = bound[0], bound[1]
    else:     start, end = -100000, 100000          # the uniform harness always passes bound=None
    start_idx = max(first_idx, round(start * fps))
    end_idx   = min(round(end * fps), max_frame)
    seg_size  = float(end_idx - start_idx) / num_segments
    return np.array([int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
                     for idx in range(num_segments)])
```

- **Content/question-blind:** indices are a pure function of `(max_frame, num_segments)`; the
  question never enters this path; pixels are read only *after* indices are fixed.
- **Evenly spaced**, but precisely **segment-midpoints** (`idx_j = seg/2 + seg·j`), *not* literal
  `linspace(0,n-1)` — first/last frames are never sampled.

### A3. cvb-974 — concrete (weighted vs even)

```
clip 1: n=3332 dur=111.18s   clip 2: n=3506 dur=116.98s
clip 3: n= 298 dur=  9.95s   clip 4: n=5805 dur=193.69s
```

**`temporal_weighted` → `[17,17,2,28]`** (clip 3 = 2 frames @ t≈2.47,7.41 s; clip 4 = 28 frames
~7 s apart). **`temporal_even` → `[16,16,16,16]`** (clip 3 oversampled ≈ every 0.6 s; clip 4
starved ≈ every 12 s). *Nuance: clip 3's `2` is the floor (rescued from quota 1.47).*

### A4. How the assembled input reaches the model (presentation format)

- Clips are presented **sequentially as separate video blocks** (no stitching): a preamble states
  the clips are independent, un-synchronized, and that the 64-frame budget is split across them;
  then each clip gets a text banner `=== Video k of K — n_k frames sampled uniformly across its
  full duration (~Ns); clip k of the sequence ===` followed by its frames labeled
  `Frame1:, Frame2:, …` (numbering restarts per clip); the question/options scaffold comes last.
- Each frame is a single **448×448 tile** (`max_num=1`, no dynamic tiling) ⇒ **256 visual tokens
  per frame**; a 64-frame question ≈ 16.4k visual tokens.

---

# Appendix B — adaptive frame selection (tested, lost — RETIRED 2026-07-02)

One controlled experiment, closed with a clear answer. Two "smarter frame" variants were tested
against the uniform baseline under identical everything (same 64-frame budget, same per-clip
counts, byte-identical prompt — only the chosen frame *indices* differed): **motion keyframes**
(`adaptive_content`) and **CLIP question-relevance frames** (`adaptive_query`). 130 questions ×
4 passes, InternVL3-8B (SLURM 58447, 2026-06-29; population std, same convention as all tables):

| method | accuracy | Δ vs uniform |
|---|---|---|
| **uniform (baseline)** | **55.8 ± 0.7%** | — |
| motion keyframes | 49.4 ± 4.3% | **−6.4 (worse)** |
| CLIP-by-question frames | 55.8 ± 2.0% | **0.0 (tie)** |

**Conclusion:** at a 64-frame budget, *which frames you take inside a clip* is not the lever —
uniform coverage is already as good as the best adaptive variant, and motion-seeking actively
hurts (it abandons the even temporal coverage that ordering questions need). This is the
empirical basis for retiring frame-level adaptivity and moving to **clip selection** (§3).

*Housekeeping:* the `adaptive_content`/`adaptive_query` methods were **removed from the harness**
(2026-07-02; the fork's `bench/methods/adaptive.py` deleted, registry entries dropped) so they
can't clutter future runs. The raw run artifacts remain for audit in the fork's bench/results
run data (`bench_cvbench_temporal_subset_internvl_adaptive_shard*.jsonl`; the table above
recomputes directly from them). The reusable piece — the CLIP text-image scorer — lives on in
`harnesses/clip_select.py`, where it now scores whole clips instead of frames.

---

## Reproduce

- **D2 figure/table:** the fork's `bench/cvbench_d2_clipcount_fig.py` (no GPU).
- **§B table:** recompute from the archived shards (no GPU) — group rows by `method`/`pass_idx`,
  accuracy = mean of `correct`, std over the 4 pass accuracies.
- The §B run command itself no longer works (methods retired by design).
