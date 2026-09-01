<!-- Numbers from the reference implementation's registered legs (Wavy-Hec/CVBench
     bench/; this repo's arms are prompt-equivalence-gated against it), media-valid
     rows only (MEVA decoded through the remuxed containers, media_remap stamped), split
     by evaluation/evidence_class_split.py against data/subsets/meva_evidence_labels.json.
     Written 2026-08-31. -->
# Empirical performance bounds — MEVA (meva1033 pool), InternVL3-8B

One dataset, one model. Three questions: what is the lowest score a sighted arm
has to beat, what do the all-view arms actually score, and what is the most a
system could score if it always picked the right view — the theoretical maximum
with an optimal view.

## 1. Setup

- **Pool** — `data/subsets/crossview_meva1033_subset.json`: 1,033 four-option
  MCQs from the MEVA temporal / event-ordering / spatial release questions,
  at most 4 views per question (955 four-view, 75 two-view, 3 three-view), every
  question's evidence cameras kept (`cap_answer_safe` on all). By evidence
  class: 431 C2, 602 C4 (`docs/evidence_taxonomy.md`).
- **Media** — the release `.avi` clips decoded through their remuxed `.mp4`
  siblings (`scripts/data/remux_avi.py`). Rows produced from the original
  containers return the wrong frame on random access and are excluded from
  every table here.
- **Model** — InternVL3-8B, one tile per frame (256 visual tokens per frame),
  direct-answer prompt (no reasoning trace), temperature 0.1, 4 passes with
  seeds 1–4. Cells are mean ± std over passes; n is the number of questions.
- **Budget** — "N frames total" is split evenly across a question's views
  (8 / 16 / 24 frames per view at K = 4 for the 32 / 64 / 96 legs), and
  `video_tokens` = 256 × N on every row, so token parity holds across arms
  within a budget.

## 2. Lower bounds — what a text-only guesser scores

| Bound | Overall | C2 | C4 | Status |
|---|---:|---:|---:|---|
| chance (uniform over the four options) | 25.00 | 25.00 | 25.00 | computed |
| pool modal letter (always the most common gold letter) | 36.21 | | | computed |
| task-conditioned modal letter, leave-one-out | **54.70** | **48.03** | **59.47** | computed — the floor arms are held to |
| blind arm (the full prompt with zero pixels — the model's own text prior) | 36.94 ± 1.49 | 45.25 ± 1.40 | 30.89 ± 1.82 | PARTIAL — 513 of 1,033 questions complete (job hit its time limit; resumed); reasoning-on prompt, T = 0.7, see section 4 |

The leave-one-out floor answers each question with the most common gold letter
among the *other* questions of its task type. It is the honest text-only bar:
the in-sample version overfits (it has seen the question's own letter), and the
pool-wide modal letter ignores a per-task prior a model can plausibly pick up.
The blind arm will show whether this model actually exploits that prior — below
54.70 it does not; above it the question text carries more than the letter
prior.

## 3. Measured all-view arms — the systems being bounded

Every media-valid leg on the pool, split by class. Selection arms keep the
prompt byte-identical to the sequential baseline and differ only in which
frames they pass.

| Arm | Budget | C2 (n=431) | C4 (n=602) | Overall |
|---|---|---|---|---:|
| cvbench_native (sequential, all views) | 32 total | 35.90 ± 2.15 | 47.88 ± 1.37 | 42.88 |
| cvbench_native | 64 total | 37.99 ± 2.05 | 50.33 ± 1.92 | 45.18 |
| cvbench_native | 96 total | 35.32 ± 2.22 | 49.96 ± 1.79 | 43.85 |
| segment_select_siglip | 32 total | 34.92 ± 1.65 | 50.79 ± 1.45 | 44.17 |
| segment_select_siglip | 64 total | 40.66 ± 1.61 | 52.82 ± 1.69 | 47.75 |
| segment_select_siglip | 96 total | **44.72 ± 0.41** | 52.24 ± 1.89 | **49.10** |
| segment_select_viclip_opt (auto K) | 32 total | 36.02 ± 1.94 | 48.96 ± 0.87 | 43.56 |
| segment_select_viclip_opt (auto K) | 64 total | 40.72 ± 0.89 | **53.16 ± 1.59** | 47.97 |
| segment_select_viclip_opt (auto K) | 96 total | 43.16 ± 1.24 | 51.99 ± 1.92 | 48.31 |

What the table says:

- **No arm clears the floor** — overall (best 49.10 vs 54.70) or within either
  class (C2 best 44.72 vs 48.03; C4 best 53.16 vs 59.47).
- **Segment selection's gain is a C2 phenomenon.** Sequential → segment
  selection at 96 total moves C2 from 35.32 to 44.72 and C4 only from 49.96 to
  52.24. Picking the right window on one camera is where the pixels start to
  pay; the cross-camera questions barely move.
- **Sequential sampling is flat in frames**: 42.88 / 45.18 / 43.85 at
  32 / 64 / 96 total. More frames spread evenly over four views do not help.

The same questions with up to 13 views delivered (the cap-13 pool; 8 frames per
view, so the total budget grows with K and is *not* matched to the ladder above;
montage cells 448 px, up to 6 tiles):

| Arm | C2 | C4 | Overall |
|---|---|---|---:|
| centralized (montage) | 43.10 ± 0.50 | 50.12 ± 1.55 | 47.19 |
| cvbench_native | 40.95 ± 1.64 | 47.76 ± 1.11 | 44.92 |
| per_stream (per-view descriptions, text-only aggregation) | 40.89 ± 1.00 | 37.75 ± 0.99 | 39.06 |

per_stream's answering call never sees pixels — only per-view descriptions —
and its C4 collapse (37.75, far below the 59.47 class floor) says the
cross-camera evidence does not survive the description step.

*Same split on the other backend (Qwen2.5-VL-7B, cap-13, for context only):*
centralized 24.94 / 64.49 / 47.99, cvbench_native 21.17 / 55.81 / 41.36,
per_stream 17.98 / 29.94 / 24.95 (C2 / C4 / overall). The C4 cell of
64.49 ± 1.18 is the only cell on either backend above its class floor, and it
sits next to a C2 at chance; it carries two confounds — the montage arm's prompt
text is arm-exclusive, and the direct-answer template suppresses refusals
differently across arms — so it needs a matched-prompt rerun before it is read
as a fusion effect.

## 4. Upper bound — the optimal-view oracle

Run `single_view1..4` over the pool — only view *i* is attached, the prompt
scaffold and the view's marker are byte-identical to the sequential arm, nothing
is renumbered — 4 passes each, and take each question's best view. Two matched
comparisons are possible:

- **per-view parity**: the single view at *n* frames against the sequential arm
  at 4*n* total (n = 8 ↔ the 32-total leg at 42.88; n = 16 ↔ the 64-total leg
  at 45.18). This isolates *which view* from *how many frames per view*.
- **total-budget parity** (the equal-budget rule): the single view at *N*
  frames against the sequential arm at *N* total — the same pixels, spent on
  one view. No 8-total sequential leg exists, so this row is open.

Estimators, per class, because the naive one is biased upward:

| Estimator | Definition | Bias |
|---|---|---|
| raw best-of-K | mean over questions of the max over views of the view's pass-mean score | inflated — a max over K noisy estimates |
| luck-best-of-K null | for each question, the expected max of K draws with replacement from its own per-view scores: Σ_i x_(i) · [(i/K)^K − ((i−1)/K)^K] over the sorted scores; the headroom is raw − luck | the null the raw number must beat |
| split-half cross-validation | choose each question's best view on passes 1–2, score that view on passes 3–4, and the mirror; average | unbiased selection under the standard 4-pass protocol; an 8-pass variant tightens it |
| worst / random view | min and mean over views | the rest of the curve |

Completeness gate: a question is complete when every *delivered* view
(`video_i` present) has all its passes — not when the release's camera count is
covered. 955 of the 1,033 records are capped, so gating on the original rig
size would silently drop them.

### 4.1 Result (single view at 8 frames; 1,033 complete questions, none dropped)

`evaluation/single_view_oracle.py` over the `mp4sv8` leg
(`multicam_results/legs/crossview_meva1033_subset_internvl_mp4sv8.jsonl.gz`,
15,916 rows; 955 four-view, 75 two-view, 3 three-view questions):

| Estimator | Overall (n=1033) | C2 (n=431) | C4 (n=602) |
|---|---:|---:|---:|
| text-only floor (task-LOO) | 54.70 | 48.03 | 59.47 |
| sequential, all views, 8 frames/view (per-view parity; reasoning off, T = 0.1) | 42.88 ± 0.68 | 35.90 ± 2.15 | 47.88 ± 1.37 |
| single view 1 alone | 37.73 | 38.17 | 37.42 |
| single view 2 alone | 35.67 | 32.95 | 37.62 |
| single view 3 alone (n=958) | 31.73 | 30.96 | 32.36 |
| single view 4 alone (n=955) | 29.58 | 29.29 | 29.81 |
| worst view | 13.48 | 10.96 | 15.28 |
| random view (mean over views) | 34.19 | 33.00 | 35.04 |
| luck-best-of-K null | 50.90 | 51.04 | 50.79 |
| raw best-of-K (inflated) | 57.65 | 58.12 | 57.31 |
| headroom: raw − luck | 6.75 | 7.08 | 6.52 |
| **optimal view, split-half CV** | **40.64** | **39.30** | **41.59** |

What the table says:

- **The raw oracle is mostly luck.** Best-of-4 over noisy pass means reads
  57.65, above the floor — but a guesser drawing four times from each
  question's own per-view scores expects 50.90, so only 6.75 points of the
  apparent margin are attributable to any view being better than another.
- **The honest oracle is under the floor and under the all-view arm.** Selecting
  the best view on two passes and scoring it on the other two gives 40.64
  overall — 14 points below the text-only floor (54.70) and 2 points below the
  sequential arm that sees all four views at the same 8 frames per view
  (42.88). Even the *right* single view does not beat the all-view baseline.
- **Views are not interchangeable.** View 1 alone scores 37.73 and view 4 alone
  29.58; the worst view per question averages 13.48. The ordering is the
  release's camera order, not a chosen one — the drop from view 1 to view 4 is
  as much a property of which camera saw the event as of the model.
- **Both classes behave the same way.** C2 (one camera, one window — the class
  where a single view *should* suffice) reaches only 39.30 with the optimal
  view against its 48.03 floor; C4 41.59 against 59.47. The class where the
  taxonomy predicts single-view sufficiency is not the class where single
  views work.

**Protocol caveat.** The single-view and blind legs were launched with the
harness defaults — the reasoning-trace prompt at temperature 0.7 — while every
sighted leg in section 3 ran the direct-answer prompt at 0.1 (the oracle tool
prints each source's protocol and flags the mismatch). The oracle-versus-floor
reading does not depend on this: the floor is protocol-free. The
oracle-versus-sequential row does, and is a cross-protocol comparison until a
`single_view1..4` sweep at the ladder protocol (`REASONING=0`, T = 0.1) is run
(~16k rows). The reasoning-on template was the earlier all-view finding's
protocol on the cap-13 pool before the ladder moved to direct answers, so
neither number is wrong; they are not matched.

## 5. Where the bounds stand

| | Overall | C2 | C4 |
|---|---:|---:|---:|
| text-only floor (LOO) | 54.70 | 48.03 | 59.47 |
| best all-view arm | 49.10 | 44.72 | 53.16 |
| gap to floor | −5.60 | −3.31 | −6.31 |
| blind arm (partial, 513 q; reasoning on, T = 0.7) | 36.94 | 45.25 | 30.89 |
| optimal-view oracle (split-half CV; reasoning on, T = 0.7) | 40.64 | 39.30 | 41.59 |
| raw best-of-4 / its luck null | 57.65 / 50.90 | 58.12 / 51.04 | 57.31 / 50.79 |

The lower bound is binding: on this pool and model, no sighted configuration
beats a guesser that knows the per-task answer-letter prior. The blind arm
(36.94 on the half of the pool that has run) sits 18 points *below* that
guesser, so the model does not exploit the letter prior the floor assumes — the
sighted arms' shortfall against the floor is not the model reading the letters
either; it is pixels that add ~6–12 points over the model's own text prior and
still land short of a statistical guesser.

The upper bound answers the question section 4 was written to ask: the deficit
is a **perception problem, not a view-selection problem**. An oracle that always
picks the best of four views, cross-validated, scores 40.64 — under the floor,
and under the all-view arm at the same frames per view. There is no large
selection margin waiting to be captured by a better view picker on this pool;
the 6.75-point raw-minus-luck headroom is the whole of it, and the honest
estimate spends it. What moves the number is what the segment-selection ladder
in section 3 already showed: which *window* is shown (C2 from 35 to 45), not
which *camera*.

Open items: the blind leg's second half (resumed under the same tag; the row
above updates in place), and the matched-protocol single-view sweep that would
turn the oracle-versus-sequential row into a like-for-like comparison.
