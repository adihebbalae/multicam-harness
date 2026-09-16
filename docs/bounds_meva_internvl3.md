<!-- Numbers from the reference implementation's registered legs (Wavy-Hec/CVBench
     bench/; this repo's arms are prompt-equivalence-gated against it), media-valid
     rows only (MEVA decoded through the remuxed containers, media_remap stamped), split
     by evaluation/evidence_class_split.py against data/subsets/meva_evidence_labels.json.
     Written 2026-08-31.
     Revised 2026-09-03: blind leg complete on the full pool; per-letter
     recall added; budget, prompt and parity claims qualified. -->
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
- **Budget** — "N frames total" is split evenly across a question's views.
  That holds exactly for the 955 four-view questions (8 / 16 / 24 frames per
  view at K = 4 for the 32 / 64 / 96 legs, `video_tokens` = 256 × N). The
  32-total sequential leg instead gives the 75 two-view questions 16 frames
  per view and the 3 three-view questions 11/11/10, and the segment-selection
  legs give two-view questions a fixed 64 frames at the 96-total budget (300
  of 4,132 rows) — so token parity across arms within a budget is exact on
  the four-view majority and approximate elsewhere.

## 2. Lower bounds — what a text-only guesser scores

| Bound | Overall | C2 | C4 | Status |
|---|---:|---:|---:|---|
| chance (uniform over the four options) | 25.00 | 25.00 | 25.00 | computed |
| pool modal letter (always the most common gold letter) | 36.21 | | | computed |
| task-conditioned modal letter, leave-one-out | **54.70** | **48.03** | **59.47** | computed — the floor arms are held to |
| blind arm (the full prompt with zero pixels — the model's own text prior) | 34.75 ± 0.89 | 45.36 ± 1.04 | 27.16 ± 1.05 | complete — 4 passes; reasoning-on prompt, T = 0.7, see section 4 |

The leave-one-out floor answers each question with the most common gold letter
among the *other* questions of its task type. It is the honest text-only bar:
the in-sample version overfits (it has seen the question's own letter), and the
pool-wide modal letter ignores a per-task prior a model can plausibly pick up.
At 34.75, the blind arm sits about 20 points under the 54.70 floor, so this
model does not exploit the per-task letter prior. By class it is near the C2
floor (45.36 vs 48.03 — spatial questions have gold A on 48% of items, and
the model's text prior leans A) and near chance on C4 (27.16).

## 3. Measured all-view arms — the systems being bounded

Every media-valid leg on the pool, split by class. The question scaffold is
identical across arms; segment selection prepends its own arm preamble and a
per-clip banner, so prompts match in the question text but are not identical
end-to-end.

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

**The C2 movement is letter-prior collapse, not perception.** Per-letter
recall by leg:

| leg (method@budget) | C2 acc | C2 balanced | C2 recall A/B/C/D | C4 acc | C4 balanced |
|---|---:|---:|---|---:|---:|
| mp4fs32 (cvbench_native@32) | 35.90 | 20.54 | 73.55/0.00/0.00/8.62 | 47.88 | 48.68 |
| mp4fs64 (cvbench_native@64) | 37.99 | 24.22 | 76.21/0.00/0.00/20.69 | 50.33 | 51.17 |
| mp4fs96 (cvbench_native@96) | 35.32 | 22.46 | 70.89/0.00/0.00/18.97 | 49.96 | 50.59 |
| mp4sg32 (segment_select_siglip@32) | 34.92 | 20.81 | 67.27/2.68/8.13/5.17 | 50.79 | 51.99 |
| mp4sg64 (segment_select_siglip@64) | 40.66 | 22.77 | 82.37/1.12/2.41/5.17 | 52.82 | 53.71 |
| mp4sg96 (segment_select_siglip@96) | 44.72 | 25.25 | 91.06/0.67/1.51/7.76 | 52.24 | 53.42 |
| mp4sgva32 (segment_select_viclip_opt@32) | 36.02 | 21.18 | 69.69/2.90/7.83/4.31 | 48.96 | 49.87 |
| mp4sgva64 (segment_select_viclip_opt@64) | 40.72 | 23.73 | 82.25/1.12/1.20/10.34 | 53.16 | 54.16 |
| mp4sgva96 (segment_select_viclip_opt@96) | 43.16 | 24.90 | 88.16/0.22/0.00/11.21 | 51.99 | 52.98 |

On every native leg the model never answers B and essentially never answers
C; gold is A on 207 of 431 C2 questions (48%). The segment-selection gain is
A-recall rising (native@96 ≈71% → segment@96 ≈91%) while B/C/D recall stays
at the floor; accuracy on the 224 non-A-gold C2 questions falls (2.46% →
1.90%). Balanced accuracy on C2 is 20–25% on all nine legs, at or below the
25% chance line — there is no perception signal on C2 in any arm. C4 is
different: predictions spread over all four letters and balanced ≈ raw, so
the C4 numbers are real.

- **Sequential sampling is flat in frames**: 42.88 / 45.18 / 43.85 at
  32 / 64 / 96 total. More frames spread evenly over four views do not help.

The same questions with up to 13 views delivered (the cap-13 pool; 8 frames per
view, so the total budget grows with K and is *not* matched to the ladder above;
montage cells 448 px, up to 6 tiles). Token parity does not hold here: 0 of
1,033 questions have equal video_tokens between centralized and native at
pass 1, and at K=13 (464 of 1,033 questions, 45% of the pool) centralized
runs at 0.38× native's video_tokens (10,240 vs 26,624):

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
differently across arms, on either backend — so it needs a matched-prompt
rerun before it is read as a fusion effect.

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

**602 of the 1,033 questions are C4 by construction** — evidence on at least
two cameras — so a single view cannot in principle hold their evidence; the
pool-wide oracle below is a ceiling over a policy class that cannot answer
58% of the pool.

Estimators, per class, because the naive one is biased upward:

| Estimator | Definition | Bias |
|---|---|---|
| raw best-of-K | mean over questions of the max over views of the view's pass-mean score | inflated — a max over K noisy estimates |
| luck-best-of-K null | for each question, the expected max of K draws with replacement from its own per-view scores: Σ_i x_(i) · [(i/K)^K − ((i−1)/K)^K] over the sorted scores; the headroom is raw − luck | the null the raw number must beat; raw ≥ luck holds by construction, so raw − luck is the noise-corrected margin, not a significance test |
| split-half cross-validation | choose each question's best view on passes 1–2, score that view on passes 3–4, and the mirror; when several views tie for best on the selection half (1,161 of the 2,066 question-halves, 56.20%), the held-out score is averaged over the tied views; average both halves | unbiased selection under the standard 4-pass protocol; an 8-pass variant tightens it |
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
| sequential, all views, 8 frames/view (per-view parity, exact on the 955 four-view questions; reasoning off, T = 0.1) | 42.88 ± 0.68 | 35.90 ± 2.15 | 47.88 ± 1.37 |
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

*The two-view questions in this leg get 16 frames per view and the
three-view questions 11/11/10, not 8 — the per-view parity label above is
exact only for the 955 four-view questions (section 1).*

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
  views work. The oracle reads *higher* on C4 (41.59) than on C2 (39.30) —
  consistent with the C2 letter-prior finding in section 3, not with genuine
  single-view perception.

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
| blind arm (reasoning on, T = 0.7) | 34.75 | 45.36 | 27.16 |
| optimal-view oracle (split-half CV; reasoning on, T = 0.7) | 40.64 | 39.30 | 41.59 |
| raw best-of-4 / its luck null | 57.65 / 50.90 | 58.12 / 51.04 | 57.31 / 50.79 |

The lower bound is binding: on this pool and model, no sighted configuration
beats a guesser that knows the per-task answer-letter prior. Blind (34.75)
sits about 20 points below the floor; pixels add about 8 points at per-view
parity (42.88) and about 14 at best (49.10) overall. By class, though, the
sighted gain is entirely C4: on C2 the blind arm (45.36) matches or beats
every sighted C2 cell (best 44.72; native 35.90–37.99), while on C4
sequential adds about 21 points over blind (47.88 vs 27.16) and the best arm
about 26 (53.16 vs 27.16).

The upper bound answers the question section 4 was written to ask: the deficit
is a **perception problem, not a view-selection problem**. An oracle that always
picks the best of four views, cross-validated, scores 40.64 — under the floor,
and under the all-view arm at the same frames per view. There is no large
selection margin waiting to be captured by a better view picker on this pool;
the 6.75-point raw-minus-luck headroom is the whole of it, and the honest
estimate spends it. The C2 movement in section 3 is the letter prior, not a
window effect, so on this pool no arm has shown a perception gain on C2; the
only real sighted signal is on C4, and the oracle says that signal is not
recoverable by choosing a camera.

Open items: the matched-protocol single-view sweep that would turn the
oracle-versus-sequential row into a like-for-like comparison.
