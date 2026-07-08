<!-- Ported from Wavy-Hec/CVBench analysis/crossview_question_types.md @ 480d6f41cddddc7efea9a09b79134811740ba17a -->
# UT Austin CrossView (Multi-Camera VQA): question-type analysis

**Source:** `crossview-release-annotations/crossview-release/` (annotations only, 36 MB).
Videos are *not* included — they come from MEVA, Ego-Exo4D, AgiBotWorld and nuScenes and
must be downloaded separately, then mapped via `video_manifest.csv` (7,760 videos).
Stats below exclude the legacy file `meva/qa_best_camera_pre_regenerate.json`
(376 items superseded by `qa_best_camera.json`).

## 1. What this benchmark tests

Unlike CVBench (1–4 *related but separate* videos, association reasoning), every CrossView
question is grounded in **synchronized cameras observing the same scene at the same time**:
surveillance networks (MEVA), ego+exo rigs (Ego-Exo4D), robot head/hand cameras (AgiBotWorld),
and 6-camera surround view (nuScenes). This is true multi-camera understanding — the model
must fuse or select among simultaneous viewpoints, which is the VideoForest setting.

## 2. Question counts — source × type (6,354 usable QAs)

| source | camera | counting | event_ordering | spatial | spatio_temporal | summarization | temporal | total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| agibot |  |  | 250 |  |  | 250 | 250 | 750 |
| ego-exo4d | 500 |  | 250 |  |  | 250 | 250 | 1250 |
| meva | 495 | 289 | 297 | 431 |  | 288 | 305 | 2105 |
| nuscenes |  | 500 | 250 | 499 | 500 | 250 | 250 | 2249 |
| **total** | 995 | 789 | 1047 | 930 | 500 | 1038 | 1055 | 6354 |

Question types: **camera** = which camera best captures X (view selection); **temporal** =
which event happened first across cameras; **event_ordering** = chronological order of
multiple events; **spatial** = relative position / frame-of-reference across views;
**spatio_temporal** = combined; **counting** = open-ended count across cameras;
**summarization** = open-ended scene/episode summary across all views.

## 3. Number of cameras per question

| source | cams (min–max, median) | distribution |
|---|---:|---:|
| agibot | 1–3 (median 3) | 1cam×45, 3cam×705 |
| ego-exo4d | 4–7 (median 5) | 4cam×3, 5cam×801, 6cam×313, 7cam×133 |
| meva | 1–16 (median 7) | 1cam×6, 2cam×232, 3cam×7, 5cam×506, 7cam×467, 11cam×10, 12cam×84, 13cam×43, 14cam×717, 15cam×22, 16cam×11 |
| nuscenes | 6–6 (median 6) | 6cam×2249 |

**This is the accuracy-vs-#cameras axis the PI wants, and it is much wider than CVBench's
1–4:** MEVA alone spans 1–16 cameras *within the same question
types*, so per-#cameras accuracy curves can be computed without changing the task.
nuScenes is fixed at 6 (surround view), AgiBotWorld at 3 (head + both hands),
Ego-Exo4D at 5–7 (1 ego + 4–6 exo).

## 4. Answer formats

- **MCQ (A–D): 4,527 / 6,354** — directly scorable like CVBench.
- **Open-ended: 1,827** — all `summarization` (1,038)
  and all `counting` (789).
  Counting is numeric-exact-match scorable; summarization needs an LLM judge — neither fits
  the MCQ harness unchanged.

## 5. Example questions (verbatim)

**meva / temporal** — 7 cameras (surveillance, cross-camera)
> Between these two moments, which happened first: the person in a blue top and blue pants closing a trunk, or the black-haired person dressed in black with black shoes, wearing a hat and carrying a bag, sitting down?
> A. The person, wearing a blue top and blue pants, closing a trunk occurred first / B. The person with black hair, wearing a black top and black pants, black shoes, with a hat, carrying a bag, sitting down occurred first / C. They occurred simultaneously / D. Cannot be determined
> **Answer: A**

**meva / camera** — 12 cameras (best-camera selection)
> Which camera shows the first appearance of a black-haired person wearing a dark plum top, a dark crimson skirt, and pink shoes as they enter the scene?
> A. Camera G424 / B. Camera G419 / C. Camera G423 / D. Camera G336
> **Answer: D**

**ego-exo4d / camera** — 6 cameras (ego↔exo view selection)
> This video shows the camera-wearer playing the piano with both hands, moving between higher and lower keys, playing chords, and performing trills. Which exocentric camera best captures the camera-wearer's actions throughout this scene?
> A. camera 2 / B. camera 5 / C. camera 1 / D. camera 3
> **Answer: C**

**agibot / event_ordering** — 3 cameras (robot manipulation)
> Identify the correct chronological order of the following robot actions observed in the episode:
> I. The robot coordinates both hands to pass the marker pen.
> II. The robot places the held ballpoint pen into the pen holder on the desk.
> III. The robot lifts its left arm to pick up the marker pen from the desk.
> A. II -> III -> I / B. I -> II -> III / C. II -> I -> III / D. III -> I -> II
> **Answer: A**

**nuscenes / spatial** — 6 cameras (driving, frame-of-reference)
> From the perspective of the barrier (consisting of concrete and plastic construction barricades, primarily in white and orange colors, arranged to redirect traffic with visible signs of wear and construction materials nearby), where is the other barrier (made up of orange and white plastic construction barricades, commonly used to cordon off work zones, connected and placed along the side of the road with a construction excavator visible in the background)?
> A. To the left of the barrier / B. In front of the barrier / C. Behind the barrier / D. To the right of the barrier
> **Answer: D**

**nuscenes / counting** — 6 cameras (open-ended numeric)
> Can you tell me the number of construction workers present in the entire video?
> **Answer: 10**

**meva / spatial** — 12 cameras (cross-camera geometry)
> In the footage, how close does the person in black pants with indigo shoes get to the person wearing a dark indigo top with dark gray pants and dark gray shoes?
> A. They approach and stay near each other (within a few meters) / B. They stay at a moderate distance (5-15 meters apart) / C. They remain far apart (more than 15 meters) / D. They walk past each other, swapping positions (cross paths)
> **Answer: A**

**ego-exo4d / summarization** — 5 cameras (open-ended free text)
> Provide a very comprehensive, well thought-out summary of the ego-actor's interactions across all views. Make it a couple sentences and around 500-1000 characters.
> **Answer: The camera-wearer performs bicycle drivetrain maintenance by lubricating and wiping the bike chain while turning the right pedal to cycle the chain through. They pick up and adjust a bottle of chain l …**

## 6. Data-quality flags

- **`question_format.md` is stale** (dated 2025-02-20): it claims agibot 934 / ego-exo4d 2,641 /
  nuscenes 1,258 questions, but the released JSONs contain 750 / 1,250 / 2,249. Treat the JSONs
  as ground truth; total 6,730 raw items across 20 files, 6,354 usable after
  dropping the legacy best-camera file.
- 3 nuscenes items double-encode the question as a JSON string with unescaped
  inner quotes (their top-level `answer` is also empty/null; the real answer sits inside the
  string — recovered automatically): nuscenes/spatial, nuscenes/spatio_temporal, nuscenes/temporal.
- 0 item(s) with an unrecoverable missing/null answer (after the recovery above).
- `meva/qa_best_camera_pre_regenerate.json` is a pre-regeneration legacy duplicate — exclude it.
- A few AgiBotWorld/MEVA items reference only 1 camera (51 items with 1 video) —
  fine as a baseline bucket for the #cameras curve.

## 7. CVBench vs CrossView — comparison & recommendation

| | CVBench (running now) | CrossView (UT Austin) |
|---|---|---|
| QAs | 1,000 | 6,354 usable |
| Cameras/question | 1–4 *separate* videos | 1–16 *synchronized* cameras |
| Nature of task | cross-video association (link entities/events across unrelated clips) | true multicam fusion (same scene, multiple simultaneous views) |
| Domains | web video mix | surveillance, ego-exo activities, robot manipulation, autonomous driving |
| Answer format | MCQ + yes/no (all scorable) | 4,527 MCQ + 1,827 open-ended |
| Videos | one HF zip, downloaded ✓ | not included; 7,760 files from 4 licensed datasets |

**Recommendation.** Keep CVBench as the *now* benchmark (pipeline already running; failure
traces + accuracy-vs-#videos land this week). CrossView is the better instrument for the
PI's actual question — multicam scaling — because the #cameras axis reaches 16 and the
cameras are genuinely synchronized. For inference on CrossView, start with **MEVA only**:
it is the widest #cameras axis (1–16), publicly downloadable (mevadata.org, AWS open bucket,
no license gate like Ego-Exo4D/AgiBotWorld), and its MCQ types (temporal, spatial,
event_ordering, camera) reuse the CVBench scoring path as-is. nuScenes-mini covers too few
scenes to be useful; full nuScenes/Ego-Exo4D/AgiBotWorld need registration + large downloads.

---
*Generated by the fork's `analysis/crossview_question_types.py`.*
