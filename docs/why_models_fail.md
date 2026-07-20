<!-- Ported from Wavy-Hec/CVBench analysis/why_models_fail.md @ 480d6f41cddddc7efea9a09b79134811740ba17a -->
# Why Vision-Language Models Fail at Multi-Camera CrossView Reasoning

*A plain-English walkthrough, grounded in verified failure traces from Qwen3-VL-8B-Thinking and InternVL3 on the 1,033-question CrossView-MEVA benchmark.*

---

## 1. The one-sentence answer

**The models fail because the input they are given — 8 wide-spaced frames per surveillance clip — almost never lands on the brief moment a person actually crosses a doorway or passes another person, so the frames look frozen and empty, and the model is forced to answer multi-camera questions about motion, distance, and timing with no visual evidence at all — falling back on commonsense language priors instead of seeing.**

This is not a "the model reasons poorly" story. It is a "the model was effectively shown a still photo and asked a question about a movie" story.

---

## 2. The chain of failure

Walk the pipeline one link at a time. Every link is where it can break.

**Link 1 — Sampling.** Each CrossView question stacks 2–4 clips from different MEVA surveillance cameras (originally up to 12–14 cameras, sub-sampled). The harness samples exactly **8 frames per clip** (16 frames for 2 clips, 24 for 3, 32 for 4), at four widely spaced timestamps — roughly **21s, 107s, 193s, 278s** of footage that runs ~280 seconds long.

**Link 2 — What the model can see.** MEVA scenes are sparse: most of a 5-minute clip is an empty doorway, stairwell, or parking lot, and the queried person passes through for only a few seconds. Eight frames spaced ~85 seconds apart almost always fall *between* the events. So every sampled frame of a given camera looks essentially identical, and the model perceives the clip as a still image. It says so out loud, constantly: the word "static" appears in **688 of 1,033 traces**, "no people / no movement" in **774**, "frames are identical" in hundreds more.

**Link 3 — Grounding (or mis-grounding).** Now the model tries to find the people the question describes (e.g. "the person in black pants with indigo shoes"). They aren't in the frames — they were never sampled, or they are tiny and distant in an aerial view. The model has no calibrated "I can't tell" option, so it does one of three things: (a) declares the scene empty and reasons as if blind; (b) **hallucinates** concrete scene contents ("two silver cars," "a basketball court") to have something to reason over; or (c) tries to recover order by treating the arbitrary clip numbering ("Video 1, 2, 3...") as a timeline.

**Link 4 — Reasoning.** Having lost the visual signal, the model fills the vacuum with a language prior — real-world plausibility ("you open a door before you enter," "two people on a path usually pass each other"). Crucially, **this is exactly how the MEVA questions were authored**, so a blind text-only model lands on the right answer surprisingly often. But the prior also misleads systematically when the true chronology contradicts naive commonsense.

**Link 5 — The wrong answer.** The output is then one of: a confident-but-guessed letter from the prior; an escape to "D: Cannot be determined"; or — on the longest questions — no answer at all, because the deliberation ran out of tokens.

The causal arrow runs **left to right**: sparse sampling → empty-looking frames → failed grounding → prior-driven reasoning → wrong answer. The reasoning is not the root cause; the missing input is.

---

## 3. The failure modes

### 3a. Event-Ordering — putting 3–4 described events in chronological order

*Qwen3-VL gets 204/297 wrong (68.7%); InternVL3 is worse at 222/297 (74.7%).*

**How it happens — static-frame abandonment.** The model reads the frames, sees they are all the same, explicitly concludes they don't show the described people, and then orders the events from a narrative prior instead.

**Root cause.** The human transit events occur *between* the few sampled frames, so the frames carry no ordering signal.

> *(id 68, gold D, pred A)* "the frames from 21.4s to 278.5s are all the same: a door with a window... No people are visible in any of those frames. So Video 1 is a static shot of a door and stairs without any people."

**How it happens — the narrative / commonsense prior.** With no visual evidence, the model ranks events by plausibility ("opening the door enables entry, so it's first") and presents the guess confidently. The blind (zero-video) run reproduces the *same* reasoning:

> *(id 90, gold C, pred A)* "The key insight is that III must be first, as it enables the other actions."

> *(id 22, BLIND / no-video run, gold C, pred D)* "After analyzing the descriptions and logical flow of actions, the most reasonable sequence is: II → I → III. This aligns with the real-world sequence of someone entering a facility, texting while inside, and then the facility door being closed."

Cross-model confirmation — InternVL3 makes the identical error on id 90 without ever referencing a frame:

> *(id 90, InternVL3, gold C, pred A)* "the correct order should be determined by the actions of the individuals... The sequence starts with someone opening a door (III)... This order logically follows the flow of people moving in and out of the facility."

**How it happens — token-budget truncation.** On 4-event questions the model enumerates permutations, repeatedly writes "both are possible," and runs out of tokens (truncated outputs average ~33K characters, max 35K) before ever emitting a closed `<answer>`. This produces **39 fully empty predictions**, all scored wrong.

> *(id 34, gold C, pred empty)* "the actual video content here is the same across all timestamps—no movement or people. Wait, but the problem mentions doorway-related moments where people enter or exit..."

> *(id 34, gold C, pred empty)* output literally terminates mid-sentence: "1. Person IV opens door (IV). 2. Person IV enters" — no closing tag, empty prediction.

### 3b. Spatial — how two people's distance evolves across synchronized cameras

*Qwen3-VL scores 151/431 = 35%; InternVL3 119/431 = 28%. Blind text-only on the 20-Q subset = 7/20 vs 9/20 with video — video adds almost nothing.*

**How it happens — the motion signal is gone.** The question asks whether two people approach, stay moderate, stay far, or cross paths. That requires a trajectory. Sparse sampling of slow pedestrians yields near-identical frames, so the model perceives no displacement.

> *(id 621, gold B, pred A)* "Video 1 has the same image at 21.4s, 107.1s, 192.8s, 278.5s. So maybe the video is static?... since the frames listed are identical, perhaps the video is a still image."

**Root cause.** 8-frame sampling over ~280 seconds destroys the relative-motion cue the task is built on.

**How it happens — prior-driven guessing.** Once perception fails, the model invokes "typical" pedestrian behavior. This biases it toward the two salient options and inflates "D, cross paths" far above its true base rate (D is predicted 81 times among wrong cases vs only 29 D-golds total; the model over-predicts A and collapses B).

> *(id 310, gold A, pred D)* "In many standard test questions, when two people are walking towards each other on a path and pass each other, the answer is D: walk past each other."

> *(id 621, gold B, pred A)* "both wearing indigo, maybe they're part of a group that's walking together, so they stay near each other."

**How it happens — the single-video shortcut.** The model picks one clip as "the relevant one" and discards the other cameras, because it has no way to track the same person across views.

> *(id 621, gold B, pred A)* "Video 1 is the first one with the aerial view of a campus with people walking. Videos 2, 3, and 4 are different scenes, like a gas station, a garage, and a transit station, so probably not relevant here. So the focus is on Video 1."

**How it happens — non-detection laundered into a distance.** When the targets are too small/brief to detect, the model declares the scene empty and converts "I can't see them" into "they're far apart."

> *(id 297, gold A, pred C)* "none of the videos show any people. But the question is about two people approaching each other. This is confusing... They remain far apart (more than 15 meters) because they aren't there."

**How it happens — D conflates a motion with a distance.** Option D ("walk past each other") is a trajectory, while A/B/C are distances. With no observed motion the model deadlocks and defaults to the richer-sounding D.

> *(id 310, gold A, pred D)* "If they walk past each other (D), they would get close (a few meters) at the crossing point. But the question is asking how close they get... A would be correct, but D is about the action."

### 3c. Temporal — "which event happened first?" (the Qwen-vs-InternVL split)

*Every Temporal question offers A/B (a real event), C ("simultaneously"), and D ("cannot be determined"). **Gold is ALWAYS A or B.** Qwen scores 75/305 = 25%; InternVL3 scores 157/305 = 51% — the only task where InternVL doubles Qwen.*

**The split is about commitment, not perception.** Qwen over-deliberates, talks itself into "the frames are static, so I can't tell," and escapes to **D (142/305) or C (8/305) — all guaranteed wrong** (gold is never C/D). InternVL commits to A/B in 292/305 cases. The decisive proof: **when both models actually commit to A/B, their accuracy is nearly identical** (Qwen 49% vs InternVL 54%). So Qwen's entire deficit is the ~150 questions it throws away on escape options; 77 of its 108 losses to InternVL are exactly those C/D escapes.

> *(id 743, gold A, pred D)* "the videos don't show these events... Therefore, there's no information to determine which happened first. The answer should be D: Cannot be determined... none of the frames show people... Therefore, the correct answer is D."

This escape is a **baked-in disposition, not perception-driven**: the blind text-only Qwen *still* picks D 10/20 = 50% of the time on Temporal — the same rate as with video. The "frames are static" line is post-hoc rationalization.

**Secondary Qwen errors feed the escapes:**

- *Mistaking sampling offsets for event times.* The prompt prints per-frame timestamps (21.4s, 107.1s...), identical across all cameras. Qwen treats them as wall-clock event times.

  > *(id 774, gold B, pred A)* "If Person A happened first (at 21.4 seconds), and Person B happened later (after 21.4 seconds), then Person A happened first. Therefore, the answer should be A."

- *Mistaking synchronized cameras for simultaneous events.* Because the clips are different views of one recording, Qwen reasons the two events must co-occur → C.

  > *(id 804, gold B, pred C)* "all videos are part of the same recording... then the events would be at the same time... Both are happening at the same time. So they are simultaneous. Therefore, answer C."

**InternVL's win is partly a lucky prior, not robust grounding.** It emits 1–3 terse sentences and almost always picks the first-listed event (answer A 222/305 times). It scores 73% when gold is A but only 26% when gold is B — and the answer key is A-skewed (165 A vs 140 B). It reads "Video 1 ends, then Video 2" as chronological order:

> *(id 789, InternVL3, gold B, pred A)* "the gold-haired person closing a trunk happens before the scene with the black-haired person talking to someone. The video transitions from the first scene to the second, indicating a chronological order."

So InternVL commits via a clip-position shortcut that happens to align with the answer key; it is *not* demonstrably perceiving order better.

### 3d. The input-level root causes (cutting across all three tasks)

These are the *upstream* causes the three tasks share:

- **8 frames render the event invisible** → the model declares the clip static and reasons as if blind. Dominant: **626 of 714 wrong answers** are classified "reasoning over emptied frames."
- **Hallucinated scene contents** fill the gap. The model invents specifics it then reasons over:
  > *(id 297, gold A, pred C)* "Video 1: Shows an outdoor parking lot with cars, a van moving... Video 2: An indoor basketball court. Empty..." — concrete contents for frames it just called empty.
- **No cross-view correspondence** → the model can't map which camera shows which event, and falls back on clip numbering as a timeline.
  > *(id 62, gold C, pred A)* "maybe each video is a different camera angle or location... But how do we know which video corresponds to which event?"
  > *(id 86, gold D, pred B)* "the four videos are separate scenes... maybe the answer is based on which video has the event."
- **Accuracy falls as cameras rise**: 41% on 2-camera questions → 25% (12 cams) → 26% (14 cams). More views, harder correspondence, worse accuracy.
- **Token-budget truncation** on the longest (Event-Ordering) questions: the answer is cut off before it's emitted.

---

## 4. Why video can HURT (the blind > video flip)

On the paired 60-question subset, the **blind text-only model (36.7%) beats the same model with video (31.7%)**. Joining by question id: **15 questions flip blind-RIGHT → video-WRONG**, vs only 12 the other way. The net +3 exactly equals the accuracy gap — the flips *are* the inversion. They are dominated by Temporal (9) and Spatial (4).

The mechanism is the same in nearly every flip: the sparse frames hide the actor and look empty. **Blind**, the model never sees this and applies the commonsense prior — which matches how the questions were authored, so its prediction equals gold on all 15 flips. **With video**, the model trusts the (uninformative) frames, concludes "the people aren't there," and either collapses to D or misreads the printed timestamps as a timeline. Seeing nothing actively overrides a correct guess.

> *(id 297, Spatial, gold A — VIDEO pred C, BLIND pred A)*
> VIDEO: "So none of the videos show any people... Since all the videos are empty, the answer must be C. Because if they aren't there, they are more than 15 meters apart."
> BLIND: "if two people are approaching each other, they get close." → A (correct).

> *(id 779, Temporal, gold A — VIDEO pred D, BLIND pred A)*
> VIDEO: "there are no people visible in any of the frames. Both videos show static scenes... we can't determine which happened first. So the answer would be D."
> BLIND: "you can't enter without opening it. So A would be correct."

> *(id 759, Temporal, gold A — VIDEO pred B, BLIND pred A)*
> VIDEO: "Video 2 starts at 21.4 seconds, which is earlier than Video 1's 107.1 seconds... So exit event (B) happens first... Therefore, the answer is B." (misreads the sampling offset as the event timeline)

The video doesn't add signal here. It adds a misleading *absence of evidence* that the blind model never has to reconcile.

---

## 5. So what

- **These benchmarks expose perception, not reasoning.** The dominant failure is upstream: the input pipeline delivers almost no temporal or cross-view motion signal, and the model is then graded on questions that require it. The blind-beats-video result is the cleanest possible proof — a model that can't see *at all* does *better* than one fed uninformative frames.

- **"Thinking" can be a liability when there's nothing to think about.** Qwen's long chains of thought don't recover signal; they manufacture uncertainty (escaping to "cannot be determined") and burn the token budget (truncating to empty answers). InternVL's terse commitment scores higher on Temporal — but largely via an answer-position prior, not better sight.

- **Language priors are doing the heavy lifting on both sides.** When the pixels go dark, both models answer from commonsense narratives about doors and pedestrians. That these priors *correlate* with the answer key is a benchmark-construction artifact, not evidence of cross-view understanding.

- **Multi-camera reasoning needs more than more cameras.** Current VLMs have no mechanism to register the same person or event across synchronized views, and no true event-time grounding (they confuse frame-sampling offsets and clip order with real chronology). Adding cameras makes accuracy *worse*, not better.

- **The fixes are architectural, not prompt-level.** Denser, event-aware frame sampling; an explicit "insufficient evidence" calibration instead of a forced multiple-choice guess; cross-view correspondence; and real timestamp grounding. Until then, multi-camera CrossView reasoning is mostly being answered by a language model with its eyes closed.
