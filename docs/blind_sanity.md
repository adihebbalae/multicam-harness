<!-- Ported from Wavy-Hec/CVBench analysis/blind_sanity.md @ 480d6f41cddddc7efea9a09b79134811740ba17a -->
# Blind (no-video) run sanity check — qwen3vl_blind

Read-only audit of the no-video baseline
(the fork's stored eval JSON `eval_subset_qwen3vl_novideo.json`, slurm job
56689, launched by the fork's `analysis/run_eval_novideo.sbatch`). Produced by
the fork's `analysis/inspect_blind.py` (all embedded assertions passed; raw data in
the fork's `analysis/inspect_blind_data.json`).

**Verdict: the blind run is clean.**

- Zero image/video tokens in the model input (checked against the tokenizer's
  own added-token inventory, all 45 prompts).
- Zero metadata leakage: no filenames, ytid stems, `_seg` markers, paths, fps,
  frame counts, frame/timestamp markers in any rendered prompt.
- Recounted accuracy = **18/45 = 40.0%**, confirmed three independent ways.
- One caveat that is inherent to the benchmark, not a bug: the *questions
  themselves* reference videos ("Which video…", "Video 1/2" in options), and
  the fixed scaffold sentence says "based on all the listed videos". The blind
  model knows videos *exist*; it just gets no information about their content.

## 1. How the blind input is built

Same script as the with-video run, one flag: `--no_video`
(`run_eval_novideo.sbatch:23-27`). In `build_messages()`
(`Video-R1/src/eval_thinking.py:117-122`) the flag skips the entire
video-interleaving loop:

```python
content = []
if not no_video:                                   # skipped when --no_video
    for k, vp in enumerate(video_paths(rec, video_root), 1):
        content.append({"type": "text", "text": f"Video {k}:"})
        content.append({"type": "video", "video": vp, "nframes": nframes})
content.append({"type": "text", "text": full_prompt})
```

So the message content is a **single text item** (asserted for all 45
questions): no `"Video k:"` marker texts, no video entries, hence
`video_inputs=None` downstream and the processor receives `videos=None` —
it emits no vision tokens. The prompt text itself is byte-identical to the
with-video run (asserted: the rendered blind prompt equals scaffold + question
+ options + scaffold exactly).

## 2. Exact inputs sent to the model (5 sample questions)

The results JSON does not store the model input, so these are deterministic
CPU re-renders: `build_messages(rec, ..., no_video=True)` +
`processor.apply_chat_template(..., add_generation_prompt=True)` with the same
pinned env as the run (transformers 5.2.0, offline HF cache). Same 5 ids as
`input_pipeline.md` (deterministic rule): 0, 4, 3, 17, 19. Each prompt below is
the **complete** input string — note the chat template pre-opens `<think>` for
the Thinking checkpoint.

### id 0 (2-video question, Cross-video Scene Recognition)

```
<|im_start|>user
Select the best answer to the following yes-no question based on all the listed videos.
Did both videos feature a team successfully scoring a decisive goal or point during the match's closing moments?
Yes.
No.
Please think about this question as if you were a human pondering deeply. Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions. It's encouraged to include self-reflection or verification in the reasoning process. Provide your detailed reasoning between the <think> and </think> tags, and then give your final answer between the <answer> and </answer> tags.
Provide only the single word (Yes or No) within the <answer> </answer> tags.<|im_end|>
<|im_start|>assistant
<think>
```

### id 4 (3-video, Joint-video Counting)

```
<|im_start|>user
Select the best answer to the following multiple-choice question based on all the listed videos.
How many key stages of the parcel lifecycle are depicted across all three videos?
A. 2 stages.
B. 3 stages.
C. 4 stages.
D. 5 stages.
Please think about this question as if you were a human pondering deeply. [... identical think instruction ...]
Provide only the single option letter (A, B, C, or D) within the <answer> </answer> tags.<|im_end|>
<|im_start|>assistant
<think>
```

### id 3 (4-video, Cross-video Event Retrieval)

```
<|im_start|>user
Select the best answer to the following multiple-choice question based on all the listed videos.
Which video features players wearing shirts labeled with country names?
A. VideoID 1
B. VideoID 2
C. VideoID 3
D. VideoID 4
Please think about this question as if you were a human pondering deeply. [... identical think instruction ...]
Provide only the single option letter (A, B, C, or D) within the <answer> </answer> tags.<|im_end|>
<|im_start|>assistant
<think>
```

### id 17 (3-video, all `_seg` files, Multi-view Scene Understanding)

```
<|im_start|>user
Select the best answer to the following multiple-choice question based on all the listed videos.
What notable connections is present across the videos?
A. All depict construction work.
B. All involve handling historical artifacts.
C. The scientist discovers a new species.
D. The historical artifacts in the videos are shown to be from different periods.
Please think about this question as if you were a human pondering deeply. [... identical think instruction ...]
Provide only the single option letter (A, B, C, or D) within the <answer> </answer> tags.<|im_end|>
<|im_start|>assistant
<think>
```

### id 19 (4-video, `_seg` files, Cross-video Counterfactual Reasoning)

```
<|im_start|>user
Select the best answer to the following multiple-choice question based on all the listed videos.
If the construction activities had not involved a helicopter or a staged vertical vehicle climb, what would have been least likely to happen?
A. Connection to cinematic production.
B. Stacking of outdoor props.
C. Completion of construction site.
D. Utilization of scaffolding structures.
Please think about this question as if you were a human pondering deeply. [... identical think instruction ...]
Provide only the single option letter (A, B, C, or D) within the <answer> </answer> tags.<|im_end|>
<|im_start|>assistant
<think>
```

(The five full strings, untruncated, are in the fork's `analysis/inspect_blind_data.json`
under `sample_prompts` — the `[...]` ellipses above elide only the
think-instruction sentence shown in full for id 0, which is identical in all
prompts.)

**Vision-token check**: candidate tokens were taken from the tokenizer's own
added-token vocabulary matching `vision|image|video|img|box|quad` (9 tokens,
including `<|vision_start|>`, `<|vision_end|>`, `<|vision_pad|>`,
`<|image_pad|>`, `<|video_pad|>`); the tokenized rendering of every one of the
45 blind prompts contains none of them, and none of the literal strings appear
either. **Zero image/video tokens confirmed.**

## 3. Leakage check

Scanned all 45 rendered blind prompts for: `.mp4` filenames, the 101 ytid stems
from the fork's `analysis/subset_videos.txt`, `_seg` markers, path fragments
(`CVBench/`, `Evaluation/`, `r1-v/`), `fps`/`nframes`/"N frames" counts,
frame/marker text (`FrameN`, "This is video", "Video N End"), and `<T seconds>`
timestamp markers. **0 findings.**

What the blind prompt *does* contain, by design:

- the fixed scaffold sentence "…based on all the listed videos." — the only
  injected reference to videos; identical in the with-video run; conveys
  existence, not content;
- the question/options text, which inherently references videos (e.g. id 3's
  options are "VideoID 1..4"). Asserted: the prompt body is *exactly*
  scaffold + question + options + scaffold, so every video mention traces to
  one of those two sources. This is intrinsic to evaluating fixed benchmark
  questions blind, not a pipeline leak — but it means 40% blind accuracy
  reflects language priors over the question/option wording, not "no
  information at all".

## 4. Accuracy recount

| Method | Blind | With-video |
|---|---|---|
| Stored `correct` flags | 18/45 = 40.0% | 28/45 = 62.2% |
| Independent re-derivation (imported `parse_choice`/`gt_choice` over stored `output` strings) | 18/45, 0 per-record mismatches vs stored flags | 28/45, 0 mismatches |
| Run stdout (the fork's `analysis/logs/56689.cvbench_novideo.out`) | `Overall: 18/45 = 40.0%` | — |

- **The 18/45 = 40% claim is confirmed.**
- `"<answer>error</answer>"` sentinel rows (written when a question raises at
  `eval_thinking.py:208-210`, which would silently count as wrong): **0** in
  both result files — all 45 answers in each run are real generations.
- Video access is worth +10 questions (+22.2 pp) over language priors.

## 5. Open questions

1. The exact production input strings/ids were not persisted by the run; §2
   shows deterministic re-renders under the same environment and chat template
   (env fingerprint in `input_pipeline.md` §0). Any future template change in
   the HF cache would change re-renders — the cache snapshot used here is the
   one the run used (`HF_HUB_OFFLINE=1` in both sbatch files).
2. Generation used the checkpoint's sampling defaults (do_sample=true,
   temperature 1.0, top_p 0.95) with `torch.manual_seed(0)`
   (`eval_thinking.py:175,201-204`) — reruns reproduce only under the same
   seed/env; the 18/45 figure is for this stored run, not a deterministic
   property of the model.
