<!-- Ported from Wavy-Hec/CVBench analysis/input_pipeline.md @ 480d6f41cddddc7efea9a09b79134811740ba17a -->
# CVBench video input pipeline: question JSON → model input

Read-only trace of how both eval harnesses turn a CVBench question record
(`id, task_type, video_1..video_4, question, options, answer`) into the actual
model input. All claims are backed by `file:line` references to the code that
ran, and the per-question numbers were produced by re-executing the **imported
production functions** (not re-implementations) on CPU via
the fork's `analysis/inspect_inputs.py` (all embedded assertions passed; raw data in
the fork's `analysis/inspect_inputs_data.json`).

*Note (port): this doc traces the __original__ Video-R1 / lmms-eval harnesses, whose
reused functions are vendored into this repo (`dataloaders/qa_json.py`,
`evaluation/scoring.py`, `models/clients.py`). The `file:line` citations below refer to
those original checkouts in the fork, as they existed at audit time.*

**Headline confirmations**

- **Frames per video = 8 in both harnesses — confirmed.** Qwen3-VL: `--nframes 8`
  (the fork's `analysis/run_eval.sbatch:48`; default also 8, `eval_thinking.py:147`).
  InternVL3: `num_frame=8` in `--model_args` (`run_eval.sbatch:57`; class default
  also 8, `internvl2.py:180`).
- **Sampling is uniform (not fps-based) in both**, but with different formulas
  (endpoint-inclusive linspace vs segment midpoints — see §3).
- **Multi-video combination order = JSON order** (`video_1 → video_4`, nulls
  skipped) in both harnesses, verified positionally and (for Qwen) via in-stream
  timestamps for 5 sample questions (§4).
- **Boundary markers exist in both, in different modalities**: Qwen3-VL gets the
  *text* `"Video k:"` before each clip; InternVL3 gets *marker image frames*
  that literally render **"This is video k"** / **"Video k End"** (the
  `lmms-eval/res/video{1-4}.png` / `end{1-4}.png` files were viewed to confirm
  their content) around each clip's 8 frames.

## 0. Environment fingerprint (what code actually ran)

| Component | Value |
|---|---|
| `qwen_vl_utils` | **pip-installed 0.0.14** in the `cvbench` conda env's site-packages (`.../site-packages/qwen_vl_utils/`) — **not** the repo copy at `Video-R1/src/qwen-vl-utils/` (which lacks `return_video_metadata`; it is dead code for this run) |
| `lmms_eval` | editable repo install, `lmms-eval/lmms_eval/` |
| transformers (Qwen leg) | 5.2.0 (cvbench conda env) |
| transformers (InternVL leg) | 4.48.3 overlay venv per `run_eval.sbatch:52-53` — **venv since deleted, see Open questions** |
| video reader | decord 0.6.0 — proven from the production log line `"qwen-vl-utils using decord to read video."` in the fork's `analysis/logs/56677.cvbench_think.err`; torchcodec not installed, so the backend chain (`vision_process.py:390-398`) lands on decord |
| launch | the fork's `analysis/run_eval.sbatch` (slurm 56677), `HF_HUB_OFFLINE=1`, `DECORD_EOF_RETRY_MAX=20480` |

## 1. Qwen3-VL (Video-R1 thinking harness)

Entry: `Video-R1/src/eval_thinking.py`, launched as
`python Video-R1/src/eval_thinking.py --model_path Qwen/Qwen3-VL-8B-Thinking
--input_json analysis/subset.json --nframes 8 --max_new_tokens 2048`
(`run_eval.sbatch:44-48`).

1. **Path resolution** — `video_paths()` (`eval_thinking.py:91-97`): joins each
   non-null `video_i` onto `--video_root` (default
   `Video-R1/src/r1-v/Evaluation/CVBench`), preserving JSON order.
2. **Message construction** — `build_messages()` (`eval_thinking.py:100-123`):

   ```python
   content = []
   for k, vp in enumerate(video_paths(rec, video_root), 1):
       content.append({"type": "text", "text": f"Video {k}:"})
       content.append({"type": "video", "video": vp, "nframes": nframes})
   content.append({"type": "text", "text": full_prompt})
   ```

   `full_prompt` = option_prompt ("Select the best answer to the following
   multiple-choice/yes-no question based on all the listed videos.") +
   question + options + the think-instruction `QUESTION_TEMPLATE`
   (`eval_thinking.py:40-48`) + answer-format line. No system prompt.
3. **Frame sampling** — `process_vision_info()` (installed
   `vision_process.py:501`) → `fetch_video()` (`:403`) → `_read_video_decord()`
   (`:292`): with explicit `nframes`, `smart_nframes()` (`:144`) returns
   `round_by_factor(8, FRAME_FACTOR=2) = 8`, and indices are
   `torch.linspace(0, total_frames-1, 8).round()` (`:317`) — **uniform,
   endpoint-inclusive (always includes frame 0 and the last frame)**, fps
   ignored for index selection.
4. **Resolution** — `smart_resize()` (`:56`) inside `fetch_video()`: resize to
   multiples of `factor = image_patch_size(16) × SPATIAL_MERGE_SIZE(2) = 32`,
   constrained to [128, 768] vision tokens per frame
   (`VIDEO_MIN/MAX_TOKEN_NUM`, `:28-29`), bicubic, aspect-preserving. Observed:
   1280×720 sources → 640×1152 model frames. The downstream processor is called
   with `do_resize=False` (`eval_thinking.py:198`) so this is the *only* resize.
5. **Why `return_video_metadata=True` matters** — with it,
   `process_vision_info` returns `(tensor, metadata)` tuples and
   `video_kwargs = {'do_sample_frames': False}` (`vision_process.py:528`),
   which stops the HF processor from re-sampling the 8 frames (the in-code
   comment at `eval_thinking.py:187-189` notes it would otherwise silently
   re-sample to 4) and gives it real timestamps.
6. **Token layout** (from decoding the actual `input_ids` produced by
   replicating `eval_thinking.py:190-200` on CPU): each video expands to
   **nframes/2 = 4 vision blocks** (temporal merge of 2 frames per block), each
   preceded by a timestamp text marker:

   ```
   <|im_start|>user
   Video 1:<10.5 seconds><|vision_start|><|video_pad|>×720<|vision_end|>
           <52.3 seconds><|vision_start|><|video_pad|>×720<|vision_end|>
           ... (4 blocks)
   Video 2:<18.4 seconds><|vision_start|>...  (4 blocks)
   <full_prompt text>
   <|im_end|>
   <|im_start|>assistant
   <think>
   ```

   `<|video_pad|>` count per block = (H/32)·(W/32) (e.g. 640×1152 → 720).
   All vision tokens sit **before** the question text; the chat template
   pre-opens `<think>` for the Thinking checkpoint. Generation uses the
   checkpoint's own `generation_config` (sampling: temperature 1.0, top_p 0.95,
   top_k 20), seeded with `torch.manual_seed(0)` (`eval_thinking.py:175,201-204`).

## 2. InternVL3-8B (lmms-eval, task `mvr_think`)

Launched from `lmms-eval/` as `python -m lmms_eval --model internvl2
--model_args "pretrained=OpenGVLab/InternVL3-8B,num_frame=8" --tasks mvr_think
--batch_size 1 --log_samples` (`run_eval.sbatch:55-60`). The dataset
`mvr_dataset/` is rebuilt from the fork's `analysis/subset.json` at the top of every run
(`run_eval.sbatch:34-41`), so both harnesses answer identical records.

1. **Visual paths** — `mvr_doc_to_visual()`
   (`lmms-eval/lmms_eval/tasks/mvr/utils.py:101-103`): returns
   `[cache_dir + "/" + video_i]` for non-null `video_i` in order; `cache_dir`
   resolves to the same absolute
   `Video-R1/src/r1-v/Evaluation/CVBench` (`mvr.yaml` `dataset_kwargs.cache_dir`,
   joined at `utils.py:47-58` — absolute, so the HF_HOME prefix is discarded).
2. **Prompt text** — `mvr_doc_to_text_think()` (`utils.py:297-308`):
   option_prompt + question + options + `THINK_INSTRUCTION` (`utils.py:286-290`)
   + answer-format line. Verified byte-identical to the `"input"` field stored
   in the run's per-sample log
   (`lmms-eval/logs/OpenGVLab__InternVL3-8B/20260610_132552_samples_mvr_think.jsonl`).
3. **Frame sampling** — `get_index()`
   (`lmms-eval/lmms_eval/models/internvl2.py:91-100`): the video is split into
   8 equal segments and the **midpoint frame of each segment** is taken
   (`int(start_idx + seg/2 + round(seg·i))`, `start_idx=0` here since `bound=None`) — uniform, fps-independent, never includes
   frame 0 or the last frame.
4. **Preprocessing** — `load_video()` (`internvl2.py:103-119`): each frame goes
   through `dynamic_preprocess(..., image_size=448, max_num=1)`; with
   `max_num=1` the only admissible tiling is 1×1, so every frame becomes a
   **single 448×448 tile (aspect ratio squashed)**, ImageNet-normalized.
   Measured: 1 tile/frame for all 40 sample frames (never assumed).
5. **Multi-video assembly** (`internvl2.py:345-362`, the locally added
   multi-video branch): for video j, pixel tensors are concatenated as
   `[marker "This is video j" image, 8 frames, marker "Video j End" image]`,
   markers loaded from `lmms-eval/res/` at 448px; `num_patches_lists +=
   [1] + [1]*8 + [1]`. The text side is

   ```
   Frame1: <image>\nFrame2: <image>\n... FrameM: <image>\n
   + "Please pay close attention to the video frames with special cues ..."  (internvl2.py:361)
   + contexts (from mvr_doc_to_text_think)
   ```

   so for an N-video question M = N×10 slots, and video j owns slots
   `Frame[10(j-1)+1 .. 10j]` (its 8 content frames at `Frame[10(j-1)+2 ..
   10(j-1)+9]`).
6. **Token expansion** — `model.chat()` replaces each `<image>` with
   `<img>` + `<IMG_CONTEXT>`×256×num_patches + `</img>`
   (cached remote code `modeling_internvl_chat.py:284`; `num_image_token =
   (448/14)² × 0.5² = 256`, `:52`), wrapped in the `internvl2_5` conversation
   template (Chinese InternVL system message, `conversation.py:384-390`).
   With 1 patch/slot: 258 vision tokens per Frame slot; a 4-video question
   carries 40 slots ≈ 10.3k vision tokens — the local sdpa-attention patch in
   `internvl2.py` exists to keep that within L40 memory.

## 3. Sampling-strategy comparison (same video, both harnesses)

`0/0Hd95ulYOT8.mp4` (7328 frames @ 50 fps, 146.6 s):

| Harness | Formula | Indices |
|---|---|---|
| Qwen3-VL | `linspace(0, 7327, 8).round()` | 0, 1047, 2093, 3140, 4187, 5234, 6280, **7327** |
| InternVL3 | segment midpoints `int(seg/2 + round(seg·i))`, seg = 7327/8 | **457**, 1373, 2289, 3205, 4121, 5036, 5952, 6868 |

Same 8-frame budget and uniform coverage, but Qwen anchors on the first/last
frames while InternVL samples segment centers (offset ≈ half a segment). Both
read indices from the file actually named in the JSON (including `_seg` files —
indices are relative to the segment file, not the parent video).

## 4. Sample-question verification (5 questions, mix of 2/3/4 videos, 2 with `_seg` files)

Sample rule (deterministic, asserted in the fork's `inspect_inputs.py`): lowest id per
n_videos bucket {2,3,4} + lowest-id `_seg` question + lowest-id 4-video `_seg`
question → ids **0, 4, 3, 17, 19**.

**Attribution methodology.** Neither harness tags frames with provenance —
attribution is **positional**: Qwen interleaves content in JSON order
(`eval_thinking.py:117-122`), InternVL concatenates `[start_j, frames_j,
end_j]` in j order (`internvl2.py:349-359`). Verification therefore
(a) recomputes each video's frame-index list with the imported production
samplers, (b) recomputes the layout offsets, and (c) for Qwen additionally
checks an *independent in-band signal*: the `<T seconds>` timestamps that the
processor embeds before each vision block must equal
`mean(frame_index_pair)/fps` of the matching `video_N` file. This held for all
16 videos below (tolerance 0.11 s, the timestamps' print precision), which ties
each vision block to the sampled frames of the correct file. For `_seg` files,
the sampler's `total_num_frames` was asserted equal to the segment file's own
decord frame count (e.g. `_seg1` 133 vs `_seg2` 276), proving indices are
segment-relative.

| id | video_N | file | frames @ fps | Qwen frame indices | InternVL frame indices | Position in combined input (Qwen / InternVL) |
|---|---|---|---|---|---|---|
| 0 | 1 | `0/0Hd95ulYOT8.mp4` | 7328 @ 50.0 | 0,1047,2093,3140,4187,5234,6280,7327 | 457,1373,2289,3205,4121,5036,5952,6868 | after "Video 1:", blocks 1-4 (t=10.5,52.3,94.2,136.1 s) / Frame1-10 (markers at 1,10) |
| 0 | 2 | `0/XW-lrijc1oQ.mp4` | 6453 @ 25.0 | 0,922,1843,2765,3687,4609,5530,6452 | 403,1209,2016,2823,3629,4435,5242,6049 | after "Video 2:", blocks 5-8 (t=18.4,92.2,165.9,239.6 s) / Frame11-20 |
| 4 | 1 | `10/6KPeYiObWHs.mp4` | 3500 @ 30.0 | 0,500,1000,1500,1999,2499,2999,3499 | 218,655,1093,1530,1968,2405,2842,3280 | blocks 1-4 (t=8.3..108.3) / Frame1-10 |
| 4 | 2 | `10/QvWk-HbwdSA.mp4` | 998 @ 30.0 | 0,142,285,427,570,712,855,997 | 62,187,311,436,560,685,810,934 | blocks 5-8 (t=2.4..30.9) / Frame11-20 |
| 4 | 3 | `10/sleeMRoccOo.mp4` | 343 @ 29.861 | 0,49,98,147,195,244,293,342 | 21,64,107,149,192,235,277,320 | blocks 9-12 (t=0.8..10.6) / Frame21-30 |
| 3 | 1 | `1/5DHFEzdgtxE.mp4` | 243 @ 30.0 | 0,35,69,104,138,173,207,242 | 15,45,75,106,136,166,197,227 | blocks 1-4 / Frame1-10 |
| 3 | 2 | `1/7J3X3cbpFAI.mp4` | 268 @ 30.0 | 0,38,76,114,153,191,229,267 | 16,49,83,116,150,183,216,250 | blocks 5-8 / Frame11-20 |
| 3 | 3 | `1/9aF1qmty6AU.mp4` | 212 @ 30.0 | 0,30,60,90,121,151,181,211 | 13,39,66,92,119,145,171,198 | blocks 9-12 / Frame21-30 |
| 3 | 4 | `1/N9PVw7G32F4.mp4` | 290 @ 30.0 | 0,41,83,124,165,206,248,289 | 18,54,90,126,162,199,235,271 | blocks 13-16 / Frame31-40 |
| 17 | 1 | `104/PB-3DQ4NuVM_seg1.mp4` | 133 @ 25.0 | 0,19,38,57,75,94,113,132 | 8,24,41,58,74,90,107,124 | blocks 1-4 (t=0.4..4.9) / Frame1-10 |
| 17 | 2 | `104/PB-3DQ4NuVM_seg2.mp4` | 276 @ 25.0 | 0,39,79,118,157,196,236,275 | 17,51,86,120,155,189,223,258 | blocks 5-8 (t=0.8..10.2) / Frame11-20 |
| 17 | 3 | `104/PB-3DQ4NuVM_seg3.mp4` | 329 @ 25.0 | 0,47,94,141,187,234,281,328 | 20,61,102,143,184,225,266,307 | blocks 9-12 (t=0.9..12.2) / Frame21-30 |
| 19 | 1 | `105/3nsWXX0zkEM_seg1.mp4` | 360 @ 23.976 | 0,51,103,154,205,256,308,359 | 22,67,112,157,202,246,291,336 | blocks 1-4 / Frame1-10 |
| 19 | 2 | `105/3nsWXX0zkEM_seg3.mp4` | 120 @ 23.976 | 0,17,34,51,68,85,102,119 | 7,22,37,52,67,81,96,111 | blocks 5-8 / Frame11-20 |
| 19 | 3 | `105/kSt4kWaRCzk_seg2.mp4` | 1349 @ 23.976 | 0,193,385,578,770,963,1155,1348 | 84,252,421,590,758,926,1095,1264 | blocks 9-12 / Frame21-30 |
| 19 | 4 | `105/T6i8LpH9JcY_seg4.mp4` | 1406 @ 25.0 | 0,201,401,602,803,1004,1204,1405 | 87,263,438,614,789,965,1141,1316 | blocks 13-16 / Frame31-40 |

(id 17 is the strongest segment test: three `_seg` slices of the *same* source
video must land at positions 1/2/3 — they do, and each is sampled from its own
file as shown by the differing frame counts 133/276/329.)

Asserted for every question: content-list structure `("Video k:", video_k)×N +
prompt`; `"Video k:"` precedes the k-th group of 4 vision blocks which precedes
`"Video k+1:"`; question text last; per-block `<|video_pad|>` count =
(H/32)·(W/32); InternVL slot bookkeeping `[1]+[1]*8+[1]` per video.

## 5. Open questions (not confirmable from code/logs)

1. **InternVL leg interpreter**: it ran via the fork's since-deleted
   `.venv-internvl` interpreter with a
   "transformers==4.48.3 overlay" (`run_eval.sbatch:52-53`), but that venv has
   been deleted; the exact transformers version of that leg is evidenced only
   by the sbatch comment (not echoed into any log).
2. **Reconstructed vs production inputs**: neither harness persists the final
   model-input string/ids. The Qwen layouts above are deterministic CPU
   re-renders under the same pinned env (fingerprint in §0); InternVL `contexts`
   were cross-checked byte-for-byte against the stored `"input"` field, but the
   `Frame…<image>` assembly happens inside `generate_until` and is logged
   nowhere — it is reconstructed from `internvl2.py:345-362` plus measured
   patch counts.
3. **Qwen timestamp semantics**: the `<T seconds>` = mean of each merged frame
   pair / fps relationship was verified empirically on all 20 videos; the
   transformers-side processor code that renders it was not traced line-by-line.
