# `inprocess/` — in-process harness arms

A self-contained implementation of the **centralized**, **decentralized**, and
**clip/frame-selection** arms, kept in its own package so it can be read, run and
compared without disturbing anything already in the repo.

**This directory adds files only.** Nothing outside `inprocess/` is modified: the
existing `harnesses/`, `dataloaders/`, `evaluation/` and `models/` packages are
untouched, so the two implementations can sit side by side and be diffed against
each other on the same subset. Take whatever is useful and leave the rest — the
package has no hooks into the rest of the tree.

## Arms

| Class | Module | What the model sees | Calls/question |
|---|---|---|---|
| `CentralizedMethod` | `harnesses.stitched` | Time-synchronized frames from all views tiled into labeled grid montages, fed as one visual input | 1 |
| `PerStreamMethod` | `harnesses.decentralized` | One query-conditioned perception pass per view, then a text-only aggregation pass over the descriptions | k+1 |
| `SummarySelectMethod` | `harnesses.clip_select` | Cached per-clip text summaries routed by the same model, which picks the clips the question needs | 1–2 |
| `ClipScoreSelectMethod` | `harnesses.clip_select` | Clips scored by CLIP/SigLIP text-image similarity over thumbnails; keep the top-m | 1 |
| `FrameSelectMethod` | `harnesses.clip_select` | One shared budget of the most question-relevant frames chosen globally across every clip, grouped by source clip | 1 |
| `OptionUnionFrameSelect` | `harnesses.option_union` | The union of frames passing each answer option's similarity threshold (absolute or per-option quantile), every option guaranteed its best frame | 1 |
| `OptionUnionClipSelect` | `harnesses.option_union` | Whole clips kept when any option's threshold passes; the `viclip` scorer embeds a clip's frames jointly (video-native) from a local OpenGVLab/ViCLIP download | 1 |
| `QuerySearchMethod` | `harnesses.option_union` | The backend writes short visual search phrases from the question and options, then the top-budget frames matching any phrase are retrieved before the answer call | 2 |

Montage geometry is `cols = ceil(sqrt(K))`, `rows = ceil(K / cols)` — 2x2 at four
views, up to 4x4 at the thirteen-slot cap.

## Usage

Every arm shares one interface, so swapping the harness is the only variable:

```python
from inprocess.harnesses.stitched import CentralizedMethod
from inprocess.harnesses.decentralized import PerStreamMethod
from inprocess.harnesses.clip_select import FrameSelectMethod

m = CentralizedMethod(backend, nframes=8)      # any backend exposing .generate()
res = m.answer(record, video_root, seed=1)     # -> Result dataclass
```

`Result` carries the prediction, the gold answer, the reasoning trace, latency,
token counts and a `frame_alloc` dict recording exactly how the frame budget was
spent — so an accuracy difference can always be checked against what each arm was
actually given.

Records are the `video_1..video_N` / `image_1..image_N` schema already used by
`data/subsets/`. Multi-view still-image records are supported by the centralized
and decentralized arms; the selection arms sample frames out of clips and raise a
clear error on a still-image record rather than silently running with no visual
input.

## MVU-Eval, end to end

MVU-Eval (multi-video QA, NeurIPS 2025 D&B) is the pool `run.py`'s usage examples
point at. It is a public HF dataset — no license gate — so this runs on any
cluster with internet and a GPU. Five steps from a fresh clone:

```bash
# 1. environment. requirements.txt covers the Qwen path. InternVL3 needs its
#    OWN environment pinned to an older transformers -- the newer one the Qwen
#    path wants breaks InternVL3's remote code, so one environment cannot
#    serve both backends.
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

# 2. the questions (small: one JSON + one CSV)
python3 scripts/data/fetch_mvueval_videos.py --qa-only

# 3. build the record pool. No GPU, no videos needed. Reproduces the committed
#    data/subsets/mvueval_qa.json byte for byte from the same release.
python3 scripts/data/build_mvueval.py --qa-json data/mvueval/MVU_Eval_QAs.json

# 4. the clips. The dev subset needs a few hundred; the full pool needs ~4.9k.
python3 scripts/data/fetch_mvueval_videos.py --subset data/subsets/mvueval_dev_subset.json
python3 scripts/data/fetch_mvueval_videos.py --check   # confirm before launching

# 5. smoke it: 5 questions, one pass, sequential baseline only
mkdir -p logs                      # Slurm opens --output before the body runs
LIMIT=5 PASSES=1 SEEDS=1 STRICT=1 METHODS=cvbench_native \
    sbatch -p <your partition> scripts/run_mvueval.sbatch
```

Then the real legs — `scripts/run_mvueval.sbatch` wraps `python -m inprocess.run`
and shards over a Slurm array. It declares no partition, so pass `-p` yourself;
the walltime, gres and cpus-per-task are one cluster's defaults, overridable with
`sbatch -t <yours> --gres=<yours>`. It activates `$VENV` or `$ENV` if you set one,
and otherwise assumes you activated the environment before submitting. `STRICT`
is required — see below.

```bash
# the two arms that need no gated weights
STRICT=1 METHODS=frame_select_siglip_optu,query_search_siglip \
    SUBSET=data/subsets/mvueval_qa.json CHUNK=8 \
    sbatch -p <your partition> --array=0-7 scripts/run_mvueval.sbatch
```

Step 3 emits the full pool plus a task x clip-count stratified dev subset and the
deduped clip list that subset needs, so a smoke run never requires the whole
release. Questions carry 2–13 videos and 2–11 options; the slot cap and the legal
letter range are imported from `inprocess/dataloaders/qa_json.py` rather than
restated, so emitted records cannot desync from the loader that reads them.

### Four things that will bite you

- **`--strict-answer-prompt` is required and has no default.** It rewrites nearly
  every prompt of a multi-option subset, so rows produced with `0` and rows
  produced with `1` **cannot be pooled**. Agree on one value with anyone else
  running these legs before either of you launches, and check the `strict_prompt`
  field on the rows before combining anything.
- **The release ships two answer keys that disagree.** `MVU_Eval_QAs.json` and
  `mvu_eval_config.csv` give different ground truth for **92 of the 1,824
  questions (5.04%)**, joined positionally — the question text matches on all
  1,824, so the join is sound. `build_mvueval.py` reads the JSON. Say which key
  you scored against when comparing against published numbers.
- **`clip_select_viclip_optu` needs a gated download.** ViCLIP is not on PyPI:
  `huggingface-cli login`, then `huggingface-cli download OpenGVLab/ViCLIP
  --local-dir $VICLIP_DIR` (default `~/models/ViCLIP`), and it must be the
  *full* download — code files and the `ViClip-InternVid-10M-FLT.pth`
  checkpoint. The other `.pth` in that repo is vision-encoder-only and is not a
  valid fallback. The other two arms need no gated weights — but note that
  `run.py`'s default `--methods` includes the viclip arm, so a run that passes
  no `--methods` exits early without that download.
- **On InternVL3, keep `--internvl-max-tiles 1`.** An image is tiled while a
  video frame is not, so a matched frame budget stops being matched once tiling
  is on; the selection arms refuse to construct rather than produce a
  tokenization artifact. The sbatch adds the flag automatically when `MODEL`
  names an InternVL checkpoint.

## Reasoning mode and option-guided selection

Two knobs were added since the initial drop, both default-off so existing calls
behave exactly as before:

- **`reasoning=False`** on any arm swaps the prompt for a direct-answer
  template. There is no model-side thinking switch — the visible trace is
  produced by the prompt — so turning it off means asking for the answer
  directly. The `<answer>` tags stay, so one parser serves both modes, and the
  parser also accepts the tagless letter-plus-option-text shape that
  direct-answer models produce.
- **`query="options"`** on `ClipScoreSelectMethod` / `FrameSelectMethod` scores
  frames against EACH answer option separately (reduced by max) instead of the
  question. Options are embedded one by one, so nothing is lost to the text
  encoder's 64/77-token cap. Rows record `query_mode` and the query count in
  `frame_alloc`, so an option-guided row is distinguishable from a
  question-guided one without the launch environment. A record with no usable
  option text falls back to the question and says so in the row.

Setting `STRICT_ANSWER_PROMPT=1` in the environment enables a v2 answer-hygiene
prompt (every legal letter enumerated, options declared exhaustive so N/A-style
refusals are ruled out). It changes generation, so it is off by default — and
because the flag lives in the environment, a runner that adopts it should record
its value alongside each row (ours stamps a `strict_prompt` field at write
time), so runs on different prompt versions can never silently pool.

The selection arms now fail loudly rather than degrade: an unreadable clip, or a
similarity scorer that fails to load or score, raises instead of silently
falling back to unguided selection — a row that looks option-guided while
carrying no guidance is worse than a crashed leg.

## How this respects the ground rules

- **Inference-only.** No training anywhere in this package; the harness is the
  only variable.
- **Equal budget.** `PerStreamMethod` takes `total_frames`, which fixes the
  *total* frames per question and splits them across its clips, so two harnesses
  can be compared at one budget rather than at equal frames-per-clip (which is not
  an equal budget when clip counts differ). `Result.frame_alloc` records the split
  and the token counts, so token parity is checkable and not assumed.
- **One change at a time.** The arms share a single prompt scaffold and answer
  parser; only the visual packaging differs between them. The selection arms'
  all-clips branch emits a prompt byte-identical to the sequential baseline, so any
  delta is attributable to the clips it pruned.
- **Never commit data, video, weights or run outputs.** This package is code only.
- **Passes.** Selection is deterministic and cached across passes of the same
  question, so a multi-pass standard deviation isolates the answer stage rather
  than re-rolling the selection.

## Provenance

Ported from the CVBench evaluation fork; each module carries a header naming the
source file and commit it was taken from, plus any deliberate delta. The
selection arms additionally guard against two failure modes that do not raise on
their own: running with zero visual items on a still-image record, and a global
top-k ranking starving whole cameras of frames, which biases exactly the
camera-count axis these arms are compared on.
