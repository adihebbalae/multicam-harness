# multicam-harness

Shared eval code for the multi-camera experiments: how should multi-camera video be
*packaged* for a frozen VLM, and — the current question — how much of the deficit is
about **which view and window** the model is shown at all?

Inference-only. No training; the harness is the variable.

## Start here — there are two entry points, and only one is current

| | `inprocess/` — **current** | `runner.py` / `run_vqa.py` — legacy |
|---|---|---|
| Model access | in-process (`transformers`), one job owns its GPU | HTTP to a vLLM server you start yourself |
| Arms | class-based `Method` objects, `inprocess/harnesses/` | function-based strategies, `harnesses/` |
| Arms available | centralized, per-stream, blind floor, clip/frame select, option-union, query-search, segment-select | uniform, stitched, decentralized |
| Last touched | actively developed | 2026-07-27 |
| Docs | [`inprocess/README.md`](inprocess/README.md) | this file, "Legacy path" below |

**If you are new to this repo, work in `inprocess/`.** Everything landed since
2026-08-24 — the blind floor, segment selection, media gating, the evidence-class
split, the CPU test gates — is there. The top-level `harnesses/` tree still works and
is kept because the older results were produced through it, but nothing new goes in.

The two trees are *not* duplicates of each other: no file pair is identical, and they
are reached by different entry points. Retiring `harnesses/` is an open decision, not
a cleanup someone forgot to finish.

## Layout

```
inprocess/            THE CURRENT HARNESS — see inprocess/README.md
  run.py              entry point: one subset x one model x N arms x P passes
  harnesses/          the arms (the independent variable)
  dataloaders/        qa_json.py — record schema, video path resolution
  models/, evaluation/

data/subsets/         question pools + the evidence labels and question bank (tracked)
data/crossview-*/     release annotations and video — gitignored, see "Data" below

scripts/data/         pool builders, video fetchers, remux, evidence labelling
evaluation/           scoring.py, chance.py, evidence_class_split.py, llm_judge.py
tests/                CPU gates: prompt equivalence vs the fork, scoring replay
configs/              datasets.yaml (categories + video roots), model_prices.yaml
docs/                 method and results docs — see "Documents" below
plotting/             plot_results.py

runner.py             LEGACY experiment loop (vLLM-server path)
run_vqa.py            LEGACY CLI
harnesses/            LEGACY arms: uniform.py, centralized/, decentralized/
```

Run everything **from the repo root** — imports are top-level packages, and
`inprocess` imports itself absolutely, so `python inprocess/run.py` puts the wrong
directory on `sys.path` and dies before argparse. Use `python -m inprocess.run`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# ffmpeg/ffprobe must be on PATH.
# OpenAI models/judge: put OPENAI_API_KEY=... (or OPENAI_API_KEY1=) in .env at repo root.
```

InternVL3-8B needs the older-transformers environment; Qwen2.5-VL-7B does not.

## Data

Scripts under `scripts/data/` expect the CrossView release at a fixed path:

```
data/crossview-release-annotations/crossview-release/
  annotations/multi-cam-dataset/meva/qa_*.json     # the release QA (annotations only)
  videos/meva/...                                  # fetched separately, see below
```

`data/` is gitignored apart from `data/subsets/`, so this never enters git. On
swarmcluster (`ece-859525`) the annotations already exist — symlink them in:

```bash
mkdir -p data/crossview-release-annotations
ln -s /mnt/Data/ah66742/crossview-release \
      data/crossview-release-annotations/crossview-release
```

That is enough to run everything annotation-only (evidence labelling, the question
bank, chance/floor computations) with no GPU and no video. Verified 2026-09-02: both
Task-3 builders below reproduce their committed outputs byte-identically from this
symlink alone.

Video is a separate, larger step. MEVA is public (CC-BY-4.0) and fetched on demand:

```bash
python3 scripts/data/fetch_meva_videos.py --subset data/subsets/crossview_meva1033_subset.json
python3 scripts/data/remux_avi.py     # writes the verified .mp4 sibling next to each .avi
```

**Run `remux_avi.py` after any fetch.** The release `.avi` containers return the wrong
frame on random access; the loader decodes the verified `.mp4` sibling, and rows
produced from the original containers are excluded from published tables.

The `video_roots.crossview` key in `configs/datasets.yaml`,
`fetch_meva_videos.py --release-root` and `remux_avi.py --root` must all agree, or the
fetched clips land where the harness never looks.

## Run — current path

```bash
python -m inprocess.run \
    --subset data/subsets/crossview_meva1033_subset.json \
    --video-root data/crossview-release-annotations/crossview-release/videos/meva \
    --model OpenGVLab/InternVL3-8B --internvl-max-tiles 1 \
    --strict-answer-prompt 1 --no-reasoning --temperature 0.1 \
    --passes 4 --seeds 1,2,3,4 --nframes 8
```

`--strict-answer-prompt` is **required and has no default**, by design: it rewrites the
prompt of essentially every record, so rows generated with `0` are not poolable with
rows generated with `1`. Agree the value with whoever else is running the same legs
before launching, and check the `strict_prompt` field on the rows before pooling.

Shard long runs with `--chunk N --offset i` (`--chunk` is the *number of shards*).
See [`inprocess/README.md`](inprocess/README.md) for the full arm table and the knobs
that must match across sites.

## Run — legacy path (vLLM server)

Kept for reproducing pre-August results. New work should not use it.

```bash
bash scripts/serve_vllm.sh Qwen/Qwen2.5-VL-7B-Instruct 0 8000   # GPU 0 -> port 8000
python run_vqa.py --dataset meva --data_dir <qa json dir> \
    --model Qwen/Qwen2.5-VL-7B-Instruct --gpu 0 --strategy uniform --num_frames 8
# --strategy uniform | stitched | decentralized
```

Results land in `results/<model>/<dataset>/<strategy>/`; request logs in `logs/`. Both
gitignored.

## Documents

| Doc | What it holds |
|---|---|
| [`docs/evidence_taxonomy.md`](docs/evidence_taxonomy.md) | the five evidence-locality classes (C1–C5) and how each question is assigned one |
| [`docs/dod_question_bank.md`](docs/dod_question_bank.md) | curated DOD-relevant MEVA questions per class, plus what the footage does **not** support |
| [`docs/bounds_meva_internvl3.md`](docs/bounds_meva_internvl3.md) | empirical lower/upper bounds on MEVA x InternVL3-8B |
| [`docs/bench_spec.md`](docs/bench_spec.md), [`docs/PORTING.md`](docs/PORTING.md) | harness contract; the byte-faithfulness rule for ported code |

Rebuild the taxonomy labels and the question bank (annotations only, no GPU):

```bash
python3 scripts/data/label_evidence_class.py    # -> data/subsets/meva_evidence_labels.json
python3 scripts/data/make_question_bank.py      # -> data/subsets/dod_question_bank.json
python -m evaluation.evidence_class_split results/<leg>_shard*.jsonl   # split a leg by class
python3 evaluation/letter_floors.py --subset data/subsets/<pool>.json  # answer-letter skew + LOO floors
```

## Ground rules

- **Inference-only.** No training; the harness is the variable.
- **Equal budget.** Compare arms at the same frame budget, and watch token parity too.
- **One change at a time.** Only the packaging differs between arms — question text,
  model call and scoring stay identical. This includes the *protocol*: reasoning
  on/off and temperature must match across any two arms being compared.
- **Baselines on every number.** Report against chance computed from the real option
  lists (`evaluation/chance.py`), not an assumed 25 %.
- **Error bars before headlines.** `--passes 4 --seeds 1,2,3,4`. Single greedy passes
  are preliminary.
- **Inspect by hand.** Watch the clips before trusting a number.
- **Never commit** data, video, weights, or run outputs (see `.gitignore`).

## Status and open items

- **Result rows** live on the `multicam-results` branch (`multicam_results/`, one
  gzipped JSONL per leg plus a registry), together with `evaluation/single_view_oracle.py`
  and the measured upper bound (`docs/bounds_meva_internvl3.md` §4.1). Not yet merged
  to master — on master that section states the protocol but carries no numbers.
- **Provenance.** The tables in `docs/bounds_meva_internvl3.md` were produced by the
  reference implementation (`Wavy-Hec/CVBench bench/`), which is not vendored here;
  this repo's arms are prompt-equivalence-gated against it (`tests/compare_prompts_vs_fork.py`)
  but do not themselves generate those numbers.
- **Open — answer-letter distribution.** On the meva1033 pool the gold letters are
  heavily skewed, and on two of the three question types part of the option set is
  never correct (temporal: C and D never; event_ordering: B never, A twice in 297).
  The published per-class "floors" equal the per-task modal-letter rates exactly, and
  the nominal 25 % chance level does not hold for those types. Both the floor and any
  conclusion drawn against it need `evaluation/chance.py` on the real option lists,
  and probably an option shuffle, before they are quoted.
- **Two harness trees.** See "Start here". Retiring `harnesses/` is undecided.
