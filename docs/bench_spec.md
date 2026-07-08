<!-- Ported from Wavy-Hec/CVBench bench/bench_spec.md @ 480d6f41cddddc7efea9a09b79134811740ba17a -->
# Multi-camera benchmark spec

Implements "Task 1: Evaluations" from the mentor's Task-1 writeup (the fork's
`Multi Camera Video Writeup and Logs.md`):
compare a **centralized** vs **decentralized** harness, across 2 VLMs, on CrossView,
producing Table 1 + Plots 1–4 with accuracy + latency segmented by question category
and camera count. Each run = N **methods** × M **backends** × P **passes**.

## Methods (`harnesses/`) — the two harnesses
- `centralized` — temporally aligns the K (≤4) clips and **spatially stitches** the
  synchronized frames into grid-montage images (`harnesses/stitched.py`), fed as ONE unified
  visual input to a single VLM call. Montages are built once per question and reused
  across passes.
- `per_stream` (decentralized) — one perception pass per clip → a **text-only** aggregator
  pass reasons over the K per-camera descriptions.
- *(A2 router / A3 NeuS-QA: roadmap, not built.)*

## Backends (`models/clients.py`) — the two models
- `qwen3vl` → `Qwen/Qwen3-VL-8B-Thinking` (cached ✅; runs under the `cvbench` env).
- `internvl3` → `OpenGVLab/InternVL3-8B` (cached ✅; **must run under the `internvl` env** —
  `cvbench`'s transformers breaks the InternVL3 remote code).
- `qwen3vl-instruct` → `Qwen/Qwen3-VL-8B-Instruct` (cached ✅; ablation backend).

## 4-pass protocol (Table 1 std)
A "pass" = one generation with decoding **sampling on** (`--temperature 0.7`, default
seeds `1,2,3,4`), with the (deterministic) frames held fixed. Std is taken over the
per-pass accuracies (`metrics.summarize_passes`).

## Metrics (one `Result` row per question × method × backend × pass)
- **M1 Accuracy** — overall + by `task_type` / `orig_num_cameras` / `source` / `cap_answer_safe`
  (never pool); mean±std over passes.
- **M2 Latency** — `latency_s` (end-to-end serial); for `per_stream` also
  `perception_latency_serial_s` / `perception_latency_par_s` (max) / `aggregate_latency_s`.
- **M3 Cost** — `input_tokens`, `video_tokens`, `output_tokens`, `num_model_calls`.
- **M4 Calibration** — `abstained` (empty prediction) → abstain rate.

## Deliverables (`plotting/plot_results.py` → `results/figs/`)
- **Table 1** — overall accuracy mean±std for the 2 models × 2 harnesses (`table1.md/.csv`).
- **Plot 1** — accuracy per question category (grouped bar, std error bars).
- **Plot 2** — latency per question (box, X = methods, Y = ms).
- **Plot 3** — accuracy vs camera count (line, X = `orig_num_cameras`).
- **Plot 4** — accuracy per category vs camera count (faceted lines).

## Pools
- Dev (100 Q): `data/subsets/crossview_combined_subset.json` (MEVA + EgoExo4D; cameras 2–12).
- Scale (1,033 Q): `data/subsets/crossview_meva1033_subset.json` (cameras 2–16).
- `video_root` = your CrossView release directory (`crossview-release`, MEVA .avi +
  EgoExo4D .mp4) — set it in `configs/datasets.yaml` or pass `--video-root`.
- NOTE: records carry `video_1..video_4` (input capped at 4 cams → montage ≤ 2×2); the
  Plot 3/4 camera axis is `orig_num_cameras` (the original difficulty count, 2–16).

## Commands

*(The CPU scoring gate — reproduce 19/60 from a stored eval JSON, no GPU — is the fork's
`bench/validate_scoring.py`; it was not ported.)*

```bash
# GPU smoke (Qwen, cvbench env), both harnesses, 4 passes:
python run_vqa.py --subset data/subsets/crossview_combined_subset.json \
    --methods centralized,per_stream --backends qwen3vl \
    --passes 4 --seeds 1,2,3,4 --temperature 0.7 --limit 5

# full meva1033 sweep, sharded 8 ways (Qwen leg under cvbench, InternVL under internvl):
ENV=cvbench SUBSET=data/subsets/crossview_meva1033_subset.json BACKENDS=qwen3vl \
    CHUNK=8 sbatch --array=0-7 scripts/run_bench.sbatch
ENV=internvl SUBSET=data/subsets/crossview_meva1033_subset.json BACKENDS=internvl3 \
    CHUNK=8 sbatch --array=0-7 scripts/run_bench.sbatch

# render Table 1 + Plots 1–4 from the (concatenated) result shards:
python plotting/plot_results.py --jsonl results/*meva1033*.jsonl --out-dir results/figs
```
The reused functions — `build_messages`, `parse_choice`, `gt_choice`, `num_videos`,
`video_paths`, `load_model` — are vendored (single source of truth) from the sibling
`Video-R1/src/eval_thinking.py` into `dataloaders/qa_json.py`, `evaluation/scoring.py`,
and `models/clients.py`. The InternVL3 preprocessing (`load_image`/`load_video`) is
ported from `lmms-eval/.../internvl2.py`.
