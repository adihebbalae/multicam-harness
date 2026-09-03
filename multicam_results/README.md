# MultiCam benchmark results

Per-question result rows from the `Wavy-Hec/MultiCam` harness, one merged JSONL per leg file group, exported at commit `8b8a04b3f716216fd0772a158ee35d3457591370`.

A row is one question x method x backend x sampling pass. **Key on `(dataset, id)`** — `id` restarts at 0 in every subset. The question records live in `data/subsets/<dataset>.json`.

## Layout

```
registry.json   one entry per (dataset, backend, method, budget): file, rows, passes, accuracy, complete
summary.csv     the same, one line per leg, for a spreadsheet
legs/           merged rows, full fields (gzip; `gzip.open(path, 'rt')` or `zcat`)
```

Several methods share one file: a file group is every arm that ran in one Slurm array. Filter on `method` and `backend`.

## Reading a leg

```python
import gzip, json, pandas as pd
rows = [json.loads(l) for l in gzip.open('multicam_results/legs/<leg>.jsonl.gz', 'rt')]
df = pd.DataFrame(rows)
df[df.method == 'cvbench_native'].groupby('pass_idx').correct.mean()   # per-pass accuracy of one arm
df.groupby(['method', 'backend']).video_tokens.mean()               # the budget each arm actually used
```

## Protocol

- passes: 4 per question (seeds 1-4, temperature 0.7) unless summary.csv says otherwise; reasoning is imposed by the prompt scaffold, `reasoning` in summary.csv says which template.
- scoring: the letter parsed from `<answer>` (or the bare reply); no letter parsed = abstained = incorrect. Rows were re-scored with the current parser (`shards_source` in registry.json).
- montage arm (`centralized`): every view resized into a fixed square cell; its token count saturates while the sequential arm's grows with views — compare `video_tokens` before reading an arm gap.
- MEVA rows decode the remuxed `.mp4` siblings of the release `.avi` (`media_remap`); see the exclusion note below.

## Row schema

| field | meaning |
|---|---|
| `id` | question index within its subset; unique only with `dataset` |
| `dataset` | subset stem the row was run against (matches data/subsets/<dataset>.json) |
| `leg` | which merged file / campaign the row came from |
| `task_type, source, orig_num_cameras, num_videos` | copied from the question record |
| `cap_answer_safe` | whether the camera cap leaves the answer recoverable |
| `method` | arm: cvbench_native (sequential), centralized (montage), per_stream, blind, single_viewN, segment_select* |
| `backend` | model name |
| `prediction, gold, correct, abstained` | parsed letter, gold letter, 0/1, no letter parsed (scored incorrect) |
| `pass_idx, seed, temperature` | sampling pass; passes are independent decodes with frames fixed |
| `latency_s, perception_latency_*_s, aggregate_latency_s` | wall-clock; the per_stream arm splits perception and aggregation |
| `input_tokens, video_tokens, output_tokens, num_model_calls` | budget actually consumed — compare across arms before reading any accuracy gap |
| `response_text, think` | full model output and the <think> block (absent in slim/) |
| `frame_alloc` | frames per view the arm chose; on segment_select legs also the per-segment scores (absent in slim/) |
| `media_remap` | 'avi->mp4' on MEVA rows decoded through the remux; the reason older MEVA legs are excluded |
| `error` | non-null when the call failed; such rows are counted in `errors` |
| `source_commit` | Wavy-Hec/MultiCam commit this bundle was exported from |

## Legs

| leg | dataset | backend | method | budget | questions | rows | acc % | complete |
|---|---|---|---|---|--:|--:|--:|---|
| `mvueval_qa_internvl_mvufull` | MVU-Eval | InternVL3-8B | centralized | 8 frames/video | 1824/1824 | 7296 | 51.95 | yes |
| `mvueval_qa_internvl_mvufull` | MVU-Eval | InternVL3-8B | cvbench_native | 8 frames/video | 1824/1824 | 7296 | 52.88 | yes |
| `mvueval_qa_internvl_mvufull` | MVU-Eval | InternVL3-8B | per_stream | 8 frames/video | 1824/1824 | 7296 | 43.98 | yes |
| `mvueval_qa_cvbench_mvufullq25` | MVU-Eval | Qwen2.5-VL-7B-Instruct | centralized | 8 frames/video | 1824/1824 | 7296 | 40.41 | yes |
| `mvueval_qa_cvbench_mvufullq25` | MVU-Eval | Qwen2.5-VL-7B-Instruct | cvbench_native | 8 frames/video | 1824/1824 | 7296 | 50.79 | yes |
| `mvueval_qa_cvbench_mvufullq25` | MVU-Eval | Qwen2.5-VL-7B-Instruct | per_stream | 8 frames/video | 1824/1824 | 7296 | 40.53 | yes |
| `mvueval_qa_internvl_mvufull32` | MVU-Eval | InternVL3-8B | centralized | 32 frames total | 1824/1824 | 7296 | 52.18 | yes |
| `mvueval_qa_internvl_mvufull32` | MVU-Eval | InternVL3-8B | cvbench_native | 32 frames total | 1824/1824 | 7296 | 52.54 | yes |
| `mvueval_qa_internvl_mvufull32` | MVU-Eval | InternVL3-8B | per_stream | 32 frames total | 1824/1824 | 7296 | 44.68 | yes |
| `mvueval_qa_internvl_mvupp32` | MVU-Eval | InternVL3-8B | cvbench_native | 32 frames/video | 1824/1824 | 7296 | 50.21 | yes |
| `mvueval_qa_internvl_mvupp32` | MVU-Eval | InternVL3-8B | per_stream | 32 frames/video | 1824/1824 | 7296 | 41.37 | yes |
| `mvueval_qa_internvl_mvublindshard` | MVU-Eval | InternVL3-8B | blind | no images | 1824/1824 | 7296 | 30.48 | yes |
| `mvueval_qa_internvl_mvufullclip` | MVU-Eval | InternVL3-8B | clip_select_top1 | 64 frames total | 1824/1824 | 7296 | 43.75 | yes |
| `mvueval_qa_internvl_mvufullclip` | MVU-Eval | InternVL3-8B | frame_select | 64 frames total | 1824/1824 | 7296 | 52.36 | yes |
| `mvueval_qa_internvl_mvufullclip` | MVU-Eval | InternVL3-8B | temporal_weighted | 64 frames total | 1824/1824 | 7296 | 51.84 | yes |
| `mvueval_noview_subset_internvl_mvusv` | MVU-Eval | InternVL3-8B | single_view1 | 8 frames/video, single view | 545/545 | 2180 | 42.06 | yes |
| `mvueval_noview_subset_internvl_mvusv` | MVU-Eval | InternVL3-8B | single_view10 | 8 frames/video, single view | 12/12 | 48 | 58.33 | yes |
| `mvueval_noview_subset_internvl_mvusv` | MVU-Eval | InternVL3-8B | single_view11 | 8 frames/video, single view | 10/10 | 40 | 40.0 | yes |
| `mvueval_noview_subset_internvl_mvusv` | MVU-Eval | InternVL3-8B | single_view12 | 8 frames/video, single view | 3/3 | 12 | 33.33 | yes |
| `mvueval_noview_subset_internvl_mvusv` | MVU-Eval | InternVL3-8B | single_view13 | 8 frames/video, single view | 2/2 | 8 | 25.0 | yes |
| `mvueval_noview_subset_internvl_mvusv` | MVU-Eval | InternVL3-8B | single_view2 | 8 frames/video, single view | 545/545 | 2180 | 41.33 | yes |
| `mvueval_noview_subset_internvl_mvusv` | MVU-Eval | InternVL3-8B | single_view3 | 8 frames/video, single view | 545/545 | 2180 | 41.88 | yes |
| `mvueval_noview_subset_internvl_mvusv` | MVU-Eval | InternVL3-8B | single_view4 | 8 frames/video, single view | 541/541 | 2164 | 40.76 | yes |
| `mvueval_noview_subset_internvl_mvusv` | MVU-Eval | InternVL3-8B | single_view5 | 8 frames/video, single view | 463/463 | 1852 | 40.01 | yes |
| `mvueval_noview_subset_internvl_mvusv` | MVU-Eval | InternVL3-8B | single_view6 | 8 frames/video, single view | 317/317 | 1268 | 44.24 | yes |
| `mvueval_noview_subset_internvl_mvusv` | MVU-Eval | InternVL3-8B | single_view7 | 8 frames/video, single view | 43/43 | 172 | 62.21 | yes |
| `mvueval_noview_subset_internvl_mvusv` | MVU-Eval | InternVL3-8B | single_view8 | 8 frames/video, single view | 33/33 | 132 | 52.27 | yes |
| `mvueval_noview_subset_internvl_mvusv` | MVU-Eval | InternVL3-8B | single_view9 | 8 frames/video, single view | 21/21 | 84 | 61.9 | yes |
| `allangles_egohumans_qa_internvl_aabfull` | All-Angles | InternVL3-8B | centralized | stills, cell_px=448 | 170/170 | 680 | 39.26 | yes |
| `allangles_egohumans_qa_internvl_aabfull` | All-Angles | InternVL3-8B | cvbench_native | stills, cell_px=448 | 170/170 | 680 | 44.12 | yes |
| `allangles_egohumans_qa_internvl_aabfull` | All-Angles | InternVL3-8B | per_stream | stills, cell_px=448 | 170/170 | 680 | 46.32 | yes |
| `allangles_egohumans_qa_cvbench_aabfull` | All-Angles | Qwen3-VL-8B-Thinking | centralized | stills, cell_px=448 | 170/170 | 680 | 39.56 | yes |
| `allangles_egohumans_qa_cvbench_aabfull` | All-Angles | Qwen3-VL-8B-Thinking | cvbench_native | stills, cell_px=448 | 170/170 | 680 | 46.91 | yes |
| `allangles_egohumans_qa_cvbench_aabfull` | All-Angles | Qwen3-VL-8B-Thinking | per_stream | stills, cell_px=448 | 77/170 | 292 | 50.0 | NO |
| `allangles_egohumans_qa_internvl_aabblind` | All-Angles | InternVL3-8B | blind | no images | 170/170 | 680 | 33.24 | yes |
| `mvueval_qa_internvl_t1iv` | MVU-Eval | InternVL3-8B | centralized | 8 frames/video | 1824/1824 | 7296 | 52.11 | yes |
| `mvueval_qa_internvl_t1iv` | MVU-Eval | InternVL3-8B | cvbench_native | 8 frames/video | 1824/1824 | 7296 | 53.23 | yes |
| `mvueval_qa_internvl_t1iv` | MVU-Eval | InternVL3-8B | per_stream | 8 frames/video | 1824/1824 | 7296 | 43.91 | yes |
| `mvueval_qa_cvbench_t1q25` | MVU-Eval | Qwen2.5-VL-7B-Instruct | centralized | 8 frames/video | 1824/1824 | 7296 | 49.49 | yes |
| `mvueval_qa_cvbench_t1q25` | MVU-Eval | Qwen2.5-VL-7B-Instruct | cvbench_native | 8 frames/video | 1824/1824 | 7296 | 53.63 | yes |
| `mvueval_qa_cvbench_t1q25` | MVU-Eval | Qwen2.5-VL-7B-Instruct | per_stream | 8 frames/video | 1824/1824 | 7296 | 40.04 | yes |
| `crossview_egoexo500_internvl_t1iv` | CrossView-EgoExo | InternVL3-8B | centralized | 8 frames/video | 500/500 | 2000 | 43.5 | yes |
| `crossview_egoexo500_internvl_t1iv` | CrossView-EgoExo | InternVL3-8B | cvbench_native | 8 frames/video | 500/500 | 2000 | 43.25 | yes |
| `crossview_egoexo500_internvl_t1iv` | CrossView-EgoExo | InternVL3-8B | per_stream | 8 frames/video | 500/500 | 2000 | 29.45 | yes |
| `crossview_egoexo500_cvbench_t1q25` | CrossView-EgoExo | Qwen2.5-VL-7B-Instruct | centralized | 8 frames/video | 500/500 | 2000 | 46.15 | yes |
| `crossview_egoexo500_cvbench_t1q25` | CrossView-EgoExo | Qwen2.5-VL-7B-Instruct | cvbench_native | 8 frames/video | 500/500 | 2000 | 45.1 | yes |
| `crossview_egoexo500_cvbench_t1q25` | CrossView-EgoExo | Qwen2.5-VL-7B-Instruct | per_stream | 8 frames/video | 500/500 | 2000 | 28.0 | yes |
| `allangles_qa_internvl_t1iv` | All-Angles | InternVL3-8B | centralized | stills, cell_px=448 | 2132/2132 | 8528 | 49.5 | yes |
| `allangles_qa_internvl_t1iv` | All-Angles | InternVL3-8B | cvbench_native | stills, cell_px=448 | 2132/2132 | 8528 | 51.04 | yes |
| `allangles_qa_internvl_t1iv` | All-Angles | InternVL3-8B | per_stream | stills, cell_px=448 | 2132/2132 | 8528 | 48.55 | yes |
| `allangles_qa_cvbench_t1q25` | All-Angles | Qwen2.5-VL-7B-Instruct | centralized | stills, cell_px=448 | 2132/2132 | 8528 | 47.14 | yes |
| `allangles_qa_cvbench_t1q25` | All-Angles | Qwen2.5-VL-7B-Instruct | cvbench_native | stills, cell_px=448 | 2132/2132 | 8528 | 49.53 | yes |
| `allangles_qa_cvbench_t1q25` | All-Angles | Qwen2.5-VL-7B-Instruct | per_stream | stills, cell_px=448 | 2132/2132 | 8528 | 39.17 | yes |
| `allangles_qa_internvl_t1iv_blind` | All-Angles | InternVL3-8B | blind | no images | 2132/2132 | 8528 | 38.59 | yes |
| `allangles_qa_cvbench_t1q25_blind` | All-Angles | Qwen2.5-VL-7B-Instruct | blind | no images | 2132/2132 | 8528 | 40.57 | yes |
| `mvueval_qa_internvl_t1iv_blind` | MVU-Eval | InternVL3-8B | blind | no images | 1824/1824 | 7296 | 30.99 | yes |
| `mvueval_qa_cvbench_t1q25_blind` | MVU-Eval | Qwen2.5-VL-7B-Instruct | blind | no images | 1824/1824 | 7296 | 33.81 | yes |
| `mvueval_qa_internvl_fs32` | MVU-Eval | InternVL3-8B | cvbench_native | 32 frames total | 1824/1824 | 7296 | 52.6 | yes |
| `mvueval_qa_internvl_fs64` | MVU-Eval | InternVL3-8B | centralized | 64 frames total | 1592/1824 | 6367 | 49.63 | NO |
| `mvueval_qa_internvl_fs64` | MVU-Eval | InternVL3-8B | cvbench_native | 64 frames total | 1824/1824 | 7296 | 52.96 | yes |
| `mvueval_qa_internvl_fs64` | MVU-Eval | InternVL3-8B | per_stream | 64 frames total | 440/1824 | 1756 | 37.07 | NO |
| `mvueval_qa_internvl_fs96` | MVU-Eval | InternVL3-8B | centralized | 96 frames total | 1824/1824 | 7296 | 51.6 | yes |
| `mvueval_qa_internvl_fs96` | MVU-Eval | InternVL3-8B | cvbench_native | 96 frames total | 1824/1824 | 7296 | 51.52 | yes |
| `mvueval_qa_internvl_fs96` | MVU-Eval | InternVL3-8B | per_stream | 96 frames total | 289/1824 | 1146 | 34.64 | NO |
| `crossview_egoexo500_internvl_optu` | CrossView-EgoExo | InternVL3-8B | clip_select_viclip_optu | matched, 8xK frames total | 500/500 | 2000 | 43.2 | yes |
| `crossview_egoexo500_internvl_optu` | CrossView-EgoExo | InternVL3-8B | frame_select_siglip_optu | matched, 8xK frames total | 500/500 | 2000 | 40.75 | yes |
| `crossview_egoexo500_internvl_optu` | CrossView-EgoExo | InternVL3-8B | query_search_siglip | matched, 8xK frames total | 500/500 | 2000 | 43.3 | yes |
| `crossview_egoexo500_cvbench_optu` | CrossView-EgoExo | Qwen2.5-VL-7B-Instruct | clip_select_viclip_optu | matched, 8xK frames total | 500/500 | 2000 | 44.5 | yes |
| `crossview_egoexo500_cvbench_optu` | CrossView-EgoExo | Qwen2.5-VL-7B-Instruct | frame_select_siglip_optu | matched, 8xK frames total | 500/500 | 2000 | 45.1 | yes |
| `crossview_egoexo500_cvbench_optu` | CrossView-EgoExo | Qwen2.5-VL-7B-Instruct | query_search_siglip | matched, 8xK frames total | 500/500 | 2000 | 45.85 | yes |
| `mvueval_qa_internvl_optu` | MVU-Eval | InternVL3-8B | clip_select_viclip_optu | matched, 8xK frames total | 1824/1824 | 7296 | 41.05 | yes |
| `mvueval_qa_internvl_optu` | MVU-Eval | InternVL3-8B | frame_select_siglip_optu | matched, 8xK frames total | 1824/1824 | 7296 | 49.48 | yes |
| `mvueval_qa_internvl_optu` | MVU-Eval | InternVL3-8B | query_search_siglip | matched, 8xK frames total | 1824/1824 | 7296 | 52.97 | yes |
| `mvueval_qa_cvbench_optu` | MVU-Eval | Qwen2.5-VL-7B-Instruct | clip_select_viclip_optu | matched, 8xK frames total | 1824/1824 | 7296 | 39.76 | yes |
| `mvueval_qa_cvbench_optu` | MVU-Eval | Qwen2.5-VL-7B-Instruct | frame_select_siglip_optu | matched, 8xK frames total | 1824/1824 | 7296 | 47.74 | yes |
| `mvueval_qa_cvbench_optu` | MVU-Eval | Qwen2.5-VL-7B-Instruct | query_search_siglip | matched, 8xK frames total | 1824/1824 | 7296 | 50.99 | yes |
| `mvueval_qa_internvl_sg32` | MVU-Eval | InternVL3-8B | segment_select_siglip | 32 frames total | 1824/1824 | 7296 | 52.85 | yes |
| `mvueval_qa_internvl_sg64` | MVU-Eval | InternVL3-8B | segment_select_siglip | 64 frames total | 1824/1824 | 7296 | 53.02 | yes |
| `mvueval_qa_internvl_sg96` | MVU-Eval | InternVL3-8B | segment_select_siglip | 96 frames total | 1824/1824 | 7296 | 51.36 | yes |
| `crossview_egoexo500_internvl_fs32` | CrossView-EgoExo | InternVL3-8B | cvbench_native | 32 frames total | 500/500 | 2000 | 45.45 | yes |
| `crossview_egoexo500_internvl_fs64` | CrossView-EgoExo | InternVL3-8B | cvbench_native | 64 frames total | 500/500 | 2000 | 43.85 | yes |
| `crossview_egoexo500_internvl_fs96` | CrossView-EgoExo | InternVL3-8B | cvbench_native | 96 frames total | 500/500 | 2000 | 43.95 | yes |
| `crossview_egoexo500_internvl_sg32` | CrossView-EgoExo | InternVL3-8B | segment_select_siglip | 32 frames total | 500/500 | 2000 | 38.65 | yes |
| `crossview_egoexo500_internvl_sg64` | CrossView-EgoExo | InternVL3-8B | segment_select_siglip | 64 frames total | 500/500 | 2000 | 40.65 | yes |
| `crossview_egoexo500_internvl_sg96` | CrossView-EgoExo | InternVL3-8B | segment_select_siglip | 96 frames total | 500/500 | 2000 | 40.8 | yes |
| `mvueval_qa_internvl_sgv32` | MVU-Eval | InternVL3-8B | segment_select_viclip_opt | 32 frames total | 1824/1824 | 7296 | 52.28 | yes |
| `mvueval_qa_internvl_sgv64` | MVU-Eval | InternVL3-8B | segment_select_viclip_opt | 64 frames total | 1824/1824 | 7296 | 51.22 | yes |
| `mvueval_qa_internvl_sgv96` | MVU-Eval | InternVL3-8B | segment_select_viclip_opt | 96 frames total | 1824/1824 | 7296 | 49.92 | yes |
| `mvueval_qa_internvl_sgva32` | MVU-Eval | InternVL3-8B | segment_select_viclip_opt | 32 frames total, auto K | 1824/1824 | 7296 | 52.17 | yes |
| `mvueval_qa_internvl_sgva64` | MVU-Eval | InternVL3-8B | segment_select_viclip_opt | 64 frames total, auto K | 1824/1824 | 7296 | 51.84 | yes |
| `mvueval_qa_internvl_sgva96` | MVU-Eval | InternVL3-8B | segment_select_viclip_opt | 96 frames total, auto K | 1824/1824 | 7296 | 49.89 | yes |
| `crossview_egoexo500_internvl_sgva32` | CrossView-EgoExo | InternVL3-8B | segment_select_viclip_opt | 32 frames total, auto K | 500/500 | 2000 | 42.4 | yes |
| `crossview_egoexo500_internvl_sgva64` | CrossView-EgoExo | InternVL3-8B | segment_select_viclip_opt | 64 frames total, auto K | 500/500 | 2000 | 42.25 | yes |
| `crossview_egoexo500_internvl_sgva96` | CrossView-EgoExo | InternVL3-8B | segment_select_viclip_opt | 96 frames total, auto K | 500/500 | 2000 | 43.3 | yes |
| `crossview_meva1033_subset_internvl_mp4fs32` | CrossView-MEVA | InternVL3-8B | cvbench_native | 32 frames total | 1033/1033 | 4132 | 42.88 | yes |
| `crossview_meva1033_subset_internvl_mp4fs64` | CrossView-MEVA | InternVL3-8B | cvbench_native | 64 frames total | 1033/1033 | 4132 | 45.18 | yes |
| `crossview_meva1033_subset_internvl_mp4fs96` | CrossView-MEVA | InternVL3-8B | cvbench_native | 96 frames total | 1033/1033 | 4132 | 43.85 | yes |
| `crossview_meva1033_subset_internvl_mp4sg32` | CrossView-MEVA | InternVL3-8B | segment_select_siglip | 32 frames total | 1033/1033 | 4132 | 44.17 | yes |
| `crossview_meva1033_subset_internvl_mp4sg64` | CrossView-MEVA | InternVL3-8B | segment_select_siglip | 64 frames total | 1033/1033 | 4132 | 47.75 | yes |
| `crossview_meva1033_subset_internvl_mp4sg96` | CrossView-MEVA | InternVL3-8B | segment_select_siglip | 96 frames total | 1033/1033 | 4132 | 49.1 | yes |
| `crossview_meva1033_subset_internvl_mp4sgva32` | CrossView-MEVA | InternVL3-8B | segment_select_viclip_opt | 32 frames total, auto K | 1033/1033 | 4132 | 43.56 | yes |
| `crossview_meva1033_subset_internvl_mp4sgva64` | CrossView-MEVA | InternVL3-8B | segment_select_viclip_opt | 64 frames total, auto K | 1033/1033 | 4132 | 47.97 | yes |
| `crossview_meva1033_subset_internvl_mp4sgva96` | CrossView-MEVA | InternVL3-8B | segment_select_viclip_opt | 96 frames total, auto K | 1033/1033 | 4132 | 48.31 | yes |
| `crossview_meva_cap13_internvl_mp4t1iv` | CrossView-MEVA | InternVL3-8B | centralized | 8 frames/video | 1033/1033 | 4132 | 47.19 | yes |
| `crossview_meva_cap13_internvl_mp4t1iv` | CrossView-MEVA | InternVL3-8B | cvbench_native | 8 frames/video | 1033/1033 | 4132 | 44.92 | yes |
| `crossview_meva_cap13_internvl_mp4t1iv` | CrossView-MEVA | InternVL3-8B | per_stream | 8 frames/video | 1033/1033 | 4132 | 39.06 | yes |
| `crossview_meva_cap13_cvbench_mp4t1q25` | CrossView-MEVA | Qwen2.5-VL-7B-Instruct | centralized | 8 frames/video | 1033/1033 | 4132 | 47.99 | yes |
| `crossview_meva_cap13_cvbench_mp4t1q25` | CrossView-MEVA | Qwen2.5-VL-7B-Instruct | cvbench_native | 8 frames/video | 1033/1033 | 4132 | 41.36 | yes |
| `crossview_meva_cap13_cvbench_mp4t1q25` | CrossView-MEVA | Qwen2.5-VL-7B-Instruct | per_stream | 8 frames/video | 1033/1033 | 4132 | 24.95 | yes |
| `crossview_meva1033_subset_internvl_mp4bd` | CrossView-MEVA | InternVL3-8B | blind | no images | 1033/1033 | 4132 | 34.75 | yes |
| `crossview_meva1033_subset_internvl_mp4sv8` | CrossView-MEVA | InternVL3-8B | single_view1 | 8 frames/view, single view | 1033/1033 | 4132 | 37.73 | yes |
| `crossview_meva1033_subset_internvl_mp4sv8` | CrossView-MEVA | InternVL3-8B | single_view2 | 8 frames/view, single view | 1033/1033 | 4132 | 35.67 | yes |
| `crossview_meva1033_subset_internvl_mp4sv8` | CrossView-MEVA | InternVL3-8B | single_view3 | 8 frames/view, single view | 958/958 | 3832 | 31.73 | yes |
| `crossview_meva1033_subset_internvl_mp4sv8` | CrossView-MEVA | InternVL3-8B | single_view4 | 8 frames/view, single view | 955/955 | 3820 | 29.58 | yes |

Legs marked `NO` stopped before every question ran; their rows are real but the accuracy is over the questions that ran. Do not pool them with a complete twin.

## Excluded: pre-remux MEVA legs

The MEVA release ships `.avi` containers whose packets carry no timestamps; random access through decord lands on keyframe 0 and decodes forward, so every seek returned a frame from the first seconds of the five-minute clip. Every MEVA leg run before 2026-08-27 saw only those seconds. Those legs are NOT in this folder and any MEVA number from an earlier bundle is superseded by the `mp4*` legs here.

| excluded group | subset | backend |
|---|---|---|
| `crossview_meva_cap13_internvl_cvmeva` | crossview_meva_cap13.json | InternVL3-8B |
| `crossview_meva_cap13_cvbench_cvmevaq25` | crossview_meva_cap13.json | Qwen2.5-VL-7B-Instruct |
| `crossview_meva_cap13_internvl_t1iv` | crossview_meva_cap13.json | InternVL3-8B |
| `crossview_meva_cap13_cvbench_t1q25` | crossview_meva_cap13.json | Qwen2.5-VL-7B-Instruct |
| `crossview_meva_cap13_internvl_fs32` | crossview_meva_cap13.json | InternVL3-8B |
| `crossview_meva_cap13_internvl_fs64` | crossview_meva_cap13.json | InternVL3-8B |
| `crossview_meva_cap13_internvl_fs96` | crossview_meva_cap13.json | InternVL3-8B |
| `crossview_meva_cap13_internvl_optu` | crossview_meva_cap13.json | InternVL3-8B |
| `crossview_meva_cap13_cvbench_optu` | crossview_meva_cap13.json | Qwen2.5-VL-7B-Instruct |
| `crossview_meva1033_subset_internvl_fs32` | crossview_meva1033_subset.json | InternVL3-8B |
| `crossview_meva1033_subset_internvl_fs64` | crossview_meva1033_subset.json | InternVL3-8B |
| `crossview_meva1033_subset_internvl_fs96` | crossview_meva1033_subset.json | InternVL3-8B |
| `crossview_meva1033_subset_internvl_sg32` | crossview_meva1033_subset.json | InternVL3-8B |
| `crossview_meva1033_subset_internvl_sg64` | crossview_meva1033_subset.json | InternVL3-8B |
| `crossview_meva1033_subset_internvl_sg96` | crossview_meva1033_subset.json | InternVL3-8B |
| `crossview_meva1033_subset_internvl_sgva32` | crossview_meva1033_subset.json | InternVL3-8B |
| `crossview_meva1033_subset_internvl_sgva64` | crossview_meva1033_subset.json | InternVL3-8B |
| `crossview_meva1033_subset_internvl_sgva96` | crossview_meva1033_subset.json | InternVL3-8B |

## Subsets

- `data/subsets/allangles_egohumans_qa.json`: present, identical
- `data/subsets/allangles_qa.json`: present, identical
- `data/subsets/crossview_egoexo500.json`: present, identical
- `data/subsets/crossview_meva1033_subset.json`: present, identical
- `data/subsets/crossview_meva_cap13.json`: present, identical
- `data/subsets/mvueval_noview_subset.json`: present, identical
- `data/subsets/mvueval_qa.json`: present, identical

## Regenerating

```
# in Wavy-Hec/MultiCam
python3 analysis/export_question_records.py
python3 analysis/export_shared_results.py --out <multicam-harness checkout> --gzip
```
