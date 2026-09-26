# MEVA segment-selection legs run on ece-859525

Three complete legs on `crossview_meva1033_subset` (1033 questions × seeds 1–4, T = 0.1, InternVL3-8B, 96 frames, reasoning off, 0 error rows). They use the same row schema as `../legs/`, plus `leg`, `source_repo` and `source_commit`. They are kept in their own folder because `export_shared_results.py` regenerates the top-level README and registry.

| leg | method | selection | acc % | Event-Ordering | Spatial | Temporal |
|---|---|---|--:|--:|--:|--:|
| `crossview_meva1033_subset_internvl_adi_pirule96` | segment_select_viclip_auto | 8-s clips, global top-12, floor 0 (PI rule) | 44.14 | 43.43 | 39.50 | 51.39 |
| `crossview_meva1033_subset_internvl_adi_sgglobal96` | segment_select_siglip | global, seg_floor 1, keep 4 | 47.14 | 42.26 | 43.21 | 57.46 |
| `crossview_meva1033_subset_internvl_adi_sgperclip96` | segment_select_siglip | per_clip, keep 4 | 48.23 | 46.21 | 43.50 | 56.89 |

- **Media.** The legs decode the `.mp4` siblings on `/nas/mars`, which are a **640×358 transcode** of the release `.avi` files, not the full-resolution remux used for the `mp4*` legs in `../legs/`. Segment picks differ between the two, so these rows do not pair question by question with those legs.
- **Seeds are not independent.** At T = 0.1 the four seeds see the same frames. Use question-level tests, not McNemar on (id, seed) pairs.
- **Code.** `pirule96` ran at `66b0b29` (pi-rule-rebased, functionally identical to PR #18 as merged). The two SigLIP legs ran at `b8e9b26` plus the scorer-on-second-GPU patch, later committed as `a2262a6`.
- The exporter is `scripts/export_meva_legs_to_harness.py` in `adihebbalae/distributed-multicam`. Per-leg details are in `registry.json` and `summary.csv`.

```python
import gzip, json
rows = [json.loads(l) for l in gzip.open('multicam_results/adi/legs/crossview_meva1033_subset_internvl_adi_pirule96.jsonl.gz', 'rt')]
```
