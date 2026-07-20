# Port Contract — CVBench `bench/` → `multicam-harness`

This repo is a faithful extraction of the multi-camera harness code from
`Wavy-Hec/CVBench` (private experiments fork), branch `crossview-benchmark`
@ **480d6f41cddddc7efea9a09b79134811740ba17a** (the "Phase-A SHA").
Every rule below is binding for the initial port and for future syncs.

## 1. Import rewrite table (mechanical — no judgment)

| Fork import | New import |
|---|---|
| `bench.methods.base` → `Method`, `Result`, `result_fields` | `harnesses.base` |
| `bench.methods.base` → `Backend`, `GenOut` | `models.clients` |
| `bench.reuse` → `build_messages`, `video_paths`, `num_videos`, `QUESTION_TEMPLATE`, `is_yesno` | `dataloaders.qa_json` (bodies vendored from `Video-R1/src/eval_thinking.py`) |
| `bench.reuse` → `parse_choice`, `gt_choice`, `extract_think`, `extract_answer` | `evaluation.scoring` (bodies vendored from `eval_thinking.py`) |
| `bench.reuse` → `load_model` | `models.clients` (vendored from `eval_thinking.py`) |
| `bench.reuse.DEFAULT_VIDEO_ROOT` | `configs/datasets.yaml` `video_roots.crossview`; surfaced only as the `run_vqa.py --video-root` default |
| `bench.methods.temporal` → `PREFIX`, `MARKER`, `SPLIT_DESC`, `allocate_frames`, method classes | `harnesses.uniform` |
| `bench.methods.stitch.sample_frame_indices` (+ any frame helper used by ≥2 modules) | `dataloaders.video` |
| `bench.backends.qwen` / `bench.backends.internvl` | `models.clients` — the InternVL import **must stay lazy** (inside the factory), exactly as in fork `run_bench.py` |

Modules are **top-level** (run everything from the repo root); imports are absolute
(`from harnesses.uniform import ...`). No package installs; no relative `..` imports.

## 2. Byte-faithfulness rule

Every ported function/class body must be **byte-identical** to its fork source except:
- import lines,
- the module docstring and the provenance header,
- path/config constants replaced per §4 (only at CLI/argparse boundaries — never inside
  method/backend logic),
- nothing else. No refactors, no renames, no reformatting, no "improvements", no added
  type hints, no reordered functions. Class and function names are unchanged; only
  module paths move. The equivalence verifier diffs with imports/docstrings stripped
  and FAILS on any other delta.

## 3. Provenance header rule

First comment line of every ported file:
`# Ported from Wavy-Hec/CVBench <fork-relative path> @ 480d0f4` — use the full form:
`# Ported from Wavy-Hec/CVBench bench/methods/temporal.py @ 480d6f41cddddc7efea9a09b79134811740ba17a`
(one line per source file if a target merges several). This is the drift anchor:
when the fork's source file changes, re-port and bump the SHA.

## 4. Parameterization rule (no absolute paths anywhere)

- Cluster/user paths → `configs/datasets.yaml` (video roots, subset paths, summary
  cache) — read by `run_vqa.py` (and scripts) to fill argparse defaults; harness and
  model code keeps taking these as ordinary parameters.
- sbatch/shell: `REPO=${REPO:-$(cd "$(dirname "$0")/.." && pwd)}`,
  `CONDA_SH=${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}`, logs to
  `logs/%A_%a.out` relative to the submit dir. `#SBATCH -p gpul40q` may remain as a
  default with a README note to override via `sbatch -p`.
- HF cache/offline env vars are pass-through only (never hardcode a cache path).

## 5. Forbidden strings (leak scan fails the build on any hit)

No file in this repo may contain: the original author's cluster username or home
directory path, NAS mount paths, collaborators' first names, the upstream fork
owner's account name, absolute conda *env* paths, or the fork's internal
meeting/status/mentor note filenames. Scan with self-non-matching patterns
(bracketed so this doc itself stays clean):

```
grep -rniE 'hector[l]ugo|/na[s]/|hokh[i]m|anaconda3/env[s]|hector_meet[i]ng|multicam_project_stat[u]s|mento[r]_' . --exclude-dir=.git
grep -rniE '(^|[^a-zA-Z])ad[i]([^a-zA-Z]|$)' . --exclude-dir=.git   # collaborator first name; ignore hits inside ordinary words
```

Allowed exception: the `$HOME/anaconda3/etc/profile.d/conda.sh` default in shell
scripts (`etc`, not `envs`). `envs/*.yml` must have no `prefix:` line.

## 6. Commit rules

Author `Wavy-Hec <hlugo576@gmail.com>`. **No co-author trailers of any kind.**
One commit per module, message prefixed with the module name
(e.g. `harnesses: port uniform/stitched/decentralized/clip_select from CVBench @ 480d6f4`).

## 7. Sync plan with the sibling repo

`dataloaders/qa_json.py` and `evaluation/scoring.py` are frozen copies of
`Video-R1/src/eval_thinking.py` @ the Phase-A SHA. **This repo is now canonical**
for those functions; the fork consumes upstream drift only via deliberate re-ports.
`tests/compare_prompts_vs_fork.py` (gated on `CVBENCH_FORK=<fork path>`) doubles as
the drift detector — run it whenever the fork's `eval_thinking.py` or `bench/` changes.
