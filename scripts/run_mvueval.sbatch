#!/bin/bash
# New file (not a port): a Slurm wrapper for `python -m inprocess.run` on
# MVU-Eval. Nothing here is ported, and it modifies nothing that already exists.
### MVU-Eval legs for the inprocess arms, with Slurm-array sharding
### (OFFSET=$SLURM_ARRAY_TASK_ID, CHUNK=array width) and the two-env split:
### Qwen runs under `cvbench`, InternVL3 under `internvl` (cvbench's newer
### transformers breaks the InternVL3 remote code).
###
### No partition is declared -- pass `-p <yours>`. The walltime, gres and
### cpus-per-task below are one cluster's defaults; override with
### `sbatch -t <yours> --gres=<yours>`. Run `mkdir -p logs` before the first
### submit: Slurm opens --output before the script body runs, and logs/ is
### gitignored so a fresh clone has none.
###
### Usage:
###   # smoke: 5 questions, one pass, the sequential baseline only
###   LIMIT=5 PASSES=1 SEEDS=1 METHODS=cvbench_native sbatch scripts/run_mvueval.sbatch
###   # dev subset, Qwen, the three option-guided arms, sharded 4 ways
###   SUBSET=data/subsets/mvueval_dev_subset.json CHUNK=4 \
###     sbatch --array=0-3 scripts/run_mvueval.sbatch
###   # full pool, InternVL3
###   ENV=internvl MODEL=OpenGVLab/InternVL3-8B \
###     SUBSET=data/subsets/mvueval_qa.json CHUNK=8 \
###     sbatch --array=0-7 scripts/run_mvueval.sbatch
#SBATCH --job-name=mvueval
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=24:00:00

set -euo pipefail
REPO=${REPO:-$(cd "$(dirname "$0")/.." && pwd)}
cd "$REPO"
mkdir -p logs results
# Activate whatever the site uses, or nothing if the caller already did. Neither
# path is assumed: VENV for the venv the README documents, ENV+CONDA_SH for conda.
set +u
if [ -n "${VENV:-}" ]; then source "$VENV/bin/activate"
elif [ -n "${ENV:-}" ]; then source "${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"; conda activate "$ENV"
fi
set -u

export DECORD_EOF_RETRY_MAX=${DECORD_EOF_RETRY_MAX:-20480}
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}

# MVU-Eval media does NOT live under any default root -- it is wherever
# scripts/data/fetch_mvueval_videos.py put it. Pass VIDEO_ROOT if you moved it.
SUBSET=${SUBSET:-data/subsets/mvueval_dev_subset.json}
VIDEO_ROOT=${VIDEO_ROOT:-data/mvueval}
MODEL=${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}
METHODS=${METHODS:-frame_select_siglip_optu,clip_select_viclip_optu,query_search_siglip}
LIMIT=${LIMIT:-0}
NFRAMES=${NFRAMES:-8}
PASSES=${PASSES:-4}
SEEDS=${SEEDS:-1,2,3,4}
TEMPERATURE=${TEMPERATURE:-0.1}
REASONING=${REASONING:-0}
# Rewrites nearly every prompt of a multi-option subset, so rows made with 0 and
# rows made with 1 are NOT poolable. run.py makes it required with no default so
# nobody picks a side by omission; this wrapper propagates that requirement.
STRICT=${STRICT:?set STRICT=0 or 1 -- rows made with different values are not poolable}
CHUNK=${CHUNK:-1}                       # NUMBER OF SHARDS, not shard size
OFFSET=${OFFSET:-${SLURM_ARRAY_TASK_ID:-0}}

ARGS=(--subset "$SUBSET" --video-root "$VIDEO_ROOT" --model "$MODEL"
      --methods "$METHODS" --strict-answer-prompt "$STRICT"
      --nframes "$NFRAMES" --passes "$PASSES" --seeds "$SEEDS"
      --temperature "$TEMPERATURE" --chunk "$CHUNK" --offset "$OFFSET")
# NB: `[ cond ] && ARGS+=(...)` would abort the whole script under `set -e`
# whenever cond is false, so these stay full if-statements.
if [ "$REASONING" = "0" ]; then ARGS+=(--no-reasoning); fi
if [ "$LIMIT" != "0" ]; then ARGS+=(--limit "$LIMIT"); fi
# InternVL tiles an image but not a video frame, so the selection arms refuse to
# construct unless tiling is off -- otherwise "same frame budget" is a
# tokenization artifact rather than a comparison.
case "$MODEL" in *InternVL*|*internvl*) ARGS+=(--internvl-max-tiles 1);; esac

echo "RUN ${ARGS[*]}"
python -m inprocess.run "${ARGS[@]}"
