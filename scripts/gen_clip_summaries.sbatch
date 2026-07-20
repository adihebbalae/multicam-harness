#!/bin/bash
# Ported from Wavy-Hec/CVBench bench/gen_clip_summaries.sbatch @ 480d6f41cddddc7efea9a09b79134811740ba17a
### Per-clip summary-cache generation for the summary_select_* methods.
### One VLM call per unique clip; resumable; shard with CHUNK + --array.
###
### Usage:
###   # the 357 unique clips of the 130-question temporal subset (~45 min):
###   SUBSET=data/subsets/cvbench_temporal_subset.json sbatch scripts/gen_clip_summaries.sbatch
###   # a full clip manifest (one relative path per line), sharded 4 ways:
###   MANIFEST=<manifest.txt> CHUNK=4 sbatch --array=0-3 \
###     scripts/gen_clip_summaries.sbatch
#SBATCH --job-name=clip-summaries
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH -N 1
#SBATCH -p gpul40q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=08:00:00

set -euo pipefail
REPO=${REPO:-$(cd "$(dirname "$0")/.." && pwd)}
cd "$REPO"
CONDA_SH=${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}
source "$CONDA_SH"
ENV=${ENV:-internvl}
conda activate "$ENV"

export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export DECORD_EOF_RETRY_MAX=${DECORD_EOF_RETRY_MAX:-20480}
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}

BACKEND=${BACKEND:-internvl3}
VIDEO_ROOT=${VIDEO_ROOT:-}   # default comes from configs/datasets.yaml (video_roots.cvbench)
OUT=${OUT:-results/clip_summaries_internvl3.jsonl}
LIMIT=${LIMIT:-0}
CHUNK=${CHUNK:-0}
OFFSET=${SLURM_ARRAY_TASK_ID:-${OFFSET:-0}}

SRC_ARG=""
if [ -n "${SUBSET:-}" ]; then SRC_ARG="--subset $SUBSET"; fi
if [ -n "${MANIFEST:-}" ]; then SRC_ARG="--manifest $MANIFEST"; fi
if [ -z "$SRC_ARG" ]; then echo "set SUBSET= or MANIFEST="; exit 2; fi
LIMIT_ARG=""
if [ "$LIMIT" != "0" ]; then LIMIT_ARG="--limit $LIMIT"; fi
SHARD_ARG=""
if [ "$CHUNK" != "0" ]; then SHARD_ARG="--chunk $CHUNK --offset $OFFSET"; fi
RETRY_ARG=""
if [ -n "${RETRY_ERRORS:-}" ]; then RETRY_ARG="--retry-errors"; fi
VR_ARG=""
if [ -n "$VIDEO_ROOT" ]; then VR_ARG="--video-root $VIDEO_ROOT"; fi

echo "SUMMARIES @ $(date '+%F %T') env=$ENV src=$SRC_ARG out=$OUT chunk=$CHUNK offset=$OFFSET"
python -m scripts.gen_clip_summaries $SRC_ARG --backend "$BACKEND" \
    --out "$OUT" $SHARD_ARG $LIMIT_ARG $RETRY_ARG $VR_ARG
echo "DONE @ $(date '+%F %T')"
