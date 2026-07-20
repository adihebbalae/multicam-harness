#!/bin/bash
# Ported from Wavy-Hec/CVBench bench/run_bench.sbatch @ 480d6f41cddddc7efea9a09b79134811740ba17a
### Multi-camera benchmark sweep (2 models x 2 harnesses x 4 passes).
### Supports Slurm-array sharding (OFFSET=$SLURM_ARRAY_TASK_ID, CHUNK=array size)
### and an env split: Qwen legs run under `cvbench`, InternVL3 under `internvl`
### (cvbench's transformers breaks the InternVL3 remote code).
###
### Usage:
###   # tiny smoke (Qwen, 5 records):
###   LIMIT=5 BACKENDS=qwen3vl sbatch scripts/run_bench.sbatch
###   # full meva1033 Qwen leg, sharded 8 ways:
###   ENV=cvbench SUBSET=data/subsets/crossview_meva1033_subset.json BACKENDS=qwen3vl \
###     CHUNK=8 sbatch --array=0-7 scripts/run_bench.sbatch
###   # full meva1033 InternVL3 leg, sharded 8 ways:
###   ENV=internvl SUBSET=data/subsets/crossview_meva1033_subset.json BACKENDS=internvl3 \
###     CHUNK=8 sbatch --array=0-7 scripts/run_bench.sbatch
#SBATCH --job-name=bench-multicam
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
ENV=${ENV:-cvbench}
conda activate "$ENV"

export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export DECORD_EOF_RETRY_MAX=${DECORD_EOF_RETRY_MAX:-20480}
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}

SUBSET=${SUBSET:-data/subsets/crossview_combined_subset.json}
METHODS=${METHODS:-centralized,per_stream}
BACKENDS=${BACKENDS:-qwen3vl}
LIMIT=${LIMIT:-0}
NFRAMES=${NFRAMES:-8}
PASSES=${PASSES:-4}
SEEDS=${SEEDS:-1,2,3,4}
TEMPERATURE=${TEMPERATURE:-0.7}
MONTAGE_FRAMES=${MONTAGE_FRAMES:-0}
MONTAGE_KIND=${MONTAGE_KIND:-camera}   # 'camera' (synced views) | 'video' (independent clips)
STREAM_KIND=${STREAM_KIND:-camera}     # per_stream labels: 'camera' (MEVA) | 'video' (CVBench)
BUDGET=${BUDGET:-64}                   # temporal_weighted: TOTAL frames per question
FLOOR=${FLOOR:-2}                      # temporal_weighted: per-clip minimum frames
WEIGHTING=${WEIGHTING:-duration}       # temporal_weighted: 'duration' (weighted) | 'even' (control)
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-8192}
INTERNVL_MAX_TILES=${INTERNVL_MAX_TILES:-1}
VIDEO_ROOT=${VIDEO_ROOT:-}   # override for non-MEVA sets (e.g. CVBench videos live elsewhere)
CHUNK=${CHUNK:-0}
# shard index: Slurm array task id if present, else OFFSET (default 0)
OFFSET=${SLURM_ARRAY_TASK_ID:-${OFFSET:-0}}

TAG=${TAG:-}   # optional suffix to separate runs that share subset+env (e.g. _dur/_even/_stitch)
BASE="results/bench_$(basename "${SUBSET%.json}")_${ENV}${TAG}"
if [ "$CHUNK" != "0" ]; then
  OUT=${OUT:-${BASE}_shard${OFFSET}.jsonl}
  SHARD_ARG="--chunk $CHUNK --offset $OFFSET"
else
  OUT=${OUT:-${BASE}.jsonl}
  SHARD_ARG=""
fi
LIMIT_ARG=""
if [ "$LIMIT" != "0" ]; then LIMIT_ARG="--limit $LIMIT"; fi
VR_ARG=""
if [ -n "$VIDEO_ROOT" ]; then VR_ARG="--video-root $VIDEO_ROOT"; fi
SUMMARIES=${SUMMARIES:-}   # summary_select_*: per-clip summary cache JSONL (or glob)
SUM_ARG=()                 # array + quoted expansion so a glob reaches python intact
if [ -n "$SUMMARIES" ]; then SUM_ARG=(--summaries "$SUMMARIES"); fi

echo "RUN @ $(date '+%F %T') env=$ENV subset=$SUBSET methods=$METHODS backends=$BACKENDS"
echo "     passes=$PASSES seeds=$SEEDS temp=$TEMPERATURE budget=$BUDGET weighting=$WEIGHTING chunk=$CHUNK offset=$OFFSET out=$OUT video_root=${VIDEO_ROOT:-<default>}"
python run_vqa.py --subset "$SUBSET" --methods "$METHODS" --backends "$BACKENDS" \
    --nframes "$NFRAMES" --passes "$PASSES" --seeds "$SEEDS" --temperature "$TEMPERATURE" \
    --montage-frames "$MONTAGE_FRAMES" --montage-kind "$MONTAGE_KIND" --stream-kind "$STREAM_KIND" \
    --budget "$BUDGET" --floor "$FLOOR" --weighting "$WEIGHTING" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --internvl-max-tiles "$INTERNVL_MAX_TILES" --out "$OUT" $SHARD_ARG $LIMIT_ARG $VR_ARG "${SUM_ARG[@]}"
echo "DONE @ $(date '+%F %T')"
