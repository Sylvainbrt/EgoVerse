#!/usr/bin/env bash
set -euo pipefail

# Sequential ViperX adaptation experiments.
#
# Usage:
#   bash scripts/run_viperx_experiments.sh
#
# Optional:
#   RUN_LOCAL=0 bash scripts/run_viperx_experiments.sh
#   RUN_ARIA=0 bash scripts/run_viperx_experiments.sh
#   RUN_CACHED=0 bash scripts/run_viperx_experiments.sh
#   RUN_STREAMING=0 bash scripts/run_viperx_experiments.sh
#   SCALE_EPISODES=250 bash scripts/run_viperx_experiments.sh

cd "$(dirname "$0")/.."

PYTHON_BIN=(python)
if [[ "${CONDA_DEFAULT_ENV:-}" != "egoverse" ]]; then
  PYTHON_BIN=(conda run -n egoverse python)
fi

RUN_LOCAL="${RUN_LOCAL:-1}"
RUN_ARIA="${RUN_ARIA:-1}"
RUN_CACHED="${RUN_CACHED:-1}"
RUN_STREAMING="${RUN_STREAMING:-1}"
SCALE_EPISODES="${SCALE_EPISODES:-250}"
CACHE_DIR="${CACHE_DIR:-/data/sybeuret/scale_zarr_cache}"

COMMON=(
  egomimic/trainHydra.py
  --config-name=train
  logger=wandb
  trainer=ddp
  trainer.strategy=ddp_find_unused_parameters_true
)

run() {
  local label="$1"
  shift
  echo
  echo "===== Starting ${label} at $(date -Is) ====="
  "${PYTHON_BIN[@]}" "${COMMON[@]}" "$@"
  echo "===== Finished ${label} at $(date -Is) ====="
}

if [[ "${RUN_LOCAL}" == "1" ]]; then
  run "viperx_local" \
    name=viperx_ablation \
    description=local_only \
    data=viperx_local \
    model=hpt_bc_flow_viperx
fi

if [[ "${RUN_ARIA}" == "1" ]]; then
  run "viperx_aria_local" \
    name=viperx_ablation \
    description=aria_local \
    data=cotrain_viperx_aria_local \
    model=hpt_cotrain_viperx_aria
fi

if [[ "${RUN_CACHED}" == "1" ]]; then
  run "viperx_scale_cached_${SCALE_EPISODES}" \
    name=viperx_ablation \
    description=scale_cached_${SCALE_EPISODES} \
    data=cotrain_viperx_scale \
    model=hpt_cotrain_viperx_scale \
    data.train_datasets.scale_bimanual.resolver._target_=egomimic.rldb.zarr.zarr_dataset_multi.S3EpisodeResolver \
    data.valid_datasets.scale_bimanual.resolver._target_=egomimic.rldb.zarr.zarr_dataset_multi.S3EpisodeResolver \
    data.train_datasets.scale_bimanual.resolver.folder_path="${CACHE_DIR}" \
    data.valid_datasets.scale_bimanual.resolver.folder_path="${CACHE_DIR}" \
    data.train_datasets.scale_bimanual.resolver.max_episodes="${SCALE_EPISODES}" \
    data.valid_datasets.scale_bimanual.resolver.max_episodes="${SCALE_EPISODES}"
fi

if [[ "${RUN_STREAMING}" == "1" ]]; then
  run "viperx_scale_streaming_${SCALE_EPISODES}" \
    name=viperx_ablation \
    description=scale_streaming_${SCALE_EPISODES} \
    data=cotrain_viperx_scale \
    model=hpt_cotrain_viperx_scale \
    data.train_datasets.scale_bimanual.resolver._target_=egomimic.rldb.zarr.zarr_dataset_multi.S3StreamingEpisodeResolver \
    data.valid_datasets.scale_bimanual.resolver._target_=egomimic.rldb.zarr.zarr_dataset_multi.S3StreamingEpisodeResolver \
    data.train_datasets.scale_bimanual.resolver.folder_path=/data/sybeuret/tmp/egoverse_unused \
    data.valid_datasets.scale_bimanual.resolver.folder_path=/data/sybeuret/tmp/egoverse_unused \
    data.train_datasets.scale_bimanual.resolver.max_episodes="${SCALE_EPISODES}" \
    data.valid_datasets.scale_bimanual.resolver.max_episodes="${SCALE_EPISODES}"
fi
