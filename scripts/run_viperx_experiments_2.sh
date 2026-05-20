#!/usr/bin/env bash

# Sequential ViperX adaptation experiments.
# Starts from experiment 2 (ARIA local).

set -uo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN=(python)
if [[ "${CONDA_DEFAULT_ENV:-}" != "egoverse" ]]; then
  PYTHON_BIN=(conda run -n egoverse python)
fi

RUN_ARIA="${RUN_ARIA:-1}"
RUN_CACHED="${RUN_CACHED:-1}"
RUN_STREAMING="${RUN_STREAMING:-1}"

SCALE_EPISODES="${SCALE_EPISODES:-250}"
CACHE_DIR="${CACHE_DIR:-/data/sybeuret/scale_zarr_cache}"

# Optional pause between trainings (seconds)
SLEEP_BETWEEN_RUNS="${SLEEP_BETWEEN_RUNS:-30}"

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
  echo "=================================================="
  echo "Starting ${label} at $(date -Is)"
  echo "=================================================="

  "${PYTHON_BIN[@]}" "${COMMON[@]}" "$@"
  EXIT_CODE=$?

  if [[ $EXIT_CODE -eq 0 ]]; then
    echo "✅ Finished ${label} successfully at $(date -Is)"
  else
    echo "❌ ${label} FAILED with exit code ${EXIT_CODE} at $(date -Is)"
  fi

  echo "Sleeping ${SLEEP_BETWEEN_RUNS}s before next run..."
  sleep "${SLEEP_BETWEEN_RUNS}"

  return 0
}

########################################################
# 2) ARIA LOCAL
########################################################
if [[ "${RUN_ARIA}" == "1" ]]; then
  run "viperx_aria_local" \
    name=viperx_ablation \
    description=aria_local \
    data=cotrain_viperx_aria_local \
    model=hpt_cotrain_viperx_aria
fi

########################################################
# 3) SCALE CACHED
########################################################
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

########################################################
# 4) SCALE STREAMING
########################################################
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

echo
echo "All requested experiments completed."
