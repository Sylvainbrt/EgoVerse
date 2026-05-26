#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EGOVERSE_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

ROBOT_DATA_DIR="${ROBOT_DATA_DIR:-/data/sybeuret/remote_data_lerobot/egoverse_data}"
SCALE_CACHE_DIR="${SCALE_CACHE_DIR:-/tmp/scale_zarr_cache}"
WANDB_LOGGER="${WANDB_LOGGER:-wandb}"
TRAINER="${TRAINER:-ddp}"
RUN_NAME="${RUN_NAME:-viperx_ablation}"

mkdir -p "${SCALE_CACHE_DIR}"

cd "${EGOVERSE_ROOT}"

run_training() {
  local description="$1"
  shift

  echo
  echo "============================================================"
  echo "Starting training: ${description}"
  echo "EgoVerse root: ${EGOVERSE_ROOT}"
  echo "Robot data: ${ROBOT_DATA_DIR}"
  echo "Scale cache: ${SCALE_CACHE_DIR}"
  echo "============================================================"
  echo

  python3 egomimic/trainHydra.py \
    --config-name=train \
    logger="${WANDB_LOGGER}" \
    trainer="${TRAINER}" \
    name="${RUN_NAME}" \
    description="${description}" \
    data.train_datasets.viperx_right_arm.folder_path="${ROBOT_DATA_DIR}" \
    data.valid_datasets.viperx_right_arm.folder_path="${ROBOT_DATA_DIR}" \
    "$@"
}

run_training \
  robot_only_local \
  data=viperx_local \
  model=hpt_bc_flow_viperx

run_training \
  robot_plus_egoverse_cached_500 \
  data=cotrain_viperx_scale \
  model=hpt_cotrain_viperx_scale \
  trainer.strategy=ddp_find_unused_parameters_true \
  data.train_datasets.scale_bimanual.resolver.folder_path="${SCALE_CACHE_DIR}" \
  data.valid_datasets.scale_bimanual.resolver.folder_path="${SCALE_CACHE_DIR}" \
  data.train_datasets.scale_bimanual.resolver.max_episodes=500 \
  data.valid_datasets.scale_bimanual.resolver.max_episodes=500

run_training \
  robot_plus_egoverse_cached_2000 \
  data=cotrain_viperx_scale \
  model=hpt_cotrain_viperx_scale \
  trainer.strategy=ddp_find_unused_parameters_true \
  data.train_datasets.scale_bimanual.resolver.folder_path="${SCALE_CACHE_DIR}" \
  data.valid_datasets.scale_bimanual.resolver.folder_path="${SCALE_CACHE_DIR}" \
  data.train_datasets.scale_bimanual.resolver.max_episodes=2000 \
  data.valid_datasets.scale_bimanual.resolver.max_episodes=2000
