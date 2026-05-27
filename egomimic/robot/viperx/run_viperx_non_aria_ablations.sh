#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EGOVERSE_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

ROBOT_DATA_DIR="${ROBOT_DATA_DIR:-/data/sybeuret/remote_data_lerobot/egoverse_data}"
SCALE_CACHE_DIR="${SCALE_CACHE_DIR:-/tmp/scale_zarr_cache}"
SCALE_MANIFEST_PATH="${SCALE_MANIFEST_PATH:-}"
SCALE_MANIFEST_LOCAL_ONLY="${SCALE_MANIFEST_LOCAL_ONLY:-0}"
WANDB_LOGGER="${WANDB_LOGGER:-wandb}"
TRAINER="${TRAINER:-ddp}"
RUN_NAME="${RUN_NAME:-viperx_ablation}"
START_AT="${START_AT:-1}"

mkdir -p "${SCALE_CACHE_DIR}"

cd "${EGOVERSE_ROOT}"

scale_resolver_overrides() {
  if [[ -z "${SCALE_MANIFEST_PATH}" ]]; then
    printf '%s\n' \
      "data.train_datasets.scale_bimanual.resolver.folder_path=${SCALE_CACHE_DIR}" \
      "data.valid_datasets.scale_bimanual.resolver.folder_path=${SCALE_CACHE_DIR}"
    return
  fi

  printf '%s\n' \
    "data.train_datasets.scale_bimanual.resolver._target_=egomimic.rldb.zarr.zarr_dataset_multi.ManifestEpisodeResolver" \
    "data.valid_datasets.scale_bimanual.resolver._target_=egomimic.rldb.zarr.zarr_dataset_multi.ManifestEpisodeResolver" \
    "data.train_datasets.scale_bimanual.resolver.folder_path=${SCALE_CACHE_DIR}" \
    "data.valid_datasets.scale_bimanual.resolver.folder_path=${SCALE_CACHE_DIR}" \
    "+data.train_datasets.scale_bimanual.resolver.manifest_path=${SCALE_MANIFEST_PATH}" \
    "+data.valid_datasets.scale_bimanual.resolver.manifest_path=${SCALE_MANIFEST_PATH}" \
    "+data.train_datasets.scale_bimanual.resolver.sync_missing=$( [[ "${SCALE_MANIFEST_LOCAL_ONLY}" == "1" ]] && echo false || echo true )" \
    "+data.valid_datasets.scale_bimanual.resolver.sync_missing=$( [[ "${SCALE_MANIFEST_LOCAL_ONLY}" == "1" ]] && echo false || echo true )"
}

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

if [[ "${START_AT}" -le 1 ]]; then
  run_training \
    robot_only_local \
    data=viperx_local \
    model=hpt_bc_flow_viperx
fi

if [[ "${START_AT}" -le 2 ]]; then
  mapfile -t SCALE_OVERRIDES < <(scale_resolver_overrides)
  run_training \
    robot_plus_egoverse_cached_500 \
    data=cotrain_viperx_scale \
    model=hpt_cotrain_viperx_scale \
    trainer.strategy=ddp_find_unused_parameters_true \
    data.train_datasets.scale_bimanual.resolver.max_episodes=500 \
    data.valid_datasets.scale_bimanual.resolver.max_episodes=500 \
    "${SCALE_OVERRIDES[@]}"
fi

if [[ "${START_AT}" -le 3 ]]; then
  mapfile -t SCALE_OVERRIDES < <(scale_resolver_overrides)
  run_training \
    robot_plus_egoverse_cached_2000 \
    data=cotrain_viperx_scale \
    model=hpt_cotrain_viperx_scale \
    trainer.strategy=ddp_find_unused_parameters_true \
    data.train_datasets.scale_bimanual.resolver.max_episodes=2000 \
    data.valid_datasets.scale_bimanual.resolver.max_episodes=2000 \
    "${SCALE_OVERRIDES[@]}"
fi
