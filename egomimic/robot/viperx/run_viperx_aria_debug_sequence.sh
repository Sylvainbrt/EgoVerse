#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EGOVERSE_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

ROBOT_DATA_DIR="${ROBOT_DATA_DIR:-/data/sybeuret/remote_data_lerobot/egoverse_data}"
ARIA_DATA_DIR="${ARIA_DATA_DIR:-/data/sybeuret/remote_aria_data/egoverse_data}"
SCALE_CACHE_DIR="${SCALE_CACHE_DIR:-/tmp/scale_zarr_cache}"
SCALE_MANIFEST_PATH="${SCALE_MANIFEST_PATH:-}"
SCALE_MANIFEST_LOCAL_ONLY="${SCALE_MANIFEST_LOCAL_ONLY:-0}"
SCALE_AUTO_EXCLUDE_ACTION_MAX_ABS="${SCALE_AUTO_EXCLUDE_ACTION_MAX_ABS:-100.0}"

LOGGER="${LOGGER:-debug}"
TRAINER="${TRAINER:-debug}"
RUN_NAME="${RUN_NAME:-viperx_aria_debug}"
START_AT="${START_AT:-1}"
DRY_RUN="${DRY_RUN:-0}"

NORM_STAT_MAX_SAMPLES="${NORM_STAT_MAX_SAMPLES:-128}"
SCALE_EPISODES_SMALL="${SCALE_EPISODES_SMALL:-50}"
SCALE_EPISODES_LARGE="${SCALE_EPISODES_LARGE:-200}"

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
  echo "Aria data: ${ARIA_DATA_DIR}"
  echo "Scale cache: ${SCALE_CACHE_DIR}"
  echo "Trainer preset: ${TRAINER}"
  echo "Logger preset: ${LOGGER}"
  echo "Norm stats max samples: ${NORM_STAT_MAX_SAMPLES}"
  echo "Scale auto-exclude max abs: ${SCALE_AUTO_EXCLUDE_ACTION_MAX_ABS}"
  echo "Dry run: ${DRY_RUN}"
  echo "============================================================"
  echo

  local -a cmd=(
    python3
    egomimic/trainHydra.py
    --config-name=train \
    logger="${LOGGER}" \
    trainer="${TRAINER}" \
    name="${RUN_NAME}" \
    description="${description}" \
    norm_stat_max_samples="${NORM_STAT_MAX_SAMPLES}" \
    data.train_datasets.viperx_right_arm.folder_path="${ROBOT_DATA_DIR}" \
    data.valid_datasets.viperx_right_arm.folder_path="${ROBOT_DATA_DIR}" \
    data.train_datasets.aria_left_arm.folder_path="${ARIA_DATA_DIR}" \
    data.valid_datasets.aria_left_arm.folder_path="${ARIA_DATA_DIR}" \
    "$@"
  )

  printf 'Command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY_RUN=1, skipping execution."
    return
  fi

  EGOVERSE_AUTO_EXCLUDE_ACTION_MAX_ABS="${EGOVERSE_AUTO_EXCLUDE_ACTION_MAX_ABS:-${SCALE_AUTO_EXCLUDE_ACTION_MAX_ABS}}" \
    "${cmd[@]}"
}

if [[ "${START_AT}" -le 1 ]]; then
  run_training \
    robot_plus_local_aria_debug \
    data=cotrain_viperx_aria_local \
    model=hpt_cotrain_viperx_aria \
    trainer.strategy=ddp_find_unused_parameters_true
fi

if [[ "${START_AT}" -le 2 ]]; then
  mapfile -t SCALE_OVERRIDES < <(scale_resolver_overrides)
  run_training \
    robot_plus_local_aria_plus_egoverse_cached_50_debug \
    data=cotrain_viperx_aria_scale \
    model=hpt_cotrain_viperx_aria_scale \
    trainer.strategy=ddp_find_unused_parameters_true \
    data.train_datasets.scale_bimanual.resolver.max_episodes="${SCALE_EPISODES_SMALL}" \
    data.valid_datasets.scale_bimanual.resolver.max_episodes="${SCALE_EPISODES_SMALL}" \
    "${SCALE_OVERRIDES[@]}"
fi

if [[ "${START_AT}" -le 3 ]]; then
  mapfile -t SCALE_OVERRIDES < <(scale_resolver_overrides)
  run_training \
    robot_plus_local_aria_plus_egoverse_cached_200_debug \
    data=cotrain_viperx_aria_scale \
    model=hpt_cotrain_viperx_aria_scale \
    trainer.strategy=ddp_find_unused_parameters_true \
    data.train_datasets.scale_bimanual.resolver.max_episodes="${SCALE_EPISODES_LARGE}" \
    data.valid_datasets.scale_bimanual.resolver.max_episodes="${SCALE_EPISODES_LARGE}" \
    "${SCALE_OVERRIDES[@]}"
fi
