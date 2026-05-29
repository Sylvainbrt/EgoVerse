#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EGOVERSE_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

ROBOT_DATA_DIR="${ROBOT_DATA_DIR:-/data/sybeuret/remote_data_lerobot/egoverse_data}"
ARIA_DATA_DIR="${ARIA_DATA_DIR:-/data/sybeuret/aria_gen2_data/egoverse_data}"
SCALE_CACHE_DIR="${SCALE_CACHE_DIR:-/tmp/scale_zarr_cache}"
SCALE_MANIFEST_PATH="${SCALE_MANIFEST_PATH:-/data/sybeuret/scale_2000_manifest.json}"
SCALE_MANIFEST_LOCAL_ONLY="${SCALE_MANIFEST_LOCAL_ONLY:-0}"
SCALE_AUTO_EXCLUDE_ACTION_MAX_ABS="${SCALE_AUTO_EXCLUDE_ACTION_MAX_ABS:-100.0}"

LOGGER="${LOGGER:-wandb}"
TRAINER="${TRAINER:-ddp}"
RUN_NAME="${RUN_NAME:-viperx_aria_ablation}"
START_AT="${START_AT:-1}"
END_AT="${END_AT:-3}"
DRY_RUN="${DRY_RUN:-0}"

SCALE_EPISODES_SMALL="${SCALE_EPISODES_SMALL:-500}"
SCALE_EPISODES_LARGE="${SCALE_EPISODES_LARGE:-2000}"

mkdir -p "${SCALE_CACHE_DIR}"

cd "${EGOVERSE_ROOT}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

in_range() {
  local step="$1"
  [[ "$step" -ge "$START_AT" && "$step" -le "$END_AT" ]]
}

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
  echo "Scale auto-exclude max abs: ${SCALE_AUTO_EXCLUDE_ACTION_MAX_ABS}"
  echo "Dry run: ${DRY_RUN}"
echo "============================================================"
  echo

  [[ "${START_AT}" =~ ^[0-9]+$ ]] || die "START_AT must be an integer"
  [[ "${END_AT}" =~ ^[0-9]+$ ]] || die "END_AT must be an integer"
  [[ "${START_AT}" -le "${END_AT}" ]] || die "START_AT must be <= END_AT"

  local -a cmd=(
    python3
    egomimic/trainHydra.py
    --config-name=train
    logger="${LOGGER}"
    trainer="${TRAINER}"
    name="${RUN_NAME}"
    description="${description}"
    data.train_datasets.viperx_right_arm.folder_path="${ROBOT_DATA_DIR}"
    data.valid_datasets.viperx_right_arm.folder_path="${ROBOT_DATA_DIR}"
    data.train_datasets.aria_left_arm.folder_path="${ARIA_DATA_DIR}"
    data.valid_datasets.aria_left_arm.folder_path="${ARIA_DATA_DIR}"
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

if in_range 1; then
  run_training \
    robot_plus_local_aria \
    data=cotrain_viperx_aria_local \
    model=hpt_cotrain_viperx_aria \
    trainer.strategy=ddp_find_unused_parameters_true
fi

if in_range 2; then
  mapfile -t SCALE_OVERRIDES < <(scale_resolver_overrides)
  run_training \
    robot_plus_local_aria_plus_egoverse_cached_500 \
    data=cotrain_viperx_aria_scale \
    model=hpt_cotrain_viperx_aria_scale \
    trainer.strategy=ddp_find_unused_parameters_true \
    data.train_datasets.scale_bimanual.resolver.max_episodes="${SCALE_EPISODES_SMALL}" \
    data.valid_datasets.scale_bimanual.resolver.max_episodes="${SCALE_EPISODES_SMALL}" \
    "${SCALE_OVERRIDES[@]}"
fi

if in_range 3; then
  mapfile -t SCALE_OVERRIDES < <(scale_resolver_overrides)
  run_training \
    robot_plus_local_aria_plus_egoverse_cached_2000 \
    data=cotrain_viperx_aria_scale \
    model=hpt_cotrain_viperx_aria_scale \
    trainer.strategy=ddp_find_unused_parameters_true \
    data.train_datasets.scale_bimanual.resolver.max_episodes="${SCALE_EPISODES_LARGE}" \
    data.valid_datasets.scale_bimanual.resolver.max_episodes="${SCALE_EPISODES_LARGE}" \
    "${SCALE_OVERRIDES[@]}"
fi
