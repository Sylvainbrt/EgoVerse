#!/usr/bin/env bash
set -euo pipefail

# Sequential two-run ViperX sweep:
#   1. robot-only local LeRobot data
#   2. robot + local Aria data with scheduled mixing
#
# Usage:
#   bash scripts/run_viperx_2way_sequential.sh
#
# Useful overrides:
#   VIPERX_DATA_ROOT=/path/to/egoverse_data bash scripts/run_viperx_2way_sequential.sh
#   ARIA_DATA_ROOT=/path/to/aria_egoverse_data bash scripts/run_viperx_2way_sequential.sh
#   RUN_ARIA=0 bash scripts/run_viperx_2way_sequential.sh

cd "$(dirname "$0")/.."

PYTHON_BIN=(python)
if [[ "${CONDA_DEFAULT_ENV:-}" != "egoverse" ]]; then
  PYTHON_BIN=(conda run -n egoverse python)
fi

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export EGOVERSE_SKIP_VALIDATION_VIZ="${EGOVERSE_SKIP_VALIDATION_VIZ:-1}"

VIPERX_DATA_ROOT="${VIPERX_DATA_ROOT:-/data/sybeuret/.local/huggingface/lerobot/lerobot/egoverse_data}"
ARIA_DATA_ROOT="${ARIA_DATA_ROOT:-/data/sybeuret/remote_aria_data/egoverse_data}"
SLEEP_BETWEEN_RUNS="${SLEEP_BETWEEN_RUNS:-30}"

RUN_ROBOT_ONLY="${RUN_ROBOT_ONLY:-1}"
RUN_ARIA="${RUN_ARIA:-1}"

COMMON=(
  egomimic/trainHydra.py
  --config-name=train
  logger=wandb
  trainer=single_gpu
  name=viperx_2way
  data.train_datasets.viperx_right_arm.folder_path="${VIPERX_DATA_ROOT}"
  data.valid_datasets.viperx_right_arm.folder_path="${VIPERX_DATA_ROOT}"
)

run() {
  local label="$1"
  shift

  echo
  echo "=================================================="
  echo "Starting ${label} at $(date -Is)"
  echo "=================================================="

  "${PYTHON_BIN[@]}" "${COMMON[@]}" "$@"

  echo "=================================================="
  echo "Finished ${label} at $(date -Is)"
  echo "=================================================="
  sleep "${SLEEP_BETWEEN_RUNS}"
}

if [[ "${RUN_ROBOT_ONLY}" == "1" ]]; then
  run "robot_only" \
    description=robot_only_45step_pyav \
    data=viperx_local \
    model=hpt_bc_flow_viperx
fi

if [[ "${RUN_ARIA}" == "1" ]]; then
  run "robot_plus_local_aria_sched" \
    description=robot_plus_local_aria_sched_45step_pyav \
    data=cotrain_viperx_aria_local \
    model=hpt_cotrain_viperx_aria_sched \
    data.train_datasets.aria_left_arm.folder_path="${ARIA_DATA_ROOT}" \
    data.valid_datasets.aria_left_arm.folder_path="${ARIA_DATA_ROOT}"
fi

echo
echo "Requested ViperX two-run sweep completed."
