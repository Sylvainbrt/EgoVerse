#!/usr/bin/env bash
set -euo pipefail

# ====== config (edit these) ======
REMOTE_USER_HOST="rpunamiya6@sky1.cc.gatech.edu"
REMOTE_PATH="/coc/flash7/rpunamiya6/Projects/EgoVerse/logs/everse_object_in_container/robot_bc_fixed_extrinsics_2025-12-15_18-12-16"   # e.g. /nethome/rpunamiya6/some/folder
LOCAL_PATH="./egomimic/robot/models/"                # where to sync into (local destination dir)
# =================================

mkdir -p "$LOCAL_PATH"

rsync -avh --progress --partial --inplace \
  --exclude='0/videos/***' \
  --exclude='0/wandb/***' \
  "${REMOTE_USER_HOST}:${REMOTE_PATH%/}/" \
  "${LOCAL_PATH%/}/"
