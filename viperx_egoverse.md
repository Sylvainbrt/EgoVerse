# ViperX Integration with EgoVerse

## Overview

Adapting a ViperX robot arm setup (recorded via LeRobot) to train with
EgoVerse.\
EgoVerse supports two training pipelines; we use the **LeRobot
pipeline** (not Zarr).

------------------------------------------------------------------------

## Pipeline

    ViperX LeRobot dataset
        ↓ (viperx_to_lerobot.py)
    EgoVerse training (parquet + mp4, 9 DoF)
        ↓ (convert + add columns)
    FolderRLDBDataset

------------------------------------------------------------------------

## Current Dataset State

Recorded dataset at:

    /data/sybeuret/.local/huggingface/lerobot/lerobot/pick_sponge

  ----------------------------------------------------------------------------
  Property              Value
  --------------------- ------------------------------------------------------
  `robot_type`          "viperx"

  `action` shape        (T, 9) --- includes 2 shadow joints at indices 2 and 4

  `observation.state`   (T, 9)

  Images                observation.images.front_img_1,
                        observation.images.right_wrist_img

  Missing               actions.joints_act, metadata.embodiment
  ----------------------------------------------------------------------------

------------------------------------------------------------------------

## Required Modifications

### 1. Add ViperX to EMBODIMENT enum

**File:** egomimic/rldb/embodiment/embodiment.py

``` python
class EMBODIMENT(Enum):
    SCALE_RIGHT_ARM = 13
    SCALE_LEFT_ARM = 14
    VIPERX_RIGHT_ARM = 15
```

------------------------------------------------------------------------

### 2. Create ViperX Embodiment class

**File:** egomimic/rldb/embodiment/viperx.py

-   Implement get_keymap()
-   Implement get_transform_list()
-   Keep indices \[0, 1, 3, 5, 6, 7, 8\]

------------------------------------------------------------------------

### 3. Dataset Conversion Script

**File:** egomimic/scripts/viperx_process/viperx_to_lerobot.py

  -----------------------------------------------------------------------
  Transform             Detail
  --------------------- -------------------------------------------------
  observation.state     (T, 9) → (T, 7)

  actions.joints_act    (T, 9) → (T, 100, 7)

  metadata.embodiment   constant value

  info.json robot_type  viperx → viperx_right_arm
  -----------------------------------------------------------------------

    POINT_GAP = 2
    CHUNK_LENGTH = 100

Run:

    python egomimic/scripts/viperx_process/viperx_to_lerobot.py \
      --input-path  /data/.../pick_sponge \
      --output-path /data/.../pick_sponge_egov \
      --repo-id     lerobot/pick_sponge_egov

------------------------------------------------------------------------

### 4. Hydra Config

``` yaml
_target_: egomimic.pl_utils.pl_data_utils.MultiDataModuleWrapper

train_datasets:
  dataset1:
    _target_: egomimic.rldb.utils.FolderRLDBDataset
    folder_path: /data/.../pick_sponge_egov
    embodiment: viperx_right_arm
    mode: train

valid_datasets:
  dataset1:
    _target_: egomimic.rldb.utils.FolderRLDBDataset
    folder_path: /data/.../pick_sponge_egov
    embodiment: viperx_right_arm
    mode: valid
```

------------------------------------------------------------------------

## TODO

-   [ ] Check configs
-   [ ] Check DataSchematic
-   [ ] Optional FK actions
-   [ ] Implement transforms

------------------------------------------------------------------------

## Files

-   embodiment.py
-   viperx.py
-   viperx_to_lerobot.py
-   viperx_local.yaml


## Running

### Prerequisites
```bash 
# 0. Create environment and setup 
git clone --recursive git@github.com:GaTech-RL2/EgoVerse.git
cd EgoVerse
conda env create -f environment.yaml
conda activate emimic
pip install -e .
pre-commit install
```

```bash
# 1. Activate environment
conda activate egoverse

# 2. Verify your source dataset looks correct
python3 -c "
import pandas as pd
from pathlib import Path
p = sorted(Path('/data/sybeuret/.local/huggingface/lerobot/lerobot/pick_and_place').rglob('*.parquet'))[0]
df = pd.read_parquet(p)
print('Columns:', df.columns.tolist())
print('Shape:',   df.shape)
print('Action shape:', df['action'][0].shape)
"

# Expected: 
# Columns: ['action', 'observation.state', 'timestamp', 'frame_index', 'episode_index', 'index', 'task_index']
# Shape: (7470, 7)
# Action shape: (9,)


# Verify VIPERX_RIGHT_ARM is already in the enum (was added manually)
python3 -c "
from egomimic.rldb.embodiment.embodiment import EMBODIMENT
print(EMBODIMENT.VIPERX_RIGHT_ARM)
"
# Expected: EMBODIMENT.VIPERX_RIGHT_ARM
```

### Run conversion

```bash
cd /data/sybeuret/codes/EgoVerse

python egomimic/scripts/viperx_process/viperx_to_lerobot.py \
  --input-path  /data/sybeuret/.local/huggingface/lerobot/lerobot/pick_and_place \
  --output-path /data/sybeuret/.local/huggingface/lerobot/lerobot/pick_and_place_egoverse \
  --repo-id     lerobot/pick_sponge_egoverse

```
