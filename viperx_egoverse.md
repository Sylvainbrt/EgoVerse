# ViperX Integration with EgoVerse

## Overview

Adapting a ViperX robot arm setup (recorded via LeRobot) to train with
EgoVerse.  
EgoVerse supports two training pipelines; we use the **LeRobot
pipeline** (not Zarr).

---

## Pipeline

    ViperX LeRobot dataset
        ↓ (viperx_to_lerobot.py)
    EgoVerse training (parquet + mp4, 9 DoF -> 7 DoF)
        ↓ (convert + add columns)
    FolderRLDBDataset

---

## Dataset State

Original recorded dataset at:

    /data/sybeuret/.local/huggingface/lerobot/lerobot/pick_sponge

| Property            | Original Value                     | Converted Value (EgoVerse)       |
| ------------------- | ---------------------------------- | -------------------------------- |
| `robot_type`        | "viperx"                           | "viperx_right_arm"               |
| Action shape        | (T, 9) - 2 shadow joints           | (T, 100, 7) chunked 7-DoF        |
| `observation.state` | (T, 9)                             | (T, 7)                           |
| Images              | observation.images...              | observation.images...            |
| Keys added          | -                                  | actions.joints_act, metadata...  |

---

## Completed Implementations

### 1. Embodiment Setup
- **`egomimic/rldb/embodiment/embodiment.py`**: Added `VIPERX_RIGHT_ARM = 15` to the `EMBODIMENT` enum.
- **`egomimic/rldb/embodiment/viperx.py`**: Created `ViperX` mapping class defining `get_keymap()` and image formats (`[B, C, H, W]`).

### 2. Dataset Processing
- **`egomimic/scripts/viperx_process/viperx_to_lerobot.py`**: Handles filtering out shadow joints (indices 2 and 4), converting from 9-DoF to 7-DoF, adding the embodiment enum to parquet, chunking actions (T=100), and rewriting metadata.
- **`egomimic/scripts/viperx_process/fix_episodes_metadata.py`**: Repairs video metadata lost or mangled during Pandas/Parquet conversions.

### 3. HPT Hydra Configs
- **`hydra_configs/model/hpt_bc_flow_viperx.yaml`**: Defined encoder/stem structures matching 7-DoF inputs and empty `camera_transforms`.
- **`hydra_configs/train.yaml`**: Updated `DataSchematic` and dataset paths to map LeRobot raw names to EgoVerse model names dynamically.

---

## How to Run the Pipeline

### Prerequisites
```bash 
# 0. Create environment and setup 
git clone --recursive git@github.com:Sylvainbrt/EgoVerse.git
cd EgoVerse
conda env create -f environment.yaml
conda activate egoverse
pip install -e .
pre-commit install

# install lerobot
git clone git@github.com:Anon-adam/lerobot.git --branch egomimic
cd lerobot
pip install -e .
# Downgrade torchcodec for compatibility with torch 2.6.0
pip install torchcodec==0.2.1
```

### 1. Verification Scripts

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
  --repo-id     lerobot/pick_and_place_egoverse

# Restore missing video columns 
python egomimic/scripts/viperx_process/fix_episodes_metadata.py

```

### Verify converted dataset

```bash
python3 -c "
import pandas as pd
from pathlib import Path
import json

root = Path('/data/sybeuret/.local/huggingface/lerobot/lerobot/pick_and_place_egoverse')

# Check parquet columns and shapes
p = sorted(root.rglob('*.parquet'))[0]
df = pd.read_parquet(p)
print('Columns:', df.columns.tolist())
print('actions.joints_act shape:', df['actions.joints_act'][0].shape)  # expect (100, 7)
print('observation.state shape:',  df['observation.state'][0].shape)   # expect (7,)
print('metadata.embodiment:',      df['metadata.embodiment'][0])       # expect [15]

# Check info.json
info = json.loads((root / 'meta/info.json').read_text())
print('robot_type:', info['robot_type'])   # expect viperx_right_arm
print('features:', list(info['features'].keys()))
"
```

### Training Run

```bash
python egomimic/trainHydra.py \
  --config-name=train \
  data=viperx_local \
  model=hpt_bc_flow_viperx \
  logger=debug \
  trainer=debug
```

### Next steps / Troubleshooting

If you encounter errors during initialization, verify the following:

Missing camera_transforms in YAML: Ensure camera_transforms: {} exists in hpt_bc_flow_viperx.yaml.
keys_of_type Argument Error: The HPT.__init__ in egomimic/algo/hpt.py may be explicitly passing the embodiment_id flag. Verify that the line data_schematic.keys_of_type("action_keys") matches the signature in rldb/utils.py.
"Data not found" during NormStats inference: If debug logs report Skipping observation.state, double check string names. Note the difference between observation.(...) (singular) and observations.(...) (plural) in your YAML files versus info.json.
