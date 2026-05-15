# ViperX Integration with EgoVerse

## Overview

Adapting a ViperX robot arm setup (recorded via LeRobot) to train with
EgoVerse.

The current setup supports:

- ViperX local LeRobot training via `FolderRLDBDataset`.
- ViperX + Scale co-training, where ViperX is local LeRobot data and Scale is
  remote or locally cached Zarr data.

---

## Pipeline

    ViperX LeRobot dataset
        ↓ (viperx_to_lerobot.py)
    EgoVerse training (parquet + mp4, 9 DoF -> 7 DoF)
        ↓ (convert + add columns)
    FolderRLDBDataset

For Scale co-training:

    Scale Zarr episodes
        ↓ (S3StreamingEpisodeResolver or S3EpisodeResolver)
    ZarrDataset + Scale transform pipeline
        ↓
    front_img_1, state_ee_pose, actions_cartesian
        ↓
    MultiDataModuleWrapper with ViperX batches

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
- **`hydra_configs/data/cotrain_viperx_scale.yaml`**: Adds ViperX + Scale co-training. ViperX is read from local LeRobot data and Scale is read from Zarr episodes.
- **`hydra_configs/model/hpt_cotrain_viperx_scale.yaml`**: Adds separate action heads for ViperX joint actions and Scale Cartesian actions while sharing the front camera representation.
- **`egomimic/algo/hpt.py`**: Supports per-domain action horizons so ViperX can use 64-step joint chunks and Scale can use 100-step Cartesian chunks in the same run.
- **`egomimic/rldb/utils.py`**: Supports normalization statistics for transformed Zarr outputs such as Scale `state_ee_pose` and `actions_cartesian`.

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
cd ..
git clone git@github.com:Anon-adam/lerobot.git --branch egomimic
cd lerobot
pip install -e . --no-deps
# Downgrade torchcodec for compatibility with torch 2.6.0
# pip install torchcodec==0.2.1

# other dependencies
pip install awscli
pip install accelerate

cd ../EgoVerse
```

### 1. Verification Scripts

# 1. Activate environment
```bash
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

# Verify VIPERX_RIGHT_ARM is already in the enum
python3 -c "
from egomimic.rldb.embodiment.embodiment import EMBODIMENT
print(EMBODIMENT.VIPERX_RIGHT_ARM)
"
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

### Training Run
## Option A: Training from scratch on local data

```bash
python egomimic/trainHydra.py \
  --config-name=train \
  data=viperx_local \
  model=hpt_bc_flow_viperx \
  logger=wandb \
  trainer=ddp
```

## Option B: Fine-Tuning from Pretrained EgoVerse Model (Recommended)

```bash
python egomimic/trainHydra.py \
  --config-name=train \
  data=viperx_local \
  model=hpt_bc_flow_viperx \
  ckpt_path=/path/to/downloaded/egoverse_base.ckpt \
  logger=wandb \
  trainer=ddp
```

## Option C: Co-Training ViperX With Scale

```bash
python egomimic/trainHydra.py \
  --config-name=train \
  data=cotrain_viperx_scale \
  model=hpt_cotrain_viperx_scale \
  logger=wandb \
  trainer=ddp \
  trainer.strategy=ddp_find_unused_parameters_true
```

This run uses two datasets at each training step:

- `viperx_right_arm`: local LeRobot data, action key `actions_joints`, shape `(64, 7)`.
- `scale_bimanual`: Scale Zarr data, action key `actions_cartesian`, shape `(100, 12)`.

The Scale raw Zarr episodes do not store `actions_cartesian` directly. They store:

- `images.front_1`
- `left.obs_ee_pose`
- `right.obs_ee_pose`
- `obs_head_pose`

The config uses `_build_aria_bimanual_transform_list` with `stride: 1` to produce:

- `front_img_1`
- `state_ee_pose`
- `actions_cartesian`

### Scale Dataset Source Modes

For direct remote streaming, use:

```yaml
resolver:
  _target_: egomimic.rldb.zarr.zarr_dataset_multi.S3StreamingEpisodeResolver
  bucket_name: rldb
  folder_path: /data/sybeuret/tmp/egoverse_unused
  max_episodes: 100
```

For local caching of a subset, use:

```yaml
resolver:
  _target_: egomimic.rldb.zarr.zarr_dataset_multi.S3EpisodeResolver
  bucket_name: rldb
  folder_path: /data/sybeuret/scale_zarr_cache
  max_episodes: 50
```

`S3EpisodeResolver` downloads selected episodes into `folder_path` and skips
episodes that are already present on later runs. This is recommended when S3
streaming produces bursty GPU utilization.

Check cache size with:

```bash
du -sh /data/sybeuret/scale_zarr_cache
```

### Normalization Settings

`train.yaml` controls normalization sampling:

```yaml
norm_stat_fraction: 0.01
norm_stat_max_samples: 512
```

`norm_stat_fraction` is the fraction of the frame-level dataset to sample.
`norm_stat_max_samples` is a hard cap on the total sampled frames per transformed
key, not per episode. For Scale, the transformed keys are currently
`state_ee_pose` and `actions_cartesian`.

When debugging, use a smaller cap such as `128` or `256`. Once training works,
increase `max_episodes` for more data diversity while keeping
`norm_stat_max_samples` capped to avoid long startup times.

### Rollout / Inference

Once your model has trained (check logs/checkpoints/ for .ckpt files), you can run inference directly on the ViperX using the wrapper built on LeRobot. Ensure the robot workspace is clear and Aria glasses are connected.

```bash
python egomimic/robot/viperx/viperx_rollout.py \
  --policy-path /path/to/your/checkpoint/last.ckpt \
  --frequency 30 \
  --query_frequency 30
```

## Rollout Controls:

- q : Quit safely and stop inference.
- r : Restart rollout state (Clears action chunking buffers).

### Next steps / Troubleshooting

If you encounter errors during initialization, verify the following:

- Missing camera_transforms in YAML: Ensure camera_transforms: {} exists in hpt_bc_flow_viperx.yaml.
- keys_of_type Argument Error: The HPT.__init__ in egomimic/algo/hpt.py may be explicitly passing the embodiment_id flag. Verify that the line data_schematic.keys_of_type("action_keys") matches the signature in rldb/utils.py.
- "Data not found" during NormStats inference: If debug logs report Skipping observation.state, double check string names. Note the difference between observation.(...) (singular) and observations.(...) (plural) in your YAML files versus info.json.
- Slow `[NormStats] transformed ...`: Scale `state_ee_pose` and `actions_cartesian` are created by transforms, so norm statistics must sample frames and run the transform pipeline. Reduce `norm_stat_max_samples` for debugging.
- Bursty training throughput: Remote Scale streaming reads from S3 in DataLoader workers. If CPU/network/GPU usage comes in peaks, cache a subset locally with `S3EpisodeResolver`, reduce Scale batch size, and use `persistent_workers`, `prefetch_factor`, and `pin_memory`.
- Shape error in Scale action loss: Ensure Scale action heads use `action_horizon: 100`, `act_dim: 12`, and `ac_keys.scale_bimanual: actions_cartesian`. HPT should use per-domain head horizons, not the shared trunk horizon, when slicing actions.
- LeRobot Driver Issues during Rollout: If ViperXInterface fails to connect, ensure no other python scripts are grabbing the camera feed or serial ports simultaneously.

---

## Scaling & Cross-Embodiment Training Possibilities

EgoVerse is designed to scale dynamically across diverse datasets and embodiments. Once your local single-robot training works, you can expand your configurations:

1. **Multi-Robot Co-Training (Local)**: Train ViperX and Aria (or other arms) simultaneously. Create a `cotrain_viperx_aria.yaml` that feeds both local datasets into a shared Heterogeneous Pretrained Transformer (HPT) trunk using separate diffusion heads for joint space (ViperX) and Cartesian space (Aria).
2. **ViperX + Scale Co-Training**: Use `cotrain_viperx_scale.yaml` with a shared visual trunk, a ViperX 7-DoF joint head, and a Scale 12-DoF Cartesian head. Start with `max_episodes: 50` or `100`, then increase after throughput and loss curves look healthy.
3. **Full Dataset Scale (S3/Cluster)**: Use S3/Zarr resolvers to leverage thousands of episodes across unified visual trunks. For single-machine experiments, prefer a local cache subset if disk space allows. For full remote streaming, expect to need more CPU/network bandwidth and multiple GPUs.
4. **Action-Free Human Co-Training**: Use Ego4D and Project Aria human demonstration datasets (`human_hands` embodiment) to heavily pre-train the model's visual trunk. The human data (which lacks explicit robot actions) improves visual representation learning, while your ViperX data fine-tunes the action-decoding head.
