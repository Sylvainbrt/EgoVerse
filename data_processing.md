# Using your own data

## Process Human Data for Training
# Robomimic HDF5
Collect Aria demonstrations via the Aria App, then transfer them to your computer, make the following structure
```
TASK_NAME_ARIA/
├── rawAria
│   ├── demo1.vrs
│   ├── demo1.vrs.json
...
│   ├── demoN.vrs
│   ├── demoN.vrs.json
└── converted (empty folder)
```

This will process your aria data into hdf5 format
```
cd egomimic
python scripts/aria_process/aria_to_robomimic.py --dataset /path/to/TASK_NAME_ARIA --out /path/to/converted/TASK_NAME.hdf5 --hand <left, right, or bimanual>
```

# RLDB Parquet (Built around LeRobot for fast I/O and more compatibility for BC training with features like prestacked actions)

Collect Aria demonstrations via the Aria App, then transfer them to your computer, make the following structure
```
TASK_NAME_ARIA/
├── rawAria
│   ├── demo1.vrs
│   ├── demo1.vrs.json
...
│   ├── demoN.vrs
│   ├── demoN.vrs.json
```
This will process your aria data into RLDB parquet format
```
python aria_process/aria_to_lerobot.py \
    --raw-path <path/to/rawAria> \
    --dataset-repo-id <hf repo-id if push to hf-hub> \
    --arm <left, right, or both> \
    --fps 30 \
    --video-encoding=false \
    --push=<True if push to hf-hub, else False> \
    --name lerobot \
    --prestack=True \
    --output-dir <path/to/output/directory> \
    --description "<task-description>" \
```

The data is stored as 
```
dataset/
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet
│       ├── episode_000001.parquet
│       ├── episode_000002.parquet
│       └── ... (additional indexed parquet files)
└── meta/
    ├── episodes.jsonl
    ├── info.json
    ├── stats.json
    ├── env.jsonl (to be added for robosuite / mujoco sim support)
    └── tasks.jsonl
```

