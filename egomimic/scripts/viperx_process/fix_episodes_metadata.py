"""
Add missing video metadata columns to episodes parquet.
Run once to fix the dataset after lerobot-record conversion.
"""

import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def fix_episodes_metadata(dataset_root: Path):
    info_path = dataset_root / "meta" / "info.json"
    info = json.loads(info_path.read_text())

    video_keys = [k for k, v in info["features"].items() if v["dtype"] == "video"]
    fps = info["fps"]

    print(f"Video keys: {video_keys}")

    ep_file = dataset_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    df = pd.read_parquet(ep_file)
    print(
        f"Existing columns (video-related): {[c for c in df.columns if 'video' in c or 'chunk' in c]}"
    )

    for vid_key in video_keys:
        chunk_col = f"videos/{vid_key}/chunk_index"
        file_col = f"videos/{vid_key}/file_index"
        from_col = f"videos/{vid_key}/from_timestamp"
        to_col = f"videos/{vid_key}/to_timestamp"

        if chunk_col not in df.columns:
            print(f"Adding video columns for: {vid_key}")

            # All episodes recorded in single chunk/file (chunk-000/file-000)
            df[chunk_col] = 0
            df[file_col] = 0

            # Compute from/to timestamps: each episode is a separate mp4 file
            # starting at t=0 in its own file
            df[from_col] = 0.0
            df[to_col] = df["length"].astype(float) / fps

    print(f"\nAdded columns: {[c for c in df.columns if 'videos/' in c]}")
    print(
        df[
            ["episode_index", "length"] + [c for c in df.columns if "videos/" in c]
        ].head(3)
    )

    table = pa.Table.from_pandas(df)
    pq.write_table(table, ep_file, compression="snappy")
    print(f"\nWritten back to {ep_file}")


if __name__ == "__main__":
    dataset_root = Path(
        "/data/sybeuret/.local/huggingface/lerobot/lerobot/egoverse_data/pick_and_place_egoverse"
    )
    fix_episodes_metadata(dataset_root)
