#!/usr/bin/env python3
import argparse
from pathlib import Path

from egomimic.utils.aws.aws_sql import create_default_engine, episode_table_to_df


def clean_str_series(series):
    return series.fillna("").astype(str).str.strip()


def add_basic_flags(df):
    df = df.copy()
    df["has_zarr"] = clean_str_series(df.get("zarr_processed_path", "")).ne("")
    df["usable"] = df["has_zarr"] & ~df.get("is_deleted", False).fillna(False).astype(
        bool
    )
    if "processing_error" in df:
        df["has_processing_error"] = clean_str_series(df["processing_error"]).ne("")
    else:
        df["has_processing_error"] = False
    if "zarr_processing_error" in df:
        df["has_zarr_error"] = clean_str_series(df["zarr_processing_error"]).ne("")
    else:
        df["has_zarr_error"] = False
    return df


def apply_filters(df, args):
    out = df
    if args.robot_name:
        out = out[clean_str_series(out["robot_name"]).isin(args.robot_name)]
    if args.task:
        task_cols = [c for c in ("task", "task_name") if c in out.columns]
        if task_cols:
            mask = False
            for col in task_cols:
                mask = mask | clean_str_series(out[col]).isin(args.task)
            out = out[mask]
    if args.lab:
        out = out[clean_str_series(out["lab"]).isin(args.lab)]
    if args.usable_only:
        out = out[out["usable"]]
    return out


def print_counts(df, group_cols, title, top=None):
    cols = [c for c in group_cols if c in df.columns]
    if not cols:
        return
    print(f"\n## {title}")
    counts = df.groupby(cols, dropna=False).size().reset_index(name="episodes")
    counts = counts.sort_values("episodes", ascending=False)
    if top is not None:
        counts = counts.head(top)
    print(counts.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Explore EgoVerse episode catalog.")
    parser.add_argument("--robot-name", action="append", default=[])
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument("--lab", action="append", default=[])
    parser.add_argument("--usable-only", action="store_true", default=True)
    parser.add_argument("--include-unusable", dest="usable_only", action="store_false")
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--sample", type=int, default=20)
    parser.add_argument("--out-csv", type=Path, default=None)
    args = parser.parse_args()

    engine = create_default_engine()
    df = add_basic_flags(episode_table_to_df(engine))
    filtered = apply_filters(df, args)

    print(f"Total rows:       {len(df)}")
    print(f"Usable zarr rows: {int(df['usable'].sum())}")
    print(f"Selected rows:    {len(filtered)}")

    print_counts(filtered, ["robot_name"], "By robot_name", args.top)
    print_counts(filtered, ["robot_name", "task"], "By robot_name/task", args.top)
    print_counts(
        filtered, ["robot_name", "task_name"], "By robot_name/task_name", args.top
    )
    print_counts(filtered, ["robot_name", "lab"], "By robot_name/lab", args.top)
    print_counts(filtered, ["robot_name", "is_eval"], "By robot_name/is_eval", args.top)

    sample_cols = [
        c
        for c in (
            "episode_hash",
            "robot_name",
            "task",
            "task_name",
            "task_description",
            "num_frames",
            "lab",
            "operator",
            "scene",
            "objects",
            "is_eval",
            "eval_success",
            "zarr_processed_path",
        )
        if c in filtered.columns
    ]
    if args.sample > 0 and len(filtered) > 0:
        print(f"\n## Sample {min(args.sample, len(filtered))}")
        print(filtered[sample_cols].head(args.sample).to_string(index=False))

    if args.out_csv is not None:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        filtered.to_csv(args.out_csv, index=False)
        print(f"\nWrote {len(filtered)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()
