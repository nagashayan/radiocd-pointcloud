from pathlib import Path
import argparse

import numpy as np
import pandas as pd


EXPECTED_COLUMNS = {
    "Frame #",
    "# Obj",
    "X",
    "Y",
    "Z",
    "Doppler",
    "Intensity",
    "Presence",
}


def get_metadata(csv_path, dataset_root):
    """
    Extract subject, environment, and object class from the directory path.

    Expected structure:
    dataset/
        subject_1/
            env_1/
                chair/
                    recording.csv
    """
    relative_path = csv_path.relative_to(dataset_root)
    parts = relative_path.parts

    if len(parts) < 4:
        raise ValueError(f"Unexpected path structure: {relative_path}")

    subject = parts[0]
    environment = parts[1]
    object_class = parts[2]

    return subject, environment, object_class


def inspect_recording(csv_path, dataset_root):
    """Inspect one RadIOCD recording."""

    subject, environment, object_class = get_metadata(
        csv_path, dataset_root
    )

    df = pd.read_csv(csv_path)

    missing_columns = sorted(EXPECTED_COLUMNS - set(df.columns))

    # --------------------------------------------------
    # Frame statistics
    # --------------------------------------------------

    if "Frame #" in df.columns:

        frame_counts = df.groupby("Frame #").size()

        num_frames = df["Frame #"].nunique()

        min_points_per_frame = frame_counts.min()
        max_points_per_frame = frame_counts.max()
        mean_points_per_frame = frame_counts.mean()

    else:

        num_frames = 0
        min_points_per_frame = 0
        max_points_per_frame = 0
        mean_points_per_frame = 0

    # --------------------------------------------------
    # Presence statistics
    # --------------------------------------------------

    if "Presence" in df.columns:

        presence = pd.to_numeric(
            df["Presence"],
            errors="coerce"
        )

        presence_zero = (presence == 0).sum()
        presence_one = (presence == 1).sum()

    else:

        presence_zero = 0
        presence_one = 0

    # --------------------------------------------------
    # Spatial / radar feature statistics
    # --------------------------------------------------

    radar_statistics = {}

    for column in [
        "X",
        "Y",
        "Z",
        "Doppler",
        "Intensity",
    ]:

        if column in df.columns:

            values = pd.to_numeric(
                df[column],
                errors="coerce"
            )

            radar_statistics[f"{column}_min"] = values.min()
            radar_statistics[f"{column}_max"] = values.max()
            radar_statistics[f"{column}_mean"] = values.mean()
            radar_statistics[f"{column}_std"] = values.std()

        else:

            radar_statistics[f"{column}_min"] = np.nan
            radar_statistics[f"{column}_max"] = np.nan
            radar_statistics[f"{column}_mean"] = np.nan
            radar_statistics[f"{column}_std"] = np.nan

    return {
        "file": str(csv_path),

        "subject": subject,
        "environment": environment,
        "class": object_class,

        "rows": len(df),
        "frames": num_frames,

        "min_points_per_frame": min_points_per_frame,
        "max_points_per_frame": max_points_per_frame,
        "mean_points_per_frame": mean_points_per_frame,

        "presence_zero": presence_zero,
        "presence_one": presence_one,

        "missing_columns": ";".join(missing_columns),

        "nan_cells": df.isna().sum().sum(),

        **radar_statistics,
    }


def main():

    parser = argparse.ArgumentParser(
        description="Inspect the RadIOCD dataset."
    )

    parser.add_argument(
        "dataset_root",
        type=Path,
        help="Path to RadIOCD dataset",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
    )

    args = parser.parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = args.output_dir.resolve()

    # --------------------------------------------------
    # Find recordings
    # --------------------------------------------------

    csv_files = sorted(
        dataset_root.rglob("*.csv")
    )

    print()
    print("RadIOCD Dataset Audit")
    print("---------------------")
    print(f"Dataset: {dataset_root}")
    print(f"CSV files: {len(csv_files)}")
    print()

    if not csv_files:
        raise RuntimeError(
            "No CSV files were found."
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    recordings = []
    errors = []

    # --------------------------------------------------
    # Scan recordings
    # --------------------------------------------------

    for i, csv_file in enumerate(
        csv_files,
        start=1
    ):

        try:

            result = inspect_recording(
                csv_file,
                dataset_root
            )

            recordings.append(result)

        except Exception as error:

            errors.append(
                {
                    "file": str(csv_file),
                    "error": str(error),
                }
            )

        if i % 500 == 0:
            print(
                f"Processed {i}/{len(csv_files)}"
            )

    print(
        f"Processed {len(csv_files)}/{len(csv_files)}"
    )

    audit = pd.DataFrame(recordings)

    # --------------------------------------------------
    # Save complete audit
    # --------------------------------------------------

    audit.to_csv(
        output_dir / "recording_audit.csv",
        index=False,
    )

    # --------------------------------------------------
    # Subject summary
    # --------------------------------------------------

    subject_summary = (
        audit.groupby("subject")
        .agg(
            recordings=("file", "count"),
            total_frames=("frames", "sum"),
            mean_frames=("frames", "mean"),
        )
        .reset_index()
    )

    subject_summary.to_csv(
        output_dir / "subject_summary.csv",
        index=False,
    )

    # --------------------------------------------------
    # Class summary
    # --------------------------------------------------

    class_summary = (
        audit.groupby("class")
        .agg(
            recordings=("file", "count"),
            total_frames=("frames", "sum"),
            mean_frames=("frames", "mean"),
            mean_points_per_frame=(
                "mean_points_per_frame",
                "mean",
            ),
        )
        .reset_index()
    )

    class_summary.to_csv(
        output_dir / "class_summary.csv",
        index=False,
    )

    # --------------------------------------------------
    # Environment / subject / class summary
    # --------------------------------------------------

    dataset_summary = (
        audit.groupby(
            [
                "subject",
                "environment",
                "class",
            ]
        )
        .agg(
            recordings=("file", "count"),
            total_frames=("frames", "sum"),
            mean_points_per_frame=(
                "mean_points_per_frame",
                "mean",
            ),
        )
        .reset_index()
    )

    dataset_summary.to_csv(
        output_dir / "dataset_summary.csv",
        index=False,
    )

    # --------------------------------------------------
    # Errors
    # --------------------------------------------------

    if errors:

        pd.DataFrame(errors).to_csv(
            output_dir / "errors.csv",
            index=False,
        )

    # --------------------------------------------------
    # Terminal summary
    # --------------------------------------------------

    print()
    print("Audit complete")
    print("--------------")

    print(
        f"Valid recordings: {len(audit)}"
    )

    print(
        f"Files with errors: {len(errors)}"
    )

    print(
        f"Subjects: {audit['subject'].nunique()}"
    )

    print(
        "Environments:",
        sorted(
            audit["environment"]
            .unique()
            .tolist()
        ),
    )

    print(
        "Classes:",
        sorted(
            audit["class"]
            .unique()
            .tolist()
        ),
    )

    print(
        "Files missing expected columns:",
        (audit["missing_columns"] != "").sum(),
    )

    print(
        "Files containing NaNs:",
        (audit["nan_cells"] > 0).sum(),
    )

    print()
    print("Class distribution")
    print("------------------")

    print(
        class_summary.to_string(
            index=False
        )
    )

    print()
    print("Output written to:")
    print(output_dir)


if __name__ == "__main__":
    main()