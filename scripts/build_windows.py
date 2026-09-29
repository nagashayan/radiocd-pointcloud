from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from tqdm import tqdm


# Parameters used in the authors' RadIOCD notebook.
THRESHOLD_FRAMES = 8
THRESHOLD_GAP = 2
WINDOW = 10
EPS = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the RadIOCD 1-second window preprocessing and "
            "13-feature extraction from the authors' notebook."
        )
    )
    parser.add_argument(
        "dataset_root",
        type=Path,
        help="Path to the RadIOCD dataset directory containing subject_* folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/features.csv"),
        help="Output CSV for processed 1-second samples (default: outputs/features.csv).",
    )
    return parser.parse_args()


def get_closest_cluster(
    filtered_points_df: pd.DataFrame,
    eps: float,
    threshold_frames: int,
) -> tuple[pd.DataFrame | None, bool]:
    """
    Match the cluster-selection logic in the authors' notebook.

    DBSCAN groups nearby XYZ detections and labels isolated points as noise (-1).
    Among non-noise clusters whose X center lies within +/-0.3 m of the radar
    centerline, keep the cluster whose 3D center is closest to the radar.
    """
    if len(filtered_points_df) < threshold_frames:
        return None, False

    clustering = DBSCAN(
        eps=eps,
        min_samples=threshold_frames,
    ).fit(filtered_points_df)

    distances = []
    indices = []

    for label in np.unique(clustering.labels_):
        points = filtered_points_df.iloc[
            np.where(clustering.labels_ == label)[0]
        ]

        if len(points) == 0:
            continue

        mean_x = points["X"].mean()
        mean_y = points["Y"].mean()
        mean_z = points["Z"].mean()

        distance = np.sqrt(
            mean_x**2 + mean_y**2 + mean_z**2
        )

        # Same centerline rule as the original notebook.
        if np.abs(mean_x) < 0.3 and label != -1:
            if distance > 0.0:
                distances.append(distance)
                indices.append(label)

    if not distances:
        return None, False

    closest_label = indices[int(np.argmin(distances))]
    points_df = filtered_points_df.iloc[
        np.where(clustering.labels_ == closest_label)[0]
    ]

    return points_df, True


def infer_metadata(
    csv_path: Path,
    dataset_root: Path,
) -> tuple[str, str, str]:
    """
    Expected path:
      dataset/subject_X/env_X/object/moving_....csv
    """
    relative = csv_path.relative_to(dataset_root)
    parts = relative.parts

    if len(parts) < 4:
        raise ValueError(
            f"Unexpected RadIOCD path structure: {relative}"
        )

    return parts[0], parts[1], parts[2]


def process_recording(
    csv_path: Path,
    dataset_root: Path,
) -> list[dict]:
    """
    Reproduce the authors' per-recording preprocessing.

    Important:
    - Presence is already included in the released CSV files, so this script
      starts after the authors' original frame-wise annotation stage.
    - A 10-frame window corresponds to 1 second at 10 Hz.
    - Windows effectively advance by one frame because candidate starts are
      taken from each frame where Presence == 1.
    """
    cdf = pd.read_csv(csv_path)
    subject, environment, object_class = infer_metadata(
        csv_path,
        dataset_root,
    )

    if "Presence" not in cdf.columns:
        return []

    presence_frames = np.sort(
        cdf.loc[cdf["Presence"] == 1, "Frame #"].unique()
    )

    # Same early rejection as the authors' notebook.
    if len(presence_frames) < THRESHOLD_FRAMES:
        return []

    all_frames = set(cdf["Frame #"].unique())
    presence_frame_set = set(presence_frames)

    rows = []

    for start_frame in presence_frames:
        frame_window = np.arange(
            start_frame,
            start_frame + WINDOW,
        )
        frame_window_set = set(frame_window)

        # Number of frames inside this 10-frame window without target Presence.
        missing_presence_frames = len(
            frame_window_set - presence_frame_set
        )

        if missing_presence_frames > THRESHOLD_GAP:
            continue

        # Require the last frame of the proposed window to exist in the CSV.
        if frame_window[-1] not in all_frames:
            continue

        current_cdf = cdf[
            (cdf["Frame #"] >= frame_window[0])
            & (cdf["Frame #"] <= frame_window[-1])
        ]

        # Retain only manually annotated target points.
        target_points = current_cdf[
            current_cdf["Presence"] == 1
        ]

        # Match the notebook: remove zero-Doppler points, then cluster XYZ only.
        filtered_points = target_points.loc[
            target_points["Doppler"].abs() > 0,
            ["X", "Y", "Z"],
        ].copy()

        points_df, estimate_feature = get_closest_cluster(
            filtered_points,
            EPS,
            THRESHOLD_FRAMES,
        )

        if not estimate_feature or points_df is None:
            continue

        mean_y = points_df["Y"].mean()
        mean_z = points_df["Z"].mean()

        # Authors' intended filtering described in the paper:
        # keep targets roughly 1-2.5 m in front of the radar.
        #
        # The original notebook also contains:
        #   np.abs(np.mean(points_df['Z']) < 1)
        # which evaluates a boolean rather than abs(mean_Z). To reproduce the
        # notebook literally, this script uses the equivalent condition mean_Z < 1.
        if not (mean_y < 2.5 and mean_y > 1.0 and mean_z < 1.0):
            continue

        rows.append(
            {
                "object": object_class,
                "subject": subject,
                "env": environment,
                "min_X": np.min(points_df["X"]),
                "max_X": np.max(points_df["X"]),
                "mean_X": np.mean(points_df["X"]),
                "std_X": np.std(points_df["X"]),
                "min_Y": np.min(points_df["Y"]),
                "max_Y": np.max(points_df["Y"]),
                "mean_Y": np.mean(points_df["Y"]),
                "std_Y": np.std(points_df["Y"]),
                "min_Z": np.min(points_df["Z"]),
                "max_Z": np.max(points_df["Z"]),
                "mean_Z": np.mean(points_df["Z"]),
                "std_Z": np.std(points_df["Z"]),
                "n_points": len(points_df),
                "csv": str(csv_path),
                "start_frame": int(frame_window[0]),
                "end_frame": int(frame_window[-1]),
            }
        )

    return rows


def main() -> None:
    args = parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    if not dataset_root.exists():
        raise SystemExit(
            f"Dataset directory does not exist: {dataset_root}"
        )

    csv_files = sorted(dataset_root.rglob("*.csv"))

    if not csv_files:
        raise SystemExit(
            f"No CSV files found under: {dataset_root}"
        )

    print("RadIOCD preprocessing reproduction")
    print("--------------------------------")
    print(f"Dataset: {dataset_root}")
    print(f"Recordings: {len(csv_files)}")
    print(
        f"Parameters: window={WINDOW}, "
        f"threshold_frames={THRESHOLD_FRAMES}, "
        f"threshold_gap={THRESHOLD_GAP}, eps={EPS}"
    )
    print()

    processed_rows = []
    errors = []

    for csv_path in tqdm(
        csv_files,
        desc="Processing recordings",
    ):
        try:
            processed_rows.extend(
                process_recording(
                    csv_path,
                    dataset_root,
                )
            )
        except Exception as exc:
            errors.append(
                {
                    "file": str(csv_path),
                    "error": repr(exc),
                }
            )

    features = pd.DataFrame(processed_rows)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    features.to_csv(
        output_path,
        index=False,
    )

    if errors:
        errors_path = output_path.parent / "preprocessing_errors.csv"
        pd.DataFrame(errors).to_csv(
            errors_path,
            index=False,
        )

    counts = Counter(features["object"]) if len(features) else Counter()

    expected = {
        "backpack": 16027,
        "chair": 19825,
        "desk": 6258,
        "human": 21393,
        "wall": 13318,
    }

    print()
    print("Processed sample counts")
    print("-----------------------")
    for class_name in [
        "backpack",
        "chair",
        "desk",
        "human",
        "wall",
    ]:
        actual = counts.get(class_name, 0)
        target = expected[class_name]
        difference = actual - target
        print(
            f"{class_name:10s} "
            f"{actual:7d} "
            f"(paper: {target:7d}, diff: {difference:+d})"
        )

    print("-----------------------")
    print(
        f"{'TOTAL':10s} "
        f"{len(features):7d} "
        f"(paper: {sum(expected.values()):7d}, "
        f"diff: {len(features) - sum(expected.values()):+d})"
    )

    print()
    print(f"Saved: {output_path}")
    print(f"Files with errors: {len(errors)}")


if __name__ == "__main__":
    main()
