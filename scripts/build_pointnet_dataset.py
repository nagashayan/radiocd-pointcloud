from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from tqdm import tqdm


WINDOW = 10
THRESHOLD_FRAMES = 8
THRESHOLD_GAP = 2
EPS = 0.5
N_POINTS = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build fixed-size XYZ point-cloud samples for PointNet from RadIOCD."
    )
    parser.add_argument(
        "dataset_root",
        type=Path,
        help="Path to the RadIOCD dataset directory containing subject_* folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/pointnet_xyz_32.npz"),
        help="Output NPZ file (default: outputs/pointnet_xyz_32.npz).",
    )
    parser.add_argument(
        "--n-points",
        type=int,
        default=N_POINTS,
        help="Fixed number of points per sample (default: 32).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12,
        help="Random seed used only for deterministic down/up-sampling.",
    )
    return parser.parse_args()


def infer_metadata(csv_path: Path, dataset_root: Path) -> tuple[str, str, str]:
    relative = csv_path.relative_to(dataset_root)
    parts = relative.parts
    if len(parts) < 4:
        raise ValueError(f"Unexpected RadIOCD path structure: {relative}")
    return parts[0], parts[1], parts[2]


def get_closest_cluster(
    target_points: pd.DataFrame,
) -> pd.DataFrame | None:
    """
    Reproduce the RadIOCD cluster selection while retaining the original
    point rows so XYZ can be saved for PointNet.
    """
    moving = target_points[target_points["Doppler"].abs() > 0].copy()

    if len(moving) < THRESHOLD_FRAMES:
        return None

    xyz = moving[["X", "Y", "Z"]].to_numpy(dtype=np.float32)

    clustering = DBSCAN(
        eps=EPS,
        min_samples=THRESHOLD_FRAMES,
    ).fit(xyz)

    best_label = None
    best_distance = np.inf

    for label in np.unique(clustering.labels_):
        if label == -1:
            continue

        mask = clustering.labels_ == label
        cluster_xyz = xyz[mask]

        mean_xyz = cluster_xyz.mean(axis=0)
        mean_x, mean_y, mean_z = mean_xyz

        distance = float(np.linalg.norm(mean_xyz))

        if abs(mean_x) < 0.3 and distance > 0 and distance < best_distance:
            best_distance = distance
            best_label = label

    if best_label is None:
        return None

    chosen = moving.iloc[np.where(clustering.labels_ == best_label)[0]].copy()

    mean_y = float(chosen["Y"].mean())
    mean_z = float(chosen["Z"].mean())

    # Match the authors' notebook behavior used in our successful 76,821-sample reproduction.
    if not (1.0 < mean_y < 2.5 and mean_z < 1.0):
        return None

    return chosen


def fix_point_count(
    xyz: np.ndarray,
    n_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    PointNet batches need a fixed tensor shape.

    If a cloud has > n_points, sample without replacement.
    If it has < n_points, keep all original points and sample extra points
    with replacement. Duplicating points is compatible with max pooling and
    avoids inventing fake zero-valued points.
    """
    n = len(xyz)

    if n == n_points:
        return xyz.astype(np.float32, copy=False)

    if n > n_points:
        indices = rng.choice(n, size=n_points, replace=False)
        return xyz[indices].astype(np.float32, copy=False)

    extra = rng.choice(n, size=n_points - n, replace=True)
    return np.concatenate([xyz, xyz[extra]], axis=0).astype(np.float32)


def process_recording(
    csv_path: Path,
    dataset_root: Path,
    n_points: int,
    rng: np.random.Generator,
) -> list[tuple[np.ndarray, str, str, str, str, int, int, int]]:
    df = pd.read_csv(csv_path)
    subject, environment, object_class = infer_metadata(csv_path, dataset_root)

    presence_frames = np.sort(
        df.loc[df["Presence"] == 1, "Frame #"].unique()
    )

    if len(presence_frames) < THRESHOLD_FRAMES:
        return []

    all_frames = set(df["Frame #"].unique())
    presence_frame_set = set(presence_frames)

    samples = []

    for start_frame in presence_frames:
        frame_window = np.arange(start_frame, start_frame + WINDOW)

        missing_presence_frames = len(
            set(frame_window) - presence_frame_set
        )
        if missing_presence_frames > THRESHOLD_GAP:
            continue

        if frame_window[-1] not in all_frames:
            continue

        current = df[
            (df["Frame #"] >= frame_window[0])
            & (df["Frame #"] <= frame_window[-1])
        ]

        target_points = current[current["Presence"] == 1]

        cluster = get_closest_cluster(target_points)
        if cluster is None:
            continue

        xyz = cluster[["X", "Y", "Z"]].to_numpy(dtype=np.float32)
        original_n = len(xyz)

        fixed_xyz = fix_point_count(
            xyz,
            n_points,
            rng,
        )

        samples.append(
            (
                fixed_xyz,
                object_class,
                subject,
                environment,
                str(csv_path),
                int(frame_window[0]),
                int(frame_window[-1]),
                int(original_n),
            )
        )

    return samples


def main() -> None:
    args = parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    if not dataset_root.exists():
        raise SystemExit(f"Dataset directory does not exist: {dataset_root}")

    csv_files = sorted(dataset_root.rglob("*.csv"))
    if not csv_files:
        raise SystemExit(f"No CSV files found under: {dataset_root}")

    rng = np.random.default_rng(args.seed)

    clouds = []
    labels = []
    subjects = []
    environments = []
    csv_names = []
    start_frames = []
    end_frames = []
    original_counts = []
    errors = []

    print("Building RadIOCD PointNet dataset")
    print("--------------------------------")
    print(f"Dataset : {dataset_root}")
    print(f"Files   : {len(csv_files)}")
    print(f"Input   : XYZ")
    print(f"Points  : {args.n_points} per sample")
    print()

    for csv_path in tqdm(csv_files, desc="Processing recordings"):
        try:
            samples = process_recording(
                csv_path,
                dataset_root,
                args.n_points,
                rng,
            )

            for (
                cloud,
                label,
                subject,
                environment,
                csv_name,
                start_frame,
                end_frame,
                original_n,
            ) in samples:
                clouds.append(cloud)
                labels.append(label)
                subjects.append(subject)
                environments.append(environment)
                csv_names.append(csv_name)
                start_frames.append(start_frame)
                end_frames.append(end_frame)
                original_counts.append(original_n)

        except Exception as exc:
            errors.append((str(csv_path), repr(exc)))

    X = np.stack(clouds).astype(np.float32)

    class_names = np.array(
        ["backpack", "chair", "desk", "human", "wall"]
    )
    class_map = {name: i for i, name in enumerate(class_names)}

    y = np.array([class_map[name] for name in labels], dtype=np.int64)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        X=X,
        y=y,
        class_names=class_names,
        subject=np.array(subjects),
        environment=np.array(environments),
        csv=np.array(csv_names),
        start_frame=np.array(start_frames, dtype=np.int64),
        end_frame=np.array(end_frames, dtype=np.int64),
        original_n_points=np.array(original_counts, dtype=np.int64),
    )

    print()
    print("PointNet dataset built")
    print("----------------------")
    print(f"Samples : {len(X)}")
    print(f"Shape   : {X.shape}  (samples, points, XYZ)")
    print(f"Saved   : {output_path}")
    print(f"Errors  : {len(errors)}")
    print()
    print("Class counts:")
    for name, idx in class_map.items():
        print(f"  {name:10s}: {int((y == idx).sum())}")

    print()
    print("Expected total from RadIOCD preprocessing: 76821")

    if errors:
        errors_path = output_path.with_suffix(".errors.csv")
        pd.DataFrame(errors, columns=["file", "error"]).to_csv(
            errors_path,
            index=False,
        )
        print(f"Error log: {errors_path}")


if __name__ == "__main__":
    main()
