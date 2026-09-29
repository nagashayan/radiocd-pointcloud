from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import KFold
from tensorflow import keras
from tensorflow.keras import layers


CLASS_NAMES = ["backpack", "chair", "desk", "human", "wall"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a PointNet-style classifier on RadIOCD XYZ point clouds."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("outputs/pointnet_xyz_32.npz"),
        help="NPZ produced by build_pointnet_dataset.py",
    )
    parser.add_argument(
        "--mode",
        choices=["10fold", "loso", "both"],
        default="10fold",
        help="Evaluation protocol (default: 10fold).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/pointnet"),
        help="Directory for results/checkpoints.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12,
    )
    return parser.parse_args()


def normalize_xyz(
    x_train: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Standardize X/Y/Z using training-fold statistics only.

    Mean/std are computed across every point from every training sample.
    """
    mean = x_train.reshape(-1, 3).mean(axis=0)
    std = x_train.reshape(-1, 3).std(axis=0)
    std[std == 0] = 1.0

    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    return x_train.astype(np.float32), x_test.astype(np.float32)


def conv_block(x, filters: int):
    """
    Shared MLP over points.

    Conv1D(kernel_size=1) applies the same learned transform independently
    to every point, which is the core PointNet idea.
    """
    x = layers.Conv1D(
        filters,
        kernel_size=1,
        use_bias=False,
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x


def dense_block(x, units: int, dropout: float = 0.3):
    x = layers.Dense(
        units,
        use_bias=False,
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(dropout)(x)
    return x


def build_pointnet(
    n_points: int,
    n_dims: int,
    n_classes: int = 5,
) -> keras.Model:
    """
    Core PointNet baseline:
      point-wise shared MLP
      -> global max pooling
      -> classifier

    This first experiment intentionally omits PointNet's T-Net alignment
    modules so we can test the basic learned point representation cleanly.
    """
    inputs = keras.Input(
        shape=(n_points, n_dims),
        name="points",
    )

    x = conv_block(inputs, 64)
    x = conv_block(x, 64)
    x = conv_block(x, 64)
    x = conv_block(x, 128)
    x = conv_block(x, 256)

    # Permutation-invariant aggregation across all points.
    x = layers.GlobalMaxPooling1D()(x)

    x = dense_block(x, 256, dropout=0.3)
    x = dense_block(x, 128, dropout=0.3)

    outputs = layers.Dense(
        n_classes,
        activation="softmax",
    )(x)

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="radiocd_pointnet_xyz",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=1e-3
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def train_one_fold(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    checkpoint_path: Path,
) -> tuple[float, float, np.ndarray, int]:
    x_train, x_test = normalize_xyz(
        x_train,
        x_test,
    )

    model = build_pointnet(
        n_points=x_train.shape[1],
        n_dims=x_train.shape[2],
        n_classes=len(CLASS_NAMES),
    )

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=30,
        mode="max",
        restore_best_weights=False,
        verbose=0,
    )

    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoint_path),
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        save_weights_only=True,
        verbose=0,
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=10,
        min_lr=1e-5,
        verbose=0,
    )

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        batch_size=128,
        epochs=300,
        verbose=0,
        callbacks=[
            early_stopping,
            checkpoint,
            reduce_lr,
        ],
    )

    model.load_weights(checkpoint_path)

    probabilities = model.predict(
        x_test,
        batch_size=512,
        verbose=0,
    )
    predictions = np.argmax(probabilities, axis=1)

    accuracy = float(np.mean(predictions == y_test))
    macro_f1 = float(
        f1_score(
            y_test,
            predictions,
            average="macro",
        )
    )

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1, 2, 3, 4],
    )

    epochs_ran = len(history.history["loss"])

    del model, probabilities
    keras.backend.clear_session()

    return accuracy, macro_f1, cm, epochs_ran


def save_confusion_matrix(
    cm: np.ndarray,
    path: Path,
) -> None:
    pd.DataFrame(
        cm,
        index=CLASS_NAMES,
        columns=CLASS_NAMES,
    ).to_csv(path)


def load_dataset(path: Path):
    data = np.load(
        path,
        allow_pickle=True,
    )

    x = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    csv_names = data["csv"].astype(str)
    subjects = data["subject"].astype(str)

    return x, y, csv_names, subjects


def run_10fold(
    x: np.ndarray,
    y: np.ndarray,
    csv_names: np.ndarray,
    output_dir: Path,
) -> None:
    print("\nPointNet 10-fold cross-validation")
    print("---------------------------------")
    print(
        "Split unit: original CSV recording "
        "(same protocol used for DDNN reproduction)."
    )

    unique_csvs = np.unique(csv_names)

    kf = KFold(
        n_splits=10,
        random_state=12,
        shuffle=True,
    )

    rows = []
    cms = []

    for fold, (train_idx, test_idx) in enumerate(
        kf.split(unique_csvs),
        start=1,
    ):
        train_csvs = unique_csvs[train_idx]
        test_csvs = unique_csvs[test_idx]

        train_mask = np.isin(
            csv_names,
            train_csvs,
        )
        test_mask = np.isin(
            csv_names,
            test_csvs,
        )

        checkpoint_path = (
            output_dir
            / "checkpoints"
            / f"10fold_{fold}.weights.h5"
        )

        accuracy, macro_f1, cm, epochs_ran = train_one_fold(
            x[train_mask],
            y[train_mask],
            x[test_mask],
            y[test_mask],
            checkpoint_path,
        )

        rows.append(
            {
                "fold": fold,
                "train_samples": int(train_mask.sum()),
                "test_samples": int(test_mask.sum()),
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "epochs_ran": epochs_ran,
            }
        )
        cms.append(cm)

        print(
            f"Fold {fold:2d}: "
            f"accuracy={accuracy:.4f}, "
            f"macro_f1={macro_f1:.4f}, "
            f"epochs={epochs_ran}"
        )

    results = pd.DataFrame(rows)
    results.to_csv(
        output_dir / "10fold_results.csv",
        index=False,
    )

    mean_cm = np.mean(cms, axis=0)
    save_confusion_matrix(
        mean_cm,
        output_dir / "10fold_mean_confusion_matrix.csv",
    )

    print("\nPointNet 10-fold mean")
    print(f"Accuracy : {results['accuracy'].mean() * 100:.2f}%")
    print(f"Macro F1 : {results['macro_f1'].mean() * 100:.2f}%")
    print()
    print("Reference DDNN reproduction:")
    print("Accuracy : 74.21%")
    print("Macro F1 : 71.59%")


def run_loso(
    x: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    output_dir: Path,
) -> None:
    print("\nPointNet Leave-One-Subject-Out")
    print("------------------------------")

    unique_subjects = sorted(
        np.unique(subjects),
        key=lambda s: int(s.split("_")[-1]),
    )

    rows = []
    cms = []

    for subject in unique_subjects:
        train_mask = subjects != subject
        test_mask = subjects == subject

        checkpoint_path = (
            output_dir
            / "checkpoints"
            / f"loso_{subject}.weights.h5"
        )

        accuracy, macro_f1, cm, epochs_ran = train_one_fold(
            x[train_mask],
            y[train_mask],
            x[test_mask],
            y[test_mask],
            checkpoint_path,
        )

        rows.append(
            {
                "held_out_subject": subject,
                "train_samples": int(train_mask.sum()),
                "test_samples": int(test_mask.sum()),
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "epochs_ran": epochs_ran,
            }
        )
        cms.append(cm)

        print(
            f"{subject:10s}: "
            f"accuracy={accuracy:.4f}, "
            f"macro_f1={macro_f1:.4f}, "
            f"epochs={epochs_ran}"
        )

    results = pd.DataFrame(rows)
    results.to_csv(
        output_dir / "loso_results.csv",
        index=False,
    )

    mean_cm = np.mean(cms, axis=0)
    save_confusion_matrix(
        mean_cm,
        output_dir / "loso_mean_confusion_matrix.csv",
    )

    print("\nPointNet LOSO mean")
    print(f"Accuracy : {results['accuracy'].mean() * 100:.2f}%")
    print(f"Macro F1 : {results['macro_f1'].mean() * 100:.2f}%")


def main() -> None:
    args = parse_args()

    tf.keras.utils.set_random_seed(args.seed)

    data_path = args.data.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    (output_dir / "checkpoints").mkdir(
        parents=True,
        exist_ok=True,
    )

    x, y, csv_names, subjects = load_dataset(
        data_path
    )

    print("RadIOCD PointNet XYZ baseline")
    print("----------------------------")
    print(f"Data    : {data_path}")
    print(f"Samples : {len(x)}")
    print(f"Shape   : {x.shape}")
    print("Input   : XYZ only")
    print("Model   : shared MLP -> global max pool -> classifier")
    print("Note    : first baseline omits PointNet T-Net alignment modules")

    if args.mode in ("10fold", "both"):
        run_10fold(
            x,
            y,
            csv_names,
            output_dir,
        )

    if args.mode in ("loso", "both"):
        run_loso(
            x,
            y,
            subjects,
            output_dir,
        )


if __name__ == "__main__":
    main()
