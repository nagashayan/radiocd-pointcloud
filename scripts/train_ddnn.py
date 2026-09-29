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


FEATURES = [
    "min_X", "max_X", "mean_X", "std_X",
    "min_Y", "max_Y", "mean_Y", "std_Y",
    "min_Z", "max_Z", "mean_Z", "std_Z",
    "n_points",
]

CLASS_MAP = {
    "backpack": 0,
    "chair": 1,
    "desk": 2,
    "human": 3,
    "wall": 4,
}

CLASS_NAMES = ["backpack", "chair", "desk", "human", "wall"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the RadIOCD DDNN baseline."
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("outputs/features.csv"),
        help="Path to features.csv produced by build_windows.py.",
    )
    parser.add_argument(
        "--mode",
        choices=["10fold", "loso", "both"],
        default="10fold",
        help="Evaluation protocol to run (default: 10fold).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/ddnn"),
        help="Directory for fold results and confusion matrices.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def z_score(
    x_train: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Match the authors' notebook:
    compute mean/std from the training fold only, then apply to train and test.
    """
    x_train = x_train.astype(np.float32, copy=True)
    x_test = x_test.astype(np.float32, copy=True)

    means = np.mean(x_train, axis=0)
    stds = np.std(x_train, axis=0)

    # Defensive guard; the original notebook does not need this for RadIOCD.
    stds[stds == 0] = 1.0

    return (x_train - means) / stds, (x_test - means) / stds


def build_model() -> keras.Model:
    """
    Published DDNN:
    13 -> 256 -> 256 -> 128 -> 128 -> 5
    ReLU after each hidden Dense layer and Dropout(0.1).
    """
    model = keras.Sequential(
        [
            layers.Input(shape=(len(FEATURES),)),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.1),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.1),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.1),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.1),
            layers.Dense(5, activation="softmax"),
        ]
    )

    # The authors use optimizer='Adam' with Keras defaults.
    model.compile(
        optimizer="Adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def prepare_dataframe(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = set(FEATURES + ["object", "subject", "csv"])
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            "features.csv is missing required columns: "
            + ", ".join(missing)
        )

    df = df.copy()
    df["object"] = df["object"].replace(CLASS_MAP)

    subject_map = {
        f"subject_{i}": i for i in range(1, 11)
    }
    df["subject"] = df["subject"].replace(subject_map)

    df["object"] = pd.to_numeric(df["object"], errors="raise").astype(int)
    df["subject"] = pd.to_numeric(df["subject"], errors="raise").astype(int)

    return df


def train_one_fold(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    checkpoint_path: Path,
) -> tuple[float, float, np.ndarray, int]:
    x_train, x_test = z_score(x_train, x_test)

    y_train_cat = keras.utils.to_categorical(
        y_train,
        num_classes=5,
    )
    y_test_cat = keras.utils.to_categorical(
        y_test,
        num_classes=5,
    )

    model = build_model()

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=50,
        mode="max",
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

    history = model.fit(
        x_train,
        y_train_cat,
        batch_size=128,
        epochs=1000,
        validation_data=(x_test, y_test_cat),
        verbose=0,
        callbacks=[early_stopping, checkpoint],
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

    # Explicit cleanup is helpful when training 10/20 models on a laptop.
    del model, y_train_cat, y_test_cat, probabilities
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


def run_10fold(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    print("\n10-fold cross-validation")
    print("------------------------")
    print(
        "Split unit: original CSV recording "
        "(overlapping 1-second windows stay together)."
    )

    unique_csvs = df["csv"].unique()
    kf = KFold(
        n_splits=10,
        random_state=12,
        shuffle=True,
    )

    rows = []
    confusion_matrices = []

    for fold, (train_idx, test_idx) in enumerate(
        kf.split(unique_csvs),
        start=1,
    ):
        csv_train = unique_csvs[train_idx]
        csv_test = unique_csvs[test_idx]

        train_mask = df["csv"].isin(csv_train)
        test_mask = df["csv"].isin(csv_test)

        x_train = df.loc[train_mask, FEATURES].to_numpy()
        y_train = df.loc[train_mask, "object"].to_numpy()

        x_test = df.loc[test_mask, FEATURES].to_numpy()
        y_test = df.loc[test_mask, "object"].to_numpy()

        checkpoint = (
            output_dir / "checkpoints" /
            f"10fold_{fold}.weights.h5"
        )

        accuracy, macro_f1, cm, epochs_ran = train_one_fold(
            x_train,
            y_train,
            x_test,
            y_test,
            checkpoint,
        )

        rows.append(
            {
                "fold": fold,
                "train_samples": len(y_train),
                "test_samples": len(y_test),
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "epochs_ran": epochs_ran,
            }
        )
        confusion_matrices.append(cm)

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

    mean_cm = np.mean(confusion_matrices, axis=0)
    save_confusion_matrix(
        mean_cm,
        output_dir / "10fold_mean_confusion_matrix.csv",
    )

    print("\n10-fold mean")
    print(f"Accuracy : {results['accuracy'].mean() * 100:.2f}%")
    print(f"Macro F1 : {results['macro_f1'].mean() * 100:.2f}%")
    print("Paper    : 76.53% accuracy, 72.55% F1")


def run_loso(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    print("\nLeave-One-Subject-Out (LOSO)")
    print("----------------------------")

    rows = []
    confusion_matrices = []

    for subject in range(1, 11):
        train_mask = df["subject"] != subject
        test_mask = df["subject"] == subject

        x_train = df.loc[train_mask, FEATURES].to_numpy()
        y_train = df.loc[train_mask, "object"].to_numpy()

        x_test = df.loc[test_mask, FEATURES].to_numpy()
        y_test = df.loc[test_mask, "object"].to_numpy()

        checkpoint = (
            output_dir / "checkpoints" /
            f"loso_subject_{subject}.weights.h5"
        )

        accuracy, macro_f1, cm, epochs_ran = train_one_fold(
            x_train,
            y_train,
            x_test,
            y_test,
            checkpoint,
        )

        rows.append(
            {
                "held_out_subject": subject,
                "train_samples": len(y_train),
                "test_samples": len(y_test),
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "epochs_ran": epochs_ran,
            }
        )
        confusion_matrices.append(cm)

        print(
            f"Subject {subject:2d}: "
            f"accuracy={accuracy:.4f}, "
            f"macro_f1={macro_f1:.4f}, "
            f"epochs={epochs_ran}"
        )

    results = pd.DataFrame(rows)
    results.to_csv(
        output_dir / "loso_results.csv",
        index=False,
    )

    mean_cm = np.mean(confusion_matrices, axis=0)
    save_confusion_matrix(
        mean_cm,
        output_dir / "loso_mean_confusion_matrix.csv",
    )

    print("\nLOSO mean")
    print(f"Accuracy : {results['accuracy'].mean() * 100:.2f}%")
    print(f"Macro F1 : {results['macro_f1'].mean() * 100:.2f}%")
    print("Paper    : 70.44% accuracy, 64.88% F1")


def main() -> None:
    args = parse_args()

    tf.keras.utils.set_random_seed(args.seed)

    features_path = args.features.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    checkpoint_dir = output_dir / "checkpoints"

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    df = prepare_dataframe(features_path)

    print("RadIOCD DDNN baseline reproduction")
    print("----------------------------------")
    print(f"Features: {features_path}")
    print(f"Samples : {len(df)}")
    print(f"Input   : {len(FEATURES)} handcrafted features")
    print("Network : 256 -> 256 -> 128 -> 128 -> 5")
    print("Dropout : 0.1 after every hidden layer")
    print("Batch   : 128")
    print("Epochs  : up to 1000")
    print("Stop    : val_accuracy patience=50")
    print()
    print("Class counts:")
    for name, label in CLASS_MAP.items():
        print(
            f"  {name:10s}: "
            f"{int((df['object'] == label).sum())}"
        )

    if args.mode in ("10fold", "both"):
        run_10fold(df, output_dir)

    if args.mode in ("loso", "both"):
        run_loso(df, output_dir)


if __name__ == "__main__":
    main()
