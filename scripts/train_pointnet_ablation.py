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
        description="Train PointNet on RadIOCD with selectable input channels."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("outputs/pointnet_xyzdi_32.npz"),
        help="NPZ produced by build_pointnet_dataset_xyzdi.py",
    )
    parser.add_argument(
        "--channels",
        type=int,
        choices=[3, 4, 5],
        default=3,
        help="3=XYZ, 4=XYZ+Doppler, 5=XYZ+Doppler+Intensity",
    )
    parser.add_argument(
        "--fold",
        type=int,
        choices=range(1, 11),
        default=1,
        help="Run a single 10-fold CV fold (1-10). Default: 1",
    )
    parser.add_argument(
        "--all-folds",
        action="store_true",
        help="Run all 10 folds instead of only --fold.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/pointnet_ablation"),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12,
    )
    return parser.parse_args()


def channel_names(n: int) -> list[str]:
    return ["X", "Y", "Z", "Doppler", "Intensity"][:n]


def normalize_channels(
    x_train: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Standardize each input channel using training-fold statistics only.
    """
    flat = x_train.reshape(-1, x_train.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std[std == 0] = 1.0

    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    return x_train.astype(np.float32), x_test.astype(np.float32)


def conv_block(x, filters: int):
    x = layers.Conv1D(filters, 1, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x


def dense_block(x, units: int, dropout: float = 0.3):
    x = layers.Dense(units, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(dropout)(x)
    return x


def build_pointnet(n_points: int, n_dims: int) -> keras.Model:
    inputs = keras.Input(shape=(n_points, n_dims), name="points")

    x = conv_block(inputs, 64)
    x = conv_block(x, 64)
    x = conv_block(x, 64)
    x = conv_block(x, 128)
    x = conv_block(x, 256)

    x = layers.GlobalMaxPooling1D()(x)

    x = dense_block(x, 256, 0.3)
    x = dense_block(x, 128, 0.3)

    outputs = layers.Dense(
        len(CLASS_NAMES),
        activation="softmax",
    )(x)

    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_fold(
    x_train,
    y_train,
    x_test,
    y_test,
    checkpoint_path: Path,
):
    x_train, x_test = normalize_channels(x_train, x_test)

    model = build_pointnet(
        n_points=x_train.shape[1],
        n_dims=x_train.shape[2],
    )

    early = keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=30,
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
        callbacks=[early, checkpoint, reduce_lr],
    )

    model.load_weights(checkpoint_path)

    probs = model.predict(x_test, batch_size=512, verbose=0)
    pred = np.argmax(probs, axis=1)

    acc = float(np.mean(pred == y_test))
    f1 = float(f1_score(y_test, pred, average="macro"))
    cm = confusion_matrix(y_test, pred, labels=[0,1,2,3,4])

    epochs_ran = len(history.history["loss"])

    del model, probs
    keras.backend.clear_session()

    return acc, f1, cm, epochs_ran


def main():
    args = parse_args()
    tf.keras.utils.set_random_seed(args.seed)

    data_path = args.data.expanduser().resolve()
    outdir = args.output_dir.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "checkpoints").mkdir(parents=True, exist_ok=True)

    data = np.load(data_path, allow_pickle=True)
    X_full = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    csv_names = data["csv"].astype(str)

    X = X_full[:, :, :args.channels]
    names = channel_names(args.channels)

    unique_csvs = np.unique(csv_names)
    kf = KFold(n_splits=10, random_state=12, shuffle=True)
    splits = list(kf.split(unique_csvs))

    folds_to_run = range(1, 11) if args.all_folds else [args.fold]

    print("RadIOCD PointNet channel ablation")
    print("--------------------------------")
    print(f"Data     : {data_path}")
    print(f"Channels : {names}")
    print(f"Shape    : {X.shape}")
    print(f"Folds    : {list(folds_to_run)}")
    print()

    rows = []
    cms = []

    for fold in folds_to_run:
        train_idx, test_idx = splits[fold - 1]

        train_csvs = unique_csvs[train_idx]
        test_csvs = unique_csvs[test_idx]

        train_mask = np.isin(csv_names, train_csvs)
        test_mask = np.isin(csv_names, test_csvs)

        tag = f"{args.channels}ch_fold{fold}"
        checkpoint = outdir / "checkpoints" / f"{tag}.weights.h5"

        acc, f1, cm, epochs_ran = train_fold(
            X[train_mask],
            y[train_mask],
            X[test_mask],
            y[test_mask],
            checkpoint,
        )

        rows.append({
            "fold": fold,
            "channels": "+".join(names),
            "accuracy": acc,
            "macro_f1": f1,
            "epochs_ran": epochs_ran,
            "train_samples": int(train_mask.sum()),
            "test_samples": int(test_mask.sum()),
        })
        cms.append(cm)

        print(
            f"Fold {fold:2d}: "
            f"accuracy={acc:.4f}, "
            f"macro_f1={f1:.4f}, "
            f"epochs={epochs_ran}"
        )

    result = pd.DataFrame(rows)
    result_path = outdir / f"results_{args.channels}ch.csv"
    result.to_csv(result_path, index=False)

    if args.all_folds:
        print()
        print("10-fold mean")
        print(f"Accuracy : {result['accuracy'].mean() * 100:.2f}%")
        print(f"Macro F1 : {result['macro_f1'].mean() * 100:.2f}%")

    print()
    print("Fold-1 reference from XYZ baseline:")
    print("Accuracy : 74.73%")
    print("Macro F1 : 70.88%")
    print(f"Saved    : {result_path}")


if __name__ == "__main__":
    main()
