"""
Skinable — Face Skin Condition Recognition
===========================================

Transfer-learning image classifier that recognizes facial skin conditions from
photos. The dataset (Kaggle: trainingdatapro/skin-defects-acne-redness-and-bags-
under-the-eyes) contains 30 subjects across 3 conditions, each photographed from
3 angles (front / left / right):

    acne   |  redness  |  bags (under the eyes)

Approach
--------
The dataset is small (~90 images), so we use transfer learning on top of a
MobileNetV2 backbone pre-trained on ImageNet:

  1. Load images + labels from skin_defects.csv.
  2. Split TRAIN/VAL by *subject* so the same person's 3 views never leak across
     the split (prevents over-optimistic validation accuracy).
  3. Train a small classification head on frozen features.
  4. Optionally fine-tune the top layers of the backbone.
  5. Save the model, a label map, training curves, and a confusion matrix.

Run:  python main.py
"""

from __future__ import annotations

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib

matplotlib.use("Agg")  # headless backend — save plots to disk
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "extracted"
IMAGES_ROOT = DATA_DIR / "files"
CSV_PATH = DATA_DIR / "skin_defects.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS_HEAD = 25          # epochs training only the new head (backbone frozen)
EPOCHS_FINETUNE = 15      # epochs fine-tuning the top of the backbone
FINE_TUNE_AT = 120        # unfreeze MobileNetV2 layers from this index onward
VAL_FRACTION = 0.2        # fraction of subjects held out for validation
SEED = 42

tf.random.set_seed(SEED)
np.random.seed(SEED)


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_dataframe() -> pd.DataFrame:
    """Read the CSV and expand each subject row into one row per image view.

    The CSV has columns: id, front, left_side, right_side, type
    where the image paths are relative to IMAGES_ROOT (e.g. /acne/0/front.jpg).
    Returns a long-format frame with columns: subject_id, path, label.
    """
    df = pd.read_csv(CSV_PATH)
    view_cols = ["front", "left_side", "right_side"]

    rows = []
    for _, r in df.iterrows():
        for col in view_cols:
            rel = str(r[col]).lstrip("/")          # "/acne/0/front.jpg" -> "acne/0/front.jpg"
            abs_path = IMAGES_ROOT / rel
            if abs_path.exists():
                rows.append(
                    {"subject_id": int(r["id"]), "path": str(abs_path), "label": r["type"]}
                )
            else:
                print(f"  [warn] missing image: {abs_path}")

    long_df = pd.DataFrame(rows)
    print(f"Loaded {len(long_df)} images across {long_df['subject_id'].nunique()} subjects.")
    print(long_df["label"].value_counts().to_string())
    return long_df


def subject_level_split(df: pd.DataFrame):
    """Split into train/val by subject so a subject's views stay together."""
    subjects = df[["subject_id", "label"]].drop_duplicates()
    train_subj, val_subj = train_test_split(
        subjects["subject_id"],
        test_size=VAL_FRACTION,
        random_state=SEED,
        stratify=subjects["label"],
    )
    train_df = df[df["subject_id"].isin(train_subj)].reset_index(drop=True)
    val_df = df[df["subject_id"].isin(val_subj)].reset_index(drop=True)
    print(f"Train: {len(train_df)} images / {train_df['subject_id'].nunique()} subjects")
    print(f"Val:   {len(val_df)} images / {val_df['subject_id'].nunique()} subjects")
    return train_df, val_df


def make_dataset(df: pd.DataFrame, class_names, training: bool) -> tf.data.Dataset:
    """Build a tf.data pipeline that decodes, resizes, and batches images."""
    label_to_idx = {name: i for i, name in enumerate(class_names)}
    paths = df["path"].tolist()
    labels = [label_to_idx[l] for l in df["label"]]

    def _load(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def build_model(num_classes: int) -> tf.keras.Model:
    """MobileNetV2 backbone + augmentation + classification head."""
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.1),
        ],
        name="augmentation",
    )

    base = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet"
    )
    base.trainable = False  # frozen for the head-training phase

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="skinable_classifier")
    return model, base


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_history(histories, path: Path):
    acc, val_acc, loss, val_loss = [], [], [], []
    for h in histories:
        acc += h.history["accuracy"]
        val_acc += h.history["val_accuracy"]
        loss += h.history["loss"]
        val_loss += h.history["val_loss"]

    epochs = range(1, len(acc) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs, acc, label="train")
    ax1.plot(epochs, val_acc, label="val")
    ax1.set_title("Accuracy"); ax1.set_xlabel("epoch"); ax1.legend()
    ax2.plot(epochs, loss, label="train")
    ax2.plot(epochs, val_loss, label="val")
    ax2.set_title("Loss"); ax2.set_xlabel("epoch"); ax2.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"Saved training curves -> {path}")


def plot_confusion(y_true, y_pred, class_names, path: Path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title("Confusion matrix")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"Saved confusion matrix -> {path}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    print("TensorFlow:", tf.__version__)
    print("GPUs:", tf.config.list_physical_devices("GPU") or "none (CPU)")

    df = load_dataframe()
    class_names = sorted(df["label"].unique().tolist())
    print("Classes:", class_names)

    train_df, val_df = subject_level_split(df)
    train_ds = make_dataset(train_df, class_names, training=True)
    val_ds = make_dataset(val_df, class_names, training=False)

    model, base = build_model(len(class_names))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=8, restore_best_weights=True
        ),
    ]

    # Phase 1 — train the head on frozen features.
    print("\n=== Phase 1: training classification head ===")
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD, callbacks=callbacks)

    # Phase 2 — fine-tune the top of the backbone at a low learning rate.
    print("\n=== Phase 2: fine-tuning backbone ===")
    base.trainable = True
    for layer in base.layers[:FINE_TUNE_AT]:
        layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FINETUNE, callbacks=callbacks)

    # Evaluation
    print("\n=== Evaluation on validation set ===")
    y_true = np.concatenate([y.numpy() for _, y in val_ds])
    y_prob = model.predict(val_ds)
    y_pred = y_prob.argmax(axis=1)
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    # Artifacts
    model_path = OUTPUT_DIR / "skinable_model.keras"
    model.save(model_path)
    print(f"Saved model -> {model_path}")
    with open(OUTPUT_DIR / "labels.json", "w") as f:
        json.dump(class_names, f, indent=2)
    plot_history([h1, h2], OUTPUT_DIR / "training_curves.png")
    plot_confusion(y_true, y_pred, class_names, OUTPUT_DIR / "confusion_matrix.png")
    print("\nDone. Artifacts written to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
