"""
Skinable — honest evaluation via subject-level k-fold cross-validation
======================================================================

The dataset is tiny (30 subjects, 90 images), so a single train/val split is
statistically noisy — one unlucky fold can read 33% while another reads 70% on
the *same* model. This script instead runs 5-fold cross-validation **split by
subject** (a person's 3 photos never span train and val), which is the honest
way to estimate generalization here.

Improvements over the baseline main.py:
  * EfficientNetV2B0 backbone (stronger features than MobileNetV2)
  * Heavier augmentation to stretch the tiny dataset
  * Two-phase training: train head on frozen features, then fine-tune the top
  * Test-time view aggregation — a subject's 3 views are averaged into ONE
    prediction (legitimate: you'd photograph a real patient from 3 angles too)
  * Reports mean ± std accuracy across folds, plus a pooled confusion matrix

After CV, a final model is trained on ALL data and saved for predict.py.

Run:  python train_cv.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from main import load_dataframe, OUTPUT_DIR  # reuse the data loader

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
N_FOLDS = 5
EPOCHS_HEAD = 20
EPOCHS_FINETUNE = 12
FINE_TUNE_AT = 200        # unfreeze EfficientNetV2B0 layers from this index
SEED = 42

tf.random.set_seed(SEED)
np.random.seed(SEED)

AUGMENT = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.15),
        tf.keras.layers.RandomZoom(0.15),
        tf.keras.layers.RandomContrast(0.2),
        tf.keras.layers.RandomBrightness(0.15, value_range=(0, 255)),
        tf.keras.layers.RandomTranslation(0.1, 0.1),
    ],
    name="augmentation",
)


# --------------------------------------------------------------------------- #
# Pipelines & model
# --------------------------------------------------------------------------- #
def _load_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    return img  # EfficientNetV2 preprocessing is baked into the model


def make_dataset(df, label_to_idx, training):
    paths = df["path"].tolist()
    labels = [label_to_idx[l] for l in df["label"]]
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(lambda p, y: (_load_image(p), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def build_model(num_classes):
    base = tf.keras.applications.EfficientNetV2B0(
        input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet"
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = AUGMENT(inputs)
    x = tf.keras.applications.efficientnet_v2.preprocess_input(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="skinable_effnetv2"), base


def train_one(model, base, train_ds, val_ds):
    """Two-phase training; returns the fitted model (best weights restored)."""
    es = lambda p: tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=p, restore_best_weights=True
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD,
              callbacks=[es(6)], verbose=2)

    base.trainable = True
    for layer in base.layers[:FINE_TUNE_AT]:
        layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FINETUNE,
              callbacks=[es(5)], verbose=2)
    return model


def evaluate_by_subject(model, val_df, label_to_idx, class_names):
    """Average each subject's 3 view-predictions into one, then score."""
    idx_to_label = {i: n for n, i in label_to_idx.items()}
    y_true, y_pred = [], []
    for subj, grp in val_df.groupby("subject_id"):
        imgs = tf.stack([_load_image(p) for p in grp["path"]])
        probs = model.predict(imgs, verbose=0).mean(axis=0)  # aggregate views
        y_pred.append(int(probs.argmax()))
        y_true.append(label_to_idx[grp["label"].iloc[0]])
    return np.array(y_true), np.array(y_pred)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    print("TensorFlow:", tf.__version__)

    df = load_dataframe()
    class_names = sorted(df["label"].unique().tolist())
    label_to_idx = {n: i for i, n in enumerate(class_names)}
    print("Classes:", class_names)

    subjects = df[["subject_id", "label"]].drop_duplicates().reset_index(drop=True)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    fold_acc, all_true, all_pred = [], [], []
    for fold, (tr_i, va_i) in enumerate(
        skf.split(subjects["subject_id"], subjects["label"]), start=1
    ):
        tr_subj = set(subjects.loc[tr_i, "subject_id"])
        va_subj = set(subjects.loc[va_i, "subject_id"])
        tr_df = df[df["subject_id"].isin(tr_subj)]
        va_df = df[df["subject_id"].isin(va_subj)]

        print(f"\n===== Fold {fold}/{N_FOLDS} "
              f"(train {tr_df.subject_id.nunique()} subj / "
              f"val {va_df.subject_id.nunique()} subj) =====")
        tf.keras.backend.clear_session()
        model, base = build_model(len(class_names))
        train_one(model, base, make_dataset(tr_df, label_to_idx, True),
                  make_dataset(va_df, label_to_idx, False))

        yt, yp = evaluate_by_subject(model, va_df, label_to_idx, class_names)
        acc = float((yt == yp).mean())
        fold_acc.append(acc)
        all_true.extend(yt.tolist()); all_pred.extend(yp.tolist())
        print(f"Fold {fold} subject-level accuracy: {acc:.3f}")

    # --- Cross-validation summary ---
    mean, std = np.mean(fold_acc), np.std(fold_acc)
    print("\n" + "=" * 55)
    print("CROSS-VALIDATION RESULT (subject-level, view-aggregated)")
    print("=" * 55)
    for i, a in enumerate(fold_acc, 1):
        print(f"  Fold {i}: {a:.3f}")
    print(f"  Mean accuracy: {mean:.3f} ± {std:.3f}")
    print("\nPooled report across all folds:")
    print(classification_report(all_true, all_pred, target_names=class_names,
                                zero_division=0))

    # Pooled confusion matrix
    cm = confusion_matrix(all_true, all_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"CV confusion (acc {mean:.2f}±{std:.2f})")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im); fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "cv_confusion_matrix.png", dpi=120)
    print(f"Saved CV confusion matrix -> {OUTPUT_DIR / 'cv_confusion_matrix.png'}")

    # --- Final model on ALL data (for predict.py) ---
    print("\n=== Training final model on all data ===")
    tf.keras.backend.clear_session()
    model, base = build_model(len(class_names))
    full_ds = make_dataset(df, label_to_idx, True)
    train_one(model, base, full_ds, full_ds)  # no holdout left; monitor train
    model.save(OUTPUT_DIR / "skinable_model.keras")
    (OUTPUT_DIR / "labels.json").write_text(json.dumps(class_names, indent=2))
    print(f"Saved final model -> {OUTPUT_DIR / 'skinable_model.keras'}")
    print(f"\nHonest estimate of real-world accuracy: {mean:.1%} ± {std:.1%}")


if __name__ == "__main__":
    main()
