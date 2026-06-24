"""
Skinable — fine-tuned skin-condition classifier (Roboflow crops)
================================================================

Upgrade over train_roboflow.py (frozen feature-extraction). Here we actually
FINE-TUNE the EfficientNetV2B0 backbone end-to-end, which usually beats frozen
features by a meaningful margin — at the cost of much more CPU time.

Two phases:
  1. Warm up a fresh head on the frozen backbone (few epochs, stable start).
  2. Unfreeze the top FINE_TUNE_AT layers and train end-to-end at a low LR,
     with augmentation + class weights to fight imbalance.

Slow on CPU (backprop through the backbone every epoch). Runs in the background.

Run:  python train_finetune.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent
CROPS_DIR = PROJECT_DIR / "data" / "crops"
OUT_DIR = PROJECT_DIR / "outputs"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_WARMUP = 8
EPOCHS_FINETUNE = 25
FINE_TUNE_AT = 150        # unfreeze EfficientNetV2B0 layers from this index
SEED = 42

tf.random.set_seed(SEED)
np.random.seed(SEED)

AUGMENT = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.15),
    tf.keras.layers.RandomBrightness(0.1, value_range=(0, 255)),
], name="augmentation")


def list_crops(split, class_names):
    label_to_idx = {c: i for i, c in enumerate(class_names)}
    paths, labels = [], []
    for cls in class_names:
        for p in sorted((CROPS_DIR / split / cls).glob("*.jpg")):
            paths.append(str(p)); labels.append(label_to_idx[cls])
    return paths, np.array(labels)


def make_ds(paths, labels, training):
    def _load(p, y):
        img = tf.io.read_file(p)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        return img, y
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def build_model(num_classes):
    base = tf.keras.applications.EfficientNetV2B0(
        input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet")
    base.trainable = False
    inp = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = AUGMENT(inp)
    x = tf.keras.applications.efficientnet_v2.preprocess_input(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inp, out, name="skinable_finetuned"), base


def main():
    OUT_DIR.mkdir(exist_ok=True)
    print("TensorFlow:", tf.__version__)
    class_names = sorted([d.name for d in (CROPS_DIR / "train").iterdir() if d.is_dir()])
    print("Classes:", class_names)

    tr_p, tr_y = list_crops("train", class_names)
    va_p, va_y = list_crops("valid", class_names)
    te_p, te_y = list_crops("test", class_names)
    print(f"train {len(tr_p)} / valid {len(va_p)} / test {len(te_p)}")

    train_ds = make_ds(tr_p, tr_y, True)
    val_ds = make_ds(va_p, va_y, False)
    test_ds = make_ds(te_p, te_y, False)

    cw = compute_class_weight("balanced", classes=np.arange(len(class_names)), y=tr_y)
    class_weight = {i: float(w) for i, w in enumerate(cw)}
    print("Class weights:", {class_names[i]: round(w, 2) for i, w in class_weight.items()})

    model, base = build_model(len(class_names))
    es = lambda p: tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=p, restore_best_weights=True)

    print("\n=== Phase 1: warm up head (frozen backbone) ===")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_WARMUP,
              class_weight=class_weight, callbacks=[es(4)], verbose=2)

    print("\n=== Phase 2: fine-tune backbone (low LR) ===")
    base.trainable = True
    for layer in base.layers[:FINE_TUNE_AT]:
        layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FINETUNE,
              class_weight=class_weight, callbacks=[es(6)], verbose=2)

    print("\n=== TEST SET evaluation ===")
    yprob = model.predict(test_ds, verbose=0)
    ypred = yprob.argmax(axis=1)
    acc = float((ypred == te_y).mean())
    print(f"Test accuracy: {acc:.3f}")
    print(classification_report(te_y, ypred, target_names=class_names, zero_division=0))

    cm = confusion_matrix(te_y, ypred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"Fine-tuned test confusion (acc {acc:.3f})")
    thresh = cm.max() / 2
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=8)
    fig.colorbar(im); fig.tight_layout()
    fig.savefig(OUT_DIR / "finetune_confusion_matrix.png", dpi=120)

    model.save(OUT_DIR / "skinable_finetuned.keras")
    (OUT_DIR / "labels.json").write_text(json.dumps(class_names, indent=2))
    print(f"\nSaved fine-tuned model -> {OUT_DIR / 'skinable_finetuned.keras'}")
    print(f"Final TEST accuracy: {acc:.1%}")


if __name__ == "__main__":
    main()
