"""
Skinable — multi-class skin-condition classifier (Roboflow crops)
=================================================================

Trains on the cropped Roboflow dataset (see prepare_crops.py). Because this
machine has no GPU (2 CPUs), we use a *cached-embeddings* strategy instead of
end-to-end fine-tuning:

  1. Run every crop through a FROZEN EfficientNetV2B0 backbone exactly ONCE and
     cache the pooled feature vectors to .npy (the expensive step, done once).
  2. Train a small MLP head on those cached features — seconds per epoch, with
     class weights to counter imbalance.
  3. Evaluate on the held-out valid + test splits and save model + report.

This is dramatically faster than fine-tuning on CPU (which repeats the backbone
forward pass every epoch) and works well when the backbone is frozen anyway.

Run:  python train_roboflow.py
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
CACHE_DIR = OUT_DIR / "embeddings"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 60
SEED = 42

tf.random.set_seed(SEED)
np.random.seed(SEED)


def list_crops(split: str, class_names):
    """Return (paths, labels) for one split, using a fixed class ordering."""
    label_to_idx = {c: i for i, c in enumerate(class_names)}
    paths, labels = [], []
    split_dir = CROPS_DIR / split
    for cls in class_names:
        for p in sorted((split_dir / cls).glob("*.jpg")):
            paths.append(str(p)); labels.append(label_to_idx[cls])
    return paths, np.array(labels)


def embed(paths, backbone, cache_path: Path):
    """Compute (or load cached) EfficientNetV2 embeddings for a list of crops."""
    if cache_path.exists():
        print(f"  loading cached embeddings {cache_path.name}")
        return np.load(cache_path)

    def _load(p):
        img = tf.io.read_file(p)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        return tf.keras.applications.efficientnet_v2.preprocess_input(img)

    ds = (tf.data.Dataset.from_tensor_slices(paths)
          .map(_load, num_parallel_calls=tf.data.AUTOTUNE)
          .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))
    feats = backbone.predict(ds, verbose=1)
    np.save(cache_path, feats)
    return feats


def build_head(input_dim, num_classes):
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ], name="skinable_head")


def plot_confusion(y_true, y_pred, class_names, acc, path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"Test confusion (acc {acc:.3f})")
    thresh = cm.max() / 2
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=8)
    fig.colorbar(im); fig.tight_layout()
    fig.savefig(path, dpi=120)
    print("Saved", path)


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print("TensorFlow:", tf.__version__)

    class_names = sorted([d.name for d in (CROPS_DIR / "train").iterdir() if d.is_dir()])
    print("Classes:", class_names)

    backbone = tf.keras.Sequential([
        tf.keras.applications.EfficientNetV2B0(
            input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet"),
        tf.keras.layers.GlobalAveragePooling2D(),
    ])
    backbone.trainable = False

    print("\nEmbedding crops (cached after first run):")
    splits = {}
    for split in ["train", "valid", "test"]:
        paths, labels = list_crops(split, class_names)
        print(f"  {split}: {len(paths)} crops")
        X = embed(paths, backbone, CACHE_DIR / f"{split}_X.npy")
        splits[split] = (X, labels)

    Xtr, ytr = splits["train"]
    Xva, yva = splits["valid"]
    Xte, yte = splits["test"]

    cw = compute_class_weight("balanced", classes=np.arange(len(class_names)), y=ytr)
    class_weight = {i: float(w) for i, w in enumerate(cw)}
    print("Class weights:", {class_names[i]: round(w, 2) for i, w in class_weight.items()})

    head = build_head(Xtr.shape[1], len(class_names))
    head.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                 loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    head.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=EPOCHS,
             batch_size=BATCH_SIZE, class_weight=class_weight,
             callbacks=[tf.keras.callbacks.EarlyStopping(
                 monitor="val_accuracy", patience=10, restore_best_weights=True)],
             verbose=2)

    print("\n=== TEST SET evaluation ===")
    yprob = head.predict(Xte, verbose=0)
    ypred = yprob.argmax(axis=1)
    acc = float((ypred == yte).mean())
    print(f"Test accuracy: {acc:.3f}")
    print(classification_report(yte, ypred, target_names=class_names, zero_division=0))
    plot_confusion(yte, ypred, class_names, acc, OUT_DIR / "roboflow_confusion_matrix.png")

    # Save a full end-to-end model (backbone + head) so predict.py works on raw images.
    inp = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = tf.keras.applications.efficientnet_v2.preprocess_input(inp)
    x = backbone(x)
    out = head(x)
    full = tf.keras.Model(inp, out, name="skinable_roboflow")
    full.save(OUT_DIR / "skinable_model.keras")
    (OUT_DIR / "labels.json").write_text(json.dumps(class_names, indent=2))
    print(f"\nSaved model -> {OUT_DIR / 'skinable_model.keras'}")
    print(f"Final TEST accuracy: {acc:.1%}")


if __name__ == "__main__":
    main()
