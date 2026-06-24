"""
Skinable — single-image inference
==================================

Predict the skin condition for one face photo using the trained model.

Usage:
    python predict.py path/to/face.jpg
"""

import sys
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "outputs" / "skinable_model.keras"
LABELS_PATH = PROJECT_DIR / "outputs" / "labels.json"
IMG_SIZE = (224, 224)


def main():
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    if not MODEL_PATH.exists():
        print(f"Model not found at {MODEL_PATH}. Train it first with: python main.py")
        sys.exit(1)

    class_names = json.loads(LABELS_PATH.read_text())
    model = tf.keras.models.load_model(MODEL_PATH)

    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    batch = tf.expand_dims(img, 0)  # preprocessing is baked into the model graph

    probs = model.predict(batch, verbose=0)[0]
    order = np.argsort(probs)[::-1]

    print(f"\nPrediction for {image_path}:")
    print(f"  -> {class_names[order[0]]}  ({probs[order[0]] * 100:.1f}%)\n")
    print("All classes:")
    for i in order:
        print(f"  {class_names[i]:<10} {probs[i] * 100:5.1f}%")


if __name__ == "__main__":
    main()
