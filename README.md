# Skinable — Face Skin Condition Recognition

A transfer-learning image classifier that recognizes facial **skin conditions**
from photos. The final model classifies an image into one of six conditions:

**Acne · Black Spot · Eyebag · Oilness · Redness · Wrinkle**

The production model is an **EfficientNetV2B0** backbone (pre-trained on
ImageNet) with a custom classification head, reaching **90.2% test accuracy**
on a held-out split.

> ⚕️ This is a learning/portfolio project, **not** a medical or diagnostic tool.

---

## Results

| Class       | precision | recall | f1   | test support |
|-------------|-----------|--------|------|--------------|
| Wrinkle     | 0.99      | 0.92   | 0.95 | 432 |
| Redness     | 0.94      | 0.91   | 0.92 | 33  |
| Acne        | 0.76      | 0.80   | 0.78 | 55  |
| Black Spot  | 0.68      | 0.88   | 0.77 | 59  |
| Oilness     | 0.60      | 0.86   | 0.71 | 7   |
| Eyebag      | 0.21      | 0.75   | 0.33 | 4   |
| **weighted avg** | **0.93** | **0.90** | **0.91** | **590** |

**Note on the numbers:** weighted accuracy (90%) is high partly because the test
set is dominated by *Wrinkle* (73%). The fairer **macro precision is ~0.70** —
the rare classes (*Eyebag*, *Oilness*) have very little data, so their metrics
are noisy. More labelled examples of those classes is the real path to improving
them. The confusion matrix is in [`outputs/roboflow_confusion_matrix.png`](outputs/roboflow_confusion_matrix.png).

---

## Dataset

**Roboflow Universe — "Skin_Analysis" (FarmasiSkinCare)**
🔗 https://universe.roboflow.com/farmasiskincare-aan8y/skin_analysis

- 12,129 images, **object-detection** format (bounding boxes), 1 version
- Classes: Acne, Black Spot, Eyebag, Oilness, Redness, Wrinkle
- Splits: train 9,550 / valid 1,828 / test 751

### Downloading it

The dataset is **not** committed to this repo (too large). Download it from
Roboflow with your own free API key:

```bash
# 1) Get the COCO export download link (returns JSON with a "link" field)
curl -s "https://api.roboflow.com/farmasiskincare-aan8y/skin_analysis/1/coco?api_key=YOUR_ROBOFLOW_API_KEY"

# 2) Download and unzip the export into data/roboflow/
curl -L -o data/roboflow.zip "PASTE_THE_LINK_FROM_STEP_1"
unzip data/roboflow.zip -d data/roboflow
```

Or, with the Roboflow Python SDK:

```python
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_ROBOFLOW_API_KEY")
project = rf.workspace("farmasiskincare-aan8y").project("skin_analysis")
project.version(1).download("coco", location="data/roboflow")
```

> The original prototype also used the Kaggle dataset
> [skin-defects-acne-redness-and-bags-under-the-eyes](https://www.kaggle.com/datasets/trainingdatapro/skin-defects-acne-redness-and-bags-under-the-eyes)
> (only ~90 images — too small to train on; see `main.py` / `train_cv.py`).

---

## Setup

```bash
# Python 3.10 recommended (a conda env named "skinable" was used in development)
conda create -n skinable python=3.10 -y
conda activate skinable

pip install -r requirements.txt
```

---

## Pipeline

The detection dataset is converted into a classification dataset by **cropping
each annotated bounding box** into its own labelled image, then training a
classifier on those crops.

```bash
# 1) Crop the Roboflow boxes into data/crops/<split>/<class>/*.jpg
python prepare_crops.py

# 2) Train the classifier (frozen EfficientNetV2 backbone + cached embeddings + head)
python train_roboflow.py        # -> outputs/skinable_model.keras  (the 90% model)

# Optional: end-to-end fine-tuning of the backbone (slow on CPU)
python train_finetune.py
```

### Predict on a new image

```bash
python predict.py path/to/face.jpg
```

```
Prediction for path/to/face.jpg:
  -> Redness  (100.0%)
All classes:
  Redness    100.0%
  Wrinkle      0.0%
  ...
```

> The model classifies a **single region/image**. On a full-face photo it
> returns the dominant condition; localized multi-defect analysis would need an
> object detector to find regions first, then classify each crop.

---

## Project layout

```
skinable/
├── main.py                 # baseline prototype on the small Kaggle set (MobileNetV2)
├── train_cv.py             # subject-level k-fold CV on the Kaggle set (EfficientNetV2)
├── prepare_crops.py        # COCO boxes -> cropped classification dataset
├── train_roboflow.py       # main training: cached embeddings + head  (90% model)
├── train_finetune.py       # optional end-to-end fine-tuning
├── predict.py              # single-image inference
├── requirements.txt
├── outputs/
│   ├── skinable_model.keras            # trained model
│   ├── labels.json                     # class index -> name
│   └── roboflow_confusion_matrix.png   # test-set confusion matrix
└── data/                   # NOT committed — download from Roboflow (see above)
```

---

## Model details

- **Backbone:** EfficientNetV2B0, ImageNet weights, frozen
- **Head:** GlobalAveragePooling → Dropout(0.3) → Dense(256, relu) → Dropout(0.3) → Dense(6, softmax)
- **Training trick:** backbone features are cached once to `.npy`, so the head
  trains in seconds — practical on a CPU-only machine
- **Imbalance handling:** balanced class weights; per-class crop caps in `prepare_crops.py`
- **Input:** 224×224 RGB (EfficientNetV2 preprocessing baked into the model graph)
