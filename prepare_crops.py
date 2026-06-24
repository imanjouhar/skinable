"""
Skinable — build a classification dataset from the Roboflow detection set
=========================================================================

The Roboflow "Skin_Analysis" dataset is object-detection (bounding boxes around
skin features). We convert it into an image-classification dataset by cropping
each annotated box into its own image, labelled by the box's class. This gives
thousands of real labelled examples — the fix for the tiny-data ceiling.

  COCO box [x, y, w, h]  ->  padded crop  ->  data/crops/<split>/<class>/<id>.jpg

Choices:
  * Skip the COCO supercategory and tiny boxes (< MIN_SIDE px) — they're noise.
  * Cap per-class counts in TRAIN to curb extreme imbalance and CPU time.
  * Keep all valid/test crops for honest evaluation.

Run:  python prepare_crops.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent
RF_DIR = PROJECT_DIR / "data" / "roboflow"
OUT_DIR = PROJECT_DIR / "data" / "crops"

CROP_SIZE = (224, 224)
PAD_FRAC = 0.12          # pad each box by 12% of its size for context
MIN_SIDE = 24            # skip boxes whose width or height is smaller than this
MAX_PER_CLASS_TRAIN = 1500   # cap train crops per class (balance + speed)
SKIP_FULL_IMAGE = True   # drop boxes that cover ~the entire frame (uninformative)


def crop_split(split: str, cap: int | None):
    ann_path = RF_DIR / split / "_annotations.coco.json"
    if not ann_path.exists():
        print(f"[skip] {ann_path} not found")
        return {}

    coco = json.loads(ann_path.read_text())
    id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    images = {img["id"]: img for img in coco["images"]}

    # Group annotations by class so we can apply per-class caps deterministically.
    by_class = defaultdict(list)
    for ann in coco["annotations"]:
        by_class[id_to_name[ann["category_id"]]].append(ann)

    counts = defaultdict(int)
    for cls, anns in sorted(by_class.items()):
        anns.sort(key=lambda a: (a["image_id"], a["id"]))  # stable order
        for ann in anns:
            if cap is not None and counts[cls] >= cap:
                break
            img_info = images[ann["image_id"]]
            x, y, w, h = ann["bbox"]
            iw, ih = img_info["width"], img_info["height"]

            if w < MIN_SIDE or h < MIN_SIDE:
                continue
            if SKIP_FULL_IMAGE and w >= 0.98 * iw and h >= 0.98 * ih:
                continue

            # pad and clip to image bounds
            px, py = w * PAD_FRAC, h * PAD_FRAC
            x0 = max(0, int(x - px)); y0 = max(0, int(y - py))
            x1 = min(iw, int(x + w + px)); y1 = min(ih, int(y + h + py))
            if x1 - x0 < MIN_SIDE or y1 - y0 < MIN_SIDE:
                continue

            src = RF_DIR / split / img_info["file_name"]
            if not src.exists():
                continue
            try:
                img = Image.open(src).convert("RGB")
                crop = img.crop((x0, y0, x1, y1)).resize(CROP_SIZE)
            except Exception as e:
                print(f"  [warn] {src.name}: {e}")
                continue

            out_dir = OUT_DIR / split / cls
            out_dir.mkdir(parents=True, exist_ok=True)
            crop.save(out_dir / f"{ann['image_id']}_{ann['id']}.jpg", quality=90)
            counts[cls] += 1

    print(f"\n[{split}] crops per class:")
    for cls in sorted(counts):
        print(f"  {cls:<12} {counts[cls]}")
    print(f"[{split}] total: {sum(counts.values())}")
    return dict(counts)


def main():
    if OUT_DIR.exists():
        print(f"Note: {OUT_DIR} already exists; new crops are added/overwritten.")
    print("Cropping boxes into a classification dataset...")
    crop_split("train", MAX_PER_CLASS_TRAIN)
    crop_split("valid", None)
    crop_split("test", None)
    print("\nDone. Crops written to", OUT_DIR)


if __name__ == "__main__":
    main()
