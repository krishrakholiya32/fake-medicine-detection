"""Reorganize OGYEIv2 images (which ship with their own train/valid/test split,
filenames encoding the class, e.g. "indastad_1_5_mg_s_035.jpg") into the
{train,val,test}/<class_name>/*.jpg layout train.py expects.

Expected input layout:
    raw_dir/train/images/<class_name>_<s|u>_<index>.jpg
    raw_dir/valid/images/...
    raw_dir/test/images/...
    (a labels/ folder with YOLO-format segmentation polygons also exists per split,
     but isn't needed here — this is whole-image classification, not detection)

Output layout:
    output_dir/{train,val,test}/<class_name>/*.jpg
    output_dir/labels.json   — sorted list of class names (index == model output index)

Usage:
    python prepare_dataset.py --raw_dir /kaggle/input/datasets/richardradli/ogyeiv2/ogyeiv2/ogyeiv2 \
        --output_dir /kaggle/working/data
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

SPLIT_MAP = {"train": "train", "valid": "val", "test": "test"}


def class_name_from_filename(filename: str) -> str:
    """'indastad_1_5_mg_s_035.jpg' -> 'indastad_1_5_mg' (strip trailing _<s|u>_<index>)."""
    stem = Path(filename).stem
    parts = stem.split("_")
    return "_".join(parts[:-2])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_classes: set[str] = set()

    for src_split, dst_split in SPLIT_MAP.items():
        images_dir = args.raw_dir / src_split / "images"
        if not images_dir.exists():
            print(f"skip (not found): {images_dir}")
            continue

        images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        print(f"[{dst_split}] {len(images)} images")

        for img in images:
            class_name = class_name_from_filename(img.name)
            all_classes.add(class_name)
            dst_dir = args.output_dir / dst_split / class_name
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst_path = dst_dir / img.name
            if not dst_path.exists():
                shutil.copy2(img, dst_path)

    labels = sorted(all_classes)
    with open(args.output_dir / "labels.json", "w") as f:
        json.dump(labels, f, indent=2)
    print(f"found {len(labels)} classes, labels.json written")


if __name__ == "__main__":
    main()
