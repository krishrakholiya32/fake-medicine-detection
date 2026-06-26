"""Evaluate a trained pill classifier checkpoint against the held-out test split.

Usage:
    python evaluate.py --config ../configs/train_config.yaml \
        --checkpoint /kaggle/working/runs/pillclf_effnetb0_v1/checkpoints/best.pt \
        --labels /kaggle/working/runs/pillclf_effnetb0_v1/labels.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import timm
import torch
import yaml
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    with open(args.labels) as f:
        labels = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tfms = transforms.Compose([
        transforms.Resize((cfg["image_size"], cfg["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    ds = datasets.ImageFolder(Path(cfg["data_dir"]) / args.split, transform=tfms)
    assert ds.classes == labels, "label order mismatch between training and this dataset split"
    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])

    model = timm.create_model(cfg["model_name"], pretrained=False, num_classes=len(labels))
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device).eval()

    all_labels, all_preds, all_top3_hits = [], [], []
    with torch.no_grad():
        for images, targets in loader:
            logits = model(images.to(device))
            preds = logits.argmax(dim=1).cpu()
            top3 = logits.topk(min(3, logits.size(1)), dim=1).indices.cpu()
            all_labels.extend(targets.numpy())
            all_preds.extend(preds.numpy())
            all_top3_hits.extend(top3.eq(targets.unsqueeze(1)).any(dim=1).numpy())

    acc = accuracy_score(all_labels, all_preds)
    top3_acc = sum(all_top3_hits) / len(all_top3_hits)

    print(f"split={args.split} n={len(all_labels)}")
    print(f"top-1 accuracy={acc:.4f} top-3 accuracy={top3_acc:.4f}")
    print(classification_report(all_labels, all_preds, target_names=labels, zero_division=0))


if __name__ == "__main__":
    main()
