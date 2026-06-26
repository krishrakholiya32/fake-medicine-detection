"""Train an EfficientNet-B0 112-way pill identity classifier.

Two-phase fine-tune: freeze backbone + train head, then unfreeze last N blocks at lower LR.
Same recipe as the deepfake-detection project's train.py, adapted to multi-class.

Usage:
    python train.py --config ../configs/train_config.yaml
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_dataloaders(cfg: dict):
    train_tfms = transforms.Compose([
        transforms.Resize((cfg["image_size"], cfg["image_size"])),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_tfms = transforms.Compose([
        transforms.Resize((cfg["image_size"], cfg["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    data_dir = Path(cfg["data_dir"])
    train_ds = datasets.ImageFolder(data_dir / "train", transform=train_tfms)
    val_ds = datasets.ImageFolder(data_dir / "val", transform=eval_tfms)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                               num_workers=cfg["num_workers"], pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False,
                             num_workers=cfg["num_workers"], pin_memory=True)
    return train_loader, val_loader, train_ds.classes


def build_model(model_name: str, num_classes: int) -> nn.Module:
    return timm.create_model(model_name, pretrained=True, num_classes=num_classes)


def set_backbone_frozen(model: nn.Module, frozen: bool, unfreeze_blocks: int):
    for param in model.parameters():
        param.requires_grad = not frozen
    if frozen:
        for param in model.get_classifier().parameters():
            param.requires_grad = True
    elif unfreeze_blocks:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.get_classifier().parameters():
            param.requires_grad = True
        blocks = list(model.blocks)
        for block in blocks[-unfreeze_blocks:]:
            for param in block.parameters():
                param.requires_grad = True


def top_k_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    topk = logits.topk(k, dim=1).indices
    correct = topk.eq(labels.unsqueeze(1)).any(dim=1)
    return correct.float().mean().item()


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(mode=train)
    total_loss = 0.0
    total_top1, total_top3, n = 0.0, 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_top1 += top_k_accuracy(logits, labels, 1) * batch_size
        total_top3 += top_k_accuracy(logits, labels, min(3, logits.size(1))) * batch_size
        n += batch_size

    return total_loss / n, total_top1 / n, total_top3 / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_loader, val_loader, classes = build_dataloaders(cfg)
    print(f"{len(classes)} classes")

    model = build_model(cfg["model_name"], len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()

    run_dir = Path(cfg["output_dir"]) / cfg["run_name"]
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "labels.json", "w") as f:
        json.dump(classes, f, indent=2)

    best_top1 = -1.0
    epochs_without_improvement = 0
    history = []

    set_backbone_frozen(model, frozen=True, unfreeze_blocks=0)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg["freeze_lr"])
    epoch = 0
    for _ in range(cfg["freeze_epochs"]):
        epoch += 1
        train_loss, train_top1, train_top3 = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_top1, val_top3 = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        print(f"[head] epoch {epoch}: train_loss={train_loss:.4f} train_top1={train_top1:.4f} "
              f"val_loss={val_loss:.4f} val_top1={val_top1:.4f} val_top3={val_top3:.4f}")
        history.append({"epoch": epoch, "phase": "head", "train_loss": train_loss, "val_loss": val_loss,
                         "val_top1": val_top1, "val_top3": val_top3})
        if val_top1 > best_top1:
            best_top1 = val_top1
            torch.save(model.state_dict(), ckpt_dir / "best.pt")

    set_backbone_frozen(model, frozen=False, unfreeze_blocks=cfg["unfreeze_blocks"])
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg["finetune_lr"],
                                  weight_decay=cfg["weight_decay"])
    for _ in range(cfg["finetune_epochs"]):
        epoch += 1
        train_loss, train_top1, train_top3 = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_top1, val_top3 = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        print(f"[finetune] epoch {epoch}: train_loss={train_loss:.4f} train_top1={train_top1:.4f} "
              f"val_loss={val_loss:.4f} val_top1={val_top1:.4f} val_top3={val_top3:.4f}")
        history.append({"epoch": epoch, "phase": "finetune", "train_loss": train_loss, "val_loss": val_loss,
                         "val_top1": val_top1, "val_top3": val_top3})

        if val_top1 > best_top1:
            best_top1 = val_top1
            epochs_without_improvement = 0
            torch.save(model.state_dict(), ckpt_dir / "best.pt")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg["early_stopping_patience"]:
                print(f"early stopping at epoch {epoch} (best val_top1={best_top1:.4f})")
                break

    with open(run_dir / "metrics.json", "w") as f:
        json.dump({"history": history, "best_val_top1": best_top1}, f, indent=2)

    print(f"done. best val_top1={best_top1:.4f}. best checkpoint: {ckpt_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
