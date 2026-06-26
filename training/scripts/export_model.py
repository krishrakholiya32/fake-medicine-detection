"""Export a trained pill classifier checkpoint to ONNX for CPU inference, and verify the export.

Usage:
    python export_model.py --config ../configs/train_config.yaml \
        --checkpoint /kaggle/working/runs/pillclf_effnetb0_v1/checkpoints/best.pt \
        --labels /kaggle/working/runs/pillclf_effnetb0_v1/labels.json \
        --output ../../models/pill_classifier_effnetb0.onnx
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import onnxruntime as ort
import timm
import torch
import yaml


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    with open(args.labels) as f:
        labels = json.load(f)

    model = timm.create_model(cfg["model_name"], pretrained=False, num_classes=len(labels))
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()

    size = cfg["image_size"]
    dummy = torch.randn(1, 3, size, size)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(args.output),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,  # legacy TorchScript-based exporter; avoids the onnxscript/onnx_ir dependency
    )
    print(f"exported to {args.output}")

    with torch.no_grad():
        torch_out = model(dummy).numpy()

    sess = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    onnx_out = sess.run(["logits"], {"input": dummy.numpy()})[0]

    max_diff = np.abs(torch_out - onnx_out).max()
    print(f"max abs diff in logits (torch vs onnx): {max_diff:.6f}")

    # Compare in probability space (post-softmax), not raw logits — small numerical
    # differences from backend kernel variation barely move predicted class probabilities.
    torch_prob = softmax(torch_out)
    onnx_prob = softmax(onnx_out)
    max_prob_diff = np.abs(torch_prob - onnx_prob).max()
    print(f"max abs diff in probability (torch vs onnx): {max_prob_diff:.6f}")
    assert max_prob_diff < 0.01, "ONNX export does not match PyTorch output — do not trust this artifact"
    print("export verified OK")

    # Place labels.json alongside the model so the backend has the same class order.
    labels_dst = args.output.parent / "labels.json"
    shutil.copy2(args.labels, labels_dst)
    print(f"copied labels to {labels_dst}")


if __name__ == "__main__":
    main()
