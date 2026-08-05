#!/usr/bin/env python3
"""Run a small UniDepthV2 inference smoke test."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from unidepth.models import UniDepthV2


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument(
        "--model",
        default="lpiccinelli/unidepth-v2-vits14",
        help="Hugging Face model identifier or local model directory",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--resolution-level", type=int, default=0)
    return parser.parse_args()


def select_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return torch.device(requested)


def tensor_summary(name, value):
    finite = torch.isfinite(value)
    summary = [
        "shape={}".format(tuple(value.shape)),
        "dtype={}".format(value.dtype),
        "device={}".format(value.device),
        "finite={}".format(bool(finite.all().item())),
    ]
    if finite.any():
        finite_values = value[finite]
        summary.extend(
            [
                "min={:.6f}".format(float(finite_values.min().item())),
                "max={:.6f}".format(float(finite_values.max().item())),
            ]
        )
    print("{}: {}".format(name, ", ".join(summary)))


def main():
    args = parse_args()
    if not args.image.is_file():
        raise FileNotFoundError(args.image)

    device = select_device(args.device)
    print("Python: {}".format(sys.version.split()[0]))
    print("PyTorch: {}".format(torch.__version__))
    print("CUDA build: {}".format(torch.version.cuda))
    print("Device: {}".format(device))
    if device.type == "cuda":
        print("GPU: {}".format(torch.cuda.get_device_name(device)))

    load_start = time.perf_counter()
    model = UniDepthV2.from_pretrained(args.model)
    model.resolution_level = args.resolution_level
    model = model.to(device).eval()
    print("Model load: {:.3f} s".format(time.perf_counter() - load_start))

    image = np.asarray(Image.open(args.image).convert("RGB")).copy()
    rgb = torch.from_numpy(image).permute(2, 0, 1)

    inference_start = time.perf_counter()
    with torch.inference_mode():
        predictions = model.infer(rgb)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    print("Inference: {:.3f} s".format(time.perf_counter() - inference_start))

    for name in sorted(predictions):
        value = predictions[name]
        if torch.is_tensor(value):
            tensor_summary(name, value)


if __name__ == "__main__":
    main()
