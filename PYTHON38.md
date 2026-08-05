# Python 3.8 compatibility

This branch keeps the UniDepthV2 inference path usable in ROS Noetic's
Python 3.8 environment. Upstream UniDepth declares Python 3.10 or newer.

The compatibility changes are intentionally small:

- postpone evaluation of Python 3.9/3.10-style type annotations;
- allow NumPy 1.24 on Python 3.8;
- avoid importing `wandb` unless training artifact logging is used; and
- provide a minimal inference-only dependency list.

Install a CUDA-compatible PyTorch build first, then install the remaining
inference dependencies and this repository without the full training stack:

```bash
python3 -m pip install -r requirements-py38-inference.txt
python3 -m pip install --no-deps -e .
```

Ubuntu 20.04 ships an older setuptools release that cannot read modern
`pyproject.toml` project metadata. The inference requirements therefore
include a Python 3.8-compatible setuptools version.

Run a smoke test with an RGB image:

```bash
python3 scripts/check_inference_py38.py --image /path/to/image.jpg
```

The model checkpoint is downloaded from Hugging Face on the first run. Set
`HF_HOME` if the cache should be stored at a specific location.

## Scope

UniDepthV2 ViT-S inference is tested. The complete training and dataset tool
chain has many optional dependencies and is not part of the Python 3.8 smoke
test. UniDepthV1 still requires xFormers for its Nystrom attention path.
