import torch


def resolve_device(device_name: str) -> torch.device:
    normalized = str(device_name).strip().lower()
    if normalized in ("", "auto"):
        normalized = "cuda" if torch.cuda.is_available() else "cpu"
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device was requested but CUDA is unavailable: "
            f"{device_name}"
        )
    return torch.device(normalized)
