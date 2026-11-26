"""Pick the compute device: NVIDIA GPU (cuda), Apple GPU (mps), or CPU."""
import os
import torch


def pick_device():
    want = os.environ.get("TM_DEVICE")              # optional override: cuda | mps | cpu
    if want:
        return want
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sync(dev):
    if dev == "cuda":
        torch.cuda.synchronize()
    elif dev == "mps":
        torch.mps.synchronize()
