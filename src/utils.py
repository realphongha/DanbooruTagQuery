import json
import os
import random
from pathlib import Path
import numpy as np
import torch

def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def device() -> torch.device:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available() and local_rank < torch.cuda.device_count():
        return torch.device("cuda", local_rank)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def is_distributed() -> bool:
    return torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1


def get_rank() -> int:
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def get_world_size() -> int:
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1


def rank0_only() -> bool:
    return get_rank() == 0

def save_checkpoint(path, state) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); torch.save(state, path)

def print_sdp_backend_status():
    if not torch.cuda.is_available():
        print("  SDP backend: CPU — no CUDA available")
        return

    flash = torch.backends.cuda.flash_sdp_enabled()
    mem_eff = torch.backends.cuda.mem_efficient_sdp_enabled()
    math = torch.backends.cuda.math_sdp_enabled()
    print("  SDP backends enabled:")
    print(f"    {'✓' if flash else '✗'} FlashAttention")
    print(f"    {'✓' if mem_eff else '✗'} MemoryEfficientAttention")
    print(f"    {'✓' if math else '✗'} Math (fallback)")

    if flash:
        print("  → FlashAttention will be used when shapes are compatible")
    elif mem_eff:
        print("  → MemoryEfficientAttention (cutlass) will be used when shapes are compatible")
    elif math:
        print("  → Math fallback")
    else:
        print("  → No SDP backend available")


def load_json(path):
    with Path(path).open() as file: return json.load(file)

def write_json(path, value) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file: json.dump(value, file, indent=2)
