import json
import random
from pathlib import Path
import numpy as np
import torch

def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_checkpoint(path, state) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); torch.save(state, path)

def load_json(path):
    with Path(path).open() as file: return json.load(file)

def write_json(path, value) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file: json.dump(value, file, indent=2)
