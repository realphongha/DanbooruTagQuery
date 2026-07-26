from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as TVF

from ._download import download_hf_file, download_url
from .base import BaseModel


_MEAN = [0.48145466, 0.4578275, 0.40821073]
_STD = [0.26862954, 0.26130258, 0.27577711]

_HF_REPO = "fancyfeast/joytag"
_GITHUB_RAW = "https://raw.githubusercontent.com/fpgaminer/joytag/master"
_REQUIRED_HF_FILES = ["model.safetensors", "config.json", "top_tags.txt"]


class JoyTag(BaseModel):
    """JoyTag (fpgaminer) using ViT-B/16 backbone."""

    def __init__(self, model_dir: Path = Path("models/joytag")):
        self._model_dir = Path(model_dir).resolve()
        self._tags: list[str] = []
        self._model: torch.nn.Module | None = None
        self._image_size: int = 448

    @property
    def name(self) -> str:
        return "JoyTag"

    @property
    def tag_names(self) -> list[str]:
        return self._tags

    def load(self) -> None:
        model_dir = self._model_dir
        model_dir.mkdir(parents=True, exist_ok=True)

        models_py = model_dir / "Models.py"
        if not models_py.is_file():
            print(f"  Downloading Models.py from GitHub ...")
            download_url(f"{_GITHUB_RAW}/Models.py", models_py, desc="Models.py")

        for fname in _REQUIRED_HF_FILES:
            dest = model_dir / fname
            if not dest.is_file():
                print(f"  Downloading {fname} from HF ...")
                download_hf_file(_HF_REPO, fname, model_dir)

        sys.path.insert(0, str(model_dir))

        print(f"  Loading JoyTag from {model_dir} ...")
        try:
            from Models import VisionModel
        except ImportError as e:
            raise ImportError(f"Cannot import Models.py from {model_dir}. Error: {e}")

        model = VisionModel.load_model(str(model_dir))
        model.eval()
        # Keep on CPU; predict() moves to device
        self._model = model
        self._image_size = model.image_size

        tags_path = model_dir / "top_tags.txt"
        with open(tags_path) as f:
            self._tags = [line.strip() for line in f if line.strip()]
        print(f"    {len(self._tags)} tags")

    def predict(self, image: Image.Image) -> np.ndarray:
        img = image.convert("RGB")
        w, h = img.size
        max_dim = max(w, h)
        pad_l = (max_dim - w) // 2
        pad_t = (max_dim - h) // 2
        padded = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
        padded.paste(img, (pad_l, pad_t))

        if max_dim != self._image_size:
            padded = padded.resize((self._image_size, self._image_size), Image.BICUBIC)

        tensor = TVF.pil_to_tensor(padded).float() / 255.0
        tensor = TVF.normalize(tensor, mean=_MEAN, std=_STD)

        dev = self.device
        self._model.to(dev)
        batch = {"image": tensor.unsqueeze(0).to(dev)}

        with torch.no_grad():
            with torch.amp.autocast(dev.type, enabled=True):
                preds = self._model(batch)
            scores = preds["tags"].sigmoid().cpu().numpy().flatten()

        return scores.astype(np.float32)
