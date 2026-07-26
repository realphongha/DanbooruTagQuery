from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from timm.data import create_transform, resolve_data_config
from torch.nn import functional as F

from .base import BaseModel


MODEL_REPO_MAP = {
    "vit": "SmilingWolf/wd-vit-tagger-v3",
    "swinv2": "SmilingWolf/wd-swinv2-tagger-v3",
    "convnext": "SmilingWolf/wd-convnext-tagger-v3",
}


def _ensure_rgb(image: Image.Image) -> Image.Image:
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGBA") if "transparency" in image.info else image.convert("RGB")
    if image.mode == "RGBA":
        canvas = Image.new("RGBA", image.size, (255, 255, 255))
        canvas.alpha_composite(image)
        image = canvas.convert("RGB")
    return image


def _pad_square(image: Image.Image) -> Image.Image:
    w, h = image.size
    px = max(w, h)
    canvas = Image.new("RGB", (px, px), (255, 255, 255))
    canvas.paste(image, ((px - w) // 2, (px - h) // 2))
    return canvas


class WDTagger(BaseModel):
    """WD-SwinV2-Tagger-v3 (SmilingWolf) via timm."""

    def __init__(self, model_name: str = "swinv2"):
        if model_name not in MODEL_REPO_MAP:
            raise ValueError(f"Unknown WD model {model_name!r}; options: {list(MODEL_REPO_MAP)}")
        self._model_name = model_name
        self._repo_id = MODEL_REPO_MAP[model_name]
        self._tags: list[str] = []
        self._tag_indices: list[int] = []  # actual output indices for each tag
        self._model: torch.nn.Module | None = None
        self._transform = None

    @property
    def name(self) -> str:
        return f"WD-{self._model_name}"

    @property
    def tag_names(self) -> list[str]:
        return self._tags

    def load(self) -> None:
        repo_id = self._repo_id
        print(f"  Loading WD tagger '{self._model_name}' from {repo_id} ...")

        model = timm.create_model("hf-hub:" + repo_id, pretrained=True).eval()
        # Keep on CPU; predict() moves to device
        self._model = model
        self._transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))

        csv_path = hf_hub_download(repo_id=repo_id, filename="selected_tags.csv")
        df = pd.read_csv(csv_path, usecols=["name", "category"])
        # Keep only general (0) and character (4); record their row indices
        mask = (df["category"] == 0) | (df["category"] == 4)
        selected = df[mask]
        self._tags = selected["name"].tolist()
        self._tag_indices = selected.index.tolist()  # original CSV row positions
        print(f"    {len(self._tags)} tags ({len(selected[selected['category']==0])} general + {len(selected[selected['category']==4])} character)")

    def predict(self, image: Image.Image) -> np.ndarray:
        img = _ensure_rgb(image)
        img = _pad_square(img)
        inputs = self._transform(img).unsqueeze(0)  # (1,3,H,W)
        inputs = inputs[:, [2, 1, 0]]  # RGB → BGR

        dev = self.device
        self._model.to(dev)
        with torch.inference_mode():
            outputs = self._model(inputs.to(dev))
            all_scores = F.sigmoid(outputs).cpu().numpy()[0]  # (N_all_tags,)

        # Extract only general + character scores at correct output positions
        return all_scores[self._tag_indices].astype(np.float32)
