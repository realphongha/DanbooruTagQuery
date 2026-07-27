from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .deepdanbooru_model import DeepDanbooruModel
from ._download import download_url
from .base import BaseModel


_DD_CHECKPOINT_URL = (
    "https://github.com/AUTOMATIC1111/TorchDeepDanbooru/releases/download/v1/"
    "model-resnet_custom_v3.pt"
)


class DeepDanbooru(BaseModel):
    """TorchDeepDanbooru (AUTOMATIC1111 port)."""

    def __init__(self, checkpoint: Path = Path("models/deepdanbooru/model-resnet_custom_v3.pt")):
        self._checkpoint = checkpoint
        self._tags: list[str] = []
        self._model: DeepDanbooruModel | None = None

    @property
    def name(self) -> str:
        return "DeepDanbooru"

    @property
    def tag_names(self) -> list[str]:
        return self._tags

    @property
    def input_size(self) -> tuple[int, int]:
        return (512, 512)

    def load(self) -> None:
        ckpt = self._checkpoint
        if not ckpt.exists():
            print(f"  Downloading DeepDanbooru checkpoint to {ckpt} ...")
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            download_url(_DD_CHECKPOINT_URL, ckpt, desc="DeepDanbooru")

        print(f"  Loading DeepDanbooru from {ckpt} ...")
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        self._tags = state.get("tags", [])

        model = DeepDanbooruModel()
        model.load_state_dict(state)
        model.eval()
        # Keep on CPU; predict() moves to device as needed
        self._model = model
        print(f"    {len(self._tags)} tags")

    def predict(self, image: Image.Image) -> np.ndarray:
        img = image.convert("RGB").resize((512, 512))
        arr = np.array(img, dtype=np.float32) / 255.0  # (512,512,3) NHWC
        dev = self.device
        # Ensure model is on the right device
        if next(self._model.parameters()).device != dev:
            self._model.to(dev)
        batch = torch.from_numpy(arr[None, ...]).to(dev)  # (1,512,512,3)
        with torch.no_grad():
            scores = self._model(batch).cpu().numpy()  # (1,9176)
        return scores[0].astype(np.float32)
