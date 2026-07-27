from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image

from ._download import download_hf_file
from .base import BaseModel


_HF_REPO = "deepghs/ml-danbooru-onnx"
_TAGS_REPO = "deepghs/imgutils-models"
_ONNX_FILENAME = "ml_caformer_m36_dec-5-97527.onnx"
_TAGS_FILENAME = "mldanbooru/mldanbooru_tags.csv"


class MLDanbooru(BaseModel):
    """ML-Danbooru (7eu7d7) via ONNX Runtime.

    Downloads ONNX model + tag CSV from HuggingFace on first run.
    No PyTorch ``src_files/`` dependency — pure ONNX inference.
    """

    def __init__(
        self,
        repo_dir: Path = Path("models/ml-danbooru"),
        model_name: str = "caformer_m36",  # kept for API compat, always caformer
        fp16: bool = False,  # ignored for ONNX
    ):
        self._repo_dir = Path(repo_dir).resolve()
        self._tags: list[str] = []
        self._session: Any = None  # onnxruntime.InferenceSession

    @property
    def name(self) -> str:
        return "ML-Danbooru"

    @property
    def tag_names(self) -> list[str]:
        return self._tags

    @property
    def input_size(self) -> tuple[int, int]:
        return (448, 448)

    def load(self) -> None:
        import onnxruntime as ort

        repo = self._repo_dir
        repo.mkdir(parents=True, exist_ok=True)

        # Download ONNX model
        onnx_path = repo / _ONNX_FILENAME
        if not onnx_path.is_file():
            print(f"  Downloading ONNX model from HF ...")
            download_hf_file(_HF_REPO, _ONNX_FILENAME, repo)

        # Download tags CSV
        csv_path = repo / _TAGS_FILENAME
        if not csv_path.is_file():
            print(f"  Downloading tags CSV from HF ...")
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                hf_hub_download(_TAGS_REPO, _TAGS_FILENAME, local_dir=str(repo),
                                local_dir_use_symlinks=False)
            except Exception:
                # Fallback: tags.csv from the ONNX repo
                download_hf_file(_HF_REPO, "tags.csv", repo)
                csv_path = repo / "tags.csv"

        # Read tags
        import pandas as pd
        df = pd.read_csv(csv_path)
        if "tag" in df.columns:
            self._tags = df["tag"].tolist()
        elif "name" in df.columns:
            self._tags = df["name"].tolist()
        else:
            self._tags = df.iloc[:, 0].tolist()
        print(f"    {len(self._tags)} tags")

        # Create ONNX session
        print(f"  Loading ONNX model ...")
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(
            str(onnx_path), sess_options,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
        )
        print(f"    ONNX session created")

    def predict(self, image: Image.Image) -> np.ndarray:
        img = image.convert("RGB")

        # Resize to square 448×448 (align to 4)
        size = 448
        img = img.resize((size, size), Image.BILINEAR)

        # HWC → CHW, float32 / 255
        arr = np.array(img, dtype=np.uint8)
        arr = arr.transpose(2, 0, 1).astype(np.float32) / 255.0

        # Add batch dim
        inp = arr.reshape(1, *arr.shape)

        # Run ONNX
        out = self._session.run(["output"], {"input": inp})[0]
        # Sigmoid
        scores = (1.0 / (1.0 + np.exp(-out))).reshape(-1)

        return scores.astype(np.float32)
