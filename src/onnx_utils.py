"""
Shared utilities for ONNX / PyTorch inference across infer.py, evaluate.py, ui.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torchvision.transforms import Compose

from .config import TrainConfig
from .model import ImageTagger
from .transforms import val_transforms


# ── helpers ──────────────────────────────────────────────────────────────────

def _resolve_model_dir(ckpt: Path) -> Path:
    """Resolve model directory from path.

    Accepts:
        - path/to/model_dir          (directory)
        - path/to/model_dir/model.onnx  (model.onnx file → parent dir)
    """
    if ckpt.name == "model.onnx":
        return ckpt.parent
    return ckpt


def is_onnx(path: str | Path) -> bool:
    p = Path(path)
    if p.name == "model.onnx" or p.suffix == ".onnx":
        return True
    if p.is_dir():
        return (p / "model.onnx").exists()
    return False


def load_tag_to_id(checkpoint: str) -> dict[str, int]:
    model_dir = _resolve_model_dir(Path(checkpoint))
    sidecar = model_dir / "tag_to_id.json"
    if sidecar.exists():
        return json.loads(sidecar.read_text())
    raise FileNotFoundError(f"Missing tag map: {sidecar}")


def load_config(checkpoint: str) -> TrainConfig:
    model_dir = _resolve_model_dir(Path(checkpoint))
    sidecar = model_dir / "config.json"
    if sidecar.exists():
        data = json.loads(sidecar.read_text())
        cfg = TrainConfig()
        cfg.image_size = data.get("image_size", cfg.image_size)
        cfg.model_name = data.get("model_name", cfg.model_name)
        cfg.num_classes = data.get("num_classes", 0)
        return cfg
    raise FileNotFoundError(f"Missing config: {sidecar}")


# ── ONNX session helpers ────────────────────────────────────────────────────


def _create_onnx_session(onnx_file: Path):
    """Create ONNX session. Prefer CUDA EP, verify with warmup, fall back to CPU.

    CUDA EP selection can succeed at session creation but fail at run time
    (e.g. missing libcudnn.so) — the warmup run catches that.
    """
    import onnxruntime as ort

    providers = [
        ("CUDAExecutionProvider", {}),
        "CPUExecutionProvider",
    ]
    try:
        sess = ort.InferenceSession(str(onnx_file), providers=providers)
        inp = sess.get_inputs()[0]
        shape = [1 if d is None or isinstance(d, str) else int(d) for d in inp.shape]
        dummy = np.zeros(shape, dtype=np.float32)
        sess.run([sess.get_outputs()[0].name], {inp.name: dummy})
        return sess
    except Exception as exc:
        print(f"  [ONNX] CUDA EP failed ({exc}); using CPU")
        return ort.InferenceSession(
            str(onnx_file), providers=["CPUExecutionProvider"]
        )


# ── Predictor (unified interface for PT + ONNX) ────────────────────────────

class Predictor:
    """Loads a checkpoint (.pt or model directory) and runs batched inference.

    Directory format:
        model_dir/
            model.onnx
            tag_to_id.json
            config.json

    Usage
    -----
    >>> p = Predictor("my_model")               # directory
    >>> p = Predictor("my_model/model.onnx")     # model.onnx → resolves to dir
    >>> scores = p.run(torch.randn(1, 3, 256, 256))   # -> np.ndarray (N, num_classes)
    """

    def __init__(self, checkpoint: str, device: str | None = None, chunk_size: int = 4):
        ckpt = Path(checkpoint)
        self.checkpoint = str(ckpt)
        self.tag_to_id = load_tag_to_id(self.checkpoint)
        self.config = load_config(self.checkpoint)
        self.val_transform: Compose = val_transforms(self.config.image_size)
        self._is_onnx = is_onnx(self.checkpoint)
        self._chunk_size = chunk_size

        if self._is_onnx:
            onnx_file = _resolve_model_dir(ckpt) / "model.onnx"
            self._sess = _create_onnx_session(onnx_file)
            self._input_name = self._sess.get_inputs()[0].name
            self._output_name = self._sess.get_outputs()[0].name
            self._model = None
        else:
            self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            state = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
            model = ImageTagger(
                self.config.model_name,
                self.config.num_classes,
                pretrained=False,
            )
            weights = state.get("model_ema") or state.get("model")
            model.load_state_dict(weights)
            model.to(self._device).eval()
            self._model = model
            self._sess = None

    @torch.inference_mode()
    def run(self, pixel_values: torch.Tensor | np.ndarray) -> np.ndarray:
        """Returns sigmoid scores as float32 array of shape (N, num_classes)."""
        if self._sess is not None:
            if isinstance(pixel_values, torch.Tensor):
                np_input = pixel_values.cpu().numpy()
            else:
                np_input = pixel_values
            if len(np_input) > self._chunk_size:
                chunks = [
                    self._sess.run(
                        [self._output_name],
                        {self._input_name: np_input[i : i + self._chunk_size]},
                    )[0]
                    for i in range(0, len(np_input), self._chunk_size)
                ]
                raw = np.concatenate(chunks, axis=0)
            else:
                raw = self._sess.run(
                    [self._output_name],
                    {self._input_name: np_input},
                )[0]
            return 1.0 / (1.0 + np.exp(-raw))  # sigmoid

        if isinstance(pixel_values, np.ndarray):
            pixel_values = torch.from_numpy(pixel_values)
        out = self._model(pixel_values.to(self._device))
        return torch.sigmoid(out).cpu().numpy()
