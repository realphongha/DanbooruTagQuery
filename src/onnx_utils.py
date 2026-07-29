"""
Shared utilities for ONNX / PyTorch inference across infer.py, evaluate.py, ui.py.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from torchvision.transforms import Compose

from .config import TrainConfig
from .model import ImageTagger
from .transforms import val_transforms


# ── helpers ──────────────────────────────────────────────────────────────────

def is_onnx(path: str | Path) -> bool:
    return Path(path).suffix == ".onnx"


def _sidecar_stem(ckpt: Path) -> str:
    """Strip variant suffixes (.fp16, .mixed) from ONNX filename stem."""
    stem = ckpt.with_suffix("").stem  # removes .onnx
    for variant in (".fp16", ".mixed"):
        if stem.endswith(variant):
            stem = stem[: -len(variant)]
            break
    return stem


def load_tag_to_id(checkpoint: str) -> dict[str, int]:
    ckpt = Path(checkpoint)
    if ckpt.suffix == ".onnx":
        sidecar = ckpt.with_name(_sidecar_stem(ckpt) + ".tag_to_id.json")
        if sidecar.exists():
            return json.loads(sidecar.read_text())
        raise FileNotFoundError(f"Missing tag map: {sidecar}")
    state = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    return state["tag_to_id"]


def load_config(checkpoint: str) -> TrainConfig:
    ckpt = Path(checkpoint)
    if ckpt.suffix == ".onnx":
        sidecar = ckpt.with_name(_sidecar_stem(ckpt) + ".config.json")
        if sidecar.exists():
            data = json.loads(sidecar.read_text())
            cfg = TrainConfig()
            cfg.image_size = data.get("image_size", cfg.image_size)
            cfg.model_name = data.get("model_name", cfg.model_name)
            cfg.head_type = data.get("head_type", cfg.head_type)
            cfg.num_classes = data.get("num_classes", 0)
            return cfg
        raise FileNotFoundError(f"Missing config: {sidecar}")
    cfg = TrainConfig()
    state = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    cfg.num_classes = len(state["tag_to_id"])
    return cfg


# ── ONNX model fixing ──────────────────────────────────────────────────────


def _onnx_cast_outputs_to_float32(onnx_path: str) -> str:
    """Cast any float16 graph outputs to float32 for CPU EP compatibility.

    Returns the path to the fixed model (same path if no fix needed, otherwise
    a temp file that should be cleaned up by the caller).
    """
    import onnx
    from onnx import helper, TensorProto

    model = onnx.load(str(onnx_path))

    needs_fix = any(
        o.type.tensor_type.elem_type == TensorProto.FLOAT16
        for o in model.graph.output
    )
    if not needs_fix:
        return onnx_path

    for output in model.graph.output:
        if output.type.tensor_type.elem_type != TensorProto.FLOAT16:
            continue

        orig_name = output.name
        cast_name = f"{orig_name}_as_fp32"

        cast_node = helper.make_node(
            "Cast",
            inputs=[orig_name],
            outputs=[cast_name],
            name=f"fix_{orig_name}_to_fp32",
            to=int(TensorProto.FLOAT),
        )
        model.graph.node.append(cast_node)

        # Redirect graph output to the Cast node's output
        output.name = cast_name
        output.type.tensor_type.elem_type = TensorProto.FLOAT

    model = onnx.shape_inference.infer_shapes(model)

    tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
    onnx.save(model, tmp.name)
    return tmp.name


# ── Predictor (unified interface for PT + ONNX) ────────────────────────────

class Predictor:
    """Loads a checkpoint (.pt or .onnx) and runs batched inference.

    Usage
    -----
    >>> p = Predictor("model.onnx")
    >>> scores = p.run(torch.randn(1, 3, 256, 256))   # -> np.ndarray (N, num_classes)
    """

    def __init__(self, checkpoint: str, device: str | None = None):
        self.checkpoint = str(checkpoint)
        self.tag_to_id = load_tag_to_id(self.checkpoint)
        self.config = load_config(self.checkpoint)
        self.val_transform: Compose = val_transforms(self.config.image_size)
        self._is_onnx = is_onnx(self.checkpoint)

        if self._is_onnx:
            import onnxruntime as ort

            # prefer CUDA EP, fall back to CPU
            providers = [
                ("CUDAExecutionProvider", {}),
                "CPUExecutionProvider",
            ]
            def _try_load(path, prov):
                return ort.InferenceSession(path, providers=prov)

            try:
                self._sess = _try_load(self.checkpoint, providers)
            except Exception:
                # fallback: CPU only; fix FP16 outputs if needed
                fixed = _onnx_cast_outputs_to_float32(self.checkpoint)
                try:
                    self._sess = _try_load(fixed, ["CPUExecutionProvider"])
                finally:
                    if fixed != self.checkpoint:
                        Path(fixed).unlink(missing_ok=True)

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
                head_type=self.config.head_type,
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
            raw = self._sess.run(
                [self._output_name],
                {self._input_name: np_input},
            )[0]
            return 1.0 / (1.0 + np.exp(-raw))  # sigmoid

        if isinstance(pixel_values, np.ndarray):
            pixel_values = torch.from_numpy(pixel_values)
        out = self._model(pixel_values.to(self._device))
        return torch.sigmoid(out).cpu().numpy()
