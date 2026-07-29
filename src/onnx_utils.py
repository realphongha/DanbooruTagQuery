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


def _fix_cast_node_attrs(model: "onnx.ModelProto") -> bool:
    """Fix Cast node 'to' attributes that mismatch inferred output types.

    ``convert_float_to_float16`` has a bug (microsoft/onnxconverter-common#320)
    where it marks Cast node outputs as float16 but leaves the node's ``to``
    attribute as float32, creating a type mismatch that ORT rejects.

    *model* must already have ``value_info`` populated (run
    ``shape_inference.infer_shapes`` before calling).

    Returns True if any attribute was fixed.
    """
    from onnx import TensorProto

    type_map: dict[str, int] = {}
    for vi in model.graph.value_info:
        try:
            type_map[vi.name] = vi.type.tensor_type.elem_type
        except Exception:
            pass

    fixed = False
    for node in model.graph.node:
        if node.op_type != "Cast":
            continue
        for attr in node.attribute:
            if attr.name != "to":
                continue
            current_to = attr.i
            for output_name in node.output:
                expected = type_map.get(output_name)
                if expected is not None and expected != current_to:
                    attr.i = expected
                    fixed = True
                    break
            break
    return fixed


def _onnx_cast_outputs_to_float32(onnx_path: str) -> str:
    """Fix FP16 ONNX model for CPU EP compatibility.

    Two-step fix:
      1. Fix Cast node ``to`` attributes mismatched by converter bug (#320).
      2. Insert Cast(float16→float32) at any graph outputs still in float16.

    Returns the path to the fixed model (same path if no fix needed, otherwise
    a temp file that should be cleaned up by the caller).
    """
    import onnx
    from onnx import helper, TensorProto, shape_inference

    model = onnx.load(str(onnx_path))

    # Step 1: propagate types so _fix_cast_node_attrs can see them
    try:
        model = shape_inference.infer_shapes(model)
    except Exception:
        pass

    # Step 2: fix Cast node 'to' attributes mismatched by converter bug (#320)
    _fix_cast_node_attrs(model)

    # Step 3: re-propagate types after fixing Cast attrs
    try:
        model = shape_inference.infer_shapes(model)
    except Exception:
        pass

    # Step 4: build type map from value_info (post-fix) + inputs
    type_map: dict[str, int] = {}
    for vi in list(model.graph.value_info) + list(model.graph.input):
        try:
            type_map[vi.name] = vi.type.tensor_type.elem_type
        except Exception:
            pass

    # Check if any output is still float16
    needs_fix = any(
        type_map.get(o.name) == TensorProto.FLOAT16
        for o in model.graph.output
    )
    if not needs_fix:
        return onnx_path

    for output in model.graph.output:
        if type_map.get(output.name) != TensorProto.FLOAT16:
            continue

        cast_node = helper.make_node(
            "Cast",
            inputs=[output.name],
            outputs=[f"{output.name}_as_fp32"],
            name=f"fix_{output.name}_to_fp32",
            to=int(TensorProto.FLOAT),
        )
        model.graph.node.append(cast_node)

        output.name = f"{output.name}_as_fp32"
        output.type.tensor_type.elem_type = TensorProto.FLOAT

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
                fixed = None
                try:
                    fixed = _onnx_cast_outputs_to_float32(self.checkpoint)
                    self._sess = _try_load(fixed, ["CPUExecutionProvider"])
                finally:
                    if fixed is not None and fixed != self.checkpoint:
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
