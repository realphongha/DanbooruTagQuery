"""ONNX predictor and preprocessing for DanbooruTagQuery.

NumPy + Pillow only — no PyTorch/torchvision.
"""

from __future__ import annotations

import ctypes
import json
from pathlib import Path

import numpy as np
from PIL import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

CATEGORY_MAP = {
    0: "general",
    1: "artist",
    3: "copyright",
    4: "character",
    5: "meta",
}


def get_category_name(cat_map: dict[str, int], tag: str) -> str:
    cat_id = cat_map.get(tag, 0)
    return CATEGORY_MAP.get(cat_id, "general")


# ── cuDNN preload ───────────────────────────────────────────────────────────

def _preload_cudnn() -> bool:
    """Load libcudnn.so before creating an ONNX session.

    ONNX Runtime's CUDA EP dlopens libcudnn at run time. In environments where
    cuDNN comes from pip's `nvidia-cudnn` wheel (no system install), the loader
    path is never set unless something imports torch first. Preload it so the
    predictor can stay torch-free.
    """
    candidates: list[str] = []

    # 1) pip nvidia-cudnn wheel
    try:
        from nvidia.cudnn import lib as cudnn_lib
        for d in cudnn_lib.__path__:
            candidates.append(str(Path(d) / "libcudnn.so.9"))
    except Exception:
        pass

    # 2) ctypes default search (system install / LD_LIBRARY_PATH)
    for name in ("libcudnn.so.9", "libcudnn.so.8", "libcudnn.so"):
        candidates.append(name)

    for cand in candidates:
        try:
            ctypes.CDLL(cand)
            return True
        except Exception:
            continue
    return False


# ── sidecar loading ─────────────────────────────────────────────────────────

def resolve_model_dir(checkpoint: str | Path) -> Path:
    """Resolve model directory from path.

    Accepts:
        - path/to/model_dir          (directory)
        - path/to/model_dir/model.onnx  (model.onnx file → parent dir)
    """
    ckpt = Path(checkpoint)
    if ckpt.name == "model.onnx":
        return ckpt.parent
    return ckpt


def load_tag_to_id(checkpoint: str | Path) -> dict[str, int]:
    model_dir = resolve_model_dir(checkpoint)
    sidecar = model_dir / "tag_to_id.json"
    if not sidecar.exists():
        raise FileNotFoundError(f"Missing tag map: {sidecar}")
    return json.loads(sidecar.read_text())


def load_config(checkpoint: str | Path) -> dict:
    model_dir = resolve_model_dir(checkpoint)
    sidecar = model_dir / "config.json"
    if not sidecar.exists():
        return {"image_size": 448}
    return json.loads(sidecar.read_text())


def load_category_map(checkpoint: str | Path) -> dict[str, int]:
    model_dir = resolve_model_dir(checkpoint)
    sidecar = model_dir / "tag_category.json"
    if not sidecar.exists():
        raise FileNotFoundError(f"Missing tag_category.json: {sidecar}")
    return json.loads(sidecar.read_text())


# ── image preprocessing ─────────────────────────────────────────────────────

def preprocess(image: Image.Image, image_size: int = 448) -> np.ndarray:
    """Resize + pad + normalize to (1, 3, H, W) float32 numpy array."""
    w, h = image.size
    scale = image_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    image = image.resize((new_w, new_h), Image.BILINEAR)

    canvas = Image.new("RGB", (image_size, image_size), (0, 0, 0))
    left = (image_size - new_w) // 2
    top = (image_size - new_h) // 2
    canvas.paste(image, (left, top))

    arr = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1) / 255.0
    arr[0] = (arr[0] - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
    arr[1] = (arr[1] - IMAGENET_MEAN[1]) / IMAGENET_STD[1]
    arr[2] = (arr[2] - IMAGENET_MEAN[2]) / IMAGENET_STD[2]
    return arr[np.newaxis, ...]


# ── ONNX predictor ──────────────────────────────────────────────────────────

class Predictor:
    """ONNX-only predictor. Prefers CUDA, falls back to CPU (or force CPU)."""

    def __init__(self, checkpoint: str | Path, use_cuda: bool = True):
        import onnxruntime as ort

        if use_cuda:
            _preload_cudnn()

        self.checkpoint = str(checkpoint)
        self.tag_to_id = load_tag_to_id(self.checkpoint)
        self.cat_map = load_category_map(self.checkpoint)
        cfg = load_config(self.checkpoint)
        self.image_size = cfg.get("image_size", 448)

        onnx_file = resolve_model_dir(checkpoint) / "model.onnx"

        if use_cuda:
            providers = [
                ("CUDAExecutionProvider", {}),
                "CPUExecutionProvider",
            ]
            try:
                sess = ort.InferenceSession(str(onnx_file), providers=providers)
                # warmup — CUDA EP can fail at run time (e.g. missing libcudnn.so)
                inp = sess.get_inputs()[0]
                shape = [1 if d is None or isinstance(d, str) else int(d) for d in inp.shape]
                dummy = np.zeros(shape, dtype=np.float32)
                sess.run([sess.get_outputs()[0].name], {inp.name: dummy})
                self._sess = sess
            except Exception as exc:
                print(f"  [ONNX] CUDA EP failed ({exc}); using CPU")
                self._sess = ort.InferenceSession(
                    str(onnx_file), providers=["CPUExecutionProvider"]
                )
        else:
            self._sess = ort.InferenceSession(
                str(onnx_file), providers=["CPUExecutionProvider"]
            )

        self._input_name = self._sess.get_inputs()[0].name
        self._output_name = self._sess.get_outputs()[0].name

    def run(self, pixel_values: np.ndarray) -> np.ndarray:
        """Raw logits to sigmoid scores, shape (N, num_classes)."""
        raw = self._sess.run([self._output_name], {self._input_name: pixel_values})[0]
        return 1.0 / (1.0 + np.exp(-raw))  # sigmoid

    @property
    def num_classes(self) -> int:
        return len(self.tag_to_id)

    def category_name(self, tag: str) -> str:
        return get_category_name(self.cat_map, tag)


def predict(
    image_path: str | Path,
    checkpoint: str | Path,
    top_k: int = 20,
    min_score: float = 0.0,
    use_cuda: bool = True,
) -> list[tuple[str, float]]:
    """Run inference on a single image. Returns [(tag, score), ...]."""
    predictor = Predictor(checkpoint, use_cuda=use_cuda)
    image = Image.open(image_path).convert("RGB")
    tensor = preprocess(image, predictor.image_size)
    scores = predictor.run(tensor)[0]
    inv = {v: k for k, v in predictor.tag_to_id.items()}
    indices = np.argsort(scores)[::-1]
    results = [(inv[int(i)], float(scores[i])) for i in indices]
    if min_score > 0.0:
        results = [(t, s) for t, s in results if s >= min_score]
    if top_k is not None:
        results = results[:top_k]
    return results