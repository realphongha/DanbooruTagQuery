"""DanbooruTagQuery ONNX deployment package.

Predictor, preprocessing, category map, Gradio UI, and CLI/hosted entry point.
No PyTorch needed. ``main`` is also the programmatic entry point used by the
Hugging Face Space (e.g. ``main(use_cuda=False)`` for CPU-only).
"""

from .core import (
    CATEGORY_MAP,
    Predictor,
    get_category_name,
    load_category_map,
    load_config,
    load_tag_to_id,
    preprocess,
    resolve_model_dir,
)
from .cli import main
from .ui import build_app, enrich_tags, format_tag, launch

# Optional hub integration (offline-safe)
from .hub import (
    download_model_variant,
    discover_model_variants,
    hf_available,
)

__all__ = [
    "CATEGORY_MAP",
    "Predictor",
    "build_app",
    "download_model_variant",
    "discover_model_variants",
    "enrich_tags",
    "format_tag",
    "get_category_name",
    "hf_available",
    "launch",
    "load_category_map",
    "load_config",
    "load_tag_to_id",
    "main",
    "preprocess",
    "resolve_model_dir",
]