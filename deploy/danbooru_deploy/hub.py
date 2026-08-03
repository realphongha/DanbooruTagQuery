"""Optional Hugging Face Hub integration.

Lazy-imports huggingface_hub — the package works fully offline without it.
Functions degrade gracefully (empty variant list) when the extra is missing.
"""

from __future__ import annotations

from pathlib import Path

HF_REPO = "realphongha/danbooru-tag-query"
MODELS_DIR = "models"
CATEGORY_JSON = "tag_category.json"

_hf_hub = None


def _import_hf_hub():
    global _hf_hub
    if _hf_hub is None:
        import huggingface_hub as h
        _hf_hub = h
    return _hf_hub


def hf_available() -> bool:
    try:
        _import_hf_hub()
        return True
    except Exception:
        return False


def discover_model_variants(
    hf_repo: str = HF_REPO, models_dir: str = MODELS_DIR
) -> list[str]:
    """List model variant names on the hub (e.g. ``DanbooruTagQuery_l16_448x448``).

    Returns [] when the hub extra is unavailable or the repo cannot be listed.
    """
    try:
        hf = _import_hf_hub()
        api = hf.HfApi()
        siblings = api.list_repo_files(hf_repo, repo_type="model")
        variants: set[str] = set()
        for path in siblings:
            if path.startswith(f"{models_dir}/") and "/" in path[len(models_dir) + 1:]:
                variant = path.split("/")[1]
                if variant:
                    variants.add(variant)
        return sorted(variants, reverse=True)
    except Exception as exc:
        print(f"Warning: could not discover models on hub: {exc}")
        return []


def download_model_variant(
    variant: str,
    hf_repo: str = HF_REPO,
    models_dir: str = MODELS_DIR,
) -> Path:
    """Download ``model.onnx`` + sidecar JSONs into the HF cache.

    Returns the path to ``model.onnx``. Sidecars are optional — missing ones
    are ignored (Predictor requires ``tag_category.json`` for hub models, so
    it is fetched here as well).
    """
    hf = _import_hf_hub()
    onnx_path = hf.hf_hub_download(
        repo_id=hf_repo,
        filename=f"{models_dir}/{variant}/model.onnx",
        repo_type="model",
    )
    for sidecar in ["config.json", "tag_to_id.json", CATEGORY_JSON]:
        try:
            hf.hf_hub_download(
                repo_id=hf_repo,
                filename=f"{models_dir}/{variant}/{sidecar}",
                repo_type="model",
            )
        except Exception:
            pass  # optional — missing is fine
    return Path(onnx_path)


def resolve_checkpoint(
    checkpoint: str | None = None,
    env_dir: str | None = None,
    hf_repo: str = HF_REPO,
    models_dir: str = MODELS_DIR,
    use_cuda: bool = True,
) -> tuple[Path, str | None, list[str]]:
    """Resolve a model checkpoint from CLI arg, env dir, or the hub.

    Returns ``(checkpoint, variant, variants)`` where ``variant`` is the
    downloaded variant name (None for local checkpoints) and ``variants`` is
    the full discovered list (for the model switcher dropdown).
    """
    variants: list[str] = []
    ckpt_path: Path | None = None
    variant: str | None = None

    if checkpoint:
        ckpt_path = Path(checkpoint)
    elif env_dir:
        env_path = Path(env_dir)
        if not env_path.is_dir():
            raise FileNotFoundError(f"MODEL_DIR {env_path} is not a directory")
        onnx_files = list(env_path.glob("*.onnx"))
        if not onnx_files:
            raise FileNotFoundError(f"no .onnx files in {env_path}")
        ckpt_path = onnx_files[0]
    else:
        # hub fallback
        if not hf_available():
            raise FileNotFoundError(
                "No checkpoint given and huggingface_hub not installed. "
                "Install the [hf] extra or pass a checkpoint."
            )
        variants = discover_model_variants(hf_repo, models_dir)
        if not variants:
            raise FileNotFoundError("No models found on hub.")
        variant = variants[0]
        ckpt_path = download_model_variant(variant, hf_repo, models_dir)
        print(f"Downloaded default model: {variant}")

    if not ckpt_path.exists():
        raise FileNotFoundError(f"{ckpt_path} not found")

    return ckpt_path, variant, variants