from .base import BaseModel


def list_available_models() -> dict[str, type["BaseModel"]]:
    """Lazy-import all model classes so missing optional deps don't crash the package."""
    models: dict[str, type[BaseModel]] = {}
    try:
        from .deepdanbooru import DeepDanbooru as _DD
        models["DeepDanbooru"] = _DD
    except ImportError as e:
        pass
    try:
        from .wd_tagger import WDTagger as _WD
        models["WD-SwinV2"] = _WD
    except ImportError as e:
        pass
    try:
        from .ml_danbooru import MLDanbooru as _ML
        models["ML-Danbooru"] = _ML
    except ImportError as e:
        pass
    try:
        from .joytag import JoyTag as _JT
        models["JoyTag"] = _JT
    except ImportError as e:
        pass
    return models
