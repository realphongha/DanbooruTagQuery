from dataclasses import asdict, dataclass
from pathlib import Path

@dataclass
class TrainConfig:
    image_size: int = 256
    batch_size: int = 128
    epochs: int = 30
    learning_rate: float = 3e-4
    weight_decay: float = 0.05
    num_workers: int = 8
    seed: int = 42
    warmup_epochs: int = 5
    model_name: str = "vit_small_patch16_dinov3.lvd1689m"
    train_parquet: Path = Path("data/danbooru2025_lite_train.parquet")
    val_parquet: Path = Path("data/danbooru2025_lite_val.parquet")
    tag_to_id: Path = Path("data/tag_to_id.json")
    checkpoint_dir: Path = Path("checkpoints")
    run_dir: Path = Path("runs")
    threshold: float = 0.5
    prefetch_factor: int = 4
    download_workers: int = 32
    download_retries: int = 3
    save_epochs: int = 1
    backbone_lr_mult: float = 0.1
    ema_decay: float = 0.999
    head_type: str = "tag_query_head"

    def to_dict(self):
        return {key: str(value) if isinstance(value, Path) else value for key, value in asdict(self).items()}
