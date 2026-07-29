from dataclasses import asdict, dataclass
from pathlib import Path

@dataclass
class TrainConfig:
    image_size: int = 448
    batch_size: int = 48
    epochs: int = 20
    learning_rate: float = 2e-4
    # batch_size: int = 128
    # epochs: int = 30
    # learning_rate: float = 3e-4
    # batch_size: int = 128
    # epochs: int = 100
    # learning_rate: float = 3e-4
    weight_decay: float = 0.05
    num_workers: int = 8
    seed: int = 42
    warmup_epochs: int = 2
    model_name: str = "vit_large_patch16_dinov3.lvd1689m"
    # model_name: str = "vit_base_patch16_dinov3.lvd1689m"
    # model_name: str = "vit_small_patch16_dinov3.lvd1689m"
    train_parquet: Path = Path("data/danbooru2025_train.parquet")
    val_parquet: Path = Path("data/danbooru2025_val.parquet")
    tag_to_id: Path = Path("data/tag_to_id.json")
    # train_parquet: Path = Path("data/danbooru2025_lite_train.parquet")
    # val_parquet: Path = Path("data/danbooru2025_lite_val.parquet")
    # tag_to_id: Path = Path("data/tag_to_id_lite.json")
    checkpoint_dir: Path = Path("checkpoints")
    run_dir: Path = Path("runs")
    threshold: float = 0.5
    prefetch_factor: int = 2
    download_workers: int = 32
    download_retries: int = 3
    save_epochs: int = 1
    backbone_lr_mult: float = 0.1
    ema_decay: float = 0.999
    head_type: str = "tag_query_head"
    use_siglip_init: bool = False
    val_interval: int = 1
    # val_interval: int = 1
    compute_multilabel_metrics: bool = False

    def to_dict(self):
        return {key: str(value) if isinstance(value, Path) else value for key, value in asdict(self).items()}
