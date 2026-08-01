"""DINOv3 L/16 from-scratch on the full ~1M dataset.

Mirrors the TrainConfig() defaults exactly — this is the reference config for
the current pipeline (kept so experiments stay reproducible).
"""

from src.config import TrainConfig

config = TrainConfig(
    image_size=448,
    batch_size=48,
    epochs=20,
    learning_rate=2e-4,
    weight_decay=0.05,
    num_workers=8,
    seed=42,
    warmup_epochs=2,
    model_name="vit_large_patch16_dinov3.lvd1689m",
    train_parquet=TrainConfig.train_parquet,
    val_parquet=TrainConfig.val_parquet,
    tag_to_id=TrainConfig.tag_to_id,
    checkpoint_dir=TrainConfig.checkpoint_dir,
    run_dir=TrainConfig.run_dir,
    threshold=0.5,
    prefetch_factor=2,
    download_workers=32,
    download_retries=3,
    save_epochs=1,
    backbone_lr_mult=0.1,
    head_lr_mult=1.0,
    proj_lr_mult=1.0,
    ema_decay=0.999,
    num_classes=0,
    val_interval=1,
    compute_multilabel_metrics=True,
    checkpoint="",
    projector="",
    teacher_path="",
    kd_weight=0.5,
)
