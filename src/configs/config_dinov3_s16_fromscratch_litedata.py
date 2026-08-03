from src.config import TrainConfig

config = TrainConfig(
    model_name="vit_small_patch16_dinov3.lvd1689m",
    backbone_lr_mult=0.1,
    batch_size=128,
    learning_rate=3e-4,
    epochs=10,
    num_workers=8,
)
