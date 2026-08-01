from src.config import TrainConfig

CKPT = "models/dtq_dinov3l16_448x448_ep13_bestmAP.pt"

config = TrainConfig(
    model_name="vit_small_patch16_dinov3.lvd1689m",
    checkpoint=CKPT,
    backbone_lr_mult=1.0,
    head_lr_mult=0.1,
    proj_lr_mult=1.0,
    batch_size=128,
    learning_rate=3e-4,
    epochs=10,
    num_workers=2,
)
