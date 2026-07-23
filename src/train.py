import argparse
import json
import math
import time
from contextlib import nullcontext
from datetime import datetime

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from .config import TrainConfig
from .dataset import make_loader
from .losses import build_loss
from .metrics import multilabel_metrics
from .model import ImageTagger
from .transforms import train_transforms, val_transforms
from .utils import device, save_checkpoint, seed_everything, load_json


def run(config):
    tag_to_id = load_json(config.tag_to_id)
    config.num_classes = len(tag_to_id)

    wandb_run = None
    if not config.no_wandb:
        import wandb
        wandb_run = wandb.init(project="danboorutagclip", config=config.to_dict())

    seed_everything(config.seed); dev = device()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = config.run_dir / timestamp; checkpoint_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True); checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config.run_dir = run_dir; config.checkpoint_dir = checkpoint_dir

    train = make_loader(config.train_parquet, config, train_transforms(config.image_size), True)
    val = make_loader(config.val_parquet, config, val_transforms(config.image_size), False)

    model = ImageTagger(config.model_name, config.num_classes).to(dev); loss_fn = build_loss()
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    steps = config.epochs * len(train); warmup = config.warmup_epochs * len(train)
    scheduler = LambdaLR(optimizer, lambda step: (step + 1) / max(1, warmup) if step < warmup else 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup))))

    best = -float("inf")
    amp = dev.type == "cuda"; ctx = torch.autocast(device_type=dev.type, dtype=torch.bfloat16) if amp else nullcontext()

    for epoch in range(config.epochs):
        started = time.time(); model.train(); total = 0.0
        for batch in tqdm(train, desc=f"Epoch {epoch + 1}/{config.epochs}"):
            optimizer.zero_grad(set_to_none=True)
            with ctx: loss = loss_fn(model(batch["image"].to(dev, non_blocking=True)), batch["labels"].to(dev, non_blocking=True))
            loss.backward(); optimizer.step(); scheduler.step(); total += loss.item()
        model.eval(); losses = []; predictions = []; targets = []
        with torch.no_grad():
            for batch in val:
                with ctx: logits = model(batch["image"].to(dev)); current = loss_fn(logits, batch["labels"].to(dev))
                losses.append(current.item()); predictions.append(torch.sigmoid(logits).float().cpu()); targets.append(batch["labels"].cpu())
        metrics = multilabel_metrics(torch.cat(predictions), torch.cat(targets))
        row = {"epoch": epoch + 1, "train_loss": total / len(train), "val_loss": sum(losses) / len(losses), **{k: v for k, v in metrics.items() if k != "per_tag_ap"}, "lr": optimizer.param_groups[0]["lr"], "duration": time.time() - started}
        with (config.run_dir / "metrics.jsonl").open("a") as file: file.write(json.dumps(row) + "\n")
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "epoch": epoch + 1, "config": config.to_dict(), "tag_to_id": tag_to_id, "metrics": row}
        save_checkpoint(config.checkpoint_dir / "last.pt", state)
        if metrics["map"] > best: best = metrics["map"]; save_checkpoint(config.checkpoint_dir / "best.pt", state)
        if config.save_epochs > 0 and (epoch + 1) % config.save_epochs == 0:
            save_checkpoint(config.checkpoint_dir / f"epoch-{epoch + 1}.pt", state)
        print(row)
        if wandb_run: wandb_run.log(row)
    if wandb_run: wandb_run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--epochs", type=int); parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args(); config = TrainConfig(); config.no_wandb = args.no_wandb
    if args.epochs: config.epochs = args.epochs
    run(config)
