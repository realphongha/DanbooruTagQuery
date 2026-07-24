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
from .metrics import multilabel_metrics, print_metrics
from .model import ImageTagger
from .transforms import train_transforms, val_transforms
from .utils import device, save_checkpoint, seed_everything, load_json

try:
    from transformers import SiglipModel, SiglipTokenizer
except ImportError:
    SiglipModel = None
    SiglipTokenizer = None


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        self.backup = {}

    def update(self, model):
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.requires_grad:
                    self.shadow[n] = self.decay * self.shadow[n] + (1 - self.decay) * p.data

    def apply(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.backup[n] = p.detach().clone()
                p.data = self.shadow[n]

    def restore(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.data = self.backup[n]


def run(config):
    tag_to_id = load_json(config.tag_to_id)
    config.num_classes = len(tag_to_id)

    dev = device()
    tag_embeddings = None
    if config.head_type == "tag_query_head" and config.use_siglip_init:
        if SiglipModel is None:
            raise RuntimeError("transformers is required for SigLIP tag query initialization")
        tags_sorted = sorted(tag_to_id, key=tag_to_id.get)
        texts = [t.replace("_", " ") for t in tags_sorted]

        siglip = SiglipModel.from_pretrained("google/siglip-base-patch16-224").to(dev).eval()
        tokenizer = SiglipTokenizer.from_pretrained("google/siglip-base-patch16-224")

        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(dev) for k, v in inputs.items()}
        with torch.no_grad():
            text_feats = siglip.get_text_features(**inputs)

        text_feats = torch.nn.functional.normalize(text_feats, dim=-1)
        mean = text_feats.mean(dim=0, keepdim=True)
        centered = text_feats - mean
        _, _, Vt = torch.linalg.svd(centered, full_matrices=False)
        tag_embeddings = (centered @ Vt[:384].T).cpu()

    wandb_run = None
    if not config.no_wandb:
        import wandb
        wandb_run = wandb.init(project="danboorutagclip", config=config.to_dict())

    seed_everything(config.seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = config.run_dir / timestamp; checkpoint_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True); checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config.run_dir = run_dir; config.checkpoint_dir = checkpoint_dir

    train = make_loader(config.train_parquet, config, train_transforms(config.image_size), True)
    val = make_loader(config.val_parquet, config, val_transforms(config.image_size), False)

    model = ImageTagger(config.model_name, config.num_classes, head_type=config.head_type, tag_embeddings=tag_embeddings).to(dev); loss_fn = build_loss()
    ema = EMA(model, config.ema_decay)
    optimizer = AdamW([
        {"params": model.backbone.parameters(), "lr": config.learning_rate * config.backbone_lr_mult},
        {"params": model.head.parameters()},
    ], lr=config.learning_rate, weight_decay=config.weight_decay)
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
            ema.update(model)
        should_val = (epoch + 1) % config.val_interval == 0 or epoch == config.epochs - 1

        if should_val:
            ema.apply(model)
            model.eval(); losses = []; predictions = []; targets = []
            with torch.no_grad():
                for batch in val:
                    with ctx: logits = model(batch["image"].to(dev)); current = loss_fn(logits, batch["labels"].to(dev))
                    losses.append(current.item()); predictions.append(torch.sigmoid(logits).float().cpu()); targets.append(batch["labels"].cpu())
            ema.restore(model)
            metrics = multilabel_metrics(torch.cat(predictions), torch.cat(targets))
            row = {"epoch": epoch + 1, "train_loss": total / len(train), "val_loss": sum(losses) / len(losses), **{k: v for k, v in metrics.items() if k != "per_tag_ap"}, "lr": optimizer.param_groups[0]["lr"], "duration": time.time() - started}
            print_metrics(metrics, tag_to_id, f"Epoch {epoch + 1}/{config.epochs}", extra={"Train loss": total / len(train), "Val loss": sum(losses) / len(losses), "LR": optimizer.param_groups[0]["lr"]}, show_per_tag=False)
            if metrics["map"] > best:
                best = metrics["map"]
                ema.apply(model)
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                ema.restore(model)
                save_checkpoint(config.checkpoint_dir / "best.pt", {"model": best_state, "model_ema": ema.shadow, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "epoch": epoch + 1, "config": config.to_dict(), "tag_to_id": tag_to_id, "metrics": row})
        else:
            row = {"epoch": epoch + 1, "train_loss": total / len(train), "lr": optimizer.param_groups[0]["lr"], "duration": time.time() - started}
            print(f"Epoch {epoch + 1}/{config.epochs}  train_loss={total / len(train):.6f}  lr={optimizer.param_groups[0]['lr']:.2e}  duration={time.time() - started:.1f}s")

        with (config.run_dir / "metrics.jsonl").open("a") as file: file.write(json.dumps(row) + "\n")
        state = {"model": model.state_dict(), "model_ema": ema.shadow, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "epoch": epoch + 1, "config": config.to_dict(), "tag_to_id": tag_to_id, "metrics": row}
        save_checkpoint(config.checkpoint_dir / "last.pt", state)
        if config.save_epochs > 0 and (epoch + 1) % config.save_epochs == 0:
            save_checkpoint(config.checkpoint_dir / f"epoch-{epoch + 1}.pt", state)
        if wandb_run: wandb_run.log(row)
    if wandb_run: wandb_run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--epochs", type=int); parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args(); config = TrainConfig(); config.no_wandb = args.no_wandb
    if args.epochs: config.epochs = args.epochs
    run(config)
