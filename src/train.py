import argparse
import json
import math
import os
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from .config import TrainConfig
from .dataset import make_loader
from .losses import build_loss
from .metrics import multilabel_metrics, print_metrics
from .model import ImageTagger, transfer_weights
from .transforms import train_transforms, val_transforms
from .utils import (
    device,
    is_distributed,
    get_rank,
    get_world_size,
    rank0_only,
    print_sdp_backend_status,
    save_checkpoint,
    seed_everything,
    load_json,
)

try:
    from transformers import SiglipModel, SiglipTokenizer
except ImportError:
    SiglipModel = None
    SiglipTokenizer = None


class EMA:
    @staticmethod
    def _strip(n: str) -> str:
        return n.removeprefix("_orig_mod.")

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {self._strip(n): p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        self.backup = {}

    def update(self, model):
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.requires_grad:
                    self.shadow[self._strip(n)] = self.decay * self.shadow[self._strip(n)] + (1 - self.decay) * p.data

    def apply(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.backup[self._strip(n)] = p.detach().clone()
                p.data = self.shadow[self._strip(n)]

    def restore(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.data = self.backup[self._strip(n)]


def _clean(sd):
    return {k.removeprefix("_orig_mod."): v for k, v in sd.items()}


def run(config):
    tag_to_id = load_json(config.tag_to_id)
    config.num_classes = len(tag_to_id)

    distributed = is_distributed()
    is_main = rank0_only()
    rank = get_rank()
    world_size = get_world_size()

    # Resume setup — compatibility checks, epoch/best restoration, run_dir derivation
    if config._resume_ckpt is not None:
        ckpt = config._resume_ckpt
        # Verify tag vocabulary matches exactly
        saved_tag_to_id = ckpt.get("tag_to_id", {})
        if saved_tag_to_id != tag_to_id:
            old_set = set(saved_tag_to_id.keys())
            new_set = set(tag_to_id.keys())
            if old_set != new_set:
                raise RuntimeError(
                    f"Checkpoint tag vocabulary mismatch: "
                    f"{len(old_set - new_set)} tags in checkpoint only, "
                    f"{len(new_set - old_set)} tags in current config. "
                    f"Use --checkpoint (weight transfer) instead of --resume for vocabulary changes."
                )
        # Verify model architecture matches
        saved_cfg = ckpt.get("config", {})
        for key in ("model_name", "head_type"):
            if saved_cfg.get(key) != getattr(config, key):
                raise RuntimeError(
                    f"Checkpoint config mismatch: {key} = {saved_cfg.get(key)} "
                    f"vs current {getattr(config, key)}. Cannot resume."
                )
        # Warn on world size mismatch — optimizer state may be slightly stale but recovery is safe
        saved_ws = ckpt.get("world_size", 1)
        if distributed and saved_ws != world_size:
            if is_main:
                print(f"  Warning: checkpoint world_size={saved_ws}, current={world_size}. Optimizer state will recover.")

        start_epoch = ckpt.get("epoch", 0)
        best = ckpt.get("best", -float("inf") if config.compute_multilabel_metrics else float("inf"))

        # Derive run_dir from checkpoint path
        ckpt_path = Path(config._resume_path)
        if ckpt_path.parent.name == "checkpoints":
            run_dir = ckpt_path.parent.parent
            checkpoint_dir = ckpt_path.parent
        else:
            run_dir = Path(config.run_dir)
            checkpoint_dir = run_dir / "checkpoints"
        config.run_dir = run_dir
        config.checkpoint_dir = checkpoint_dir

        if is_main:
            print(f"Resuming from epoch {start_epoch}/{config.epochs}  (best={'mAP' if config.compute_multilabel_metrics else 'val_loss'}: {best:.6f})")
            print(f"  Run dir: {run_dir}")
    else:
        start_epoch = 0
        best = -float("inf") if config.compute_multilabel_metrics else float("inf")

    dev = device()
    if dev.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
    if is_main:
        print_sdp_backend_status()
        if distributed:
            print(f"  DDP: {world_size} GPUs, rank {rank}")

    tag_embeddings = None
    if config.head_type == "tag_query_head" and config.use_siglip_init:
        if SiglipModel is None:
            raise RuntimeError("transformers is required for SigLIP tag query initialization")
        if is_main:
            print("Initializing tag query head with SigLIP...")
            tags_sorted = sorted(tag_to_id, key=tag_to_id.get)
            texts = [t.replace("_", " ") for t in tags_sorted]
            siglip = SiglipModel.from_pretrained("google/siglip-base-patch16-224").to(dev).eval()
            tokenizer = SiglipTokenizer.from_pretrained("google/siglip-base-patch16-224")
            inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
            inputs = {k: v.to(dev) for k, v in inputs.items()}
            with torch.no_grad():
                text_feats = siglip.text_model(**inputs).pooler_output
            text_feats = torch.nn.functional.normalize(text_feats, dim=-1)
            mean = text_feats.mean(dim=0, keepdim=True)
            centered = text_feats - mean
            _, _, Vt = torch.linalg.svd(centered, full_matrices=False)
            n = min(centered.size(0), Vt.size(0), 384)
            tag_embeddings = centered @ Vt[:n].T
            if n < 384:
                tag_embeddings = torch.cat(
                    [tag_embeddings, torch.zeros(tag_embeddings.size(0), 384 - n)],
                    dim=1,
                )
        if distributed:
            tag_embeddings = (
                tag_embeddings if is_main else torch.zeros(len(tag_to_id), 384)
            )
            torch.distributed.broadcast(tag_embeddings, src=0)

    wandb_run = None
    if is_main and not config.no_wandb:
        import wandb
        wandb_kwargs = {"project": "danboorutagquery", "config": config.to_dict()}
        if config._wandb_run_id:
            wandb_kwargs["id"] = config._wandb_run_id
            wandb_kwargs["resume"] = "must"
        wandb_run = wandb.init(**wandb_kwargs)

    seed_everything(config.seed)
    if config._resume_ckpt is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = config.run_dir / timestamp
        checkpoint_dir = run_dir / "checkpoints"
        config.run_dir = run_dir
        config.checkpoint_dir = checkpoint_dir
    if is_main:
        config.run_dir.mkdir(parents=True, exist_ok=True)
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train = make_loader(config.train_parquet, config, train_transforms(config.image_size), True)
    val = make_loader(config.val_parquet, config, val_transforms(config.image_size), False)

    model = ImageTagger(config.model_name, config.num_classes, head_type=config.head_type, tag_embeddings=tag_embeddings).to(dev)
    if args.checkpoint:
        stats = transfer_weights(model, tag_to_id, args.checkpoint, weights_key="model_ema", verbose=is_main)
        if is_main:
            print(f"  Transferred {stats['query_transfer_count']} / {stats['new_tag_count']} tag queries")

    if config._resume_ckpt is not None:
        model.load_state_dict(config._resume_ckpt["model"], strict=True)
        if is_main:
            print(f"  Restored model weights from checkpoint (epoch {config._resume_ckpt.get('epoch', 0)})")

    if is_main:
        # EMA lives only on rank 0.  Gradients are synced by DDP so model
        # weights are identical across ranks — rank 0's EMA is correct for all.
        ema = EMA(model, config.ema_decay)
        if config._resume_ckpt is not None and "model_ema" in config._resume_ckpt:
            ema.shadow = config._resume_ckpt["model_ema"]

    if dev.type == "cuda":
        compile_mode = "default" if distributed else "max-autotune"
        try:
            model = torch.compile(model, mode=compile_mode, dynamic=False)
            if is_main:
                print(f"  [torch.compile] enabled with {compile_mode}")
        except Exception as e:
            if is_main:
                print(f"  [torch.compile] failed, using eager mode: {e}")

    if distributed:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            gradient_as_bucket_view=True,
            static_graph=True,  # fixed computation graph
        )

    loss_fn = build_loss()

    backbone_params = (
        model.module.backbone.parameters() if distributed else model.backbone.parameters()
    )
    head_params = (
        model.module.head.parameters() if distributed else model.head.parameters()
    )
    try:
        optimizer = AdamW([
            {"params": backbone_params, "lr": config.learning_rate * config.backbone_lr_mult},
            {"params": head_params},
        ], lr=config.learning_rate, weight_decay=config.weight_decay, fused=True)
    except TypeError:
        optimizer = AdamW([
            {"params": backbone_params, "lr": config.learning_rate * config.backbone_lr_mult},
            {"params": head_params},
        ], lr=config.learning_rate, weight_decay=config.weight_decay)

    steps = config.epochs * len(train); warmup = config.warmup_epochs * len(train)
    scheduler = LambdaLR(optimizer, lambda step: (step + 1) / max(1, warmup) if step < warmup else 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup))))

    if config._resume_ckpt is not None:
        optimizer.load_state_dict(config._resume_ckpt["optimizer"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(dev)
        scheduler.load_state_dict(config._resume_ckpt["scheduler"])
        if is_main:
            print(f"  Restored optimizer & scheduler state (step ~{scheduler.last_epoch})")

    amp = dev.type == "cuda"; ctx = torch.autocast(device_type=dev.type, dtype=torch.bfloat16) if amp else nullcontext()

    for epoch in range(start_epoch, config.epochs):
        if distributed and hasattr(train.sampler, "set_epoch"):
            train.sampler.set_epoch(epoch)

        started = time.time(); model.train(); total = 0.0
        train_iter = tqdm(train, desc=f"Epoch {epoch + 1}/{config.epochs}") if is_main else train
        for batch in train_iter:
            optimizer.zero_grad(set_to_none=True)
            with ctx: loss = loss_fn(model(batch["image"].to(dev, non_blocking=True)), batch["labels"].to(dev, non_blocking=True))
            loss.backward(); optimizer.step(); scheduler.step(); total += loss.item()
            if is_main:
                raw = model.module if distributed else model
                ema.update(raw)
        should_val = (epoch + 1) % config.val_interval == 0 or epoch == config.epochs - 1

        if should_val:
            raw = model.module if distributed else model
            if is_main:
                ema.apply(raw)
            model.eval(); val_loss_sum = 0.0; val_loss_cnt = 0
            all_preds = []; all_targets = []
            with torch.no_grad():
                for batch in val:
                    with ctx: logits = model(batch["image"].to(dev)); current = loss_fn(logits, batch["labels"].to(dev))
                    bsz = batch["labels"].size(0)
                    val_loss_sum += current.item() * bsz
                    val_loss_cnt += bsz
                    if config.compute_multilabel_metrics:
                        all_preds.append(torch.sigmoid(logits).float().cpu())
                        all_targets.append(batch["labels"].cpu())
            if is_main:
                ema.restore(raw)

            # gather loss across GPUs (weighted by batch size, handles uneven last batch)
            if distributed:
                loss_data = torch.tensor([val_loss_sum, val_loss_cnt], device=dev)
                torch.distributed.all_reduce(loss_data)
                val_loss = loss_data[0].item() / loss_data[1].item()
            else:
                val_loss = val_loss_sum / val_loss_cnt

            # gather predictions + targets to rank 0 for metric computation
            if distributed and config.compute_multilabel_metrics:
                local_preds = torch.cat(all_preds)
                local_targets = torch.cat(all_targets)
                gathered_preds = [None] * world_size if is_main else None
                gathered_targets = [None] * world_size if is_main else None
                torch.distributed.gather_object(local_preds, gathered_preds if is_main else None, dst=0)
                torch.distributed.gather_object(local_targets, gathered_targets if is_main else None, dst=0)
                if is_main:
                    all_preds = gathered_preds
                    all_targets = gathered_targets
                else:
                    all_preds = []; all_targets = []  # discard on non-main ranks

            if config.compute_multilabel_metrics and is_main:
                metrics = multilabel_metrics(torch.cat(all_preds), torch.cat(all_targets))
                row = {"epoch": epoch + 1, "train_loss": total / len(train), "val_loss": val_loss, **{k: v for k, v in metrics.items() if k != "per_tag_ap"}, "lr": optimizer.param_groups[0]["lr"], "duration": time.time() - started}
                print_metrics(metrics, tag_to_id, f"Epoch {epoch + 1}/{config.epochs}", extra={"Train loss": total / len(train), "Val loss": val_loss, "LR": optimizer.param_groups[0]["lr"]}, show_per_tag=False)
                if metrics["map"] > best:
                    best = metrics["map"]
                    ema.apply(raw)
                    best_state = _clean(raw.state_dict())
                    ema.restore(raw)
                    save_checkpoint(config.checkpoint_dir / "best.pt", {"model": best_state, "model_ema": ema.shadow, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "epoch": epoch + 1, "config": config.to_dict(), "tag_to_id": tag_to_id, "metrics": row, "best": best, "world_size": get_world_size()})
            elif is_main:
                row = {"epoch": epoch + 1, "train_loss": total / len(train), "val_loss": val_loss, "lr": optimizer.param_groups[0]["lr"], "duration": time.time() - started}
                if val_loss < best:
                    best = val_loss
                    ema.apply(raw)
                    best_state = _clean(raw.state_dict())
                    ema.restore(raw)
                    save_checkpoint(config.checkpoint_dir / "best.pt", {"model": best_state, "model_ema": ema.shadow, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "epoch": epoch + 1, "config": config.to_dict(), "tag_to_id": tag_to_id, "metrics": row, "best": best, "world_size": get_world_size()})
                print(f"Epoch {epoch + 1}/{config.epochs}  train_loss={total / len(train):.6f}  val_loss={val_loss:.6f}  lr={optimizer.param_groups[0]['lr']:.2e}  duration={time.time() - started:.1f}s")
        else:
            row = {"epoch": epoch + 1, "train_loss": total / len(train), "lr": optimizer.param_groups[0]["lr"], "duration": time.time() - started}
            if is_main:
                print(f"Epoch {epoch + 1}/{config.epochs}  train_loss={total / len(train):.6f}  lr={optimizer.param_groups[0]['lr']:.2e}  duration={time.time() - started:.1f}s")

        if is_main:
            with (config.run_dir / "metrics.jsonl").open("a") as file: file.write(json.dumps(row) + "\n")
            raw = model.module if distributed else model
            state = {"model": _clean(raw.state_dict()), "model_ema": ema.shadow, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "epoch": epoch + 1, "config": config.to_dict(), "tag_to_id": tag_to_id, "metrics": row, "best": best, "world_size": get_world_size()}
            save_checkpoint(config.checkpoint_dir / "last.pt", state)
            if config.save_epochs > 0 and (epoch + 1) % config.save_epochs == 0:
                save_checkpoint(config.checkpoint_dir / f"epoch-{epoch + 1}.pt", state)
            if wandb_run: wandb_run.log(row)
    if wandb_run: wandb_run.finish()
    if distributed:
        torch.distributed.barrier()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to pretrained checkpoint for weight transfer")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume training from checkpoint (e.g., runs/.../checkpoints/last.pt)")
    parser.add_argument("--wandb-run-id", type=str, default=None,
                        help="Wandb run ID to resume (creates new run if omitted)")
    parser.add_argument("--batch-size", type=int,
                        help="Override batch size per GPU (e.g., 24 for 2x4090)")
    args = parser.parse_args()

    if args.resume and args.checkpoint:
        parser.error("--resume and --checkpoint are mutually exclusive")
    if args.resume and (args.epochs is not None or args.batch_size is not None):
        parser.error("--resume is mutually exclusive with --epochs and --batch-size (checkpoint config is authoritative)")

    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if local_rank >= 0:
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        ckpt_cfg = ckpt.get("config", {})
        config = TrainConfig()
        for field in TrainConfig.__dataclass_fields__:
            if field in ckpt_cfg:
                setattr(config, field, ckpt_cfg[field])
        config.no_wandb = args.no_wandb
        config._resume_ckpt = ckpt
        config._resume_path = args.resume
    else:
        config = TrainConfig()
        config.no_wandb = args.no_wandb
        if args.epochs:
            config.epochs = args.epochs
        if args.batch_size:
            config.batch_size = args.batch_size
        config._resume_ckpt = None
        config._resume_path = None
    config._wandb_run_id = args.wandb_run_id

    run(config)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
