import argparse
import gc
import importlib.util
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
from .losses import build_loss, kd_logits_loss
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


def _describe_model(model, config) -> list[str]:
    """Compact model card printed before training (backbone / head / projector)."""
    backbone_n = sum(p.numel() for p in model.backbone.parameters())
    head_n = sum(p.numel() for p in model.head.parameters())
    total = backbone_n + head_n
    lines = [
        f"  Model: {config.model_name}",
        f"    Backbone : {backbone_n / 1e6:.1f}M params, embed dim {model.backbone.num_features}",
        f"    Head     : {head_n / 1e6:.1f}M params, embed dim {model.head.tag_queries.shape[1]}, {config.num_classes} tags",
    ]
    if getattr(model, "projector", None) is not None:
        proj = model.projector.proj
        proj_n = sum(p.numel() for p in model.projector.parameters())
        lines.append(f"    Projector: {proj_n / 1e6:.1f}M params ({proj.in_features} -> {proj.out_features})")
        total += proj_n
    lines.append(f"    Total    : {total / 1e6:.1f}M trainable params")
    return lines


def load_config_module(path: str) -> TrainConfig:
    """Load an experiment config module that exports `config = TrainConfig(...)`."""
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    spec = importlib.util.spec_from_file_location(f"dtq_cfg_{p.stem}", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = getattr(mod, "config", None)
    if not isinstance(cfg, TrainConfig):
        raise TypeError(
            f"Config module {path} must expose `config` as a TrainConfig instance, got {type(cfg).__name__}"
        )
    return cfg


def _peek_head_embed_dim(state: dict, weights_key: str = "model_ema") -> int:
    """Recover the head embed dim from a checkpoint's state dict."""
    w = state[weights_key]
    key = "head.cross_attn.in_proj_weight"
    if key in w:
        return w[key].shape[0] // 3  # MultiheadAttention in_proj: (3*d, d)
    key2 = "head.tag_queries"
    if key2 in w:
        return w[key2].shape[1]
    raise KeyError(
        f"Cannot determine head embed dim from checkpoint (missing {key} / {key2})"
    )


class _KdTeacher:
    """Frozen logits teacher for distillation: PyTorch (.pt) or ONNX dir.

    logits(images) -> float32 logits tensor on the training device.  The .pt
    path builds the teacher from its own saved config (model_name, projector),
    loads model_ema (fallback model), freezes it, and validates that the
    student's tag vocabulary is a SUBSET of the teacher's (exact match and
    subsets are both fine; student-only tags raise).  For subsets, expose
    `student_slice` (teacher indices in student tag order) so the loss can
    align teacher logits to the student's vocab.
    """

    def __init__(self, path: str, tag_to_id: dict[str, int], dev, is_main: bool = True):
        p = Path(path)
        if p.name == "model.onnx":
            onnx_file = p
        elif p.is_dir() and (p / "model.onnx").exists():
            onnx_file = p / "model.onnx"
        else:
            onnx_file = None
        self.dev = dev
        self._model = None
        self._sess = None
        self._input = None
        self._output = None
        self.student_slice = None  # LongTensor (teacher indices) when student vocab is a subset

        if onnx_file is not None:
            import onnxruntime as ort

            self._sess = ort.InferenceSession(
                str(onnx_file),
                providers=[("CUDAExecutionProvider", {}), "CPUExecutionProvider"],
            )
            self._input = self._sess.get_inputs()[0].name
            self._output = self._sess.get_outputs()[0].name
            sidecar = onnx_file.parent / "tag_to_id.json"
            if sidecar.exists():
                saved = json.loads(sidecar.read_text())
                self._set_vocab_mapping(saved, tag_to_id)
        else:
            try:
                state = torch.load(path, map_location="cpu", mmap=True)  # weights_only=True (safe: our ckpts)
            except (RuntimeError, OSError, ValueError, TypeError):
                state = torch.load(path, map_location="cpu", weights_only=False)
            saved = state.get("tag_to_id", {})
            self._set_vocab_mapping(saved, tag_to_id)
            tcfg = state.get("config", {})
            tname = tcfg.get("model_name")
            if not tname:
                raise RuntimeError(f"Teacher checkpoint missing 'config.model_name': {path}")
            proj = tcfg.get("projector") or ""
            head_embed = None
            if proj:
                parts = str(proj).split(":")
                if len(parts) == 2:
                    head_embed = int(parts[1])
                else:
                    raise RuntimeError(f"Invalid teacher projector field: {proj!r}")
            teacher = ImageTagger(tname, len(saved), pretrained=False, head_embed_dim=head_embed)
            weights = state.get("model_ema") or state.get("model")
            if weights is None:
                raise RuntimeError(f"Teacher checkpoint has neither 'model_ema' nor 'model': {path}")
            teacher.load_state_dict(weights)
            teacher.to(dev).eval()
            for p in teacher.parameters():
                p.requires_grad_(False)
            self._model = teacher

        if is_main:
            kind = "ONNX" if self._sess is not None else "PyTorch"
            vocab = ""
            if self.student_slice is not None:
                vocab = f" (vocab subset {len(self.student_slice)}/{len(saved)})"
            print(f"  [KD] teacher: {Path(path).name} ({kind}, frozen, eval{vocab})")

    def _set_vocab_mapping(self, teacher_vocab: dict, student_vocab: dict) -> None:
        """Align student tags to teacher indices.

        Student tags are ordered by their id (student logits are positional).
        Every student tag must exist in the teacher; student-only tags raise.
        """
        student_sorted = sorted(student_vocab.items(), key=lambda kv: kv[1])
        missing = [name for name, _ in student_sorted if name not in teacher_vocab]
        if missing:
            shown = ", ".join(missing[:20])
            raise RuntimeError(
                f"Teacher is missing {len(missing)} student tags (can't distill unknown tags): {shown}"
            )
        slices = [teacher_vocab[name] for name, _ in student_sorted]
        if len(teacher_vocab) == len(slices) and slices == list(range(len(slices))):
            self.student_slice = None  # identical vocab, already aligned
        else:
            self.student_slice = torch.tensor(slices, dtype=torch.long)

    @torch.no_grad()
    def logits(self, images) -> torch.Tensor:
        if self._sess is not None:
            np_in = images.float().cpu().numpy()
            raw = self._sess.run([self._output], {self._input: np_in})[0]
            return torch.as_tensor(raw, dtype=torch.float32, device=self.dev)
        return self._model(images)


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
        for key in ("model_name",):
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

    # Transfer source: config.checkpoint (CLI --checkpoint already applied to
    # config earlier).  Peek the teacher's head embed dim so the head is built
    # at the teacher's dim and a projector bridges any backbone dim mismatch.
    # mmap=True keeps the 4.8GB source out of RAM; it is freed after transfer.
    transfer_state = None
    head_embed = None
    if config.checkpoint and config._resume_ckpt is None:
        try:
            transfer_state = torch.load(config.checkpoint, map_location="cpu", mmap=True)
        except (RuntimeError, OSError, ValueError):
            transfer_state = torch.load(config.checkpoint, map_location="cpu")
        head_embed = _peek_head_embed_dim(transfer_state)
    elif config.projector_dims():
        head_embed = config.projector_dims()[1]

    model = ImageTagger(
        config.model_name,
        config.num_classes,
        tag_embeddings=tag_embeddings,
        head_embed_dim=head_embed,
    ).to(dev)

    if transfer_state is not None:
        teacher_embed = _peek_head_embed_dim(transfer_state)
        backbone_dim = model.backbone.num_features
        if teacher_embed != backbone_dim:
            config.projector = f"{backbone_dim}:{teacher_embed}"
        else:
            config.projector = ""
        if is_main:
            print(f"  Transfer source: {config.checkpoint} (head embed dim {teacher_embed})")
        stats = transfer_weights(model, tag_to_id, config.checkpoint, weights_key="model_ema", verbose=is_main, state=transfer_state)
        if is_main:
            print(f"  Transferred {stats['query_transfer_count']} / {stats['new_tag_count']} tag queries")
        del transfer_state
        gc.collect()
        if is_main:
            print("  Freed transfer checkpoint from RAM (teacher for KD loads separately)")

    if config._resume_ckpt is not None:
        model.load_state_dict(config._resume_ckpt["model"], strict=True)
        if is_main:
            print(f"  Restored model weights from checkpoint (epoch {config._resume_ckpt.get('epoch', 0)})")

    if is_main:
        for line in _describe_model(model, config):
            print(line)
        # print(model)

    if is_main:
        # EMA lives only on rank 0.  Gradients are synced by DDP so model
        # weights are identical across ranks — rank 0's EMA is correct for all.
        ema = EMA(model, config.ema_decay)
        if config._resume_ckpt is not None and "model_ema" in config._resume_ckpt:
            ema.shadow = {k: v.to(dev) for k, v in config._resume_ckpt["model_ema"].items()}

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

    raw = model.module if distributed else model
    backbone_params = raw.backbone.parameters()
    head_params = raw.head.parameters()
    param_groups = [
        {"params": backbone_params, "lr": config.learning_rate * config.backbone_lr_mult},
        {"params": head_params, "lr": config.learning_rate * config.head_lr_mult},
    ]
    if raw.projector is not None:
        param_groups.append(
            {"params": raw.projector.parameters(), "lr": config.learning_rate * config.proj_lr_mult}
        )
    try:
        optimizer = AdamW(param_groups, lr=config.learning_rate, weight_decay=config.weight_decay, fused=True)
    except TypeError:
        optimizer = AdamW(param_groups, lr=config.learning_rate, weight_decay=config.weight_decay)

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

    teacher = None
    if config.teacher_path:
        if config.kd_weight > 0:
            teacher = _KdTeacher(config.teacher_path, tag_to_id, dev, is_main=is_main)
            if is_main:
                print(f"  [KD] kd_weight={config.kd_weight}")
        elif is_main:
            print(f"  [KD] teacher_path set but kd_weight=0 — teacher NOT loaded; remove teacher_path to fully disable KD")

    for epoch in range(start_epoch, config.epochs):
        if distributed and hasattr(train.sampler, "set_epoch"):
            train.sampler.set_epoch(epoch)

        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats()  # measure this epoch only (post-compile)

        started = time.time(); model.train(); total = 0.0
        train_iter = tqdm(train, desc=f"Epoch {epoch + 1}/{config.epochs}") if is_main else train
        for batch in train_iter:
            optimizer.zero_grad(set_to_none=True)
            with ctx:
                student_logits = model(batch["image"].to(dev, non_blocking=True))
                loss = loss_fn(student_logits, batch["labels"].to(dev, non_blocking=True))
                if teacher is not None:
                    teacher_logits = teacher.logits(batch["image"].to(dev, non_blocking=True))
                    loss = loss + config.kd_weight * kd_logits_loss(
                        student_logits, teacher_logits, teacher_slice=teacher.student_slice
                    )
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
            if dev.type == "cuda":
                print(
                    f"  VRAM: peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB | "
                    f"reserved {torch.cuda.memory_reserved() / 2**30:.2f} GiB"
                )
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
    parser.add_argument("--config", type=str, default=None,
                        help="Path to experiment config module exporting `config` (a TrainConfig instance)")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to pretrained checkpoint for weight transfer (overrides config.checkpoint)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume training from checkpoint (e.g., runs/.../checkpoints/last.pt)")
    parser.add_argument("--wandb-run-id", type=str, default=None,
                        help="Wandb run ID to resume (creates new run if omitted)")
    parser.add_argument("--batch-size", type=int,
                        help="Override batch size per GPU (e.g., 24 for 2x4090)")
    parser.add_argument("--teacher-path", type=str, default=None,
                        help="Teacher checkpoint (.pt) or ONNX dir for logits distillation (overrides config.teacher_path)")
    parser.add_argument("--kd-weight", type=float, default=None,
                        help="KD loss weight, default 0.5 (requires a teacher source)")
    args = parser.parse_args()

    if args.resume and args.checkpoint:
        parser.error("--resume and --checkpoint are mutually exclusive")
    if args.resume and (args.epochs is not None or args.batch_size is not None):
        parser.error("--resume is mutually exclusive with --epochs and --batch-size (checkpoint config is authoritative)")
    if args.resume and args.config:
        parser.error("--resume and --config are mutually exclusive (checkpoint config is authoritative)")
    if args.kd_weight is not None and args.kd_weight < 0:
        parser.error("--kd-weight must be >= 0")

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
                val = ckpt_cfg[field]
                # Checkpoint stores Paths as strings; restore to Path
                ann = TrainConfig.__annotations__.get(field)
                if ann == Path:
                    val = Path(val)
                setattr(config, field, val)
        config.no_wandb = args.no_wandb
        if args.teacher_path:
            config.teacher_path = args.teacher_path
        if args.kd_weight is not None:
            config.kd_weight = args.kd_weight
        config._resume_ckpt = ckpt
        config._resume_path = args.resume
    else:
        config = load_config_module(args.config) if args.config else TrainConfig()
        config.no_wandb = args.no_wandb
        if args.epochs:
            config.epochs = args.epochs
        if args.batch_size:
            config.batch_size = args.batch_size
        if args.checkpoint:
            config.checkpoint = args.checkpoint
        if args.teacher_path:
            config.teacher_path = args.teacher_path
        if args.kd_weight is not None:
            config.kd_weight = args.kd_weight
        config._resume_ckpt = None
        config._resume_path = None
    config._wandb_run_id = args.wandb_run_id

    if args.kd_weight is not None and not config.teacher_path:
        parser.error("--kd-weight requires --teacher-path (or config.teacher_path)")
    if config.kd_weight < 0:
        parser.error("--kd-weight must be >= 0")

    run(config)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
