# Tutorial — Step-by-Step Guide

Copy-paste commands. No troubleshooting. No architecture discussion.

---

## Setup

Install dependencies:

```bash
uv sync
```

Dataset parquet files (two sizes):

```text
data/danbooru2025_lite_train.parquet   # ~10K images, 50 tags
data/danbooru2025_lite_val.parquet

data/danbooru2025_train.parquet        # ~1M images
data/danbooru2025_val.parquet
```

Images do not need to be present to create the tag vocabulary.

---

## Create Vocabulary

All unique tags used as classes, sorted by descending frequency (ties broken alphabetically).

**Full dataset (default):**

```bash
uv run python -m src.create_tag_vocab \
  data/danbooru2025_train.parquet \
  data/danbooru2025_val.parquet \
  --output data/tag_to_id.json
```

**Lite dataset:**

```bash
uv run python -m src.create_tag_vocab \
  data/danbooru2025_lite_train.parquet \
  data/danbooru2025_lite_val.parquet \
  --output data/tag_to_id_lite.json
```

---

## Download Images

Download images and add an `image_path` column to each Parquet file:

```bash
uv run python -m src.download_images \
  data/danbooru2025_lite_train.parquet \
  data/danbooru2025_lite_val.parquet
```

Images stored in a directory derived from the Parquet filename:

```text
data/images/
├── danbooru2025_lite_train/
│   └── 6301081.jpg
└── danbooru2025_lite_val/
    └── 6223583.jpg
```

Downloader: 32 concurrent workers, skips existing files, retries transient failures, logs failures to `failed_downloads.csv`. Configure with `--workers` and `--retries`.

---

## Train

After downloading images, start training:

```bash
uv run python -m src.train
```

Default: 448px images, batch size 48, 20 epochs, AdamW, cosine decay + 2 warmup epochs, BCE-with-logits loss, BF16 AMP on CUDA.

Checkpoints written to `checkpoints/best.pt` and `checkpoints/last.pt`.

**Smoke test (1 epoch):**

```bash
uv run python -m src.train --epochs 1
```

Training metrics appended to `runs/metrics.jsonl`.

### Config files & experiments

Per-experiment Python configs in `src/configs/` (each exports a `TrainConfig` instance). Load with `--config`:

```bash
uv run python -m src.train --config src/configs/config_dinov3_l16_fromscratch_fulldata.py
```

| Config | Purpose |
|---|---|
| `config_dinov3_l16_fromscratch_fulldata.py` | Reference: DINOv3 L/16 from scratch, full dataset (mirrors defaults) |
| `config_dinov3_l16_to_b16_transfer.py` | Transfer: B/16 student reusing the trained L/16 head via a projector |
| `config_dinov3_b16_kd.py` | Knowledge distillation: B/16 student trained against the frozen L/16 teacher |
| `config_dinov3_l16_to_s16_transfer_kd_litedata.py` | Transfer + KD on lite data: S/16 student reusing L/16 head AND distilling from frozen L/16 teacher |

CLI flags (`--epochs`, `--batch-size`, `--checkpoint`, `--teacher-path`, `--kd-weight`) override config file fields. Paths inside config files are relative to repo root.

**Vocab note:** transfer remaps vocabulary by tag name (different tag counts fine — matched tags transfer, student-only tags start random). KD aligns teacher logits to student vocab for tags teacher knows: student vocab must be a subset of teacher's (trained L/16 teacher covers 11,424 of 11,516 tags in `data/tag_to_id.json`). KD configs point `tag_to_id` at `data/tag_to_id_l16_teacher.json` (from checkpoint) or a lite vocab.

### Transfer learning (scaling down)

Reuse the trained L/16 head on a smaller backbone. When checkpoint's head embed dim differs from backbone dim, a trainable projector (`nn.Linear`) inserted automatically; head built at teacher's dim so it transfers 1:1 (tag queries remap by vocab index):

```bash
uv run python -m src.train --config src/configs/config_dinov3_l16_to_b16_transfer.py
```

Per-run controls (config fields, overridable via CLI):

- `checkpoint` — transfer source `.pt` (CLI `--checkpoint`)
- `head_lr_mult` — LR multiplier for transferred head (0.01: keep teacher init)
- `proj_lr_mult` — LR multiplier for random projector (1.0: new layer must adapt)
- `backbone_lr_mult` — LR multiplier for pretrained backbone (0.1: standard finetune)

Projector participates in optimizer (separate param group), scheduler, and EMA. Checkpoint `config` records `projector` ("in:out") so resume, ONNX export, and inference rebuild correctly.

### Knowledge distillation

Logits distillation from frozen teacher (`.pt` or ONNX dir). Loss per batch: `BCE(gt, student) + kd_weight * KL(log_softmax(student), log_softmax(teacher))` (softmax KL, temperature 1):

```bash
uv run python -m src.train --config src/configs/config_dinov3_b16_kd.py
# or explicitly:
uv run python -m src.train --config ... --teacher-path models/dtq_dinov3l16_448x448_ep13_bestmAP.pt --kd-weight 0.5
```

- `--teacher-path` — teacher `.pt` checkpoint (built from its own saved config, frozen, eval) or ONNX model dir/file (session I/O names resolved automatically).
- `--kd-weight` — default 0.5; requires teacher source.
- Teacher tag vocabulary must match student's exactly.
- Validation loss stays pure BCE; best-checkpoint selection unchanged.

### Multi-GPU Training (DDP)

Use `torchrun` to spawn one process per GPU. Script auto-detects `LOCAL_RANK` and sets up NCCL-based DDP:

```bash
torchrun --standalone --nproc_per_node=4 \
  -m src.train --epochs 30 --batch-size 32
```

**Key details:**

- `--batch-size` is **per-GPU**. Total effective batch = `batch-size × num-gpus`. Example: 4x4090 with `--batch-size 32` = 128 total.
- `torch.compile` switches from `max-autotune` to `default` mode under DDP for faster warmup.
- EMA lives on rank 0 only — gradients synced, weights identical across ranks.
- Validation loss reduced via `all_reduce`; predictions/targets gathered to rank 0 for metric computation.
- **Resume requires same GPU count** — checkpoint stores `world_size` and rejects mismatch.
- Single-GPU fallback: `uv run python -m src.train` (no `torchrun`).

```bash
# 2x4090 — effective batch 64
torchrun --standalone --nproc_per_node=2 \
  -m src.train --batch-size 32

# Resume (same GPU count)
torchrun --standalone --nproc_per_node=4 \
  -m src.train --resume runs/20250101_120000/checkpoints/last.pt
```

---

## Evaluate

Evaluate a checkpoint on the validation set:

```bash
uv run python -m src.evaluate checkpoints/best.pt
```

Reported: mAP, macro F1, micro F1, per-tag average precision.

---

## Inference

**CLI — single image:**

```bash
uv run python -m src.infer \
  data/images/danbooru2025_lite_val/6223583.jpg \
  checkpoints/best.pt \
  --top-k 10
```

**Web UI** (via standalone `danbooru-deploy` package; `src.infer` is CLI-only):

```bash
uv run python deploy/main.py --cpu          # CPU; omit --cpu → CUDA-if-available
```

`deploy/main.py` resolves model from CLI arg → `MODEL_DIR` env → HF hub default (auto-downloads). See [`deploy/README.md`](deploy/README.md) for the `danbooru-deploy` pip package entry point.

---

## ONNX Export

Convert a PyTorch checkpoint to ONNX for faster CPU inference or deployment:

```bash
uv run python -m src.convert_onnx checkpoints/best.pt -o models/model
```

Produces:

| File | Contents |
|---|---|
| `models/model.onnx` | Exported ONNX model |
| `models/model.tag_to_id.json` | Tag name → index mapping |
| `models/model.config.json` | Export-time config (image size, model name, ...) |

**ONNX inference — same CLI commands, pass `.onnx` file:**

```bash
uv run python -m src.infer image.jpg models/model.onnx --top-k 10
uv run python -m src.evaluate models/model.onnx
```

CLI auto-detects file extension: `onnxruntime` for ONNX, PyTorch for `.pt`. No code changes.

---

## Validation Visualization

Create a grid of validation images with ground-truth and predicted tags:

```bash
uv run python -m src.visualize \
  checkpoints/best.pt \
  --output runs/validation_examples.png
```

---

## Model Explorer

Interactive research tool: patch↔tag inspector + infinite semantic graph.

```bash
uv run python -m tools.explorer models/dtq_dinov3b16_448x448_ft_ep9bestmap.pt
```

Open **http://127.0.0.1:8000**. Requires PyTorch checkpoint (`.pt`). Full docs in [`EXPLORER.md`](EXPLORER.md).
