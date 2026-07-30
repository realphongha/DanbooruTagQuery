# DanbooruTagQuery

Lightweight multi-label anime image tagger using a pretrained ViT backbone and a linear classification head.

## Setup

Install the project dependencies with:

```bash
uv sync
```

Parquet files contain the `tags` column with image labels. The dataset comes in two sizes:

```text
data/danbooru2025_lite_train.parquet   # ~10K images, 50 tags
data/danbooru2025_lite_val.parquet

data/danbooru2025_train.parquet        # ~1M images
data/danbooru2025_val.parquet
```

Images do not need to be present to create the tag vocabulary.

## Create Vocabulary

All unique tags are used as classes, sorted by descending frequency (ties
broken alphabetically so runs are deterministic).

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

## Download Images

Download images and add an `image_path` column to each Parquet file:

```bash
uv run python -m src.download_images \
  data/danbooru2025_lite_train.parquet \
  data/danbooru2025_lite_val.parquet
```

Images are stored in a directory derived from the Parquet filename:

```text
data/images/
├── danbooru2025_lite_train/
│   └── 6301081.jpg
└── danbooru2025_lite_val/
    └── 6223583.jpg
```

The downloader uses up to 32 concurrent workers by default, skips existing files, retries transient failures, and records failures in `failed_downloads.csv`. Configure concurrency and retries with `--workers` and `--retries`.

## Train

After downloading the images, start training with:

```bash
uv run python -m src.train
```

The default configuration uses 256px images, batch size 128, 30 epochs, AdamW, cosine decay with five warmup epochs, BCE-with-logits loss, and BF16 automatic mixed precision on CUDA.

Checkpoints are written to:

```text
checkpoints/best.pt
checkpoints/last.pt
```

Run a shorter smoke training session with:

```bash
uv run python -m src.train --epochs 1
```

Training metrics are appended to `runs/metrics.jsonl`.

### Multi-GPU Training (DDP)

Use `torchrun` to spawn one process per GPU. The script auto-detects `LOCAL_RANK` and sets up NCCL-based DistributedDataParallel:

```bash
torchrun --standalone --nproc_per_node=4 \
  -m src.train --epochs 30 --batch-size 32
```

**Key details:**

- `--batch-size` is **per-GPU**. Total effective batch = `batch-size × num-gpus`. Example: 4x4090 with `--batch-size 32` gives 128 total.
- `torch.compile` switches from `max-autotune` to `default` mode under DDP for faster warmup.
- EMA lives on rank 0 only — gradients are synced, so weights stay identical across ranks.
- Validation loss is reduced via `all_reduce`; predictions/targets gathered to rank 0 for metric computation.
- **Resume requires same GPU count** — checkpoint stores `world_size` and rejects mismatch.
- Single-GPU fallback: just run `uv run python -m src.train` (no `torchrun`).

```bash
# 2x4090 example — effective batch 64
torchrun --standalone --nproc_per_node=2 \
  -m src.train --batch-size 32

# Resume (must use same GPU count)
torchrun --standalone --nproc_per_node=4 \
  -m src.train --resume runs/20250101_120000/checkpoints/last.pt
```

## Evaluate

Evaluate a checkpoint on the validation set:

```bash
uv run python -m src.evaluate checkpoints/best.pt
```

Reported metrics include mAP, macro F1, micro F1, and per-tag average precision.

## Inference

Predict the top tags for one image:

```bash
uv run python -m src.infer \
  data/images/danbooru2025_lite_val/6223583.jpg \
  checkpoints/best.pt \
  --top-k 10
```

### Web UI

Launch a Gradio web interface:

```bash
uv run python -m src.infer checkpoints/best.pt --ui
```

## ONNX Export

Convert a PyTorch checkpoint to ONNX for faster CPU inference or deployment:

```bash
uv run python -m src.convert_onnx checkpoints/best.pt -o models/model
```

This produces three files:

| File | Contents |
|---|---|
| `models/model.onnx` | Exported ONNX model |
| `models/model.tag_to_id.json` | Tag name → index mapping |
| `models/model.config.json` | Export-time config (image size, model name, …) |

### ONNX Inference

Use the ONNX model with the same CLI commands — just pass the `.onnx` file:

```bash
uv run python -m src.infer image.jpg models/model.onnx --top-k 10
```

```bash
uv run python -m src.evaluate models/model.onnx
```

```bash
uv run python -m src.infer models/model.onnx --ui
```

The CLI auto-detects the file extension and uses `onnxruntime` for ONNX files or PyTorch for `.pt` files. No code changes needed.

## Validation Visualization

Create a grid of validation images with ground-truth and predicted tags:

```bash
uv run python -m src.visualize \
  checkpoints/best.pt \
  --output runs/validation_examples.png
```

## Model Explorer

Interactive research tool for understanding how the model represents visual concepts and makes predictions.

- **Patch ↔ Tag Inspector** — click image patches to see which tags fire, click tags to see which image regions they attend to
- **Infinite Semantic Graph** — explore learned tag relationships through an expandable nearest-neighbour graph

```bash
uv run python -m src.explorer models/dtq_dinov3b16_448x448_ft_ep9bestmap.pt
```

Open **http://127.0.0.1:8000** in your browser. Requires a PyTorch checkpoint (`.pt`). Full documentation in [`EXPLORER.md`](EXPLORER.md).

## Project Structure

```text
src/
├── config.py          # Hyperparameters and paths
├── create_tag_vocab.py
├── dataset.py
├── download_images.py
├── evaluate.py
├── infer.py
├── losses.py
├── metrics.py
├── model.py
├── train.py
├── transforms.py
├── utils.py
└── visualize.py
```
