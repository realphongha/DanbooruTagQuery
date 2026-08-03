# Danbooru-deploy

Standalone ONNX deployment for DanbooruTagQuery.

- NumPy + PIL only predictor (no PyTorch/torchvision).
- CUDA accelerator preferred when available, CPU fallback automatic.
- Force CPU with `use_cuda=False` (used by the HF space).
- Optional HF-hub auto-download (`[hf]` extra) + runtime model switcher.
- Gradio UI + CLI.

## Install

Local/offline (no hub):

```bash
pip install git+https://github.com/realphongha/DanbooruTagQuery.git#subdirectory=deploy
```

With HF-hub discovery/download:

```bash
pip install "git+https://github.com/realphongha/DanbooruTagQuery.git#subdirectory=deploy[hf]"
```

For GPU (onnxruntime-gpu):

```bash
pip install "git+https://github.com/realphongha/DanbooruTagQuery.git#subdirectory=deploy[gpu]"
```

## Model layout

Expects a model directory:

```
model_dir/
  model.onnx
  config.json
  tag_to_id.json
  tag_category.json
```

Or pass the `model.onnx` file path directly.

## Quick start

Run straight from the repo (no pip install of the package needed):

```bash
uv run python deploy/main.py --cpu     # CPU; omit --cpu → CUDA-if-available
```

Examples:

```bash
python deploy/main.py --cpu                      # hub default model, web UI, CPU
python deploy/main.py path/to/model_dir         # local model dir, auto-CUDA UI
python deploy/main.py path/to/model_dir img.jpg # tag a single image
```

Or pip-install and use the `danbooru-deploy` console script (identical CLI):

```bash
pip install "git+https://github.com/realphongha/DanbooruTagQuery.git#subdirectory=deploy[hf]"
danbooru-deploy --cpu
```

Hub auto-download (no model arg) needs `huggingface_hub` installed — `uv add huggingface_hub` or the `[hf]` extra. Local models need nothing extra.

## Usage

```python
from danbooru_deploy import Predictor

p = Predictor("model_dir")                # CUDA if available
p = Predictor("model_dir", use_cuda=False)  # force CPU

scores = p.run(preprocess(image, p.image_size))
```

Or run the Gradio UI / CLI:

```bash
danbooru-deploy model_dir
danbooru-deploy model_dir image.jpg --top-k 20 --min-score 0.2
```

## Programmatic / HF Space

`main` is the single orchestration entry point (models resolved from CLI arg,
then `MODEL_DIR` env, then HF hub), and is what the Hugging Face Space calls:

```python
from danbooru_deploy import main

app = main(
    use_cuda=False,      # CPU-forced (HF Space has no GPU)
    host=...,            # or rely on GRADIO_SERVER_NAME/HOST env
    port=...,            # or rely on GRADIO_SERVER_PORT/PORT env
)
```

Use `use_cuda="auto"` (default) to prefer CUDA with automatic CPU fallback.