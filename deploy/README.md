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

Auto-download default model from HF hub and launch the web UI (CPU):

```bash
pip install "git+...danbooru-deploy"
danbooru-deploy --cpu
```

Run the web UI against a local model (CUDA if available):

```bash
danbooru-deploy path/to/model_dir
```

Force CPU for the UI:

```bash
danbooru-deploy --cpu path/to/model_dir
```

Tag a single image, print top-20:

```bash
danbooru-deploy path/to/model_dir image.jpg
```

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