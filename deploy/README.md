# Danbooru-deploy

Standalone ONNX deployment for DanbooruTagQuery.

- NumPy + PIL only predictor (no PyTorch/torchvision).
- CUDA accelerator preferred when available, CPU fallback automatic.
- Force CPU with `use_cuda=False` (used by the HF space).
- Gradio UI + CLI.

## Install

```bash
pip install git+https://github.com/realphongha/DanbooruTagQuery.git#subdirectory=deploy
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