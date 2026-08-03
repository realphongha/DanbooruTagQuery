# Evaluation

## Metrics

| Metric | Definition |
|---|---|
| **mAP** | Mean Average Precision across all tags. AP per tag = area under the precision-recall curve. Final score = `nanmean` (tags with zero ground-truth positives excluded). Computed via batched CUDA sort. Primary ranking metric. |
| **Macro F1** | Unweighted mean of per-tag F1 scores. Threshold searched over `[0.10, 0.15, …, 0.95]` (step 0.05); reported value uses the threshold that maximizes Macro F1. |
| **Micro F1** | Global F1 from aggregated TP/FP/FN across all tags. Reported at the same best-Macro-F1 threshold — not independently optimized. |
| **Thresh** | The threshold from the search grid that maximizes Macro F1. Both Macro F1 and Micro F1 are computed at this single value. |

## Comparison Methodology
We compared our model to other popular methods from the community.
All models evaluated on the same **custom intersection evaluation set**: 3383 tags that appear in every model's vocabulary. Models with larger or smaller vocabularies are reduced to this common subset so scores are directly comparable.

- **Input resolution** listed per model — where models differ (DeepDanbooru at 512×512 vs 448×448), we use each model's native resolution.
- **Latency** measured on a single NVIDIA RTX 5090 (PyTorch eager, batch size 1, CUDA timing). ONNX runtime used for ONNX-exported models.
- **Best checkpoint** selected by validation mAP during training.
- **Threshold** searched over `[0.10, 0.15, …, 0.95]` (step 0.05). Best value maximizes Macro F1. Micro F1 reported at the same threshold — not independently optimized.

## Results

### Comparison Table — Intersection (3383 tags)

| Model | Params | Input | Latency | mAP | Macro F1 | Micro F1 | Thresh |
|---|---|---|---|---|---|---|---|
| **Ours (L/16)** | 319.0M | 448×448 | 36.6ms | **0.5352** | **0.4775** | **0.6884** | 0.20 |
| **Ours (B/16)** | 96.8M | 448×448 | 24.9ms | 0.4693 | 0.4195 | 0.6684 | 0.20 |
| DeepDanbooru | 161.0M | 512×512 | 33.6ms | 0.2100 | 0.1920 | 0.4692 | 0.15 |
| WD-eva02 | 315.2M | 448×448 | 50.3ms | 0.4822 | 0.4344 | 0.6684 | 0.30 |
| WD-SwinV2 | 98.0M | 448×448 | 35.8ms | 0.4603 | 0.4140 | 0.6474 | 0.15 |
| ML-Danbooru | 68.9M | 448×448 | 34.0ms | 0.4023 | 0.3490 | 0.5952 | 0.60 |
| JoyTag | 91.5M | 448×448 | 20.2ms | 0.3783 | 0.3429 | 0.6179 | 0.35 |

### Key Results

- **Ours (L/16)** leads all baselines by **+0.053 mAP** over WD-eva02 (the strongest comparable model) with 33% faster inference (36.6ms vs 50.3ms).
- **Ours (B/16)** outperforms all models in the <100M class (WD-SwinV2 +0.009 mAP, JoyTag +0.091 mAP) at 24.9ms latency.

### Baseline Models

All baseline implementations live in [`src/compared_models/`](src/compared_models/). Each implements the `BaseModel` interface (`name`, `tag_names`, `load()`, `predict()`).

| Model | Architecture | Implementation | Source |
|---|---|---|---|
| DeepDanbooru | ResNet (custom variant, TorchDeepDanbooru port) | [`deepdanbooru.py`](src/compared_models/deepdanbooru.py) — custom PyTorch model def | [AUTOMATIC1111/TorchDeepDanbooru](https://github.com/AUTOMATIC1111/TorchDeepDanbooru) (Torch implementation of [KichangKim/DeepDanbooru](https://github.com/KichangKim/DeepDanbooru/)) |
| WD-eva02 | EVA-02 Large ViT + MLP head (via timm) | [`wd_tagger.py`](src/compared_models/wd_tagger.py) — `WDTagger("eva02-large")` | [SmilingWolf/wd-eva02-large-tagger-v3](https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3) |
| WD-SwinV2 | SwinV2-Tiny + MLP head (via timm) | [`wd_tagger.py`](src/compared_models/wd_tagger.py) — `WDTagger("swinv2")` | [SmilingWolf/wd-swinv2-tagger-v3](https://huggingface.co/SmilingWolf/wd-swinv2-tagger-v3) |
| ML-Danbooru | CAFormer-M36 + ONNX Runtime | [`ml_danbooru.py`](src/compared_models/ml_danbooru.py) — pure ONNX inference | [deepghs/ml-danbooru-onnx](https://huggingface.co/deepghs/ml-danbooru-onnx) (ONNX implementation of [7eu7d7/ML-Danbooru](https://huggingface.co/7eu7d7/ML-Danbooru)) |
| JoyTag | ViT-B/16 + MLP head | [`joytag.py`](src/compared_models/joytag.py) — original `Models.py` via `VisionModel` | [fancyfeast/joytag](https://huggingface.co/fancyfeast/joytag) |
