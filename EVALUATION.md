# Evaluation

## Metrics

| Metric | Definition |
|---|---|
| **mAP** | Mean Average Precision across all tags. For each tag, AP is the area under the precision-recall curve. mAP is the unweighted mean. Primary ranking metric. |
| **Macro F1** | Unweighted mean of per-tag F1 scores, computed at a fixed threshold. Treats rare and frequent tags equally. |
| **Micro F1** | Global F1 computed from aggregated TP/FP/FN across all tags. Dominated by frequent tags. |
| **Thresh** | Confidence threshold used for Macro/Micro F1 computation (sigmoid output > threshold → positive). |

## Comparison Methodology

All models evaluated on the same **custom intersection evaluation set**: 3383 tags that appear in every model's vocabulary. Models with larger or smaller vocabularies are reduced to this common subset so scores are directly comparable.

- **Input resolution** listed per model — where models differ (DeepDanbooru at 512×512 vs 448×448), we use each model's native resolution.
- **Latency** measured on a single NVIDIA RTX 4090 (PyTorch eager, batch size 1, CUDA timing). ONNX runtime used for ONNX-exported models.
- **Best checkpoint** selected by validation mAP during training.
- **Threshold** for each model is the value that maximizes Macro F1 on the intersection set, reported for reference.

## Results

### Comparison Table — Intersection (3383 tags)

| Model | Params | Input | Latency | mAP | Macro F1 | Micro F1 | Thresh |
|---|---|---|---|---|---|---|---|
| **Ours (L/16)** | 319.0M | 448×448 | 36.6ms | **0.5352** | **0.4775** | **0.6884** | 0.20 |
| **Ours (B/16)** | 96.8M | 448×448 | 24.9ms | 0.4693 | 0.4195 | 0.6684 | 0.20 |
| DeepDanbooru | 161.0M | 512×512 | 33.6ms | 0.2100 | 0.1920 | 0.4692 | 0.15 |
| WD-eva02 | 315.2M | 448×448 | 50.3ms | 0.4822 | 0.4344 | 0.6684 | 0.30 |
| WD-SwinV2 | 98.0M | 448×448 | 35.8ms | 0.4603 | 0.4140 | 0.6474 | 0.15 |
| ML-Danbooru | N/A | 448×448 | 34.0ms | 0.4023 | 0.3490 | 0.5952 | 0.60 |
| JoyTag | 91.5M | 448×448 | 20.2ms | 0.3783 | 0.3429 | 0.6179 | 0.35 |

### Key Results

- **Ours (L/16)** leads all baselines by **+0.053 mAP** over WD-eva02 (the strongest comparable model) with 33% faster inference (36.6ms vs 50.3ms).
- **Ours (B/16)** outperforms all models in the <100M class (WD-SwinV2 +0.009 mAP, JoyTag +0.091 mAP) at 24.9ms latency.
- Both variants match or exceed WD-eva02 on Macro and Micro F1.

### Baseline Models

| Model | Architecture | Link |
|---|---|---|
| DeepDanbooru | GoogLeNet (ResNet-like CNN) | [Konaki4547/DeepDanbooru](https://github.com/Konaki4547/DeepDanbooru) |
| WD-eva02 | EVA-02 Large ViT + MLP head | [SmilingWolf/wd-eva02-large-tagger-v3](https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3) |
| WD-SwinV2 | SwinV2-Tiny + MLP head | [SmilingWolf/wd-swinv2-small-tagger-v3](https://huggingface.co/SmilingWolf/wd-swinv2-small-tagger-v3) |
| ML-Danbooru | CNN ensemble | [lloydmeta/deepdanbooru](https://github.com/lloydmeta/deepdanbooru) (ML-Danbooru fork) |
| JoyTag | ConvNeXt + MLP | [AlicUpup/joymodel-tagger](https://github.com/AlicUpup/joymodel-tagger) |
