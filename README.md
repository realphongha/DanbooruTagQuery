# DanbooruTagQuery

Lightweight multi-label anime image tagger using a pretrained DINOv3 ViT backbone and a cross-attention tag query head.

---

## Architecture

**Tag Query Head** — each tag is a learned query vector that cross-attends to ViT patch tokens to produce classification logits. Modular design: the head is the sole component swapped between experiments (linear prototype → query decoder → future multi-block decoder).

**DDP Training** — NCCL-based multi-GPU with per-GPU batching, automatic `torch.compile` mode switch (`max-autotune` → `default`), EMA on rank 0 only, and `all_reduce` validation. Resume requires matching GPU count.

**Vocabulary** — all unique tags from the dataset, sorted by descending frequency with deterministic alphabetical tie-breaking.

**Transfer Learning** — when teacher/student backbone dims differ, a trainable projector (`nn.Linear`) is inserted automatically. Vocabulary remapped by tag name — mismatched tag counts work (matched tags transfer, student-only tags start random).

---

## Features

| Feature | Summary |
|---|---|
| **Data pipeline** | Parquet → vocab creation → concurrent image download → train → eval, fully modular |
| **Knowledge distillation** | BCE + KL loss with frozen teacher (`.pt` or ONNX); validation stays pure BCE for checkpoint selection |
| **ONNX export** | Full export with tag vocab and config; CLI auto-selects PyTorch or ONNX runtime by file extension |
| **Model Explorer** | Patch↔tag cross-attention inspector + infinite semantic graph via nearest-neighbour expansion — see [`EXPLORER.md`](EXPLORER.md) |

---

## Quick Links

- [Tutorial](TUTORIAL.md) — step-by-step setup, training, inference commands
- [Evaluation](EVALUATION.md) — metrics, comparison vs baseline models
- [Model Explorer](EXPLORER.md) — interactive visualization docs
- [Deploy](deploy/README.md) — `danbooru-deploy` Web UI package

---

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
tools/
├── explorer.py        # Model Explorer (FastAPI + HTML/JS)
```
