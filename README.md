# DanbooruTagQuery

Lightweight multi-label anime image tagger using a pretrained DINOv3 ViT backbone and a cross-attention tag query head.

---

## Architecture

Image → ViT backbone (DINOv3) → patch tokens → cross-attention → per-tag logits.

```
Input image (448×448)
    │
    ▼
┌─────────────────────┐
│        DINOv3       │  pretrained ViT backbone│
└─────────┬───────────┘
          │  tokens: (B, N_patches+5, D)
          ▼
┌─────────────────────┐
│   Tag Query Head    │
│                     │
│  tag_queries:       │  learned (num_tags, D) — each tag = one query vector
│  (num_tags, D) ─────┼─→ cross-attention ──→ tag features (B, num_tags, D)
│                     │     queries attend to ViT patch tokens
│  classifier:        │
│  Linear(D→1) ───────┘  → logits (B, num_tags)
└─────────────────────┘
          │
          ▼
  sigmoid(logits) → per-tag probabilities
```

**Cross-attention tag query mechanism** — each tag is a single learnable embedding vector (a "query"). The `nn.MultiheadAttention` layer computes attention between all tag queries and all ViT patch tokens. Output per tag: the query embedding attended over spatial features, projected through a scalar linear classifier. No MLP, no positional encoding — the query itself encodes "what to look for". This is modular: swap the head for a multi-block decoder without touching dataset, training loop, or metrics.

**Modular backbone → head boundary** — `ImageTagger` separates `backbone.forward_features()` from `head(tokens)`. The head is the sole component changed between experiments (linear prototype → cross-attention query → future multi-block decoder).

---

## Dataset Creation

Source: [trojblue/danbooru2025-metadata](https://huggingface.co/datasets/trojblue/danbooru2025-metadata) on HuggingFace.

**1. Clean** (`clean_data.ipynb`) — download the 20 GB metadata dataset, filter deleted/banned/flagged/pending images, enforce score ≥ 10, file size 50 KB–20 MB, dimensions ≥ 224×224. Extract the 360×360 variant URL. Output: `data/danbooru2025_cleaned.parquet`.

**2. Remove noise tags** (`clean_ignored_tags.ipynb`) — strip tags listed in `data/ignored_tags.txt` (e.g. `real_life`, `text`, metadata artifacts). Re-saves cleaned train/val parquet files in-place.

**3. Full set** (`get_full_set.ipynb`) — filter cleaned data (score > 10), combine `tag_string_general` + `tag_string_character` + `tag_string_copyright` into a unified `tags` list. Keep tags with frequency ≥ 100 occurrences. Keep images with ≥ 2 tags. Train/val split: last 100K images as validation. Output: `data/danbooru2025_train.parquet` + `data/danbooru2025_val.parquet`.

**4. Lite set** (`get_lite_set.ipynb`) — same as full set but with extra filters: score > 50, rating = `g` (safe only). Keep only top-50 most frequent tags. Sample 20K images (16K train / 4K val). Output: `data/danbooru2025_lite_train.parquet` + `data/danbooru2025_lite_val.parquet`.

**5. Test set** (`get_test_set_2024.ipynb`) — extract images from the val set with post ID > 7220105 to avoid contamination with WD-SwinV2 training data. Sample 5K, filter by active tags. Output: `data/danbooru_after2024_testset.parquet`.

---

## Features

| Feature | Summary |
|---|---|
| **Data pipeline** | Parquet → vocab creation → concurrent image download → train → eval, fully modular |
| **Knowledge distillation** | BCE + KL loss with frozen teacher (`.pt` or ONNX); validation stays pure BCE for checkpoint selection |
| **ONNX export** | Full export with tag vocab and config; CLI auto-selects PyTorch or ONNX runtime by file extension |
| **Model Explorer** | Patch↔tag cross-attention inspector + infinite semantic graph via nearest-neighbour expansion — see [`EXPLORER.md`](EXPLORER.md) |

---

## Experimental: Knowledge Distillation & Transfer Learning

Scale a trained L/16 model down to smaller backbones (B/16, S/16) while retaining tagging quality.

**Transfer learning** — reuse the trained head on a smaller backbone. When teacher/student backbone dims differ, a trainable projector (`nn.Linear`) is inserted automatically. Vocabulary remapped by tag name — mismatched tag counts work (matched tags transfer, student-only tags start random). Separate LR multipliers for backbone, head, and projector.

**Knowledge distillation** — train a student against a frozen teacher (`.pt` or ONNX). Loss: `BCE(gt, student) + kd_weight × KL(student, teacher)`. Teacher can be PyTorch (built from saved config) or ONNX (auto-resolved I/O).

See [Tutorial → Train](TUTORIAL.md#train) for commands and [configs/](src/configs/) for preset experiments.

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
