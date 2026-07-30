# DTQ Model Explorer V0.1

Interactive research tool for understanding how the image tagging model represents visual concepts. FastAPI backend + HTML/JS frontend.

## Quick Start

```bash
uv run python -m src.explorer models/dtq_dinov3b16_448x448_ft_ep9bestmap.pt
```

Open **http://127.0.0.1:8000** in your browser.

**Requires:** PyTorch checkpoint (`.pt`). ONNX not supported — intermediate tensors aren't available in the exported format.

---

## Architecture

```
Browser (HTML/JS)                Python (FastAPI)
─────────────────                ────────────────
  Upload image ──POST /infer──→  run_with_intermediates()
                                   ↓
  Canvas click ──POST /patch-tags → pixel_to_patch() → top-30 tags
                                   ↓
  Tag click ────POST /tag-info──→  attention heatmap + neighbours
                                   ↓
  Graph click ──GET /neighbors/{tag} → precomputed kNN
                                   ↓
  Search ──────GET /tag-names?q=  → fuzzy tag lookup
```

### Token layout

The DINOv2 backbone outputs 789 tokens per image:

| Index | Content |
|---|---|
| 0 | CLS token (global image representation) |
| 1–784 | Image patch tokens (28×28 grid, 16 px each) |
| 785–788 | Register tokens (ignored in visualisation) |

---

## Tab 1 — Patch ↔ Tag Inspector

### Patch → Tags

1. **Upload an image.** Inference runs automatically.
2. **Click any patch** on the 28×28 grid overlay.
3. Right panel shows the top 30 tags ranked by **cross-attention weight** from that tag query to that patch token.

### Tag → Patches

1. **Click a tag row** in the list.
2. Bottom panel shows:
   - **3-panel attention heatmap** (image / overlay / grid)
   - **Tag details**: score, category, top-5 patches, 15 nearest neighbours

### Connected navigation

```
Image → click patch → see tags → click tag → heatmap + neighbours
                                              → "Show in Graph"
```

---

## Tab 2 — Semantic Graph

vis-network interactive graph of learned tag relationships.

- **Nodes** = learned tag queries. Size ≈ prediction score. Colour = category (blue=general, purple=copyright, red=character).
- **Edges** = cosine similarity between normalised query embeddings. Thickness = similarity strength.
- **Click any node** → 20 nearest neighbours are added (expand indefinitely).
- **Search box** → type tag name + Enter → add tag + 20 neighbours to graph.
- **Physics toggle** → freeze layout for precise dragging.
- **Clear** → reset graph.

### Cross-tab

In the Inspector tab, select a tag and click **📊 Show in Graph** — the tag and 15 neighbours appear in the graph.

---

## Interpreting the Attention Heatmap

| Panel | What to look for |
|---|---|
| Image | The input as the model sees it (448×448, padded to square) |
| Overlay | Red/yellow regions = high attention from this tag query |
| Grid | Raw 28×28 attention matrix for spatial pattern analysis |

A well-learned tag should show high attention in semantically relevant regions:
- `smile` → mouth area
- `blue_eyes` → eye region
- `thighs` → lower body
- `simple_background` → edges or uniform distribution

Uniform or random-looking attention suggests the tag is poorly learned or unrelated to the image content.

---

## API Endpoints

| Endpoint | Method | Input | Returns |
|---|---|---|---|
| `/` | GET | — | Single-page HTML app |
| `/infer` | POST | `file` (multipart) | Base64 image + top-10 tags |
| `/patch-tags` | POST | `x`, `y` (form) | Patch index + top-30 tags |
| `/tag-info` | POST | `tag` (form) | Heatmap base64 + neighbours + metadata |
| `/neighbors/{tag}` | GET | `?limit=N` | Neighbour list for graph expansion |
| `/tag-names` | GET | `?query=...` | Matching tag names for autocomplete |

---

## Caveats

- **PyTorch only.** ONNX checkpoints don't expose intermediate tensors.
- **ViT patch size hardcoded at 16.** Currently works with DINOv2-based backbones only.
- **vis-network loaded from CDN.** Requires internet on first load. Cached by browser afterwards.
- **Single-user session.** The server keeps one inference result in memory. Uploading a new image replaces it.

## File

| File | Purpose |
|---|---|
| `src/explorer.py` | FastAPI app, inference wrapper, inline HTML/JS frontend |
