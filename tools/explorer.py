"""
DTQ Model Explorer V0.1 — FastAPI backend + HTML/JS frontend.

Usage
-----
    uv run python -m tools.explorer models/dtq_dinov3b16_448x448_ft_ep9bestmap.pt
    # Open http://127.0.0.1:8000
"""

import argparse
import base64
import json
import io
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.model import ImageTagger
from src.transforms import val_transforms

# ── constants ───────────────────────────────────────────────────────────────

CATEGORY_MAP = {0: "general", 3: "copyright", 4: "character"}
CATEGORY_COLORS = {"general": "#4a90d9", "copyright": "#7b61ff", "character": "#e8522e", "unknown": "#999999"}
PATCH_SIZE = 16
_TOKEN_BREAKDOWN = "DINOv2: [CLS(0), patch(1..784), register(785..788)]"


# ═════════════════════════════════════════════════════════════════════════════
#  Inference engine
# ═════════════════════════════════════════════════════════════════════════════

class ExplorerInference:
    """Loads .pt checkpoint, runs inference, captures intermediates."""

    def __init__(self, checkpoint: str | Path, device: str | None = None):
        self.checkpoint = str(checkpoint)
        if str(checkpoint).endswith(".onnx"):
            raise ValueError("Explorer needs PyTorch (.pt), not ONNX.")

        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        state = torch.load(self.checkpoint, map_location="cpu", weights_only=False)

        self.tag_to_id: dict[str, int] = state["tag_to_id"]
        self.id_to_tag: dict[int, str] = {v: k for k, v in self.tag_to_id.items()}
        self.num_classes = len(self.tag_to_id)

        cfg = state.get("config", {})
        self.image_size = cfg.get("image_size", 448)
        self.model_name = cfg.get("model_name", "vit_base_patch16_dinov3.lvd1689m")
        self.model = ImageTagger(
            self.model_name, self.num_classes,
            pretrained=False,
        )
        weights = state.get("model_ema") or state.get("model")
        self.model.load_state_dict(weights)
        self.model.to(self._device).eval()

        self._precompute_tag_similarity(top_k=50)
        self._load_categories()
        self.val_transform = val_transforms(self.image_size)
        self.num_image_patches = (self.image_size // PATCH_SIZE) ** 2

    def _load_categories(self):
        p = Path("models/tag_category.json")
        self.tag_category = json.loads(p.read_text()) if p.exists() else {}

    def _precompute_tag_similarity(self, top_k: int = 50):
        with torch.no_grad():
            q = self.model.head.tag_queries.detach().cpu()
            q_norm = F.normalize(q, dim=1)
            sim = q_norm @ q_norm.T
            topk = sim.topk(top_k + 1, dim=1, largest=True)
            self.tag_neighbors = topk.indices[:, 1:].clone()
            self.tag_similarities = topk.values[:, 1:].clone()

    def get_tag_similarity(self, tag_idx: int) -> list[tuple[str, float]]:
        nbrs, sims = self.tag_neighbors[tag_idx], self.tag_similarities[tag_idx]
        return [(self.id_to_tag[int(n)], float(s)) for n, s in zip(nbrs, sims)]

    def get_tag_category(self, tag: str) -> str:
        cid = self.tag_category.get(tag)
        return CATEGORY_MAP.get(cid, "unknown") if cid is not None else "unknown"

    def _compute_patch_tag_sim(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """Compute cosine similarity between projected patch features (W_k) and projected tag queries (W_q).
        This approximates P(tag | patch) by asking "which tag query is most compatible with this patch?"
        patch_tokens: (B, N_tok, D)
        Returns: (N_patches, C) similarity matrix
        """
        attn_mod = self.model.head.cross_attn
        d = attn_mod.embed_dim
        h = attn_mod.num_heads
        hd = d // h

        w = attn_mod.in_proj_weight
        b = attn_mod.in_proj_bias

        # Extract patches (skip CLS, registers)
        patches = patch_tokens[0, 1:1+self.num_image_patches].detach() # (N_p, D)
        queries = self.model.head.tag_queries.detach()                 # (N_t, D)

        # Project K and Q
        p_proj = (patches @ w[d:2*d].T + b[d:2*d]).reshape(-1, h, hd) # (N_p, H, hd)
        q_proj = (queries @ w[0:d].T + b[0:d]).reshape(-1, h, hd)     # (N_t, H, hd)

        # Normalize per head
        p_norm = F.normalize(p_proj, dim=2)
        q_norm = F.normalize(q_proj, dim=2)

        # Cosine similarity per head, then average
        sim_heads = torch.einsum('phd,qhd->phq', p_norm, q_norm) # (N_p, H, N_t)
        return sim_heads.mean(dim=1)                          # (N_p, N_t)

    def run_with_intermediates(self, pil_image: Image.Image) -> dict:
        tensor = self.val_transform(pil_image.convert("RGB"))
        inp = tensor.unsqueeze(0).to(self._device)

        with torch.no_grad():
            patch_tokens = self.model.backbone.forward_features(inp)
            queries = self.model.head.tag_queries.unsqueeze(0)
            tag_features, attn_weights = self.model.head.cross_attn(
                query=queries, key=patch_tokens, value=patch_tokens,
                need_weights=True, average_attn_weights=False,
            )
            logits = self.model.head.classifier(tag_features).squeeze(-1)
            scores = torch.sigmoid(logits).squeeze(0)

        self._last_scores = scores.cpu()
        # torch.nn.MultiheadAttention with batch_first=True + average_attn_weights=False:
        # returns (B, num_heads, N_queries, L_src). Squeeze batch → (H, C, N_tok).
        attn_transposed = attn_weights.squeeze(0)  # (H, C, N_tok) — no permute needed
        self._last_attn_weights = attn_transposed.cpu()
        self._num_heads = attn_transposed.shape[0]

        # Patch-to-tag semantic similarity (W_k(patch) ↔ W_q(tag))
        pt_sim = self._compute_patch_tag_sim(patch_tokens)

        # Denormalise
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        display = (tensor.cpu() * std + mean).clamp(0, 1)
        display_np = (display.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        display_pil = Image.fromarray(display_np)

        return {
            "attn_weights": attn_transposed.cpu(),     # (H, C, N_tok)
            "patch_tag_sim": pt_sim.cpu(),             # (N_p, C)
            "scores": scores.cpu(),                   # (C,)
            "processed_pil": display_pil,
            "processed_np": display_np,
        }


# ═════════════════════════════════════════════════════════════════════════════
#  Coordinate helpers
# ═════════════════════════════════════════════════════════════════════════════

def compute_pad(pil_size: tuple[int, int], target_size: int) -> tuple[int, int, int, int]:
    w, h = pil_size
    scale = target_size / max(w, h)
    nw, nh = int(w * scale), int(h * scale)
    left = (target_size - nw) // 2
    top = (target_size - nh) // 2
    return left, top, target_size - nw - left, target_size - nh - top


def pixel_to_patch(x: float, y: float, image_size: int = 448,
                   patch_size: int = 16, pad_info: tuple | None = None) -> int | None:
    """Map canvas pixel to model patch index.

    The model sees the full image_size x image_size as a patch_size grid.
    CLICK coordinates are already in the processed-image canvas space,
    so we do NOT subtract padding — the model's patches are absolute.
    """
    px, py = int(x), int(y)
    col, row = px // patch_size, py // patch_size
    ppr = image_size // patch_size
    if 0 <= col < ppr and 0 <= row < ppr:
        return row * ppr + col
    return None


# ═════════════════════════════════════════════════════════════════════════════
#  Heatmap rendering
# ═════════════════════════════════════════════════════════════════════════════

def _aggregate_attn(attn_per_head: torch.Tensor, tag_idx: int, np_: int,
                    mode: str = "avg") -> np.ndarray:
    """Aggregate per-head attention for one tag across all patches.

    attn_per_head: (H, C, N_tok)
    mode: "avg" | "max" | int (single head index)
    """
    if isinstance(mode, int):
        attn = attn_per_head[mode, tag_idx, 1:1 + np_]
    elif mode == "max":
        attn = attn_per_head[:, tag_idx, 1:1 + np_].max(dim=0).values
    else:
        attn = attn_per_head[:, tag_idx, 1:1 + np_].mean(dim=0)
    return attn.cpu().numpy()


def render_heatmap_base64(attn_weights: torch.Tensor, tag_idx: int,
                           image_np: np.ndarray, image_size: int = 448,
                           patch_size: int = 16, head_mode: str = "avg") -> str:
    """Return base64 PNG of 3-panel heatmap.

    head_mode: "avg" (mean over heads), "max" (max over heads), or int.
    """
    np_ = (image_size // patch_size) ** 2
    ppr = image_size // patch_size

    attn = _aggregate_attn(attn_weights, tag_idx, np_, head_mode)
    attn_2d = attn.reshape(ppr, -1)

    # Stats for annotation
    v_min, v_med, v_95, v_max = np.percentile(attn, [0, 50, 95, 100])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax in axes:
        ax.axis("off")

    axes[0].imshow(image_np)
    axes[0].set_title("Image")

    axes[1].imshow(image_np)
    up = np.kron(attn_2d, np.ones((patch_size, patch_size)))
    axes[1].imshow(up, cmap="magma", alpha=0.6)
    head_label = head_mode if isinstance(head_mode, int) else f"heads:{head_mode}"
    axes[1].set_title(f"Overlay ({head_label})")

    im = axes[2].imshow(attn_2d, cmap="magma", interpolation="nearest", origin='upper', vmin=0)
    axes[2].set_title(f"Attention (softmax QKᵀ/√d)\n"
                      f"min={v_min:.4f} med={v_med:.4f} 95th={v_95:.4f} max={v_max:.4f}")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


# ═════════════════════════════════════════════════════════════════════════════
#  FastAPI app
# ═════════════════════════════════════════════════════════════════════════════

# ── inline frontend HTML ────────────────────────────────────────────────────
_FRONTEND_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DTQ Model Explorer V0.1</title>
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#0f0f1a;color:#ddd;min-height:100vh}
a{color:#6af}
.header{background:#1a1a2e;padding:16px 24px;border-bottom:1px solid #333}
.header h1{font-size:20px;color:#fff}
.header p{font-size:13px;color:#888;margin-top:4px}
.tabs{display:flex;gap:0;background:#16162a;border-bottom:1px solid #333}
.tab{ padding:10px 24px;cursor:pointer;font-size:14px;border-bottom:3px solid transparent;color:#888;transition:.15s}
.tab:hover{color:#ddd}
.tab.active{color:#fff;border-bottom-color:#4a90d9}
.panel{display:none;padding:16px}
.panel.active{display:flex;flex-direction:column;gap:16px}
.row{display:flex;gap:16px;flex-wrap:wrap}
.col{flex:1;min-width:280px}
.card{background:#1a1a2e;border-radius:8px;border:1px solid #2a2a3e;padding:12px}
.card h3{font-size:13px;color:#888;margin-bottom:8px;text-transform:uppercase;letter-spacing:.5px}
.btn{background:#4a90d9;color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:13px;cursor:pointer;transition:.15s}
.btn:hover{background:#5a9fe9}
.btn-sm{padding:4px 10px;font-size:12px}
.btn-secondary{background:#2a2a3e;color:#ccc}
.btn-secondary:hover{background:#3a3a4e}
.btn-danger{background:#c0392b}
.btn-danger:hover{background:#e74c3c}
input[type=text]{background:#0f0f1a;border:1px solid #333;border-radius:6px;padding:8px 12px;color:#ddd;font-size:13px;width:100%}
input[type=text]:focus{outline:none;border-color:#4a90d9}
#image-canvas-wrapper{position:relative;display:inline-block;cursor:crosshair}
#image-canvas{border-radius:4px;display:block;max-width:100%}
#patch-overlay{position:absolute;top:0;left:0;pointer-events:none;max-width:100%}
.tag-row{display:flex;align-items:center;padding:4px 8px;border-radius:4px;cursor:pointer;font-size:13px;gap:8px;transition:.1s}
.tag-row:hover{background:rgba(74,144,217,0.15)}
.tag-row.active{background:rgba(74,144,217,0.25)}
.tag-rank{color:#555;width:24px;text-align:right;font-variant-numeric:tabular-nums}
.tag-dot{width:8px;height:8px;border-radius:50%;display:inline-block;flex-shrink:0}
.tag-name{flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.tag-val{font-variant-numeric:tabular-nums;color:#888;width:52px;text-align:right}
.tag-cat{font-size:11px;color:#555;width:56px;text-align:right}
#top-tags-list{max-height:200px;overflow-y:auto;margin-bottom:4px;scrollbar-width:thin}
#top-tags-list::-webkit-scrollbar{width:4px}
#top-tags-list::-webkit-scrollbar-thumb{background:#333;border-radius:2px}
#tag-list{max-height:400px;overflow-y:auto;scrollbar-width:thin}
#tag-list::-webkit-scrollbar{width:4px}
#tag-list::-webkit-scrollbar-thumb{background:#333;border-radius:2px}
.heatmap-img{width:100%;border-radius:4px;max-width:720px}
#graph-container{width:100%;height:520px;background:#12122a;border-radius:8px}
.neighbor-row{display:flex;align-items:center;padding:2px 4px;gap:6px;font-size:12px;cursor:pointer;border-radius:3px}
.neighbor-row:hover{background:rgba(74,144,217,0.12)}
#tag-detail{font-size:13px;line-height:1.6}
#tag-detail b{color:#fff}
#tag-detail code{background:#2a2a3e;padding:1px 6px;border-radius:3px;font-size:12px}
.status{font-size:13px;color:#888;padding:8px 0}
.status.ok{color:#4a9}
.status.err{color:#e74c3c}
.graph-controls{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
#graph-search{flex:1;min-width:160px}
.node-info{font-size:13px;line-height:1.6;padding:8px;background:#0f0f1a;border-radius:6px;max-height:180px;overflow-y:auto}
.node-info b{color:#fff}
.node-info .neighbor-row{font-size:12px}
.split{display:flex;gap:16px;flex-wrap:wrap}
.split-main{flex:3;min-width:400px}
.split-side{flex:1;min-width:240px}
</style>
</head>
<body>

<div class="header">
  <h1>🔬 DTQ Model Explorer V0.1</h1>
  <p>Cross-attn = softmax(QKᵀ/√d). Switch modes: Attention (tag→patch) vs Semantic (patch↔tag).</p>
</div>

<div class="tabs" id="tabs">
  <div class="tab active" data-tab="0">🖼️ Patch ↔ Tag Inspector</div>
  <div class="tab" data-tab="1">🕸️ Semantic Graph</div>
</div>

<!-- ════════════ Tab 0: Inspector ════════════ -->
<div class="panel active" id="panel-0">
  <div class="row">
    <div class="col" style="flex:1.2">
      <!-- Top Tags card -->
      <div class="card" style="margin-bottom:16px">
        <h3>🏆 Top Tags (inference)</h3>
        <div id="top-tags-list"><span style="color:#555;font-size:13px">Load an image</span></div>
      </div>
      <!-- Image card -->
      <div class="card">
        <h3>Upload Image</h3>
        <input type="file" id="image-upload" accept="image/*" style="margin-bottom:8px">
        <div id="image-canvas-wrapper">
          <canvas id="image-canvas" width="448" height="448"></canvas>
          <canvas id="patch-overlay" width="448" height="448"></canvas>
        </div>
        <div class="status" id="infer-status">Load an image to begin.</div>
      </div>
    </div>
    <div class="col" style="flex:0.8">
      <!-- Patch tags card -->
      <div class="card" style="height:100%">
        <h3 id="patch-info">Selected Patches <span style="color:#555;font-weight:normal" id="patch-count"></span></h3>
        <div style="display:flex;gap:6px;margin:6px 0">
          <button class="btn btn-sm btn-secondary mode-btn active" data-mode="attention">Attention (tag→patch)</button>
          <button class="btn btn-sm btn-secondary mode-btn" data-mode="semantic">Semantic (patch↔tag)</button>
        </div>
        <p id="mode-desc" style="font-size:11px;color:#666;margin-bottom:6px">⚠ Attention shows where tags looked, weighted by score. Not P(tag|patch).</p>
        <div id="tag-list" style="max-height:430px"></div>
      </div>
    </div>
  </div>

  <div class="card">
    <div class="split">
      <div class="split-main">
        <h3>Attention Heatmap (softmax QKᵀ/√d, not saliency)</h3>
        <div style="display:flex;gap:8px;margin-bottom:6px;align-items:center">
          <label style="font-size:12px;color:#888">Head:</label>
          <select id="head-select" style="background:#0f0f1a;border:1px solid #333;color:#ddd;padding:4px 8px;border-radius:4px;font-size:12px">
            <option value="avg">Average</option>
            <option value="max">Max</option>
          </select>
        </div>
        <div id="heatmap-container"><span style="color:#555;font-size:13px">Click a tag to see heatmap</span></div>
      </div>
      <div class="split-side">
        <h3>Tag Details</h3>
        <div id="tag-detail"><span style="color:#555;font-size:13px">Select a tag</span></div>
        <div style="margin-top:8px">
          <button class="btn btn-sm" id="send-to-graph-btn" disabled>📊 Show in Graph</button>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- ════════════ Tab 1: Graph ════════════ -->
<div class="panel" id="panel-1">
  <div class="row">
    <div class="col" style="flex:1;min-width:220px">
      <div class="card">
        <h3>Controls</h3>
        <input type="text" id="graph-search" placeholder="Tag name + Enter" style="margin-bottom:8px">
        <div class="graph-controls">
          <button class="btn btn-sm btn-secondary" id="graph-physics">🔄 Physics</button>
          <button class="btn btn-sm btn-danger" id="graph-clear">🧹 Clear</button>
        </div>
        <div class="status" id="graph-status">Add tags from the inspector or search above.</div>
        <div id="graph-node-info" class="node-info"></div>
      </div>
    </div>
    <div class="col" style="flex:3">
      <div class="card" style="padding:0">
        <div id="graph-container"></div>
      </div>
    </div>
  </div>
</div>

<script>
// ── state ──────────────────────────────────────────────────────────────────
const S = {
    imageNp: null, padInfo: null, imageSize: 448, numHeads: 8,
    selectedTag: null, headMode: 'avg', patchMode: 'attention',
    selectedPatches: new Map(),  // patchIdx => {row, col}
    top10Tags: [],               // [{tag, score, category}]
    graphNodes: new vis.DataSet(),
    graphEdges: new vis.DataSet(),
    network: null, physics: true,
};

// ── canvas setup ───────────────────────────────────────────────────────────
const canvas = document.getElementById('image-canvas');
const ctx = canvas.getContext('2d');
const overlay = document.getElementById('patch-overlay');
const octx = overlay.getContext('2d');
const PS = 16, PPR = 28;

// ── tab switching ──────────────────────────────────────────────────────────
document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
        tab.classList.add('active');
        document.getElementById('panel-' + tab.dataset.tab).classList.add('active');
        if (tab.dataset.tab === '1' && S.network) S.network.fit();
    });
});

// ── helpers ────────────────────────────────────────────────────────────────
const CAT_COLORS = {'general':'#4a90d9','copyright':'#7b61ff','character':'#e8522e','unknown':'#999999'};

function showError(id, msg) {
    const el = document.getElementById(id);
    el.textContent = '❌ ' + msg;
    el.className = 'status err';
}

function tagRowHtml(t, showVal) {
    const color = CAT_COLORS[t.category] || '#999';
    const valKey = t.mode === 'semantic' ? 'similarity' : 'val';
    const valTitle = t.mode === 'semantic' ? 'cosine(Wk, Wq)' : 'attn × score';
    return `<div class="tag-row" data-tag="${t.tag}">
      <span class="tag-dot" style="background:${color}"></span>
      <span class="tag-name">${t.tag}</span>
      ${showVal ? `<span class="tag-val" title="${valTitle}">${(t[valKey]||0).toFixed(4)}</span>` : ''}
      <span class="tag-val">${(t.score||0).toFixed(4)}</span>
      <span class="tag-cat"><code>${t.category}</code></span>
    </div>`;
}

// ── draw grid overlay ─────────────────────────────────────────────────────
function drawGrid() {
    octx.clearRect(0, 0, overlay.width, overlay.height);
    const w = overlay.width, h = overlay.height;
    octx.strokeStyle = 'rgba(255,255,255,0.12)';
    octx.lineWidth = 0.5;
    for (let i = 0; i <= w; i += PS) { octx.beginPath(); octx.moveTo(i,0); octx.lineTo(i,h); octx.stroke(); }
    for (let i = 0; i <= h; i += PS) { octx.beginPath(); octx.moveTo(0,i); octx.lineTo(w,i); octx.stroke(); }
    // Highlight selected patches
    for (const [pid, rc] of S.selectedPatches) {
        const x = rc.col * PS, y = rc.row * PS;
        octx.fillStyle = 'rgba(66,133,244,0.25)';
        octx.fillRect(x, y, PS, PS);
        octx.strokeStyle = '#4285f4';
        octx.lineWidth = 2;
        octx.strokeRect(x, y, PS, PS);
    }
}

// ── show heatmap for a tag ─────────────────────────────────────────────────
async function showTagHeatmap(tag) {
    S.selectedTag = tag;
    document.getElementById('send-to-graph-btn').disabled = false;
    const fd = new FormData(); fd.append('tag', tag);
    if (S.headMode !== 'avg') fd.append('head_mode', S.headMode);
    try {
        const r = await fetch('/tag-info', { method:'POST', body:fd });
        const d = await r.json();
        if (d.error) { document.getElementById('tag-detail').innerHTML = d.error; return; }
        document.getElementById('heatmap-container').innerHTML =
            `<img class="heatmap-img" src="data:image/png;base64,${d.heatmap}" alt="heatmap">`;
        const catColor = CAT_COLORS[d.category] || '#999';
        const noiseWarn = d.attn_noise_ratio !== undefined && d.attn_noise_ratio < 0.05
            ? '<p style="color:#e67e22;font-size:11px;margin-top:4px">⚠ Flat attention — top patches are noise, not meaningful</p>' : '';
        document.getElementById('tag-detail').innerHTML = `
          <b>${d.tag}</b><br>
          Score: <b>${d.score.toFixed(4)}</b> &middot; Cat: <code style="background:${catColor}22;color:${catColor}">${d.category}</code><br>
          Top patches: ${d.top_patches.map(p => `(${p.row},${p.col})`).join(', ')}<br>
          ${noiseWarn}<br>
          <b>Nearest tags:</b><br>
          ${d.neighbours.slice(0,15).map(n =>
            `<div class="neighbor-row" data-tag="${n.tag}">
               <span class="tag-dot" style="background:${n.color}"></span>
               <span>${n.tag}</span>
               <span style="color:#555;margin-left:auto">${n.similarity.toFixed(3)}</span>
             </div>`
          ).join('')}`;
        // Neighbour click → heatmap
        document.querySelectorAll('#tag-detail .neighbor-row').forEach(el => {
            el.addEventListener('click', () => showTagHeatmap(el.dataset.tag));
        });
    } catch (err) {
        showError('infer-status', 'Fetch error: ' + err.message);
    }
}

// ── attach tag-row click listeners ─────────────────────────────────────────
function attachTagRowListeners(containerId) {
    document.querySelectorAll(`#${containerId} .tag-row`).forEach(row => {
        row.addEventListener('click', () => {
            const tag = row.dataset.tag;
            // Highlight active row
            document.querySelectorAll(`#${containerId} .tag-row`).forEach(r => r.classList.remove('active'));
            row.classList.add('active');
            showTagHeatmap(tag);
        });
    });
}

// ── fetch aggregated tags for selected patches ─────────────────────────────
async function fetchSelectedPatchTags() {
    const ids = [...S.selectedPatches.keys()];
    const count = document.getElementById('patch-count');
    const list = document.getElementById('tag-list');

    if (ids.length === 0) {
        count.textContent = '';
        list.innerHTML = '<span style="color:#555;font-size:13px">Click patches to explore</span>';
        return;
    }
    count.textContent = `(${ids.length} patch${ids.length>1?'es':''})`;

    const fd = new FormData();
    fd.append('patches', ids.join(','));
    fd.append('mode', S.patchMode);
    try {
        const r = await fetch('/multi-patch-tags', { method:'POST', body:fd });
        const d = await r.json();
        if (d.error) { list.innerHTML = d.error; return; }
        list.innerHTML = d.tags.map(t => tagRowHtml(t, true)).join('');
        attachTagRowListeners('tag-list');
    } catch (err) {
        list.innerHTML = 'Error: ' + err.message;
    }
}

// ── image upload ───────────────────────────────────────────────────────────
document.getElementById('image-upload').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const status = document.getElementById('infer-status');
    status.textContent = '⏳ Running inference...';
    status.className = 'status';

    S.selectedPatches.clear();
    S.selectedTag = null;

    const fd = new FormData();
    fd.append('file', file);
    try {
        const resp = await fetch('/infer', { method: 'POST', body: fd });
        const data = await resp.json();
        if (data.error) { showError('infer-status', data.error); return; }

        // Draw image
        const img = new Image();
        img.onload = () => {
            S.imageSize = data.image_size;
            S.padInfo = data.pad_info;
            canvas.width = img.width; canvas.height = img.height;
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            overlay.width = canvas.width; overlay.height = canvas.height;
            drawGrid();
        };
        img.src = 'data:image/png;base64,' + data.image;

        // Store top10
        S.top10Tags = data.top10;

        // Populate head selector
        if (data.num_heads) {
            S.numHeads = data.num_heads;
            const sel = document.getElementById('head-select');
            sel.innerHTML = '<option value="avg">Average</option>' +
                            '<option value="max">Max</option>' +
                            Array.from({length: data.num_heads}, (_,i) => `<option value="${i}">Head ${i}</option>`).join('');
        }

        // Render top tags box
        document.getElementById('top-tags-list').innerHTML =
            data.top10.map(t => tagRowHtml(t, false)).join('');
        attachTagRowListeners('top-tags-list');

        // Reset patch area
        document.getElementById('patch-count').textContent = '';
        document.getElementById('tag-list').innerHTML = '<span style="color:#555;font-size:13px">Click patches to explore</span>';
        document.getElementById('heatmap-container').innerHTML = '<span style="color:#555;font-size:13px">Click a tag to see heatmap</span>';
        document.getElementById('tag-detail').innerHTML = '<span style="color:#555;font-size:13px">Select a tag</span>';
        document.getElementById('send-to-graph-btn').disabled = true;

        status.textContent = `✅ ${data.num_tags} tags · Top: ${data.top10.map(t=>t.tag).join(', ')}`;
        status.className = 'status ok';
    } catch (err) {
        showError('infer-status', 'Fetch error: ' + err.message);
    }
});

// ── click on canvas → toggle patch selection ───────────────────────────────
canvas.addEventListener('click', async (e) => {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;

    // Get patch from server
    const fd = new FormData();
    fd.append('x', x); fd.append('y', y);
    fd.append('mode', S.patchMode);
    try {
        const r = await fetch('/patch-tags', { method: 'POST', body: fd });
        const d = await r.json();
        if (d.error || d.patch == null) return;

        const pid = d.patch;
        // Toggle selection
        if (S.selectedPatches.has(pid)) {
            S.selectedPatches.delete(pid);
        } else {
            S.selectedPatches.set(pid, { row: d.row, col: d.col });
        }
        drawGrid();
        await fetchSelectedPatchTags();
    } catch (err) {
        showError('infer-status', 'Fetch error: ' + err.message);
    }
});

// ── send to graph ──────────────────────────────────────────────────────────
document.getElementById('send-to-graph-btn').addEventListener('click', () => {
    if (!S.selectedTag) return;
    addTagToGraph(S.selectedTag);
    document.querySelector('.tab[data-tab="1"]').click();
});

// ── Semantic Graph ─────────────────────────────────────────────────────────
function initGraph() {
    const container = document.getElementById('graph-container');
    const data = { nodes: S.graphNodes, edges: S.graphEdges };
    const options = {
        nodes: { shape: 'dot', font: { size: 11, color: '#ddd', strokeWidth: 2, strokeColor: '#222' } },
        edges: { smooth: { type: 'continuous' } },
        physics: {
            enabled: true, solver: 'forceAtlas2Based',
            forceAtlas2Based: { gravitationalConstant: -40, centralGravity: 0.005, springLength: 160, springConstant: 0.02, damping: 0.4 },
        },
        interaction: { hover: true, tooltipDelay: 200, navigationButtons: true, keyboard: true },
    };
    S.network = new vis.Network(container, data, options);
    S.network.on('click', p => { if (p.nodes.length) expandGraphNode(p.nodes[0]); });
    // Seed
    fetch('/neighbors/1girl?limit=8').then(r => r.json()).then(d => {
        if (d.error) return;
        addNode('1girl', 'general', 0);
        d.neighbors.forEach(n => { addNode(n.tag, n.category, 0); addEdge('1girl', n.tag, n.similarity); });
    });
}

function addNode(id, category, score) {
    if (!S.graphNodes.get(id)) {
        const color = CAT_COLORS[category] || '#999';
        S.graphNodes.add({ id, label: id, title: `${id}<br>${category}<br>score:${score.toFixed(4)}`, color: { background: color, border: color }, size: Math.max(10, 10 + score * 60) });
    }
}
function addEdge(from, to, weight) {
    const key = from < to ? from+'~'+to : to+'~'+from;
    if (!S.graphEdges.get(key)) {
        S.graphEdges.add({ id: key, from, to, value: weight, width: Math.max(0.5, weight * 6), color: { color: 'rgba(255,255,255,0.25)', highlight: 'rgba(255,255,255,0.5)' }, title: `sim:${weight.toFixed(3)}` });
    }
}

async function expandGraphNode(tag) {
    try {
        const r = await fetch(`/neighbors/${encodeURIComponent(tag)}?limit=20`);
        const d = await r.json();
        if (d.error) return;
        addNode(d.tag, '', 0);
        let c = 0;
        d.neighbors.forEach(n => { if (!S.graphNodes.get(n.tag)) { c++; addNode(n.tag, n.category, 0); } addEdge(d.tag, n.tag, n.similarity); });
        document.getElementById('graph-status').textContent = `Expanded ${tag} +${c} (${S.graphNodes.length} total)`;
    } catch (err) { document.getElementById('graph-status').textContent = 'Error: ' + err.message; }
}

async function addTagToGraph(tag) {
    try {
        const r = await fetch(`/neighbors/${encodeURIComponent(tag)}?limit=15`);
        const d = await r.json();
        if (d.error) return;
        addNode(d.tag, '', 0);
        d.neighbors.forEach(n => { addNode(n.tag, n.category, 0); addEdge(d.tag, n.tag, n.similarity); });
        document.getElementById('graph-status').textContent = `Added ${tag} +${d.neighbors.length} neighbors`;
        if (S.network) S.network.fit();
    } catch (err) { document.getElementById('graph-status').textContent = 'Error: ' + err.message; }
}

document.getElementById('graph-search').addEventListener('keydown', async (e) => {
    if (e.key !== 'Enter') return;
    const tag = e.target.value.trim();
    if (!tag) return;
    e.target.value = '';
    const rt = await fetch(`/neighbors/${encodeURIComponent(tag)}?limit=20`);
    const d = await rt.json();
    if (d.error) { document.getElementById('graph-status').textContent = d.error; return; }
    addNode(d.tag, d.category || 'general', 0);
    d.neighbors.forEach(n => { addNode(n.tag, n.category, 0); addEdge(d.tag, n.tag, n.similarity); });
    document.getElementById('graph-status').textContent = `Added ${tag} +${d.neighbors.length} neighbors`;
    if (S.network) S.network.fit();
});

document.getElementById('graph-physics').addEventListener('click', () => {
    S.physics = !S.physics;
    S.network.setOptions({ physics: { enabled: S.physics } });
    document.getElementById('graph-status').textContent = `Physics ${S.physics ? 'ON' : 'OFF'}`;
});

document.getElementById('graph-clear').addEventListener('click', () => {
    S.graphNodes.clear(); S.graphEdges.clear();
    document.getElementById('graph-status').textContent = 'Cleared.';
});

// ── head selector ─────────────────────────────────────────────────────────
document.getElementById('head-select').addEventListener('change', (e) => {
    S.headMode = e.target.value;
    if (S.selectedTag) showTagHeatmap(S.selectedTag);
});

// ── patch mode toggle ─────────────────────────────────────────────────────
document.querySelectorAll('.mode-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        S.patchMode = btn.dataset.mode;
        const desc = document.getElementById('mode-desc');
        if (S.patchMode === 'semantic') {
            desc.textContent = 'Semantic: cosine(Wk(patch), Wq(tag)). Shows which tag queries are most compatible with selected patches.';
        } else {
            desc.textContent = 'Attention: softmax(QKᵀ/√d) × score. Shows where tags looked, weighted by confidence.';
        }
        fetchSelectedPatchTags();
    });
});

// ── init ───────────────────────────────────────────────────────────────────
initGraph();
</script>
</body>
</html>"""


def create_app(inf: ExplorerInference):
    from fastapi import FastAPI, UploadFile, File, Form
    from fastapi.responses import HTMLResponse
    import uvicorn

    app = FastAPI(title="DTQ Model Explorer")

    # ── state per request ───────────────────────────────────────────────
    session = {
        "intermediates": None,
        "image_np": None,
        "pad_info": None,
        "image_size": inf.image_size,
    }

    # ══════════════════════════════════════════════════════════════════════
    #  HTML frontend (single page)
    # ══════════════════════════════════════════════════════════════════════

    @app.get("/", response_class=HTMLResponse)
    def index():
        return HTMLResponse(_FRONTEND_HTML)

    # ══════════════════════════════════════════════════════════════════════
    #  API endpoints
    # ══════════════════════════════════════════════════════════════════════

    @app.post("/infer")
    async def api_infer(file: UploadFile = File(...)):
        """Upload image → run inference → return scores + image data."""
        raw = await file.read()
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
        orig_size = pil.size

        inter = inf.run_with_intermediates(pil)
        session["intermediates"] = inter
        session["image_np"] = inter["processed_np"]
        session["pad_info"] = compute_pad(orig_size, inf.image_size)

        scores = inter["scores"]
        top10_idx = scores.argsort(descending=True)[:10]
        top10 = [
            {"tag": inf.id_to_tag[int(i)], "score": float(scores[i]),
             "category": inf.get_tag_category(inf.id_to_tag[int(i)])}
            for i in top10_idx
        ]

        # Encode processed image as base64
        buf = io.BytesIO()
        inter["processed_pil"].save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        return {
            "image": img_b64,
            "image_size": inf.image_size,
            "num_heads": inf._num_heads,
            "num_tags": inf.num_classes,
            "top10": top10,
            "pad_info": list(session["pad_info"]),
        }

    @app.post("/patch-tags")
    def api_patch_tags(x: float = Form(), y: float = Form(), mode: str = Form(default="attention")):
        """Given click coordinates, return top-30 tags.

        mode: "attention" → P(patch|tag)×score (where tag looked)
              "semantic"  → cosine(W_k(patch), W_q(tag)) (what patch depicts)
        """
        inter = session.get("intermediates")
        if inter is None:
            return {"error": "No image loaded"}

        pid = pixel_to_patch(x, y, inf.image_size, PATCH_SIZE, session.get("pad_info"))
        if pid is None:
            return {"error": "Click on image area (not padding)", "patch": None}

        scores = inter["scores"]
        if mode == "semantic":
            # Cosine similarity between projected patch & projected tag queries
            sim = inter["patch_tag_sim"]  # (N_p, C)
            vals = sim[pid, :]            # (C,)
            mode_label = "semantic"
        else:
            # Cross-attention weighted by prediction score
            attn = inter["attn_weights"]  # (H, C, N_tok)
            patch_attn = attn[:, :, pid + 1].mean(dim=0)  # (C,)
            vals = patch_attn * scores    # (C,)
            mode_label = "attention"

        top_idx = vals.argsort(descending=True)[:30]

        tags = []
        for rank, ti in enumerate(top_idx):
            ti_i = int(ti)
            tag = inf.id_to_tag[ti_i]
            row = {
                "rank": rank + 1,
                "tag": tag,
                "score": float(scores[ti_i]),
                "category": inf.get_tag_category(tag),
                "color": CATEGORY_COLORS.get(inf.get_tag_category(tag), "#999"),
                "mode": mode_label,
                "val": float(vals[ti]),
            }
            if mode == "attention":
                row["attention"] = float(attn[:, :, pid + 1].mean(dim=0)[ti])
            else:
                row["similarity"] = float(inter["patch_tag_sim"][pid, ti])
            tags.append(row)

        ppr = inf.image_size // PATCH_SIZE
        return {
            "patch": pid,
            "row": pid // ppr,
            "col": pid % ppr,
            "tags": tags,
        }

    @app.post("/multi-patch-tags")
    def api_multi_patch_tags(patches: str = Form(), mode: str = Form(default="attention")):
        """Given comma-separated patch IDs, return top-30 tags.

        mode: "attention" or "semantic"
        """
        inter = session.get("intermediates")
        if inter is None:
            return {"error": "No image loaded"}
        pids = [int(p.strip()) for p in patches.split(",") if p.strip().isdigit()]
        if not pids:
            return {"error": "No valid patch IDs"}

        scores = inter["scores"]
        if mode == "semantic":
            sim = inter["patch_tag_sim"]  # (N_p, C)
            vals = sim[[pid for pid in pids], :].mean(dim=0)  # avg sim across patches → (C,)
            mode_label = "semantic"
        else:
            attn = inter["attn_weights"]   # (H, C, N_tok)
            patch_attn = attn[:, :, [pid + 1 for pid in pids]].mean(dim=2).mean(dim=0)  # (C,)
            vals = patch_attn * scores
            mode_label = "attention"

        top_idx = vals.argsort(descending=True)[:30]

        tags = []
        for rank, ti in enumerate(top_idx):
            ti_i = int(ti)
            tag = inf.id_to_tag[ti_i]
            row = {
                "rank": rank + 1,
                "tag": tag,
                "score": float(scores[ti_i]),
                "category": inf.get_tag_category(tag),
                "color": CATEGORY_COLORS.get(inf.get_tag_category(tag), "#999"),
                "mode": mode_label,
                "val": float(vals[ti]),
            }
            if mode == "attention":
                row["attention"] = float(attn[:, :, [pid + 1 for pid in pids]].mean(dim=2).mean(dim=0)[ti])
            else:
                row["similarity"] = float(sim[[pid for pid in pids], ti].mean())
            tags.append(row)
        return {"tags": tags, "num_patches": len(pids)}

    @app.post("/tag-info")
    def api_tag_info(tag: str = Form(), head_mode: str = Form(default="avg")):
        """Return heatmap + neighbours + metadata for a tag.

        head_mode: "avg" (mean over heads), "max" (max over heads), or int.
        """
        inter = session.get("intermediates")
        if inter is None:
            return {"error": "No image loaded"}
        ti = inf.tag_to_id.get(tag)
        if ti is None:
            return {"error": f"Unknown tag: {tag}"}

        # Parse head_mode
        hm = "avg"
        if head_mode != "avg":
            try:
                hm = int(head_mode)
            except ValueError:
                hm = head_mode  # "max" or other

        # Heatmap
        heatmap_b64 = render_heatmap_base64(
            inter["attn_weights"], ti, session["image_np"],
            inf.image_size, PATCH_SIZE, hm,
        )

        # Neighbours
        neighbours = [
            {"tag": t, "similarity": s,
             "category": inf.get_tag_category(t),
             "color": CATEGORY_COLORS.get(inf.get_tag_category(t), "#999")}
            for t, s in inf.get_tag_similarity(ti)[:20]
        ]

        # Top patches (by avg attention across heads)
        np_ = inf.num_image_patches
        attn = inter["attn_weights"]  # (H, C, N_tok) — already on CPU
        patch_attn = attn[:, ti, 1:1 + np_].mean(dim=0)  # (np_,)
        sorted_idx = patch_attn.argsort(descending=True)
        top_p = sorted_idx[:5].tolist()
        top_vals = float(patch_attn[sorted_idx[0]])
        min_val = float(patch_attn.min())
        noise_ratio = (top_vals - min_val) / (min_val + 1e-8)
        ppr = inf.image_size // PATCH_SIZE

        return {
            "tag": tag,
            "score": float(inter["scores"][ti]),
            "category": inf.get_tag_category(tag),
            "top_patches": [{"idx": int(p), "row": int(p) // ppr, "col": int(p) % ppr} for p in top_p],
            "attn_noise_ratio": noise_ratio,
            "heatmap": heatmap_b64,
            "neighbours": neighbours,
        }

    @app.get("/neighbors/{tag_name}")
    def api_neighbors(tag_name: str, limit: int = 20):
        """Return top-K neighbours of a tag (for graph expansion)."""
        ti = inf.tag_to_id.get(tag_name)
        if ti is None:
            return {"error": f"Unknown tag: {tag_name}"}
        nbrs = [
            {"tag": t, "similarity": s,
             "category": inf.get_tag_category(t),
             "color": CATEGORY_COLORS.get(inf.get_tag_category(t), "#999")}
            for t, s in inf.get_tag_similarity(ti)[:limit]
        ]
        return {"tag": tag_name, "neighbors": nbrs}

    @app.get("/tag-names")
    def api_tag_names(query: str = "", limit: int = 50):
        """Search tag names (for autocomplete)."""
        if not query:
            return {"tags": []}
        q = query.lower()
        matches = [t for t in inf.tag_to_id if q in t.lower()][:limit]
        return {"tags": matches}

    return app


# ═════════════════════════════════════════════════════════════════════════════
#  Launch
# ═════════════════════════════════════════════════════════════════════════════

def launch(checkpoint: str, host: str = "127.0.0.1", port: int = 8000):
    import uvicorn
    print(f"Loading checkpoint: {checkpoint}")
    inf = ExplorerInference(checkpoint)
    print(f"  Model       : {inf.model_name}")
    print(f"  Image size  : {inf.image_size}")
    print(f"  Tags        : {inf.num_classes}")
    print(f"  Device      : {inf._device}")
    print(f"  Server      : http://{host}:{port}")

    app = create_app(inf)
    uvicorn.run(app, host=host, port=port, log_level="info")


def main():
    p = argparse.ArgumentParser(description="DTQ Model Explorer V0.1")
    p.add_argument("checkpoint", help="Path to .pt checkpoint")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    launch(args.checkpoint, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
