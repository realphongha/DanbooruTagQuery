"""
Gradio web UI for DanbooruTagCLIP inference.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from PIL import Image

from .api import (
    CATEGORY_MAP,
    configure as api_configure,
    get_auth_info,
    wiki_body,
)
from .cache import TagCache
from .config import TrainConfig
from .defaults import DEFAULT_TOP_K, DEFAULT_MIN_SCORE
from .onnx_utils import Predictor
from .transforms import val_transforms

# ── globals ──────────────────────────────────────────────────────────────────

_CACHE = TagCache()  # module-level, shared across all sessions
_CATEGORY_NAMES = sorted(CATEGORY_MAP.values())

# ── model helpers ────────────────────────────────────────────────────────────

@torch.inference_mode()
def predict_all(image: Image.Image, predictor: Predictor):
    tensor = predictor.val_transform(image.convert("RGB")).unsqueeze(0)
    scores = predictor.run(tensor)[0]  # (num_classes,) numpy
    inv = {v: k for k, v in predictor.tag_to_id.items()}
    indices = np.argsort(scores)[::-1]
    return [(inv[int(i)], float(scores[i])) for i in indices]


def format_tag(tag: str, use_underscore: bool) -> str:
    return tag if use_underscore else tag.replace("_", " ")


# ── tag metadata enrichment (cache-only, no API) ──────────────────────────

def enrich_tags(tags_scores: list[tuple[str, float]]) -> dict[str, dict]:
    """Return dict: tag_name -> {score, category, category_name, wiki_body}.

    Only reads from cache — no API calls. Wiki body is None unless
    previously cached; lazy-fetched on hover via get_tag_description().
    """
    result: dict[str, dict] = {}
    for tag, score in tags_scores:
        cached = _CACHE.get(tag)
        if cached is not None:
            cat_id, wbody = cached
        else:
            cat_id, wbody = None, None
        result[tag] = {
            "score": score,
            "category": cat_id,
            "category_name": CATEGORY_MAP.get(cat_id) if cat_id is not None else None,
            "wiki_body": wbody,
        }
    return result


def get_tag_description(tag: str) -> str:
    """Fetch wiki body for *tag* — checks cache, falls back to API."""
    cached = _CACHE.get(tag)
    if cached is not None and cached[1] is not None:
        return cached[1]
    try:
        body = wiki_body(tag) or ""
    except Exception:
        body = ""
    cat = cached[0] if cached else None
    _CACHE.set(tag, cat, body or None)
    return body


# ── Gradio app ───────────────────────────────────────────────────────────────

def build_app(predictor: Predictor):
    state = {
        "all_logits": None,
        "tag_metadata": None,
        "current_image": None,
        "predictor": predictor,
    }

    # load initial auth info for display
    init_user, init_key_masked = get_auth_info()

    with gr.Blocks(title="DanbooruTagCLIP", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🏷️ DanbooruTagCLIP")

        with gr.Row():
            # ── left: image + credentials ──
            with gr.Column(scale=1):
                image_input = gr.Image(
                    label="Image",
                    type="pil",
                    sources=["upload", "clipboard"],
                    height=300,
                )
                url_input = gr.Textbox(
                    label="Image URL",
                    placeholder="Paste image URL and press Enter",
                )
                with gr.Row():
                    analyze_btn = gr.Button("🔍 Analyze", variant="primary", scale=2)
                    clear_btn = gr.Button("🗑️ Clear", scale=1)

                gr.Markdown("### 🔑 Danbooru API")
                with gr.Row():
                    api_user = gr.Textbox(
                        label="User",
                        value=init_user or "",
                        placeholder="DANBOORU_USER",
                        scale=1,
                    )
                    api_key = gr.Textbox(
                        label="API Key",
                        value=init_key_masked or "",
                        placeholder="DANBOORU_API_KEY",
                        type="password",
                        scale=2,
                    )
                auth_status = gr.Markdown(
                    "✅ Loaded from `.env`" if init_user else "⚠️ No credentials"
                )

            # ── right: params + categories ──
            with gr.Column(scale=1):
                top_k = gr.Number(
                    label="Top-K", value=DEFAULT_TOP_K, minimum=0, step=1
                )
                min_score = gr.Slider(
                    label="Min Score",
                    value=DEFAULT_MIN_SCORE,
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                )
                sort_by = gr.Radio(
                    label="Sort by", choices=["score", "name"], value="score"
                )
                use_underscore = gr.Checkbox(
                    label="Use underscore (_)", value=False
                )
                categories = gr.CheckboxGroup(
                    label="Categories",
                    choices=_CATEGORY_NAMES,
                    value=["general"],
                )

        # ── hover lazy-load (hidden) ──
        hover_tag = gr.Textbox(visible=False, elem_id="hover-tag")
        hover_desc = gr.Textbox(visible=False, elem_id="hover-desc")
        hover_tag.change(fn=get_tag_description, inputs=[hover_tag], outputs=[hover_desc])

        # ── outputs ──
        with gr.Tabs():
            with gr.TabItem("📋 Tag list"):
                tag_table = gr.HTML(label="Tags")
            with gr.TabItem("📝 Comma-separated"):
                tag_string = gr.Textbox(label="Tags", lines=6)

        # ── status row ──
        with gr.Row():
            status = gr.Markdown("Ready. Load an image and click **Analyze**.")
            clear_cache_btn = gr.Button("🧹 Clear tag cache", size="sm")

        # ── helpers ────────────────────────────────────────────────────────

        def make_tooltip(body: str | None) -> str:
            if not body:
                return ""
            # strip wiki markup for a clean tooltip
            clean = body.replace("[[", "").replace("]]", "")
            # keep first 300 chars
            return clean[:300].replace('"', "&quot;")

        def refresh_results(
            _top_k, _min_score, _sort_by, _use_underscore, _categories,
        ):
            if state["all_logits"] is None or state["tag_metadata"] is None:
                return "<i>No results yet.</i>", ""

            # filter by category
            meta = state["tag_metadata"]
            all_tags = list(meta.keys())
            if _categories:
                all_tags = [
                    t for t in all_tags
                    if meta[t]["category_name"] in _categories
                ]

            # build list sorted by score (original order)
            items = [(t, meta[t]["score"]) for t in all_tags]

            if _sort_by == "name":
                items.sort(key=lambda x: format_tag(x[0], _use_underscore))
            else:
                items.sort(key=lambda x: x[1], reverse=True)

            # min-score filter
            items = [(t, s) for t, s in items if s >= _min_score]

            # top-k
            if _top_k is not None and _top_k > 0:
                items = items[:_top_k]

            if not items:
                return "<i>No tags pass the filters.</i>", ""

            # table with tooltip
            rows = []
            for tag, score in items:
                m = meta[tag]
                tt = make_tooltip(m["wiki_body"])
                cat_badge = m["category_name"] or "?"
                link = (
                    f'<a href="https://danbooru.donmai.us/posts?tags={tag}"'
                    f' target="_blank">{tag}</a>'
                )
                display = format_tag(tag, _use_underscore)
                rows.append(
                    f'<tr data-tag="{tag}" title="{tt}">'
                    f"<td>{link}</td>"
                    f"<td>{display}</td>"
                    f"<td style='text-align:right'>{score:.4f}</td>"
                    f"<td><code>{cat_badge}</code></td>"
                    f"</tr>"
                )
            table = (
                '<table style="width:100%">'
                '<thead><tr>'
                '<th>Link</th><th>Tag</th>'
                '<th style="text-align:right">Score</th>'
                '<th>Category</th>'
                '</tr></thead>'
                '<tbody>' + "".join(rows) + '</tbody></table>'
                '<script>'
                '(function(){'
                'const input = document.querySelector("#hover-tag input, #hover-tag textarea");'
                'const output = document.querySelector("#hover-desc input, #hover-desc textarea");'
                'if(!input||!output)return;'
                'const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,"value").set;'
                'document.addEventListener("mouseover",function(e){'
                'const row=e.target.closest("[data-tag]");'
                'if(!row||row.dataset.descLoading)return;'
                'const tag=row.dataset.tag;'
                'if(row.title&&row.title!=="")return;'
                'row.dataset.descLoading="1";'
                'setter.call(input,tag);'
                'input.dispatchEvent(new Event("input",{bubbles:true}));'
                'input.dispatchEvent(new Event("change",{bubbles:true}));'
                'const ob=new MutationObserver(function(){'
                'if(output.value){'
                'row.title=output.value;'
                'delete row.dataset.descLoading;'
                'ob.disconnect();'
                '}'
                '});'
                'ob.observe(output,{attributes:true,attributeFilter:["value"]});'
                '})'
                '})();'
                '</script>'
            )

            # comma-separated
            csv = ", ".join(format_tag(t, _use_underscore) for t, _ in items)
            return table, csv

        # ── callbacks ──────────────────────────────────────────────────────

        def on_auth_change(user, key):
            # if the user typed a masked placeholder like "e7SG…Bag", skip
            if key and "…" in key:
                # re-apply env values
                from dotenv import load_dotenv
                import os
                load_dotenv()
                user = user or os.environ.get("DANBOORU_USER")
                key = os.environ.get("DANBOORU_API_KEY")
            api_configure(user, key)
            u, k_masked = get_auth_info()
            return (
                u or "⚠️ No user",
                f"✅ Credentials set" if u else "⚠️ No credentials",
            )

        for w in [api_user, api_key]:
            w.change(
                fn=on_auth_change,
                inputs=[api_user, api_key],
                outputs=[api_user, auth_status],
            )

        def on_analyze(image, url):
            if image is None and not url:
                return *refresh_results(
                    top_k.value, min_score.value,
                    sort_by.value, use_underscore.value,
                    categories.value,
                ), "⚠️ No image loaded."

            pil = image
            if pil is None and url:
                import requests as std_requests
                try:
                    resp = std_requests.get(url, timeout=30)
                    resp.raise_for_status()
                    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                    tmp.write(resp.content)
                    tmp.close()
                    pil = Image.open(tmp.name).convert("RGB")
                except Exception as exc:
                    return "<i>Error loading URL.</i>", "", f"❌ {exc}"

            state["current_image"] = pil
            all_logits = predict_all(pil, state["predictor"])
            state["all_logits"] = all_logits

            # enrich with category + wiki from API (cached)
            state["tag_metadata"] = enrich_tags(all_logits)

            table, csv = refresh_results(
                top_k.value, min_score.value,
                sort_by.value, use_underscore.value,
                categories.value,
            )
            n = len(state["tag_metadata"])
            cached = _CACHE.size()
            return table, csv, f"✅ {n} tags · {cached} cached"

        analyze_btn.click(
            fn=on_analyze,
            inputs=[image_input, url_input],
            outputs=[tag_table, tag_string, status],
        )

        url_input.submit(
            fn=on_analyze,
            inputs=[image_input, url_input],
            outputs=[tag_table, tag_string, status],
        )

        def on_clear():
            state["all_logits"] = None
            state["tag_metadata"] = None
            state["current_image"] = None
            return None, "", "<i>No results yet.</i>", "Cleared."

        clear_btn.click(
            fn=on_clear,
            inputs=[],
            outputs=[image_input, url_input, tag_table, tag_string, status],
        )

        def on_clear_cache():
            _CACHE.clear()
            return f"🧹 Cache cleared (0 entries)"

        clear_cache_btn.click(fn=on_clear_cache, inputs=[], outputs=[status])

        # param changes → refresh from cache
        for widget in [top_k, min_score, sort_by, use_underscore, categories]:
            widget.change(
                fn=refresh_results,
                inputs=[top_k, min_score, sort_by, use_underscore, categories],
                outputs=[tag_table, tag_string],
            )

    return app


def launch(checkpoint: str, share: bool = False, **kwargs):
    predictor = Predictor(checkpoint)
    app = build_app(predictor)
    app.launch(share=share, **kwargs)
