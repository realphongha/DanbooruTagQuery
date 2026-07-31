"""
Gradio web UI for DanbooruTagQuery inference.

Uses tag_category.json sidecar — no SQLite, no API calls.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

# ── constants ────────────────────────────────────────────────────────────────

DEFAULT_TOP_K = None
DEFAULT_MIN_SCORE = 0.2

CATEGORY_MAP = {
    0: "general",
    1: "artist",
    3: "copyright",
    4: "character",
    5: "meta",
}

_CATEGORY_NAMES = sorted(CATEGORY_MAP.values())


def get_category_name(cat_map: dict[str, int], tag: str) -> str:
    cat_id = cat_map.get(tag, 0)
    return CATEGORY_MAP.get(cat_id, "general")


def format_tag(tag: str, use_underscore: bool) -> str:
    return tag if use_underscore else tag.replace("_", " ")


# ── tag metadata enrichment (static cat_map) ────────────────────────────────

def enrich_tags(
    tags_scores: list[tuple[str, float]], cat_map: dict[str, int]
) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for tag, score in tags_scores:
        result[tag] = {
            "score": score,
            "category_name": get_category_name(cat_map, tag),
        }
    return result


# ── Gradio app ───────────────────────────────────────────────────────────────

def build_app(predict_fn, cat_map: dict[str, int]):
    """Build Gradio app.

    Parameters
    ----------
    predict_fn : callable
        ``predict_fn(pil_image: PIL.Image) -> list[tuple[str, float]]``
        Returns [(tag, score), …] sorted by score descending.
    cat_map : dict[str, int]
        tag -> category_id mapping from tag_category.json.
    """
    state = {
        "all_logits": None,
        "tag_metadata": None,
        "current_image": None,
    }

    css = """
    #csv-wrap { position: relative; }
    #copy-csv-btn { position: absolute; top: 4px; right: 4px; z-index: 10;
                    min-width: 0; padding: 0 6px; height: 24px;
                    font-size: 13px; line-height: 24px; }
    """

    with gr.Blocks(title="DanbooruTagQuery", theme=gr.themes.Soft(), css=css) as app:
        gr.Markdown("# 🏷️ DanbooruTagQuery")

        with gr.Row():
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

        # ── outputs ──
        with gr.Tabs():
            with gr.TabItem("📋 Tag list"):
                tag_table = gr.HTML(label="Tags")
            with gr.TabItem("📝 Comma-separated"):
                with gr.Column(elem_id="csv-wrap"):
                    tag_string = gr.Textbox(label="Tags", lines=6, elem_id="csv-text")
                    copy_btn = gr.Button("📋", elem_id="copy-csv-btn")

        # ── status row ──
        with gr.Row():
            status = gr.Markdown("Ready. Load an image and click **Analyze**.")

        # ── tag score query ──
        gr.Markdown("### 🔍 Tag score query")
        with gr.Row():
            tag_query = gr.Textbox(
                label="Search tag",
                placeholder="Type to search…",
                scale=3,
            )
        tag_query_output = gr.HTML(label="Results")

        # ── helpers ────────────────────────────────────────────────────────

        def refresh_results(
            _top_k, _min_score, _sort_by, _use_underscore, _categories,
        ):
            if state["all_logits"] is None or state["tag_metadata"] is None:
                return "<i>No results yet.</i>", ""

            meta = state["tag_metadata"]
            all_tags = list(meta.keys())
            if _categories:
                all_tags = [
                    t for t in all_tags
                    if meta[t]["category_name"] in _categories
                ]

            items = [(t, meta[t]["score"]) for t in all_tags]

            if _sort_by == "name":
                items.sort(key=lambda x: format_tag(x[0], _use_underscore))
            else:
                items.sort(key=lambda x: x[1], reverse=True)

            items = [(t, s) for t, s in items if s >= _min_score]

            if _top_k is not None and _top_k > 0:
                items = items[:_top_k]

            if not items:
                return "<i>No tags pass the filters.</i>", ""

            rows = []
            for tag, score in items:
                m = meta[tag]
                link = (
                    f'<a href="https://danbooru.donmai.us/posts?tags={tag}"'
                    f' target="_blank">{tag}</a>'
                )
                display = format_tag(tag, _use_underscore)
                rows.append(
                    f"<tr>"
                    f"<td>{link}</td>"
                    f"<td>{display}</td>"
                    f"<td style='text-align:right'>{score:.4f}</td>"
                    f"<td><code>{m['category_name']}</code></td>"
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
            )

            csv = ", ".join(format_tag(t, _use_underscore) for t, _ in items)
            return table, csv

        # ── callbacks ──────────────────────────────────────────────────────

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
            import time
            t0 = time.time()
            all_logits = predict_fn(pil)
            state["all_logits"] = all_logits

            state["tag_metadata"] = enrich_tags(all_logits, cat_map)

            table, csv = refresh_results(
                top_k.value, min_score.value,
                sort_by.value, use_underscore.value,
                categories.value,
            )
            elapsed = time.time() - t0
            n = len(state["tag_metadata"])
            return table, csv, f"✅ {n} tags · {elapsed:.2f}s"

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
            return None, "", "<i>No results yet.</i>", "Cleared.", "", ""

        clear_btn.click(
            fn=on_clear,
            inputs=[],
            outputs=[image_input, url_input, tag_table, tag_string, status, tag_query, tag_query_output],
        )

        # param changes → refresh
        for widget in [top_k, min_score, sort_by, use_underscore, categories]:
            widget.change(
                fn=refresh_results,
                inputs=[top_k, min_score, sort_by, use_underscore, categories],
                outputs=[tag_table, tag_string],
            )

        # tag score query (dynamic search)
        def query_tag_score(query):
            meta = state.get("tag_metadata")
            if not meta or not query:
                return ""
            query_l = query.lower()
            matches = sorted(
                [(t, meta[t]["score"]) for t in meta if query_l in t.lower()],
                key=lambda x: x[1], reverse=True,
            )[:20]
            if not matches:
                return "<i>No matching tags.</i>"
            rows = "".join(
                f"<tr><td>{t}</td><td>{s:.4f}</td>"
                f"<td><code>{meta[t]['category_name']}</code></td></tr>"
                for t, s in matches
            )
            return (f"<table style='width:100%'>"
                    f"<tr><th>Tag</th><th>Score</th><th>Category</th></tr>"
                    f"{rows}</table>")

        tag_query.change(
            fn=query_tag_score,
            inputs=[tag_query],
            outputs=[tag_query_output],
        )

        # copy button JS
        copy_btn.click(
            fn=lambda: None,
            inputs=[],
            outputs=[],
            js="""() => {
                const tb = document.querySelector('#csv-text textarea');
                if (tb) { navigator.clipboard.writeText(tb.value); }
            }"""
        )

    return app


def launch(predict_fn, cat_map: dict[str, int], share: bool = False, **kwargs):
    """Launch Gradio UI with a predict function and category map."""
    app = build_app(predict_fn, cat_map)
    app.launch(share=share, **kwargs)
