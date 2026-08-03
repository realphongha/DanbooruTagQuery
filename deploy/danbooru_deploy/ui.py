"""Gradio web UI for DanbooruTagQuery inference.

Uses tag_category.json sidecar — no SQLite, no API calls. Supports an
optional runtime model switcher (HF hub variants) and a harmless ZeroGPU
marker for Hugging Face Spaces (MockSpaces stub when ``spaces`` absent).
"""

from __future__ import annotations

import tempfile
import time

import gradio as gr
from PIL import Image

from .core import CATEGORY_MAP, get_category_name

try:
    import spaces
except ImportError:
    class MockSpaces:
        def GPU(self, func=None, **kwargs):
            if func is not None and callable(func):
                # Used as @spaces.GPU
                return func

            # Used as @spaces.GPU(...)
            def decorator(f):
                return f
            return decorator

    spaces = MockSpaces()

DEFAULT_TOP_K = None
DEFAULT_MIN_SCORE = 0.2

_CATEGORY_NAMES = sorted(CATEGORY_MAP.values())


def format_tag(tag: str, use_underscore: bool) -> str:
    return tag if use_underscore else tag.replace("_", " ")


def enrich_tags(
    tags_scores: list[tuple[str, float]], cat_map: dict[str, int]
) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for tag, score in tags_scores:
        result[tag] = {
            "score": score,
            "category": cat_map.get(tag, 0),
            "category_name": get_category_name(cat_map, tag),
        }
    return result


def build_app(
    predict_fn,
    cat_map: dict[str, int],
    model_choices: list[str] | None = None,
    get_model_predict_fn=None,
    hf_repo: str | None = None,
):
    """Build Gradio app.

    Parameters
    ----------
    predict_fn : callable
        ``predict_fn(pil_image: PIL.Image) -> list[tuple[str, float]]``
        Returns [(tag, score), …] sorted by score descending.
    cat_map : dict[str, int]
        tag -> category_id mapping from tag_category.json.
    model_choices : list[str] | None
        Model variant names for the switcher dropdown. None → no switcher.
    get_model_predict_fn : callable | None
        ``get_model_predict_fn(variant) -> (predict_fn, cat_map)`` used by the
        model switcher at runtime. Required when model_choices is given.
    hf_repo : str | None
        Model card link target (defaults to no card when None).
    """
    state = {
        "all_logits": None,
        "tag_metadata": None,
        "current_image": None,
        "predict_fn": predict_fn,
        "cat_map": cat_map,
    }

    css = """
    #csv-wrap { position: relative; }
    #copy-csv-btn { position: absolute; top: 4px; right: 4px; z-index: 10;
                    min-width: 0; padding: 0 6px; height: 24px;
                    font-size: 13px; line-height: 24px; }

    /* wider layout — don't cap the container */
    .gradio-container { max-width: 100% !important; }
    /* image box fills the left column, matching the params box height on the
       right; object-fit: contain letterboxes (no crop) */
    #dq-image { display: flex; flex-direction: column; }
    #dq-image .image-container,
    #dq-image .image-preview-container {
        flex: 1;
        min-height: 320px;
    }
    #dq-image .image-container img {
        object-fit: contain !important;   /* whole image, padded, no crop */
        width: 100% !important;
        height: 100% !important;
        background: #000;
    }
    """

    with gr.Blocks(title="DanbooruTagQuery", theme=gr.themes.Soft(), css=css) as app:
        gr.Markdown("# 🏷️ DanbooruTagQuery")

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    label="Image",
                    type="pil",
                    sources=["upload", "clipboard"],
                    elem_id="dq-image",
                )
                url_input = gr.Textbox(
                    label="Image URL",
                    placeholder="Paste image URL and press Enter",
                )
                with gr.Row():
                    analyze_btn = gr.Button("🔍 Analyze", variant="primary", scale=2)
                    clear_btn = gr.Button("🗑️ Clear", scale=1)
                    if hf_repo:
                        gr.Markdown(
                            f'[📄 **Model Card**](https://huggingface.co/{hf_repo})'
                        )

                model_dropdown = None
                model_status = None
                if model_choices:
                    gr.Markdown("### 🤖 Model")
                    model_dropdown = gr.Dropdown(
                        choices=model_choices,
                        value=model_choices[0] if model_choices else None,
                        label="Model variant",
                        interactive=True,
                    )
                    model_status = gr.Markdown("Ready")

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
            fn = state["predict_fn"]
            if fn is None:
                return "<i>No model loaded.</i>", "", "❌ No model loaded."

            t0 = time.time()
            all_logits = fn(pil)
            state["all_logits"] = all_logits
            state["tag_metadata"] = enrich_tags(all_logits, state["cat_map"])

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
            return None, "", "<i>No results yet.</i>", "", "Cleared.", "", ""

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

        # HF ZeroGPU requires @spaces.GPU somewhere in code (harmless stub)
        gpu_hidden_state = gr.State(value=None)

        @spaces.GPU
        def _gpu_dummy():
            return None

        gpu_hidden_state.change(
            fn=_gpu_dummy, inputs=[gpu_hidden_state], outputs=[gpu_hidden_state]
        )

        # ── model switcher ────────────────────────────────────────────────

        if model_dropdown is not None and get_model_predict_fn is not None:

            def on_model_change(variant):
                if not variant:
                    return "⚠️ No model selected"
                try:
                    fn, cat = get_model_predict_fn(variant)
                    state["predict_fn"] = fn
                    state["cat_map"] = cat
                    state["all_logits"] = None
                    state["tag_metadata"] = None
                    return f"✅ Switched to {variant}"
                except Exception as exc:
                    return f"❌ Failed to load model: {exc}"

            model_dropdown.change(
                fn=on_model_change,
                inputs=[model_dropdown],
                outputs=[model_status],
            )

    return app


def launch(
    predict_fn,
    cat_map: dict[str, int],
    share: bool = False,
    model_choices: list[str] | None = None,
    get_model_predict_fn=None,
    hf_repo: str | None = None,
    **kwargs,
):
    """Launch Gradio UI with a predict function and category map."""
    app = build_app(
        predict_fn,
        cat_map,
        model_choices=model_choices,
        get_model_predict_fn=get_model_predict_fn,
        hf_repo=hf_repo,
    )
    app.launch(share=share, **kwargs)