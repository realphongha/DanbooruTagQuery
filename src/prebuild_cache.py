"""
Pre-populate tag cache from qdlabs/danbooru-tags (Hugging Face).

Fills category for every tag in tag_to_id.json — no API calls, fast.
Wiki body is lazy-loaded on hover in the UI.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from .cache import TagCache

HF_DATASET = "qdlabs/danbooru-tags"


def _load_category_map() -> dict[str, int]:
    ds = load_dataset(HF_DATASET, split="train")
    return {row["name"]: row["category"] for row in ds}


def prebuild(tag_to_id_path: str = "data/tag_to_id.json"):
    tag_to_id = json.loads(Path(tag_to_id_path).read_text())
    tags = sorted(tag_to_id.keys(), key=lambda t: tag_to_id[t])
    cache = TagCache()

    print(f"Loading category map from {HF_DATASET} …")
    cat_map = _load_category_map()
    found = sum(1 for t in tags if t in cat_map)
    print(f"Tags: {len(tags)} total, {found} have category in dataset")

    todo = []
    already = 0
    for tag in tags:
        cached = cache.get(tag)
        if cached is not None and cached[0] is not None:
            already += 1
            continue
        cat = cat_map.get(tag)
        existing_wiki = cached[1] if cached else None
        todo.append((tag, cat, existing_wiki))

    print(f"Already cached: {already}, to insert: {len(todo)}")
    if todo:
        for _ in tqdm([0], desc="Inserting", disable=True):
            cache.bulk_set(todo)
        print(f"Inserted {len(todo)} tags in one batch")

    # NOTE: tags not found in HF dataset have category=None in cache
    print(f"Done — {cache.size()} total cached")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-populate tag cache from HF dataset (no API calls)"
    )
    parser.add_argument(
        "--tag-map", default="data/tag_to_id.json",
        help="Path to tag_to_id.json",
    )
    args = parser.parse_args()
    prebuild(args.tag_map)
