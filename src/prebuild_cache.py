"""
Pre-populate tag category lookup from HF dataset + Danbooru API.

Fills category for every tag in tag_to_id.json.
Writes both tag_category.db (SQLite) and tag_category.json.

Usage:
  python -m src.prebuild_cache                    # uses data/tag_to_id.json
  python -m src.prebuild_cache --tag-map path/to/tag_to_id.json
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from .api import tag_info
from .cache import TagCache

HF_DATASET = "qdlabs/danbooru-tags"
API_WORKERS = 2


def _load_category_map() -> dict[str, int]:
    ds = load_dataset(HF_DATASET, split="train")
    return {row["name"]: row["category"] for row in ds}


def _api_lookup(tag: str) -> int | None:
    """Fetch tag category from Danbooru API. Returns None on failure."""
    try:
        info = tag_info(tag)
        if info is not None:
            return info["category"]
    except Exception as exc:
        print(f"  API error for '{tag}': {exc}")
    return None


def _parallel_api_lookup(tags: list[str]) -> dict[str, int]:
    """Fetch categories in parallel via thread pool."""
    results: dict[str, int] = {}

    def fetch(tag: str) -> tuple[str, int]:
        cat = _api_lookup(tag)
        return tag, cat if cat is not None else 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=API_WORKERS) as pool:
        futs = [pool.submit(fetch, tag) for tag in tags]
        for fut in tqdm(
            concurrent.futures.as_completed(futs),
            total=len(futs),
            desc="API",
        ):
            tag, cat = fut.result()
            results[tag] = cat

    return results


def _load_existing(out_dir: Path, tags: list[str]) -> dict[str, int]:
    """Load already-known categories from existing JSON & SQLite files."""
    existing: dict[str, int] = {}

    json_path = out_dir / "tag_category.json"
    if json_path.exists():
        existing.update(json.loads(json_path.read_text()))

    db_path = out_dir / "tag_category.db"
    if db_path.exists():
        cache = TagCache(str(db_path))
        for tag in tags:
            if tag not in existing:
                cached = cache.get(tag)
                if cached and cached[0] is not None:
                    existing[tag] = cached[0]

    return existing


def prebuild(tag_to_id_path: str | Path = "data/tag_to_id.json"):
    tag_to_id_path = Path(tag_to_id_path)
    tag_to_id = json.loads(tag_to_id_path.read_text())
    tags = sorted(tag_to_id.keys(), key=lambda t: tag_to_id[t])
    out_dir = tag_to_id_path.parent

    # load existing (from previous runs)
    existing = _load_existing(out_dir, tags)

    # tags still needing a category
    missing = [t for t in tags if t not in existing]
    print(f"Tags: {len(tags)} total, {len(existing)} already cached, {len(missing)} missing")

    if not missing:
        print("All tags have categories — nothing to do.")
        return

    # query HF dataset
    print(f"Loading category map from {HF_DATASET} …")
    cat_map = _load_category_map()
    found_in_ds = sum(1 for t in missing if t in cat_map)
    print(f"  {found_in_ds}/{len(missing)} found in dataset")

    lookup: dict[str, int] = {}
    api_needed: list[str] = []
    for tag in missing:
        cat = cat_map.get(tag)
        if cat is not None:
            lookup[tag] = cat
        else:
            api_needed.append(tag)

    # API fallback for rest
    if api_needed:
        print(f"  {len(api_needed)} not in dataset, fetching from Danbooru API ({API_WORKERS} workers) …")
        lookup.update(_parallel_api_lookup(api_needed))

    api_found = sum(1 for t in api_needed if lookup.get(t, 0) != 0)
    defaulted = len(api_needed) - api_found
    if defaulted:
        print(f"  {defaulted} defaulted to general (API lookup failed)")

    merge = {**existing, **lookup}

    # ── write JSON ──────────────────────────────────────────────────────
    json_path = out_dir / "tag_category.json"
    json_path.write_text(json.dumps(merge, indent=0) + "\n")
    print(f"Written {json_path} ({len(merge)} tags)")

    # ── write SQLite ────────────────────────────────────────────────────
    cache = TagCache(str(out_dir / "tag_category.db"))
    cache.bulk_set([(tag, merge[tag], None) for tag in tags])
    print(f"Written {out_dir / 'tag_category.db'} ({cache.size()} entries)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-build tag category lookup (dataset + API fallback)"
    )
    parser.add_argument(
        "--tag-map", default="data/tag_to_id.json",
        help="Path to tag_to_id.json",
    )
    args = parser.parse_args()
    prebuild(args.tag_map)
