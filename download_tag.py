"""
Download all images for a tag from Danbooru.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from src.api import posts


def _download_one(url: str, dest: Path) -> bool:
    """Download a single file. Returns True on success."""
    try:
        from curl_cffi import requests

        resp = requests.get(url, impersonate="chrome", timeout=120)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        return True
    except Exception:
        return False


def download_tag(
    tag: str,
    output_dir: str = "downloads",
    workers: int = 4,
    limit: int | None = None,
):
    out = Path(output_dir) / tag.replace(" ", "_")
    out.mkdir(parents=True, exist_ok=True)

    # get page 1 to find total pages
    page = 1
    all_posts = []
    while True:
        batch = posts(f"{tag} order:id", limit=200)
        if not batch:
            break
        all_posts.extend(batch)
        if limit and len(all_posts) >= limit:
            all_posts = all_posts[:limit]
            break
        if len(batch) < 200:
            break
        page += 1

    print(f"Found {len(all_posts)} posts for '{tag}'")

    # collect URLs
    urls: list[tuple[str, Path]] = []
    for p in all_posts:
        file_url = p.get("file_url")
        if not file_url:
            continue
        ext = Path(file_url).suffix or ".jpg"
        # use md5 or id as filename
        img_id = p["id"]
        dest = out / f"{img_id}{ext}"
        if dest.exists():
            continue
        urls.append((file_url, dest))

    print(f"To download: {len(urls)} (already have {len(all_posts) - len(urls)})")

    if not urls:
        print("All up to date.")
        return

    ok = 0
    fail = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_download_one, url, dest): dest for url, dest in urls}
        pbar = tqdm(total=len(futs), unit="img", desc="Downloading")
        for fut in as_completed(futs):
            if fut.result():
                ok += 1
            else:
                fail += 1
                dest = futs[fut]
                tqdm.write(f"  FAILED: {dest.name}")
            pbar.update(1)
            pbar.set_postfix(ok=ok, fail=fail)
        pbar.close()

    print(f"Done — {ok} OK, {fail} failed, saved to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download all images for a Danbooru tag")
    parser.add_argument("tag", help="Tag name to download")
    parser.add_argument("-o", "--output", default="downloads", help="Output directory")
    parser.add_argument("-w", "--workers", type=int, default=4, help="Download threads")
    parser.add_argument("--limit", type=int, default=None, help="Max images")
    args = parser.parse_args()
    download_tag(args.tag, args.output, args.workers, args.limit)
