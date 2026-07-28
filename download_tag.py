"""
Download all images for a tag from Danbooru.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from src.api import posts, get_auth_info


def _format_tags(post: dict) -> str:
    """Combine general + character tags into comma-separated labels, underscores -> spaces.
    Matches the format of infer.py's _save_labels."""
    general = (post.get("tag_string_general") or "").strip()
    character = (post.get("tag_string_character") or "").strip()
    combined = " ".join(p for p in (general, character) if p)
    tags = combined.replace("_", " ").split()
    return ", ".join(tags)


def _download_one(url: str, dest: Path, tags: str | None = None) -> bool:
    """Download a single file. Returns True on success.
    Writes to a temp file first, renames atomically to avoid partial files.
    If tags is provided, saves a .txt file with labels alongside the image.
    """
    try:
        from curl_cffi import requests

        resp = requests.get(url, impersonate="chrome", timeout=120)
        resp.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        tmp.write_bytes(resp.content)
        tmp.rename(dest)

        if tags:
            txt_path = dest.with_suffix(".txt")
            txt_path.write_text(tags)

        return True
    except Exception as exc:
        print(f"  FAIL: {dest.name} — {exc}", file=sys.stderr)
        return False


def download_tag(
    tag: str,
    output_dir: str = "downloads",
    workers: int = 4,
    limit: int | None = None,
    min_score: int | None = None,
):
    # check credentials
    user, masked = get_auth_info()
    if not user:
        print("ERROR: No Danbooru credentials found.", file=sys.stderr)
        print("Create a .env file with:", file=sys.stderr)
        print("  DANBOORU_USER=your_username", file=sys.stderr)
        print("  DANBOORU_API_KEY=your_api_key", file=sys.stderr)
        print("Get your API key at: https://danbooru.donmai.us/profile", file=sys.stderr)
        sys.exit(1)
    print(f"Authenticated as: {user} ({masked})")

    out = Path(output_dir) / tag.replace(" ", "_")
    out.mkdir(parents=True, exist_ok=True)
    # clean stale partials from previous runs
    for p in out.glob("*.tmp"):
        p.unlink()
    print(f"Output directory: {out}")

    # paginate through all posts
    query = tag
    if min_score is not None:
        query += f" score:>={min_score}"
    query += " order:id"

    page = 1
    all_posts = []
    print(f"Fetching posts for tag '{tag}' …")
    while True:
        print(f"  Page {page} …", end=" ", flush=True)
        batch = posts(query, limit=200, page=page)
        if not batch:
            print("empty page, done.")
            break
        print(f"{len(batch)} posts")
        all_posts.extend(batch)
        if limit and len(all_posts) >= limit:
            all_posts = all_posts[:limit]
            print(f"Reached limit of {limit}")
            break
        if len(batch) < 200:
            print("Last page reached.")
            break
        page += 1

    print(f"\nTotal: {len(all_posts)} posts for '{tag}'")

    # collect unique file URLs with tags
    urls: list[tuple[str, Path, str]] = []
    skipped = 0
    no_url = 0
    for p in all_posts:
        file_url = p.get("file_url")
        if not file_url:
            no_url += 1
            continue
        ext = Path(file_url).suffix or ".jpg"
        img_id = p["id"]
        dest = out / f"{img_id}{ext}"
        txt_dest = dest.with_suffix(".txt")
        if dest.exists() and txt_dest.exists():
            skipped += 1
            continue
        tag_labels = _format_tags(p)
        urls.append((file_url, dest, tag_labels))

    print(f"  {len(urls)} to download, {skipped} already exist, {no_url} without file_url")

    if not urls:
        print("All up to date.")
        return

    # download
    ok = 0
    fail = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_download_one, url, dest, tags): (dest, tags) for url, dest, tags in urls}
        pbar = tqdm(total=len(futs), unit="img", desc="Downloading")
        for fut in as_completed(futs):
            if fut.result():
                ok += 1
            else:
                fail += 1
            pbar.update(1)
            pbar.set_postfix(ok=ok, fail=fail)
        pbar.close()

    print(f"\nDone — {ok} OK, {fail} failed")
    print(f"Saved to: {out.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download all images for a Danbooru tag"
    )
    parser.add_argument("tag", help="Tag name to download")
    parser.add_argument(
        "-o", "--output", default="downloads",
        help="Output directory (default: downloads/)",
    )
    parser.add_argument(
        "-w", "--workers", type=int, default=4,
        help="Download threads (default: 4)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max images to download",
    )
    parser.add_argument(
        "--min-score", type=int, default=None,
        help="Minimum post score (e.g. 100 for score>=100)",
    )
    args = parser.parse_args()
    download_tag(args.tag, args.output, args.workers, args.limit, args.min_score)
