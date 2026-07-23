import argparse
import csv
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
_DANBOORU_USER = os.environ.get("DANBOORU_USER")
_DANBOORU_API_KEY = os.environ.get("DANBOORU_API_KEY")
if not _DANBOORU_USER or not _DANBOORU_API_KEY:
    print("ERROR: DANBOORU_USER and DANBOORU_API_KEY environment variables must be set.")
    print("Get your API key at: https://danbooru.donmai.us/profile")
    exit(1)

_SESSION = requests.Session()
_SESSION.auth = (_DANBOORU_USER, _DANBOORU_API_KEY)
_SESSION.headers.update({
    "User-Agent": f"DanbooruTagCLIP/1.0 (by {_DANBOORU_USER} on Danbooru)",
    "Referer": "https://danbooru.donmai.us/",
})


def download_one(row, output_dir, retries):
    suffix = Path(urlparse(row.image_url).path).suffix.lower() or ".jpg"
    path = output_dir / f"{int(row.id)}{suffix}"
    if path.exists() and path.stat().st_size > 0: return "skipped", row.id, row.image_url, ""
    for attempt in range(retries + 1):
        try:
            response = _SESSION.get(row.image_url, timeout=(5, 30))
            if response.status_code == 200 and response.content:
                temporary = path.with_suffix(path.suffix + ".part")
                temporary.write_bytes(response.content)
                if temporary.stat().st_size > 0: os.replace(temporary, path); return "downloaded", row.id, row.image_url, ""
                temporary.unlink(missing_ok=True)
                return "failed", row.id, row.image_url, "empty response"
            retryable = response.status_code >= 500
            reason = f"HTTP {response.status_code}"
        except requests.RequestException as error:
            retryable, reason = True, str(error)
        if not retryable or attempt == retries: return "failed", row.id, row.image_url, reason
        time.sleep(2 ** attempt)


def update_parquet(parquet, workers=32, retries=3):
    parquet = Path(parquet); output_dir = Path("data/images") / parquet.stem; output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(parquet); results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(download_one, row, output_dir, retries) for row in frame.itertuples()]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading images"):
            status, identifier, url, reason = future.result()
            if status == "failed":
                print(f"FAILED [{identifier}] {url} -> {reason}")
            results.append((status, identifier, url, reason))
    paths = {int(identifier): str(output_dir / f"{int(identifier)}{Path(urlparse(url).path).suffix.lower() or '.jpg'}") for _, identifier, url, _ in results}
    frame["image_path"] = frame.id.map(paths)
    frame.to_parquet(parquet, index=False, compression="zstd")
    failed = [(identifier, url, reason) for status, identifier, url, reason in results if status == "failed"]
    if failed:
        with Path("failed_downloads.csv").open("w", newline="") as file:
            writer = csv.writer(file); writer.writerow(["id", "url", "reason"]); writer.writerows(failed)
    print(f"Downloaded : {sum(x[0] == 'downloaded' for x in results)}\nSkipped : {sum(x[0] == 'skipped' for x in results)}\nFailed : {len(failed)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("parquet", nargs="+", type=Path); parser.add_argument("--workers", type=int, default=32); parser.add_argument("--retries", type=int, default=3); args = parser.parse_args()
    try:
        for parquet in args.parquet: update_parquet(parquet, args.workers, args.retries)
    except KeyboardInterrupt: print("\nDownload interrupted; completed files are preserved.")
