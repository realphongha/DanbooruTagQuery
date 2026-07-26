"""Download helpers for model weights and source files."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urljoin

import requests
from huggingface_hub import hf_hub_download
from tqdm import tqdm


def download_url(url: str, dest: Path, desc: str = "") -> None:
    """Download a single file from *url* to *dest* with progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, stream=True, timeout=300)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    chunk_size = 1024 * 1024  # 1 MiB
    with tqdm(total=total, unit="B", unit_scale=True, desc=desc or dest.name) as pbar:
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


def download_hf_file(repo_id: str, filename: str, dest_dir: Path) -> Path:
    """Download a file from HuggingFace Hub to *dest_dir*, return local path."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(dest_dir),
                           local_dir_use_symlinks=False)
    return Path(path)


def download_github_dir(repo: str, subdir: str, dest_dir: Path) -> None:
    """Download an entire subdirectory from a GitHub repo via archive extraction.

    Uses ``https://github.com/{repo}/archive/master.zip`` and extracts
    ``{repo_dir}/{subdir}/*`` into *dest_dir*.
    """
    import zipfile
    import io

    repo_dir_name = repo.split("/")[-1]  # e.g. "ML-Danbooru"
    archive_url = f"https://github.com/{repo}/archive/master.zip"
    prefix = f"{repo_dir_name}-master/{subdir}/"

    resp = requests.get(archive_url, stream=True, timeout=300)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    chunk_size = 8 * 1024 * 1024
    with tqdm(total=total, unit="B", unit_scale=True, desc=f"{subdir}/") as pbar:
        chunks = []
        for chunk in resp.iter_content(chunk_size=chunk_size):
            if chunk:
                chunks.append(chunk)
                pbar.update(len(chunk))
    data = b"".join(chunks)

    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for name in zf.namelist():
            if not name.startswith(prefix) or name == prefix:
                continue
            # Skip directory entries (trailing /)
            if name.endswith("/"):
                continue
            relative = Path(name).relative_to(prefix)
            target = dest_dir / subdir / relative
            # If a previous buggy run left a file where a directory should go, remove it
            if target.parent.exists() and not target.parent.is_dir():
                target.parent.unlink()
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(zf.read(name))
