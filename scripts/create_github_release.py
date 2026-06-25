#!/usr/bin/env python3
"""Create a GitHub Release and upload deployment artifacts.

Usage:
    set GITHUB_TOKEN=ghp_xxx
    python scripts/create_github_release.py

Requires ``requests``.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

REPO = "Slayer025/pubmed-graphrag-v2"
TAG = "v2.1-hnsw"
TITLE = "Phase 6 HNSW Artifacts"
NOTES = (
    "Multi-index embeddings (semantic, fixed, sentence), graph CSVs, and "
    "pre-built HNSW approximate-nearest-neighbor indexes for Phase 6 deployment."
)

# Order matches bootstrap_artifacts.py _CACHE_LOGICAL_PATHS, with HNSW indexes appended.
ASSETS: list[tuple[str, str]] = [
    ("data/chunks/chunks_semantic.jsonl.gz", "chunks_semantic.jsonl.gz"),
    ("data/chunks/chunks_fixed.jsonl.gz", "chunks_fixed.jsonl.gz"),
    ("data/chunks/chunks_sentence.jsonl.gz", "chunks_sentence.jsonl.gz"),
    ("data/embeddings/semantic_embeddings.npy", "semantic_embeddings.npy"),
    ("data/embeddings/fixed_embeddings.npy", "fixed_embeddings.npy"),
    ("data/embeddings/sentence_embeddings.npy", "sentence_embeddings.npy"),
    ("data/graph/mentions.csv", "mentions.csv"),
    ("data/graph/has_chunk.csv", "has_chunk.csv"),
    ("data/graph/entities.csv", "entities.csv"),
    ("data/hnsw/semantic_index.bin", "semantic_index.bin"),
    ("data/hnsw/semantic_chunk_ids.json", "semantic_chunk_ids.json"),
    ("data/hnsw/fixed_index.bin", "fixed_index.bin"),
    ("data/hnsw/fixed_chunk_ids.json", "fixed_chunk_ids.json"),
    ("data/hnsw/sentence_index.bin", "sentence_index.bin"),
    ("data/hnsw/sentence_chunk_ids.json", "sentence_chunk_ids.json"),
    ("data/hnsw/manifest.json", "manifest.json"),
]


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def _get_token() -> str:
    token = os.environ.get("GITHUB_TOKEN", os.environ.get("GH_TOKEN", "")).strip()
    if not token:
        logger.error("GITHUB_TOKEN or GH_TOKEN environment variable is required.")
        sys.exit(1)
    return token


def _headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _get_release(token: str) -> dict | None:
    url = f"https://api.github.com/repos/{REPO}/releases/tags/{TAG}"
    resp = requests.get(url, headers=_headers(token), timeout=30)
    if resp.status_code == 200:
        return resp.json()
    if resp.status_code == 404:
        return None
    logger.error("Failed to check existing release: %s %s", resp.status_code, resp.text)
    return None


def _create_release(token: str) -> dict:
    existing = _get_release(token)
    if existing is not None:
        logger.info("Release %s already exists.", TAG)
        return existing

    url = f"https://api.github.com/repos/{REPO}/releases"
    payload = {
        "tag_name": TAG,
        "name": TITLE,
        "body": NOTES,
        "draft": False,
        "prerelease": False,
    }
    resp = requests.post(url, headers=_headers(token), json=payload, timeout=30)
    if resp.status_code >= 400:
        logger.error("Failed to create release: %s %s", resp.status_code, resp.text)
        sys.exit(1)
    logger.info("Created release %s.", TAG)
    return resp.json()


def _upload_asset(upload_url: str, token: str, local_path: Path, remote_name: str) -> bool:
    if not local_path.is_file():
        logger.error("Local file not found: %s", local_path)
        return False

    size_mb = local_path.stat().st_size / (1024 * 1024)
    logger.info("Uploading %s -> %s (%.2f MB)", local_path, remote_name, size_mb)

    url = upload_url.replace("{?name,label}", f"?name={remote_name}")
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "Content-Type": "application/octet-stream",
    }
    with open(local_path, "rb") as handle:
        data = handle.read()

    for attempt in range(3):
        try:
            resp = requests.post(url, headers=headers, data=data, timeout=300)
            if resp.status_code in (201, 422):
                # 422 often means asset already exists.
                logger.info("Uploaded %s (status=%s).", remote_name, resp.status_code)
                return True
            logger.warning("Upload attempt %d failed for %s: %s %s", attempt + 1, remote_name, resp.status_code, resp.text)
            time.sleep(2 ** attempt)
        except Exception as exc:
            logger.warning("Upload attempt %d error for %s: %s", attempt + 1, remote_name, exc)
            time.sleep(2 ** attempt)

    logger.error("Failed to upload %s after retries.", remote_name)
    return False


def _list_assets(token: str, release_id: int) -> list[dict]:
    url = f"https://api.github.com/repos/{REPO}/releases/{release_id}/assets"
    resp = requests.get(url, headers=_headers(token), timeout=30)
    resp.raise_for_status()
    return resp.json()


def main() -> int:
    _configure_logging()
    token = _get_token()
    project_root = Path(__file__).resolve().parent.parent

    release = _create_release(token)
    upload_url = release["upload_url"]
    release_id = release["id"]

    all_ok = True
    for local, remote in ASSETS:
        local_path = project_root / local
        if not _upload_asset(upload_url, token, local_path, remote):
            all_ok = False

    if all_ok:
        logger.info("All assets uploaded successfully.")
    else:
        logger.error("Some assets failed to upload.")

    assets = _list_assets(token, release_id)
    logger.info("Release assets (%d):", len(assets))
    for asset in assets:
        size_mb = asset["size"] / (1024 * 1024)
        logger.info("  - %s (%.2f MB) -> %s", asset["name"], size_mb, asset["browser_download_url"])

    print(f"\nARTIFACT_BASE_URL=https://github.com/{REPO}/releases/download/{TAG}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
