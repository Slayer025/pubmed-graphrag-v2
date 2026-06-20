"""Artifact loader for Phase 1/2 data files."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import requests

from src.config import AppConfig
from src.embeddings import normalize_embeddings
from src.infrastructure.storage.csv_loader import load_csv
from src.storage import iter_jsonl_gz

logger = logging.getLogger(__name__)

MIN_VALID_SIZE_DEFAULT = 1024
LOCK_RETRY_MS = 300
LOCK_RETRY_ATTEMPTS = 10

# Base URL for Streamlit Cloud / fresh-container bootstrap.
# Override via ARTIFACT_BASE_URL env var (no trailing slash required).
ARTIFACT_BASE_URL = os.environ.get("ARTIFACT_BASE_URL", "TODO_SET_THIS")

_ARTIFACT_REMOTE_NAMES: dict[str, str] = {
    "data/chunks/chunks_semantic.jsonl.gz": "chunks_semantic.jsonl.gz",
    "data/embeddings/semantic_embeddings.npy": "semantic_embeddings.npy",
    "data/graph/mentions.csv": "mentions.csv",
    "data/graph/entities.csv": "entities.csv",
    "data/graph/has_chunk.csv": "has_chunk.csv",
}

_MIN_VALID_SIZES: dict[str, int] = {
    "data/chunks/chunks_semantic.jsonl.gz": MIN_VALID_SIZE_DEFAULT,
    "data/embeddings/semantic_embeddings.npy": 1024 * 1024,
    "data/graph/mentions.csv": MIN_VALID_SIZE_DEFAULT,
    "data/graph/entities.csv": MIN_VALID_SIZE_DEFAULT,
    "data/graph/has_chunk.csv": MIN_VALID_SIZE_DEFAULT,
}


@dataclass(frozen=True)
class _ArtifactPaths:
    """All filesystem paths for one artifact, resolved once."""

    base: Path
    ready: Path
    lock: Path
    part: Path


def _repo_root() -> Path:
    """Return repository root without relying on process cwd."""
    return Path(__file__).resolve().parents[3]


def _resolve_artifact_path(path: Path | str) -> Path:
    """Normalize to a single absolute resolved path (no relative paths)."""
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = _repo_root() / candidate
    return candidate.resolve()


def _artifact_paths(path: Path | str) -> _ArtifactPaths:
    """Build consistent resolved paths for base, .ready, .lock, and .part."""
    base = _resolve_artifact_path(path)
    return _ArtifactPaths(
        base=base,
        ready=Path(f"{base}.ready").resolve(),
        lock=Path(f"{base}.lock").resolve(),
        part=Path(f"{base}.part").resolve(),
    )


def _relative_key(base: Path) -> str:
    return base.relative_to(_repo_root()).as_posix()


def _effective_min_size(base: Path, min_size: int = MIN_VALID_SIZE_DEFAULT) -> int:
    return _MIN_VALID_SIZES.get(_relative_key(base), min_size)


def _read_ready_content(ready_path: Path) -> str | None:
    if not ready_path.exists():
        return None
    try:
        return ready_path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _artifact_is_ready(path: Path | str, min_size: int = MIN_VALID_SIZE_DEFAULT, *, log: bool = True) -> bool:
    """Single source of truth for artifact readiness."""
    paths = _artifact_paths(path)
    required_size = _effective_min_size(paths.base, min_size)
    reason = "ok"
    ready = False

    if not paths.base.exists():
        reason = "missing file"
    elif not paths.base.is_file():
        reason = "not a file"
    else:
        size = os.path.getsize(paths.base)
        if size <= 0:
            reason = "empty file"
        elif size < required_size:
            reason = f"size {size} < minimum {required_size}"
        elif not paths.ready.exists():
            reason = f"missing ready file at {paths.ready}"
        elif _read_ready_content(paths.ready) != "ok":
            reason = f"invalid ready content at {paths.ready}"
        else:
            ready = True

    if log:
        status = "TRUE" if ready else "FALSE"
        logger.info("READY CHECK: %s → %s (%s)", paths.base, status, reason)
        logger.info("ARTIFACT READY = %s", status)
        print(f"READY CHECK: {paths.base} → {status}", flush=True)
        print(f"ARTIFACT READY = {status}", flush=True)

    return ready


def _log_skip_download(paths: _ArtifactPaths) -> None:
    print("SKIP DOWNLOAD TRIGGERED", flush=True)
    print("SKIP DOWNLOAD", flush=True)
    logger.info("SKIP DOWNLOAD TRIGGERED for %s", paths.base)
    logger.info("SKIP DOWNLOAD: artifact validated for %s", paths.base)


def _remove_invalid_artifact(paths: _ArtifactPaths) -> None:
    """Remove corrupt or partial artifacts before a fresh download."""
    if _artifact_is_ready(paths.base, log=False):
        return
    for stale in (paths.part, paths.ready, paths.base):
        if stale.exists():
            try:
                stale.unlink()
                logger.info("Removed stale artifact path: %s", stale)
            except OSError as exc:
                logger.warning("Failed to remove stale artifact %s: %s", stale, exc)


def _create_process_lock(lock_path: Path) -> None:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text("downloading", encoding="utf-8")


def _remove_process_lock(lock_path: Path) -> None:
    try:
        lock_path.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("Failed to remove process lock %s: %s", lock_path, exc)


def _write_ready_marker(ready_path: Path) -> None:
    ready_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ready_path, "w", encoding="utf-8") as handle:
        handle.write("ok")
        handle.flush()
        os.fsync(handle.fileno())


def download_if_missing(url: str, path: Path | str) -> Path:
    """Download ``url`` to ``path`` when the artifact is not already ready."""
    paths = _artifact_paths(path)

    # Mandatory readiness gate — no bypass.
    if _artifact_is_ready(paths.base):
        _log_skip_download(paths)
        return paths.base

    # A. Peer lock wait
    if paths.lock.exists():
        for _ in range(LOCK_RETRY_ATTEMPTS):
            if _artifact_is_ready(paths.base, log=False):
                _log_skip_download(paths)
                return paths.base
            if not paths.lock.exists():
                break
            time.sleep(LOCK_RETRY_MS / 1000.0)
        if _artifact_is_ready(paths.base, log=False):
            _log_skip_download(paths)
            return paths.base
        if paths.lock.exists():
            logger.info("Peer lock still active; skipping download for %s", paths.base)
            return paths.base

    _remove_invalid_artifact(paths)

    # B. Acquire lock
    _create_process_lock(paths.lock)
    paths.base.parent.mkdir(parents=True, exist_ok=True)

    logger.info("DOWNLOAD STARTED: %s", paths.base)
    print("ARTIFACT DOWNLOAD", flush=True)
    logger.info("ARTIFACT DOWNLOAD: fetching %s -> %s", url, paths.base)

    try:
        response = requests.get(url, timeout=300, stream=True)
        response.raise_for_status()

        # C. Download to .part
        with open(paths.part, "wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
            # D. Flush + fsync
            handle.flush()
            os.fsync(handle.fileno())

        # E. Atomic replace
        os.replace(paths.part, paths.base)
        # F. Mark ready only after successful replace
        _write_ready_marker(paths.ready)

        logger.info("DOWNLOAD COMPLETED: %s (%d bytes)", paths.base, os.path.getsize(paths.base))
        return paths.base
    except Exception:
        if paths.part.exists():
            try:
                paths.part.unlink()
            except OSError:
                pass
        if paths.ready.exists():
            try:
                paths.ready.unlink()
            except OSError:
                pass
        raise
    finally:
        # G. Always remove .lock
        _remove_process_lock(paths.lock)


def _ensure_artifact(path: Path | str) -> Path:
    """Ensure a pipeline artifact exists locally, downloading if configured."""
    paths = _artifact_paths(path)

    rel = _relative_key(paths.base)
    remote_name = _ARTIFACT_REMOTE_NAMES.get(rel)
    if remote_name is None:
        raise FileNotFoundError(f"No remote mapping for artifact: {paths.base}")

    base_url = ARTIFACT_BASE_URL.rstrip("/")
    if base_url == "TODO_SET_THIS":
        raise FileNotFoundError(
            f"Artifact missing: {paths.base}. Set the ARTIFACT_BASE_URL environment variable "
            f"to a base URL hosting deployment artifacts, or generate data/ locally."
        )

    url = f"{base_url}/{remote_name}"
    return download_if_missing(url, paths.base)


def _download_if_missing() -> tuple[str, ...]:
    """Ensure all deployment artifacts exist on disk (idempotent; no Streamlit)."""
    cfg = AppConfig.default()
    artifact = cfg.artifact
    paths = (
        artifact.chunks_path,
        artifact.embeddings_path,
        artifact.mentions_path,
        artifact.has_chunk_path,
        artifact.entities_path,
    )
    ensured: list[str] = []
    for path in paths:
        resolved = _ensure_artifact(path)
        ensured.append(str(_artifact_paths(resolved).base))
    return tuple(ensured)


@lru_cache(maxsize=1)
def ensure_deployment_artifacts() -> tuple[str, ...]:
    """Download missing deployment artifacts once per process (file-safe)."""
    return _download_if_missing()


@dataclass(frozen=True)
class LoadedArtifacts:
    """Container for all loaded pipeline artifacts."""

    chunks: list[dict[str, Any]]
    embeddings: np.ndarray
    mentions: list[dict[str, str]]
    has_chunk: list[dict[str, str]]
    entities: list[dict[str, str]]


class ArtifactLoader:
    """Load and validate chunks, embeddings, mentions, and graph edges."""

    @staticmethod
    def load(config: AppConfig) -> LoadedArtifacts:
        artifact = config.artifact

        ensure_deployment_artifacts()

        chunks_path = _artifact_paths(artifact.chunks_path).base
        embeddings_path = _artifact_paths(artifact.embeddings_path).base
        mentions_path = _artifact_paths(artifact.mentions_path).base
        has_chunk_path = _artifact_paths(artifact.has_chunk_path).base
        entities_path = _artifact_paths(artifact.entities_path).base

        chunks = list(iter_jsonl_gz(chunks_path))
        embeddings = np.load(embeddings_path)

        if embeddings.shape[0] != len(chunks):
            raise ValueError(
                f"Embedding rows ({embeddings.shape[0]}) do not match chunk count ({len(chunks)})."
            )

        expected_dim = config.embedding.embedding_dim
        if embeddings.shape[1] != expected_dim:
            raise ValueError(
                f"Embedding dimension ({embeddings.shape[1]}) does not match config ({expected_dim})."
            )

        embeddings = normalize_embeddings(embeddings)

        mentions = load_csv(mentions_path, ["chunk_id", "entity_id"])
        has_chunk = load_csv(has_chunk_path, ["article_id", "chunk_id"])
        entities = load_csv(entities_path, ["entity_id", "name", "label"])

        ArtifactLoader._validate_mentions(chunks, mentions)

        return LoadedArtifacts(
            chunks=chunks,
            embeddings=embeddings,
            mentions=mentions,
            has_chunk=has_chunk,
            entities=entities,
        )

    @staticmethod
    def _validate_mentions(chunks: list[dict[str, Any]], mentions: list[dict[str, str]]) -> None:
        chunk_id_set = {str(chunk["chunk_id"]) for chunk in chunks}
        unknown_chunks = {rel["chunk_id"] for rel in mentions if rel["chunk_id"] not in chunk_id_set}
        if unknown_chunks:
            sample = sorted(unknown_chunks)[:5]
            raise ValueError(f"mentions.csv references unknown chunk_ids: {sample}")
