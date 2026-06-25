"""Phase-A artifact download (process start only; never during Streamlit runtime)."""

from __future__ import annotations

import logging
import os
import shutil
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, TypedDict

import requests

from src.config import AppConfig
from src.infrastructure.storage.safety_guard import (
    assert_no_repo_write,
    detect_repo_root,
    safe_mkdir,
    safe_write_file,
    verify_no_repo_writes,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.infrastructure.storage.artifact_loader import LoadedArtifacts

MIN_VALID_SIZE_DEFAULT = 1024
ARTIFACT_BASE_URL = os.environ.get("ARTIFACT_BASE_URL", "TODO_SET_THIS")

_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503})
_RETRY_BACKOFF_SECONDS = (1, 2, 4)

# Order matches artifact_paths() and ArtifactLoader.load_from_paths().
_CACHE_LOGICAL_PATHS: tuple[str, ...] = (
    "data/chunks/chunks_semantic.jsonl.gz",
    "data/chunks/chunks_fixed.jsonl.gz",
    "data/chunks/chunks_sentence.jsonl.gz",
    "data/embeddings/semantic_embeddings.npy",
    "data/embeddings/fixed_embeddings.npy",
    "data/embeddings/sentence_embeddings.npy",
    "data/graph/mentions.csv",
    "data/graph/has_chunk.csv",
    "data/graph/entities.csv",
    "data/hnsw/semantic_index.bin",
    "data/hnsw/semantic_chunk_ids.json",
    "data/hnsw/fixed_index.bin",
    "data/hnsw/fixed_chunk_ids.json",
    "data/hnsw/sentence_index.bin",
    "data/hnsw/sentence_chunk_ids.json",
    "data/hnsw/manifest.json",
)

_ARTIFACT_REMOTE_NAMES: dict[str, str] = {
    "data/chunks/chunks_semantic.jsonl.gz": "chunks_semantic.jsonl.gz",
    "data/chunks/chunks_fixed.jsonl.gz": "chunks_fixed.jsonl.gz",
    "data/chunks/chunks_sentence.jsonl.gz": "chunks_sentence.jsonl.gz",
    "data/embeddings/semantic_embeddings.npy": "semantic_embeddings.npy",
    "data/embeddings/fixed_embeddings.npy": "fixed_embeddings.npy",
    "data/embeddings/sentence_embeddings.npy": "sentence_embeddings.npy",
    "data/graph/mentions.csv": "mentions.csv",
    "data/graph/entities.csv": "entities.csv",
    "data/graph/has_chunk.csv": "has_chunk.csv",
    "data/hnsw/semantic_index.bin": "semantic_index.bin",
    "data/hnsw/semantic_chunk_ids.json": "semantic_chunk_ids.json",
    "data/hnsw/fixed_index.bin": "fixed_index.bin",
    "data/hnsw/fixed_chunk_ids.json": "fixed_chunk_ids.json",
    "data/hnsw/sentence_index.bin": "sentence_index.bin",
    "data/hnsw/sentence_chunk_ids.json": "sentence_chunk_ids.json",
    "data/hnsw/manifest.json": "manifest.json",
}

_MIN_VALID_SIZES: dict[str, int] = {
    "data/chunks/chunks_semantic.jsonl.gz": MIN_VALID_SIZE_DEFAULT,
    "data/chunks/chunks_fixed.jsonl.gz": MIN_VALID_SIZE_DEFAULT,
    "data/chunks/chunks_sentence.jsonl.gz": MIN_VALID_SIZE_DEFAULT,
    "data/embeddings/semantic_embeddings.npy": 1024 * 1024,
    "data/embeddings/fixed_embeddings.npy": 1024 * 1024,
    "data/embeddings/sentence_embeddings.npy": 1024 * 1024,
    "data/graph/mentions.csv": MIN_VALID_SIZE_DEFAULT,
    "data/graph/entities.csv": MIN_VALID_SIZE_DEFAULT,
    "data/graph/has_chunk.csv": MIN_VALID_SIZE_DEFAULT,
    "data/hnsw/semantic_index.bin": 1024 * 1024,
    "data/hnsw/fixed_index.bin": 1024 * 1024,
    "data/hnsw/sentence_index.bin": 1024 * 1024,
    "data/hnsw/semantic_chunk_ids.json": MIN_VALID_SIZE_DEFAULT,
    "data/hnsw/fixed_chunk_ids.json": MIN_VALID_SIZE_DEFAULT,
    "data/hnsw/sentence_chunk_ids.json": MIN_VALID_SIZE_DEFAULT,
    "data/hnsw/manifest.json": 100,
}

_streamlit_runtime = False
_bootstrap_complete = False
_downloading_allowed = False
_preloaded_artifacts: "LoadedArtifacts | None" = None
_last_bootstrap_status: "BootstrapStatus | None" = None
_download_locks: dict[str, threading.Lock] = {}
_download_locks_guard = threading.Lock()


class BootstrapStatus(TypedDict):
    success: bool
    missing: list[str]
    failed: list[str]
    cached: list[str]


def mark_streamlit_runtime() -> None:
    """Mark that Streamlit has been imported; blocks further bootstrap calls."""
    global _streamlit_runtime
    _streamlit_runtime = True


def is_streamlit_runtime() -> bool:
    return _streamlit_runtime


def is_bootstrap_complete() -> bool:
    return _bootstrap_complete


def get_last_bootstrap_status() -> BootstrapStatus | None:
    return _last_bootstrap_status


def is_bootstrap_successful() -> bool:
    """Return True only when bootstrap completed with all required artifacts."""
    return _last_bootstrap_status is not None and _last_bootstrap_status["success"]


def require_bootstrap_success() -> None:
    """Fail fast when artifact bootstrap did not complete successfully."""
    if not is_bootstrap_successful():
        status = _last_bootstrap_status or _empty_status()
        raise RuntimeError(
            "Artifact bootstrap failed; pipeline cannot be built. "
            f"missing={status['missing']}, failed={status['failed']}"
        )


@contextmanager
def _downloading_phase() -> Iterator[None]:
    global _downloading_allowed
    _downloading_allowed = True
    try:
        yield
    finally:
        _downloading_allowed = False


def assert_downloading_allowed() -> None:
    if not _downloading_allowed:
        raise RuntimeError("Artifact downloads are only allowed during bootstrap_artifacts()")


def default_cache_dir() -> str:
    env_dir = os.environ.get("ARTIFACT_CACHE_DIR", "").strip()
    return env_dir or "/tmp/pubmed-graphrag"


def artifact_paths(cache_dir: str) -> tuple[str, ...]:
    """Return deterministic on-disk paths for all indexes (no filesystem access)."""
    root = Path(cache_dir).resolve()
    return tuple(str(root / logical) for logical in _CACHE_LOGICAL_PATHS)


def core_artifact_paths(cache_dir: str) -> tuple[str, str, str, str, str]:
    """Return paths for the core semantic + graph artifacts used by ArtifactLoader."""
    paths = artifact_paths(cache_dir)
    # _CACHE_LOGICAL_PATHS order:
    # 0=semantic chunks, 3=semantic embeddings, 6=mentions, 7=has_chunk, 8=entities
    return (
        paths[0],
        paths[3],
        paths[6],
        paths[7],
        paths[8],
    )


def _repo_root() -> Path:
    return detect_repo_root()


def _cache_path(cache_dir: str, logical: str) -> Path:
    dest = (Path(cache_dir).resolve() / logical).resolve()
    assert_no_repo_write(str(dest))
    return dest


def _cache_hit(path: Path) -> bool:
    return path.is_file() and os.path.getsize(path) > 0


def _artifact_file_valid(path: Path, logical_key: str) -> bool:
    if not _cache_hit(path):
        return False
    return os.path.getsize(path) >= _MIN_VALID_SIZES.get(logical_key, MIN_VALID_SIZE_DEFAULT)


def _download_lock(dest: Path) -> threading.Lock:
    key = str(dest.resolve())
    with _download_locks_guard:
        if key not in _download_locks:
            _download_locks[key] = threading.Lock()
        return _download_locks[key]


def _use_stale_or_fail(dest: Path, logical_key: str) -> bool:
    if _artifact_file_valid(dest, logical_key):
        logger.info("USING STALE LOCAL ARTIFACT: %s", dest)
        return True
    logger.error("Artifact unavailable and no local cache: %s", dest)
    return False


def _materialize_to_cache(source: Path, cache_path: Path, logical_key: str) -> bool:
    """Copy a read-only repo artifact into the external cache directory."""
    if _artifact_file_valid(cache_path, logical_key):
        return True
    safe_mkdir(cache_path.parent)
    try:
        shutil.copy2(source, cache_path)
    except OSError as exc:
        logger.error("Failed to copy artifact into cache %s: %s", cache_path, exc)
        return False
    return _artifact_file_valid(cache_path, logical_key)


def _remove_part_file(part_path: Path) -> None:
    if part_path.exists():
        try:
            part_path.unlink()
        except OSError:
            pass


def _finalize_part_download(part_path: Path, dest: Path, logical_key: str) -> bool:
    """Atomically promote a completed .part file to the final artifact path."""
    if _artifact_file_valid(dest, logical_key):
        _remove_part_file(part_path)
        return True
    if not part_path.is_file():
        logger.warning("Cannot finalize download; part file missing: %s", part_path)
        return False
    if os.path.getsize(part_path) == 0:
        logger.error("Cannot finalize download; part file is empty: %s", part_path)
        _remove_part_file(part_path)
        return False
    try:
        assert_no_repo_write(str(dest))
        os.replace(part_path, dest)
    except FileNotFoundError:
        if _artifact_file_valid(dest, logical_key):
            return True
        logger.warning("Finalize skipped; part file disappeared before replace: %s", part_path)
        return False
    except OSError as exc:
        if _artifact_file_valid(dest, logical_key):
            _remove_part_file(part_path)
            return True
        logger.error("Failed to finalize artifact download %s: %s", dest, exc)
        _remove_part_file(part_path)
        return False
    return _artifact_file_valid(dest, logical_key)


def _write_response_to_part(response: requests.Response, part_path: Path) -> bool:
    try:
        with safe_write_file(part_path, "wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        logger.error("Failed writing artifact part file %s: %s", part_path, exc)
        return False

    if not part_path.is_file() or os.path.getsize(part_path) == 0:
        logger.error("Download produced empty file: %s", part_path)
        return False

    return True


def _download_if_missing(url: str, dest: Path, logical_key: str) -> bool:
    """Download artifact to dest. Returns True when dest is usable, False otherwise."""
    assert_downloading_allowed()

    with _download_lock(dest):
        if _artifact_file_valid(dest, logical_key):
            logger.info("USING CACHED ARTIFACT: %s", dest)
            return True

        safe_mkdir(dest.parent)
        part_path = Path(f"{dest}.part").resolve()
        assert_no_repo_write(str(part_path))

        logger.info("Artifact download URL: %s", url)
        logger.info("Artifact destination: %s", dest)

        attempts = len(_RETRY_BACKOFF_SECONDS) + 1
        for attempt in range(attempts):
            if _artifact_file_valid(dest, logical_key):
                _remove_part_file(part_path)
                return True

            if attempt > 0:
                backoff = _RETRY_BACKOFF_SECONDS[attempt - 1]
                logger.warning(
                    "Retrying artifact download in %ss (attempt %d/%d): %s",
                    backoff,
                    attempt + 1,
                    attempts,
                    url,
                )
                time.sleep(backoff)

            _remove_part_file(part_path)

            try:
                response = requests.get(url, timeout=300, stream=True)
            except requests.exceptions.RequestException as exc:
                logger.warning("Artifact download request failed: %s (%s)", url, exc)
                if attempt + 1 >= attempts:
                    return _use_stale_or_fail(dest, logical_key)
                continue

            if response.status_code == 404:
                logger.warning("Artifact missing on remote, skipping download: %s", url)
                response.close()
                return _use_stale_or_fail(dest, logical_key)

            if response.status_code in _RETRYABLE_STATUS_CODES:
                logger.warning(
                    "Retryable HTTP %s for artifact download: %s",
                    response.status_code,
                    url,
                )
                response.close()
                if attempt + 1 >= attempts:
                    return _use_stale_or_fail(dest, logical_key)
                continue

            try:
                response.raise_for_status()
            except requests.exceptions.HTTPError as exc:
                logger.warning("Artifact download HTTP error: %s (%s)", url, exc)
                response.close()
                if attempt + 1 >= attempts:
                    return _use_stale_or_fail(dest, logical_key)
                continue

            if not _write_response_to_part(response, part_path):
                response.close()
                _remove_part_file(part_path)
                if attempt + 1 >= attempts:
                    return _use_stale_or_fail(dest, logical_key)
                continue

            response.close()
            if not _finalize_part_download(part_path, dest, logical_key):
                if attempt + 1 >= attempts:
                    return _use_stale_or_fail(dest, logical_key)
                continue

            logger.info("DOWNLOAD COMPLETED: %s (%d bytes)", dest, os.path.getsize(dest))
            return True

        return _use_stale_or_fail(dest, logical_key)


def _ensure_artifact(
    cache_dir: str,
    logical: str,
    status: BootstrapStatus,
) -> Path | None:
    """Resolve or download one artifact. Never raises on remote failure."""
    assert_downloading_allowed()
    logical_key = logical
    cache_path = _cache_path(cache_dir, logical)

    if _artifact_file_valid(cache_path, logical_key):
        logger.info("USING CACHED ARTIFACT: %s", cache_path)
        status["cached"].append(logical)
        return cache_path

    repo_path = (_repo_root() / logical).resolve()
    if _artifact_file_valid(repo_path, logical_key):
        logger.info("Using existing repo artifact (read-only): %s", repo_path)
        if _materialize_to_cache(repo_path, cache_path, logical_key):
            status["cached"].append(logical)
            return cache_path
        status["failed"].append(logical)
        return None

    remote_name = _ARTIFACT_REMOTE_NAMES.get(logical_key)
    if remote_name is None:
        logger.error("No remote mapping for artifact: %s", logical_key)
        status["failed"].append(logical)
        return None

    base_url = ARTIFACT_BASE_URL.rstrip("/")
    if base_url == "TODO_SET_THIS":
        logger.warning(
            "ARTIFACT_BASE_URL not set; cannot download %s. Checking local cache only.",
            logical_key,
        )
        if _use_stale_or_fail(cache_path, logical_key):
            status["cached"].append(logical)
            return cache_path
        status["missing"].append(logical)
        return None

    url = f"{base_url}/{remote_name}"
    if _download_if_missing(url, cache_path, logical_key) and _artifact_file_valid(
        cache_path, logical_key
    ):
        status["cached"].append(logical)
        return cache_path

    status["failed"].append(logical)
    return None


def _empty_status() -> BootstrapStatus:
    return {"success": False, "missing": [], "failed": [], "cached": []}


def _logical_paths_for_cache(paths: list[str]) -> list[tuple[str, str]]:
    """Pair logical artifact keys with on-disk cache paths."""
    return list(zip(_CACHE_LOGICAL_PATHS, paths, strict=True))


def _finalize_status(status: BootstrapStatus, paths: list[str]) -> BootstrapStatus:
    present = {
        logical
        for logical, path in _logical_paths_for_cache(paths)
        if _artifact_file_valid(Path(path), logical)
    }
    status["missing"] = [logical for logical in _CACHE_LOGICAL_PATHS if logical not in present]
    status["success"] = not status["missing"] and not status["failed"]
    return status


def bootstrap_artifacts(cache_dir: str | None = None) -> BootstrapStatus:
    """Download deployment artifacts at process start.

    Raises ``RuntimeError`` when any required artifact cannot be materialized.
    """
    global _bootstrap_complete, _last_bootstrap_status, _preloaded_artifacts

    if _bootstrap_complete and _last_bootstrap_status is not None:
        if not _last_bootstrap_status["success"]:
            require_bootstrap_success()
        return _last_bootstrap_status

    if _streamlit_runtime:
        raise RuntimeError(
            "bootstrap_artifacts() must run before Streamlit is imported; "
            "call it from scripts/demo.py at process start."
        )

    resolved_cache_dir = cache_dir or default_cache_dir()
    assert_no_repo_write(resolved_cache_dir)

    status: BootstrapStatus = _empty_status()

    with _downloading_phase():
        for logical in _CACHE_LOGICAL_PATHS:
            try:
                _ensure_artifact(resolved_cache_dir, logical, status)
            except Exception as exc:
                logger.error("Unexpected error ensuring artifact %s: %s", logical, exc)
                if logical not in status["failed"]:
                    status["failed"].append(logical)

    paths = list(artifact_paths(resolved_cache_dir))
    status = _finalize_status(status, paths)

    try:
        verify_no_repo_writes(paths)
    except RuntimeError as exc:
        logger.warning("Repo write verification warning during bootstrap: %s", exc)

    all_present = all(
        _artifact_file_valid(Path(path), logical)
        for logical, path in _logical_paths_for_cache(paths)
    )
    _preloaded_artifacts = None
    if all_present:
        try:
            from src.infrastructure.storage.artifact_loader import ArtifactLoader

            cfg = AppConfig.default()
            _preloaded_artifacts = ArtifactLoader.load_from_paths(
                *core_artifact_paths(resolved_cache_dir),
                embedding_dim=cfg.embedding.embedding_dim,
            )
        except Exception as exc:
            logger.error("Failed to preload artifacts into memory: %s", exc)
            status["success"] = False
            _preloaded_artifacts = None

    if status["failed"]:
        logger.error("Artifact bootstrap failures: %s", status["failed"])
    if status["missing"]:
        logger.error("Artifact bootstrap missing: %s", status["missing"])

    _bootstrap_complete = True
    _last_bootstrap_status = status

    if status["success"]:
        message = "ARTIFACT PHASE COMPLETE (ALL FILES LOCAL)"
        logger.info(message)
        print(message, flush=True)
        return status

    message = "ARTIFACT PHASE FAILED"
    logger.error(message)
    print(message, flush=True)
    raise RuntimeError(
        "Artifact bootstrap failed; required artifacts are unavailable. "
        f"missing={status['missing']}, failed={status['failed']}"
    )


def get_preloaded_artifacts() -> "LoadedArtifacts":
    """Return artifacts loaded during bootstrap (read-only after bootstrap)."""
    require_bootstrap_success()
    if _preloaded_artifacts is None:
        raise RuntimeError(
            "bootstrap_artifacts() must complete successfully before pipeline build; "
            "call it at process start before importing Streamlit."
        )
    return _preloaded_artifacts
