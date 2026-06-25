"""Process-wide environment defaults for deployment runtimes."""


def configure_environment() -> None:
    """Configure cache env vars before any Streamlit or ML imports."""
    import os

    os.environ.setdefault("HF_HOME", "/tmp/hf_cache")
    os.environ.setdefault("TORCH_HOME", "/tmp/torch_cache")
    os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/hf_cache/datasets")
    os.environ.setdefault("HF_HUB_CACHE", "/tmp/hf_cache/hub")
