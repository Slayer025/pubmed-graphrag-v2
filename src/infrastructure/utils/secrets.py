"""Secret scrubbing helpers.

Prevent API tokens and other sensitive values from appearing in logs, UI
output, or exception messages.
"""

from __future__ import annotations

import os
import re


# Known secret environment variable names. Values are redacted wherever they
# appear in text.
_SENSITIVE_ENV_VARS = [
    "HF_API_TOKEN",
    "OPENAI_API_KEY",
    "OLLAMA_URL",
    "EMBEDDING_SERVICE_URL",
]

# Generic patterns that look like API tokens / secrets.
_TOKEN_PATTERNS = [
    re.compile(r"hf_[A-Za-z0-9_]{20,}", re.IGNORECASE),
    re.compile(r"sk-[A-Za-z0-9]{20,}", re.IGNORECASE),
    re.compile(r"ghp_[A-Za-z0-9]{20,}", re.IGNORECASE),
]

_REDACTED = "***REDACTED***"


def _collect_known_secrets() -> set[str]:
    """Gather non-empty secret values from the environment."""
    secrets: set[str] = set()
    for name in _SENSITIVE_ENV_VARS:
        value = os.environ.get(name)
        if value:
            secrets.add(value)
    return secrets


def scrub_secrets(text: str | None) -> str:
    """Return ``text`` with any known or suspected secrets replaced.

    This is a defensive, best-effort scrubber. It is not a substitute for
    keeping secrets out of strings in the first place.
    """
    if text is None:
        return ""
    result = str(text)
    for secret in _collect_known_secrets():
        result = result.replace(secret, _REDACTED)
    for pattern in _TOKEN_PATTERNS:
        result = pattern.sub(_REDACTED, result)
    return result
