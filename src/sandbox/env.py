"""Minimal .env loader. No external dependency.

Reads `crs-sandbox/.env`, strips optional surrounding quotes, and writes the
keys into os.environ if not already present. Call `load_env()` once at the
top of any script.
"""

from __future__ import annotations

import os
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_ENV_PATH = _REPO_ROOT / ".env"


def load_env(env_path: Path | None = None) -> dict[str, str]:
    """Load .env into os.environ. Returns the parsed dict."""
    path = env_path or _ENV_PATH
    parsed: dict[str, str] = {}
    if not path.exists():
        return parsed
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
            val = val[1:-1]
        parsed[key] = val
        os.environ.setdefault(key, val)
    return parsed
