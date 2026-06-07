"""Catalog loading helpers.

Centralizes the JSONL → list[dict] conversion so tools don't each reinvent it,
and provides a small cache so repeated calls within a session are cheap.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def category_config_path(category: str) -> Path:
    return REPO_ROOT / "configs" / f"{category}.yaml"


def load_config(category: str) -> dict[str, Any]:
    with category_config_path(category).open() as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=64)
def load_catalog(category: str) -> tuple[dict[str, Any], ...]:
    """Read products.jsonl for a category. Cached, returns tuple for hashability."""
    config = load_config(category)
    path = REPO_ROOT / config["catalog_path"]
    products = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                products.append(json.loads(line))
    return tuple(products)


def get_product(category: str, asin: str) -> dict[str, Any] | None:
    for p in load_catalog(category):
        if p.get("asin") == asin:
            return p
    return None
