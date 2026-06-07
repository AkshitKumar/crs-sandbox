"""CLI: build a vector index for a category.

Usage:
    python scripts/build_index.py laptop
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from sandbox.index.builder import build_index  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("category")
    args = parser.parse_args()

    config_path = REPO_ROOT / "configs" / f"{args.category}.yaml"
    with config_path.open() as f:
        config = yaml.safe_load(f)

    catalog_path = REPO_ROOT / config["catalog_path"]
    index_dir = REPO_ROOT / config["index_dir"]
    build_index(catalog_path=catalog_path, index_dir=index_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
