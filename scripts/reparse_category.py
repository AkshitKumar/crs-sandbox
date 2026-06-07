"""Reparse cached HTML for a category without re-scraping.

Useful when the parser changes (e.g., Amazon updates a selector).
Overwrites products.jsonl from the contents of _raw/.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from sandbox.scraper.amazon import parse_pdp  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("category")
    args = parser.parse_args()

    config_path = REPO_ROOT / "configs" / f"{args.category}.yaml"
    with config_path.open() as f:
        config = yaml.safe_load(f)
    reviews_per_product = config["scrape"].get("reviews_per_product", 8)

    cat_dir = REPO_ROOT / Path(config["catalog_path"]).parent
    raw_dir = cat_dir / "_raw"
    out_path = cat_dir / "products.jsonl"

    # Preserve sponsored_in_search and discovered_via_query from existing JSONL.
    existing: dict[str, dict] = {}
    if out_path.exists():
        with out_path.open() as f:
            for line in f:
                rec = json.loads(line)
                existing[rec["asin"]] = rec

    htmls = sorted(raw_dir.glob("*.html"))
    print(f"Reparsing {len(htmls)} cached PDPs from {raw_dir}")

    with out_path.open("w") as out_f:
        for path in htmls:
            asin = path.stem
            html = path.read_text()
            result = parse_pdp(html, asin)
            result.review_excerpts = result.review_excerpts[:reviews_per_product]
            # Restore search-time metadata.
            prev = existing.get(asin, {})
            result.sponsored_in_search = prev.get("sponsored_in_search", False)
            result.discovered_via_query = prev.get("discovered_via_query")
            result.url = prev.get("url") or f"https://www.amazon.com/dp/{asin}"
            out_f.write(json.dumps(asdict(result)) + "\n")

    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
