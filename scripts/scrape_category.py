"""CLI entry for the Amazon scraper.

Two backends, selected by which env vars are present:

    SCRAPERAPI_KEY set:  httpx-based, concurrent (default concurrency=5).
                         Fast, reliable, ~$0.05–$0.10 per category at $49/mo plan.

    Neither set:         Playwright-based, sequential, scrapes from your IP.
                         Amazon will CAPTCHA after ~90 requests.

Use `--no-proxy` to force Playwright/residential even if SCRAPERAPI_KEY is set.

Usage:
    uv run python scripts/scrape_category.py laptop                  # fast (ScraperAPI)
    uv run python scripts/scrape_category.py laptop --concurrency 8  # bump in-flight count
    uv run python scripts/scrape_category.py laptop --render         # JS rendering (~10x cost)
    uv run python scripts/scrape_category.py laptop --no-proxy       # force residential
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from sandbox.env import load_env  # noqa: E402


def _str2bool(s: str) -> bool:
    return s.lower() in {"1", "true", "yes", "y", "t"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("category", help="category name (matches configs/<name>.yaml)")
    parser.add_argument("--target", type=int, default=None, help="override target product count from config")
    parser.add_argument("--concurrency", type=int, default=5, help="parallel in-flight requests (ScraperAPI Hobby = 5)")
    parser.add_argument("--render", action="store_true", help="enable JS rendering via ScraperAPI (~10x cost)")
    parser.add_argument("--no-proxy", action="store_true", help="force Playwright residential scraping")
    parser.add_argument("--headless", type=_str2bool, default=True, help="(residential mode only) headless browser")
    args = parser.parse_args()

    load_env()

    config_path = REPO_ROOT / "configs" / f"{args.category}.yaml"
    if not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        return 1
    with config_path.open() as f:
        config = yaml.safe_load(f)

    output_dir = REPO_ROOT / Path(config["catalog_path"]).parent
    seed_queries = config["scrape"]["seed_queries"]
    target = args.target or config["scrape"]["target_products"]
    reviews_per_product = config["scrape"].get("reviews_per_product", 8)

    api_key = os.environ.get("SCRAPERAPI_KEY")
    if api_key and not args.no_proxy:
        from sandbox.scraper.fast import scrape_category_fast
        print(f"Mode: ScraperAPI (concurrency={args.concurrency}, render={args.render})")
        asyncio.run(
            scrape_category_fast(
                category=args.category,
                seed_queries=seed_queries,
                target_products=target,
                output_dir=output_dir,
                api_key=api_key,
                reviews_per_product=reviews_per_product,
                concurrency=args.concurrency,
                render=args.render,
            )
        )
    else:
        from sandbox.scraper.amazon import scrape_category
        print("Mode: residential (Playwright)")
        asyncio.run(
            scrape_category(
                category=args.category,
                seed_queries=seed_queries,
                target_products=target,
                output_dir=output_dir,
                reviews_per_product=reviews_per_product,
                proxy=None,
                headless=args.headless,
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
