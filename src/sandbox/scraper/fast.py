"""Concurrent Amazon scraper via ScraperAPI's API endpoint mode.

Why a separate module from `amazon.py`:
    - `amazon.py` uses Playwright (browser-based, sequential, good for residential
      stealth scraping).
    - This module uses `httpx` async (no browser, parallel, used when we have
      a ScraperAPI key — they handle anti-bot for us, so the browser is overhead).

Reuses `parse_pdp` and `_parse_search_results` from `amazon.py` since the
HTML parsing is identical.

Concurrency: bounded by an asyncio.Semaphore (default 5 = ScraperAPI Hobby plan).
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import asdict
from pathlib import Path

import httpx

from sandbox.scraper.amazon import (
    AMAZON_BASE,
    SEARCH_URL,
    PDP_URL,
    ScrapeResult,
    _looks_like_captcha,
    _parse_search_results,
    parse_pdp,
    build_scraperapi_url,
)


REQUEST_TIMEOUT = 90.0  # ScraperAPI for Amazon: ~3–15s per request


async def _fetch_via_scraperapi(
    client: httpx.AsyncClient,
    target_url: str,
    api_key: str,
    render: bool,
    sem: asyncio.Semaphore,
) -> tuple[str | None, str | None]:
    """Fetch one URL through ScraperAPI. Returns (html, error)."""
    wrapped = build_scraperapi_url(target_url, api_key=api_key, render=render)
    async with sem:
        try:
            r = await client.get(wrapped, timeout=REQUEST_TIMEOUT)
            if r.status_code != 200:
                return None, f"http {r.status_code}"
            return r.text, None
        except Exception as e:
            return None, f"{type(e).__name__}: {e}"


async def _discover_asins_fast(
    client: httpx.AsyncClient,
    api_key: str,
    queries: list[str],
    target_count: int,
    sem: asyncio.Semaphore,
    max_pages_per_query: int = 4,
    render: bool = False,
) -> list[tuple[str, bool, str]]:
    """Fan out search-page fetches concurrently, then collect ASINs."""
    # Build all (query, page) pairs up to the cap.
    pairs: list[tuple[str, int]] = []
    for q in queries:
        for p in range(1, max_pages_per_query + 1):
            pairs.append((q, p))

    async def fetch_one(q: str, p: int):
        url = SEARCH_URL.format(query=q.replace(" ", "+"), page=p)
        html, err = await _fetch_via_scraperapi(client, url, api_key, render, sem)
        return q, p, html, err

    results = await asyncio.gather(*(fetch_one(q, p) for q, p in pairs))

    collected: dict[str, tuple[bool, str]] = {}
    for q, p, html, err in results:
        if err:
            print(f"  [search] {q!r} page {p}: error {err}")
            continue
        if _looks_like_captcha(html):
            print(f"  [search] {q!r} page {p}: CAPTCHA in body")
            continue
        tiles = _parse_search_results(html)
        new = 0
        for asin, sponsored in tiles:
            if asin not in collected:
                collected[asin] = (sponsored, q)
                new += 1
        print(f"  [search] {q!r} page {p}: +{new} new (total {len(collected)})")
        if len(collected) >= target_count:
            break

    return [(asin, sp, q) for asin, (sp, q) in list(collected.items())[:target_count]]


def _load_existing_asins(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    asins: set[str] = set()
    with out_path.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
                if rec.get("asin"):
                    asins.add(rec["asin"])
            except json.JSONDecodeError:
                continue
    return asins


async def scrape_category_fast(
    category: str,
    seed_queries: list[str],
    target_products: int,
    output_dir: Path,
    api_key: str,
    reviews_per_product: int = 8,
    concurrency: int = 5,
    render: bool = False,
) -> dict:
    """Concurrent scrape via ScraperAPI API endpoint mode.

    Returns a summary dict for logging.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "_raw"
    raw_dir.mkdir(exist_ok=True)
    products_path = output_dir / "products.jsonl"

    existing = _load_existing_asins(products_path)
    print(f"[{category}] starting. {len(existing)} ASINs already scraped. concurrency={concurrency} render={render}")

    sem = asyncio.Semaphore(concurrency)
    summary = {"category": category, "fetched": 0, "errors": 0, "captchas": 0}

    async with httpx.AsyncClient(http2=False, follow_redirects=True) as client:
        # 1. Discover.
        print(f"[{category}] discovering ASINs (parallel)...")
        discovered = await _discover_asins_fast(
            client=client,
            api_key=api_key,
            queries=seed_queries,
            target_count=target_products,
            sem=sem,
            render=render,
        )
        print(f"[{category}] discovered {len(discovered)} unique ASINs.")

        # 2. Fetch PDPs concurrently.
        to_fetch = [d for d in discovered if d[0] not in existing]
        print(f"[{category}] fetching {len(to_fetch)} new PDPs in parallel (max {concurrency} in flight)...")

        async def fetch_pdp(asin: str, sponsored: bool, via_query: str):
            url = PDP_URL.format(asin=asin)
            html, err = await _fetch_via_scraperapi(client, url, api_key, render, sem)
            return asin, sponsored, via_query, url, html, err

        # gather concurrently; semaphore bounds in-flight count.
        # Append to JSONL as each completes (use a single writer to avoid corruption).
        write_lock = asyncio.Lock()
        completed = 0

        async def fetch_and_save(asin, sponsored, via_query):
            nonlocal completed
            asin, sponsored, via_query, url, html, err = await fetch_pdp(asin, sponsored, via_query)
            completed += 1
            if err:
                summary["errors"] += 1
                print(f"  [{completed}/{len(to_fetch)}] {asin}: {err}")
                return
            if _looks_like_captcha(html):
                summary["captchas"] += 1
                print(f"  [{completed}/{len(to_fetch)}] {asin}: CAPTCHA")
                return
            raw_path = raw_dir / f"{asin}.html"
            raw_path.write_text(html)

            result = parse_pdp(html, asin)
            result.sponsored_in_search = sponsored
            result.discovered_via_query = via_query
            result.url = url
            result.review_excerpts = result.review_excerpts[:reviews_per_product]

            async with write_lock:
                with products_path.open("a") as f:
                    f.write(json.dumps(asdict(result)) + "\n")
            summary["fetched"] += 1
            title_preview = (result.title or "?")[:60]
            print(
                f"  [{completed}/{len(to_fetch)}] {asin}  {title_preview!r:62}  "
                f"price={result.price}  specs={len(result.spec_table)}"
            )

        await asyncio.gather(*(fetch_and_save(a, s, q) for a, s, q in to_fetch))

    print(f"[{category}] done. fetched={summary['fetched']} errors={summary['errors']} captchas={summary['captchas']}")
    return summary
