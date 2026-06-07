"""Amazon scraper. Discovers ASINs via search, fetches PDPs, parses to JSONL.

Designed to be resumable: skips ASINs already in the output JSONL.
Raw HTML is persisted under data/categories/<cat>/_raw/ so parsing can be re-run
without re-scraping.

Proxy support:
- ScraperAPI: set SCRAPERAPI_KEY in .env. The scraper auto-constructs the
  proxy URL `http://scraperapi.country_code=us:KEY@proxy-server.scraperapi.com:8001`.
  Pass `render=True` to enable JS rendering (~10x cost; only needed if Amazon
  starts serving CAPTCHAs through ScraperAPI).
- Custom proxy: set PROXY_URL directly (takes precedence over SCRAPERAPI_KEY).
- No proxy: scrapes from your IP (residential). Amazon will CAPTCHA after
  ~90 requests.

When a proxy is used, polite delays are reduced (the proxy handles rotation).
CAPTCHA-detected pages cause the scraper to abort the current run, log the
ASIN, and exit; the JSONL written so far is preserved.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any
from urllib.parse import quote

from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, Browser, BrowserContext, Page


AMAZON_BASE = "https://www.amazon.com"
SEARCH_URL = AMAZON_BASE + "/s?k={query}&page={page}"
PDP_URL = AMAZON_BASE + "/dp/{asin}"

DEFAULT_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/129.0.0.0 Safari/537.36"
)


@dataclass
class ScrapeResult:
    asin: str
    title: str | None = None
    brand: str | None = None
    bullets: list[str] = field(default_factory=list)
    description: str | None = None
    price: float | None = None
    price_str: str | None = None
    avg_rating: float | None = None
    num_reviews: int | None = None
    review_excerpts: list[str] = field(default_factory=list)
    # Structured spec/details — merged from all `table.a-keyvalue` blocks on the PDP.
    # Common keys: "Color", "Brand", "Screen Size", "Hard-Drive Size", "RAM",
    # "Operating System", "Graphics Description", "Item Weight", etc.
    spec_table: dict[str, str] = field(default_factory=dict)
    sponsored_in_search: bool = False
    discovered_via_query: str | None = None
    url: str | None = None
    parse_errors: list[str] = field(default_factory=list)


SCRAPERAPI_BASE = "http://api.scraperapi.com/"


def build_scraperapi_url(
    target: str, api_key: str, render: bool = False, country: str = "us", premium: bool = False
) -> str:
    """Wrap a target URL through ScraperAPI's API endpoint.

    This is the alternative to proxy-mode. We send GET to api.scraperapi.com
    and pass the target URL as a query param; ScraperAPI fetches the target
    from a managed proxy pool and returns the raw HTML.

    Cheaper than proxy-mode for non-JS pages (Amazon mostly server-renders).
    """
    params = [
        f"api_key={api_key}",
        f"url={quote(target, safe='')}",
    ]
    if country:
        params.append(f"country_code={country}")
    if render:
        params.append("render=true")
    if premium:
        params.append("premium=true")
    return SCRAPERAPI_BASE + "?" + "&".join(params)


def resolve_fetch_mode(
    api_key: str | None, proxy_url: str | None, render: bool = False
) -> tuple[str, dict[str, Any]]:
    """Return (mode, config) where mode ∈ {"scraperapi_api", "proxy", "direct"}.

    Resolution order:
      1. PROXY_URL (custom proxy, takes precedence)
      2. SCRAPERAPI_KEY (use api endpoint mode)
      3. neither (scrape from this machine's IP)
    """
    if proxy_url:
        return "proxy", {"proxy_url": proxy_url}
    if api_key:
        return "scraperapi_api", {"api_key": api_key, "render": render}
    return "direct", {}


async def _new_context(browser: Browser) -> BrowserContext:
    ctx = await browser.new_context(
        user_agent=DEFAULT_UA,
        viewport={"width": 1440, "height": 900},
        locale="en-US",
        ignore_https_errors=True,  # safe for ScraperAPI MITM and benign otherwise
    )
    await ctx.add_init_script(
        "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
    )
    return ctx


async def _polite_sleep(min_s: float = 2.0, max_s: float = 5.0) -> None:
    await asyncio.sleep(random.uniform(min_s, max_s))


def _looks_like_captcha(html: str) -> bool:
    lowered = html.lower()
    return (
        "enter the characters you see below" in lowered
        or "to discuss automated access" in lowered
        or "captcha" in lowered[:5000]
    )


async def _goto(page: Page, url: str) -> str:
    await page.goto(url, wait_until="domcontentloaded", timeout=45_000)
    # Brief settle for client-rendered bits.
    await asyncio.sleep(random.uniform(0.8, 1.8))
    return await page.content()


# ---------------------------------------------------------------------------
# Discovery: search → ASINs + sponsored flag
# ---------------------------------------------------------------------------


def _parse_search_results(html: str) -> list[tuple[str, bool]]:
    """Return [(asin, sponsored), ...] from a search results page."""
    soup = BeautifulSoup(html, "lxml")
    out: list[tuple[str, bool]] = []
    seen: set[str] = set()
    for tile in soup.select("[data-component-type='s-search-result']"):
        asin = tile.get("data-asin") or ""
        if not asin or asin in seen:
            continue
        seen.add(asin)
        # Sponsored detection: Amazon marks sponsored tiles a few ways.
        sponsored = bool(
            tile.select_one("[data-component-type='sp-sponsored-result']")
            or tile.find(string=re.compile(r"\bSponsored\b"))
        )
        out.append((asin, sponsored))
    return out


async def discover_asins(
    page: Page,
    queries: list[str],
    target_count: int,
    max_pages_per_query: int = 4,
    delay_min: float = 2.0,
    delay_max: float = 5.0,
) -> list[tuple[str, bool, str]]:
    """Search each query across N pages, collect ASINs.

    Returns [(asin, sponsored_in_search, discovered_via_query), ...].
    Stops early if target_count unique ASINs found.
    """
    collected: dict[str, tuple[bool, str]] = {}
    for query in queries:
        if len(collected) >= target_count:
            break
        for page_num in range(1, max_pages_per_query + 1):
            url = SEARCH_URL.format(query=query.replace(" ", "+"), page=page_num)
            try:
                html = await _goto(page, url)
            except Exception as e:
                print(f"  [search] {query!r} page {page_num}: error {e}")
                continue
            if _looks_like_captcha(html):
                print(f"  [search] CAPTCHA on {query!r} page {page_num}, stopping query")
                break
            results = _parse_search_results(html)
            if not results:
                print(f"  [search] {query!r} page {page_num}: no tiles parsed")
                break
            new_count = 0
            for asin, sponsored in results:
                if asin not in collected:
                    collected[asin] = (sponsored, query)
                    new_count += 1
            print(
                f"  [search] {query!r} page {page_num}: +{new_count} new "
                f"(total {len(collected)})"
            )
            await _polite_sleep(delay_min, delay_max)
            if len(collected) >= target_count:
                break
    return [(asin, sp, q) for asin, (sp, q) in collected.items()]


# ---------------------------------------------------------------------------
# PDP parsing
# ---------------------------------------------------------------------------


_PRICE_RE = re.compile(r"\$?\s*([0-9][0-9,]*\.?[0-9]*)")


def _to_float(s: str | None) -> float | None:
    if not s:
        return None
    m = _PRICE_RE.search(s.replace(",", ""))
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", ""))
    except ValueError:
        return None


def _to_int(s: str | None) -> int | None:
    if not s:
        return None
    digits = re.sub(r"[^0-9]", "", s)
    return int(digits) if digits else None


def parse_pdp(html: str, asin: str) -> ScrapeResult:
    result = ScrapeResult(asin=asin)
    soup = BeautifulSoup(html, "lxml")

    # Title
    title_el = soup.select_one("#productTitle")
    if title_el:
        result.title = title_el.get_text(strip=True)
    else:
        result.parse_errors.append("no_title")

    # Feature bullets
    bullets: list[str] = []
    for li in soup.select("#feature-bullets ul li"):
        text = li.get_text(" ", strip=True)
        if text and "hide" not in (li.get("class") or []):
            bullets.append(text)
    result.bullets = bullets

    # Description
    desc_el = soup.select_one("#productDescription")
    if desc_el:
        result.description = desc_el.get_text(" ", strip=True)

    # Price — try several locations.
    price_text = None
    for sel in [
        "#corePriceDisplay_desktop_feature_div .a-offscreen",
        "#corePrice_feature_div .a-offscreen",
        ".priceToPay .a-offscreen",
        "#price .a-offscreen",
        ".a-price .a-offscreen",
    ]:
        el = soup.select_one(sel)
        if el and el.get_text(strip=True):
            price_text = el.get_text(strip=True)
            break
    result.price_str = price_text
    result.price = _to_float(price_text)

    # Average rating
    rating_el = soup.select_one("#acrPopover")
    if rating_el:
        title_attr = rating_el.get("title", "")
        m = re.search(r"([0-9.]+)\s+out of", title_attr)
        if m:
            try:
                result.avg_rating = float(m.group(1))
            except ValueError:
                pass
    if result.avg_rating is None:
        alt = soup.select_one("[data-hook='rating-out-of-text']")
        if alt:
            m = re.search(r"([0-9.]+)", alt.get_text())
            if m:
                try:
                    result.avg_rating = float(m.group(1))
                except ValueError:
                    pass

    # Number of reviews
    nr_el = soup.select_one("#acrCustomerReviewText")
    if nr_el:
        result.num_reviews = _to_int(nr_el.get_text())

    # Review excerpts. Amazon selectors as of late 2025:
    #   [data-hook="reviewText"]  — review body
    #   [data-hook="review-body"] — older selector, kept as fallback
    reviews: list[str] = []
    seen_review_texts: set[str] = set()
    for sel in ["[data-hook='reviewText']", "[data-hook='review-body']"]:
        for body in soup.select(sel):
            text = body.get_text(" ", strip=True)
            # Strip Amazon's accessibility prefix and trailing expand/collapse UI.
            text = re.sub(
                r"^(Brief content visible[^.]*\.\s*)?"
                r"(Full content visible[^.]*\.\s*)?",
                "",
                text,
            )
            text = re.sub(r"\s*Read more\s*Read less\s*$", "", text).strip()
            if text and text not in seen_review_texts:
                seen_review_texts.add(text)
                reviews.append(text[:800])
    result.review_excerpts = reviews

    # Brand — from the byline link near the title.
    byline = soup.select_one("#bylineInfo")
    if byline:
        text = byline.get_text(" ", strip=True)
        # Amazon often renders as "Visit the X Store" or "Brand: X".
        m = re.search(r"(?:Visit the\s+|Brand:\s*|Stores\s*)?(.+?)(?:\s+Store)?$", text)
        if m:
            result.brand = m.group(1).strip()

    # Spec table — merge all `table.a-keyvalue` (Amazon's structured-spec component).
    # Each table has rows of th/td pairs. There are typically 5–15 such tables
    # grouping different aspects (general, display, connectivity, ports, etc.).
    specs: dict[str, str] = {}
    for t in soup.select("table.a-keyvalue"):
        for row in t.select("tr"):
            th = row.find("th")
            td = row.find("td")
            if not th or not td:
                continue
            key = th.get_text(" ", strip=True)
            val = td.get_text(" ", strip=True)
            # Skip empties and noisy duplicates (first occurrence wins).
            if key and val and key not in specs:
                specs[key] = val
    # Also extract brand from the spec table if we didn't get it from the byline.
    # Different category templates use different keys; check the most common ones.
    if not result.brand:
        for key in ("Brand", "Brand Name", "Manufacturer", "Brand Name "):
            if key in specs and specs[key]:
                result.brand = specs[key].strip()
                break
    result.spec_table = specs

    return result


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _load_existing_asins(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    asins: set[str] = set()
    with out_path.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
                asin = rec.get("asin")
                if asin:
                    asins.add(asin)
            except json.JSONDecodeError:
                continue
    return asins


async def scrape_category(
    category: str,
    seed_queries: list[str],
    target_products: int,
    output_dir: Path,
    reviews_per_product: int = 8,
    proxy: str | None = None,
    headless: bool = True,
) -> None:
    """Scrape one category. Writes JSONL incrementally so partial runs are useful."""
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "_raw"
    raw_dir.mkdir(exist_ok=True)
    products_path = output_dir / "products.jsonl"

    existing = _load_existing_asins(products_path)
    via_proxy = bool(proxy)
    print(f"[{category}] starting. {len(existing)} ASINs already scraped. proxy={'yes' if via_proxy else 'no'}")

    # Polite delay: tight when proxied (proxy handles rotation), loose otherwise.
    delay_min, delay_max = (0.3, 0.8) if via_proxy else (2.0, 5.0)

    async with async_playwright() as p:
        launch_args = ["--ignore-certificate-errors"] if via_proxy else []
        launch_kwargs: dict[str, Any] = {"headless": headless, "args": launch_args}
        if proxy:
            launch_kwargs["proxy"] = {"server": proxy}
        browser = await p.chromium.launch(**launch_kwargs)
        ctx = await _new_context(browser, via_proxy=via_proxy)
        page = await ctx.new_page()

        # 1. Discover ASINs.
        print(f"[{category}] discovering ASINs...")
        discovered = await discover_asins(
            page=page,
            queries=seed_queries,
            target_count=target_products,
            delay_min=delay_min,
            delay_max=delay_max,
        )
        print(f"[{category}] discovered {len(discovered)} unique ASINs.")

        # 2. Fetch PDPs.
        to_fetch = [d for d in discovered if d[0] not in existing]
        print(f"[{category}] fetching {len(to_fetch)} new PDPs.")

        with products_path.open("a") as out_f:
            for i, (asin, sponsored, via_query) in enumerate(to_fetch, 1):
                url = PDP_URL.format(asin=asin)
                try:
                    html = await _goto(page, url)
                except Exception as e:
                    print(f"  [{i}/{len(to_fetch)}] {asin}: navigation error {e}")
                    continue
                if _looks_like_captcha(html):
                    print(f"  [{i}/{len(to_fetch)}] {asin}: CAPTCHA, aborting run")
                    break
                raw_path = raw_dir / f"{asin}.html"
                raw_path.write_text(html)

                result = parse_pdp(html, asin)
                result.sponsored_in_search = sponsored
                result.discovered_via_query = via_query
                result.url = url
                # Cap reviews to configured limit.
                result.review_excerpts = result.review_excerpts[:reviews_per_product]

                out_f.write(json.dumps(asdict(result)) + "\n")
                out_f.flush()
                title_preview = (result.title or "?")[:60]
                print(
                    f"  [{i}/{len(to_fetch)}] {asin}  {title_preview!r:62}  "
                    f"price={result.price}  reviews={len(result.review_excerpts)}"
                )
                await _polite_sleep(delay_min, delay_max)

        await browser.close()

    print(f"[{category}] done. Output: {products_path}")
