"""Inspection tools: zoom in on individual products or summarize the catalog.

    - `get_product_details(asin)`  → full structured record for one product.
    - `compare(asins, aspects)`    → side-by-side attribute table.
    - `catalog_overview(category)` → high-level snapshot of the catalog.

None of these mutate the CandidateBus; they return data the agent can fold
into its prompt or relay to the user.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from sandbox.catalog import get_product, load_catalog


# ---------------------------------------------------------------------------
# get_product_details
# ---------------------------------------------------------------------------


def get_product_details(category: str, asin: str) -> dict[str, Any] | None:
    """Return the full product record, with review excerpts trimmed for prompt-fitness."""
    p = get_product(category, asin)
    if p is None:
        return None
    # Return a copy with a few prompt-friendly shapings.
    out = {
        "asin": p["asin"],
        "title": p.get("title"),
        "brand": p.get("brand"),
        "price": p.get("price"),
        "price_str": p.get("price_str"),
        "avg_rating": p.get("avg_rating"),
        "num_reviews": p.get("num_reviews"),
        "bullets": p.get("bullets") or [],
        "description": (p.get("description") or "")[:1500] or None,
        "spec_table": p.get("spec_table") or {},
        "review_excerpts": [r[:400] for r in (p.get("review_excerpts") or [])[:5]],
        "url": p.get("url"),
        "sponsored_in_search": p.get("sponsored_in_search", False),
    }
    return out


# ---------------------------------------------------------------------------
# compare: side-by-side table
# ---------------------------------------------------------------------------


def compare(
    category: str, asins: list[str], aspects: list[str] | None = None
) -> dict[str, Any]:
    """Return a structured side-by-side comparison.

    `aspects`: spec-table keys to include. If None, defaults to the universally
    populated fields: price, rating, num_reviews, brand, plus the top 8
    spec_table keys that appear in all of the requested asins.
    """
    products = [get_product(category, a) for a in asins]
    products = [p for p in products if p is not None]
    if not products:
        return {"asins": asins, "products": [], "aspects": [], "table": {}}

    # Build the aspect list if not provided.
    if aspects is None:
        aspects = ["title", "brand", "price", "avg_rating", "num_reviews"]
        # Pick spec_table keys present in ALL listed products.
        common_specs = set((products[0].get("spec_table") or {}).keys())
        for p in products[1:]:
            common_specs &= set((p.get("spec_table") or {}).keys())
        # Take the 10 most useful — sort by length of value (shorter = more atomic).
        ranked = sorted(
            common_specs,
            key=lambda k: -sum(1 for p in products if (p.get("spec_table") or {}).get(k)),
        )
        aspects.extend(ranked[:10])

    table: dict[str, list[Any]] = {}
    for asp in aspects:
        row = []
        for p in products:
            if asp in {"title", "brand", "price", "avg_rating", "num_reviews"}:
                v = p.get(asp)
            else:
                v = (p.get("spec_table") or {}).get(asp)
            row.append(v)
        table[asp] = row

    return {
        "asins": [p["asin"] for p in products],
        "aspects": aspects,
        "table": table,
    }


# ---------------------------------------------------------------------------
# catalog_overview: high-level snapshot
# ---------------------------------------------------------------------------


def catalog_overview(category: str) -> dict[str, Any]:
    """High-level snapshot: how many products, price range, top brands, top features.

    Read by the agent before it makes promises about what the catalog can offer.
    Cheap, no LLM call.
    """
    catalog = list(load_catalog(category))
    n = len(catalog)
    if n == 0:
        return {"category": category, "n_products": 0}

    prices = sorted(p["price"] for p in catalog if p.get("price") is not None)
    brands = Counter(p["brand"] for p in catalog if p.get("brand"))
    feature_terms: Counter[str] = Counter()
    for p in catalog:
        for b in (p.get("bullets") or [])[:3]:
            # Crude tokenization for "popular feature words"
            for token in b.lower().split():
                token = token.strip(".,!()-—:;\"")
                if len(token) > 4 and token.isalpha():
                    feature_terms[token] += 1

    return {
        "category": category,
        "n_products": n,
        "price_min": prices[0] if prices else None,
        "price_p25": prices[len(prices) // 4] if prices else None,
        "price_median": prices[len(prices) // 2] if prices else None,
        "price_p75": prices[3 * len(prices) // 4] if prices else None,
        "price_max": prices[-1] if prices else None,
        "top_brands": [{"brand": b, "n": n} for b, n in brands.most_common(10)],
        "common_feature_terms": [t for t, _ in feature_terms.most_common(20)],
    }
