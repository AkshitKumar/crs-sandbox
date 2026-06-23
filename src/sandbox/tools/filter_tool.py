"""Filter tool: hard constraints over the catalog.

The single most important tool for narrowing a 100–500 product catalog down
to a manageable candidate set before semantic search runs.

Two surfaces:
    - `available_filters(category)` — what attributes are filterable, and
      what values they take. The agent calls this first to know its options.
    - `preview_filter(bus, constraints)` — count survivors without changing
      the bus.
    - `apply_filter(bus, constraints)` — restrict the bus by hard constraints.

Constraint schema:
    {
      "price_max":      float | None,   # USD
      "price_min":      float | None,
      "rating_min":     float | None,   # 0.0–5.0
      "min_reviews":    int | None,
      "brand_in":       list[str] | None,   # case-insensitive substring match
      "brand_not_in":   list[str] | None,
      "spec_contains":  dict[str, str] | None,   # {field: substring} — value-side
      "spec_equals":    dict[str, str] | None,   # {field: exact_value} — case-insensitive
    }

All constraints are AND-ed. Missing constraint keys are ignored.
A product missing a field that's being constrained on is treated as not matching.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from sandbox.catalog import load_catalog
from sandbox.tools.candidate_bus import CandidateBus


# ---------------------------------------------------------------------------
# available_filters: introspect the catalog so the agent knows what's filterable
# ---------------------------------------------------------------------------


def available_filters(category: str, min_coverage: float = 0.30) -> dict[str, Any]:
    """Return the filterable attributes for a category.

    A spec field shows up only if it's populated in at least `min_coverage`
    fraction of products (default 30%) — this keeps the schema focused on
    fields that are reliably present.
    """
    catalog = load_catalog(category)
    n = len(catalog)
    if n == 0:
        return {"category": category, "n_products": 0}

    # Price + rating + reviews summary.
    prices = [p["price"] for p in catalog if p.get("price") is not None]
    ratings = [p["avg_rating"] for p in catalog if p.get("avg_rating") is not None]
    num_revs = [p["num_reviews"] for p in catalog if p.get("num_reviews") is not None]

    # Brand value list.
    brands = sorted({p["brand"].strip() for p in catalog if p.get("brand")})

    # Spec field coverage and value distributions.
    field_counts: Counter[str] = Counter()
    field_values: dict[str, Counter[str]] = {}
    for p in catalog:
        for k, v in (p.get("spec_table") or {}).items():
            field_counts[k] += 1
            field_values.setdefault(k, Counter())[v] += 1

    # Only fields that appear in ≥ min_coverage of products.
    spec_fields: dict[str, dict[str, Any]] = {}
    for field, count in field_counts.most_common():
        if count / n < min_coverage:
            continue
        top_values = field_values[field].most_common(8)
        spec_fields[field] = {
            "coverage": count,
            "coverage_pct": round(100 * count / n, 1),
            "example_values": [v for v, _ in top_values],
        }

    return {
        "category": category,
        "n_products": n,
        "price": _range_summary(prices),
        "rating": _range_summary(ratings),
        "num_reviews": _range_summary(num_revs),
        "brands": brands,
        "spec_fields": spec_fields,
    }


def _range_summary(xs: list[float]) -> dict[str, float | None]:
    if not xs:
        return {"min": None, "median": None, "max": None, "n": 0}
    s = sorted(xs)
    return {
        "min": s[0],
        "median": s[len(s) // 2],
        "max": s[-1],
        "n": len(s),
    }


# ---------------------------------------------------------------------------
# apply_filter: AND together all constraints, restrict the bus
# ---------------------------------------------------------------------------


def apply_filter(bus: CandidateBus, constraints: dict[str, Any]) -> CandidateBus:
    """Narrow the bus by hard constraints. Returns the same bus, mutated."""
    kept = _matching_asins(bus, constraints)
    note = "filter(" + ", ".join(_describe_constraints(constraints)) + ")"
    return bus.restrict(kept, note=note)


def preview_filter(bus: CandidateBus, constraints: dict[str, Any]) -> dict[str, Any]:
    """Return filter impact without mutating the bus."""
    kept = _matching_asins(bus, constraints)
    return {
        "before": bus.size(),
        "after": len(kept),
        "dropped": bus.size() - len(kept),
        "constraints": constraints,
        "note": "filter(" + ", ".join(_describe_constraints(constraints)) + ")",
        "sample_asins": kept[:5],
    }


def _matching_asins(bus: CandidateBus, constraints: dict[str, Any]) -> list[str]:
    """Compute matching ASINs for constraints over the current bus."""
    catalog = load_catalog(bus.category)
    by_asin = {p["asin"]: p for p in catalog if p.get("asin")}

    # Build a sequence of per-product predicates, each returning bool.
    preds: list = []

    # Note: each lambda captures its threshold via default-arg to avoid
    # Python's late-binding closure trap.
    if (price_max := constraints.get("price_max")) is not None:
        preds.append(lambda p, v=price_max: p.get("price") is not None and p["price"] <= v)
    if (price_min := constraints.get("price_min")) is not None:
        preds.append(lambda p, v=price_min: p.get("price") is not None and p["price"] >= v)
    if (rating_min := constraints.get("rating_min")) is not None:
        preds.append(lambda p, v=rating_min: p.get("avg_rating") is not None and p["avg_rating"] >= v)
    if (min_reviews := constraints.get("min_reviews")) is not None:
        preds.append(lambda p, v=min_reviews: p.get("num_reviews") is not None and p["num_reviews"] >= v)

    if (brands_in := constraints.get("brand_in")):
        wanted = [b.lower() for b in brands_in]
        preds.append(
            lambda p, w=wanted: (p.get("brand") or "").lower().strip() in w
            or any(s in (p.get("brand") or "").lower() for s in w)
        )
    if (brands_not := constraints.get("brand_not_in")):
        unwanted = [b.lower() for b in brands_not]
        preds.append(
            lambda p, u=unwanted: not any(s in (p.get("brand") or "").lower() for s in u)
        )

    if (sc := constraints.get("spec_contains")):
        for field, substr in sc.items():
            substrings = substr if isinstance(substr, list) else [substr]
            substrings_lower = [str(s).lower() for s in substrings]
            preds.append(
                lambda p, f=field, ss=substrings_lower: any(
                    s in str((p.get("spec_table") or {}).get(f, "")).lower()
                    for s in ss
                )
            )
    if (se := constraints.get("spec_equals")):
        for field, value in se.items():
            value_lower = value.lower()
            preds.append(
                lambda p, f=field, v=value_lower: (p.get("spec_table") or {}).get(f, "").lower().strip() == v
            )

    # Apply predicates.
    kept: list[str] = []
    for asin in bus.asins:
        p = by_asin.get(asin)
        if p and all(pred(p) for pred in preds):
            kept.append(asin)
    return kept


def _describe_constraints(c: dict[str, Any]) -> list[str]:
    """Render a short description of which constraints were applied."""
    parts = []
    if c.get("price_max") is not None:
        parts.append(f"≤${c['price_max']:.0f}")
    if c.get("price_min") is not None:
        parts.append(f"≥${c['price_min']:.0f}")
    if c.get("rating_min") is not None:
        parts.append(f"≥{c['rating_min']:.1f}★")
    if c.get("min_reviews") is not None:
        parts.append(f"≥{c['min_reviews']} revs")
    if c.get("brand_in"):
        parts.append("brand∈" + ",".join(c["brand_in"]))
    if c.get("brand_not_in"):
        parts.append("brand∉" + ",".join(c["brand_not_in"]))
    if c.get("spec_contains"):
        for k, v in c["spec_contains"].items():
            parts.append(f'{k}~"{v}"')
    if c.get("spec_equals"):
        for k, v in c["spec_equals"].items():
            parts.append(f'{k}="{v}"')
    return parts or ["no-op"]
