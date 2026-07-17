"""Shared category-agnostic product-to-text format for retrieval and ranking."""

from __future__ import annotations

from typing import Any


PRODUCT_TEXT_VERSION = "spec-first-v1"


def serialize_product(product: dict[str, Any]) -> str:
    lines = [f"Title: {product.get('title') or ''}"]
    if product.get("brand"):
        lines.append(f"Brand: {product['brand']}")
    price = product.get("price")
    if isinstance(price, (int, float)):
        lines.append(f"Price: ${float(price):.2f}")
    rating = product.get("avg_rating")
    if isinstance(rating, (int, float)):
        lines.append(f"Rating: {float(rating):.1f} stars")

    specs = product.get("spec_table") or {}
    if specs:
        spec_text = "; ".join(f"{key}: {value}" for key, value in specs.items())
        lines.append(f"Specifications: {spec_text[:2400]}")

    bullets = product.get("bullets") or []
    if not isinstance(bullets, list):
        bullets = [bullets]
    if bullets:
        lines.append("Features: " + "; ".join(str(item) for item in bullets[:6])[:1800])
    description = product.get("description") or ""
    if description:
        lines.append(f"Description: {str(description)[:600]}")
    return "\n".join(lines)
