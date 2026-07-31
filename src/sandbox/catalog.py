"""Category data access and stateless full-catalog retrieval."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sentence_transformers import SentenceTransformer

from sandbox.index.builder import INDEX_SCHEMA_VERSION, MODEL_NAME, MODEL_REVISION
from sandbox.product_text import PRODUCT_TEXT_VERSION, serialize_product


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def category_config_path(category: str) -> Path:
    return REPO_ROOT / "configs" / f"{category}.yaml"


@lru_cache(maxsize=64)
def load_config(category: str) -> dict[str, Any]:
    path = category_config_path(category)
    if not path.is_file():
        raise ValueError(f"unsupported category: {category!r}")
    return yaml.safe_load(path.read_text())


@lru_cache(maxsize=64)
def load_catalog(category: str) -> tuple[dict[str, Any], ...]:
    path = REPO_ROOT / load_config(category)["catalog_path"]
    products: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                products.append(json.loads(line))
    return tuple(products)


def get_product(category: str, asin: str) -> dict[str, Any] | None:
    return next((product for product in load_catalog(category) if product.get("asin") == asin), None)


def get_products(category: str, asins: list[str]) -> list[dict[str, Any]]:
    by_asin = {product.get("asin"): product for product in load_catalog(category)}
    return [by_asin[asin] for asin in asins if asin in by_asin]


def product_details(category: str, asins: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "asin": product.get("asin"),
            "title": product.get("title"),
            "brand": product.get("brand"),
            "price": product.get("price"),
            "avg_rating": product.get("avg_rating"),
            "num_reviews": product.get("num_reviews"),
            "bullets": (product.get("bullets") or [])[:6],
            "description": str(product.get("description") or "")[:1200],
            "spec_table": product.get("spec_table") or {},
            "review_excerpts": [
                str(review)[:500] for review in (product.get("review_excerpts") or [])[:5]
            ],
        }
        for product in get_products(category, asins)
    ]


def catalog_overview(category: str) -> dict[str, Any]:
    products = list(load_catalog(category))
    prices = sorted(
        float(product["price"])
        for product in products
        if isinstance(product.get("price"), (int, float))
    )
    brands = Counter(product.get("brand") for product in products if product.get("brand"))
    return {
        "category": category,
        "products": len(products),
        "price_min": prices[0] if prices else None,
        "price_median": prices[len(prices) // 2] if prices else None,
        "price_max": prices[-1] if prices else None,
        "top_brands": [brand for brand, _ in brands.most_common(10)],
    }


def default_slate(category: str) -> list[dict[str, Any]]:
    asins = list(load_config(category).get("default_slate") or [])
    products = get_products(category, asins)
    if len(asins) != 3 or len(set(asins)) != 3 or len(products) != 3:
        raise ValueError(f"{category!r} default_slate must contain three unique catalog ASINs")
    return products


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ordered_asin_hash(products: list[dict[str, Any]]) -> str:
    payload = "\n".join(str(product.get("asin") or "") for product in products)
    return hashlib.sha256(payload.encode()).hexdigest()


@lru_cache(maxsize=1)
def _dense_model() -> SentenceTransformer:
    return SentenceTransformer(
        MODEL_NAME,
        revision=MODEL_REVISION,
        local_files_only=True,
    )


def warm_dense_model() -> None:
    _dense_model()


@lru_cache(maxsize=8)
def _load_index(category: str) -> tuple[np.ndarray, list[dict[str, Any]]]:
    config = load_config(category)
    index_dir = REPO_ROOT / config["index_dir"]
    embeddings_path = index_dir / "embeddings.npy"
    manifest_path = index_dir / "index_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"dense index manifest missing for {category!r}")
    manifest = json.loads(manifest_path.read_text())
    expected = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "model_id": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "product_text_version": PRODUCT_TEXT_VERSION,
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"dense index metadata mismatch for {category!r}: {mismatches}")
    embeddings = np.load(embeddings_path)
    products = list(load_catalog(category))
    actual = {
        "product_count": len(products),
        "embedding_shape": list(embeddings.shape),
        "embedding_dtype": str(embeddings.dtype),
        "catalog_sha256": _sha256(REPO_ROOT / config["catalog_path"]),
        "ordered_asin_sha256": _ordered_asin_hash(products),
        "embeddings_sha256": _sha256(embeddings_path),
    }
    stale = {
        key: (manifest.get(key), value)
        for key, value in actual.items()
        if manifest.get(key) != value
    }
    if stale:
        raise ValueError(f"dense index is stale for {category!r}: {stale}")
    return embeddings, products


def validate_category(category: str) -> None:
    config = load_config(category)
    products = list(load_catalog(category))
    asins = [product.get("asin") for product in products]
    if not products or any(not asin for asin in asins) or len(asins) != len(set(asins)):
        raise ValueError(f"{category!r} catalog has missing or duplicate ASINs")
    default_slate(category)
    questions = REPO_ROOT / config["questions_path"]
    personas = questions.with_name("personas.json")
    if not questions.is_file() or not personas.is_file():
        raise ValueError(f"{category!r} is missing questions or personas")
    _load_index(category)


def _dense_scores(category: str, query: str) -> dict[str, float]:
    embeddings, products = _load_index(category)
    vector = _dense_model().encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    )[0].astype(np.float32)
    scores = embeddings @ vector
    return {
        product["asin"]: float(scores[index])
        for index, product in enumerate(products)
        if product.get("asin")
    }


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.casefold())


@lru_cache(maxsize=64)
def _bm25_scores(category: str, query: str) -> dict[str, float]:
    products = [product for product in load_catalog(category) if product.get("asin")]
    documents = [_tokens(serialize_product(product)) for product in products]
    if not documents:
        return {}
    frequencies = [Counter(document) for document in documents]
    document_frequency: Counter[str] = Counter()
    for frequency in frequencies:
        document_frequency.update(frequency.keys())
    average_length = sum(map(len, documents)) / len(documents)
    scores: dict[str, float] = {}
    for product, document, frequency in zip(products, documents, frequencies):
        score = 0.0
        for term in _tokens(query):
            containing = document_frequency[term]
            inverse = math.log(
                1 + (len(documents) - containing + 0.5) / (containing + 0.5)
            )
            count = frequency[term]
            denominator = count + 1.2 * (0.25 + 0.75 * len(document) / average_length)
            if denominator:
                score += inverse * count * 2.2 / denominator
        scores[product["asin"]] = score
    return scores


@dataclass(frozen=True)
class RetrievalResult:
    products: list[dict[str, Any]]
    dense_scores: dict[str, float]
    lane_sources: dict[str, list[str]] = field(default_factory=dict)
    eligible_count: int = 0


def retrieve_catalog(
    category: str,
    query: str,
    *,
    focus_query: str | None = None,
    max_price: float | None = None,
    limit: int = 15,
) -> RetrievalResult:
    """Round-robin dense, BM25, and focused-dense recall over the full catalog."""
    if limit < 1:
        raise ValueError("retrieval limit must be positive")
    eligible = [
        product
        for product in load_catalog(category)
        if product.get("asin")
        and isinstance(product.get("price"), (int, float))
        and (max_price is None or float(product["price"]) <= max_price)
    ]
    dense = _dense_scores(category, query)
    lexical = _bm25_scores(category, query)
    lane_specs: list[tuple[str, dict[str, float]]] = [
        ("dense", dense),
        ("bm25", lexical),
    ]
    focused_query = (focus_query or "").strip()
    if focused_query and focused_query.casefold() != query.strip().casefold():
        lane_specs.append(("focused_dense", _dense_scores(category, focused_query)))
    lanes = [
        sorted(
            eligible,
            key=lambda product, values=values: (-values[product["asin"]], product["asin"]),
        )
        for _, values in lane_specs
    ]
    selected: list[dict[str, Any]] = []
    sources: dict[str, list[str]] = {}
    seen: set[str] = set()
    position = 0
    while len(selected) < limit and any(position < len(lane) for lane in lanes):
        for (name, _), lane in zip(lane_specs, lanes):
            if position >= len(lane):
                continue
            product = lane[position]
            asin = product["asin"]
            sources.setdefault(asin, []).append(name)
            if asin not in seen:
                selected.append(product)
                seen.add(asin)
                if len(selected) == limit:
                    break
        position += 1
    return RetrievalResult(
        products=selected,
        dense_scores={product["asin"]: dense[product["asin"]] for product in selected},
        lane_sources={product["asin"]: sources[product["asin"]] for product in selected},
        eligible_count=len(eligible),
    )
