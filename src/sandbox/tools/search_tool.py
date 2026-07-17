"""Semantic search + narrow_search over a CandidateBus.

`semantic_search` ranks ALL products in a category by cosine similarity to a
natural-language query and keeps the top-K as the bus contents.

`narrow_search` does the same but only over the ASINs currently in the bus —
the agent uses this after a `filter` pass to re-rank what's left.

Both tools cache the bge model so subsequent calls within a session are cheap.
"""

from __future__ import annotations

import json
import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from sandbox.catalog import REPO_ROOT, load_config
from sandbox.index.builder import INDEX_SCHEMA_VERSION, MODEL_NAME, MODEL_REVISION
from sandbox.product_text import PRODUCT_TEXT_VERSION
from sandbox.tools.candidate_bus import CandidateBus


DEFAULT_MODEL = MODEL_NAME


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ordered_asin_sha256(products: list[dict]) -> str:
    payload = "\n".join(str(product.get("asin") or "") for product in products)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@lru_cache(maxsize=1)
def _model(model_name: str = DEFAULT_MODEL) -> SentenceTransformer:
    # Evaluation machines build the index first, which also populates this
    # exact revision in the cache. Offline-only loading prevents an upstream
    # model update or a metadata request from changing/failing a run.
    return SentenceTransformer(
        model_name,
        revision=MODEL_REVISION,
        local_files_only=True,
    )


def warm_dense_model() -> None:
    """Load the shared embedding model before starting parallel workers."""
    _model()


@lru_cache(maxsize=8)
def _load_index(category: str) -> tuple[np.ndarray, list[dict]]:
    """Return (embeddings, products_in_index_order) for a category."""
    config = load_config(category)
    index_dir = REPO_ROOT / config["index_dir"]
    embeddings_path = index_dir / "embeddings.npy"
    products_path = index_dir / "products.json"
    manifest_path = index_dir / "index_manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"dense index manifest is missing for {category!r}; rebuild the index")
    manifest = json.loads(manifest_path.read_text())
    expected_metadata = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "model_id": DEFAULT_MODEL,
        "model_revision": MODEL_REVISION,
        "product_text_version": PRODUCT_TEXT_VERSION,
    }
    mismatches = {
        key: (manifest.get(key), expected)
        for key, expected in expected_metadata.items()
        if manifest.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"dense index metadata mismatch for {category!r}: {mismatches}")

    embeddings = np.load(embeddings_path)
    products = json.loads(products_path.read_text())
    catalog_path = REPO_ROOT / config["catalog_path"]
    actual = {
        "product_count": len(products),
        "embedding_shape": list(embeddings.shape),
        "embedding_dtype": str(embeddings.dtype),
        "catalog_sha256": _sha256(catalog_path),
        "ordered_asin_sha256": _ordered_asin_sha256(products),
        "products_sha256": _sha256(products_path),
        "embeddings_sha256": _sha256(embeddings_path),
    }
    stale = {
        key: (manifest.get(key), value)
        for key, value in actual.items()
        if manifest.get(key) != value
    }
    if stale:
        raise ValueError(f"dense index files are stale for {category!r}: {stale}")
    if embeddings.ndim != 2 or embeddings.shape[0] != len(products):
        raise ValueError(f"dense index row count does not match products for {category!r}")
    asins = [product.get("asin") for product in products]
    if any(not asin for asin in asins) or len(asins) != len(set(asins)):
        raise ValueError(f"dense index contains missing or duplicate ASINs for {category!r}")
    return embeddings, products


def _score_query(category: str, query: str) -> tuple[np.ndarray, list[dict]]:
    """Return (per-product cosine similarities, products) for a query."""
    embeddings, products = _load_index(category)
    q_emb = _model().encode([query], normalize_embeddings=True, convert_to_numpy=True)[0].astype(np.float32)
    with np.errstate(all="ignore"):
        sims = embeddings @ q_emb
    return sims, products


def semantic_scores(category: str, query: str) -> dict[str, float]:
    """Return full-catalog dense scores keyed by ASIN without mutating a bus."""
    sims, products = _score_query(category, query)
    return {
        product["asin"]: float(sims[index])
        for index, product in enumerate(products)
        if product.get("asin")
    }


def semantic_search(
    bus: CandidateBus, query: str, top_k: int = 20, full_catalog: bool = True
) -> CandidateBus:
    """Rank by semantic similarity to `query`.

    If `full_catalog=True` (default), score every product in the index and
    keep the top-K, regardless of what's currently in the bus. This is what
    you call at the start of a session to seed the candidate set.

    If `full_catalog=False`, only score ASINs currently in the bus. Use this
    after a `filter` pass to re-rank what's left without re-introducing
    filtered-out items.
    """
    sims, products = _score_query(bus.category, query)
    # Map ASIN → (sim, idx).
    asin_to_score: dict[str, float] = {
        p["asin"]: float(sims[i]) for i, p in enumerate(products) if p.get("asin")
    }

    if full_catalog:
        candidates = list(asin_to_score.keys())
    else:
        candidates = [a for a in bus.asins if a in asin_to_score]

    # Sort by score desc, take top K.
    candidates.sort(key=lambda a: asin_to_score[a], reverse=True)
    top = candidates[:top_k]

    scores = {a: asin_to_score[a] for a in top}

    if full_catalog:
        # Replace bus contents.
        bus.asins = top
        bus.scores = scores
        bus.notes.append(f"semantic_search({query!r}): scored {len(products)}, top {len(top)}")
        return bus

    return bus.reorder(top, scores, note=f"narrow_search({query!r})")


def narrow_search(bus: CandidateBus, query: str, top_k: int = 20) -> CandidateBus:
    """Convenience wrapper: re-rank within the current bus contents only."""
    return semantic_search(bus, query, top_k=top_k, full_catalog=False)
