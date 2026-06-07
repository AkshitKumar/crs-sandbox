"""Semantic search + narrow_search over a CandidateBus.

`semantic_search` ranks ALL products in a category by cosine similarity to a
natural-language query and keeps the top-K as the bus contents.

`narrow_search` does the same but only over the ASINs currently in the bus —
the agent uses this after a `filter` pass to re-rank what's left.

Both tools cache the bge model so subsequent calls within a session are cheap.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from sandbox.catalog import REPO_ROOT, load_config
from sandbox.tools.candidate_bus import CandidateBus


DEFAULT_MODEL = "BAAI/bge-large-en-v1.5"


@lru_cache(maxsize=1)
def _model(model_name: str = DEFAULT_MODEL) -> SentenceTransformer:
    return SentenceTransformer(model_name)


@lru_cache(maxsize=8)
def _load_index(category: str) -> tuple[np.ndarray, list[dict]]:
    """Return (embeddings, products_in_index_order) for a category."""
    config = load_config(category)
    index_dir = REPO_ROOT / config["index_dir"]
    embeddings = np.load(index_dir / "embeddings.npy")
    products = json.loads((index_dir / "products.json").read_text())
    return embeddings, products


def _score_query(category: str, query: str) -> tuple[np.ndarray, list[dict]]:
    """Return (per-product cosine similarities, products) for a query."""
    embeddings, products = _load_index(category)
    q_emb = _model().encode([query], normalize_embeddings=True, convert_to_numpy=True)[0].astype(np.float32)
    with np.errstate(all="ignore"):
        sims = embeddings @ q_emb
    return sims, products


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
