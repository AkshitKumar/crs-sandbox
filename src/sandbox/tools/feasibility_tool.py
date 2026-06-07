"""check_category_supported: does our catalog cover this kind of product?

Per-category centroid embedding is computed once (here, cached to disk) and
loaded at runtime. When the user says "I want X", we embed the query, take
the max cosine similarity against per-category centroids, and decide whether
to proceed with the best-matching category or apologize.

This is what lets the agent say "Sorry, I don't carry power tools" instead
of inventing recommendations from an irrelevant catalog.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from sandbox.catalog import REPO_ROOT, load_config


DEFAULT_MODEL = "BAAI/bge-large-en-v1.5"
# A query is supported when (a) absolute similarity to the best category is high
# enough AND (b) it dominates the second-best category by a margin. With only
# 2 categories indexed today, the margin is the load-bearing signal.
#
# BEFORE the embedding-based check runs, we look for a direct keyword match
# (e.g., user literally said "laptop"). That's the most reliable signal of
# intent and trumps any cosine-based margin.
CONFIDENCE_THRESHOLD = 0.55  # cosine to best category must exceed this
MARGIN_THRESHOLD = 0.08      # best - second_best must exceed this
SUGGEST_THRESHOLD = 0.40     # below this → "we don't carry that" rather than guessing


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    return SentenceTransformer(DEFAULT_MODEL)


def _category_dir(category: str) -> Path:
    return REPO_ROOT / Path(load_config(category)["catalog_path"]).parent


def _centroid_path(category: str) -> Path:
    return _category_dir(category) / "index" / "centroid.npy"


def list_available_categories() -> list[str]:
    """All categories that have an index built."""
    configs_dir = REPO_ROOT / "configs"
    cats = []
    for yml in sorted(configs_dir.glob("*.yaml")):
        cat = yml.stem
        if (REPO_ROOT / Path(load_config(cat)["index_dir"]) / "embeddings.npy").exists():
            cats.append(cat)
    return cats


def build_centroid(category: str) -> np.ndarray:
    """Compute (and cache) the centroid embedding for a category."""
    path = _centroid_path(category)
    if path.exists():
        return np.load(path)

    index_dir = REPO_ROOT / Path(load_config(category)["index_dir"])
    embeddings = np.load(index_dir / "embeddings.npy")
    centroid = embeddings.mean(axis=0)
    # L2-normalize so dot product with normalized queries is cosine.
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid = centroid / norm
    np.save(path, centroid.astype(np.float32))
    return centroid.astype(np.float32)


@lru_cache(maxsize=64)
def _category_centroids() -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for cat in list_available_categories():
        out[cat] = build_centroid(cat)
    return out


def _category_aliases(cat: str) -> list[str]:
    """Return naive surface forms for a category name (singular + plural)."""
    base = cat.lower().replace("_", " ").strip()
    forms = {base}
    if base.endswith("y") and not base.endswith(("ay", "ey", "iy", "oy", "uy")):
        forms.add(base[:-1] + "ies")
    elif base.endswith("s"):
        pass
    else:
        forms.add(base + "s")
    return sorted(forms, key=lambda s: -len(s))  # longest first


def _keyword_match(query: str, categories: list[str]) -> str | None:
    """Return the category whose name (or simple pluralization) appears as a
    whole-word match in the query. None if no clean match.
    """
    q_lower = " " + re.sub(r"[^a-z0-9 ]+", " ", query.lower()) + " "
    best: tuple[int, str] | None = None  # (form length, category)
    for cat in categories:
        for form in _category_aliases(cat):
            if f" {form} " in q_lower:
                if best is None or len(form) > best[0]:
                    best = (len(form), cat)
                break
    return best[1] if best else None


def check_category_supported(query: str) -> dict[str, Any]:
    """Decide which catalog category the user is asking about.

    Order of evidence:
        1. Explicit keyword match in the query (most reliable).
        2. Embedding similarity to per-category centroids with margin check.

    Returns:
        {
          "best_category":         str | None,
          "best_score":            float,
          "margin_over_runner_up": float,
          "supported":             bool,
          "should_suggest":        bool,
          "matched_via":           "keyword" | "embedding" | "none",
          "ranked":                [(category, score), ...],
        }
    """
    centroids = _category_centroids()
    if not centroids:
        return {
            "best_category": None,
            "best_score": 0.0,
            "margin_over_runner_up": 0.0,
            "supported": False,
            "should_suggest": False,
            "matched_via": "none",
            "ranked": [],
            "note": "no categories indexed yet",
        }

    available_cats = list(centroids.keys())

    # 1. Keyword fast-path.
    kw_match = _keyword_match(query, available_cats)
    if kw_match is not None:
        # Still compute embedding scores for the audit trail.
        q_emb = _model().encode([query], normalize_embeddings=True, convert_to_numpy=True)[0].astype(np.float32)
        ranked = sorted(
            ((cat, float(np.dot(q_emb, centroid))) for cat, centroid in centroids.items()),
            key=lambda x: x[1],
            reverse=True,
        )
        best_score = next((s for c, s in ranked if c == kw_match), 0.0)
        margin = best_score - max((s for c, s in ranked if c != kw_match), default=0.0)
        return {
            "best_category": kw_match,
            "best_score": best_score,
            "margin_over_runner_up": margin,
            "supported": True,
            "should_suggest": True,
            "matched_via": "keyword",
            "ranked": ranked,
        }

    # 2. Embedding-based fallback.
    q_emb = _model().encode([query], normalize_embeddings=True, convert_to_numpy=True)[0].astype(np.float32)
    ranked = sorted(
        ((cat, float(np.dot(q_emb, centroid))) for cat, centroid in centroids.items()),
        key=lambda x: x[1],
        reverse=True,
    )
    best_cat, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = best_score - second_score
    supported = best_score >= CONFIDENCE_THRESHOLD and margin >= MARGIN_THRESHOLD
    return {
        "best_category": best_cat if supported else None,
        "best_score": best_score,
        "margin_over_runner_up": margin,
        "supported": supported,
        "should_suggest": best_score >= SUGGEST_THRESHOLD and margin >= MARGIN_THRESHOLD * 0.5,
        "matched_via": "embedding" if supported else "none",
        "ranked": ranked,
    }
