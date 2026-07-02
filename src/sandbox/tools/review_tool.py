"""LLM-driven review tools, cached on disk.

    - `summarize_reviews(category, asin, aspect)` — 2–3 sentence summary of what
      reviewers say (overall, or focused on a specific aspect like "battery").
    - `rank_by_review_sentiment(bus, aspect)` — rerank the bus by how positively
      reviews speak about an aspect.

Cache file: data/categories/<cat>/review_cache.json, keyed by (asin, aspect|"_overall").
Calls gpt-5-mini with reasoning_effort=low (cheap; this is mostly summarization).

Cost rough estimate:
    - summarize_reviews:           ~$0.0003 per call (300 input + 100 output tokens).
    - rank_by_review_sentiment(20 items): ~$0.006 per call total.
Caching means each (asin, aspect) is paid for exactly once across the whole project.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import openai._base_client as openai_base_client
from openai import OpenAI

from sandbox.catalog import REPO_ROOT, load_catalog, get_product, load_config
from sandbox.openai_responses import OPENAI_MAX_RETRIES, response_to_text
from sandbox.tools.candidate_bus import CandidateBus


MODEL = "gpt-5-mini"
REASONING = "low"


def _client() -> OpenAI:
    openai_base_client.INITIAL_RETRY_DELAY = 2.0
    openai_base_client.MAX_RETRY_DELAY = 64.0
    return OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), max_retries=OPENAI_MAX_RETRIES)


def _cache_path(category: str) -> Path:
    return REPO_ROOT / Path(load_config(category)["catalog_path"]).parent / "review_cache.json"


def _load_cache(category: str) -> dict[str, dict[str, Any]]:
    path = _cache_path(category)
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _save_cache(category: str, cache: dict[str, dict[str, Any]]) -> None:
    _cache_path(category).write_text(json.dumps(cache))


def _cache_key(asin: str, aspect: str | None) -> str:
    return f"{asin}__{aspect or '_overall'}"


# ---------------------------------------------------------------------------
# summarize_reviews
# ---------------------------------------------------------------------------


def summarize_reviews(category: str, asin: str, aspect: str | None = None) -> dict[str, Any]:
    """Summarize what reviewers say. Returns {summary, sentiment_score, evidence}.

    sentiment_score is in [0, 1], 0 = very negative, 1 = very positive,
    0.5 = mixed/neutral. None if no reviews are available for the product.
    """
    cache = _load_cache(category)
    key = _cache_key(asin, aspect)
    if key in cache:
        return cache[key]

    product = get_product(category, asin)
    if product is None:
        return {"summary": "product not found", "sentiment_score": None, "evidence": ""}
    reviews = product.get("review_excerpts") or []
    if not reviews:
        result = {"summary": "no reviews available", "sentiment_score": None, "evidence": ""}
        cache[key] = result
        _save_cache(category, cache)
        return result

    reviews_block = "\n---\n".join(r[:400] for r in reviews[:6])
    aspect_clause = (
        f"Focus specifically on what reviewers say about: {aspect!r}. "
        if aspect
        else "Cover the most salient strengths and complaints. "
    )
    title = (product.get("title") or "")[:120]
    prompt = (
        f"You are summarizing customer reviews for a product on Amazon.\n"
        f"Product: {title}\n\n"
        f"Customer reviews:\n{reviews_block}\n\n"
        f"{aspect_clause}"
        f"Return a JSON object with exactly these fields:\n"
        f"  summary: 2-3 sentences in plain language\n"
        f"  sentiment_score: number in [0, 1] (0=very negative, 0.5=mixed, 1=very positive). "
        f"If asked about an aspect that reviews don't address, use 0.5.\n"
        f"  evidence: 1 short sentence quoting or paraphrasing the most representative review snippet."
    )

    resp = _client().responses.create(
        model=MODEL,
        input=prompt,
        reasoning={"effort": REASONING},
        text={"format": {"type": "json_object"}},
    )
    raw = response_to_text(resp) or "{}"
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {"summary": raw[:300], "sentiment_score": 0.5, "evidence": ""}

    # Normalize.
    score = parsed.get("sentiment_score")
    if isinstance(score, (int, float)):
        score = max(0.0, min(1.0, float(score)))
    else:
        score = 0.5
    result = {
        "summary": str(parsed.get("summary", ""))[:600],
        "sentiment_score": score,
        "evidence": str(parsed.get("evidence", ""))[:300],
    }

    cache[key] = result
    _save_cache(category, cache)
    return result


# ---------------------------------------------------------------------------
# rank_by_review_sentiment
# ---------------------------------------------------------------------------


def rank_by_review_sentiment(
    bus: CandidateBus, aspect: str, max_items: int = 20
) -> CandidateBus:
    """Rerank the top-`max_items` of the bus by sentiment on `aspect`.

    Items beyond max_items keep their current order. Sentiment scores
    get persisted in `bus.scores`. Note: each new (asin, aspect) pair
    costs ~$0.0003 in API spend, but is cached on disk after the first
    call so the chat UI doesn't repay.
    """
    head_asins = bus.top(max_items)
    tail_asins = bus.asins[max_items:]

    sentiments: dict[str, float] = {}
    for asin in head_asins:
        result = summarize_reviews(bus.category, asin, aspect=aspect)
        s = result["sentiment_score"]
        sentiments[asin] = float(s) if s is not None else 0.5

    head_sorted = sorted(head_asins, key=lambda a: sentiments[a], reverse=True)
    return bus.reorder(
        head_sorted + tail_asins,
        scores={**sentiments, **{a: 0.0 for a in tail_asins}},
        note=f"rank_by_review_sentiment({aspect!r}, head={len(head_asins)})",
    )
