"""Ranking tools: re-rank a CandidateBus by various objectives.

Three tools, in order of complexity:
    - `rank_by_match`     — weighted combo of semantic sim + rating + popularity
    - `rank_by_commission`— P(purchase) × price, the v0 commission-objective ranker
    - `rank_by_price`     — simple price ascending / descending (utility)

All operate over the bus's current ASIN set; none re-introduce filtered-out items.

`rank_by_commission` uses a hand-tuned logistic for P(purchase). The coefficients
will be re-fit from buyer-simulator data once that pipeline is up; the v0 form
is good enough to drive meaningful rank differences and test the agent's logic.

P(purchase | persona, item) ≈ sigmoid(
    β_0
  + β_1 * semantic_sim                  # match to elicited preferences
  + β_2 * z(rating)                     # higher rating helps
  + β_3 * z(log_reviews)                # social proof
  - β_4 * z(price)                      # price hurts
  + β_5 * 1{price ≤ elicited_budget}    # within budget is a big boost
)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from sandbox.catalog import load_catalog
from sandbox.tools.candidate_bus import CandidateBus


# ---------------------------------------------------------------------------
# Helpers: per-catalog normalization
# ---------------------------------------------------------------------------


def _normalizers(catalog: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Compute z-score normalizers (mean/std) for price, rating, log_reviews
    across the catalog. Cached at call time (caller can hold the result)."""
    prices = np.array([p["price"] for p in catalog if p.get("price") is not None], dtype=float)
    ratings = np.array([p["avg_rating"] for p in catalog if p.get("avg_rating") is not None], dtype=float)
    log_revs = np.array([math.log1p(p["num_reviews"]) for p in catalog if p.get("num_reviews") is not None], dtype=float)

    def _stats(xs: np.ndarray) -> dict[str, float]:
        if xs.size == 0:
            return {"mean": 0.0, "std": 1.0}
        m = float(xs.mean())
        s = float(xs.std()) or 1.0
        return {"mean": m, "std": s}

    return {
        "price": _stats(prices),
        "rating": _stats(ratings),
        "log_reviews": _stats(log_revs),
    }


def _safe_z(value: float | None, stats: dict[str, float], fallback: float = 0.0) -> float:
    if value is None:
        return fallback
    return (float(value) - stats["mean"]) / stats["std"]


def _safe_log_revs(num_reviews: int | None) -> float:
    if num_reviews is None or num_reviews <= 0:
        return 0.0
    return math.log1p(float(num_reviews))


# ---------------------------------------------------------------------------
# rank_by_match: weighted combo of semantic + rating + popularity
# ---------------------------------------------------------------------------


@dataclass
class MatchWeights:
    semantic: float = 1.0
    rating: float = 0.0
    popularity: float = 0.0
    price_penalty: float = 0.0  # subtract `price_penalty * z(price)`


def rank_by_match(
    bus: CandidateBus, weights: MatchWeights | dict[str, float] | None = None
) -> CandidateBus:
    """Reorder the bus by a weighted combination of attributes.

    Reuses the per-asin semantic scores currently held in `bus.scores` (set
    by the most recent `semantic_search` / `narrow_search` call). Items
    without a score get 0 for that component.
    """
    if weights is None:
        weights = MatchWeights()
    elif isinstance(weights, dict):
        weights = MatchWeights(**{k: v for k, v in weights.items() if k in MatchWeights.__dataclass_fields__})

    catalog = list(load_catalog(bus.category))
    by_asin = {p["asin"]: p for p in catalog if p.get("asin")}
    norms = _normalizers(catalog)

    new_scores: dict[str, float] = {}
    for asin in bus.asins:
        p = by_asin.get(asin)
        if p is None:
            continue
        sem = bus.scores.get(asin, 0.0)
        rating_z = _safe_z(p.get("avg_rating"), norms["rating"])
        pop_z = _safe_z(_safe_log_revs(p.get("num_reviews")), norms["log_reviews"])
        price_z = _safe_z(p.get("price"), norms["price"])
        score = (
            weights.semantic * sem
            + weights.rating * rating_z
            + weights.popularity * pop_z
            - weights.price_penalty * price_z
        )
        new_scores[asin] = score

    ordered = sorted(new_scores.keys(), key=lambda a: new_scores[a], reverse=True)
    note = (
        f"rank_by_match(sem={weights.semantic}, rating={weights.rating}, "
        f"pop={weights.popularity}, -price={weights.price_penalty})"
    )
    return bus.reorder(ordered, new_scores, note=note)


# ---------------------------------------------------------------------------
# rank_by_commission: expected revenue per impression
# ---------------------------------------------------------------------------


@dataclass
class PurchaseModel:
    """Hand-tuned v0 logistic for P(purchase | item, elicited preferences)."""

    beta_0: float = -1.0
    beta_semantic: float = 2.5
    beta_rating: float = 0.5
    beta_log_reviews: float = 0.3
    beta_price: float = -0.6
    beta_in_budget: float = 1.5   # large jump for fitting under the user's hard budget cap

    def prob(
        self,
        semantic_sim: float,
        rating_z: float,
        log_revs_z: float,
        price_z: float,
        in_budget: bool,
    ) -> float:
        x = (
            self.beta_0
            + self.beta_semantic * semantic_sim
            + self.beta_rating * rating_z
            + self.beta_log_reviews * log_revs_z
            + self.beta_price * price_z
            + self.beta_in_budget * (1.0 if in_budget else 0.0)
        )
        return 1.0 / (1.0 + math.exp(-x))


def rank_by_commission(
    bus: CandidateBus,
    budget_max: float | None = None,
    model: PurchaseModel | None = None,
) -> CandidateBus:
    """Rank by expected commission ≈ price × P(purchase).

    `budget_max` is the user's elicited hard price cap; items at or below
    this get a P(purchase) boost. If None, the in-budget term is dropped.
    """
    model = model or PurchaseModel()
    catalog = list(load_catalog(bus.category))
    by_asin = {p["asin"]: p for p in catalog if p.get("asin")}
    norms = _normalizers(catalog)

    new_scores: dict[str, float] = {}
    for asin in bus.asins:
        p = by_asin.get(asin)
        if p is None or p.get("price") is None:
            continue
        sem = bus.scores.get(asin, 0.0)
        rating_z = _safe_z(p.get("avg_rating"), norms["rating"])
        log_revs_z = _safe_z(_safe_log_revs(p.get("num_reviews")), norms["log_reviews"])
        price_z = _safe_z(p.get("price"), norms["price"])
        in_budget = budget_max is not None and p["price"] <= budget_max

        p_buy = model.prob(sem, rating_z, log_revs_z, price_z, in_budget)
        expected_commission = float(p["price"]) * p_buy
        new_scores[asin] = expected_commission

    ordered = sorted(new_scores.keys(), key=lambda a: new_scores[a], reverse=True)
    note = f"rank_by_commission(budget≤{budget_max})" if budget_max else "rank_by_commission"
    return bus.reorder(ordered, new_scores, note=note)


# ---------------------------------------------------------------------------
# rank_by_price: trivial utility
# ---------------------------------------------------------------------------


def rank_by_price(bus: CandidateBus, ascending: bool = True) -> CandidateBus:
    catalog = list(load_catalog(bus.category))
    by_asin = {p["asin"]: p for p in catalog if p.get("asin")}
    ordered = sorted(
        bus.asins,
        key=lambda a: (by_asin.get(a, {}).get("price") or float("inf")),
        reverse=not ascending,
    )
    return bus.reorder(ordered, None, note=f"rank_by_price({'asc' if ascending else 'desc'})")
