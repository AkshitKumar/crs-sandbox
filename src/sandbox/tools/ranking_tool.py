"""Optional objective-specific ranking tools available to the ReAct agent.

Fixed policies do not assign category-specific weights. The agent may still
invoke an explicit ranking objective when the conversation or experiment calls
for it; final selection is bounded and validated by the shared pipeline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from sandbox.catalog import load_catalog
from sandbox.tools.candidate_bus import CandidateBus


def _normalizers(catalog: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    prices = np.array([p["price"] for p in catalog if p.get("price") is not None], dtype=float)
    ratings = np.array([p["avg_rating"] for p in catalog if p.get("avg_rating") is not None], dtype=float)
    log_reviews = np.array(
        [math.log1p(p["num_reviews"]) for p in catalog if p.get("num_reviews") is not None],
        dtype=float,
    )

    def stats(values: np.ndarray) -> dict[str, float]:
        if values.size == 0:
            return {"mean": 0.0, "std": 1.0}
        return {"mean": float(values.mean()), "std": float(values.std()) or 1.0}

    return {
        "price": stats(prices),
        "rating": stats(ratings),
        "log_reviews": stats(log_reviews),
    }


def _safe_z(value: float | None, stats: dict[str, float], fallback: float = 0.0) -> float:
    if value is None:
        return fallback
    return (float(value) - stats["mean"]) / stats["std"]


def _log_reviews(num_reviews: int | None) -> float | None:
    if num_reviews is None:
        return None
    return math.log1p(max(0, float(num_reviews)))


@dataclass
class MatchWeights:
    semantic: float = 1.0
    rating: float = 0.0
    popularity: float = 0.0
    price_penalty: float = 0.0


def rank_by_match(
    bus: CandidateBus,
    weights: MatchWeights | dict[str, float] | None = None,
) -> CandidateBus:
    if weights is None:
        weights = MatchWeights()
    elif isinstance(weights, dict):
        weights = MatchWeights(
            **{
                key: value
                for key, value in weights.items()
                if key in MatchWeights.__dataclass_fields__
            }
        )

    catalog = list(load_catalog(bus.category))
    by_asin = {product["asin"]: product for product in catalog if product.get("asin")}
    norms = _normalizers(catalog)
    scores: dict[str, float] = {}
    for asin in bus.asins:
        product = by_asin.get(asin)
        if product is None:
            continue
        scores[asin] = (
            weights.semantic * bus.scores.get(asin, 0.0)
            + weights.rating * _safe_z(product.get("avg_rating"), norms["rating"])
            + weights.popularity
            * _safe_z(_log_reviews(product.get("num_reviews")), norms["log_reviews"])
            - weights.price_penalty * _safe_z(product.get("price"), norms["price"])
        )
    ordered = sorted(scores, key=scores.get, reverse=True)
    return bus.reorder(
        ordered,
        scores,
        note=(
            f"rank_by_match(sem={weights.semantic}, rating={weights.rating}, "
            f"pop={weights.popularity}, -price={weights.price_penalty})"
        ),
    )


@dataclass
class PurchaseModel:
    """Illustrative hand-tuned objective used only for commission experiments."""

    beta_0: float = -1.0
    beta_semantic: float = 2.5
    beta_rating: float = 0.5
    beta_log_reviews: float = 0.3
    beta_price: float = -0.6
    beta_in_budget: float = 1.5

    def prob(
        self,
        semantic_sim: float,
        rating_z: float,
        log_reviews_z: float,
        price_z: float,
        in_budget: bool,
    ) -> float:
        value = (
            self.beta_0
            + self.beta_semantic * semantic_sim
            + self.beta_rating * rating_z
            + self.beta_log_reviews * log_reviews_z
            + self.beta_price * price_z
            + self.beta_in_budget * float(in_budget)
        )
        return 1.0 / (1.0 + math.exp(-value))


def rank_by_commission(
    bus: CandidateBus,
    budget_max: float | None = None,
    model: PurchaseModel | None = None,
) -> CandidateBus:
    model = model or PurchaseModel()
    catalog = list(load_catalog(bus.category))
    by_asin = {product["asin"]: product for product in catalog if product.get("asin")}
    norms = _normalizers(catalog)
    scores: dict[str, float] = {}
    for asin in bus.asins:
        product = by_asin.get(asin)
        if product is None or product.get("price") is None:
            continue
        probability = model.prob(
            bus.scores.get(asin, 0.0),
            _safe_z(product.get("avg_rating"), norms["rating"]),
            _safe_z(_log_reviews(product.get("num_reviews")), norms["log_reviews"]),
            _safe_z(product.get("price"), norms["price"]),
            budget_max is not None and product["price"] <= budget_max,
        )
        scores[asin] = float(product["price"]) * probability
    ordered = sorted(scores, key=scores.get, reverse=True)
    note = f"rank_by_commission(budget≤{budget_max})" if budget_max else "rank_by_commission"
    return bus.reorder(ordered, scores, note=note)


def rank_by_price(bus: CandidateBus, ascending: bool = True) -> CandidateBus:
    catalog = list(load_catalog(bus.category))
    by_asin = {product["asin"]: product for product in catalog if product.get("asin")}
    ordered = sorted(
        bus.asins,
        key=lambda asin: by_asin.get(asin, {}).get("price") or float("inf"),
        reverse=not ascending,
    )
    return bus.reorder(
        ordered,
        None,
        note=f"rank_by_price({'asc' if ascending else 'desc'})",
    )
