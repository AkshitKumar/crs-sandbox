"""Transcript-driven recommendation service shared by every policy."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

from sandbox.catalog import (
    PRODUCT_CARD_BULLET_CHAR_LIMIT,
    PRODUCT_CARD_BULLET_LIMIT,
    default_slate,
    get_products,
    retrieve_catalog,
)
from sandbox.openai_responses import (
    UsageTracker,
    create_response,
    make_client,
    response_to_text,
)


DEFAULT_MODEL = "gpt-6-luna"
PREVIOUS_RECOMMENDATION_LABEL = "[PREVIOUSLY RECOMMENDED]"
CONTINUITY_NOTE = (
    "Keep a [PREVIOUSLY RECOMMENDED] product if it is still among the best fits "
    "for what the shopper has said, and when the shopper's answers don't "
    "distinguish products, keep the earlier ones."
)


class RecommendationError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def dialogue_text(dialogue: list[dict[str, str]]) -> str:
    """Serialize the exact visible prefix without inferring a preference state."""
    labels = {"user": "Buyer", "assistant": "Recommender"}
    return "\n".join(
        f"{labels.get(item.get('role'), item.get('role', 'Unknown'))}: {item.get('content', '')}"
        for item in dialogue
    )


def render_product_cards(products: list[dict[str, Any]]) -> str:
    """Render the product facts used by both ranking and buyer evaluation."""
    cards: list[str] = []
    for number, product in enumerate(products, start=1):
        price = product.get("price")
        specs = "; ".join(
            f"{key}: {value}" for key, value in (product.get("spec_table") or {}).items()
        )[:2200]
        bullets = "; ".join(
            str(item)
            for item in (product.get("bullets") or [])[:PRODUCT_CARD_BULLET_LIMIT]
        )[:PRODUCT_CARD_BULLET_CHAR_LIMIT]
        reviews = " | ".join(
            text[:500] + ("..." if len(text) > 500 else "")
            for item in (product.get("review_excerpts") or [])[:5]
            for text in [str(item)]
        )
        cards.append(
            f"{number}. {product.get('title', '')}\n"
            f"ASIN: {product.get('asin')}\n"
            f"Price: ${float(price):.2f}; rating: {product.get('avg_rating', 'unknown')}; "
            f"ratings: {product.get('num_reviews', 'unknown')}\n"
            f"Features: {bullets or 'unavailable'}\n"
            f"Specifications: {specs or 'unavailable'}\n"
            f"Review excerpts: {reviews or 'unavailable'}"
        )
    return "\n\n".join(cards)


@dataclass(frozen=True)
class RecommendationPlan:
    focus_query: str
    max_price: float | None
    target_price: float | None = None
    min_price: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "focus_query": self.focus_query,
            "max_price": self.max_price,
            "target_price": self.target_price,
            "min_price": self.min_price,
        }


@dataclass(frozen=True)
class RecommendationResult:
    recommendations: list[dict[str, Any]]
    plan: RecommendationPlan
    retrieved_asins: list[str]
    default_slate_asins: list[str]
    carried_forward_asins: list[str]
    candidate_asins: list[str]
    lane_sources: dict[str, list[str]]
    eligible_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "recommendations": self.recommendations,
            "plan": self.plan.to_dict(),
            "retrieved_asins": self.retrieved_asins,
            "default_slate_asins": self.default_slate_asins,
            "carried_forward_asins": self.carried_forward_asins,
            "candidate_asins": self.candidate_asins,
            "lane_sources": self.lane_sources,
            "eligible_count": self.eligible_count,
        }


@dataclass
class RecommendationService:
    category: str
    model: str = DEFAULT_MODEL
    retrieval_limit: int = 15
    assortment_size: int = 3
    tracker: UsageTracker = field(default_factory=UsageTracker)
    client: OpenAI = field(default_factory=make_client)

    def __post_init__(self) -> None:
        if self.assortment_size < 1:
            raise ValueError("assortment_size must be positive")

    def recommend(
        self,
        dialogue: list[dict[str, str]],
        *,
        prior_recommendations: list[dict[str, Any]] | None = None,
    ) -> RecommendationResult:
        transcript = dialogue_text(dialogue)
        plan = self._plan(transcript)
        retrieval = retrieve_catalog(
            self.category,
            transcript,
            focus_query=plan.focus_query,
            max_price=plan.max_price,
            target_price=plan.target_price,
            min_price=plan.min_price,
            limit=self.retrieval_limit,
        )

        pool = list(retrieval.products)
        seen = {product["asin"] for product in pool}
        lane_sources = {
            asin: list(sources) for asin, sources in retrieval.lane_sources.items()
        }
        # Preserve recall for defaults at later checkpoints; turn zero still
        # returns the fixed slate directly through default_recommendation().
        default_asins: list[str] = []
        for product in default_slate(self.category):
            price = product.get("price")
            if not isinstance(price, (int, float)):
                continue
            if plan.max_price is not None and float(price) > plan.max_price:
                continue
            if plan.min_price is not None and float(price) < plan.min_price:
                continue
            asin = product["asin"]
            default_asins.append(asin)
            lane_sources.setdefault(asin, []).append("default_slate")
            if asin not in seen:
                pool.append(product)
                seen.add(asin)
        carried_forward_asins: list[str] = []
        prior_asins = [
            str(product.get("asin") or "")
            for product in (prior_recommendations or [])
            if product.get("asin")
        ]
        # Keep earlier recommendations available below the retrieval price
        # floor, while still enforcing the shopper's stated hard maximum.
        for product in get_products(self.category, prior_asins):
            price = product.get("price")
            if not isinstance(price, (int, float)):
                continue
            if plan.max_price is not None and float(price) > plan.max_price:
                continue
            asin = product["asin"]
            carried_forward_asins.append(asin)
            sources = lane_sources.setdefault(asin, [])
            if "carry_forward" not in sources:
                sources.append("carry_forward")
            if asin not in seen:
                pool.append(product)
                seen.add(asin)
        if len(pool) < self.assortment_size:
            raise RecommendationError(
                "INSUFFICIENT_BUDGET_ELIGIBLE_PRODUCTS",
                f"only {len(pool)} products are eligible for a "
                f"{self.assortment_size}-product recommendation",
            )
        selection_pool = pool
        if carried_forward_asins:
            prior_asin_set = set(carried_forward_asins)
            selection_pool = []
            for product in pool:
                copy = dict(product)
                if str(product.get("asin") or "") in prior_asin_set:
                    copy["title"] = (
                        f"{PREVIOUS_RECOMMENDATION_LABEL} "
                        f"{product.get('title', '')}"
                    )
                selection_pool.append(copy)

        selected = self._select(
            transcript, selection_pool, include_continuity=bool(carried_forward_asins)
        )
        originals = {str(product["asin"]): product for product in pool}
        recommendations: list[dict[str, Any]] = []
        for item in selected:
            product = dict(originals[str(item["asin"])])
            product["rank"] = item["rank"]
            product["recommendation_explanation"] = item[
                "recommendation_explanation"
            ]
            recommendations.append(product)
        return RecommendationResult(
            recommendations=recommendations,
            plan=plan,
            retrieved_asins=[product["asin"] for product in retrieval.products],
            default_slate_asins=default_asins,
            carried_forward_asins=carried_forward_asins,
            candidate_asins=[product["asin"] for product in pool],
            lane_sources=lane_sources,
            eligible_count=retrieval.eligible_count,
        )

    def default_recommendation(self) -> RecommendationResult:
        """Return the configured turn-zero slate without retrieval or ranking."""

        products = []
        for rank, source in enumerate(default_slate(self.category), start=1):
            product = dict(source)
            product["rank"] = rank
            product["recommendation_explanation"] = (
                "Default option shown before the customer provides preference information."
            )
            products.append(product)
        asins = [product["asin"] for product in products]
        return RecommendationResult(
            recommendations=products,
            plan=RecommendationPlan(focus_query="", max_price=None),
            retrieved_asins=[],
            default_slate_asins=asins,
            carried_forward_asins=[],
            candidate_asins=asins,
            lane_sources={asin: ["default_slate"] for asin in asins},
            eligible_count=len(products),
        )

    def _plan(self, transcript: str) -> RecommendationPlan:
        response = create_response(
            self.client,
            self.tracker,
            kind="recommendation_plan",
            model=self.model,
            service_tier="default",
            instructions=(
                "Choose a 2-4 word product search query. Keep the buyer's named "
                "specialized activity, adding its stated problem or demanding workload "
                "only when useful. If the activity is routine, use the distinctive "
                "stated problem or most differentiating stated feature. Use familiar "
                "listing terms. "
                "Set max_price only when the buyer has explicitly stated a numeric ceiling; "
                "otherwise use null. Set target_price only when the buyer has explicitly "
                "stated a numeric target; otherwise use null."
            ),
            input=transcript,
            reasoning={"effort": "low"},
            text={
                "format": {
                    "type": "json_schema",
                    "name": "recommendation_plan",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "focus_query": {"type": "string"},
                            "max_price": {"type": ["number", "null"]},
                            "target_price": {"type": ["number", "null"]},
                        },
                        "required": ["focus_query", "max_price", "target_price"],
                        "additionalProperties": False,
                    },
                }
            },
        )
        raw = response_to_text(response) or "{}"
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RecommendationError("INVALID_RECOMMENDATION_PLAN", raw[:500]) from exc
        focus = payload.get("focus_query")
        maximum = payload.get("max_price")
        target = payload.get("target_price")
        if not isinstance(focus, str) or not focus.strip():
            raise RecommendationError("INVALID_RECOMMENDATION_PLAN", "focus_query is empty")
        if maximum is not None and (
            not isinstance(maximum, (int, float)) or isinstance(maximum, bool) or maximum <= 0
        ):
            raise RecommendationError("INVALID_RECOMMENDATION_PLAN", "max_price is invalid")
        if target is not None and (
            not isinstance(target, (int, float)) or isinstance(target, bool) or target <= 0
        ):
            raise RecommendationError("INVALID_RECOMMENDATION_PLAN", "target_price is invalid")
        return RecommendationPlan(
            focus_query=focus.strip(),
            max_price=float(maximum) if maximum is not None else None,
            target_price=float(target) if target is not None else None,
            min_price=float(target) * 0.4 if target is not None else None,
        )

    def _select(
        self,
        transcript: str,
        candidates: list[dict[str, Any]],
        *,
        include_continuity: bool = False,
    ) -> list[dict[str, Any]]:
        selection_instructions = (
            f"Select {self.assortment_size} products to recommend for the shopper. "
            "Based on the shopper's expressed preferences, choose the "
            f"{self.assortment_size} best fits. Give a brief explanation for each product."
        )
        if include_continuity:
            selection_instructions += "\n\n" + CONTINUITY_NOTE
        selection_input = (
            f"Transcript:\n{transcript}\n\n"
            f"Products (indexed 1-{len(candidates)}):\n{render_product_cards(candidates)}"
        )
        response = create_response(
            self.client,
            self.tracker,
            kind="recommendation_selection",
            model=self.model,
            service_tier="default",
            instructions=selection_instructions,
            input=selection_input,
            reasoning={"effort": "medium"},
            text={
                "format": {
                    "type": "json_schema",
                    "name": "recommendations",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "recommendations": {
                                "type": "array",
                                "minItems": self.assortment_size,
                                "maxItems": self.assortment_size,
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "product_number": {"type": "integer"},
                                        "explanation": {"type": "string"},
                                    },
                                    "required": ["product_number", "explanation"],
                                    "additionalProperties": False,
                                },
                            }
                        },
                        "required": ["recommendations"],
                        "additionalProperties": False,
                    },
                }
            },
        )
        raw = response_to_text(response) or "{}"
        try:
            items = json.loads(raw).get("recommendations")
        except (json.JSONDecodeError, AttributeError) as exc:
            raise RecommendationError("INVALID_RECOMMENDATION_SELECTION", raw[:500]) from exc
        if not isinstance(items, list) or len(items) != self.assortment_size:
            raise RecommendationError("INVALID_RECOMMENDATION_SELECTION", raw[:500])
        numbers = [item.get("product_number") for item in items if isinstance(item, dict)]
        explanations = [item.get("explanation") for item in items if isinstance(item, dict)]
        if (
            len(numbers) != self.assortment_size
            or len(set(numbers)) != self.assortment_size
            or not all(type(number) is int and 1 <= number <= len(candidates) for number in numbers)
            or not all(isinstance(text, str) and text.strip() for text in explanations)
        ):
            raise RecommendationError("INVALID_RECOMMENDATION_SELECTION", raw[:500])
        recommendations = [
            {
                "rank": rank,
                **candidates[number - 1],
                "recommendation_explanation": explanation.strip(),
            }
            for rank, (number, explanation) in enumerate(zip(numbers, explanations), start=1)
        ]
        return recommendations


def render_recommendations(recommendations: list[dict[str, Any]]) -> str:
    lines = [f"I recommend these {len(recommendations)} options:"]
    for product in recommendations:
        price = product.get("price")
        price_text = f"${float(price):.2f}" if isinstance(price, (int, float)) else "price unavailable"
        lines.append(
            f"{product.get('rank')}. {product.get('title')} — {price_text}. "
            f"{product.get('recommendation_explanation', '')}"
        )
    return "\n\n".join(lines)
