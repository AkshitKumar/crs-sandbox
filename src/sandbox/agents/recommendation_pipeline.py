"""Shared retrieval, bounded selection, and explanation services for the CRS.

This module has no conversation loop.  ``CRSAgentSession`` remains the only
recommender agent; the classes here implement the catalog operation performed
when that agent finalizes a recommendation, plus hidden checkpoint snapshots.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Protocol

from sandbox.catalog import load_catalog, load_config
from sandbox.openai_responses import OPENAI_MAX_RETRIES, response_to_text
from sandbox.product_text import serialize_product
from sandbox.tools.search_tool import semantic_scores


PROTOCOL_VERSION = "react-query-recommend-v3"
DEFAULT_MODEL = "gpt-5-mini-2025-08-07"
DEFAULT_RETRIEVAL_LIMIT = 15

_HARD_CAP_RE = re.compile(
    r"(?:hard\s+(?:cap|maximum|max|limit)|must\s+(?:stay|be)\s+(?:under|below)|"
    r"(?:cannot|can['’]t|won['’]t|will\s+not)\s+(?:(?:go|spend|pay)\s+)?(?:over|above|exceed)|"
    r"need\s+to\s+keep(?:\s+(?:it|this|the\s+(?:price|cost)))?\s+(?:under|below)|"
    r"(?:no|not)\s+more\s+than|at\s+most|do\s+not\s+(?:go|spend|pay)\s+(?:over|above))"
    r"[^$\d]{0,24}\$?\s*([0-9][0-9,]*(?:\.\d{1,2})?)",
    re.IGNORECASE,
)
_FIRM_CAP_RE = re.compile(
    r"(?:(?:firm|strict|absolute|non-negotiable)\s+"
    r"(?:budget|cap|limit|maximum|max)|absolute\s+maximum)"
    r"[^$\d]{0,24}\$?\s*([0-9][0-9,]*(?:\.\d{1,2})?)|"
    r"(?:budget|upper\s+limit|cap|maximum|max)[^$\d]{0,24}"
    r"\b(?:firm|strict|absolute|non-negotiable)\b[^$\d]{0,12}"
    r"\$?\s*([0-9][0-9,]*(?:\.\d{1,2})?)|"
    r"(?:budget|upper\s+limit|cap|maximum|max)\s*(?:is|of|at)?\s*"
    r"\$?\s*([0-9][0-9,]*(?:\.\d{1,2})?)"
    r"[^.\n]{0,24}\b(?:firm|strict|absolute|non-negotiable)\b",
    re.IGNORECASE,
)
_AMOUNT_THEN_FIRM_CAP_RE = re.compile(
    r"\$?\s*([0-9][0-9,]*(?:\.\d{1,2})?)\s*"
    r"(?:is\s+)?(?:a\s+)?(?:firm|strict|absolute|non-negotiable)\s+"
    r"(?:(?:hard|upper)\s+)?(?:budget|cap|limit|maximum|max)\b",
    re.IGNORECASE,
)
_FLEXIBLE_CAP_RE = re.compile(
    r"\b(?:flexible|some\s+flexibility|can\s+stretch|could\s+stretch|"
    r"willing\s+to\s+(?:go|spend|pay)\s+(?:over|above|more))\b",
    re.IGNORECASE,
)


def render_product_cards(products: list[dict[str, Any]]) -> list[str]:
    """Render private, numbered evidence cards for selector/prose calls."""
    cards: list[str] = []
    for number, product in enumerate(products, start=1):
        price = product.get("price")
        price_text = f"${price:.2f}" if isinstance(price, (int, float)) else "price unavailable"
        bullets = "; ".join(
            str(value)[:240] for value in (product.get("bullets") or [])[:5]
        )[:1800]
        specs = "; ".join(
            f"{key}: {value}" for key, value in (product.get("spec_table") or {}).items()
        )[:2400]
        review_count = product.get("num_reviews")
        review_text = (
            f" from {int(review_count):,} ratings"
            if isinstance(review_count, (int, float))
            else ""
        )
        cards.append(
            f"{number}. {product.get('title', '')}\n"
            f"Price: {price_text}; rating: {product.get('avg_rating', 'unknown')}{review_text}\n"
            f"Catalog evidence: {bullets or 'unavailable'}\n"
            f"Specifications: {specs or 'unavailable'}"
        )
    return cards


@dataclass(frozen=True)
class RecommendationResult:
    product_numbers: list[int]
    explanations: list[str] = field(default_factory=list)
    raw: str = ""


@dataclass(frozen=True)
class RecommendationProse:
    explanations: list[str]
    raw: str = ""


@dataclass(frozen=True)
class CandidateRetrieval:
    products: list[dict[str, Any]]
    scores: dict[str, float]
    lane_sources: dict[str, list[str]] = field(default_factory=dict)


class RecommendationGenerator(Protocol):
    def choose_and_explain(
        self,
        *,
        category: str,
        customer_statements: list[str],
        candidates: list[dict[str, Any]],
    ) -> RecommendationResult: ...

    def explain(
        self,
        *,
        category: str,
        customer_statements: list[str],
        recommendations: list[dict[str, Any]],
    ) -> RecommendationProse: ...


class CandidateRetriever(Protocol):
    def retrieve(
        self,
        *,
        category: str,
        query: str,
        key_query: str | None,
        eligible_products: list[dict[str, Any]],
        limit: int,
    ) -> CandidateRetrieval: ...


@dataclass
class OpenAIRecommendationGenerator:
    """Choose three numbered products and briefly explain each choice."""

    model: str = DEFAULT_MODEL
    reasoning_effort: str = "low"

    def choose_and_explain(
        self,
        *,
        category: str,
        customer_statements: list[str],
        candidates: list[dict[str, Any]],
    ) -> RecommendationResult:
        from openai import OpenAI

        cards = render_product_cards(candidates)
        output_example = {
            "recommendations": [
                {"product_number": number, "explanation": "one short sentence"}
                for number in range(1, 4)
            ]
        }
        prompt = (
            "The customer said these sentences, verbatim:\n"
            f"{json.dumps(customer_statements, ensure_ascii=False)}\n\n"
            f"Candidate {category} products:\n" + "\n\n".join(cards) + "\n\n"
            "Choose the three products that best fit the customer, in best-to-worst order. "
            "Give greatest weight to the customer's decisive requirements and intended use, "
            "while treating other preferences as tradeoffs rather than filters. Use only facts "
            "shown in the product cards; missing information is unknown. Prefer meaningfully "
            "different products over near-identical variants. For each product, give one short "
            "sentence stating its strongest fit and any important shortfall. Use only the "
            "numbered products. Return JSON in this form: "
            f"{json.dumps(output_example)}"
        )
        response = OpenAI(max_retries=OPENAI_MAX_RETRIES).responses.create(
            model=self.model,
            instructions="Choose three products and briefly explain each choice.",
            input=prompt,
            reasoning={"effort": self.reasoning_effort},
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
                                "minItems": 3,
                                "maxItems": 3,
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
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"recommendation call returned malformed JSON: {raw[:500]}") from exc
        items = payload.get("recommendations") if set(payload) == {"recommendations"} else None
        if not isinstance(items, list):
            return RecommendationResult(product_numbers=[], explanations=[], raw=raw)
        return RecommendationResult(
            product_numbers=[item.get("product_number") for item in items if isinstance(item, dict)],
            explanations=[item.get("explanation") for item in items if isinstance(item, dict)],
            raw=raw,
        )

    def explain(
        self,
        *,
        category: str,
        customer_statements: list[str],
        recommendations: list[dict[str, Any]],
    ) -> RecommendationProse:
        from openai import OpenAI

        cards = render_product_cards(recommendations)
        output_example = json.dumps(
            {"explanations": [f"reason for product {index}" for index in range(1, len(cards) + 1)]}
        )
        prompt = (
            "The customer said these sentences, verbatim:\n"
            f"{json.dumps(customer_statements, ensure_ascii=False)}\n\n"
            f"Already-selected {category} products:\n" + "\n\n".join(cards) + "\n\n"
            "Write one short sentence explaining why each product fits, in the same order. "
            f"Return JSON in this form: {output_example}."
        )
        response = OpenAI(max_retries=OPENAI_MAX_RETRIES).responses.create(
            model=self.model,
            instructions="Briefly explain the fixed recommendations without changing them.",
            input=prompt,
            reasoning={"effort": self.reasoning_effort},
            text={
                "format": {
                    "type": "json_schema",
                    "name": "fixed_recommendation_explanations",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "explanations": {
                                "type": "array",
                                "items": {"type": "string"},
                                "minItems": len(cards),
                                "maxItems": len(cards),
                            }
                        },
                        "required": ["explanations"],
                        "additionalProperties": False,
                    },
                }
            },
        )
        raw = response_to_text(response) or "{}"
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"recommendation prose generator returned malformed JSON: {raw[:500]}") from exc
        explanations = payload.get("explanations") if set(payload) == {"explanations"} else None
        if (
            not isinstance(explanations, list)
            or len(explanations) != len(cards)
            or not all(isinstance(item, str) and item.strip() for item in explanations)
        ):
            raise ValueError(f"recommendation prose generator returned invalid explanations: {raw[:500]}")
        return RecommendationProse(
            explanations=[item.strip() for item in explanations],
            raw=raw,
        )


@dataclass
class DenseCandidateRetriever:
    def retrieve(
        self,
        *,
        category: str,
        query: str,
        key_query: str | None = None,
        eligible_products: list[dict[str, Any]],
        limit: int,
    ) -> CandidateRetrieval:
        scores = semantic_scores(category, query)
        missing = [
            product["asin"] for product in eligible_products if product.get("asin") not in scores
        ]
        if missing:
            raise ValueError(f"dense index is missing {len(missing)} eligible catalog ASINs; rebuild it")
        ordered = sorted(
            eligible_products,
            key=lambda product: (-scores[product["asin"]], product["asin"]),
        )[:limit]
        return CandidateRetrieval(
            products=ordered,
            scores={product["asin"]: scores[product["asin"]] for product in ordered},
            lane_sources={product["asin"]: ["dense_full"] for product in ordered},
        )


_LEXICAL_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _lexical_tokens(text: str) -> list[str]:
    return _LEXICAL_TOKEN_RE.findall(text.casefold())


@lru_cache(maxsize=64)
def _bm25_scores(category: str, query: str) -> dict[str, float]:
    """Score the full category with standard BM25 (k1=1.2, b=0.75)."""
    products = [product for product in load_catalog(category) if product.get("asin")]
    documents = [_lexical_tokens(serialize_product(product)) for product in products]
    if not documents:
        return {}

    term_frequencies: list[Counter[str]] = []
    document_frequency: Counter[str] = Counter()
    for document in documents:
        frequencies = Counter(document)
        term_frequencies.append(frequencies)
        document_frequency.update(frequencies.keys())

    average_length = sum(map(len, documents)) / len(documents)
    query_terms = _lexical_tokens(query)
    scores: dict[str, float] = {}
    for product, document, frequencies in zip(products, documents, term_frequencies):
        score = 0.0
        for term in query_terms:
            containing = document_frequency[term]
            inverse_frequency = math.log(
                1.0 + (len(documents) - containing + 0.5) / (containing + 0.5)
            )
            frequency = frequencies[term]
            denominator = frequency + 1.2 * (
                0.25 + 0.75 * len(document) / average_length
            )
            if denominator:
                score += inverse_frequency * frequency * 2.2 / denominator
        scores[product["asin"]] = score
    return scores


@dataclass
class HybridCandidateRetriever:
    """Union three general retrieval lanes without combining arbitrary weights.

    The lanes are full-preference dense retrieval, exact-term BM25 retrieval,
    and dense retrieval for the customer's most important revealed requirement.
    Round-robin union reserves roughly equal recall capacity for each lane.
    """

    def retrieve(
        self,
        *,
        category: str,
        query: str,
        key_query: str | None = None,
        eligible_products: list[dict[str, Any]],
        limit: int,
    ) -> CandidateRetrieval:
        dense_full_scores = semantic_scores(category, query)
        focused_query = (key_query or query).strip() or query
        dense_key_scores = semantic_scores(category, focused_query)
        lexical_scores = _bm25_scores(category, query)

        for name, scores in {
            "dense_full": dense_full_scores,
            "bm25_full": lexical_scores,
            "dense_key": dense_key_scores,
        }.items():
            missing = [
                product["asin"]
                for product in eligible_products
                if product.get("asin") not in scores
            ]
            if missing:
                raise ValueError(
                    f"{name} is missing {len(missing)} eligible catalog ASINs"
                )

        lane_specs = (
            ("dense_full", dense_full_scores),
            ("bm25_full", lexical_scores),
            ("dense_key", dense_key_scores),
        )
        lanes = [
            sorted(
                eligible_products,
                key=lambda product, values=scores: (
                    -values[product["asin"]],
                    product["asin"],
                ),
            )
            for _, scores in lane_specs
        ]

        selected: list[dict[str, Any]] = []
        lane_sources: dict[str, list[str]] = {}
        seen: set[str] = set()
        position = 0
        while len(selected) < limit and any(position < len(lane) for lane in lanes):
            for (lane_name, _), lane in zip(lane_specs, lanes):
                if position >= len(lane):
                    continue
                product = lane[position]
                asin = product["asin"]
                lane_sources.setdefault(asin, []).append(lane_name)
                if asin in seen:
                    continue
                selected.append(product)
                seen.add(asin)
                if len(selected) == limit:
                    break
            position += 1

        return CandidateRetrieval(
            products=selected,
            scores={product["asin"]: dense_full_scores[product["asin"]] for product in selected},
            lane_sources={product["asin"]: lane_sources[product["asin"]] for product in selected},
        )


@dataclass
class PreferenceLedger:
    """Append-only revealed evidence and category-general hard-budget detection."""

    entries: list[dict[str, str]] = field(default_factory=list)
    hard_budget_max: float | None = None

    def observe(
        self,
        text: str,
        *,
        source: str,
        question_id: str | None = None,
        topic: str | None = None,
        question_text: str | None = None,
    ) -> None:
        clean = str(text).strip()
        if not clean:
            return
        entry = {"source": source, "text": clean}
        for key, value in {
            "question_id": question_id,
            "topic": topic,
            "question_text": question_text,
        }.items():
            if value is not None:
                entry[key] = value
        self.entries.append(entry)
        matches = list(_HARD_CAP_RE.findall(clean))
        matches.extend(
            next((value for value in groups if value), "") for groups in _FIRM_CAP_RE.findall(clean)
        )
        matches.extend(_AMOUNT_THEN_FIRM_CAP_RE.findall(clean))
        matches = [value for value in matches if value]
        if _FLEXIBLE_CAP_RE.search(clean):
            matches = []
        if matches:
            cap = min(float(value.replace(",", "")) for value in matches)
            self.hard_budget_max = cap if self.hard_budget_max is None else min(self.hard_budget_max, cap)

    def as_text(self) -> str:
        return "\n".join(self.statements())

    def statements(self) -> list[str]:
        """Return exactly what the customer said, in chronological order."""
        return [entry["text"] for entry in self.entries]

    def prefix_hash(self) -> str:
        return hashlib.sha256(self.as_text().encode("utf-8")).hexdigest()


@dataclass
class RecommendationSnapshot:
    index: int
    asks_so_far: int
    question_ids: list[str]
    preference_ledger: list[dict[str, str]]
    prefix_hash: str
    hard_budget_max: float | None
    recommendations: list[dict[str, Any]] | None
    recommendation_source: str | None
    recommendation_validation_error: str | None = None
    recommendation_selection_raw: str | None = None
    recommendation_prose_raw: str | None = None
    recommendation_product_numbers: list[int] = field(default_factory=list)
    retrieval_query: str | None = None
    retrieval_key_query: str | None = None
    retrieval_query_raw: str | None = None
    retrieval_candidate_asins: list[str] = field(default_factory=list)
    retrieval_scores: dict[str, float] = field(default_factory=dict)
    retrieval_lane_sources: dict[str, list[str]] = field(default_factory=dict)
    eligible_count: int = 0
    terminal_status: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "asks_so_far": self.asks_so_far,
            "question_ids": list(self.question_ids),
            "preference_ledger": list(self.preference_ledger),
            "prefix_hash": self.prefix_hash,
            "hard_budget_max": self.hard_budget_max,
            "recommendations": self.recommendations,
            "recommendation_source": self.recommendation_source,
            "recommendation_validation_error": self.recommendation_validation_error,
            "recommendation_selection_raw": self.recommendation_selection_raw,
            "recommendation_prose_raw": self.recommendation_prose_raw,
            "recommendation_product_numbers": list(self.recommendation_product_numbers),
            "retrieval_query": self.retrieval_query,
            "retrieval_key_query": self.retrieval_key_query,
            "retrieval_query_raw": self.retrieval_query_raw,
            "retrieval_candidate_asins": list(self.retrieval_candidate_asins),
            "retrieval_scores": dict(self.retrieval_scores),
            "retrieval_lane_sources": {
                asin: list(sources) for asin, sources in self.retrieval_lane_sources.items()
            },
            "eligible_count": self.eligible_count,
            "terminal_status": self.terminal_status,
        }


@dataclass
class RecommendationPipeline:
    """Stateless recommendation operation used by the one conversational agent."""

    category: str
    model: str = DEFAULT_MODEL
    retrieval_limit: int = DEFAULT_RETRIEVAL_LIMIT
    recommendation_limit: int = 3
    generator: RecommendationGenerator | None = None
    retriever: CandidateRetriever | None = None

    def __post_init__(self) -> None:
        load_config(self.category)
        if self.recommendation_limit < 1:
            raise ValueError("recommendation_limit must be positive")
        if self.retrieval_limit < self.recommendation_limit:
            raise ValueError("retrieval_limit must be at least recommendation_limit")

    def default_snapshot(
        self,
        *,
        ledger: PreferenceLedger,
        question_ids: list[str],
        asks_so_far: int,
    ) -> RecommendationSnapshot:
        products = self._default_slate(ledger)
        return self._snapshot_from_fixed_slate(
            ledger=ledger,
            question_ids=question_ids,
            asks_so_far=asks_so_far,
            selected=products,
            source="curated_default_slate",
        )

    def prefix_snapshot(
        self,
        *,
        ledger: PreferenceLedger,
        question_ids: list[str],
        asks_so_far: int,
        query: str | None,
        key_query: str | None = None,
    ) -> RecommendationSnapshot:
        if asks_so_far == 0 and ledger.hard_budget_max is None:
            return self.default_snapshot(
                ledger=ledger,
                question_ids=question_ids,
                asks_so_far=asks_so_far,
            )
        eligible = self._eligible_products(ledger)
        if not eligible:
            return self._no_match(ledger, question_ids, asks_so_far, 0)
        query_text = (query or ledger.as_text()).strip()
        if not query_text:
            return self.default_snapshot(
                ledger=ledger,
                question_ids=question_ids,
                asks_so_far=asks_so_far,
            )
        retrieval = (self.retriever or HybridCandidateRetriever()).retrieve(
            category=self.category,
            query=query_text,
            key_query=key_query,
            eligible_products=eligible,
            limit=self.retrieval_limit,
        )
        source = "agent_query_hybrid_then_model" if query else "ledger_fallback_hybrid_then_model"
        return self._select_and_explain(
            ledger=ledger,
            question_ids=question_ids,
            asks_so_far=asks_so_far,
            candidates=retrieval.products,
            source=source,
            retrieval=retrieval,
            retrieval_query=query_text or None,
            retrieval_key_query=(key_query or query_text).strip() or None,
            eligible_count=len(eligible),
        )

    def select_for_query(
        self,
        *,
        ledger: PreferenceLedger,
        question_ids: list[str],
        asks_so_far: int,
        query: str,
        key_query: str | None = None,
    ) -> RecommendationSnapshot:
        """Retrieve 15 products from the agent's query, then select and explain three."""
        query_text = query.strip()
        if not query_text:
            raise ValueError("recommend() requires a non-empty embedding query")
        eligible = self._eligible_products(ledger)
        if not eligible:
            return self._no_match(ledger, question_ids, asks_so_far, 0)
        retrieval = (self.retriever or HybridCandidateRetriever()).retrieve(
            category=self.category,
            query=query_text,
            key_query=key_query,
            eligible_products=eligible,
            limit=self.retrieval_limit,
        )
        return self._select_and_explain(
            ledger=ledger,
            question_ids=question_ids,
            asks_so_far=asks_so_far,
            candidates=retrieval.products,
            source="agent_query_hybrid_then_model",
            retrieval=retrieval,
            retrieval_query=query_text,
            retrieval_key_query=(key_query or query_text).strip(),
            eligible_count=len(eligible),
        )

    def _select_and_explain(
        self,
        *,
        ledger: PreferenceLedger,
        question_ids: list[str],
        asks_so_far: int,
        candidates: list[dict[str, Any]],
        source: str,
        retrieval: CandidateRetrieval,
        retrieval_query: str | None,
        retrieval_key_query: str | None,
        eligible_count: int,
    ) -> RecommendationSnapshot:
        if len(candidates) < self.recommendation_limit:
            return self._no_match(ledger, question_ids, asks_so_far, eligible_count)
        generator = self.generator or OpenAIRecommendationGenerator(model=self.model)
        selection = generator.choose_and_explain(
            category=self.category,
            customer_statements=ledger.statements(),
            candidates=candidates,
        )
        selected, error = self._validate_selection(selection, candidates)
        if error:
            raise ValueError(f"invalid reranker selection ({error}): {selection.raw[:500]}")
        recommendations = [
            {"rank": index, **product, "recommendation_explanation": explanation}
            for index, (product, explanation) in enumerate(
                zip(selected, selection.explanations), start=1
            )
        ]
        return RecommendationSnapshot(
            index=asks_so_far,
            asks_so_far=asks_so_far,
            question_ids=list(question_ids),
            preference_ledger=list(ledger.entries),
            prefix_hash=ledger.prefix_hash(),
            hard_budget_max=ledger.hard_budget_max,
            recommendations=recommendations,
            recommendation_source=source,
            recommendation_selection_raw=selection.raw,
            recommendation_prose_raw=None,
            recommendation_product_numbers=list(selection.product_numbers),
            retrieval_query=retrieval_query,
            retrieval_key_query=retrieval_key_query,
            retrieval_query_raw=None,
            retrieval_candidate_asins=[product["asin"] for product in candidates],
            retrieval_scores=dict(retrieval.scores),
            retrieval_lane_sources={
                asin: list(sources) for asin, sources in retrieval.lane_sources.items()
            },
            eligible_count=eligible_count,
            terminal_status="RECOMMENDED",
        )

    def _snapshot_from_fixed_slate(
        self,
        *,
        ledger: PreferenceLedger,
        question_ids: list[str],
        asks_so_far: int,
        selected: list[dict[str, Any]],
        source: str,
    ) -> RecommendationSnapshot:
        generator = self.generator or OpenAIRecommendationGenerator(model=self.model)
        prose = generator.explain(
            category=self.category,
            customer_statements=ledger.statements(),
            recommendations=selected,
        )
        recommendations = [
            {"rank": index, **product, "recommendation_explanation": explanation}
            for index, (product, explanation) in enumerate(zip(selected, prose.explanations), start=1)
        ]
        return RecommendationSnapshot(
            index=asks_so_far,
            asks_so_far=asks_so_far,
            question_ids=list(question_ids),
            preference_ledger=list(ledger.entries),
            prefix_hash=ledger.prefix_hash(),
            hard_budget_max=ledger.hard_budget_max,
            recommendations=recommendations,
            recommendation_source=source,
            recommendation_prose_raw=prose.raw,
            retrieval_candidate_asins=[],
            eligible_count=len(self._eligible_products(ledger)),
            terminal_status="RECOMMENDED",
        )

    def _eligible_products(self, ledger: PreferenceLedger) -> list[dict[str, Any]]:
        products = [
            product for product in load_catalog(self.category)
            if product.get("asin") and isinstance(product.get("price"), (int, float))
        ]
        if ledger.hard_budget_max is not None:
            products = [
                product for product in products
                if float(product["price"]) <= ledger.hard_budget_max
            ]
        return products

    def _default_slate(self, ledger: PreferenceLedger) -> list[dict[str, Any]]:
        configured = load_config(self.category).get("default_slate") or []
        if len(configured) != self.recommendation_limit or len(set(configured)) != len(configured):
            raise ValueError(
                f"{self.category!r} default_slate must contain exactly "
                f"{self.recommendation_limit} unique ASINs"
            )
        by_asin = {product["asin"]: product for product in self._eligible_products(ledger)}
        missing = [asin for asin in configured if asin not in by_asin]
        if missing:
            raise ValueError(
                f"{self.category!r} default_slate contains missing, unpriced, or over-budget products: {missing}"
            )
        return [by_asin[asin] for asin in configured]

    def _validate_selection(
        self,
        selection: RecommendationResult,
        candidates: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], str | None]:
        numbers = selection.product_numbers
        if not isinstance(numbers, list):
            return [], "product_numbers_not_a_list"
        if not all(type(number) is int and 1 <= number <= len(candidates) for number in numbers):
            return [], "out_of_range_product_number"
        expected = min(self.recommendation_limit, len(candidates))
        if len(numbers) != expected or len(set(numbers)) != len(numbers):
            return [], "wrong_count_or_duplicate_product_numbers"
        explanations = selection.explanations
        if (
            not isinstance(explanations, list)
            or len(explanations) != expected
            or not all(isinstance(item, str) and item.strip() for item in explanations)
        ):
            return [], "invalid_explanations"
        return [candidates[number - 1] for number in numbers], None

    def _no_match(
        self,
        ledger: PreferenceLedger,
        question_ids: list[str],
        asks_so_far: int,
        eligible_count: int,
    ) -> RecommendationSnapshot:
        return RecommendationSnapshot(
            index=asks_so_far,
            asks_so_far=asks_so_far,
            question_ids=list(question_ids),
            preference_ledger=list(ledger.entries),
            prefix_hash=ledger.prefix_hash(),
            hard_budget_max=ledger.hard_budget_max,
            recommendations=None,
            recommendation_source=None,
            eligible_count=eligible_count,
            terminal_status="NO_FEASIBLE_MATCH",
        )
