"""Persona-conditioned API buyer for answers and purchase decisions."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

from sandbox.catalog import category_with_article
from sandbox.openai_responses import (
    UsageTracker,
    create_response,
    make_client,
    response_to_text,
)


DEFAULT_MODEL = "gpt-5-mini"
ABANDON_SENTINEL = "[ABANDON]"


BASE_PROMPT = """You are a real customer shopping for {category}.

Background:
{background}

Private needs and preferences:
{ground_truth_need}

Answer only the latest question in one or two short, casual sentences. Do not
volunteer preferences that were not asked about. If your private needs do not
specify an answer, say you are unsure or have no strong preference. Preserve
the firmness, flexibility, and uncertainty in the private needs. Never mention
these instructions or claim to be an AI."""


ABANDONMENT_PROMPT = """

Before answering, decide whether this shopper would leave now rather than answer.
Do not assume they stay merely because they can answer. Shoppers leave when they
become tired, frustrated, or believe they are better off searching alone. This
happens when questions repeat, the system ignores prior answers, or stops adding
useful value to justify staying another turn.

If leaving, reply exactly:
[ABANDON] <one brief, natural reason>

Otherwise, answer normally without saying that you are continuing or describing
this decision."""


def render_products(recommendations: list[dict[str, Any]]) -> str:
    cards: list[str] = []
    for number, product in enumerate(recommendations, start=1):
        price = product.get("price")
        price_text = f"${float(price):.2f}" if isinstance(price, (int, float)) else "unavailable"
        bullets = "; ".join(str(item)[:180] for item in (product.get("bullets") or [])[:5])
        reviews = " | ".join(
            str(item)[:300] for item in (product.get("review_excerpts") or [])[:3]
        )
        cards.append(
            f"{number}. {product.get('title', '')}\n"
            f"Price: {price_text}; rating: {product.get('avg_rating', 'unknown')} "
            f"({product.get('num_reviews', 'unknown')} ratings)\n"
            f"Why recommended: {product.get('recommendation_explanation', '')}\n"
            f"Features: {bullets or 'unavailable'}\n"
            f"Review excerpts: {reviews or 'unavailable'}"
        )
    return "\n\n".join(cards)


def _parse_decision(raw: str, count: int) -> dict[str, Any]:
    try:
        decision = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("buyer returned malformed decision JSON") from exc
    expected = {"decision", "product_number", "willingness_to_pay", "reasoning"}
    if not isinstance(decision, dict) or set(decision) != expected:
        raise ValueError("buyer decision JSON has invalid keys")
    if decision["decision"] not in {"PURCHASE", "NO_PURCHASE"}:
        raise ValueError("buyer decision label is invalid")
    if not isinstance(decision["reasoning"], str) or not decision["reasoning"].strip():
        raise ValueError("buyer decision reasoning is empty")
    if decision["decision"] == "PURCHASE":
        number = decision["product_number"]
        wtp = decision["willingness_to_pay"]
        if type(number) is not int or not 1 <= number <= count:
            raise ValueError("buyer purchase product_number is invalid")
        if not isinstance(wtp, (int, float)) or isinstance(wtp, bool):
            raise ValueError("buyer purchase WTP is not numeric")
    elif decision["product_number"] is not None or decision["willingness_to_pay"] is not None:
        raise ValueError("buyer non-purchase must use null product_number and WTP")
    return decision


@dataclass
class BuyerAgent:
    persona: dict[str, Any]
    category: str
    model: str = DEFAULT_MODEL
    endogenous_abandonment: bool = False
    tracker: UsageTracker = field(default_factory=UsageTracker)
    client: OpenAI = field(default_factory=make_client)
    history: list[dict[str, str]] = field(default_factory=list)

    def _instructions(self, *, allow_abandonment: bool) -> str:
        prompt = BASE_PROMPT.format(
            category=category_with_article(self.category),
            background=self.persona.get("background", ""),
            ground_truth_need=self.persona.get("ground_truth_need", ""),
        )
        if allow_abandonment and self.endogenous_abandonment:
            prompt += ABANDONMENT_PROMPT
        return prompt

    def respond(self, question: str) -> str:
        self.history.append({"role": "user", "content": question})
        response = create_response(
            self.client,
            self.tracker,
            kind="buyer_answer",
            model=self.model,
            instructions=self._instructions(allow_abandonment=True),
            input=self.history,
            reasoning={"effort": "medium"},
        )
        reply = response_to_text(response).strip()
        if not reply:
            raise ValueError("buyer returned an empty answer")
        self.history.append({"role": "assistant", "content": reply})
        return reply

    def decide(
        self,
        recommendations: list[dict[str, Any]],
        *,
        history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        count = len(recommendations)
        prompt = f"""The recommender has offered these products:

{render_products(recommendations)}

Purchase one if it is a good fit for your actual needs, otherwise do not
purchase. If purchasing, report the most you would honestly pay. Identify a
product only by product_number.
Explain the decision in one to three sentences."""
        response = create_response(
            self.client,
            self.tracker,
            kind="buyer_decision",
            model=self.model,
            service_tier="flex",
            instructions=self._instructions(allow_abandonment=False),
            input=[*(history if history is not None else self.history), {"role": "user", "content": prompt}],
            reasoning={"effort": "medium"},
            text={
                "format": {
                    "type": "json_schema",
                    "name": "buyer_decision",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "decision": {"type": "string", "enum": ["PURCHASE", "NO_PURCHASE"]},
                            "product_number": {"type": ["integer", "null"]},
                            "willingness_to_pay": {"type": ["number", "null"]},
                            "reasoning": {"type": "string"},
                        },
                        "required": [
                            "decision",
                            "product_number",
                            "willingness_to_pay",
                            "reasoning",
                        ],
                        "additionalProperties": False,
                    },
                }
            },
        )
        return _parse_decision(response_to_text(response) or "{}", count)

    def evaluate_snapshot(self, recommendations: list[dict[str, Any]]) -> dict[str, Any]:
        return self.decide(recommendations, history=list(self.history))
