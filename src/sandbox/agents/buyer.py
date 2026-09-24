"""Persona-conditioned API buyer for answers and purchase decisions."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

from sandbox.agents.recommender import render_product_cards
from sandbox.catalog import category_with_article
from sandbox.openai_responses import (
    UsageTracker,
    create_response,
    make_client,
    response_to_text,
)


DEFAULT_MODEL = "gpt-5.6-luna"
ABANDON_SENTINEL = "[ABANDON]"


BASE_PROMPT = """You are a customer shopping for {category}.

Background:
{background}

Private needs and preferences:
{ground_truth_need}

You are interacting with an online conversational recommender. Answer naturally and concisely based on your needs; don't volunteer unasked information, or improvise any new preferences."""


ABANDONMENT_PROMPT = """You are interacting with an online conversational recommender. Based on the conversation thus far, either continue the conversation (CONTINUE) or leave the interaction (ABANDON) at this point. Briefly explain your choice."""


PURCHASE_PROMPT = """The recommender has offered these products:

<offered_products>
{products}
</offered_products>

Based on the shopper's preferences, choose whether to purchase one of these products at the current price. If no product is a good fit, choose NO_PURCHASE. Otherwise, choose PURCHASE and identify the best product by its product_number. Briefly explain your choice."""


def _parse_abandonment_decision(raw: str) -> dict[str, str]:
    try:
        decision = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("buyer returned malformed abandonment JSON") from exc
    if not isinstance(decision, dict) or set(decision) != {"action", "reason"}:
        raise ValueError("buyer abandonment JSON has invalid keys")
    if decision["action"] not in {"CONTINUE", "ABANDON"}:
        raise ValueError("buyer abandonment action is invalid")
    if not isinstance(decision["reason"], str) or not decision["reason"].strip():
        raise ValueError("buyer abandonment reason is empty")
    return {
        "action": decision["action"],
        "reason": decision["reason"].strip(),
    }


def render_products(recommendations: list[dict[str, Any]]) -> str:
    return render_product_cards(recommendations)


def _parse_decision(raw: str, count: int) -> dict[str, Any]:
    try:
        decision = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("buyer returned malformed decision JSON") from exc
    expected = {"reason", "decision", "product_number"}
    if not isinstance(decision, dict) or set(decision) != expected:
        raise ValueError("buyer decision JSON has invalid keys")
    if decision["decision"] not in {"PURCHASE", "NO_PURCHASE"}:
        raise ValueError("buyer decision label is invalid")
    if not isinstance(decision["reason"], str) or not decision["reason"].strip():
        raise ValueError("buyer decision reason is empty")
    if decision["decision"] == "PURCHASE":
        number = decision["product_number"]
        if type(number) is not int or not 1 <= number <= count:
            raise ValueError("buyer purchase product_number is invalid")
    elif decision["product_number"] is not None:
        raise ValueError("buyer non-purchase must use null product_number")
    return {
        "reasoning": decision["reason"].strip(),
        "decision": decision["decision"],
        "product_number": decision["product_number"],
        # Retain the legacy output field while no longer eliciting WTP.
        "willingness_to_pay": None,
    }


@dataclass
class BuyerAgent:
    persona: dict[str, Any]
    category: str
    model: str = DEFAULT_MODEL
    endogenous_abandonment: bool = False
    tracker: UsageTracker = field(default_factory=UsageTracker)
    client: OpenAI = field(default_factory=make_client)
    history: list[dict[str, str]] = field(default_factory=list)

    def _instructions(self) -> str:
        return BASE_PROMPT.format(
            category=category_with_article(self.category),
            background=self.persona.get("background", ""),
            ground_truth_need=self.persona.get("ground_truth_need", ""),
        )

    def decide_abandonment(self, question: str) -> dict[str, str]:
        if not self.endogenous_abandonment:
            raise ValueError("endogenous abandonment is disabled")
        transcript = [f"Shopper: I'm looking for {category_with_article(self.category)}."]
        labels = {"user": "Recommender", "assistant": "Shopper"}
        transcript.extend(
            f"{labels[item['role']]}: {item['content']}" for item in self.history
        )
        transcript.append(f"Recommender: {question}")
        instructions = f"""Background:
{self.persona.get('background', '')}

Private needs and preferences:
{self.persona.get('ground_truth_need', '')}

{ABANDONMENT_PROMPT}"""
        response = create_response(
            self.client,
            self.tracker,
            kind="buyer_abandonment",
            model=self.model,
            service_tier="flex",
            instructions=instructions,
            input="\n".join(transcript),
            reasoning={"effort": "low"},
            text={
                "format": {
                    "type": "json_schema",
                    "name": "abandonment_decision",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "reason": {"type": "string"},
                            "action": {
                                "type": "string",
                                "enum": ["CONTINUE", "ABANDON"],
                            },
                        },
                        "required": ["reason", "action"],
                        "additionalProperties": False,
                    },
                }
            },
        )
        return _parse_abandonment_decision(response_to_text(response) or "{}")

    def respond(self, question: str) -> str:
        self.history.append({"role": "user", "content": question})
        response = create_response(
            self.client,
            self.tracker,
            kind="buyer_answer",
            model=self.model,
            service_tier="flex",
            instructions=self._instructions(),
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
        prompt = PURCHASE_PROMPT.format(products=render_products(recommendations))
        instructions = (
            f"Background:\n{self.persona.get('background', '')}\n\n"
            "Private needs and preferences:\n"
            f"{self.persona.get('ground_truth_need', '')}"
        )
        response = create_response(
            self.client,
            self.tracker,
            kind="buyer_decision",
            model=self.model,
            service_tier="flex",
            instructions=instructions,
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
                            "reason": {"type": "string"},
                            "decision": {"type": "string", "enum": ["PURCHASE", "NO_PURCHASE"]},
                            "product_number": {"type": ["integer", "null"]},
                        },
                        "required": [
                            "reason",
                            "decision",
                            "product_number",
                        ],
                        "additionalProperties": False,
                    },
                }
            },
        )
        return _parse_decision(response_to_text(response) or "{}", count)

    def evaluate_snapshot(self, recommendations: list[dict[str, Any]]) -> dict[str, Any]:
        return self.decide(recommendations, history=list(self.history))
