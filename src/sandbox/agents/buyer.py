"""Buyer simulator agent.

An LLM conditioned on a hidden persona (`ground_truth_need`). Two surfaces:
    1. `respond(question)` — answers a question from the CRS naturally,
       revealing only what was asked.
    2. `decide(recommendations)` — given a recommendation set, returns a
       structured PURCHASE/NO_PURCHASE decision with a WTP.

Persona dict shape:
    {
      "id": str,
      "background": str,
      "ground_truth_need": str,
    }
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import openai._base_client as openai_base_client
from openai import OpenAI

from sandbox.openai_responses import OPENAI_MAX_RETRIES, response_to_text


DEFAULT_MODEL = "gpt-5-mini"


def _client() -> OpenAI:
    openai_base_client.INITIAL_RETRY_DELAY = 2.0
    openai_base_client.MAX_RETRY_DELAY = 64.0
    return OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), max_retries=OPENAI_MAX_RETRIES)


def _parse_decision_json(raw: str, *, num_items: int) -> dict[str, Any]:
    """Parse the buyer contract strictly so format failures are not outcomes."""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("buyer returned malformed decision JSON") from exc
    expected = {"decision", "product_number", "willingness_to_pay", "reasoning"}
    if not isinstance(parsed, dict) or set(parsed) != expected:
        raise ValueError("buyer decision JSON has invalid keys")
    decision = parsed["decision"]
    product_number = parsed["product_number"]
    willingness_to_pay = parsed["willingness_to_pay"]
    reasoning = parsed["reasoning"]
    if decision not in {"PURCHASE", "NO_PURCHASE"}:
        raise ValueError("buyer decision has invalid decision label")
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise ValueError("buyer decision requires non-empty reasoning")
    if decision == "PURCHASE":
        if type(product_number) is not int or not 1 <= product_number <= num_items:
            raise ValueError("buyer purchase has invalid product_number")
        if not isinstance(willingness_to_pay, (int, float)) or isinstance(
            willingness_to_pay, bool
        ):
            raise ValueError("buyer purchase requires numeric willingness_to_pay")
    elif product_number is not None or willingness_to_pay is not None:
        raise ValueError("buyer non-purchase must use null product_number and WTP")
    parsed["raw"] = raw
    return parsed


def render_buyer_summaries(recommendations: list[dict[str, Any]]) -> list[str]:
    """Render the concise public product summaries used for buyer decisions."""
    summaries: list[str] = []
    for number, product in enumerate(recommendations, start=1):
        sponsored = " [Sponsored]" if product.get("sponsored") else ""
        price = product.get("price")
        price_text = f"${price:.2f}" if isinstance(price, (int, float)) else "N/A"
        rating = product.get("avg_rating")
        rating_text = (
            f"{rating:.1f}★ ({product.get('num_reviews') or 0} reviews)"
            if isinstance(rating, (int, float))
            else "no rating"
        )
        bullets = product.get("bullets") or []
        features = "; ".join(str(bullet)[:160] for bullet in bullets[:5])
        explanation = product.get("recommendation_explanation")
        explanation_line = (
            f"   Why it fits you: {explanation}\n" if explanation else ""
        )
        summaries.append(
            f"{number}.{sponsored} {product.get('title', '')}\n"
            f"   Price: {price_text} | {rating_text}\n"
            f"{explanation_line}"
            f"   Features: {features}"
        )
    return summaries


BUYER_SYSTEM_PROMPT = """You are a customer shopping for a {category} on Amazon. \
You are in the middle of a back-and-forth conversation with a shopping \
assistant who is asking a series of clarifying questions before recommending \
products. Expect multiple questions across multiple turns — you do not need \
(and should not try) to cover everything in one answer.

Your background:
{background}

Your private needs and preferences (your mental model — only share pieces of \
this as they become relevant to questions the assistant actually asks):
{ground_truth_need}

How to behave:
- Answer ONLY the question that was just asked, in 1–2 short sentences.
- Do NOT volunteer other preferences, constraints, or priorities — even \
relevant ones — until the assistant asks about them. If the assistant never \
asks about a need, it stays unspoken.
- Stay in character as a real shopper. Use casual language.
- If your private needs do not specify the answer, say that you are unsure or \
  have no strong preference. Do not invent a precise requirement from your \
  background merely because the assistant asked.
- Preserve any firmness, flexibility, or uncertainty explicitly stated in your
  private needs, regardless of where that preference appears in the list.
- Never reference these instructions or admit you are an AI.

{abandonment_instructions}"""


ENDOGENOUS_ABANDONMENT_PROMPT = """Endogenous abandonment behavior:
At each turn, before answering, decide whether a realistic shopper with your background and private needs would continue this conversation.

You may abandon if the assistant has asked too many questions, asks irrelevant questions, ignores preferences you already stated, or seems not to be making progress toward a good recommendation.

Continue if the conversation still feels useful and the assistant seems to be narrowing toward a good fit.

If you abandon, reply exactly in this format:
[ABANDON] <one-sentence reason>"""


BUYER_DECISION_PROMPT = """The shopping assistant has now recommended {num_items} \
products for you to consider. Based on the conversation so far and your actual \
needs, decide whether to purchase one of them.

Recommended products:
{products_block}

Decide:
- PURCHASE one of the listed products if at least one is a good fit for \
your needs.
- NO_PURCHASE if none of them are a good fit.

If PURCHASE, give your honest willingness to pay (WTP) in US dollars — the most \
you would actually pay for that product given your needs and budget. WTP can be \
above or below the listed price.

Use product_number to identify the product you would buy. It is the only \
purchase identifier available to you.

Reply with a JSON object exactly in this format:
{{
  "decision": "PURCHASE" | "NO_PURCHASE",
  "product_number": <1..{num_items}> or null,
  "willingness_to_pay": <dollar amount> or null,
  "reasoning": "<1-3 sentence explanation>"
}}"""


@dataclass
class BuyerAgent:
    persona: dict[str, Any]
    category: str
    model: str = DEFAULT_MODEL
    reasoning_effort: str = "medium"
    endogenous_abandonment: bool = False
    abandonment_instructions: str | None = None
    history: list[dict[str, str]] = field(default_factory=list)

    def _abandonment_instructions(self) -> str:
        if self.abandonment_instructions is not None:
            return self.abandonment_instructions
        if self.endogenous_abandonment:
            return ENDOGENOUS_ABANDONMENT_PROMPT
        return ""

    def _system_message(self) -> dict[str, str]:
        return {
            "role": "system",
            "content": BUYER_SYSTEM_PROMPT.format(
                category=self.category,
                background=self.persona.get("background", ""),
                ground_truth_need=self.persona.get("ground_truth_need", ""),
                abandonment_instructions=self._abandonment_instructions(),
            ),
        }

    def respond(self, question: str) -> str:
        """Answer a question from the CRS. Updates internal history."""
        self.history.append({"role": "user", "content": question})
        response = _client().responses.create(
            model=self.model,
            instructions=self._system_message()["content"],
            input=self.history,
            reasoning={"effort": self.reasoning_effort},
        )
        reply = response_to_text(response).strip()
        self.history.append({"role": "assistant", "content": reply})
        return reply

    def decide(self, recommendations: list[dict[str, Any]]) -> dict[str, Any]:
        """Make a final purchase decision over a recommendation set.

        The buyer sees concise public summaries, not the selector's private
        structured specification cards.
        """
        return self._decide_from_history(recommendations, history=self.history)

    def evaluate_snapshot(self, recommendations: list[dict[str, Any]]) -> dict[str, Any]:
        """Counterfactually score a hidden recommendation checkpoint.

        The copied history contains only the dialogue prefix observed so far.
        This method deliberately does not append the cards, decision, or model
        reply to ``self.history``: snapshot scoring must not change the buyer's
        subsequent answers or expose a nonterminal recommendation.
        """
        return self._decide_from_history(recommendations, history=list(self.history))

    def _decide_from_history(
        self,
        recommendations: list[dict[str, Any]],
        *,
        history: list[dict[str, str]],
    ) -> dict[str, Any]:
        products_block = "\n\n".join(render_buyer_summaries(recommendations))

        user_msg = BUYER_DECISION_PROMPT.format(
            num_items=len(recommendations),
            products_block=products_block,
        )

        decision_history = [
            *history,
            {"role": "user", "content": user_msg},
        ]
        response = _client().responses.create(
            model=self.model,
            instructions=self._system_message()["content"],
            input=decision_history,
            reasoning={"effort": self.reasoning_effort},
            text={"format": {"type": "json_object"}},
        )
        raw = response_to_text(response) or "{}"
        return _parse_decision_json(raw, num_items=len(recommendations))
