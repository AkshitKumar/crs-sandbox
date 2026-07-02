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
- Your needs are listed in a priority order --- express uncertainty or flexibility
  for preferences that are secondary or appear later in your preference.
- If asked something your needs don't specify, give a reasonable answer \
consistent with your background.
- You can express vagueness about priorities or needs when it is realistic.
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
- PURCHASE one of the listed products if at least one is a reasonable fit for \
your needs.
- NO_PURCHASE if none of them are a reasonable fit.

If PURCHASE, give your honest willingness to pay (WTP) in US dollars — the most \
you would actually pay for that product given your needs and budget. WTP can be \
above or below the listed price.

Reply with a JSON object exactly in this format:
{{
  "decision": "PURCHASE" | "NO_PURCHASE",
  "product_number": <1..{num_items}> or null,
  "asin": <the chosen product's ASIN> or null,
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

        `recommendations` is a list of dicts with keys: rank, asin, title,
        price, avg_rating, num_reviews, bullets, sponsored.
        """
        lines = []
        for i, r in enumerate(recommendations, start=1):
            sponsored = " [Sponsored]" if r.get("sponsored") else ""
            price_str = (
                f"${r['price']:.2f}" if isinstance(r.get("price"), (int, float)) else "N/A"
            )
            rating = r.get("avg_rating")
            rating_str = f"{rating:.1f}★ ({r.get('num_reviews') or 0} reviews)" if rating else "no rating"
            bullets = r.get("bullets") or []
            bullets_short = "; ".join(b[:160] for b in bullets[:5])
            lines.append(
                f"{i}.{sponsored} {r['title']}\n"
                f"   Price: {price_str} | {rating_str}\n"
                f"   Features: {bullets_short}"
            )
        products_block = "\n\n".join(lines)

        user_msg = BUYER_DECISION_PROMPT.format(
            num_items=len(recommendations),
            products_block=products_block,
        )

        decision_history = [
            *self.history,
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
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {
                "decision": "NO_PURCHASE",
                "product_number": None,
                "asin": None,
                "willingness_to_pay": None,
                "reasoning": f"failed to parse decision JSON: {raw[:200]}",
            }
        parsed["raw"] = raw
        return parsed
