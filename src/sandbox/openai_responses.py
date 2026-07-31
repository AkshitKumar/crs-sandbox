"""OpenAI Responses helpers and per-conversation usage accounting."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI


OPENAI_MAX_RETRIES = 7

# USD per million tokens. Costs are estimates, while token counts come from
# each API response's usage object.
MODEL_PRICES = {
    "gpt-5-mini": {"input": 0.25, "cached_input": 0.025, "output": 2.00},
}


def make_client() -> OpenAI:
    return OpenAI(max_retries=OPENAI_MAX_RETRIES)


def content_to_text(content: Any) -> str:
    text = getattr(content, "text", None)
    if isinstance(text, str):
        return text
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(filter(None, (content_to_text(item) for item in content))).strip()
    if isinstance(content, dict):
        for key in ("text", "refusal", "content", "output_text"):
            if key in content:
                value = content_to_text(content[key])
                if value:
                    return value
        return ""
    nested = getattr(content, "content", None)
    return content_to_text(nested) if nested is not None else ""


def response_to_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text:
        return output_text
    return content_to_text(getattr(response, "output", None))


def response_output_items(response: Any) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for item in getattr(response, "output", None) or []:
        if hasattr(item, "model_dump"):
            items.append(item.model_dump(exclude_none=True))
        elif isinstance(item, dict):
            items.append(dict(item))
    return items


def _price_for_model(model: str) -> dict[str, float] | None:
    return next((rates for prefix, rates in MODEL_PRICES.items() if model.startswith(prefix)), None)


@dataclass
class UsageTracker:
    calls: list[dict[str, Any]] = field(default_factory=list)

    def record(self, response: Any, *, kind: str, latency_s: float) -> None:
        usage = getattr(response, "usage", None)
        model = str(getattr(response, "model", "") or "")
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        details = getattr(usage, "input_tokens_details", None)
        cached_tokens = int(getattr(details, "cached_tokens", 0) or 0)
        rates = _price_for_model(model)
        estimated_cost = None
        if rates is not None:
            estimated_cost = (
                max(0, input_tokens - cached_tokens) * rates["input"]
                + cached_tokens * rates["cached_input"]
                + output_tokens * rates["output"]
            ) / 1_000_000
        self.calls.append(
            {
                "kind": kind,
                "model": model,
                "response_id": getattr(response, "id", None),
                "input_tokens": input_tokens,
                "cached_input_tokens": cached_tokens,
                "output_tokens": output_tokens,
                "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
                "latency_s": round(latency_s, 3),
                "estimated_cost_usd": (
                    round(estimated_cost, 8) if estimated_cost is not None else None
                ),
            }
        )

    def summary(self) -> dict[str, Any]:
        known_costs = [
            call["estimated_cost_usd"]
            for call in self.calls
            if call["estimated_cost_usd"] is not None
        ]
        unpriced_requests = len(self.calls) - len(known_costs)
        return {
            "requests": len(self.calls),
            "input_tokens": sum(call["input_tokens"] for call in self.calls),
            "cached_input_tokens": sum(
                call["cached_input_tokens"] for call in self.calls
            ),
            "output_tokens": sum(call["output_tokens"] for call in self.calls),
            "latency_s": round(sum(call["latency_s"] for call in self.calls), 3),
            "estimated_cost_usd": (
                round(sum(known_costs), 8) if not unpriced_requests else None
            ),
            "unpriced_requests": unpriced_requests,
            "pricing_usd_per_million_tokens": MODEL_PRICES,
            "calls": list(self.calls),
        }


def create_response(
    client: OpenAI,
    tracker: UsageTracker,
    *,
    kind: str,
    **kwargs: Any,
) -> Any:
    started = time.perf_counter()
    response = client.responses.create(**kwargs)
    tracker.record(response, kind=kind, latency_s=time.perf_counter() - started)
    return response
