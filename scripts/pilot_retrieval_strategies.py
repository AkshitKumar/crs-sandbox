"""Pilot simple controlled-retrieval strategies on prefix-only persona evidence.

This is a diagnostic harness, not the batch evaluator.  It deliberately keeps
the candidate-generation variants small and auditable so a retrieval change is
tested before it becomes part of the experimental protocol.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from sandbox.agents.buyer import BuyerAgent  # noqa: E402
from sandbox.agents.recommendation_pipeline import (  # noqa: E402
    OpenAIRecommendationGenerator,
    PreferenceLedger,
)
from sandbox.catalog import load_catalog  # noqa: E402
from sandbox.env import load_env  # noqa: E402
from sandbox.tools.search_tool import semantic_scores, warm_dense_model  # noqa: E402


DEFAULT_MODEL = "gpt-5-mini-2025-08-07"
DEFAULT_CASES = (
    ("laptop", "laptop_001"),
    ("laptop", "laptop_002"),
    ("laptop", "laptop_006"),
    ("air_purifier", "air purifier_001"),
    ("air_purifier", "air purifier_002"),
    ("air_purifier", "air purifier_069"),
)

@dataclass(frozen=True)
class PilotCase:
    category: str
    persona: dict[str, Any]
    statements: list[str]
    hard_budget_max: float | None


def _load_persona(category: str, persona_id: str) -> dict[str, Any]:
    path = REPO_ROOT / "data" / "categories" / category / "personas.json"
    payload = json.loads(path.read_text())
    personas = payload.get("personas", payload)
    return next(persona for persona in personas if persona.get("id") == persona_id)


def _clean_statement(statement: str) -> str:
    return statement.replace("[[", "").replace("]]", "").strip()


def _hard_budget(statements: list[str]) -> float | None:
    ledger = PreferenceLedger()
    for statement in statements:
        ledger.observe(statement, source="scripted_revealed_prefix")
    return ledger.hard_budget_max


def _case(category: str, persona_id: str, depth: int) -> PilotCase:
    persona = _load_persona(category, persona_id)
    trail = [_clean_statement(item) for item in persona.get("context_trail") or []]
    statements = trail[:depth]
    return PilotCase(
        category=category,
        persona=persona,
        statements=statements,
        hard_budget_max=_hard_budget(statements),
    )


def _raw_query(case: PilotCase) -> str:
    return f"Category: {case.category}. " + " ".join(case.statements)


def _eligible(case: PilotCase) -> list[dict[str, Any]]:
    products = [
        product
        for product in load_catalog(case.category)
        if product.get("asin") and isinstance(product.get("price"), (int, float))
    ]
    if case.hard_budget_max is not None:
        products = [
            product
            for product in products
            if float(product["price"]) <= case.hard_budget_max
        ]
    return products


def _dense_candidates(case: PilotCase, query: str, limit: int) -> list[dict[str, Any]]:
    scores = semantic_scores(case.category, query)
    return sorted(
        _eligible(case),
        key=lambda product: (-scores.get(product["asin"], float("-inf")), product["asin"]),
    )[:limit]


def _hybrid_candidates(
    case: PilotCase,
    query: str,
    *,
    dense_limit: int = 12,
    popularity_limit: int = 3,
) -> list[dict[str, Any]]:
    dense = _dense_candidates(case, query, dense_limit)
    seen = {product["asin"] for product in dense}
    popular = sorted(
        (product for product in _eligible(case) if product["asin"] not in seen),
        key=lambda product: (
            -float(product.get("num_reviews") or 0),
            -float(product.get("avg_rating") or 0),
            product["asin"],
        ),
    )[:popularity_limit]
    return [*dense, *popular]


def _selection_record(
    case: PilotCase,
    *,
    strategy: str,
    query: str,
    query_raw: str | None,
    candidates: list[dict[str, Any]],
    model: str,
) -> dict[str, Any]:
    selector = OpenAIRecommendationGenerator(model=model)
    selection = selector.choose_and_explain(
        category=case.category,
        customer_statements=case.statements,
        candidates=candidates,
    )
    numbers = selection.product_numbers
    valid = (
        isinstance(numbers, list)
        and len(numbers) == 3
        and len(set(numbers)) == 3
        and all(type(number) is int and 1 <= number <= len(candidates) for number in numbers)
    )
    if not valid:
        raise ValueError(
            f"invalid selector output for {case.persona.get('id')}: {selection.raw}"
        )
    selected = [candidates[number - 1] for number in numbers]
    ranked = [
        {"rank": rank, **product, "recommendation_explanation": explanation}
        for rank, (product, explanation) in enumerate(
            zip(selected, selection.explanations), start=1
        )
    ]
    decision = BuyerAgent(
        persona=case.persona,
        category=case.category,
        model=model,
        reasoning_effort="medium",
    ).decide(ranked)
    return {
        "category": case.category,
        "persona_id": case.persona.get("id"),
        "depth": len(case.statements),
        "strategy": strategy,
        "hard_budget_max": case.hard_budget_max,
        "query": query,
        "query_raw": query_raw,
        "candidate_count": len(candidates),
        "candidate_asins": [product["asin"] for product in candidates],
        "candidate_titles": [product.get("title") for product in candidates],
        "selector_raw": selection.raw,
        "selector_valid": valid,
        "selected_asins": [product["asin"] for product in selected],
        "selected_titles": [product.get("title") for product in selected],
        "selected_prices": [product.get("price") for product in selected],
        "explanations": selection.explanations,
        "buyer_decision": decision,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--case-start", type=int, default=0)
    parser.add_argument("--case-limit", type=int, default=len(DEFAULT_CASES))
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="optional JSONL artifact; flushed after every completed strategy",
    )
    parser.add_argument(
        "--strategies",
        default="raw15,raw30,hybrid15",
        help="comma-separated subset of raw15,raw30,hybrid15",
    )
    args = parser.parse_args()
    load_env()
    requested = [item.strip() for item in args.strategies.split(",") if item.strip()]
    unknown = set(requested) - {"raw15", "raw30", "hybrid15"}
    if unknown:
        parser.error(f"unknown strategies: {sorted(unknown)}")

    warm_dense_model()
    selected_cases = DEFAULT_CASES[
        max(0, args.case_start):max(0, args.case_start) + max(0, args.case_limit)
    ]
    output_handle = None
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        output_handle = args.out.open("w")
    try:
        for category, persona_id in selected_cases:
            case = _case(category, persona_id, args.depth)
            raw_query = _raw_query(case)

            for strategy in requested:
                if strategy == "raw15":
                    query, query_raw = raw_query, None
                    candidates = _dense_candidates(case, query, 15)
                elif strategy == "raw30":
                    query, query_raw = raw_query, None
                    candidates = _dense_candidates(case, query, 30)
                else:
                    query, query_raw = raw_query, None
                    candidates = _hybrid_candidates(case, query)
                assert isinstance(query, str)
                rendered = json.dumps(
                    _selection_record(
                        case,
                        strategy=strategy,
                        query=query,
                        query_raw=query_raw,
                        candidates=candidates,
                        model=args.model,
                    ),
                    sort_keys=True,
                )
                print(rendered, flush=True)
                if output_handle is not None:
                    output_handle.write(rendered + "\n")
                    output_handle.flush()
    finally:
        if output_handle is not None:
            output_handle.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
