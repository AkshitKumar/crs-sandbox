"""Test the production retrieve-15/select-and-explain operation with real API calls.

This intentionally bypasses the ReAct conversation and buyer. By default the
full query is the persona's revealed statements joined together; pass
``--query`` and ``--key-query`` to test queries written as the ReAct agent would write them.
Each case makes one OpenAI call after local embedding retrieval.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from sandbox.agents.recommendation_pipeline import (  # noqa: E402
    PreferenceLedger,
    RecommendationPipeline,
)
from sandbox.env import load_env  # noqa: E402
from sandbox.tools.search_tool import warm_dense_model  # noqa: E402


CASES = (
    ("laptop", "laptop_001"),
    ("laptop", "laptop_002"),
    ("laptop", "laptop_006"),
    ("air_purifier", "air purifier_001"),
    ("air_purifier", "air purifier_002"),
    ("air_purifier", "air purifier_069"),
)


def _persona(category: str, persona_id: str) -> dict:
    path = REPO_ROOT / "data" / "categories" / category / "personas.json"
    payload = json.loads(path.read_text())
    personas = payload.get("personas", payload)
    return next(persona for persona in personas if persona.get("id") == persona_id)


def _run_case(
    category: str,
    persona_id: str,
    model: str,
    query_override: str | None,
    key_query_override: str | None,
) -> dict:
    persona = _persona(category, persona_id)
    ledger = PreferenceLedger()
    pipeline = RecommendationPipeline(
        category=category,
        model=model,
    )
    statements = [
        str(statement).replace("[[", "").replace("]]", "").strip()
        for statement in persona.get("context_trail") or []
    ]
    for statement in statements:
        ledger.observe(statement, source="scripted_revealed_prefix")
    query = (query_override or " ".join(statements)).strip()
    key_query = (key_query_override or (statements[1] if len(statements) > 1 else query)).strip()
    snapshot = pipeline.select_for_query(
        ledger=ledger,
        question_ids=[],
        asks_so_far=len(statements),
        query=query,
        key_query=key_query,
    )
    chosen = snapshot.recommendations or []
    return {
        "category": category,
        "persona_id": persona_id,
        "ground_truth_need": persona.get("ground_truth_need"),
        "model": model,
        "revealed_statements": statements,
        "retrieval_query": snapshot.retrieval_query,
        "retrieval_key_query": snapshot.retrieval_key_query,
        "eligible_count": snapshot.eligible_count,
        "candidate_count": len(snapshot.retrieval_candidate_asins),
        "candidate_asins": snapshot.retrieval_candidate_asins,
        "retrieval_scores": snapshot.retrieval_scores,
        "retrieval_lane_sources": snapshot.retrieval_lane_sources,
        "product_numbers": snapshot.recommendation_product_numbers,
        "raw_model_response": snapshot.recommendation_selection_raw,
        "validation_error": snapshot.recommendation_validation_error,
        "explanations": [product.get("recommendation_explanation") for product in chosen],
        "selected_products": [
            {
                "asin": product.get("asin"),
                "title": product.get("title"),
                "price": product.get("price"),
                "avg_rating": product.get("avg_rating"),
                "num_reviews": product.get("num_reviews"),
            }
            for product in chosen
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="gpt-5-mini-2025-08-07")
    parser.add_argument("--parallel", type=int, default=3)
    parser.add_argument(
        "--query",
        default=None,
        help="embedding query to use (best with one --case); defaults to joined statements",
    )
    parser.add_argument(
        "--key-query",
        default=None,
        help="focused query for the most important revealed preference",
    )
    parser.add_argument(
        "--case",
        action="append",
        metavar="CATEGORY:PERSONA_ID",
        help=(
            "run only this case; repeat for multiple cases (default: the six documented cases)"
        ),
    )
    args = parser.parse_args()
    load_env()
    warm_dense_model()

    cases = list(CASES)
    if args.case:
        cases = []
        for value in args.case:
            category, separator, persona_id = value.partition(":")
            if not separator or not category or not persona_id:
                parser.error(f"invalid --case {value!r}; expected CATEGORY:PERSONA_ID")
            cases.append((category, persona_id))

    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
        records = list(
            pool.map(
                lambda case: _run_case(
                    *case,
                    args.model,
                    args.query,
                    args.key_query,
                ),
                cases,
            )
        )
    for record in records:
        print(json.dumps(record, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
