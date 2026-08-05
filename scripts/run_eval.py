"""Run API-backed buyer/recommender simulations for one product category."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from sandbox.catalog import (  # noqa: E402
    REPO_ROOT as CATALOG_ROOT,
    default_slate,
    load_config,
    validate_category,
    warm_dense_model,
)
from sandbox.env import load_env  # noqa: E402
from sandbox.simulation import Outcome, Policy, Simulation  # noqa: E402
from sandbox.questions import QuestionBank  # noqa: E402


def _load_personas(category: str) -> list[dict[str, Any]]:
    path = REPO_ROOT / "data" / "categories" / category / "personas.json"
    payload = json.loads(path.read_text())
    personas = list(payload.get("personas", payload))
    ids = [persona.get("id") for persona in personas]
    if not personas or any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError(f"{category!r} personas have missing or duplicate IDs")
    return personas


def _persona_slice(value: str, total: int) -> tuple[int, int]:
    try:
        start_text, end_text = value.split("-", 1)
        start, end = int(start_text), int(end_text)
    except ValueError as exc:
        raise ValueError("--persona-range must look like START-END, for example 51-100") from exc
    if start < 1 or end < start:
        raise ValueError("--persona-range uses positive 1-based inclusive bounds")
    if end > total:
        raise ValueError(f"--persona-range ends at {end}, but the category has {total} personas")
    return start - 1, end


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_one(
    persona: dict[str, Any],
    category: str,
    policy: Policy,
    args: argparse.Namespace,
) -> Outcome:
    return Simulation(
        persona=persona,
        category=category,
        policy=policy,
        max_turns=args.max_turns,
        endogenous_abandonment=args.endogenous_abandonment,
        buyer_model=args.buyer_model,
        recommender_model=args.recommender_model,
        retrieval_limit=args.retrieval_k,
        assortment_size=args.assortment_size,
    ).run()


def _mean(values: list[float]) -> float | None:
    return round(mean(values), 4) if values else None


def _aggregate_usage(outcomes: list[Outcome]) -> dict[str, Any]:
    calls = [call for outcome in outcomes for call in outcome.api_usage.get("calls", [])]
    by_kind: dict[str, Counter[str]] = defaultdict(Counter)
    for call in calls:
        kind = str(call.get("kind") or "unknown")
        by_kind[kind].update(
            requests=1,
            input_tokens=int(call.get("input_tokens") or 0),
            cached_input_tokens=int(call.get("cached_input_tokens") or 0),
            output_tokens=int(call.get("output_tokens") or 0),
        )
    known_costs = [
        float(call["estimated_cost_usd"])
        for call in calls
        if isinstance(call.get("estimated_cost_usd"), (int, float))
    ]
    unpriced_requests = len(calls) - len(known_costs)
    return {
        "requests": len(calls),
        "input_tokens": sum(int(call.get("input_tokens") or 0) for call in calls),
        "cached_input_tokens": sum(
            int(call.get("cached_input_tokens") or 0) for call in calls
        ),
        "output_tokens": sum(int(call.get("output_tokens") or 0) for call in calls),
        "latency_s": round(sum(float(call.get("latency_s") or 0) for call in calls), 3),
        "estimated_cost_usd": (
            round(sum(known_costs), 6) if not unpriced_requests else None
        ),
        "unpriced_requests": unpriced_requests,
        "by_kind": {kind: dict(counts) for kind, counts in sorted(by_kind.items())},
    }


def _checkpoint_metrics(outcomes: list[Outcome]) -> dict[str, Any]:
    grouped: dict[int, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    by_persona: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for outcome in outcomes:
        for checkpoint in outcome.checkpoints:
            depth = checkpoint.get("depth")
            if isinstance(depth, int):
                grouped[depth].append((outcome.persona_id, checkpoint))
                by_persona[outcome.persona_id][depth] = checkpoint

    summary: dict[str, Any] = {}
    for depth, entries in sorted(grouped.items()):
        expected_n = len(outcomes)
        scored = [checkpoint for _, checkpoint in entries if checkpoint.get("evaluation_status") == "scored"]
        purchases = [checkpoint for checkpoint in scored if checkpoint.get("outcome") == "PURCHASE"]
        revenue = [float(checkpoint.get("revenue") or 0) for checkpoint in scored]
        surplus = [float(checkpoint.get("consumer_surplus") or 0) for checkpoint in scored]
        purchase_surplus = [float(checkpoint["consumer_surplus"]) for checkpoint in purchases]
        wtp = [float(checkpoint["wtp"]) for checkpoint in purchases]

        gains = losses = same = 0
        slate_turnover: list[float] = []
        selected_turnover: list[float] = []
        if depth > 0:
            for persona_id, current in entries:
                previous = by_persona[persona_id].get(depth - 1)
                if (
                    previous is None
                    or previous.get("evaluation_status") != "scored"
                    or current.get("evaluation_status") != "scored"
                ):
                    continue
                previous_purchase = previous.get("outcome") == "PURCHASE"
                current_purchase = current.get("outcome") == "PURCHASE"
                if current_purchase and not previous_purchase:
                    gains += 1
                elif previous_purchase and not current_purchase:
                    losses += 1
                else:
                    same += 1
                previous_asins = {
                    product.get("asin")
                    for product in previous["recommendation_result"].get("recommendations", [])
                }
                current_asins = {
                    product.get("asin")
                    for product in current["recommendation_result"].get("recommendations", [])
                }
                previous_asins.discard(None)
                current_asins.discard(None)
                denominator = max(len(previous_asins), len(current_asins), 1)
                slate_turnover.append(1 - len(previous_asins & current_asins) / denominator)
                selected_turnover.append(
                    float(previous.get("purchased_asin") != current.get("purchased_asin"))
                )
        paired = gains + losses + same
        summary[str(depth)] = {
            "expected_n": expected_n,
            "observed_n": len(entries),
            "scored_n": len(scored),
            "protocol_error_n": len(entries) - len(scored),
            "missing_n": expected_n - len(entries),
            "purchase_rate": round(len(purchases) / expected_n, 4) if expected_n else None,
            "mean_wtp_among_purchases": _mean(wtp),
            "total_revenue": round(sum(revenue), 2),
            "mean_revenue_per_persona": (
                round(sum(revenue) / expected_n, 4) if expected_n else None
            ),
            "average_order_value": _mean(
                [float(checkpoint["actual_price"]) for checkpoint in purchases]
            ),
            "mean_consumer_surplus_all_personas": (
                round(sum(surplus) / expected_n, 4) if expected_n else None
            ),
            "mean_consumer_surplus_among_purchases": _mean(purchase_surplus),
            "stepwise_purchase_gains": gains,
            "stepwise_purchase_losses": losses,
            "persona_regression_rate": round(losses / paired, 4) if paired else None,
            "mean_recommendation_set_turnover": _mean(slate_turnover),
            "mean_selected_product_turnover": _mean(selected_turnover),
        }
    return summary


def _summarize(outcomes: list[Outcome]) -> dict[str, Any]:
    n = len(outcomes)
    counts = Counter(outcome.outcome for outcome in outcomes)
    purchases = [outcome for outcome in outcomes if outcome.outcome == "PURCHASE"]
    revenue = [outcome.revenue for outcome in outcomes]
    surplus = [outcome.consumer_surplus for outcome in outcomes]
    engagement_decisions = [
        decision
        for outcome in outcomes
        for decision in outcome.engagement_decisions
    ]
    engagement_fallbacks = [
        decision
        for decision in engagement_decisions
        if decision.get("evaluation_status") == "fallback_answer"
    ]
    topics = Counter(topic for outcome in outcomes for topic in outcome.question_topics if topic)
    return {
        "n": n,
        "outcomes": dict(counts),
        "purchase_rate": round(len(purchases) / n, 4) if n else None,
        "abandonment_rate": round(counts["ABANDONED"] / n, 4) if n else None,
        "protocol_error_rate": round(counts["PROTOCOL_ERROR"] / n, 4) if n else None,
        "mean_wtp_among_purchases": _mean([outcome.wtp for outcome in purchases if outcome.wtp is not None]),
        "total_revenue": round(sum(revenue), 2),
        "mean_revenue_per_persona": _mean(revenue),
        "average_order_value": _mean(
            [outcome.actual_price for outcome in purchases if outcome.actual_price is not None]
        ),
        "mean_consumer_surplus_all_personas": _mean(surplus),
        "mean_consumer_surplus_among_purchases": _mean(
            [outcome.consumer_surplus for outcome in purchases]
        ),
        "mean_turns": _mean([float(outcome.turns_used) for outcome in outcomes]),
        "mean_questions": _mean([float(outcome.asks) for outcome in outcomes]),
        "question_topics": dict(topics),
        "engagement_decision_count": len(engagement_decisions),
        "engagement_decision_fallback_count": len(engagement_fallbacks),
        "checkpoint_metrics": _checkpoint_metrics(outcomes),
        "api_usage": _aggregate_usage(outcomes),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("category")
    parser.add_argument(
        "--policy",
        choices=["adaptive", "rec", "single_atr", "branching_atr"],
        default="adaptive",
    )
    parser.add_argument("--numquestions", type=int)
    parser.add_argument(
        "--n-personas",
        type=int,
        default=10,
        help="number of personas from the beginning; ignored by explicit ID/range selection",
    )
    persona_selection = parser.add_mutually_exclusive_group()
    persona_selection.add_argument("--persona-ids")
    persona_selection.add_argument(
        "--persona-range",
        metavar="START-END",
        help="1-based inclusive positions in the persona file, for example 51-100",
    )
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--max-turns", type=int, default=16)
    parser.add_argument("--retrieval-k", type=int, default=15)
    parser.add_argument("--assortment-size", type=int, default=5)
    parser.add_argument("--buyer-model", default="gpt-5.6-luna")
    parser.add_argument("--recommender-model", default="gpt-5.6-luna")
    parser.add_argument("--endogenous-abandonment", action="store_true")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--fail-on-protocol-error", action="store_true")
    args = parser.parse_args()

    load_env()
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key or api_key == "sk-...":
        parser.error("OPENAI_API_KEY is required for evaluation")
    try:
        policy = Policy.make(args.policy, args.numquestions)
        validate_category(args.category)
        personas = _load_personas(args.category)
    except ValueError as exc:
        parser.error(str(exc))
    if args.n_personas < 1:
        parser.error("--n-personas must be positive")
    if args.parallel < 1:
        parser.error("--parallel must be positive")
    if args.max_turns < 1:
        parser.error("--max-turns must be positive")
    if args.assortment_size < 1:
        parser.error("--assortment-size must be positive")
    if args.retrieval_k < args.assortment_size:
        parser.error("--retrieval-k must be at least --assortment-size")
    if policy.numquestions > len(QuestionBank.load(args.category).questions):
        parser.error("policy requests more questions than the category defines")
    if policy.name != "adaptive" and args.max_turns < policy.numquestions + 1:
        parser.error("--max-turns must include all questions plus the recommendation")

    if args.persona_range:
        try:
            start, end = _persona_slice(args.persona_range, len(personas))
        except ValueError as exc:
            parser.error(str(exc))
        personas = personas[start:end]
    elif args.persona_ids:
        wanted = {value.strip() for value in args.persona_ids.split(",") if value.strip()}
        personas = [persona for persona in personas if persona.get("id") in wanted]
        if len(personas) != len(wanted):
            parser.error("one or more --persona-ids were not found")
    else:
        personas = personas[: args.n_personas]
    if not personas:
        parser.error("no personas selected")
    warm_dense_model()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = args.out_dir or (
        REPO_ROOT / "results" / "local" / f"{args.category}_{policy.name}_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    partial_path = output_dir / "transcripts.partial.jsonl"
    transcripts_path = output_dir / "transcripts.jsonl"
    summary_path = output_dir / "summary.json"
    partial_path.write_text("")

    started = time.time()
    outcomes_by_index: dict[int, Outcome] = {}
    with partial_path.open("a") as partial:
        with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
            futures = {
                pool.submit(_run_one, persona, args.category, policy, args): (index, persona)
                for index, persona in enumerate(personas)
            }
            for future in as_completed(futures):
                index, persona = futures[future]
                try:
                    outcome = future.result()
                except Exception as exc:
                    outcome = Outcome(
                        persona_id=str(persona.get("id") or ""),
                        category=args.category,
                        policy=policy.name,
                        numquestions=policy.numquestions,
                        outcome="PROTOCOL_ERROR",
                        turns_used=0,
                        asks=0,
                        error_code=type(exc).__name__.upper(),
                        error=f"{type(exc).__name__}: {exc}",
                    )
                outcomes_by_index[index] = outcome
                partial.write(json.dumps({"index": index, "outcome": outcome.to_dict()}) + "\n")
                partial.flush()
                print(f"{outcome.persona_id}: {outcome.outcome}", flush=True)

    outcomes = [outcomes_by_index[index] for index in range(len(personas))]
    # Rewrite the temporary file in canonical order, then atomically promote it.
    with partial_path.open("w") as handle:
        for outcome in outcomes:
            handle.write(json.dumps(outcome.to_dict()) + "\n")
    partial_path.replace(transcripts_path)

    config = load_config(args.category)
    catalog_path = CATALOG_ROOT / config["catalog_path"]
    personas_path = catalog_path.with_name("personas.json")
    questions_path = catalog_path.with_name("questions.yaml")
    elapsed = time.time() - started
    summary = {
        "schema_version": "crs-evaluation-v2",
        "run": {
            "category": args.category,
            "policy": policy.name,
            "numquestions": policy.numquestions,
            "persona_range": args.persona_range,
            "persona_ids": [outcome.persona_id for outcome in outcomes],
            "max_turns": args.max_turns,
            "retrieval_k": args.retrieval_k,
            "assortment_size": args.assortment_size,
            "carry_forward_recommendations": policy.name == "branching_atr",
            "maximum_reranker_candidates": (
                args.retrieval_k
                + len(default_slate(args.category))
                + (args.assortment_size if policy.name == "branching_atr" else 0)
            ),
            "default_slate": [product["asin"] for product in default_slate(args.category)],
            "endogenous_abandonment": args.endogenous_abandonment,
            "buyer_model": args.buyer_model,
            "recommender_model": args.recommender_model,
            "catalog_sha256": _sha256(catalog_path),
            "personas_sha256": _sha256(personas_path),
            "questions_sha256": _sha256(questions_path),
            "wall_clock_s": round(elapsed, 2),
        },
        "metrics": _summarize(outcomes),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["metrics"], indent=2))
    print(f"wrote {transcripts_path}")
    print(f"wrote {summary_path}")
    if args.fail_on_protocol_error and summary["metrics"]["protocol_error_rate"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
