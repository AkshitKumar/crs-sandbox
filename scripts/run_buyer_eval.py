"""Run buyer-simulator evaluation for one category.

Runs N personas from the category's personas.json through the
rufus-femto CRS agent, collects outcomes (purchase/no-purchase/WTP/turns/abandonment),
writes per-conversation transcripts + a summary aggregate to results/.

Usage:
    python scripts/run_buyer_eval.py laptop --n-personas 10
    python scripts/run_buyer_eval.py air_purifier --n-personas 20 --eta 0.05
    python scripts/run_buyer_eval.py laptop --persona-ids laptop_001,laptop_002
    python scripts/run_buyer_eval.py laptop --policy rec
    python scripts/run_buyer_eval.py laptop --policy atr --numquestions 3
    python scripts/run_buyer_eval.py laptop --policy checkpoint_atr --numquestions 3
    python scripts/run_buyer_eval.py laptop --endogenous-abandonment

Cost rough estimate: ~$0.15–0.30 per conversation. 10 personas ≈ $2–3.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from sandbox.env import load_env  # noqa: E402

load_env()

from sandbox.elicitation_policy import ElicitationPolicy, make_policy  # noqa: E402
from sandbox.catalog import load_config  # noqa: E402
from sandbox.orchestrator.sim_conversation import SimConversation, SimOutcome  # noqa: E402


def _load_personas(category: str) -> list[dict]:
    """Load the persona list for a category from its self-contained file."""
    path = REPO_ROOT / "data" / "categories" / category / "personas.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No personas at {path}. Add them with the gen pipeline or copy from your dataset."
        )
    data = json.loads(path.read_text())
    return data.get("personas", data)


def _run_one(
    persona: dict,
    category: str,
    max_turns: int,
    eta: float,
    abandonment_seed: int,
    elicitation_policy: ElicitationPolicy | None,
    endogenous_abandonment: bool,
    buyer_model: str,
    recommender_model: str,
    retrieval_limit: int,
) -> SimOutcome:
    sim = SimConversation(
        persona=persona,
        category=category,
        max_turns=max_turns,
        eta=eta,
        abandonment_seed=abandonment_seed,
        elicitation_policy=elicitation_policy,
        endogenous_abandonment=endogenous_abandonment,
        buyer_model=buyer_model,
        recommender_model=recommender_model,
        retrieval_limit=retrieval_limit,
    )
    return sim.run()


def _summarize(outcomes: list[SimOutcome]) -> dict:
    """Aggregate metrics over a batch of outcomes."""
    n = len(outcomes)
    if n == 0:
        return {"n": 0}

    by_outcome = {}
    abandonment_types = {}
    abandonment_reasons = {}
    for o in outcomes:
        by_outcome[o.outcome] = by_outcome.get(o.outcome, 0) + 1
        if o.outcome == "ABANDONED":
            if o.abandonment_type:
                abandonment_types[o.abandonment_type] = abandonment_types.get(o.abandonment_type, 0) + 1
            if o.abandonment_reason:
                abandonment_reasons[o.abandonment_reason] = abandonment_reasons.get(o.abandonment_reason, 0) + 1
    purchased = [o for o in outcomes if o.outcome == "PURCHASE"]
    finished = [
        o for o in outcomes if o.outcome in ("PURCHASE", "NO_PURCHASE", "NO_FEASIBLE_MATCH")
    ]

    wtp_values = [o.wtp for o in purchased if isinstance(o.wtp, (int, float))]
    surplus_values = [o.consumer_surplus for o in purchased if isinstance(o.consumer_surplus, (int, float))]
    turns_values = [o.turns_used for o in finished]
    asks_values = [o.asks for o in finished]

    return {
        "n": n,
        "outcomes": by_outcome,
        "purchase_rate": round(len(purchased) / n, 4),
        "abandonment_rate": round(by_outcome.get("ABANDONED", 0) / n, 4),
        "abandonment_types": abandonment_types,
        "abandonment_reasons": abandonment_reasons,
        "protocol_error_rate": round(by_outcome.get("PROTOCOL_ERROR", 0) / n, 4),
        "no_feasible_match_rate": round(by_outcome.get("NO_FEASIBLE_MATCH", 0) / n, 4),
        "mean_turns_among_finished": round(mean(turns_values), 2) if turns_values else None,
        "mean_asks_among_finished": round(mean(asks_values), 2) if asks_values else None,
        "mean_wtp_among_purchased": round(mean(wtp_values), 2) if wtp_values else None,
        "mean_consumer_surplus_among_purchased": (
            round(mean(surplus_values), 2) if surplus_values else None
        ),
    }


def _summarize_checkpoints(outcomes: list[SimOutcome]) -> dict[str, dict]:
    """Aggregate paired hidden-prefix observations by checkpoint index.

    ``current_*`` metrics describe that prefix's card set. ``best_observed_*``
    is historical over the snapshots already measured; it is intentionally not
    presented as a currently feasible recommendation after later constraints.
    """
    grouped: dict[int, list[dict]] = {}
    for outcome in outcomes:
        for checkpoint in outcome.recommendation_checkpoints:
            index = checkpoint.get("index")
            if isinstance(index, int):
                grouped.setdefault(index, []).append(checkpoint)

    summary: dict[str, dict] = {}
    for index, checkpoints in sorted(grouped.items()):
        scored = [item for item in checkpoints if item.get("evaluation_status") == "scored"]
        current_purchases = [item for item in scored if item.get("current_purchase")]
        current_surplus = [
            item["counterfactual_consumer_surplus"]
            for item in current_purchases
            if isinstance(item.get("counterfactual_consumer_surplus"), (int, float))
        ]
        current_utility = [
            item["current_utility"] for item in scored if isinstance(item.get("current_utility"), (int, float))
        ]
        best_utility = [
            item["best_observed_utility"]
            for item in scored
            if isinstance(item.get("best_observed_utility"), (int, float))
        ]
        degradation_observations = [item for item in scored if index > 0]
        summary[str(index)] = {
            "n": len(checkpoints),
            "scored_n": len(scored),
            "snapshot_evaluator_error_n": sum(
                item.get("evaluation_status") == "snapshot_evaluator_error" for item in checkpoints
            ),
            "current_purchase_rate": round(
                sum(item.get("current_purchase") is True for item in scored) / len(scored), 4
            ) if scored else None,
            "best_observed_purchase_rate": round(
                sum(item.get("best_observed_purchase") is True for item in scored) / len(scored), 4
            ) if scored else None,
            "mean_current_utility": round(mean(current_utility), 2) if current_utility else None,
            "mean_best_observed_utility": round(mean(best_utility), 2) if best_utility else None,
            "mean_conditional_surplus_among_current_purchases": (
                round(mean(current_surplus), 2) if current_surplus else None
            ),
            "degradation_rate_from_previous": round(
                sum(item.get("degraded_from_previous") is True for item in degradation_observations)
                / len(degradation_observations),
                4,
            ) if degradation_observations else None,
        }
    return summary


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_value(*args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _dependency_versions() -> dict[str, str | None]:
    packages = [
        "openai",
        "langchain-openai",
        "langchain-core",
        "langgraph",
        "PyYAML",
        "numpy",
        "sentence-transformers",
        "torch",
        "transformers",
        "tokenizers",
        "huggingface-hub",
    ]
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _build_manifest(
    *,
    category: str,
    personas: list[dict],
    args: argparse.Namespace,
    policy: ElicitationPolicy | None,
) -> dict:
    """Capture the inputs that must be held fixed across policy arms."""
    catalog_path = REPO_ROOT / "data" / "categories" / category / "products.jsonl"
    personas_path = REPO_ROOT / "data" / "categories" / category / "personas.json"
    questions_path = REPO_ROOT / "data" / "categories" / category / "questions.yaml"
    config_path = REPO_ROOT / "configs" / f"{category}.yaml"
    protocol_path = REPO_ROOT / "src" / "sandbox" / "agents" / "recommendation_pipeline.py"
    buyer_path = REPO_ROOT / "src" / "sandbox" / "agents" / "buyer.py"
    behavior_sources = {
        "recommendation_pipeline": protocol_path,
        "recommender": REPO_ROOT / "src" / "sandbox" / "agents" / "langgraph_crs.py",
        "buyer_simulator": buyer_path,
        "orchestrator": REPO_ROOT / "src" / "sandbox" / "orchestrator" / "sim_conversation.py",
        "elicitation_policy": REPO_ROOT / "src" / "sandbox" / "elicitation_policy.py",
        "question_bank": REPO_ROOT / "src" / "sandbox" / "tools" / "question_bank.py",
        "dense_search": REPO_ROOT / "src" / "sandbox" / "tools" / "search_tool.py",
        "product_text": REPO_ROOT / "src" / "sandbox" / "product_text.py",
        "index_builder": REPO_ROOT / "src" / "sandbox" / "index" / "builder.py",
        "openai_responses": REPO_ROOT / "src" / "sandbox" / "openai_responses.py",
    }
    index_manifest_path = (
        REPO_ROOT / load_config(category)["index_dir"] / "index_manifest.json"
    )
    index_manifest = json.loads(index_manifest_path.read_text())
    status = _git_value("status", "--porcelain") or ""
    return {
        "schema_version": "evaluation-manifest-v7",
        "category": category,
        "policy": {
            "name": policy.name if policy is not None else "adaptive",
            "target_asks": policy.target_asks if policy is not None else None,
        },
        "default_slate": list(load_config(category).get("default_slate") or []),
        "persona_ids": [persona.get("id", "") for persona in personas],
        "abandonment_seeds": [args.abandonment_seed + index for index in range(len(personas))],
        "max_turns": args.max_turns,
        "eta": args.eta,
        "endogenous_abandonment": args.endogenous_abandonment,
        "fail_on_protocol_error": args.fail_on_protocol_error,
        "models": {
            "buyer": args.buyer_model,
            "recommender": args.recommender_model,
            "recommendation_selector_and_prose": args.recommender_model,
        },
        "candidate_order": (
            "curated default slate"
            if policy is not None and policy.name == "rec"
            else (
                "checkpoint 0 curated; later prefixes use ReAct-supplied query, "
                "key query, hard-budget eligibility, and round-robin hybrid retrieval"
                if policy is not None and policy.has_nonterminal_checkpoints
                else "ReAct-supplied full/key queries; hard-budget eligibility; hybrid retrieval"
            )
        ),
        "recommendation_selection": {
            "local_training": False,
            "retrieval_limit": (
                None if policy is not None and policy.name == "rec" else args.retrieval_k
            ),
            "retrieval_query_policy": (
                "latest full and key queries supplied by ReAct; direct revealed-ledger fallback "
                "if the agent has not supplied a full query"
                if policy is not None and policy.has_nonterminal_checkpoints
                else None
            ),
            "method": (
                "curated default slate followed by one fixed-list personalized-prose API call"
                if policy is not None and policy.name == "rec"
                else (
                    "checkpoint 0 uses the curated slate; later prefixes use a ReAct-supplied "
                    "full query plus key query, round-robin BGE/BM25/key-query recall, and one "
                    "combined select-and-explain API call"
                    if policy is not None and policy.has_nonterminal_checkpoints
                    else (
                        "ReAct supplies full and key queries; round-robin BGE/BM25/key-query "
                        "retrieval produces the candidate pool; one API call selects three and "
                        "writes one short explanation for each"
                    )
                )
            ),
        },
        "checkpoint_evaluator": {
            "enabled": policy.has_nonterminal_checkpoints if policy is not None else False,
            "model": (
                args.buyer_model
                if policy is not None and policy.has_nonterminal_checkpoints
                else None
            ),
            "timing": "hidden prefix snapshots 0..k; buyer history is not mutated",
        },
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_dirty": bool(status),
        "catalog_sha256": _sha256(catalog_path),
        "personas_sha256": _sha256(personas_path),
        "config_sha256": _sha256(config_path),
        "question_bank_sha256": _sha256(questions_path),
        "index_manifest_sha256": _sha256(index_manifest_path),
        "dense_index": index_manifest,
        "recommendation_pipeline_source_sha256": _sha256(protocol_path),
        "recommender_source_sha256": _sha256(
            REPO_ROOT / "src" / "sandbox" / "agents" / "langgraph_crs.py"
        ),
        "buyer_prompt_source_sha256": _sha256(buyer_path),
        "behavior_source_sha256": {name: _sha256(path) for name, path in behavior_sources.items()},
        "dependency_versions": _dependency_versions(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("category", help="category name (must have personas.json + a question bank)")
    parser.add_argument("--n-personas", type=int, default=10, help="how many personas to run (default 10)")
    parser.add_argument(
        "--persona-ids",
        default=None,
        help="comma-separated list of persona ids to run (overrides --n-personas)",
    )
    parser.add_argument("--max-turns", type=int, default=16)
    parser.add_argument("--eta", type=float, default=0.0, help="per-turn abandonment hazard (0.0 = no attrition)")
    parser.add_argument(
        "--endogenous-abandonment",
        action="store_true",
        help="allow the buyer LLM to abandon when the conversation feels unhelpful",
    )
    parser.add_argument(
        "--abandonment-seed",
        "--seed",
        dest="abandonment_seed",
        type=int,
        default=0,
        help="seed for exogenous abandonment draws only (does not seed model calls)",
    )
    parser.add_argument("--parallel", type=int, default=4, help="max concurrent conversations")
    parser.add_argument(
        "--policy",
        choices=["rec", "atr", "checkpoint_atr"],
        default=None,
        metavar="{rec,atr,checkpoint_atr}",
        help=(
            "optional controlled policy; omit for the ordinary adaptive ReAct recommender"
        ),
    )
    parser.add_argument(
        "--numquestions",
        type=int,
        default=None,
        help="number of clarifying questions for --policy atr or checkpoint_atr",
    )
    parser.add_argument("--buyer-model", default="gpt-5-mini-2025-08-07")
    parser.add_argument("--recommender-model", default="gpt-5-mini-2025-08-07")
    parser.add_argument(
        "--retrieval-k",
        type=int,
        default=15,
        help="dense candidates shown to the combined selector/explainer (default 15)",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="directory to write transcripts + summary (default: results/eval_<category>_<ts>/)",
    )
    parser.add_argument(
        "--fail-on-protocol-error",
        action="store_true",
        help=(
            "return a nonzero process status after writing outputs if any conversation "
            "ended in PROTOCOL_ERROR"
        ),
    )
    args = parser.parse_args()

    elicitation_policy = None
    if args.policy is not None:
        try:
            elicitation_policy = make_policy(args.policy, args.numquestions)
        except ValueError as e:
            parser.error(str(e))
    elif args.numquestions is not None:
        parser.error("--numquestions requires --policy atr or checkpoint_atr")
    if (
        elicitation_policy is not None
        and args.max_turns < elicitation_policy.target_asks + 1
    ):
        parser.error(
            f"--max-turns must be at least {elicitation_policy.target_asks + 1} for "
            f"{elicitation_policy.label} (questions plus the terminal recommendation)"
        )
    if elicitation_policy is not None and args.retrieval_k < 3:
        parser.error("--retrieval-k must be at least 3")
    if elicitation_policy is not None and elicitation_policy.has_nonterminal_checkpoints and (
        args.eta != 0.0 or args.endogenous_abandonment
    ):
        parser.error(
            "checkpoint_atr requires --eta 0 and no --endogenous-abandonment so every prefix 0..k is observed"
        )

    personas = _load_personas(args.category)
    if args.persona_ids:
        wanted = {p.strip() for p in args.persona_ids.split(",") if p.strip()}
        personas = [p for p in personas if p.get("id") in wanted]
        if not personas:
            print(f"No personas matched --persona-ids {wanted}", file=sys.stderr)
            return 1
    else:
        personas = personas[: args.n_personas]

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "results" / f"eval_{args.category}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Running {len(personas)} personas against {args.category}")
    print(f"  parallel={args.parallel}  eta={args.eta}  max_turns={args.max_turns}")
    print(f"  endogenous_abandonment={args.endogenous_abandonment}")
    if elicitation_policy is not None:
        print(f"  policy={elicitation_policy.name}  numquestions={elicitation_policy.target_asks}")
    else:
        print("  policy=adaptive (ordinary ReAct recommender)")
    print(f"  output: {out_dir}")

    transcripts_path = out_dir / "transcripts.jsonl"
    partial_transcripts_path = out_dir / "transcripts.partial.jsonl"
    summary_path = out_dir / "summary.json"
    manifest_path = out_dir / "run_manifest.json"
    transcripts_path.write_text("")
    partial_transcripts_path.write_text("")

    manifest = _build_manifest(
        category=args.category,
        personas=personas,
        args=args,
        policy=elicitation_policy,
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    if elicitation_policy is not None and elicitation_policy.target_asks > 0:
        # SentenceTransformer construction is not reliably thread-safe. Load
        # the pinned query encoder once before conversation workers start.
        from sandbox.tools.search_tool import warm_dense_model

        warm_dense_model()

    outcomes: list[SimOutcome] = []
    started = time.time()

    with partial_transcripts_path.open("a") as partial_file:
        with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
            futures = {
                pool.submit(
                    _run_one,
                    p,
                    args.category,
                    args.max_turns,
                    args.eta,
                    args.abandonment_seed + i,
                    elicitation_policy,
                    args.endogenous_abandonment,
                    args.buyer_model,
                    args.recommender_model,
                    args.retrieval_k,
                ): (i, p)
                for i, p in enumerate(personas)
            }
            outcomes_by_index: dict[int, SimOutcome] = {}
            for fut in as_completed(futures):
                index, persona = futures[fut]
                try:
                    out = fut.result()
                except Exception as e:
                    out = SimOutcome(
                        persona_id=persona.get("id", ""),
                        category=args.category,
                        outcome="PROTOCOL_ERROR",
                        turns_used=0,
                        asks=0,
                        policy=(elicitation_policy.name if elicitation_policy else None),
                        numquestions=(
                            elicitation_policy.target_asks if elicitation_policy else None
                        ),
                        error=f"{type(e).__name__}: {e}",
                    )
                outcomes_by_index[index] = out
                partial_file.write(json.dumps({"index": index, "outcome": out.to_dict()}) + "\n")
                partial_file.flush()
                marker = {
                    "PURCHASE": "✅", "NO_PURCHASE": "✘", "NO_FEASIBLE_MATCH": "⊘",
                    "ABANDONED": "⏸", "PROTOCOL_ERROR": "⚠",
                }.get(out.outcome, "?")
                wtp_s = f" wtp=${out.wtp}" if out.wtp is not None else ""
                err = f"  [{out.error[:80]}]" if out.error else ""
                print(f"  {marker} {out.persona_id:<24} {out.outcome:<18} turns={out.turns_used} asks={out.asks}{wtp_s}{err}")

    outcomes = [outcomes_by_index[index] for index in range(len(personas))]
    with transcripts_path.open("w") as out_f:
        for out in outcomes:
            out_f.write(json.dumps(out.to_dict()) + "\n")

    elapsed = time.time() - started
    summary = _summarize(outcomes)
    summary["category"] = args.category
    summary["max_turns"] = args.max_turns
    summary["eta"] = args.eta
    summary["endogenous_abandonment"] = args.endogenous_abandonment
    if elicitation_policy is not None:
        summary["policy"] = elicitation_policy.name
        summary["numquestions"] = elicitation_policy.target_asks
    try:
        summary["manifest_path"] = str(manifest_path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        summary["manifest_path"] = str(manifest_path)
    summary["manifest"] = manifest
    summary["completed_personas"] = len(outcomes)
    summary["expected_personas"] = len(personas)
    summary["partial_transcripts_path"] = str(partial_transcripts_path)
    recommendation_sources: dict[str, int] = {}
    for outcome in outcomes:
        if outcome.recommendation_source:
            source = outcome.recommendation_source
            recommendation_sources[source] = recommendation_sources.get(source, 0) + 1
    summary["recommendation_sources"] = recommendation_sources
    summary["checkpoint_metrics"] = _summarize_checkpoints(outcomes)
    summary["wall_clock_s"] = round(elapsed, 1)
    try:
        summary["transcripts_path"] = str(
            transcripts_path.resolve().relative_to(REPO_ROOT.resolve())
        )
    except ValueError:
        summary["transcripts_path"] = str(transcripts_path)

    summary_path.write_text(json.dumps(summary, indent=2))

    print()
    print(f"Done in {elapsed:.1f}s.")
    print(f"  purchase_rate: {summary['purchase_rate']:.0%}")
    print(f"  abandonment_rate: {summary['abandonment_rate']:.0%}")
    print(f"  protocol_error_rate: {summary['protocol_error_rate']:.0%}")
    print(f"  mean_turns_among_finished: {summary['mean_turns_among_finished']}")
    print(f"  mean_asks_among_finished: {summary['mean_asks_among_finished']}")
    print(f"  mean_wtp_among_purchased: ${summary['mean_wtp_among_purchased']}")
    print(f"  transcripts:  {transcripts_path}")
    print(f"  manifest:     {manifest_path}")
    print(f"  summary:      {summary_path}")
    if args.fail_on_protocol_error and summary["protocol_error_rate"] > 0:
        print("  failing process because --fail-on-protocol-error was set", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
