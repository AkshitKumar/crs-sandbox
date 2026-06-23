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
    python scripts/run_buyer_eval.py laptop --policy atr_recs --numquestions 3

Cost rough estimate: ~$0.15–0.30 per conversation. 10 personas ≈ $2–3.
"""

from __future__ import annotations

import argparse
import json
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
    seed: int,
    elicitation_policy: ElicitationPolicy | None,
) -> SimOutcome:
    sim = SimConversation(
        persona=persona,
        category=category,
        max_turns=max_turns,
        eta=eta,
        seed=seed,
        elicitation_policy=elicitation_policy,
    )
    return sim.run()


def _summarize(outcomes: list[SimOutcome]) -> dict:
    """Aggregate metrics over a batch of outcomes."""
    n = len(outcomes)
    if n == 0:
        return {"n": 0}

    by_outcome = {}
    for o in outcomes:
        by_outcome[o.outcome] = by_outcome.get(o.outcome, 0) + 1
    purchased = [o for o in outcomes if o.outcome == "PURCHASE"]
    finished = [o for o in outcomes if o.outcome in ("PURCHASE", "NO_PURCHASE")]

    wtp_values = [o.wtp for o in purchased if isinstance(o.wtp, (int, float))]
    surplus_values = [o.consumer_surplus for o in purchased if isinstance(o.consumer_surplus, (int, float))]
    turns_values = [o.turns_used for o in finished]
    asks_values = [o.asks for o in finished]

    return {
        "n": n,
        "outcomes": by_outcome,
        "purchase_rate": round(len(purchased) / n, 4),
        "abandonment_rate": round(by_outcome.get("ABANDONED", 0) / n, 4),
        "error_rate": round(by_outcome.get("ERROR", 0) / n, 4),
        "mean_turns_among_finished": round(mean(turns_values), 2) if turns_values else None,
        "mean_asks_among_finished": round(mean(asks_values), 2) if asks_values else None,
        "mean_wtp_among_purchased": round(mean(wtp_values), 2) if wtp_values else None,
        "mean_consumer_surplus_among_purchased": (
            round(mean(surplus_values), 2) if surplus_values else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("category", help="category name (must have personas.json + index built)")
    parser.add_argument("--n-personas", type=int, default=10, help="how many personas to run (default 10)")
    parser.add_argument(
        "--persona-ids",
        default=None,
        help="comma-separated list of persona ids to run (overrides --n-personas)",
    )
    parser.add_argument("--max-turns", type=int, default=16)
    parser.add_argument("--eta", type=float, default=0.0, help="per-turn abandonment hazard (0.0 = no attrition)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--parallel", type=int, default=4, help="max concurrent conversations")
    parser.add_argument(
        "--policy",
        choices=["rec", "atr", "atr_recs"],
        default=None,
        help="optional fixed elicitation policy; omit to keep the current adaptive behavior",
    )
    parser.add_argument(
        "--numquestions",
        type=int,
        default=None,
        help="number of clarifying questions for --policy atr or atr_recs",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="directory to write transcripts + summary (default: results/eval_<category>_<ts>/)",
    )
    args = parser.parse_args()

    elicitation_policy = None
    if args.policy is not None:
        try:
            elicitation_policy = make_policy(args.policy, args.numquestions)
        except ValueError as e:
            parser.error(str(e))
    elif args.numquestions is not None:
        parser.error("--numquestions requires --policy atr or atr_recs")

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
    if elicitation_policy is not None:
        print(f"  policy={elicitation_policy.name}  numquestions={elicitation_policy.target_asks}")
    print(f"  output: {out_dir}")

    transcripts_path = out_dir / "transcripts.jsonl"
    summary_path = out_dir / "summary.json"
    transcripts_path.unlink(missing_ok=True)

    outcomes: list[SimOutcome] = []
    started = time.time()

    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
        futures = {
            pool.submit(
                _run_one,
                p,
                args.category,
                args.max_turns,
                args.eta,
                args.seed + i,
                elicitation_policy,
            ): p
            for i, p in enumerate(personas)
        }
        with transcripts_path.open("a") as out_f:
            for fut in as_completed(futures):
                persona = futures[fut]
                try:
                    out = fut.result()
                except Exception as e:
                    out = SimOutcome(
                        persona_id=persona.get("id", ""),
                        category=args.category,
                        outcome="ERROR",
                        turns_used=0,
                        asks=0,
                        policy=elicitation_policy.name if elicitation_policy else None,
                        numquestions=elicitation_policy.target_asks if elicitation_policy else None,
                        error=f"{type(e).__name__}: {e}",
                    )
                outcomes.append(out)
                out_f.write(json.dumps(out.to_dict()) + "\n")
                out_f.flush()
                marker = {"PURCHASE": "✅", "NO_PURCHASE": "✘", "ABANDONED": "⏸", "ERROR": "⚠"}.get(out.outcome, "?")
                wtp_s = f" wtp=${out.wtp}" if out.wtp else ""
                err = f"  [{out.error[:80]}]" if out.error else ""
                print(f"  {marker} {out.persona_id:<24} {out.outcome:<12} turns={out.turns_used} asks={out.asks}{wtp_s}{err}")

    elapsed = time.time() - started
    summary = _summarize(outcomes)
    summary["category"] = args.category
    summary["max_turns"] = args.max_turns
    summary["eta"] = args.eta
    if elicitation_policy is not None:
        summary["policy"] = elicitation_policy.name
        summary["numquestions"] = elicitation_policy.target_asks
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
    print(f"  mean_turns_among_finished: {summary['mean_turns_among_finished']}")
    print(f"  mean_asks_among_finished: {summary['mean_asks_among_finished']}")
    print(f"  mean_wtp_among_purchased: ${summary['mean_wtp_among_purchased']}")
    print(f"  transcripts:  {transcripts_path}")
    print(f"  summary:      {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
