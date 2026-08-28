"""Apply the production abandonment evaluator to a completed control run.

Each persona is replayed sequentially.  The evaluator sees the same persona,
question, and stored conversation prefix that it would see in a live run.  A
persona stops at the first ABANDON decision; no later abandonment calls are
made.  Stored answers, checkpoints, recommendations, and purchase decisions
are never regenerated.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sandbox.agents.buyer import (  # noqa: E402
    ABANDONMENT_PROMPT,
    ABANDON_SENTINEL,
    BuyerAgent,
    DEFAULT_MODEL,
)
from sandbox.env import load_env  # noqa: E402
from sandbox.openai_responses import make_client  # noqa: E402


SEARCH_ROOTS = (
    REPO_ROOT / "results" / "bouchet",
    REPO_ROOT / "results" / "bouchet_day",
    REPO_ROOT / "results" / "local",
)
MAX_PARALLEL = 20
_thread_local = threading.local()


def _sha256(path: Path) -> str:
    """Return a content hash used to identify the exact source artifacts."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a non-empty JSONL file and report malformed lines precisely."""

    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: malformed JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path} is empty")
    return rows


def _resolve_source_run(value: str) -> Path:
    """Resolve either a run-directory path or an exact run title."""

    supplied = Path(value).expanduser()
    if supplied.is_dir():
        return supplied.resolve()
    matches = [root / value for root in SEARCH_ROOTS if (root / value).is_dir()]
    if not matches:
        roots = ", ".join(str(root) for root in SEARCH_ROOTS)
        raise ValueError(f"run {value!r} was not found under: {roots}")
    if len(matches) > 1:
        rendered = ", ".join(str(path) for path in matches)
        raise ValueError(f"run title {value!r} is ambiguous: {rendered}")
    return matches[0].resolve()


def _load_source_run(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load and validate a complete, non-abandonment branching run."""

    summary_path = run_dir / "summary.json"
    transcripts_path = run_dir / "transcripts.jsonl"
    if not summary_path.is_file() or not transcripts_path.is_file():
        raise ValueError(f"{run_dir} must contain summary.json and transcripts.jsonl")
    summary = json.loads(summary_path.read_text())
    rows = _load_jsonl(transcripts_path)
    run = summary.get("run") or {}
    if run.get("endogenous_abandonment"):
        raise ValueError("source run already has endogenous abandonment enabled")
    if run.get("policy") != "branching_atr":
        raise ValueError("source run must use the branching_atr policy")
    expected_ids = run.get("persona_ids") or []
    actual_ids = [row.get("persona_id") for row in rows]
    if expected_ids and actual_ids != expected_ids:
        raise ValueError("source transcript order does not match summary persona_ids")
    if len(actual_ids) != len(set(actual_ids)):
        raise ValueError("source transcripts contain duplicate persona IDs")
    for row in rows:
        if row.get("outcome") == "PROTOCOL_ERROR":
            raise ValueError(f"source persona {row.get('persona_id')} has a protocol error")
        question_count = len(row.get("question_ids") or [])
        if question_count != int(row.get("numquestions") or -1):
            raise ValueError(
                f"source persona {row.get('persona_id')} does not contain every question"
            )
        if len(row.get("checkpoints") or []) != question_count + 1:
            raise ValueError(
                f"source persona {row.get('persona_id')} has incomplete checkpoints"
            )
    return summary, rows


def _load_personas(category: str) -> dict[str, dict[str, Any]]:
    """Load the private persona records used by the production buyer."""

    path = REPO_ROOT / "data" / "categories" / category / "personas.json"
    payload = json.loads(path.read_text())
    rows = list(payload.get("personas", payload))
    personas = {str(row.get("id")): row for row in rows}
    if len(personas) != len(rows) or "None" in personas:
        raise ValueError(f"{path} contains missing or duplicate persona IDs")
    return personas


def _question_answer_pairs(row: dict[str, Any]) -> list[tuple[str, str]]:
    """Extract and validate the fixed question/answer sequence from a transcript."""

    dialogue = row.get("dialogue") or []
    question_count = len(row.get("question_ids") or [])
    if not dialogue or dialogue[0].get("role") != "user":
        raise ValueError(f"{row.get('persona_id')}: missing shopper opener")
    pairs: list[tuple[str, str]] = []
    for offset in range(question_count):
        question_index = 2 * offset + 1
        answer_index = question_index + 1
        if answer_index >= len(dialogue):
            raise ValueError(f"{row.get('persona_id')}: incomplete dialogue at question {offset + 1}")
        question = dialogue[question_index]
        answer = dialogue[answer_index]
        if question.get("role") != "assistant" or answer.get("role") != "user":
            raise ValueError(f"{row.get('persona_id')}: unexpected dialogue roles")
        if str(answer.get("content") or "").startswith(ABANDON_SENTINEL):
            raise ValueError(f"{row.get('persona_id')}: source dialogue already abandons")
        pairs.append((str(question.get("content") or ""), str(answer.get("content") or "")))
    return pairs


def _thread_client() -> Any:
    """Reuse one OpenAI client per worker thread."""

    if not hasattr(_thread_local, "client"):
        _thread_local.client = make_client()
    return _thread_local.client


def _fallback_decision(exc: Exception) -> dict[str, Any]:
    """Match live simulation behavior when an abandonment call cannot be scored."""

    return {
        "evaluation_status": "fallback_continue",
        "action": "CONTINUE",
        "reason": (
            "Defaulted to CONTINUE because the abandonment decision "
            "could not be evaluated."
        ),
        "error_code": getattr(exc, "code", None) or type(exc).__name__.upper(),
        "error": f"{type(exc).__name__}: {exc}",
    }


def _censor_at_abandonment(
    source: dict[str, Any],
    question_number: int,
    reason: str,
) -> dict[str, Any]:
    """Convert a complete control row into the row observable at an exit."""

    derived = copy.deepcopy(source)
    derived.update(
        {
            "outcome": "ABANDONED",
            "turns_used": question_number,
            "asks": question_number,
            "purchased_asin": None,
            "wtp": None,
            "actual_price": None,
            "revenue": 0.0,
            "consumer_surplus": 0.0,
            "decision": None,
            "recommendation_result": None,
            "abandonment_reason": reason,
            "error_code": None,
            "error": None,
        }
    )
    visible_dialogue = derived["dialogue"][: 2 * question_number]
    visible_dialogue.append(
        {"role": "user", "content": f"{ABANDON_SENTINEL} {reason}"}
    )
    derived["dialogue"] = visible_dialogue
    derived["question_ids"] = derived["question_ids"][:question_number]
    derived["question_topics"] = derived["question_topics"][:question_number]
    derived["tool_calls_per_turn"] = derived["tool_calls_per_turn"][:question_number]
    # In branching ATR, checkpoint d is evaluated before question d + 1.
    derived["checkpoints"] = derived["checkpoints"][:question_number]
    return derived


def _replay_one(
    source: dict[str, Any],
    persona: dict[str, Any],
    model: str,
    *,
    strict_errors: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate one persona sequentially and stop immediately after ABANDON."""

    category = str(source["category"])
    agent = BuyerAgent(
        persona=persona,
        category=category,
        model=model,
        endogenous_abandonment=True,
        client=_thread_client(),
    )
    engagement_decisions: list[dict[str, Any]] = []
    abandoned_at: int | None = None
    abandonment_reason: str | None = None
    pairs = _question_answer_pairs(source)
    question_ids = source["question_ids"]

    for index, (question, stored_answer) in enumerate(pairs, start=1):
        record: dict[str, Any] = {
            "question_number": index,
            "question_id": question_ids[index - 1],
            "question": question,
        }
        try:
            decision = agent.decide_abandonment(question)
            record.update({"evaluation_status": "scored", **decision})
        except Exception as exc:
            if strict_errors:
                raise
            record.update(_fallback_decision(exc))
        engagement_decisions.append(record)
        if record["action"] == "ABANDON":
            abandoned_at = index
            abandonment_reason = str(record["reason"])
            break
        # BuyerAgent normally appends these in respond(); replay uses the exact
        # stored answer and therefore must update history without another API call.
        agent.history.extend(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": stored_answer},
            ]
        )

    if abandoned_at is not None and abandonment_reason is not None:
        derived = _censor_at_abandonment(source, abandoned_at, abandonment_reason)
    else:
        derived = copy.deepcopy(source)
        derived["abandonment_reason"] = None

    source_usage = derived.pop("api_usage", {})
    overlay_usage = agent.tracker.summary()
    derived["engagement_decisions"] = engagement_decisions
    derived["api_usage"] = overlay_usage
    derived["counterfactual_abandonment"] = {
        "source_outcome": source.get("outcome"),
        "source_api_usage": source_usage,
        "abandoned_at_question": abandoned_at,
        "model": model,
        "abandonment_prompt_sha256": hashlib.sha256(
            ABANDONMENT_PROMPT.encode()
        ).hexdigest(),
    }
    overlay = {
        "persona_id": source["persona_id"],
        "source_outcome": source.get("outcome"),
        "counterfactual_outcome": derived["outcome"],
        "abandoned_at_question": abandoned_at,
        "abandonment_reason": abandonment_reason,
        "engagement_decisions": engagement_decisions,
        "api_usage": overlay_usage,
    }
    return derived, overlay


def _aggregate_usage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate only the new abandonment-evaluation API calls."""

    calls = [call for row in rows for call in (row.get("api_usage") or {}).get("calls", [])]
    known_costs = [
        float(call["estimated_cost_usd"])
        for call in calls
        if isinstance(call.get("estimated_cost_usd"), (int, float))
    ]
    unpriced = len(calls) - len(known_costs)
    return {
        "requests": len(calls),
        "input_tokens": sum(int(call.get("input_tokens") or 0) for call in calls),
        "cached_input_tokens": sum(
            int(call.get("cached_input_tokens") or 0) for call in calls
        ),
        "output_tokens": sum(int(call.get("output_tokens") or 0) for call in calls),
        "latency_s": round(sum(float(call.get("latency_s") or 0) for call in calls), 3),
        "estimated_cost_usd": round(sum(known_costs), 8) if not unpriced else None,
        "unpriced_requests": unpriced,
    }


def _summarize(derived_rows: list[dict[str, Any]], overlays: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize exits and the retained terminal control outcomes."""

    n = len(derived_rows)
    outcomes = Counter(str(row.get("outcome")) for row in derived_rows)
    exits = Counter(
        int(row["abandoned_at_question"])
        for row in overlays
        if row.get("abandoned_at_question") is not None
    )
    revenues = [float(row.get("revenue") or 0) for row in derived_rows]
    engagement = [
        decision
        for row in overlays
        for decision in row.get("engagement_decisions", [])
    ]
    return {
        "n": n,
        "outcomes": dict(sorted(outcomes.items())),
        "abandonment_rate": round(outcomes["ABANDONED"] / n, 4) if n else None,
        "purchase_rate": round(outcomes["PURCHASE"] / n, 4) if n else None,
        "abandonments_by_question": {
            str(question): count for question, count in sorted(exits.items())
        },
        "mean_questions": (
            round(mean(float(row.get("asks") or 0) for row in derived_rows), 4)
            if derived_rows
            else None
        ),
        "total_revenue": round(sum(revenues), 2),
        "mean_revenue_per_persona": round(sum(revenues) / n, 4) if n else None,
        "engagement_decision_count": len(engagement),
        "engagement_decision_fallback_count": sum(
            decision.get("evaluation_status") == "fallback_continue"
            for decision in engagement
        ),
        "overlay_api_usage": _aggregate_usage(overlays),
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write JSONL in deterministic persona order."""

    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source_run",
        help="control run directory or exact run title",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument(
        "--buyer-model",
        help="override the source run's buyer model",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate inputs and print the plan without making API calls",
    )
    args = parser.parse_args()

    load_env()
    try:
        source_dir = _resolve_source_run(args.source_run)
        source_summary, source_rows = _load_source_run(source_dir)
        run = source_summary.get("run") or {}
        category = str(run.get("category") or source_rows[0].get("category") or "")
        personas_path = REPO_ROOT / "data" / "categories" / category / "personas.json"
        source_personas_sha256 = str(run.get("personas_sha256") or "")
        if source_personas_sha256 and _sha256(personas_path) != source_personas_sha256:
            raise ValueError(
                "current personas.json does not match the source run's personas_sha256"
            )
        personas = _load_personas(category)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    if not 1 <= args.parallel <= MAX_PARALLEL:
        parser.error(f"--parallel must be between 1 and {MAX_PARALLEL}")
    missing = [row["persona_id"] for row in source_rows if row["persona_id"] not in personas]
    if missing:
        parser.error(f"persona records not found: {', '.join(missing)}")
    model = args.buyer_model or str(run.get("buyer_model") or DEFAULT_MODEL)
    expected_requests = sum(len(row["question_ids"]) for row in source_rows)
    plan = {
        "source_run": str(source_dir),
        "category": category,
        "persona_count": len(source_rows),
        "model": model,
        "parallel": args.parallel,
        "maximum_api_requests": expected_requests,
        "source_personas_sha256": source_personas_sha256 or None,
        "questions_replayed_from_source_transcripts": True,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        parser.error("OPENAI_API_KEY is required unless --dry-run is used")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (
        REPO_ROOT
        / "results"
        / "local"
        / f"{source_dir.name}_counterfactual_abandonment_{timestamp}"
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error(f"output directory is not empty: {output_dir}")

    started = time.time()
    derived_by_index: dict[int, dict[str, Any]] = {}
    overlay_by_index: dict[int, dict[str, Any]] = {}
    # Score the first persona strictly before launching the worker pool. This
    # catches missing connectivity, credentials, and incompatible API responses
    # instead of silently turning an entire replay into fallback CONTINUEs.
    try:
        first_derived, first_overlay = _replay_one(
            source_rows[0],
            personas[source_rows[0]["persona_id"]],
            model,
            strict_errors=True,
        )
    except Exception as exc:
        parser.error(f"abandonment API preflight failed: {type(exc).__name__}: {exc}")
    derived_by_index[0] = first_derived
    overlay_by_index[0] = first_overlay
    first_marker = first_overlay["abandoned_at_question"] or "continue"
    print(f"{first_overlay['persona_id']}: {first_marker}", flush=True)

    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {
            pool.submit(
                _replay_one,
                row,
                personas[row["persona_id"]],
                model,
            ): index
            for index, row in enumerate(source_rows[1:], start=1)
        }
        for future in as_completed(futures):
            index = futures[future]
            derived, overlay = future.result()
            derived_by_index[index] = derived
            overlay_by_index[index] = overlay
            marker = overlay["abandoned_at_question"] or "continue"
            print(f"{overlay['persona_id']}: {marker}", flush=True)

    derived_rows = [derived_by_index[index] for index in range(len(source_rows))]
    overlays = [overlay_by_index[index] for index in range(len(source_rows))]
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "transcripts.jsonl", derived_rows)
    _write_jsonl(output_dir / "abandonment_overlay.jsonl", overlays)

    source_summary_path = source_dir / "summary.json"
    source_transcripts_path = source_dir / "transcripts.jsonl"
    summary = {
        "schema_version": "crs-counterfactual-abandonment-v1",
        "run": {
            **plan,
            "source_summary_sha256": _sha256(source_summary_path),
            "source_transcripts_sha256": _sha256(source_transcripts_path),
            "abandonment_prompt": ABANDONMENT_PROMPT,
            "abandonment_prompt_sha256": hashlib.sha256(
                ABANDONMENT_PROMPT.encode()
            ).hexdigest(),
            "wall_clock_s": round(time.time() - started, 2),
            "derived_transcripts": True,
        },
        "metrics": _summarize(derived_rows, overlays),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["metrics"], indent=2))
    print(f"wrote {output_dir / 'transcripts.jsonl'}")
    print(f"wrote {output_dir / 'abandonment_overlay.jsonl'}")
    print(f"wrote {output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
