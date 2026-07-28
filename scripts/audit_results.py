"""Audit saved simulation transcripts without calling a model.

Older result directories predate the controlled protocol. This script treats
them as diagnostic evidence and reports outcome/accounting invariants rather
than pooling them into a policy conclusion.

Usage:
    uv run python scripts/audit_results.py
    uv run python scripts/audit_results.py --results-root results/local
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent


def _empty_counts() -> Counter[str]:
    return Counter(
        rows=0,
        malformed_json=0,
        purchases=0,
        protocol_errors=0,
        max_turn_failures=0,
        asks_exceed_turns=0,
        purchase_not_in_displayed_cards=0,
        purchase_missing_price=0,
        negative_surplus_purchase=0,
        missing_summary=0,
        summary_n_mismatch=0,
    )


def _audit_row(row: dict[str, Any], counts: Counter[str]) -> None:
    counts["rows"] += 1
    outcome = row.get("outcome")
    error = str(row.get("error") or "")
    if outcome in {"ERROR", "PROTOCOL_ERROR"}:
        counts["protocol_errors"] += 1
    if "max_turns_reached_without_final_recommendation" in error:
        counts["max_turn_failures"] += 1

    asks = row.get("asks")
    turns = row.get("turns_used")
    if isinstance(asks, int) and isinstance(turns, int) and asks > turns:
        counts["asks_exceed_turns"] += 1

    if outcome != "PURCHASE":
        return
    counts["purchases"] += 1
    cards = row.get("crs_recommendations") or []
    displayed_asins = {card.get("asin") for card in cards if isinstance(card, dict)}
    purchased_asin = row.get("purchased_asin")
    if not purchased_asin or purchased_asin not in displayed_asins:
        counts["purchase_not_in_displayed_cards"] += 1
    if row.get("actual_price") is None:
        counts["purchase_missing_price"] += 1
    surplus = row.get("consumer_surplus")
    if isinstance(surplus, (int, float)) and surplus < 0:
        counts["negative_surplus_purchase"] += 1


def audit(results_root: Path) -> dict[str, Any]:
    totals = _empty_counts()
    per_run: dict[str, dict[str, int]] = {}
    transcript_paths = sorted(results_root.rglob("transcripts.jsonl"))
    for path in transcript_paths:
        counts = _empty_counts()
        with path.open() as handle:
            for raw in handle:
                if not raw.strip():
                    continue
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError:
                    counts["malformed_json"] += 1
                    continue
                if isinstance(row, dict):
                    _audit_row(row, counts)
        for key, value in counts.items():
            totals[key] += value
        summary_path = path.parent / "summary.json"
        if not summary_path.exists():
            counts["missing_summary"] += 1
            totals["missing_summary"] += 1
        else:
            try:
                summary = json.loads(summary_path.read_text())
                if isinstance(summary, dict) and isinstance(summary.get("n"), int) and summary["n"] != counts["rows"]:
                    counts["summary_n_mismatch"] += 1
                    totals["summary_n_mismatch"] += 1
            except json.JSONDecodeError:
                counts["summary_n_mismatch"] += 1
                totals["summary_n_mismatch"] += 1
        per_run[str(path.parent.relative_to(results_root))] = dict(counts)
    return {
        "results_root": str(results_root),
        "runs_with_transcripts": len(transcript_paths),
        "totals": dict(totals),
        "per_run": per_run,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-root", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    report = audit(args.results_root)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.json_out:
        args.json_out.write_text(rendered + "\n")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
