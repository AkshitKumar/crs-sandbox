"""Simple terminal REPL to talk to the CRS agent.

Useful for debugging before the Streamlit chat UI exists.

Usage:
    python scripts/repl.py
    python scripts/repl.py --start-category laptop

Commands:
    /reset     start a new conversation
    /trace     show the tool calls from the last turn
    /bus       show the current Candidate Bus contents and notes
    /quit      exit
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from sandbox.env import load_env
from sandbox.agents.langgraph_crs import CRSAgentSession


def _print_tool_trace(calls: list[dict]) -> None:
    if not calls:
        print("  (no tool calls)")
        return
    for c in calls:
        args = json.dumps(c.get("args", {}), default=str)
        if len(args) > 90:
            args = args[:87] + "..."
        print(f"  ▸ {c['tool']}({args}) → {c.get('result', '')[:100]}")


def main() -> int:
    load_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-category", default=None)
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--reasoning", default="medium")
    args = parser.parse_args()

    session = CRSAgentSession(model=args.model, reasoning_effort=args.reasoning)
    if args.start_category:
        session.category = args.start_category

    print("=" * 80)
    print("crs-sandbox REPL  (commands: /reset /trace /bus /quit)")
    print("=" * 80)
    last_calls: list[dict] = []

    while True:
        try:
            user = input("\nYOU: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not user:
            continue
        if user == "/quit":
            return 0
        if user == "/reset":
            session.reset()
            print("[reset]")
            continue
        if user == "/trace":
            print("Last turn's tool calls:")
            _print_tool_trace(last_calls)
            continue
        if user == "/bus":
            if session.bus:
                print(f"bus size: {session.bus.size()}, top 5 asins: {session.bus.top(5)}")
                print("notes:")
                for n in session.bus.notes:
                    print(f"  • {n}")
            else:
                print("(no bus yet)")
            continue

        out = session.chat(user)
        last_calls = out["tool_calls_this_turn"]
        print(f"\nAGENT: {out['reply']}")
        if last_calls:
            print(f"\n  [tool calls: {len(last_calls)}]")
            _print_tool_trace(last_calls[:6])
            if len(last_calls) > 6:
                print(f"  ... +{len(last_calls) - 6} more")
        if out["recommendations"]:
            print(f"\n  [recommendation set finalized: {len(out['recommendations'])} products]")
            for r in out["recommendations"]:
                print(f"    #{r['rank']} ${r['price']} ★{r['avg_rating']} — {r['title'][:75]}")


if __name__ == "__main__":
    sys.exit(main())
