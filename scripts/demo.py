"""Streamlit demo for live simulations, timed replay, and run comparison."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import streamlit as st


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from sandbox.env import load_env  # noqa: E402
from sandbox.catalog import get_products  # noqa: E402
from sandbox.simulation import Policy, Simulation  # noqa: E402


load_env()
st.set_page_config(page_title="Conversational recommender demo", layout="wide")
st.title("Conversational recommender demo")


def _runs() -> list[Path]:
    root = REPO_ROOT / "results"
    return sorted(
        (path.parent for path in root.rglob("transcripts.jsonl")) if root.exists() else [],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def _records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _personas(category: str) -> list[dict[str, Any]]:
    path = REPO_ROOT / "data" / "categories" / category / "personas.json"
    payload = json.loads(path.read_text())
    return list(payload.get("personas", payload))


def _money(value: Any) -> str:
    return "—" if value is None else f"${float(value):,.2f}"


def _render_tools(calls: list[dict[str, Any]]) -> None:
    if not calls:
        return
    with st.expander(f"{len(calls)} tool call(s)"):
        for call in calls:
            st.code(
                f"{call.get('tool')}({json.dumps(call.get('args') or {})})\n"
                f"→ {str(call.get('result') or '')[:600]}"
            )


def _render_recommendations(recommendations: list[dict[str, Any]]) -> None:
    if not recommendations:
        return
    st.subheader("Recommendations")
    for product in recommendations:
        with st.container(border=True):
            st.markdown(
                f"**{product.get('rank', '?')}. {product.get('title', '')}**  \n"
                f"{_money(product.get('price'))} · {product.get('avg_rating', '—')} stars  \n"
                f"{product.get('recommendation_explanation', '')}"
            )


def _hydrate_recommendations(
    category: str, recommendations: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Join compact transcript selections to the matching local catalog."""
    asins = [str(product.get("asin") or "") for product in recommendations]
    catalog = {product.get("asin"): product for product in get_products(category, asins)}
    return [
        {
            "rank": index,
            **catalog.get(product.get("asin"), {}),
            **product,
        }
        for index, product in enumerate(recommendations, start=1)
    ]


def _render_decision(record: dict[str, Any]) -> None:
    decision = record.get("decision") or record.get("buyer_decision_raw") or {}
    outcome = record.get("outcome") or decision.get("decision")
    st.subheader("Buyer decision")
    st.write(
        f"**{outcome}** · WTP {_money(record.get('wtp'))} · "
        f"revenue {_money(record.get('revenue', record.get('actual_price') or 0))} · "
        f"consumer surplus {_money(record.get('consumer_surplus'))}"
    )
    if decision.get("reasoning"):
        st.info(decision["reasoning"])


def _replay(record: dict[str, Any], delay: float) -> None:
    dialogue = record.get("dialogue") or []
    tool_turns = record.get("tool_calls_per_turn") or record.get("crs_tool_calls_per_turn") or []
    assistant_index = 0
    for message in dialogue:
        role = "assistant" if message.get("role") == "assistant" else "user"
        with st.chat_message(role):
            st.markdown(message.get("content") or "")
            if role == "assistant":
                calls = tool_turns[assistant_index] if assistant_index < len(tool_turns) else []
                _render_tools(calls)
                assistant_index += 1
        if delay:
            time.sleep(delay)
    recommendation_result = record.get("recommendation_result") or {}
    recommendations = (
        recommendation_result.get("recommendations")
        or record.get("recommendations")
        or record.get("crs_recommendations")
        or []
    )
    _render_recommendations(
        _hydrate_recommendations(str(record.get("category") or ""), recommendations)
    )
    _render_decision(record)
    checkpoints = record.get("checkpoints") or record.get("recommendation_checkpoints") or []
    if checkpoints:
        with st.expander("Branching checkpoints"):
            for checkpoint in checkpoints:
                depth = checkpoint.get("depth", checkpoint.get("index"))
                outcome = checkpoint.get("outcome", checkpoint.get("counterfactual_outcome"))
                st.write(
                    f"depth {depth}: {checkpoint.get('evaluation_status')} · {outcome} · "
                    f"revenue {_money(checkpoint.get('revenue', checkpoint.get('counterfactual_actual_price') or 0))}"
                )


def _render_live_event(event: dict[str, Any]) -> None:
    event_type = event.get("type")
    if event_type in {"buyer_message", "recommender_message"}:
        role = "user" if event_type == "buyer_message" else "assistant"
        with st.chat_message(role):
            st.markdown(event.get("content") or "")
            _render_tools(event.get("tool_calls") or [])
    elif event_type == "checkpoint":
        checkpoint = event["checkpoint"]
        st.caption(
            f"hidden depth {checkpoint.get('depth')}: "
            f"{checkpoint.get('evaluation_status')} · {checkpoint.get('outcome', '')}"
        )
    elif event_type == "recommendations":
        _render_recommendations(event.get("items") or [])
    elif event_type == "decision":
        _render_decision(event)
    elif event_type == "abandoned":
        st.warning(f"Buyer abandoned: {event.get('reason') or 'no reason given'}")
    elif event_type == "error":
        st.error(f"{event.get('error_code')}: {event.get('error')}")


mode = st.sidebar.radio("Mode", ["Live simulation", "Replay result", "Compare runs"])

if mode == "Live simulation":
    category = st.sidebar.selectbox("Category", ["laptop", "air_purifier"])
    personas = _personas(category)
    persona_id = st.sidebar.selectbox("Persona", [persona["id"] for persona in personas])
    persona = next(persona for persona in personas if persona["id"] == persona_id)
    policy_name = st.sidebar.selectbox(
        "Policy", ["adaptive", "rec", "single_atr", "branching_atr"]
    )
    numquestions = 0
    if policy_name in {"single_atr", "branching_atr"}:
        numquestions = st.sidebar.number_input("Questions", 0, 20, 4)
    endogenous = st.sidebar.checkbox("Endogenous abandonment", value=False)
    if st.sidebar.button("Run simulation", type="primary"):
        simulation = Simulation(
            persona=persona,
            category=category,
            policy=Policy.make(policy_name, int(numquestions)),
            endogenous_abandonment=endogenous,
        )
        for event in simulation.run_iter():
            _render_live_event(event)

elif mode == "Replay result":
    runs = _runs()
    if not runs:
        st.info("No saved transcripts found.")
        st.stop()
    run = st.sidebar.selectbox("Run", runs, format_func=lambda path: str(path.relative_to(REPO_ROOT)))
    records = _records(run / "transcripts.jsonl")
    persona_id = st.sidebar.selectbox("Persona", [record.get("persona_id") for record in records])
    delay = st.sidebar.slider("Seconds between messages", 0.0, 2.0, 0.35, 0.05)
    record = next(record for record in records if record.get("persona_id") == persona_id)
    if st.sidebar.button("Play replay", type="primary"):
        _replay(record, delay)

else:
    runs = [run for run in _runs() if (run / "summary.json").is_file()]
    selected = st.sidebar.multiselect(
        "Runs",
        runs,
        default=runs[: min(3, len(runs))],
        format_func=lambda path: str(path.relative_to(REPO_ROOT)),
    )
    rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    for run in selected:
        summary = json.loads((run / "summary.json").read_text())
        metrics = summary.get("metrics", summary)
        rows.append(
            {
                "run": run.name,
                "purchase_rate": metrics.get("purchase_rate"),
                "total_revenue": metrics.get("total_revenue"),
                "mean_surplus": metrics.get(
                    "mean_consumer_surplus_all_personas",
                    metrics.get("mean_consumer_surplus_among_purchased"),
                ),
                "protocol_error_rate": metrics.get("protocol_error_rate"),
                "estimated_cost_usd": (metrics.get("api_usage") or {}).get("estimated_cost_usd"),
            }
        )
        for depth, values in (metrics.get("checkpoint_metrics") or {}).items():
            checkpoint_rows.append(
                {"run": run.name, "depth": int(depth), "purchase_rate": values.get("purchase_rate", values.get("current_purchase_rate"))}
            )
    st.dataframe(rows, use_container_width=True)
    if checkpoint_rows:
        st.line_chart(checkpoint_rows, x="depth", y="purchase_rate", color="run")
