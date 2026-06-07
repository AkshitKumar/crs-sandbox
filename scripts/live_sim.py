"""rufus-femto · live buyer × CRS simulator.

Watch a persona-conditioned buyer LLM talk to rufus-femto in real time.

Run with:
    streamlit run scripts/live_sim.py --server.port=8503

Choose a category and persona, hit ▶ Run, and watch the dialogue, tool calls,
recommendations, and final purchase decision render as they're generated.
Cost: ~$0.15–0.30 per simulation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from sandbox.env import load_env  # noqa: E402

load_env()

from sandbox.orchestrator.sim_conversation import SimConversation  # noqa: E402
from sandbox.tools.feasibility_tool import list_available_categories  # noqa: E402

st.set_page_config(page_title="rufus-femto · live sim", layout="wide", page_icon="🎬")

# ---------------- CSS polish ----------------

st.markdown(
    """
    <style>
      .vh-header { display:flex; align-items:baseline; gap:0.6em; margin-bottom:0.5em; }
      .vh-header h1 { margin:0; font-weight:700; letter-spacing:-0.5px; }
      .vh-header .tag { color:#888; font-size:0.9em; }
      .outcome-pill {
          display:inline-block; padding:0.15em 0.7em; border-radius:999px;
          font-size:0.85em; font-weight:600;
      }
      .pill-purchase   { background:#d4edda; color:#155724; }
      .pill-nopurchase { background:#f8d7da; color:#721c24; }
      .pill-abandoned  { background:#fff3cd; color:#856404; }
      .pill-error      { background:#e2e3e5; color:#383d41; }
      .rank-badge {
          display:inline-block; background:#0f6efd; color:white;
          border-radius:999px; padding:0.05em 0.6em;
          font-size:0.8em; font-weight:600; margin-right:0.4em;
      }
      .price-pill { background:#fff7e6; color:#8a5800;
          border-radius:6px; padding:0.15em 0.5em;
          font-weight:600; display:inline-block; }
      .rating-line { color:#b58900; }
      .brand-tag { background:#f1f3f5; color:#444;
          padding:0.1em 0.55em; border-radius:4px;
          font-size:0.85em; display:inline-block; }
      .tool-call { font-family:'SF Mono',Menlo,monospace; font-size:0.85em; padding:0.15em 0; }
      .tool-name { color:#0f6efd; font-weight:600; }
      .tool-args { color:#666; }
      .tool-result { color:#444; font-size:0.9em; padding-left:1em; }
      .ground-truth-card {
          background:#fffae6; border:1px solid #f0d878; border-radius:8px;
          padding:0.8em 1em; margin:0.5em 0;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- helpers ----------------


def _load_personas(category: str) -> list[dict]:
    path = REPO_ROOT / "data" / "categories" / category / "personas.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    return data.get("personas", data)


def _fmt_price(p) -> str:
    if p is None:
        return "—"
    try:
        return f"${float(p):,.0f}"
    except (TypeError, ValueError):
        return f"${p}"


def _fmt_rating(rating, n_reviews) -> str:
    if rating is None:
        return "no ratings yet"
    full = int(round(float(rating)))
    stars = "★" * full + "☆" * (5 - full)
    label = f"{rating:.1f}"
    rev_part = f" · {n_reviews:,} review{'s' if (n_reviews or 0) != 1 else ''}" if n_reviews else ""
    return f"<span class='rating-line'>{stars}</span> {label}{rev_part}"


def _outcome_pill(outcome: str) -> str:
    label = outcome.replace("_", " ")
    cls = {
        "PURCHASE": "pill-purchase",
        "NO_PURCHASE": "pill-nopurchase",
        "ABANDONED": "pill-abandoned",
        "ERROR": "pill-error",
    }.get(outcome, "pill-error")
    return f"<span class='outcome-pill {cls}'>{label}</span>"


# ---------------- session state ----------------


if "playing" not in st.session_state:
    st.session_state.playing = False
if "current_outcome" not in st.session_state:
    st.session_state.current_outcome = None


# ---------------- header ----------------

st.markdown(
    "<div class='vh-header'><h1>🎬 rufus-femto · live sim</h1>"
    "<span class='tag'>watch a buyer LLM talk to the CRS in real time</span></div>",
    unsafe_allow_html=True,
)


# ---------------- sidebar: pick + Play ----------------

with st.sidebar:
    st.markdown("### Pick a persona")

    available = list_available_categories()
    if not available:
        st.error("No indexed categories. Run `python scripts/build_index.py <cat>` first.")
        st.stop()
    category = st.selectbox("Category", options=available, index=0)

    personas = _load_personas(category)
    if not personas:
        st.warning(f"No personas at data/categories/{category}/personas.json")
        st.stop()

    persona_ids = [p["id"] for p in personas]
    persona_id = st.selectbox(
        "Persona", options=persona_ids,
        format_func=lambda pid: pid,
    )
    persona = next(p for p in personas if p["id"] == persona_id)

    st.divider()
    st.markdown("### Simulation settings")
    max_turns = st.slider("Max turns", min_value=4, max_value=24, value=16)
    eta = st.slider(
        "Per-turn abandonment hazard (η)", min_value=0.0, max_value=0.5,
        value=0.0, step=0.05,
        help="Probability the buyer gives up before the next turn. Set 0 for noiseless eval.",
    )
    seed = st.number_input("Seed", min_value=0, value=0, step=1)

    st.divider()
    play = st.button("▶ Run simulation", type="primary", use_container_width=True)
    if st.button("Clear", use_container_width=True):
        st.session_state.current_outcome = None
        st.rerun()


# ---------------- persona reveal ----------------

c_persona, c_metrics = st.columns([3, 1])
with c_persona:
    st.markdown("### Persona (hidden from CRS)")
    st.caption(f"`{persona['id']}` · category `{category}`")
    st.markdown(
        f"<div class='ground-truth-card'><b>Background</b><br>{persona.get('background', '')}</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='ground-truth-card'><b>Ground-truth need</b> (only the buyer sees this; CRS must elicit)<br>"
        f"{persona.get('ground_truth_need', '')}</div>",
        unsafe_allow_html=True,
    )

with c_metrics:
    st.markdown("### Outcome")
    placeholder_outcome = st.empty()
    if st.session_state.current_outcome:
        o = st.session_state.current_outcome
        with placeholder_outcome.container():
            st.markdown(_outcome_pill(o.outcome), unsafe_allow_html=True)
            st.metric("turns", o.turns_used)
            st.metric("asks", o.asks)
            if o.outcome == "PURCHASE":
                st.metric("WTP", _fmt_price(o.wtp))
                st.metric("paid", _fmt_price(o.actual_price))
                if o.consumer_surplus is not None:
                    st.metric("surplus", f"${o.consumer_surplus:,.0f}")
    else:
        placeholder_outcome.info("Hit ▶ Run to watch the simulation.")

st.divider()

# ---------------- transcript area ----------------

st.markdown("### Live transcript")
transcript_container = st.container()
recommendation_container = st.container()
decision_container = st.container()


# ---------------- run the simulation ----------------

if play:
    st.session_state.playing = True
    st.session_state.current_outcome = None

    sim = SimConversation(
        persona=persona,
        category=category,
        max_turns=int(max_turns),
        eta=float(eta),
        seed=int(seed),
    )

    final_outcome = None
    with transcript_container:
        for event in sim.run_iter():
            etype = event.get("type")
            if etype == "buyer_message":
                with st.chat_message("user", avatar="🧍"):
                    st.markdown(event["content"])
            elif etype == "crs_message":
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(event["content"])
                    calls = event.get("tool_calls") or []
                    if calls:
                        with st.expander(f"🔧 {len(calls)} tool call(s) · bus={event.get('bus_size')}"):
                            for c in calls:
                                args = json.dumps(c.get("args", {}), default=str)
                                if len(args) > 110:
                                    args = args[:107] + "…"
                                st.markdown(
                                    f"<div class='tool-call'>"
                                    f"<span class='tool-name'>{c['tool']}</span>"
                                    f"<span class='tool-args'>({args})</span></div>"
                                    f"<div class='tool-result'>↳ {(c.get('result') or '')[:200]}</div>",
                                    unsafe_allow_html=True,
                                )
            elif etype == "abandoned":
                st.warning(f"Buyer abandoned at turn {event['turn']}.")
            elif etype == "error":
                st.error(f"{event.get('where', 'error')}: {event.get('error', '')}")
            elif etype == "recommendations":
                with recommendation_container:
                    st.markdown("### Final recommendation set")
                    for rec in event["items"]:
                        with st.container(border=True):
                            title = rec.get("title") or "—"
                            rank = rec.get("rank", "?")
                            sponsored = (
                                " <span style='background:#fff3cd;color:#856404;padding:0.05em 0.5em;border-radius:4px;font-size:0.75em'>SPONSORED</span>"
                                if rec.get("sponsored") else ""
                            )
                            st.markdown(
                                f"<span class='rank-badge'>#{rank}</span><strong>{title[:130]}</strong>{sponsored}",
                                unsafe_allow_html=True,
                            )
                            cp, cr, cb = st.columns([1, 2, 1])
                            cp.markdown(
                                f"<span class='price-pill'>{_fmt_price(rec.get('price'))}</span>",
                                unsafe_allow_html=True,
                            )
                            cr.markdown(
                                _fmt_rating(rec.get("avg_rating"), rec.get("num_reviews")),
                                unsafe_allow_html=True,
                            )
                            brand = rec.get("brand")
                            if brand:
                                cb.markdown(
                                    f"<span class='brand-tag'>{brand}</span>",
                                    unsafe_allow_html=True,
                                )
                            bullets = rec.get("bullets") or []
                            if bullets:
                                with st.expander("Key bullets"):
                                    for b in bullets[:5]:
                                        st.markdown(f"- {b}")
            elif etype == "decision":
                with decision_container:
                    st.markdown("### Buyer's decision")
                    d = event["decision"]
                    label = event["outcome_label"]
                    if label == "PURCHASE":
                        n = d.get("product_number")
                        asin = event.get("purchased_asin")
                        wtp = event.get("wtp")
                        price = event.get("actual_price")
                        surplus = event.get("consumer_surplus")
                        st.success(
                            f"PURCHASED product #{n} (`{asin}`) · WTP {_fmt_price(wtp)} · paid {_fmt_price(price)}"
                            + (f" · surplus ${surplus:,.0f}" if surplus is not None else "")
                        )
                    else:
                        st.info("No purchase")
                    reasoning = d.get("reasoning", "")
                    if reasoning:
                        st.markdown(f"> {reasoning}")
            elif etype == "outcome":
                final_outcome = event["outcome"]

    if final_outcome is not None:
        st.session_state.current_outcome = final_outcome
        # Refresh sidebar/metrics with final values.
        with c_metrics:
            placeholder_outcome.empty()
            with placeholder_outcome.container():
                st.markdown(_outcome_pill(final_outcome.outcome), unsafe_allow_html=True)
                st.metric("turns", final_outcome.turns_used)
                st.metric("asks", final_outcome.asks)
                if final_outcome.outcome == "PURCHASE":
                    st.metric("WTP", _fmt_price(final_outcome.wtp))
                    st.metric("paid", _fmt_price(final_outcome.actual_price))
                    if final_outcome.consumer_surplus is not None:
                        st.metric("surplus", f"${final_outcome.consumer_surplus:,.0f}")
    st.session_state.playing = False
