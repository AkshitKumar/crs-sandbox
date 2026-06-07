"""rufus-femto — Streamlit chat UI for the agentic CRS.

Two-pane layout:
    LEFT  — customer-facing chat with formatted recommendation cards.
    RIGHT — agent thinking: per-turn tool calls + live Candidate Bus state.

Run with:
    streamlit run scripts/chat.py
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

from sandbox.agents.langgraph_crs import CRSAgentSession  # noqa: E402

st.set_page_config(page_title="rufus-femto", layout="wide", page_icon="🛍️")

# ---------------------------------------------------------------------------
# Light CSS polish (Streamlit's defaults are workable but the cards benefit
# from a bit of breathing room and a clearer visual hierarchy)
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
      .rufus-header {
          display: flex; align-items: baseline; gap: 0.6em;
          margin-bottom: 0.2em;
      }
      .rufus-header h1 { margin: 0; font-weight: 700; letter-spacing: -0.5px; }
      .rufus-header .tag { color: #888; font-size: 0.9em; }
      .rank-badge {
          display: inline-block; background:#0f6efd; color:white;
          border-radius: 999px; padding: 0.05em 0.6em;
          font-size: 0.8em; font-weight: 600;
          margin-right: 0.4em; vertical-align: 1px;
      }
      .price-pill {
          background: #fff7e6; color: #8a5800;
          border-radius: 6px; padding: 0.15em 0.5em;
          font-weight: 600; font-size: 1.1em;
          display: inline-block;
      }
      .rating-line { color: #b58900; font-size: 1.05em; }
      .brand-tag {
          background: #f1f3f5; color: #444;
          padding: 0.1em 0.55em; border-radius: 4px;
          font-size: 0.85em; display: inline-block;
      }
      .sponsored-tag {
          background: #fff3cd; color: #856404;
          padding: 0.05em 0.5em; border-radius: 4px;
          font-size: 0.75em; display: inline-block;
          letter-spacing: 0.04em;
      }
      .tool-call {
          font-family: 'SF Mono', Menlo, monospace;
          font-size: 0.85em;
          padding: 0.2em 0;
      }
      .tool-name { color: #0f6efd; font-weight: 600; }
      .tool-args { color: #666; }
      .tool-result { color: #444; font-size: 0.95em; padding-left: 1em; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------


def _init_state() -> None:
    if "session" not in st.session_state:
        st.session_state.session = CRSAgentSession()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "trace" not in st.session_state:
        st.session_state.trace = []


def _reset() -> None:
    st.session_state.session = CRSAgentSession()
    st.session_state.messages = []
    st.session_state.trace = []


_init_state()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Header + sidebar
# ---------------------------------------------------------------------------

st.markdown(
    "<div class='rufus-header'><h1>🛍️ rufus-femto</h1>"
    "<span class='tag'>agentic shopping assistant · human-in-the-loop test mode</span></div>",
    unsafe_allow_html=True,
)
st.divider()


with st.sidebar:
    st.markdown("### Session")
    if st.button("🔄  Start over", use_container_width=True):
        _reset()
        st.rerun()
    st.divider()

    st.markdown("### Status")
    sess: CRSAgentSession = st.session_state.session
    st.metric("category", sess.category or "(not set)")
    st.metric("candidate bus", sess.bus.size() if sess.bus else 0)
    st.metric("clarifying asks", sess.asks_so_far)

    st.divider()
    st.markdown("### Try")
    st.caption(
        "- *I need a new laptop for college*\n"
        "- *Looking for a quiet air purifier for my bedroom*\n"
        "- *Recommend a gaming laptop under $1200*"
    )


# ---------------------------------------------------------------------------
# Layout: left = chat, right = agent trace
# ---------------------------------------------------------------------------


left, right = st.columns([3, 2], gap="large")


# ===================== LEFT: chat + recommendation cards =====================


with left:
    chat_box = st.container()
    with chat_box:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Recommendation cards after the latest assistant message.
        latest = st.session_state.trace[-1] if st.session_state.trace else None
        if latest and latest.get("recommendations"):
            st.markdown("### Recommendations")
            for rec in latest["recommendations"]:
                with st.container(border=True):
                    # Header row: rank badge + title + sponsored tag (if any)
                    title = rec.get("title") or "—"
                    rank = rec.get("rank", "?")
                    sponsored_html = (
                        " <span class='sponsored-tag'>SPONSORED</span>"
                        if rec.get("sponsored_in_search") else ""
                    )
                    st.markdown(
                        f"<span class='rank-badge'>#{rank}</span>"
                        f"<strong>{title[:130]}</strong>{sponsored_html}",
                        unsafe_allow_html=True,
                    )

                    # Stats row: price · rating · brand
                    c_price, c_rating, c_brand = st.columns([1, 2, 1])
                    with c_price:
                        st.markdown(
                            f"<span class='price-pill'>{_fmt_price(rec.get('price'))}</span>",
                            unsafe_allow_html=True,
                        )
                    with c_rating:
                        st.markdown(
                            _fmt_rating(rec.get("avg_rating"), rec.get("num_reviews")),
                            unsafe_allow_html=True,
                        )
                    with c_brand:
                        brand = rec.get("brand")
                        if brand:
                            st.markdown(
                                f"<span class='brand-tag'>{brand}</span>",
                                unsafe_allow_html=True,
                            )

                    # Description / bullets
                    bullets = rec.get("bullets") or []
                    if bullets:
                        with st.expander("Key features"):
                            for b in bullets[:6]:
                                st.markdown(f"- {b}")
                    desc = (rec.get("description") or "").strip()
                    if desc:
                        short_desc = desc[:280] + ("…" if len(desc) > 280 else "")
                        st.caption(short_desc)

                    # Footer: link out
                    url = rec.get("url")
                    if url:
                        st.markdown(
                            f"<a href='{url}' target='_blank' "
                            f"style='text-decoration:none; color:#0f6efd; font-weight:500;'>"
                            f"View on Amazon →</a>",
                            unsafe_allow_html=True,
                        )

    # Chat input
    user_input = st.chat_input("Type your message…")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.spinner("rufus-femto is thinking…"):
            out = st.session_state.session.chat(user_input)
        st.session_state.messages.append({"role": "assistant", "content": out["reply"]})
        st.session_state.trace.append({
            "turn": len(st.session_state.trace) + 1,
            "user": user_input,
            "tool_calls": out["tool_calls_this_turn"],
            "bus_size": out["bus_size"],
            "asks_so_far": out["asks_so_far"],
            "category": out["category"],
            "recommendations": out["recommendations"],
        })
        st.rerun()


# ===================== RIGHT: agent trace + bus inspector =====================


with right:
    st.markdown("### Agent thinking")
    if not st.session_state.trace:
        st.info("Tool calls and bus operations will show up here as you chat.")
    else:
        for entry in reversed(st.session_state.trace):
            label = (
                f"Turn {entry['turn']}  ·  bus={entry['bus_size']}  ·  asks={entry['asks_so_far']}"
                + ("  ·  ✅ RECOMMENDED" if entry["recommendations"] else "")
            )
            with st.expander(label, expanded=(entry == st.session_state.trace[-1])):
                st.caption(f"You said: *{entry['user']}*")
                calls = entry["tool_calls"]
                if not calls:
                    st.caption("(no tool calls this turn)")
                else:
                    for c in calls:
                        args = json.dumps(c.get("args", {}), default=str)
                        if len(args) > 100:
                            args = args[:97] + "…"
                        st.markdown(
                            f"<div class='tool-call'>"
                            f"<span class='tool-name'>{c['tool']}</span>"
                            f"<span class='tool-args'>({args})</span></div>"
                            f"<div class='tool-result'>↳ {c.get('result', '')[:160]}</div>",
                            unsafe_allow_html=True,
                        )

    st.divider()
    st.markdown("### Candidate Bus")
    sess: CRSAgentSession = st.session_state.session
    if sess.bus and sess.bus.size() > 0:
        st.caption(f"{sess.bus.size()} candidates · {len(sess.bus.notes)} ops")
        with st.expander("Audit trail"):
            for n in sess.bus.notes:
                st.markdown(f"- {n}")
        with st.expander("Top 10 ASINs"):
            from sandbox.catalog import load_catalog
            cat = list(load_catalog(sess.category))
            by_asin = {p["asin"]: p for p in cat if p.get("asin")}
            for i, asin in enumerate(sess.bus.top(10), 1):
                p = by_asin.get(asin, {})
                title = (p.get("title") or "")[:55]
                price = p.get("price")
                score = sess.bus.scores.get(asin)
                price_str = f"${price:,.0f}" if price else "—"
                score_str = f"  · score {score:.3f}" if score is not None else ""
                st.markdown(f"{i}. `{asin}` · {price_str}{score_str} — {title}")
    else:
        st.caption("Bus is empty — agent hasn't initialized a category yet.")
