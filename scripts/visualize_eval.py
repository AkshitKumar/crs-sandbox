"""rufus-femto · transcript visualizer.

Browse buyer ↔ CRS simulated conversations from an eval run. Pick a result
directory, pick a persona, and see:
    - the persona's HIDDEN ground truth (left sidebar) — what the buyer
      knows but only reveals when asked
    - the full chat transcript (main)
    - the tool calls the CRS made each turn (inline expanders)
    - the recommendation set
    - the buyer's purchase decision + reasoning

Run with:
    streamlit run scripts/visualize_eval.py --server.port=8502
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

st.set_page_config(page_title="rufus-femto · transcript viewer", layout="wide", page_icon="🔎")

# ----------------- CSS polish -----------------

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
      .pill-purchase    { background:#d4edda; color:#155724; }
      .pill-nopurchase  { background:#f8d7da; color:#721c24; }
      .pill-abandoned   { background:#fff3cd; color:#856404; }
      .pill-error       { background:#e2e3e5; color:#383d41; }
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

# ----------------- helpers -----------------


def _list_eval_dirs() -> list[Path]:
    root = REPO_ROOT / "results"
    if not root.exists():
        return []
    return sorted(
        [p for p in root.iterdir() if p.is_dir() and (p / "transcripts.jsonl").exists()],
        reverse=True,
    )


def _load_transcripts(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_personas_for_category(category: str) -> dict[str, dict]:
    """Map persona_id → persona dict for the category, for ground-truth display."""
    path = REPO_ROOT / "data" / "categories" / category / "personas.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    personas = data.get("personas", data)
    return {p["id"]: p for p in personas if p.get("id")}


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


# ----------------- Page -----------------

st.markdown(
    "<div class='vh-header'><h1>🔎 rufus-femto · transcript viewer</h1>"
    "<span class='tag'>browse buyer × CRS simulated conversations</span></div>",
    unsafe_allow_html=True,
)

# Eval run + persona selection
dirs = _list_eval_dirs()
if not dirs:
    st.warning(
        "No eval runs found. Run `python scripts/run_buyer_eval.py <category> "
        "--n-personas 5` first, then refresh this page."
    )
    st.stop()

with st.sidebar:
    st.markdown("### Run")
    selected_dir_name = st.selectbox(
        "Eval run",
        options=[d.name for d in dirs],
        index=0,
        help="Sorted newest-first.",
    )
    selected_dir = next(d for d in dirs if d.name == selected_dir_name)
    rows = _load_transcripts(selected_dir / "transcripts.jsonl")
    if not rows:
        st.error("Selected eval directory has no transcripts.")
        st.stop()

    persona_ids = [r["persona_id"] for r in rows]
    selected_id = st.selectbox("Persona", options=persona_ids, index=0)
    record = next(r for r in rows if r["persona_id"] == selected_id)
    category = record["category"]
    persona_map = _load_personas_for_category(category)
    persona = persona_map.get(selected_id, {})

    # Run-level summary
    st.divider()
    st.markdown("### Run summary")
    n_total = len(rows)
    n_purchase = sum(1 for r in rows if r["outcome"] == "PURCHASE")
    n_nop = sum(1 for r in rows if r["outcome"] == "NO_PURCHASE")
    n_ab = sum(1 for r in rows if r["outcome"] == "ABANDONED")
    n_err = sum(1 for r in rows if r["outcome"] == "ERROR")
    st.metric("personas", n_total)
    st.metric("purchase rate", f"{n_purchase / n_total:.0%}")
    st.caption(f"{n_purchase} purchased · {n_nop} no-purchase · {n_ab} abandoned · {n_err} error")

# ----------------- main: this persona -----------------

c_left, c_right = st.columns([1, 2], gap="large")

# Left: persona ground truth + outcome
with c_left:
    st.markdown("### Persona")
    st.caption(f"`{selected_id}` · category `{category}`")
    if persona:
        st.markdown(
            f"<div class='ground-truth-card'><b>Background</b><br>{persona.get('background', '')}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='ground-truth-card'><b>Hidden ground-truth need</b> (visible to buyer only)<br>{persona.get('ground_truth_need', '')}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.warning(f"No persona file found for category `{category}`.")

    st.markdown("### Outcome")
    st.markdown(_outcome_pill(record["outcome"]), unsafe_allow_html=True)
    cols = st.columns(2)
    cols[0].metric("turns used", record.get("turns_used", "—"))
    cols[1].metric("clarifying asks", record.get("asks", "—"))
    if record["outcome"] == "PURCHASE":
        cols2 = st.columns(2)
        wtp = record.get("wtp")
        price = record.get("actual_price")
        surplus = record.get("consumer_surplus")
        cols2[0].metric("WTP", _fmt_price(wtp))
        cols2[1].metric("paid", _fmt_price(price))
        if surplus is not None:
            st.metric("consumer surplus", f"${surplus:,.0f}")
    if record.get("abandoned_at_turn"):
        st.caption(f"abandoned at turn {record['abandoned_at_turn']}")
    if record.get("error"):
        st.error(record["error"])

# Right: transcript with tool calls + final recommendations
with c_right:
    st.markdown("### Transcript")
    dialogue = record.get("dialogue", [])
    tool_log = record.get("crs_tool_calls_per_turn", [])

    # CRS speaks every other message starting with index 1 (buyer goes first
    # with the opener). Track which CRS turn we're on so we can attach tools.
    crs_turn_idx = -1
    for i, msg in enumerate(dialogue):
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "assistant":
            crs_turn_idx += 1
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(content)
                calls = tool_log[crs_turn_idx] if 0 <= crs_turn_idx < len(tool_log) else []
                if calls:
                    with st.expander(f"🔧 {len(calls)} tool call(s)"):
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
        else:
            with st.chat_message("user", avatar="🧍"):
                st.markdown(content)

    # Recommendations
    recs = record.get("crs_recommendations") or []
    if recs:
        st.markdown("### Final recommendation set")
        for rec in recs:
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
                cr.markdown(_fmt_rating(rec.get("avg_rating"), rec.get("num_reviews")), unsafe_allow_html=True)
                brand = rec.get("brand")
                if brand:
                    cb.markdown(f"<span class='brand-tag'>{brand}</span>", unsafe_allow_html=True)
                bullets = rec.get("bullets") or []
                if bullets:
                    with st.expander("Key bullets"):
                        for b in bullets[:5]:
                            st.markdown(f"- {b}")

    # Buyer's decision reasoning
    decision = record.get("buyer_decision_raw") or {}
    if decision:
        st.markdown("### Buyer's decision")
        d = decision.get("decision")
        chosen_n = decision.get("product_number")
        chosen_asin = decision.get("asin") or record.get("purchased_asin")
        wtp = decision.get("willingness_to_pay")
        reasoning = decision.get("reasoning", "")
        if d == "PURCHASE":
            st.success(
                f"PURCHASED product #{chosen_n} (`{chosen_asin}`) · WTP ${wtp}"
            )
        else:
            st.info("No purchase")
        if reasoning:
            st.markdown(f"> {reasoning}")
