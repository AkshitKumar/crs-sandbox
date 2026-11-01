"""Streamlit demo for live simulations, timed replay, and run comparison."""

from __future__ import annotations

import html
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

# Follow new messages: when the reader is near the bottom and content grows,
# scroll smoothly to it. Scrolling up to read pauses this until they return.
# The script is installed once in the parent page so it survives reruns.
AUTOSCROLL_JS = """
<script>
const doc = window.parent.document;
if (!window.parent.__crsAutoscroll) {
  window.parent.__crsAutoscroll = true;
  const script = doc.createElement("script");
  script.textContent = `
    (() => {
      const NEAR = 300;          // px from the bottom that still counts as "following"
      const DURATION = 1000;     // ms per scroll animation
      const scroller = () =>
        document.querySelector('[data-testid="stMain"]') ||
        document.querySelector('[data-testid="stAppViewContainer"]') ||
        document.scrollingElement;
      const distance = (el) => el.scrollHeight - el.scrollTop - el.clientHeight;
      const ease = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
      let follow = true, pending = false, frame = null;
      const stop = () => { if (frame) cancelAnimationFrame(frame); frame = null; };
      const glide = () => {
        stop();
        const el = scroller(), from = el.scrollTop, started = performance.now();
        const step = (now) => {
          const t = Math.min(1, (now - started) / DURATION);
          // Re-read the target each frame so content that keeps growing is reached.
          el.scrollTop = from + (el.scrollHeight - el.clientHeight - from) * ease(t);
          frame = t < 1 ? requestAnimationFrame(step) : null;
        };
        frame = requestAnimationFrame(step);
      };
      // The reader taking over cancels the animation; their position decides following.
      for (const kind of ["wheel", "touchstart", "keydown", "mousedown"]) {
        document.addEventListener(kind, () => { stop(); }, { capture: true, passive: true });
      }
      document.addEventListener("scroll", (event) => {
        const el = scroller();
        if (frame) return;  // our own animation
        if (event.target === el || event.target === document) follow = distance(el) < NEAR;
      }, true);
      new MutationObserver(() => {
        // A running animation already re-targets the growing bottom each frame.
        if (!follow || pending || frame) return;
        pending = true;
        requestAnimationFrame(() => {
          pending = false;
          if (distance(scroller()) > 1) glide();
        });
      }).observe(document.body, { childList: true, subtree: true });
    })();
  `;
  doc.head.appendChild(script);
}
</script>
"""
st.iframe(AUTOSCROLL_JS.strip(), height=1)


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


def _md(text: Any) -> str:
    """Escape dollar signs so Streamlit does not render prices as LaTeX."""
    return str(text or "").replace("$", "\\$")


def _money(value: Any) -> str:
    return "—" if value is None else f"${float(value):,.2f}"


def _rating(value: Any) -> str:
    return "no rating" if value in (None, "", "None") else f"{value} stars"


def _adaptive_step(call: dict[str, Any]) -> str | None:
    """One readable line (or short block) per adaptive controller step."""
    tool, args, result = call.get("tool"), call.get("args") or {}, call.get("result") or {}
    if tool == "recommend":
        # ATR logs recommend without a source; leave its display unchanged.
        return f"**Recommend:** from {args['source']}" if "source" in args else None
    if not isinstance(result, dict):
        return None
    if tool == "candidate_preview":
        plan = result.get("plan") or {}
        if result.get("source") == "default_slate":
            return "**Preview:** default slate (nothing asked yet)"
        price = ""
        if plan.get("max_price") or plan.get("target_price"):
            price = f" · target {_money(plan.get('target_price'))}, max {_money(plan.get('max_price'))}"
        return (
            f"**Preview:** query “{plan.get('focus_query', '')}”{price} · "
            f"{len(result.get('asins') or [])} candidates of {result.get('eligible_count')} eligible"
        )
    if tool == "decide_next_action":
        threshold = result.get("threshold")
        lines = [
            f"**Decide:** {result.get('action')}"
            + (" (forced)" if args.get("forced") else "")
            + (f" · threshold {threshold}" if threshold is not None else "")
            + f" — {result.get('reason', '')}"
        ]
        metrics = result.get("metrics") or {}
        if metrics:
            lines.append("metrics: " + ", ".join(f"{key} {value}" for key, value in metrics.items()))
        for item in result.get("assessments") or []:
            probability = item.get("swap_probability")
            marker = "▲" if threshold is not None and probability is not None and probability >= threshold else "·"
            lines.append(f"{marker} `{item.get('question_id')}` {probability} — {item.get('how', '')}")
        return "  \n".join(lines)
    if tool == "choose_question":
        return f"**Ask:** `{result.get('question_id')}` — {result.get('reason', '')}"
    return None


def _render_tools(calls: list[dict[str, Any]]) -> None:
    if not calls:
        return
    with st.expander(f"{len(calls)} tool call(s)"):
        for call in calls:
            step = _adaptive_step(call)
            if step is not None:
                st.markdown(_md(step))
                continue
            st.code(
                f"{call.get('tool')}({json.dumps(call.get('args') or {})})\n"
                f"→ {str(call.get('result') or '')[:600]}"
            )


CARD_CSS = """
<style>
.rec-row { display: flex; gap: 0.75rem; align-items: stretch; margin: 0.25rem 0 0.5rem; }
.rec-card { flex: 1 1 0; min-width: 0; border: 2px solid rgba(128,128,128,0.35);
  border-radius: 0.6rem; padding: 0.7rem 0.8rem; display: flex; flex-direction: column; gap: 0.35rem; }
.rec-card.bought { border-color: #22c55e; background: rgba(34,197,94,0.10); }
.rec-card.passed { border-color: rgba(128,128,128,0.6); opacity: 0.6; }
.rec-card.declined { border-color: #ef4444; background: rgba(239,68,68,0.08); }
.rec-badge.none { color: #ef4444; }
.rec-head { display: flex; justify-content: space-between; font-size: 0.8rem; opacity: 0.8; }
.rec-badge { color: #22c55e; font-weight: 700; }
.rec-title { font-weight: 600; font-size: 0.95rem; line-height: 1.3; display: -webkit-box;
  -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; }
.rec-price { font-weight: 700; }
.rec-why { font-size: 0.85rem; line-height: 1.35; opacity: 0.85; display: -webkit-box;
  -webkit-line-clamp: 4; -webkit-box-orient: vertical; overflow: hidden; }
.rec-row.mini { gap: 0.4rem; margin: 0.1rem 0 0.3rem; }
.rec-row.mini .rec-card { padding: 0.4rem 0.5rem; gap: 0.15rem; border-radius: 0.45rem; }
.rec-row.mini .rec-head { font-size: 0.68rem; }
.rec-row.mini .rec-title { font-size: 0.75rem; -webkit-line-clamp: 2; }
.rec-row.mini .rec-meta { font-size: 0.72rem; }
.rec-row.mini .rec-why { display: none; }
</style>
"""


def _html(text: Any) -> str:
    """Escape for raw HTML; &#36; keeps Streamlit from reading prices as LaTeX."""
    return html.escape(str(text or "")).replace("$", "&#36;")


def _cards_html(
    recommendations: list[dict[str, Any]], decision: dict[str, Any] | None, *, mini: bool = False
) -> str:
    """Horizontal product cards: green on the purchase (others grey), all red on no purchase."""
    bought = (decision or {}).get("purchased_asin")
    outcome = (decision or {}).get("outcome")
    cards = []
    for index, product in enumerate(recommendations, start=1):
        if outcome == "PURCHASE":
            state = "bought" if product.get("asin") == bought else "passed"
        elif outcome == "NO_PURCHASE":
            state = "declined"
        else:
            state = ""
        badge = {
            "bought": f'<span class="rec-badge">{"BOUGHT" if mini else "PURCHASED"}</span>',
            "declined": f'<span class="rec-badge none">{"NO" if mini else "NO PURCHASE"}</span>',
        }.get(state, "")
        # Older runs can name products that have since left the catalog.
        title = product.get("title") or f"{product.get('asin')} (no longer in catalog)"
        cards.append(
            f'<div class="rec-card {state}">'
            f'<div class="rec-head"><span>{"" if mini else "Option "}{index}</span>{badge}</div>'
            f'<div class="rec-title" title="{_html(title)}">{_html(title)}</div>'
            f'<div class="rec-meta"><span class="rec-price">{_html(_money(product.get("price")))}</span> · '
            f'{_html(_rating(product.get("avg_rating")))}</div>'
            f'<div class="rec-why">{_html(product.get("recommendation_explanation"))}</div>'
            "</div>"
        )
    return CARD_CSS + f'<div class="rec-row{" mini" if mini else ""}">{"".join(cards)}</div>'


def _render_cards(
    slot: Any, recommendations: list[dict[str, Any]], decision: dict[str, Any] | None = None
) -> None:
    if recommendations:
        slot.markdown(_cards_html(recommendations, decision), unsafe_allow_html=True)


def _is_recommendation(calls: list[dict[str, Any]]) -> bool:
    return any(call.get("tool") == "recommend" for call in calls or [])


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


def _render_decision(record: dict[str, Any], recommendations: list[dict[str, Any]]) -> None:
    """The buyer's purchase decision as their final chat message."""
    decision = record.get("decision") or record.get("buyer_decision_raw") or {}
    outcome = record.get("outcome") or decision.get("decision")
    with st.chat_message("user"):
        if outcome == "PURCHASE":
            bought = record.get("purchased_asin")
            index, product = next(
                ((i, p) for i, p in enumerate(recommendations, start=1) if p.get("asin") == bought),
                (decision.get("product_number"), {}),
            )
            st.markdown(
                _md(
                    f":green[**PURCHASE**] · option {index}: {product.get('title') or bought} "
                    f"({_money(record.get('actual_price') or product.get('price'))})"
                )
            )
        else:
            st.markdown(f":red[**{outcome or 'NO_PURCHASE'}**]".replace("_", " "))
        if decision.get("reasoning"):
            st.markdown(_md(decision["reasoning"]))


def _render_checkpoint(checkpoint: dict[str, Any], category: str) -> None:
    """A hidden branching-ATR checkpoint: mini cards plus the buyer's decision."""
    depth = int(checkpoint.get("depth") or 0)
    label = (
        "Before any question (default slate)"
        if depth == 0
        else f"After {depth} answer{'' if depth == 1 else 's'}"
    )
    if checkpoint.get("evaluation_status") != "scored":
        st.caption(_md(f"**{label}** · {checkpoint.get('error_code') or 'not scored'}"))
        return
    recommendations = _hydrate_recommendations(
        category, (checkpoint.get("recommendation_result") or {}).get("recommendations") or []
    )
    if checkpoint.get("outcome") == "PURCHASE":
        index = next(
            (i for i, p in enumerate(recommendations, start=1)
             if p.get("asin") == checkpoint.get("purchased_asin")),
            "?",
        )
        verdict = f":green[**PURCHASE**] option {index} ({_money(checkpoint.get('actual_price'))})"
    else:
        verdict = ":red[**NO PURCHASE**]"
    st.markdown(_md(f"**{label}** · ") + verdict)
    st.markdown(_cards_html(recommendations, checkpoint, mini=True), unsafe_allow_html=True)
    reasoning = (checkpoint.get("decision") or {}).get("reasoning")
    if reasoning:
        with st.expander("Buyer reasoning"):
            st.markdown(_md(reasoning))


ROW_WIDTHS = [12, 1.3, 8]


def _left(state: dict[str, Any]) -> Any:
    """Where the next conversation item goes.

    Branching runs give every message its own row so a checkpoint can sit level
    with the answer it followed: [message | connector | checkpoint].
    """
    if not state.get("branching"):
        return state.setdefault("main", st.container())
    if not state.get("headed"):
        left, _, right = st.columns(ROW_WIDTHS, gap="small")
        left.markdown("##### Conversation")
        right.markdown("##### Hidden checkpoints")
        right.caption("What the buyer would do if recommended to after each answer.")
        state["headed"] = True
    state["row"] = st.columns(ROW_WIDTHS, gap="small", vertical_alignment="center")
    return state["row"][0]


def _connector(outcome: str | None) -> str:
    color = {"PURCHASE": "#22c55e", "NO_PURCHASE": "#ef4444"}.get(outcome or "", "rgba(128,128,128,0.6)")
    return (
        f'<div style="display:flex;align-items:center;width:100%">'
        f'<div style="flex:1;border-top:2px dotted {color}"></div>'
        f'<div style="width:7px;height:7px;border-radius:50%;background:{color}"></div></div>'
    )


TIMELINE_CSS = """
<style>
.tl-row { display: flex; align-items: stretch; gap: 0.3rem; margin: 0.5rem 0 1rem; }
.tl-arrow { align-self: center; opacity: 0.5; font-size: 1.1rem; }
.tl-card { flex: 1 1 0; min-width: 0; border: 2px solid rgba(128,128,128,0.4); border-radius: 0.6rem;
  padding: 0.55rem 0.6rem; display: flex; flex-direction: column; gap: 0.25rem; }
.tl-card.buy { border-color: #22c55e; background: rgba(34,197,94,0.10); }
.tl-card.no { border-color: #ef4444; background: rgba(239,68,68,0.08); }
.tl-step { font-size: 0.72rem; opacity: 0.75; text-transform: uppercase; letter-spacing: 0.03em; }
.tl-verdict { font-weight: 700; font-size: 0.85rem; }
.tl-card.buy .tl-verdict { color: #22c55e; }
.tl-card.no .tl-verdict { color: #ef4444; }
.tl-title { font-size: 0.78rem; line-height: 1.3; display: -webkit-box; -webkit-line-clamp: 3;
  -webkit-box-orient: vertical; overflow: hidden; }
.tl-note { font-size: 0.7rem; opacity: 0.7; }
</style>
"""


def _timeline_html(checkpoints: list[dict[str, Any]], steps: dict[int, str], category: str) -> str:
    """One tile per checkpoint: what the buyer would have done at that depth."""
    tiles, previous = [], None
    for checkpoint in sorted(checkpoints, key=lambda item: item.get("depth", 0)):
        depth = int(checkpoint.get("depth") or 0)
        step = "Start" if depth == 0 else f"After Q{depth}" + (f": {steps[depth]}" if steps.get(depth) else "")
        if checkpoint.get("evaluation_status") != "scored":
            tiles.append(f'<div class="tl-card"><div class="tl-step">{_html(step)}</div>'
                         f'<div class="tl-verdict">not scored</div></div>')
            continue
        asin = checkpoint.get("purchased_asin")
        if checkpoint.get("outcome") == "PURCHASE":
            product = next(iter(_hydrate_recommendations(category, [{"asin": asin}])), {})
            title = product.get("title") or f"{asin} (no longer in catalog)"
            note = "same pick" if asin == previous else ("new pick" if previous else "")
            tiles.append(
                f'<div class="tl-card buy"><div class="tl-step">{_html(step)}</div>'
                f'<div class="tl-verdict">PURCHASE · {_html(_money(checkpoint.get("actual_price")))}</div>'
                f'<div class="tl-title" title="{_html(title)}">{_html(title)}</div>'
                + (f'<div class="tl-note">{note}</div>' if note else "")
                + "</div>"
            )
        else:
            tiles.append(f'<div class="tl-card no"><div class="tl-step">{_html(step)}</div>'
                         f'<div class="tl-verdict">NO PURCHASE</div></div>')
        previous = asin or previous
    return TIMELINE_CSS + '<div class="tl-row">' + '<div class="tl-arrow">→</div>'.join(tiles) + "</div>"


def _render_timeline(state: dict[str, Any]) -> None:
    checkpoints = state.get("checkpoints") or []
    if not checkpoints:
        return
    st.markdown("##### Purchase decision after each answer")
    st.caption("Each checkpoint is an independent decision on that depth's recommendations.")
    st.markdown(
        _timeline_html(checkpoints, state.get("steps") or {}, state.get("category", "")),
        unsafe_allow_html=True,
    )


def _render_event(event: dict[str, Any], state: dict[str, Any]) -> None:
    event_type = event.get("type")
    if event_type in {"buyer_message", "recommender_message"}:
        role = "user" if event_type == "buyer_message" else "assistant"
        calls = event.get("tool_calls") or []
        if state.get("branching") and role == "assistant":
            if _is_recommendation(calls):
                return  # identical to the last checkpoint; the timeline replaces it
            asked = next((call.get("result") for call in calls if call.get("tool") == "ask_question"), None)
            state.setdefault("steps", {})[len(state.get("steps") or {}) + 1] = str(asked or "").replace("_", " ")
        with _left(state), st.chat_message(role):
            if role == "assistant" and _is_recommendation(calls):
                # Cards arrive with the next event; fill this slot then.
                st.markdown(_md(str(event.get("content") or "").split("\n")[0]))
                state["slot"] = st.empty()
            else:
                st.markdown(_md(event.get("content")))
            _render_tools(calls)
    elif event_type == "checkpoint":
        checkpoint = event["checkpoint"]
        if state.get("branching"):
            state.setdefault("checkpoints", []).append(checkpoint)
            # Fill the row of the answer this checkpoint followed.
            if state.get("row") is None:
                _left(state)
            _, line, right = state["row"]
            line.markdown(_connector(checkpoint.get("outcome")), unsafe_allow_html=True)
            with right:
                _render_checkpoint(checkpoint, state.get("category", ""))
            state["row"] = None
        else:
            st.caption(
                f"hidden depth {checkpoint.get('depth')}: "
                f"{checkpoint.get('evaluation_status')} · {checkpoint.get('outcome', '')}"
            )
    elif event_type == "recommendations":
        state["items"] = event.get("items") or []
        if state.get("branching"):
            return
        _render_cards(state.get("slot") or _left(state).empty(), state["items"])
    elif event_type == "decision":
        if state.get("branching"):
            _render_timeline(state)
            return
        if state.get("slot") is not None:
            _render_cards(state["slot"], state.get("items") or [], event)
        with _left(state):
            _render_decision(event, state.get("items") or [])
    elif event_type == "abandoned":
        with _left(state):
            st.warning(_md(f"Buyer abandoned: {event.get('reason') or 'no reason given'}"))
    elif event_type == "error":
        with _left(state):
            st.error(_md(f"{event.get('error_code')}: {event.get('error')}"))


# Events that put something new on screen; pacing applies between these.
PACED_EVENTS = {"buyer_message", "recommender_message", "checkpoint", "decision", "abandoned"}


def _record_events(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Rebuild the live event stream from a saved transcript."""
    dialogue = record.get("dialogue") or []
    tool_turns = record.get("tool_calls_per_turn") or record.get("crs_tool_calls_per_turn") or []
    checkpoints = sorted(
        record.get("checkpoints") or [], key=lambda checkpoint: checkpoint.get("depth", 0)
    )
    recommendation_result = record.get("recommendation_result") or {}
    recommendations = _hydrate_recommendations(
        str(record.get("category") or ""),
        recommendation_result.get("recommendations")
        or record.get("recommendations")
        or record.get("crs_recommendations")
        or [],
    )
    events: list[dict[str, Any]] = []
    assistant_index = 0
    for position, message in enumerate(dialogue + [None]):
        events += [
            {"type": "checkpoint", "checkpoint": checkpoint}
            for checkpoint in checkpoints
            if checkpoint.get("dialogue_message_count") == position
        ]
        if message is None:
            break
        if message.get("role") == "assistant":
            calls = tool_turns[assistant_index] if assistant_index < len(tool_turns) else []
            assistant_index += 1
            events.append({"type": "recommender_message", "content": message.get("content"), "tool_calls": calls})
        else:
            events.append({"type": "buyer_message", "content": message.get("content")})
    if recommendations:
        events.append({"type": "recommendations", "items": recommendations})
    if record.get("outcome") in {"PURCHASE", "NO_PURCHASE"}:
        events.append({"type": "decision", **record})
    return events


def _replay(record: dict[str, Any], delay: float) -> None:
    state = {
        "branching": bool(record.get("checkpoints")),
        "category": str(record.get("category") or ""),
    }
    for event in _record_events(record):
        _render_event(event, state)
        if delay and event["type"] != "recommendations":
            time.sleep(delay)


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
    fixed_order = False
    if policy_name == "adaptive":
        fixed_order = st.sidebar.checkbox("Fixed question order (as in ATR)", value=False)
    endogenous = st.sidebar.checkbox("Endogenous abandonment", value=False)
    min_gap = st.sidebar.slider(
        "Minimum seconds between messages",
        0.0,
        5.0,
        2.0,
        0.25,
        help="Fast steps wait this long before appearing; slow API calls are not delayed further.",
    )
    if st.sidebar.button("Run simulation", type="primary"):
        simulation = Simulation(
            persona=persona,
            category=category,
            policy=Policy.make(policy_name, int(numquestions)),
            endogenous_abandonment=endogenous,
            fixed_order=fixed_order,
        )
        live_state: dict[str, Any] = {
            "branching": policy_name == "branching_atr",
            "category": category,
        }
        shown_at = None
        for event in simulation.run_iter():
            if event.get("type") in PACED_EVENTS:
                if shown_at is not None:
                    time.sleep(max(0.0, min_gap - (time.monotonic() - shown_at)))
                shown_at = time.monotonic()
            _render_event(event, live_state)

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
