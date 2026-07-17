"""LangGraph-based tool-using CRS agent.

The agent is a ReAct-style loop wrapped by `langgraph.prebuilt.create_react_agent`.
Each conversation lives in a `CRSAgentSession` that holds:
    - a CandidateBus     (mutated by tools)
    - a QuestionTool     (tracks which openers/followups have been asked)
    - an asks_so_far     counter
    - a dialogue ledger  (for inspection / chat-UI rendering)

Tools are defined inside `_make_tools` as closures over the session, so the
LLM doesn't need to pass the bus around in arguments — it just calls
`semantic_search_full(...)` or `rank_by_match(...)` and the session updates its
bus.

When the agent decides to recommend, it calls `recommend(query=..., key_query=...)`.
The shared pipeline builds a 15-product hybrid recall pool and asks one model
call to choose and briefly explain three of them.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

import openai._base_client as openai_base_client
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.prebuilt import create_react_agent

from sandbox.catalog import load_catalog, load_config
from sandbox.agents.recommendation_pipeline import (
    PROTOCOL_VERSION,
    PreferenceLedger,
    RecommendationPipeline,
    RecommendationSnapshot,
)
from sandbox.elicitation_policy import ElicitationPolicy
from sandbox.tools.candidate_bus import CandidateBus
from sandbox.tools.feasibility_tool import (
    check_category_supported as _check_category_supported,
    list_available_categories,
)
from sandbox.tools.filter_tool import apply_filter, available_filters, preview_filter as _preview_filter
from sandbox.tools.inspect_tool import catalog_overview, compare, get_product_details
from sandbox.tools.question_bank import QuestionBank, QuestionTool
from sandbox.tools.ranking_tool import (
    MatchWeights,
    rank_by_commission,
    rank_by_match,
    rank_by_price,
)
from sandbox.tools.search_tool import narrow_search, semantic_search
from sandbox.tools.uncertainty_tool import ActionThresholds, compute_uncertainty, suggest_next_action
from sandbox.openai_responses import OPENAI_MAX_RETRIES, message_to_text


DEFAULT_MODEL = "gpt-5-mini"
DEFAULT_REASONING = "medium"


SYSTEM_PROMPT = """You are a shopping assistant on Amazon helping a customer pick the right product.
You have access to a real product catalog through a set of tools — you cannot
see products directly, only through tool calls. Your goal is to ask enough
clarifying questions to understand what the customer needs, then recommend
3 well-chosen products.

# Categories you support right now

{available_categories}

If the customer asks about something outside these categories, use
`check_category_supported` to confirm and then politely say you don't carry
that. Do not invent products.

# How to drive the conversation

You are working with a hidden "Candidate Bus" while you explore the catalog.
Search, filtering, ranking, and inspection tools can change or examine this bus,
and it persists across tool calls. It is a workspace for reasoning, not the final
recommendation set: `recommend(query=..., key_query=...)` always searches the
full eligible catalog again from the queries you provide.

Typical flow (use judgment, this is not a script or a checklist):

  1. If the customer's initial message is ambiguous about the product type,
     call `check_category_supported` first. If supported, set the category with
     `set_category`. From then on, every catalog tool acts on that category.
  2. Use `catalog_overview` early when knowing the catalog's price range, brands,
     or general contents would help.
  3. Ask clarifying questions with `ask_question()`. Ask only one question at a
     time. After each answer, decide whether another answer would materially
     improve the recommendation or whether you know enough to recommend.
  4. Use filters to remove clear mismatches, not to enforce every preference
     exactly. Apply exact filters only to genuine hard constraints. For other
     preferences, filter loosely enough to leave room for tradeoffs; if there is
     no sensible loose filter, use search or ranking instead.
  5. Use search, inspection, comparison, ranking, and uncertainty tools when they
     help you understand the catalog or decide what to ask. They do not lock the
     final products. Avoid long tool loops; use only the tools that help.
  6. When ready, call `recommend(query=..., key_query=...)`. The query should
     concisely cover all revealed preferences. The key query should be a short
     phrase for the most important revealed requirement. Do not add assumptions
     or hidden requirements. The tool retrieves 15 products from the full
     hard-budget-eligible catalog and returns exactly three products with a short
     personalized explanation for each. Those products and their order are final.

# Style

- Conversational, concise. Ask one question at a time — never stack multiple
  questions or `ask_question()` calls in one turn.
- Don't repeat back the customer's words verbatim. Acknowledge briefly and ask.
- Be thoughtful when using tools; prefer using fewer tools over many.
- Only your final no-tool assistant message for the turn is shown to the customer.
  It must be a complete customer-facing response.
- NEVER reveal you're using tools or anything about the internal mechanics, such
  as the catalog, Candidate Bus, embedding retrieval, or ASINs. Keep the
  conversation focused on the customer and their preferences.

# Failure modes to avoid

- Do not interpret every strongly worded preference as a hard filter. Unless the
  customer gives a genuinely non-negotiable constraint, let retrieval and final
  selection handle the tradeoff.
- Do not assume that products currently at the top of the Candidate Bus will be
  recommended. The final `recommend` query is the complete input to fresh
  full-catalog retrieval, so include every important revealed preference in the
  full query.
- Use the exact products and order returned by `recommend()`. Do not replace,
  rename, or add products in your customer-facing response.
- Do not provide generic recommendations without specific catalog products.
- Never ask the customer for ASINs, Amazon links, screenshots, live listing text,
  or product-page fields. Use the catalog information available to you.
- Avoid inventing product attributes you did not see. If a detail is unavailable,
  state that uncertainty briefly and make the best recommendation from the
  available evidence.
- If the catalog has no exact match, recommend the closest available products;
  do not ask the customer to find products for you.

"""


def _policy_prompt(policy: ElicitationPolicy) -> str:
    if policy.name == "rec":
        return """

# Fixed elicitation policy for this run

The experiment has already fixed the product category. Ask zero clarifying
questions and call recommend(query="") during this turn. For this zero-information
condition, recommend() freezes the category's curated initial slate; do not
attempt to replace those products in prose.
"""

    checkpoint_instruction = ""
    if policy.has_nonterminal_checkpoints:
        checkpoint_instruction = (
            "After each customer answer, call semantic_search_full once with a concise full query "
            "and a short key query for the most important preference revealed so far before asking "
            "the next question."
        )
    return f"""

# Internal question budget for this run

The experiment has already fixed the product category. Use an internal question
budget of {policy.target_asks}. Spend the full budget on one-at-a-time
clarifying questions before recommending. ask_question() returns the next fixed
experimental question; its topic argument cannot change that order. You may use
the other catalog tools to interpret answers and prepare candidates.
{checkpoint_instruction}
Once the budget is exhausted, call recommend(query=..., key_query=...) with the
current revealed preferences and the most important revealed requirement.
The tools enforce the boundary. Do not
mention the question budget, question number, or experimental policy to the
customer.
"""


@dataclass
class CRSAgentSession:
    """One conversation. Holds the live bus + tool state.

    Multi-turn memory is provided by a per-session LangGraph checkpointer
    keyed by `thread_id`. Each call to `chat()` reuses the same thread, so
    the agent sees its full prior message history (including tool calls
    and results).
    """

    category: Optional[str] = None
    bus: Optional[CandidateBus] = None
    qtool: Optional[QuestionTool] = None
    asks_so_far: int = 0
    asked_question_this_turn: bool = False
    pending_question_text: Optional[str] = None
    recommendation_finalized_this_turn: bool = False
    recommendation_prose_raw: Optional[str] = None
    visible_reply_to_sync: Optional[str] = None
    recommendations: Optional[list] = None
    visible_dialogue: list[dict[str, str]] = field(default_factory=list)
    tool_log: list = field(default_factory=list)
    model: str = DEFAULT_MODEL
    reasoning_effort: str = DEFAULT_REASONING
    elicitation_policy: Optional[ElicitationPolicy] = None
    retrieval_limit: int = 15
    ledger: PreferenceLedger = field(default_factory=PreferenceLedger)
    question_ids: list[str] = field(default_factory=list)
    recommendation_source: Optional[str] = None
    recommendation_validation_error: Optional[str] = None
    recommendation_selection_raw: Optional[str] = None
    recommendation_product_numbers: list[int] = field(default_factory=list)
    retrieval_query: Optional[str] = None
    retrieval_key_query: Optional[str] = None
    retrieval_query_raw: Optional[str] = None
    latest_embedding_query: Optional[str] = None
    latest_key_query: Optional[str] = None
    retrieval_candidate_asins: list[str] = field(default_factory=list)
    retrieval_scores: dict[str, float] = field(default_factory=dict)
    retrieval_lane_sources: dict[str, list[str]] = field(default_factory=dict)
    eligible_count: Optional[int] = None
    terminal_status: Optional[str] = None
    checkpoints: list[RecommendationSnapshot] = field(default_factory=list)

    # The compiled LangGraph agent (lazily built once self exists).
    _agent: Any = None
    _checkpointer: Any = None
    _thread_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    _pipeline: Optional[RecommendationPipeline] = None
    _new_checkpoints_this_turn: list[RecommendationSnapshot] = field(default_factory=list)
    _awaiting_answer_context: Optional[dict[str, str]] = None

    def __post_init__(self) -> None:
        if self.elicitation_policy is None:
            return
        if not self.category:
            raise ValueError("controlled elicitation policies require a known category")
        qtool = self._ensure_qtool()
        available = len(qtool.bank.openers) + len(qtool.bank.followups_ordered)
        if self.elicitation_policy.target_asks > available:
            raise ValueError(
                f"policy requests {self.elicitation_policy.target_asks} questions but "
                f"{self.category!r} has only {available}"
            )

    # ------------------------------------------------------------------
    # Public surface
    # ------------------------------------------------------------------

    def chat(self, user_message: str, max_steps: int = 90) -> dict[str, Any]:
        """Send one user turn through the agent. Returns the reply + audit info.

        `max_steps` is LangGraph's recursion_limit: roughly each tool call
        counts as 2 steps (agent decides → tool runs → agent reads result).
        90 → ~45 tool calls per turn, generous for our 18-tool taxonomy.

        If the agent hits the limit, we still return whatever tool calls
        happened plus a fallback message so the chat doesn't break.
        """
        self.visible_dialogue.append({"role": "user", "content": user_message})
        context = self._awaiting_answer_context or {"source": "initial_request"}
        self.ledger.observe(user_message, **context)
        self._awaiting_answer_context = None
        self._new_checkpoints_this_turn = []
        pending_checkpoint: tuple[int, list[str]] | None = None
        if (
            self.elicitation_policy is not None
            and self.elicitation_policy.has_nonterminal_checkpoints
            and self.asks_so_far < self.elicitation_policy.target_asks
        ):
            pending_checkpoint = (self.asks_so_far, list(self.question_ids))

        if self._agent is None:
            self._agent = self._build_agent()

        self.asked_question_this_turn = False
        self.pending_question_text = None
        self.recommendation_finalized_this_turn = False
        before_recs = self.recommendations
        config = {
            "configurable": {"thread_id": self._thread_id},
            "recursion_limit": max_steps,
        }
        messages = []
        if self.visible_reply_to_sync:
            messages.append({"role": "assistant", "content": self.visible_reply_to_sync})
            self.visible_reply_to_sync = None
        messages.append({"role": "user", "content": user_message})

        try:
            result = self._agent.invoke(
                {"messages": messages},
                config=config,
            )
            reply = message_to_text(result["messages"][-1])
        except GraphRecursionError:
            reply = (
                "Sorry — I got a bit tangled up trying to answer that. "
                "Let me try a simpler approach. Could you restate what you're looking for "
                "in one or two sentences?"
            )
        original_reply = reply
        if self.pending_question_text:
            reply = self.pending_question_text
        elif self.recommendation_finalized_this_turn and self.recommendations is not before_recs:
            reply = self._render_recommendation_reply()
        elif (
            self.elicitation_policy is not None
            and not self.elicitation_policy.allows_early_recommendations
            and self.asks_so_far < self.elicitation_policy.target_asks
            and not self.recommendations
        ):
            forced_question = self._force_next_question()
            if forced_question:
                reply = forced_question
        elif (
            self.elicitation_policy is not None
            and self.asks_so_far >= self.elicitation_policy.target_asks
            and not self.recommendations
            and self.terminal_status is None
        ):
            self._finalize_recommendations(
                query=self.latest_embedding_query or self.ledger.as_text(),
                key_query=self.latest_key_query,
            )
            if self.terminal_status == "NO_FEASIBLE_MATCH":
                reply = (
                    "I couldn’t find a catalog option within the hard budget you gave me, "
                    "so I won’t recommend an item that violates it."
                )
            else:
                reply = self._render_recommendation_reply()
        if pending_checkpoint is not None:
            checkpoint_asks, checkpoint_question_ids = pending_checkpoint
            self._capture_hidden_checkpoint(
                asks_so_far=checkpoint_asks,
                question_ids=checkpoint_question_ids,
                query=self.latest_embedding_query,
                key_query=self.latest_key_query,
            )
        if reply != original_reply:
            self.visible_reply_to_sync = reply
        self.visible_dialogue.append({"role": "assistant", "content": reply})
        new_recs = self.recommendations if self.recommendations is not before_recs else None
        return {
            "reply": reply,
            "recommendations": new_recs,
            "bus_size": self.bus.size() if self.bus else None,
            "asks_so_far": self.asks_so_far,
            "category": self.category,
            "question_ids": list(self.question_ids),
            "recommendation_source": self.recommendation_source,
            "recommendation_validation_error": self.recommendation_validation_error,
            "recommendation_selection_raw": self.recommendation_selection_raw,
            "recommendation_prose_raw": self.recommendation_prose_raw,
            "recommendation_product_numbers": list(self.recommendation_product_numbers),
            "retrieval_query": self.retrieval_query,
            "retrieval_key_query": self.retrieval_key_query,
            "retrieval_query_raw": self.retrieval_query_raw,
            "latest_embedding_query": self.latest_embedding_query,
            "latest_key_query": self.latest_key_query,
            "retrieval_candidate_asins": list(self.retrieval_candidate_asins),
            "retrieval_scores": dict(self.retrieval_scores),
            "retrieval_lane_sources": {
                asin: list(sources) for asin, sources in self.retrieval_lane_sources.items()
            },
            "eligible_count": self.eligible_count,
            "terminal_status": self.terminal_status,
            "new_checkpoints": [item.to_dict() for item in self._new_checkpoints_this_turn],
            "checkpoints": [item.to_dict() for item in self.checkpoints],
            "protocol_version": PROTOCOL_VERSION,
            "tool_calls_this_turn": self._drain_tool_log(),
        }

    def reset(self) -> None:
        """Wipe conversation state and start a fresh thread."""
        # Controlled simulations receive their category from the harness;
        # interactive adaptive sessions rediscover it after reset.
        self.category = self.category if self.elicitation_policy is not None else None
        self.bus = None
        self.qtool = None
        self.asks_so_far = 0
        self.asked_question_this_turn = False
        self.pending_question_text = None
        self.recommendation_finalized_this_turn = False
        self.recommendation_prose_raw = None
        self.visible_reply_to_sync = None
        self.recommendations = None
        self.visible_dialogue = []
        self.tool_log = []
        self.ledger = PreferenceLedger()
        self.question_ids = []
        self.recommendation_source = None
        self.recommendation_validation_error = None
        self.recommendation_selection_raw = None
        self.recommendation_product_numbers = []
        self.retrieval_query = None
        self.retrieval_key_query = None
        self.retrieval_query_raw = None
        self.latest_embedding_query = None
        self.latest_key_query = None
        self.retrieval_candidate_asins = []
        self.retrieval_scores = {}
        self.retrieval_lane_sources = {}
        self.eligible_count = None
        self.terminal_status = None
        self.checkpoints = []
        self._new_checkpoints_this_turn = []
        self._awaiting_answer_context = None
        self._pipeline = None
        self._agent = None
        self._checkpointer = None
        self._thread_id = uuid.uuid4().hex[:12]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_agent(self) -> Any:
        # `reasoning_effort` is a top-level kwarg on ChatOpenAI in
        # langchain-openai >= 0.3.x; passing via model_kwargs raises a warning.
        openai_base_client.INITIAL_RETRY_DELAY = 2.0
        openai_base_client.MAX_RETRY_DELAY = 64.0
        llm = ChatOpenAI(
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            use_responses_api=True,
            max_retries=OPENAI_MAX_RETRIES,
        )
        prompt = SYSTEM_PROMPT.format(
            available_categories=", ".join(list_available_categories()) or "(none indexed)"
        )
        if self.elicitation_policy is not None:
            prompt += _policy_prompt(self.elicitation_policy)
        self._checkpointer = InMemorySaver()
        return create_react_agent(
            model=llm,
            tools=self._make_tools(),
            prompt=prompt,
            checkpointer=self._checkpointer,
        )

    def _drain_tool_log(self) -> list[dict[str, Any]]:
        out, self.tool_log = self.tool_log, []
        return out

    def _record(self, name: str, args: dict[str, Any], result_summary: str) -> None:
        self.tool_log.append({"tool": name, "args": args, "result": result_summary})

    def _render_recommendation_reply(self) -> str:
        if not self.recommendations:
            return "I found a few options, but I need to refresh the recommendation set before showing them."
        lines = ["I recommend these three options:"]
        for rec in self.recommendations[:3]:
            price = rec.get("price")
            price_text = f"${price:.2f}" if isinstance(price, (int, float)) else "price unavailable"
            rating = rec.get("avg_rating")
            rating_text = f"{rating:.1f} stars" if isinstance(rating, (int, float)) else "no rating"
            bullets = rec.get("bullets") or []
            explanation = rec.get("recommendation_explanation")
            detail = explanation or (bullets[0] if bullets else rec.get("description") or "")
            detail = str(detail).strip() if explanation else str(detail)[:180].strip()
            line = f"{rec.get('rank')}. {rec.get('title')} — {price_text}, {rating_text}."
            if detail:
                line += f" {detail}"
            lines.append(line)
        return "\n\n".join(lines)

    def _ensure_pipeline(self) -> RecommendationPipeline:
        category = self._ensure_category()
        if self._pipeline is None or self._pipeline.category != category:
            self._pipeline = RecommendationPipeline(
                category=category,
                model=self.model,
                retrieval_limit=self.retrieval_limit,
            )
        return self._pipeline

    def _capture_hidden_checkpoint(
        self,
        *,
        asks_so_far: int,
        question_ids: list[str],
        query: str | None,
        key_query: str | None,
    ) -> None:
        snapshot = self._ensure_pipeline().prefix_snapshot(
            ledger=self.ledger,
            question_ids=question_ids,
            asks_so_far=asks_so_far,
            query=query,
            key_query=key_query,
        )
        self.checkpoints.append(snapshot)
        self._new_checkpoints_this_turn.append(snapshot)

    def _apply_snapshot(self, snapshot: RecommendationSnapshot) -> None:
        self.recommendations = snapshot.recommendations
        self.recommendation_source = snapshot.recommendation_source
        self.recommendation_validation_error = snapshot.recommendation_validation_error
        self.recommendation_selection_raw = snapshot.recommendation_selection_raw
        self.recommendation_prose_raw = snapshot.recommendation_prose_raw
        self.recommendation_product_numbers = list(snapshot.recommendation_product_numbers)
        self.retrieval_query = snapshot.retrieval_query
        self.retrieval_key_query = snapshot.retrieval_key_query
        self.retrieval_query_raw = snapshot.retrieval_query_raw
        self.retrieval_candidate_asins = list(snapshot.retrieval_candidate_asins)
        self.retrieval_scores = dict(snapshot.retrieval_scores)
        self.retrieval_lane_sources = {
            asin: list(sources) for asin, sources in snapshot.retrieval_lane_sources.items()
        }
        self.eligible_count = snapshot.eligible_count
        self.terminal_status = snapshot.terminal_status

    def _finalize_recommendations(self, query: str = "", key_query: str | None = None) -> str:
        pipeline = self._ensure_pipeline()
        if self.elicitation_policy is not None and self.elicitation_policy.name == "rec":
            snapshot = pipeline.default_snapshot(
                ledger=self.ledger,
                question_ids=self.question_ids,
                asks_so_far=self.asks_so_far,
            )
        else:
            snapshot = pipeline.select_for_query(
                ledger=self.ledger,
                question_ids=self.question_ids,
                asks_so_far=self.asks_so_far,
                query=query,
                key_query=key_query,
            )
        self._apply_snapshot(snapshot)
        self.recommendation_finalized_this_turn = snapshot.recommendations is not None
        if (
            self.elicitation_policy is not None
            and self.elicitation_policy.has_nonterminal_checkpoints
        ):
            self.checkpoints.append(snapshot)
            self._new_checkpoints_this_turn.append(snapshot)

        if snapshot.terminal_status == "NO_FEASIBLE_MATCH":
            self._record(
                "recommend",
                {"query": query, "key_query": key_query},
                "no feasible products after hard-budget eligibility",
            )
            return "NO_FEASIBLE_MATCH: no catalog product satisfies the explicit hard budget."

        details = snapshot.recommendations or []
        self._record(
            "recommend",
            {"query": query, "key_query": key_query},
            f"finalized {len(details)} products via {snapshot.recommendation_source}",
        )
        summary = "\n".join(
            f"  #{index + 1} ${product.get('price')} ★{product.get('avg_rating')} — "
            f"{str(product.get('title', ''))[:80]}"
            for index, product in enumerate(details)
        )
        return f"recommendation finalized ({len(details)} products):\n{summary}"

    def _force_next_question(self) -> str | None:
        try:
            qtool = self._ensure_qtool()
        except Exception:
            return None
        r = qtool.ask_fixed() if self.elicitation_policy is not None else qtool.ask(topic=None)
        question = r.get("question_text")
        if not question:
            return None
        self.asks_so_far += 1
        self.asked_question_this_turn = True
        self.pending_question_text = question
        self.question_ids.append(str(r["question_id"]))
        self._awaiting_answer_context = {
            "source": "answer",
            "question_id": str(r["question_id"]),
            "topic": str(r["topic"]),
            "question_text": str(question),
        }
        self._record("ask_question", {"topic": None, "forced": True}, f"tier={r['tier']} q={r['question_id']}")
        return question

    def _ensure_category(self) -> str:
        if not self.category:
            raise ValueError("category not set yet — call set_category first")
        return self.category

    def _ensure_bus(self) -> CandidateBus:
        cat = self._ensure_category()
        if self.bus is None:
            self.bus = CandidateBus.full(cat, list(load_catalog(cat)))
        return self.bus

    def _ensure_qtool(self) -> QuestionTool:
        cat = self._ensure_category()
        if self.qtool is None:
            config = load_config(cat)
            from pathlib import Path
            from sandbox.catalog import REPO_ROOT
            self.qtool = QuestionTool(bank=QuestionBank.load(REPO_ROOT / config["questions_path"]))
        return self.qtool

    # ------------------------------------------------------------------
    # Tool wrappers (closures over self)
    # ------------------------------------------------------------------

    def _make_tools(self) -> list:
        s = self

        @tool
        def check_category_supported(query: str) -> str:
            """Check whether the user's request matches any supported catalog category.
            Use this when the user's initial message is ambiguous about product type.
            Returns the best-matching category (if confident), a score, and the ranked alternatives."""
            r = _check_category_supported(query)
            s._record("check_category_supported", {"query": query},
                      f"best={r['best_category']} score={r['best_score']:.2f} supported={r['supported']}")
            return (
                f"best_category={r['best_category']}  score={r['best_score']:.3f}  "
                f"margin={r['margin_over_runner_up']:.3f}  supported={r['supported']}  "
                f"ranked={[(c, round(v, 3)) for c, v in r['ranked']]}"
            )

        @tool
        def set_category(category: str) -> str:
            """Set the working category for this conversation. Call this once after
            you've confirmed the customer is asking about a supported category.
            Initializes the Candidate Bus with all products in that category."""
            if category not in list_available_categories():
                return f"ERROR: '{category}' is not a supported category. Supported: {list_available_categories()}"
            if s.elicitation_policy is not None and s.category and category != s.category:
                return (
                    f"ERROR: this controlled run is fixed to category {s.category!r}; "
                    "do not change categories."
                )
            if s.category == category:
                bus = s._ensure_bus()
                s._record("set_category", {"category": category}, "category already set; no state reset")
                return (
                    f"category already set to {category}. Candidate bus has {bus.size()} products."
                )
            s.category = category
            s.bus = CandidateBus.full(category, list(load_catalog(category)))
            s.qtool = None  # reset question state for new category
            s._pipeline = None
            s._record("set_category", {"category": category}, f"bus initialized with {s.bus.size()} products")
            return f"category set to {category}. Candidate bus initialized with {s.bus.size()} products."

        @tool
        def catalog_overview_tool() -> str:
            """High-level snapshot of the current category's catalog: count, price range,
            top brands, common feature terms. Useful early in the conversation to know
            what the catalog actually offers."""
            ov = catalog_overview(s._ensure_category())
            s._record("catalog_overview", {}, f"n={ov['n_products']} ${ov['price_min']}–${ov['price_max']}")
            return (
                f"category={ov['category']}, n_products={ov['n_products']}\n"
                f"prices: ${ov['price_min']}–${ov['price_max']} (median ${ov['price_median']}, p75 ${ov['price_p75']})\n"
                f"top brands: {[b['brand'] for b in ov['top_brands'][:8]]}\n"
                f"common feature terms: {ov['common_feature_terms'][:12]}"
            )

        @tool
        def available_filters_tool() -> str:
            """List the filterable attributes for the current category and example values.
            Rarely needed. Call this only if the user gave one explicit hard
            constraint and you are considering a simple filter."""
            af = available_filters(s._ensure_category())
            top_fields = list(af["spec_fields"].items())[:10]
            field_summary = "\n".join(
                f"  - {k} (covers {info['coverage_pct']:.0f}%, examples: {info['example_values'][:3]})"
                for k, info in top_fields
            )
            s._record("available_filters", {}, f"{len(af['spec_fields'])} filterable fields")
            return (
                f"category={af['category']}, n_products={af['n_products']}\n"
                f"brands: {af['brands'][:10]}{'...' if len(af['brands']) > 10 else ''}\n"
                f"spec_fields:\n{field_summary}"
            )

        @tool
        def preview_filter(
            price_max: Optional[float] = None,
            price_min: Optional[float] = None,
            rating_min: Optional[float] = None,
            min_reviews: Optional[int] = None,
            brand_in: Optional[list] = None,
            brand_not_in: Optional[list] = None,
            spec_contains: Optional[dict] = None,
        ) -> str:
            """Preview an exceptional hard filter before applying it.
            Prefer using this only for simple explicit constraints, especially a
            budget ceiling. For preferences like RAM, storage, HEPA, portability,
            build quality, or quiet operation, use search/ranking instead."""
            constraints = {
                k: v for k, v in {
                    "price_max": price_max, "price_min": price_min,
                    "rating_min": rating_min, "min_reviews": min_reviews,
                    "brand_in": brand_in, "brand_not_in": brand_not_in,
                    "spec_contains": spec_contains,
                }.items() if v is not None
            }
            bus = s._ensure_bus()
            result = _preview_filter(bus, constraints)
            s._record(
                "preview_filter",
                constraints,
                f"{result['before']} → {result['after']} (no change)",
            )
            if result["after"] == 0:
                return (
                    f"preview only: {result['before']} → 0 products. "
                    f"Do not apply this filter as-is; it would empty the candidate bus. "
                    f"{result['note']}"
                )
            return (
                f"preview only: {result['before']} → {result['after']} products "
                f"({result['dropped']} would be dropped). Candidate bus unchanged. "
                f"{result['note']}"
            )

        @tool
        def filter_products(
            price_max: Optional[float] = None,
            price_min: Optional[float] = None,
            rating_min: Optional[float] = None,
            min_reviews: Optional[int] = None,
            brand_in: Optional[list] = None,
            brand_not_in: Optional[list] = None,
            spec_contains: Optional[dict] = None,
        ) -> str:
            """Remove products for one simple explicit hard constraint.
            Use this mainly for clear budget constraints like 'under $1000'.
            Rank/search instead for product attributes and softer preferences.
            spec_contains is a dict mapping a spec-table field name to a substring
            that the value must contain.
            Avoid spec_contains unless the user made the spec an explicit
            non-negotiable and preview shows a broad candidate set remains."""
            constraints = {
                k: v for k, v in {
                    "price_max": price_max, "price_min": price_min,
                    "rating_min": rating_min, "min_reviews": min_reviews,
                    "brand_in": brand_in, "brand_not_in": brand_not_in,
                    "spec_contains": spec_contains,
                }.items() if v is not None
            }
            bus = s._ensure_bus()
            before = bus.size()
            apply_filter(bus, constraints)
            s._record("filter_products", constraints, f"{before} → {bus.size()}")
            if bus.size() == 0:
                return f"WARNING: no products survived. {bus.notes[-1]}. Consider relaxing constraints."
            return f"filtered: {before} → {bus.size()} products. {bus.notes[-1]}"

        @tool
        def reset_bus_to_full_catalog() -> str:
            """Discard all filters and ranking on the current bus and restart with
            the entire category catalog. Use this when the current candidate set
            looks too narrow or stale and you want to rebuild from scratch."""
            cat = s._ensure_category()
            s.bus = CandidateBus.full(cat, list(load_catalog(cat)))
            s._record("reset_bus_to_full_catalog", {}, f"bus reset to {s.bus.size()} products")
            return f"bus reset to full catalog: {s.bus.size()} products."

        @tool
        def semantic_search_full(query: str, key_query: str = "", top_k: int = 30) -> str:
            """Score the FULL catalog (not just current bus) by semantic similarity to
            a natural-language description of what the customer wants. Replaces the
            bus contents with the top-K. The key query is a short phrase for the
            customer's most important revealed requirement and is saved for final
            multi-lane retrieval."""
            bus = s._ensure_bus()
            semantic_search(bus, query=query, top_k=top_k, full_catalog=True)
            s.latest_embedding_query = query.strip() or s.latest_embedding_query
            s.latest_key_query = key_query.strip() or s.latest_key_query
            s._record(
                "semantic_search_full",
                {"query": query, "key_query": key_query, "top_k": top_k},
                f"bus={bus.size()}",
            )
            return f"bus refreshed: top {bus.size()} products matching '{query}'."

        @tool
        def narrow_search_tool(query: str, top_k: int = 15) -> str:
            """Re-rank ONLY the current bus contents by semantic similarity to a query.
            Does not bring back products outside the current bus. Use only when
            the current bus is already a good broad candidate set."""
            bus = s._ensure_bus()
            before = bus.size()
            narrow_search(bus, query=query, top_k=top_k)
            s.latest_embedding_query = query.strip() or s.latest_embedding_query
            s._record("narrow_search", {"query": query, "top_k": top_k}, f"{before} → {bus.size()}")
            return f"bus re-ranked within {before} → kept top {bus.size()}."

        @tool
        def rank_by_match_tool(semantic: float = 1.0, rating: float = 0.3, popularity: float = 0.1) -> str:
            """Reorder the bus by a weighted combination of semantic similarity (from
            most recent search), rating, and popularity. Default weights are sensible."""
            bus = s._ensure_bus()
            rank_by_match(bus, MatchWeights(semantic=semantic, rating=rating, popularity=popularity))
            s._record("rank_by_match", {"sem": semantic, "rating": rating, "pop": popularity},
                      f"reranked {bus.size()}")
            return f"bus reranked. {bus.notes[-1]}"

        @tool
        def rank_by_commission_tool(budget_max: Optional[float] = None) -> str:
            """Reorder the bus by expected commission (price × P(purchase)). Surfaces
            higher-priced items the user is still likely to buy. Pass the customer's
            elicited budget to apply the in-budget boost — items above it are penalized."""
            bus = s._ensure_bus()
            rank_by_commission(bus, budget_max=budget_max)
            s._record("rank_by_commission", {"budget_max": budget_max}, f"reranked {bus.size()}")
            return f"bus reranked by expected commission. {bus.notes[-1]}"

        @tool
        def rank_by_price_tool(ascending: bool = True) -> str:
            """Sort the bus by price (ascending by default)."""
            bus = s._ensure_bus()
            rank_by_price(bus, ascending=ascending)
            return f"bus sorted by price ({'asc' if ascending else 'desc'})."

        @tool
        def get_product_details_tool(asin: str) -> str:
            """Full structured details on one specific product."""
            d = get_product_details(s._ensure_category(), asin)
            if d is None:
                return f"ERROR: ASIN {asin} not found."
            s._record("get_product_details", {"asin": asin}, d.get("title", "")[:60])
            return (
                f"asin={d['asin']}  title={d['title']!r}\n"
                f"brand={d['brand']}  price=${d['price']}  rating={d['avg_rating']} ({d['num_reviews']} revs)\n"
                f"bullets: {d['bullets'][:6]}\n"
                f"key specs: {dict(list((d['spec_table'] or {}).items())[:10])}\n"
                f"review excerpts: {d['review_excerpts'][:2]}"
            )

        @tool
        def compare_products(asins: list, aspects: Optional[list] = None) -> str:
            """Side-by-side comparison of 2–4 products on selected aspects.
            aspects can include 'title', 'brand', 'price', 'avg_rating', and any
            spec-table key (e.g., 'Screen Size', 'RAM Memory Installed')."""
            c = compare(s._ensure_category(), asins, aspects=aspects)
            s._record("compare", {"asins": asins, "aspects": aspects}, f"{len(c['asins'])} products")
            lines = [f"compared {len(c['asins'])} products on {len(c['aspects'])} aspects:"]
            for asp, row in c["table"].items():
                cells = " | ".join(str(v)[:40] if v is not None else "—" for v in row)
                lines.append(f"  {asp}: {cells}")
            return "\n".join(lines)

        @tool
        def compute_uncertainty_tool(top_k: int = 10) -> str:
            """Compute entropy of retrieval scores + attribute diversity over the top-K
            of the bus. Low entropy / clear top item = ready to recommend.
            High entropy = ask another clarifying question."""
            bus = s._ensure_bus()
            sig = compute_uncertainty(bus, top_k=top_k)
            s._record("compute_uncertainty", {"top_k": top_k},
                      f"H_norm={sig.entropy_normalized:.2f} gap={sig.score_gap:.2f}")
            return (
                f"top_k_size={sig.top_k_size}, entropy_normalized={sig.entropy_normalized:.3f}\n"
                f"score_gap (top1 vs top_K)={sig.score_gap:.3f}\n"
                f"split_attributes (high diversity = ambiguous): {sig.top_k_diversity}"
            )

        @tool
        def suggest_next_action_tool() -> str:
            """Composite recommendation: ASK, RECOMMEND, or KEEP_ASKING. Based on
            entropy of the current bus + ask budget. Use as a soft guide; you can override."""
            thresholds = None
            if s.elicitation_policy is not None:
                target = s.elicitation_policy.target_asks
                if s.elicitation_policy.name == "rec":
                    thresholds = ActionThresholds(min_asks_before_recommend=0, max_asks=0)
                elif s.elicitation_policy.allows_early_recommendations:
                    thresholds = ActionThresholds(min_asks_before_recommend=0, max_asks=target)
                else:
                    thresholds = ActionThresholds(min_asks_before_recommend=target, max_asks=target)
            r = suggest_next_action(s._ensure_bus(), asks_so_far=s.asks_so_far, thresholds=thresholds)
            s._record("suggest_next_action", {}, f"{r['action']}: {r['reason']}")
            return f"suggested={r['action']}  reason={r['reason']}  signals={r['signals']}"

        @tool
        def ask_question_tool(topic: Optional[str] = None) -> str:
            """Get the next clarifying question to ask the customer. Without `topic`,
            returns the next opener question (or any unasked followup if openers are
            done). With a `topic`, returns a followup on that topic.
            You must ASK this question to the customer — the tool returns the text.
            You must use this tool to ask any question, you cannot use it more than once in a turn."""
            if s.asked_question_this_turn:
                msg = (
                    "You have already called ask_question() this turn. Send the previous question "
                    "to the buyer; you may ask again once the buyer answers."
                )
                s._record("ask_question", {"topic": topic}, msg)
                return msg

            if s.elicitation_policy is not None and s.asks_so_far >= s.elicitation_policy.target_asks:
                s._record(
                    "ask_question",
                    {"topic": topic},
                    f"policy limit reached at {s.asks_so_far}/{s.elicitation_policy.target_asks}",
                )
                return (
                    "POLICY_LIMIT_REACHED: do not ask another clarifying question. "
                    "Recommend now using the current conversation and candidate set."
                )
            qtool = s._ensure_qtool()
            r = qtool.ask_fixed() if s.elicitation_policy is not None else qtool.ask(topic=topic)
            if r.get("question_text"):
                s.asks_so_far += 1
                s.asked_question_this_turn = True
                s.pending_question_text = r["question_text"]
                s.question_ids.append(str(r["question_id"]))
                s._awaiting_answer_context = {
                    "source": "answer",
                    "question_id": str(r["question_id"]),
                    "topic": str(r["topic"]),
                    "question_text": str(r["question_text"]),
                }
                s._record("ask_question", {"topic": topic},
                          f"tier={r['tier']} q={r['question_id']}")
                return (
                    f"NEXT QUESTION TO RELAY (tier={r['tier']}, id={r['question_id']}): "
                    f"{r['question_text']}\n"
                    f"remaining_openers={r['remaining_openers']}, "
                    f"uncovered_topics={r['uncovered_topics'][:5]}"
                )
            return f"no more questions available. uncovered_topics={r.get('uncovered_topics', [])}"

        @tool
        def recommend(query: str = "", key_query: str = "") -> str:
            """Retrieve 15 products, then finalize three. Query should cover all
            revealed preferences; key_query should focus only on the most important
            revealed requirement."""
            if s.recommendation_finalized_this_turn or s.recommendations:
                return (
                    "RECOMMENDATION_ALREADY_FINALIZED: use the fixed products already returned; "
                    "do not call recommend() again or change their order."
                )
            if (
                s.elicitation_policy is not None
                and not s.elicitation_policy.allows_early_recommendations
                and s.asks_so_far < s.elicitation_policy.target_asks
            ):
                remaining = s.elicitation_policy.target_asks - s.asks_so_far
                s._record(
                    "recommend",
                    {"query": query, "key_query": key_query},
                    f"blocked by policy: {s.asks_so_far}/{s.elicitation_policy.target_asks} asks",
                )
                return (
                    f"POLICY_REQUIRES_MORE_QUESTIONS: ask {remaining} more "
                    f"clarifying question(s) before recommending."
                )

            if s.asked_question_this_turn and s.pending_question_text:
                s._record(
                    "recommend",
                    {"query": query, "key_query": key_query},
                    "blocked: already asked a question this turn",
                )
                return (
                    "ASK_ALREADY_FINALIZED_THIS_TURN: send the question you just received "
                    "from ask_question() to the buyer. Recommend only after the buyer answers."
                )

            if s.elicitation_policy is None or s.elicitation_policy.name != "rec":
                if not query.strip():
                    return "QUERY_REQUIRED: call recommend with a concise embedding query."
                s.latest_embedding_query = query.strip()
                s.latest_key_query = key_query.strip() or s.latest_key_query or query.strip()
            return s._finalize_recommendations(
                query=query,
                key_query=key_query.strip() or s.latest_key_query or query.strip(),
            )

        return [
            check_category_supported,
            set_category,
            catalog_overview_tool,
            available_filters_tool,
            preview_filter,
            filter_products,
            reset_bus_to_full_catalog,
            semantic_search_full,
            narrow_search_tool,
            rank_by_match_tool,
            rank_by_commission_tool,
            rank_by_price_tool,
            get_product_details_tool,
            compare_products,
            compute_uncertainty_tool,
            suggest_next_action_tool,
            ask_question_tool,
            recommend,
        ]
