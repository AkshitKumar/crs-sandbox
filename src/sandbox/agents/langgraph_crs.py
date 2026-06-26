"""LangGraph-based tool-using CRS agent.

The agent is a ReAct-style loop wrapped by `langgraph.prebuilt.create_react_agent`.
Each conversation lives in a `CRSAgentSession` that holds:
    - a CandidateBus     (mutated by tools)
    - a QuestionTool     (tracks which openers/followups have been asked)
    - an asks_so_far     counter
    - a dialogue ledger  (for inspection / chat-UI rendering)

Tools are defined inside `_make_tools` as closures over the session, so the
LLM doesn't need to pass the bus around in arguments — it just calls
`filter_products(price_max=1100)` and the session updates its bus.

When the agent decides to recommend, it calls `recommend(top_k)` which freezes
the current bus's top-K into `session.recommendations` for the chat UI to
render as cards.
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
from sandbox.tools.review_tool import summarize_reviews
from sandbox.tools.search_tool import narrow_search, semantic_search
from sandbox.tools.uncertainty_tool import compute_uncertainty, suggest_next_action


DEFAULT_MODEL = "gpt-5-mini"
DEFAULT_REASONING = "medium"
OPENAI_MAX_RETRIES = 7


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

You are working with a hidden "Candidate Bus" — the set of products you're
considering. Tools restrict, rerank, or inspect the bus; the bus persists
across tool calls within this conversation. You do not need to track it
yourself; each tool's return value reports the current bus size.

Typical flow (use judgment, this is not a script or a checklist):

  1. If the customer's initial message is ambiguous about the product type,
     call `check_category_supported` first. If supported, then set the category
     by calling `set_category`. From then on every tool acts on that category.
  2. Call `catalog_overview` and/or `available_filters` once early to know
     what the catalog actually contains (price range, brands, filterable specs).
  3. Ask the FIRST opener question via `ask_question()` (no topic argument).
     Always ask the openers in their canonical order — they cover the broadest
     attributes (use case, budget, form factor). Then, use ask_question()
     whenever you ask follow-up questions.
  4. After each customer answer, decide: ASK or RECOMMEND.
     - If you decide to ASK, you should ask only one question at a time using
       `ask_question`. 
     - Call `compute_uncertainty` to get an entropy/diversity reading on the
       current bus. Low entropy = ready to recommend. High entropy = ask more.
     - You can also call `suggest_next_action` which composites this for you.
  5. Use `filter_products` for HARD constraints (price ceiling, must-have
     features, brand preferences). Don't filter too aggressively --- some 
     preferences are less important, and should be used for ranking with
     narrow_search within the filtered set instead. Use 'preview_filter' 
     before applying any restrictive or uncertain filter to see how many
     products remain. If after filtering the bus has fewer than 5 products, 
     either ask the customer to confirm an inferred preference (turning it 
     into a stated one), or relax the most recent filter.
     
     CRITICAL: Filter ONLY on attributes the customer has explicitly stated
     that are important for their preference. If they say "under $1000", 
     filter ONLY on price — do NOT also add "dedicated GPU", "16GB RAM", or 
     any other constraint you inferred. Each filter strips items; stacking 
     inferred filters quickly leaves the bus too small to make a useful 
     recommendation. If the user provides many constraints, you should only 
     filter on some of these and incorporate others into a semantic or narrow 
     search query to re-rank products rather than filtering.
     
  6. Use `semantic_search` to seed the bus from the entire catalog using a 
     natural-language description of what the user wants. Use `narrow_search` 
     to re-rank within the current bus contents after a filter.
  7. Use `rank_by_match` for general "best match" ordering.
     Use `rank_by_commission(budget_max=...)` when you've been told to
     optimize for higher-revenue recommendations; pass the customer's budget.
  8. Before recommending, optionally use `summarize_reviews` on the top 1–2
     candidates to enrich your pitch.
  9. When ready, call `recommend(top_k=3)` to finalize. Return a friendly
     message explaining why each item fits the customer's needs.

# Style

- Conversational, concise. Ask one question at a time — never stack multiple
  questions or ask_question() calls in one turn.
- Don't repeat back the user's words verbatim. Acknowledge briefly and ask.
- Be thoughtful when using tools; prefer using fewer tools over many. 
- Only your final no-tool assistant message for the turn is shown to the customer.
  It must be a complete customer-facing response.
- NEVER reveal you're using tools or anything about the internal mechanics;
  keep the conversation focused on the user and their preferences. 

# Failure modes to avoid

- Avoid treating all stated preferences as hard constraints --- use extra preference
  information as a ranking signal rather than filtering everything if they are not
  the main priorities. 
- If you filter, keep filters simple --- avoid stacking filters on many parts of 
  the product to avoid overly shrinking the candidate bus. Use preview_filter() 
  to avoid loops of trying filters. 
- If you are in a repetitive loop of using tools, ask a question with ask_question(). 
- Avoid recommending without elicitation when the user has only said vague things.
- Avoid recommending products that are not explicitly what recommend() returns. 
  You must use recommend() and the products returned by recommend() whenever you 
  make a recommendation.
- Avoid asking too many questions when the bus is already concentrated. Trust low-entropy
  signals — once the candidate set has clearly converged, recommend.
- Avoid inventing product attributes you didn't see in tool output.
- If the catalog does not have a product/exact match for the user, acknowledge this
  and recommend the nearest possible products within the catalog. Do not ask the 
  user for ideas, just relax some restrictions and try to find the best match available.

"""


def _policy_prompt(policy: ElicitationPolicy) -> str:
    if policy.name == "rec":
        return """

# Fixed elicitation policy for this run

You are immediately recommending products to the user; ask zero clarifying 
questions. Use the customer's initial request to search, filter, rank, and 
recommend immediately.
"""

    if policy.name == "atr_recs":
        return f"""

# Internal question budget for this run

Use an internal question budget of {policy.target_asks}. Ask at most that many
clarifying questions before ending the conversation. You may recommend before
the budget is exhausted, and early recommendations without a purchase do not
end the conversation. Once the budget is exhausted, make a final recommendation
and end. Your goal is to provide the best possible recommendation with the
information gathered within the question budget. Do not mention the question
budget to the customer, what question you are on, etc. to the user. 
"""

    return f"""

# Internal question budget for this run

Use an internal question budget of {policy.target_asks}. Spend the full budget
on one-at-a-time clarifying questions before recommending. Once the budget is
exhausted, you must recommend with recommend(). The ask_question() and 
recommend() tools will return with an indication to recommend at the correct 
turn. Your goal is to provide the best possible recommendation with the 
information gathered within the question budget. Do not mention the question 
budget to the customer, what question you are on, etc. to the user. 
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
    recommendations: Optional[list] = None
    tool_log: list = field(default_factory=list)
    model: str = DEFAULT_MODEL
    reasoning_effort: str = DEFAULT_REASONING
    elicitation_policy: Optional[ElicitationPolicy] = None

    # The compiled LangGraph agent (lazily built once self exists).
    _agent: Any = None
    _checkpointer: Any = None
    _thread_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    # ------------------------------------------------------------------
    # Public surface
    # ------------------------------------------------------------------

    def chat(self, user_message: str, max_steps: int = 90) -> dict[str, Any]:
        """Send one user turn through the agent. Returns the reply + audit info.

        `max_steps` is LangGraph's recursion_limit: roughly each tool call
        counts as 2 steps (agent decides → tool runs → agent reads result).
        90 → ~45 tool calls per turn, generous for our 17-tool taxonomy.

        If the agent hits the limit, we still return whatever tool calls
        happened plus a fallback message so the chat doesn't break.
        """
        if self._agent is None:
            self._agent = self._build_agent()

        self.asked_question_this_turn = False
        before_recs = self.recommendations
        config = {
            "configurable": {"thread_id": self._thread_id},
            "recursion_limit": max_steps,
        }
        try:
            result = self._agent.invoke(
                {"messages": [{"role": "user", "content": user_message}]},
                config=config,
            )
            reply = result["messages"][-1].content
        except GraphRecursionError:
            reply = (
                "Sorry — I got a bit tangled up trying to answer that. "
                "Let me try a simpler approach. Could you restate what you're looking for "
                "in one or two sentences?"
            )
        new_recs = self.recommendations if self.recommendations is not before_recs else None
        return {
            "reply": reply,
            "recommendations": new_recs,
            "bus_size": self.bus.size() if self.bus else None,
            "asks_so_far": self.asks_so_far,
            "category": self.category,
            "tool_calls_this_turn": self._drain_tool_log(),
        }

    def reset(self) -> None:
        """Wipe conversation state and start a fresh thread."""
        self.category = None
        self.bus = None
        self.qtool = None
        self.asks_so_far = 0
        self.asked_question_this_turn = False
        self.recommendations = None
        self.tool_log = []
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
            s.category = category
            s.bus = CandidateBus.full(category, list(load_catalog(category)))
            s.qtool = None  # reset question state for new category
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
            Call this before `filter_products` if you're not sure what to constrain on."""
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
            """Preview HARD constraints without changing the candidate bus.
            Use this before `filter_products` when a constraint may be too
            restrictive. The schema is identical to `filter_products`."""
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
            """Apply HARD constraints to the candidate bus. Use this when the user
            states a non-negotiable like 'under $1000' or 'has to have HEPA filter'.
            spec_contains is a dict mapping a spec-table field name to a substring
            that the value must contain, e.g. {"Graphics Description": "Dedicated"};
            you can use | to indicate OR conditions for the values, e.g. {"Processor": "Intel|AMD"}.
            Call `available_filters` first to see what fields exist."""
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
            the entire category catalog. Use this when a previous filter chain was
            too restrictive and you want to rebuild from scratch."""
            cat = s._ensure_category()
            s.bus = CandidateBus.full(cat, list(load_catalog(cat)))
            s._record("reset_bus_to_full_catalog", {}, f"bus reset to {s.bus.size()} products")
            return f"bus reset to full catalog: {s.bus.size()} products."

        @tool
        def semantic_search_full(query: str, top_k: int = 30) -> str:
            """Score the FULL catalog (not just current bus) by semantic similarity to
            a natural-language description of what the customer wants. Replaces the
            bus contents with the top-K. Use early in the conversation to seed the bus
            with semantically relevant products, OR after a filter removed too much."""
            bus = s._ensure_bus()
            semantic_search(bus, query=query, top_k=top_k, full_catalog=True)
            s._record("semantic_search_full", {"query": query, "top_k": top_k}, f"bus={bus.size()}")
            return f"bus refreshed: top {bus.size()} products matching '{query}'."

        @tool
        def narrow_search_tool(query: str, top_k: int = 15) -> str:
            """Re-rank ONLY the current bus contents by semantic similarity to a query.
            Does not bring back filtered-out products. Use after a filter call to
            sharpen ranking within what survived."""
            bus = s._ensure_bus()
            before = bus.size()
            narrow_search(bus, query=query, top_k=top_k)
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
        def summarize_reviews_tool(asin: str, aspect: Optional[str] = None) -> str:
            """Cached LLM-generated summary of what reviewers say about a product.
            Pass an `aspect` (e.g., 'battery life', 'noise level') to focus the summary."""
            r = summarize_reviews(s._ensure_category(), asin, aspect=aspect)
            s._record("summarize_reviews", {"asin": asin, "aspect": aspect},
                      f"sentiment={r.get('sentiment_score')}")
            return f"sentiment={r['sentiment_score']}\nsummary: {r['summary']}\nevidence: {r['evidence']}"

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
            r = suggest_next_action(s._ensure_bus(), asks_so_far=s.asks_so_far)
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
            r = qtool.ask(topic=topic)
            if r.get("question_text"):
                s.asks_so_far += 1
                s.asked_question_this_turn = True
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
        def recommend(top_k: int = 3, justification: str = "") -> str:
            """FINALIZE the recommendation by taking the top-K of the current bus.
            After calling this you should write a customer-facing message explaining
            why each item fits. The chat UI renders the recommendation cards from
            the bus's top-K automatically. You must use this tool if you are recommending
            any products.

            IMPORTANT: if the bus has fewer than `top_k` products, this tool will
            REFUSE and tell you to widen first. Do not work around this — the
            customer wants choices, not a single forced result. Common ways to
            widen: relax the tightest filter, or call semantic_search_full again
            with a slightly broader query.

            `justification`: a short note about why these items were chosen."""
            if (
                s.elicitation_policy is not None
                and not s.elicitation_policy.allows_early_recommendations
                and s.asks_so_far < s.elicitation_policy.target_asks
            ):
                remaining = s.elicitation_policy.target_asks - s.asks_so_far
                s._record(
                    "recommend",
                    {"top_k": top_k, "justification": justification},
                    f"blocked by policy: {s.asks_so_far}/{s.elicitation_policy.target_asks} asks",
                )
                return (
                    f"POLICY_REQUIRES_MORE_QUESTIONS: ask {remaining} more "
                    f"clarifying question(s) before recommending."
                )

            bus = s._ensure_bus()
            available = bus.size()

            # Guard: don't ship a thin recommendation.
            if available < top_k:
                return (
                    f"REFUSED: bus has only {available} product(s), need at least {top_k}. "
                    f"Do NOT recommend yet — widen the candidate set first. "
                    f"Recent bus operations: {bus.notes[-3:]}. "
                    f"Suggestions: (1) relax the most recent filter (e.g., raise price_max, "
                    f"drop a brand/spec restriction), (2) re-run semantic_search_full with a "
                    f"broader query, then narrow back with rank_by_match."
                )

            top = bus.top(top_k)
            details = [get_product_details(s._ensure_category(), a) for a in top]
            details = [d for d in details if d]
            s.recommendations = [{
                "rank": i + 1, **d
            } for i, d in enumerate(details)]
            s._record("recommend", {"top_k": top_k, "justification": justification},
                      f"finalized {len(details)} products from bus of {available}")
            summary = "\n".join(
                f"  #{i+1} ${d['price']} ★{d['avg_rating']} — {d['title'][:80]}"
                for i, d in enumerate(details)
            )
            return f"recommendation finalized ({len(details)} products):\n{summary}"

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
            summarize_reviews_tool,
            compute_uncertainty_tool,
            suggest_next_action_tool,
            ask_question_tool,
            recommend,
        ]
