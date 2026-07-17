# crs-sandbox — Project Plan

Living document. Edit / annotate / push back. Updated as we make decisions.

---

## 1. What we are building

A **good agentic conversational recommender system** — the kind of thing Amazon Rufus aspires to be — that can:

- Hold a multi-turn conversation about what a shopper wants.
- Use a rich toolbox (search, filter, rank, inspect, compare, summarize reviews, gauge its own uncertainty) to find good products in a real catalog.
- **Endogenously decide** when to ask a clarifying question vs when to recommend, based on how uncertain it is about the user's preferences and what's in the catalog.
- Adapt its ranking to platform-side objectives (consumer match vs commission vs conversion) without exposing those mechanics to the user.

## 2. How we evaluate it

Two evaluation modes, built in this order:

1. **Human-in-the-loop chat UI** (built first). A small browser-based chat where a human user (you) talks to the agent in real time, sees its tool calls, candidate sets, entropy values, and final recommendations. This is for *qualitative debugging* — catching failure modes, weird tool-use, bad questions, etc.
2. **LLM buyer simulator** (built second). The persona-conditioned buyer LLM we already have. This is the *scaled quantitative* evaluation — runs thousands of conversations across personas/categories, produces aggregate metrics. Replaces costly human-subject recruitment.

If we want to replicate or extend the Kumar–Manshadi–Tumu experiments, that becomes possible once both eval modes are in place. It's a side benefit, not the goal.

## 3. What "agentic" means here

Borrowed from Microsoft InteRecAgent's blueprint, with extensions inspired by Rufus and the entropy-policy paper (Frey et al. 2025).

The agent is an LLM that **orchestrates tools**. It doesn't see the catalog directly — it only sees what tools return. Tools share a **Candidate Bus**: a per-conversation in-memory candidate set that flows between tool calls, so the LLM doesn't have to keep large lists in its prompt.

The full tool taxonomy is in §6 below.

## 4. The endogenous ask-vs-recommend gate

This is the central design question — the agent needs to decide each turn whether it has enough info to recommend or should ask another question.

**Approach**: convert text into a numerical signal via the **entropy of retrieval scores**.

After each user message, the agent silently runs `semantic_search(running_query, top_k=20)`, gets back the cosine similarities, softmaxes them with a temperature, computes entropy:

```
H = -Σ p_i * log(p_i)
  where p_i = exp(score_i / τ) / Σ exp(score_j / τ)
```

Two thresholds split the action space:

- `H < H_recommend`: a few candidates clearly dominate → **RECOMMEND**
- `H > H_ask`: many candidates score similarly → ambiguous → **ASK**
- in between: LLM judgment call

Layered on top of entropy:
- **Attribute coverage**: track which user-need attributes have been elicited. If a known-important attribute (e.g., "intended use") is still unspecified, prefer ASK.
- **Turn budget**: soft penalty as turn count grows, to model realistic impatience.
- **Diversity in the top-K**: if the top-K candidates split on a single attribute, that's the question to ask next.

The exact thresholds need calibration against real conversations — this is what the human chat mode helps tune.

## 5. Current state (as of 2026-06-07)

| Component | State | Path |
|---|---|---|
| Scraper — ScraperAPI (httpx async, concurrent) | ✅ | `src/sandbox/scraper/fast.py` |
| Scraper — Playwright (residential fallback) | ✅ | `src/sandbox/scraper/amazon.py` |
| Spec-table parsing (median 60 fields/laptop, 26/air_purifier) | ✅ | `parse_pdp` in `amazon.py` |
| Laptop catalog | ✅ 167 products | `data/categories/laptop/products.jsonl` |
| Air purifier catalog | ✅ 100 products | `data/categories/air_purifier/products.jsonl` |
| Question banks | ✅ both pilot categories | `data/categories/<cat>/questions.yaml` |
| Personas (self-contained) | ✅ 100 each | `data/categories/<cat>/personas.json` |
| Index builder (bge embeddings + per-cat centroid) | ✅ | `src/sandbox/index/builder.py` |
| Shared dense top-15 + bounded LLM reranking | ✅ | `src/sandbox/agents/recommendation_pipeline.py` |
| Laptop + air_purifier indices | ✅ built | `data/categories/<cat>/index/` |
| Tool layer (10 modules) | ✅ | `src/sandbox/tools/` |
| Buyer agent (LLM, multi-turn-aware) | ✅ | `src/sandbox/agents/buyer.py` |
| rufus-femto agent (LangGraph ReAct, 18 tool wrappers) | ✅ | `src/sandbox/agents/langgraph_crs.py` |
| Sim orchestrator (buyer ↔ rufus-femto, `run` + `run_iter`) | ✅ | `src/sandbox/orchestrator/sim_conversation.py` |
| Streamlit · human chat | ✅ | `scripts/chat.py` (port 8501) |
| Streamlit · live buyer×CRS sim | ✅ | `scripts/live_sim.py` (port 8503) |
| Streamlit · transcript browser | ✅ | `scripts/visualize_eval.py` (port 8502) |
| Batch eval CLI | ✅ | `scripts/run_buyer_eval.py` |
| Terminal REPL | ✅ | `scripts/repl.py` |

**What's worth iterating on next (open):**

- Calibration of `compute_uncertainty` thresholds against more real chat sessions.
- Re-fitting `rank_by_commission`'s P(purchase) coefficients from buyer-simulator outcome data once we have it at scale (a few hundred conversations).
- Surfacing the entropy signal in the live UI so you can watch the ask-vs-recommend gate fire in real time.
- More categories — the embedding-based feasibility fallback gets stronger margins as the catalog of categories grows.
- A "diff against ground truth" panel that highlights which persona attributes were elicited vs stayed hidden.

### Controlled-evaluation ranking note (2026-07-13)

The recommender no longer uses category-specific evidence weights. `REC` and
checkpoint 0 use the curated three-product slate declared in the category
config. `CRSAgentSession` is the conversational agent in every arm. For a
terminal recommendation it supplies a natural-language embedding query; the
shared pipeline uses the pinned BGE index to retrieve 15 and one constrained
`gpt-5-mini` call chooses three structured cards and briefly explains each.
Hidden checkpoints reuse the latest query the ReAct agent actually supplied,
without a separate query-normalization call. Lightweight cross-encoders
were rejected after whole-catalog checks exposed SEO-heavy rankings. Only
explicit hard budgets are structurally enforced; other responses remain soft
preferences. Controlled questions use fixed YAML order and neutral wording. The buyer reports
uncertainty rather than inventing a requirement when its persona is silent.

## 6. Tool layer specification

This is the contract for what each tool does and how it's implemented. The CRS agent calls these via OpenAI tool-calling.

### Discovery / feasibility

#### `check_category_supported(query: str) -> {supported: bool, suggested: list[str]}`
- **What**: Does our catalog cover the kind of product the user is asking about?
- **How**: Embed the query, find the nearest catalog centroid across all 40 categories, return the category if cosine sim > 0.5; otherwise return top-3 suggested categories with similarities.
- **Why**: Lets the agent say "Sorry, I don't carry power tools" instead of making up recommendations from a wrong category.
- **Stored data needed**: One per-category centroid embedding, precomputed from product embeddings.

#### `catalog_overview(category: str) -> {n_products, price_min/median/max, top_brands, common_features}`
- **What**: High-level snapshot of what's in the catalog.
- **How**: Precomputed at index-build time and persisted as `data/categories/<cat>/index/overview.json`. No LLM call needed at runtime.
- **Why**: Helps the agent calibrate expectations — if the user says "I want it under $200" and the catalog median is $800, the agent knows to surface that constraint early.

### Filtering (hard constraints)

#### `filter(category: str, constraints: dict) -> CandidateBus`
- **What**: Apply hard constraints to narrow the catalog.
- **Constraints schema**: `{price_max, price_min, must_have_features: [str], must_not_have_features: [str], brand_in: [str], rating_min, min_reviews}`
- **How**: Iterate over catalog rows, apply each constraint, return the surviving subset as the Candidate Bus.
- **Implementation detail**: `features` aren't free-text — they're matched against a **per-product attribute set** that we precompute once.

#### `available_filters(category: str) -> {price_range, brands, features, ...}`
- **What**: What filterable attributes exist for this category, and what values they take.
- **How**: Static metadata derived once at index build.

### Retrieval (soft, semantic)

#### `semantic_search(query: str, top_k: int = 20, restrict_to: CandidateBus | None = None) -> CandidateBus`
- **What**: bge embedding + cosine similarity over the (optionally filtered) catalog.
- **How**: This is what `ProductSearchTool.search` already does. Just renamed and refactored.

### Ranking

#### `rank_by_match(bus: CandidateBus, weights: dict) -> CandidateBus`
- **What**: Re-rank within the bus using weighted attributes (semantic similarity, rating, popularity).
- **How**: Already implemented inside `ProductSearchTool` — pull it out as a standalone callable.

#### `rank_by_commission(bus: CandidateBus) -> CandidateBus`
- **What**: Rank by `expected_commission = price × P(purchase | persona, item)`.
- **How — computing P(purchase)**:
  1. **v0 (immediate)**: a calibrated logistic on observable features:
     ```
     P_purchase = sigmoid(
         β_0
       + β_1 * semantic_sim
       + β_2 * z_score(rating)
       + β_3 * z_score(log_reviews)
       - β_4 * z_score(price)
       + β_5 * (price <= elicited_max_budget ? 1 : 0)
     )
     ```
     We pick the `β`s by hand at first (e.g., `β_1=2, β_4=1, β_5=2`), then re-fit them from observed simulator outcomes once we have data.
  2. **v1 (better)**: few-shot LLM-as-judge with the elicited persona + product summary, asking for `P(purchase)` on a 0–1 scale. More expensive but better-calibrated.
  3. **v2 (best, far-future)**: a fine-tuned small model on collected `(persona, product, purchased?)` pairs from buyer-simulator runs.
- **Why this matters**: this is exactly the mechanism the paper's "commission objective" formalizes. Here, the agent surfaces high-price items the user is likely to actually buy — boosting expected commission, possibly at the cost of pure match quality.

### Inspection

#### `get_product_details(asin: str) -> ProductRecord`
- **What**: Full structured record for one product.
- **How**: Trivial lookup in the catalog JSONL.

#### `compare(asins: list[str], aspects: list[str]) -> ComparisonTable`
- **What**: Side-by-side attribute table for 2–4 products on specified aspects.
- **How**: For each (asin, aspect), pull from the structured-spec table if scraped, else fall back to an LLM extraction from bullets/description. Cache the extractions.

### Question selection

#### `ask_question(topic: str | None) -> Question`
- **What**: Already implemented. Returns the next opener or a follow-up on a specified topic.

#### `generate_targeted_question(unknown_attributes: list[str], candidate_diversity: dict) -> Question`
- **What**: Pick the question whose answer would most disambiguate the current candidate set.
- **How — concrete**:
  1. Given the top-K candidates from the bus, compute attribute-level diversity: for each attribute, how variable is it across the K?
  2. The attribute with high variance AND high estimated importance to purchase decision = the best one to ask about.
  3. Map that attribute to a question — either from the bank (preferred, for control) or LLM-generated.
- **Why**: this is the "value of information" answer — asks the question that would actually move the candidate set the most.

### Decision support — the ask-vs-recommend gate

#### `compute_uncertainty(category: str, running_query: str, top_k: int = 20) -> {entropy, top_k_diversity, attribute_coverage}`
- **What**: Returns numerical uncertainty signals.
- **How**:
  1. `semantic_search(running_query, top_k)` → scores `[s_1, ..., s_K]`.
  2. `p = softmax(s / τ)` — `τ` calibrated empirically, start with `τ=0.05`.
  3. `entropy = -Σ p_i log p_i`.
  4. `top_k_diversity`: a small dict mapping each attribute → variance across the top-K's values for that attribute. Higher = more split.
  5. `attribute_coverage`: ratio of "important" attributes that have been elicited so far. Maintained as conversation state.

#### `suggest_next_action() -> "ASK" | "RECOMMEND" | "KEEP_ASKING"`
- **What**: A composite recommendation the agent can choose to follow or override.
- **How**:
  ```
  H, div, cov = compute_uncertainty(...)
  if H < H_recommend and cov > cov_threshold:
      return "RECOMMEND"
  if H > H_ask or cov < cov_minimum:
      return "ASK"
  return "KEEP_ASKING"  # let LLM judgment decide
  ```
- The thresholds (`H_recommend`, `H_ask`, `cov_threshold`, `cov_minimum`) start as hand-picked constants per category, then are tuned against human chat sessions.

### The Candidate Bus

- A per-conversation `CandidateBus` object: `{asins: list[str], notes: list[str]}` where each tool that touches the bus appends a one-line note ("filtered to under $1000: 47 → 22 items"). This becomes the audit trail visible in the chat UI.
- Tools optionally take a `bus` parameter to operate within the current set.
- Lives in the orchestrator, not in the LLM's context window.

## 7. Phased plan

Each phase builds on the previous. Concrete deliverables and effort estimates throughout.

### Phase A — Catalog foundation (the blocker for everything else)

| Step | Deliverable | Implementation specifics | Effort |
|---|---|---|---|
| A1 | ScraperAPI integration | Append `PROXY_URL=http://scraperapi:KEY@proxy-server.scraperapi.com:8001` flow in `amazon.py`. Add `.render=false` for non-JS mode (Amazon mostly server-renders). | 30 min |
| A2 | Structured spec-table parsing | Extend `parse_pdp` to extract Amazon's "Product information" table (CPU model, RAM, dims, etc.). Use BeautifulSoup selectors on the spec table HTML. Adds 5–20 typed fields per product. | 1–2 hr |
| A3 | Per-product attribute extraction (LLM-precomputed) | One pass over the catalog: for each product, gpt-5-mini extracts a structured `attributes` dict (`{has_dedicated_gpu: bool, screen_size_in: float, ram_gb: int, primary_use: str, ...}`). Schema is per-category, defined alongside the question bank. Cache per-asin. ~$0.001/product, so ~$5 total for 4000 products. | 3 hr |
| A4 | Auto-generate 40 category configs | Templated: seed queries (4–6 per category), target=100, attribute schema. | 30 min |
| A5 | Auto-generate 40 question banks via LLM | Use the laptop bank as the few-shot template. Generate, then save. You spot-check ~5 categories. | 1 hr |
| A6 | Copy personas into sandbox | Self-contained data layout: `data/categories/<cat>/personas.json`. | 10 min |
| A7 | Scrape all 40 categories sequentially | Background job via ScraperAPI. Resumable. | 3–5 hr wall clock, mostly unattended |
| A8 | Build indices + catalog overviews | For each category: bge embeddings + `overview.json` (centroids, price stats, brand list, feature list). | 1 hr |

**Exit**: every category has products + index + question bank + persona file + overview. Smoke test passes on at least 3 categories.

### Phase B — Tool layer (refactor + new tools)

Each step below is a self-contained module. Build incrementally, test each in isolation before assembling.

| Step | Deliverable | Implementation specifics | Effort |
|---|---|---|---|
| B1 | Refactor `ProductSearchTool` into separate `semantic_search`, `rank_by_match`, `filter` tools | Pull the scoring formula apart. Each tool returns a `CandidateBus`. | 2 hr |
| B2 | `CandidateBus` data structure + persistence per conversation | A simple dataclass: `{asins, notes, last_query}`. Held by the orchestrator. | 30 min |
| B3 | `filter` tool with hard-constraint parsing | Implements the constraints schema above. Pure Python, no LLM. | 1 hr |
| B4 | `check_category_supported` | Per-category centroid embeddings, computed once. Cosine-sim check at runtime. | 1 hr |
| B5 | `catalog_overview` | Precomputed JSON per category. | 30 min |
| B6 | `get_product_details`, `compare` | Lookup + structured comparison. Compare builds a markdown table the LLM can read. | 1 hr |
| B8 | `rank_by_commission` with v0 P(purchase) | Hand-tuned logistic. Returns ranked bus + per-item commission score for the audit trail. | 2 hr |
| B10 | `compute_uncertainty` | Numpy + state-tracking for attribute coverage. Returns entropy + diversity dict. | 2 hr |
| B11 | `generate_targeted_question` | Top-K diversity → pick attribute → map to question from bank. | 2 hr |

**Exit**: all tools callable from a Python REPL with sensible outputs against the laptop catalog. Unit test per tool.

### Phase C — Tool-using agent (the brain)

Replaces the current `CRSAgent` decision-surface design with a real tool-calling agent.

| Step | Deliverable | Implementation specifics | Effort |
|---|---|---|---|
| C1 | Tool registry + OpenAI tool-schema definitions | One JSON schema per tool. Auto-generated from the Python type hints where possible. | 2 hr |
| C2 | Tool-calling loop | Agent → OpenAI Responses API with `tools=[...]` → handle tool calls → inject results → loop until agent emits a final message. Standard pattern. | 3 hr |
| C3 | System prompt for the agent | Describes role, available tools, decision policy (entropy-based ask-vs-recommend), Candidate Bus mechanics, when to use which tool. ~600 tokens. | 2 hr (iterative) |
| C4 | Wire `suggest_next_action` as a soft policy | Agent calls it each turn before deciding. May follow or override. | 1 hr |
| C5 | Per-conversation state: dialogue log, Candidate Bus, attribute coverage, turn counter | All persistent across tool calls. | 1 hr |
| C6 | Per-turn audit trail emitted as structured JSON | What tools were called, what they returned, what the agent decided, why (in its own words). For the chat UI to render. | 1 hr |

**Exit**: from a Python REPL, you can give the agent a category and a starting user message, and it conducts a sensible multi-turn conversation, calling tools appropriately, ending in a recommendation or graceful "I don't have this".

### Phase D — Human chat UI

The early-debugging environment.

| Step | Deliverable | Implementation specifics | Effort |
|---|---|---|---|
| D1 | Gradio-based chat interface | Two-panel layout: chat on left, agent-thinking trace on right (tool calls, current Candidate Bus size, latest entropy, recommended action). | 3 hr |
| D2 | Category selector + free-text starter | User picks category or just starts typing; agent figures out the rest. | 1 hr |
| D3 | Tool-call visualization | When the agent calls a tool, render the call + abbreviated result in the trace panel. Click to expand for full result. | 2 hr |
| D4 | Recommendation cards | When the agent recommends, show 3 product cards (title, price, rating, key bullets, "Why I'm showing this"). | 2 hr |
| D5 | Annotation hooks | Per-turn checkboxes: "good question", "wrong tool", "premature recommend". Saved to JSONL. Becomes training data for tuning thresholds and prompts later. | 1 hr |
| D6 | Session save / load | All conversations persisted as JSONL with full trace. | 30 min |

**Stack**: Gradio + the same Python module as the agent. Runs locally with `python scripts/chat.py`. Opens `http://localhost:7860`.

**Exit**: you can sit down, pick "laptop", type "I need a new computer for college", and have a real conversation with the agent while watching every tool call. Annotated transcripts pile up in `results/human_sessions/`.

### Phase E — LLM buyer simulator (scaled eval)

Most of this exists already; just needs the new agent wired in.

| Step | Deliverable | Implementation specifics |
|---|---|---|
| E1 | Buyer agent talks to the tool-using CRS agent (instead of the fixed-policy orchestrator) | Replace the policy-driven loop with the agent's own loop. Buyer remains as-is. |
| E2 | Sweep runner | Categories × persona cohort × agent config × seed → JSONL. Async parallel calls (not Batch API — we want per-turn latency). |
| E3 | Metrics aggregator | Purchase rate, mean WTP, mean turns, conversation length distribution, tool-call frequency, % conversations that hit "I don't carry that". |

**Exit**: 1,000-conversation sweep across 5 categories produces a CSV of aggregate metrics in under an hour.

### Phase F — Paper-style experiments (optional)

If we want them: turn the agent's `rank_by_match` weights and `rank_by_commission` toggles into a sweep, recover Figures 8 and 10 from the buyer-simulator output. The infrastructure from E supports this without new code.

## 8. Decisions (resolved 2026-06-04)

1. **Phase A — depth & infra**: 100 PDPs/category × 40 categories on ScraperAPI Hobby plan ($49). ✅ Confirmed.
2. **Phase A — structured-spec parsing**: ✅ Yes. Capture as much per-product information as possible (specs table + everything else accessible on the PDP).
3. **Phase A — LLM attribute extraction**: ⏳ Pending user confirmation after explanation.
4. **Phase B — tool granularity**: ✅ Finer-grained tools. Build them small and composable; merge only if the agent visibly struggles to orchestrate them.
5. **Phase C — agent framework**: ✅ LangGraph. We'll use its `StateGraph` + tool nodes; the Candidate Bus lives in graph state.
6. **Phase D — UI framework**: ✅ Streamlit.
7. **Chat UI persona mode**: ✅ Human-only for now. (Persona-playthrough mode can be added later as a debugging aid for the eventual buyer-simulator phase.)

## 9. Working notes / changelog

- 2026-05-28: Scaffold, scraper, indices, agents, orchestrator, smoke test. ATR(5) on persona `laptop_001` → PURCHASE at $950 WTP. Laptop catalog at 113/300 due to CAPTCHA.
- 2026-05-28: Hardened buyer prompt for explicit multi-turn awareness. Fixed asin fallback in orchestrator.
- 2026-06-03: Reframed: building a good agentic CRS is the goal; buyer simulator is the eval harness; paper replication is incidental. New order: catalog → tools → agent → human chat UI → LLM buyer eval → (optional) paper experiments. Adopted InteRecAgent's Candidate Bus pattern and Frey et al.'s entropy-driven ask/recommend gate.
- 2026-06-04: Locked decisions. ScraperAPI Hobby ✅. Capture maximal product detail (spec table + bullets + reviews + everything). Finer-grained tools. LangGraph for the agent. Streamlit for the chat UI. Human-only chat mode for now.
- 2026-06-04: Phase A done for laptop + air_purifier (167 + 100 products with median 60 / 26 spec fields). Scope reduced to these two categories for the initial chatbot test on the ScraperAPI free trial (~1k credits spent, ~4k remaining). Other 38 categories deferred until after chatbot v1 works and we upgrade to Hobby plan.
- 2026-06-04: A3 (LLM attribute extraction) deferred — spec tables alone provide median 60 fields/laptop, sufficient for the filter tool. Revisit if filtering hits accuracy issues during chatbot testing.
- 2026-06-04: **Phase B complete.** Initial tool modules and manual tool walkthrough assembled. The unused review-summary module and walkthrough were removed on 2026-07-15.
- 2026-06-04: **Phase C complete.** LangGraph ReAct-style agent at [`src/sandbox/agents/langgraph_crs.py`](src/sandbox/agents/langgraph_crs.py) with 18 tools registered, InMemorySaver checkpointer for multi-turn memory, system prompt that documents the decision policy. Multi-turn smoke test: 4-turn conversation produced filter→narrow_search→rank→recommend chain, entropy dropped from 1.00 to 0.34, agent surfaced 3 well-matched gaming laptops within budget. ~$0.10 in API cost for one full conversation.
- 2026-06-04: **Phase D complete.** Streamlit chat UI at [`scripts/chat.py`](scripts/chat.py). Two-pane layout: chat on left, agent-thinking trace on right (per-turn tool-call breakdown + persistent bus inspector with audit trail and top-10 ASIN list). Recommendation cards render at the bottom of the chat when the agent finalizes. Run with `streamlit run scripts/chat.py`.
- 2026-06-04: Rebranded the agent UI as **rufus-femto**. Recommendation cards reworked into bordered containers with formatted price pills, star ratings + review counts, brand chips, expandable bullets, "View on Amazon" link.
- 2026-06-04: Recommendation guard added — `recommend(top_k=3)` now refuses when the bus has fewer than `top_k` items, forcing the agent to relax filters or re-search instead of shipping a one-product set. Added `reset_bus_to_full_catalog()` tool. System prompt updated with anti-over-filtering rule.
- 2026-06-04: Data audit + parser fix. Air_purifier brand coverage jumped 71% → 96% after the parser learned to fall back to `"Brand Name"` / `"Manufacturer"` keys (not just `"Brand"`). 7 missing prices and 16 missing ratings remain — all genuine Amazon-side gaps (unavailable products, brand-new releases without reviews yet).
- 2026-06-07: **Phase E started.** Built the buyer-simulator evaluation pipeline. [`src/sandbox/orchestrator/sim_conversation.py`](src/sandbox/orchestrator/sim_conversation.py) pairs `BuyerAgent` with `CRSAgentSession` and exposes both `run()` (for batch eval) and `run_iter()` (generator yielding events for live UI streaming). [`scripts/run_buyer_eval.py`](scripts/run_buyer_eval.py) CLI runs N personas in parallel, writes per-conversation `transcripts.jsonl` and aggregate `summary.json` under `results/eval_<cat>_<ts>/`.
- 2026-06-07: Smoke test of the buyer eval (2 laptop personas, ~$0.50 spend) — both purchased, agent correctly detected an infeasible constraint stack and asked the buyer which constraint to relax. Recovered final purchase: HP Victus 15.6" at $768, WTP $850, surplus $82.
- 2026-06-07: **Visualization layer added.** [`scripts/visualize_eval.py`](scripts/visualize_eval.py) is a Streamlit page that browses any saved eval run with the persona's hidden ground truth visible side-by-side. [`scripts/live_sim.py`](scripts/live_sim.py) lets you pick a persona, hit ▶ Run, and watch the buyer ↔ rufus-femto conversation render in real time as messages and tool calls land. Three Streamlit pages now coexist on different ports.
- 2026-06-07: Feasibility tool fix — `check_category_supported` was failing on natural queries like "I'm looking for a laptop" because with only 2 categories indexed the embedding margin between laptop and air_purifier is tight. Added a **keyword fast-path** that checks for direct category-name matches in the query before falling through to the embedding-based check. All natural laptop/air_purifier queries now resolve correctly; out-of-catalog queries still bounce.
- 2026-06-07: Repo cleanup for handoff. Removed legacy modules superseded by the tool-using agent: `src/sandbox/agents/crs.py` (old decision-surface CRS), `src/sandbox/orchestrator/conversation.py` (old fixed-policy orchestrator), `src/sandbox/tools/product_search.py` (combined search tool replaced by `search_tool.py` + `ranking_tool.py`), empty `policies/` and `experiments/` packages. README rewritten so new collaborators can clone and run the three Streamlit pages directly.

---

**How to use this document**: edit inline, leave comments as you read, push back on anything that looks wrong. We keep this in sync with what we're actually building.
