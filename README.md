# crs-sandbox · rufus-femto

A sandbox for building and evaluating an **agentic conversational shopping assistant** (think mini Amazon Rufus) against a real scraped product catalog, with an LLM-based buyer simulator as the cheap stand-in for human evaluation.

The shopping assistant — codenamed **rufus-femto** — is a LangGraph ReAct-style agent with 18 tools spanning catalog feasibility, hard-constraint filtering, semantic search, multi-objective ranking (match / commission / price), inspection, review summarization, and an entropy-based decision policy that drives endogenous ask-vs-recommend transitions.

## What's in the box

| Layer | What | Where |
|---|---|---|
| **Catalog** | Real Amazon product data scraped via Playwright or ScraperAPI; spec tables (median 60 fields/product), prices, ratings, review excerpts | `data/categories/<cat>/products.jsonl` |
| **Tools** | 10 tool modules the agent orchestrates: feasibility, filter, semantic_search, ranking, uncertainty (entropy), inspection, review summarization, question bank, candidate-bus shared state | `src/sandbox/tools/` |
| **rufus-femto agent** | LangGraph ReAct loop wrapping the tool layer; per-conversation Candidate Bus state; entropy-driven ask-vs-recommend gate | `src/sandbox/agents/langgraph_crs.py` |
| **Buyer simulator** | LLM conditioned on a hidden persona; answers only what's asked; makes a structured PURCHASE / NO_PURCHASE + WTP decision when shown recommendations | `src/sandbox/agents/buyer.py` |
| **Sim orchestrator** | Pairs buyer × rufus-femto, runs the dialogue loop, optionally with per-turn abandonment hazard. Exposes both `run()` and a `run_iter()` generator for live UI streaming. | `src/sandbox/orchestrator/sim_conversation.py` |
| **Streamlit pages** | Three local UIs — see [Interactive pages](#interactive-pages) below | `scripts/chat.py`, `scripts/live_sim.py`, `scripts/visualize_eval.py` |
| **Eval CLI** | Runs N personas through rufus-femto in parallel, writes transcripts JSONL + aggregate summary | `scripts/run_buyer_eval.py` |

## Quick start

```bash
git clone <your-fork>
cd crs-sandbox
pip install -r requirements.txt
playwright install chromium      # only needed if you re-scrape

# Set up secrets
cp .env.example .env
# edit .env to add OPENAI_API_KEY (and SCRAPERAPI_KEY if scraping new categories)

# Three things you can do, in increasing order of "automated":
streamlit run scripts/chat.py        --server.port=8501   # talk to rufus-femto yourself
streamlit run scripts/live_sim.py    --server.port=8503   # watch buyer LLM talk to it
streamlit run scripts/visualize_eval.py --server.port=8502 # browse batch eval results

# Or run a batch evaluation from CLI (10 laptop personas, ~$2):
python scripts/run_buyer_eval.py laptop --n-personas 10
```

The repo ships with two categories already scraped and indexed: **laptop** (167 products) and **air_purifier** (100 products). To add more categories, see [Adding a category](#adding-a-category) below.

## Interactive pages

| Port | Page | What it's for |
|---|---|---|
| 8501 | [`scripts/chat.py`](scripts/chat.py) | You ↔ rufus-femto. Type a message, watch tool calls in the right pane, see recommendations as cards. |
| 8502 | [`scripts/visualize_eval.py`](scripts/visualize_eval.py) | Browse saved transcripts from a `run_buyer_eval.py` run. Side-by-side view of the persona's hidden ground truth and the conversation that actually happened. |
| 8503 | [`scripts/live_sim.py`](scripts/live_sim.py) | Pick a persona, hit ▶ Run, watch the buyer LLM talk to rufus-femto in real time as messages and tool calls render incrementally. |

All three can run simultaneously on different ports.

## Architecture

```
              BUYER SIMULATOR (LLM)              rufus-femto CRS AGENT (LangGraph)
              ─────────────────────              ────────────────────────────────
              persona-conditioned                10 tool modules (feasibility,
              answers only what asked  ◀─ chat ─▶    filter, search, rank,
              makes purchase decision            inspect, uncertainty, review)
              emits WTP                          Candidate Bus shared across calls
                     │                                       │
                     └─────────── SIM ORCHESTRATOR ──────────┘
                                        │
                                        ▼
                          transcript + outcome (JSONL)
                                        │
                                        ▼
                              aggregate metrics
                              (purchase rate, mean WTP,
                               mean turns, abandonment)
```

The CRS agent's decision-making is driven by **entropy of retrieval scores** ([Frey et al. 2025](https://arxiv.org/abs/2509.06185)): after each user message, it computes uncertainty over the current candidate set and asks one more question when high, recommends when low. This is the empirical answer to "when should a CRS converse?" — replacing the fixed `REC` / `ATR(k)` policies typical in CRS literature.

## Repo layout

```
crs-sandbox/
├── README.md                       # this file
├── PLAN.md                         # design + decision log (living)
├── .env.example                    # copy to .env, fill in keys
├── .gitignore                      # ignores .env, raw HTML, embeddings, results
├── requirements.txt
├── configs/
│   └── <category>.yaml             # seed queries, catalog path, ranking defaults
├── data/
│   └── categories/<category>/
│       ├── products.jsonl          # scraped catalog (committed)
│       ├── personas.json           # LLM-generated buyer personas (committed)
│       ├── questions.yaml          # tier-1 openers + tier-2 followups (committed)
│       ├── review_cache.json       # cached review summaries (committed, regen-able)
│       └── index/
│           ├── embeddings.npy      # bge-large-en-v1.5 (gitignored, regen-able)
│           ├── products.json       # ordered metadata (committed)
│           └── centroid.npy        # for check_category_supported (committed, regen-able)
├── src/sandbox/
│   ├── agents/
│   │   ├── langgraph_crs.py        # the rufus-femto agent (18 tool wrappers)
│   │   └── buyer.py                # persona-conditioned buyer LLM
│   ├── tools/                      # the 10 tool modules the agent orchestrates
│   ├── orchestrator/
│   │   └── sim_conversation.py     # pairs buyer ↔ rufus-femto (run + run_iter)
│   ├── scraper/
│   │   ├── fast.py                 # httpx + ScraperAPI (concurrent)
│   │   └── amazon.py               # Playwright (residential fallback)
│   ├── index/builder.py            # builds bge embeddings + per-cat centroids
│   ├── catalog.py                  # central catalog/config loader
│   └── env.py                      # minimal .env loader
└── scripts/
    ├── chat.py                     # Streamlit: human ↔ rufus-femto
    ├── live_sim.py                 # Streamlit: live buyer × rufus-femto
    ├── visualize_eval.py           # Streamlit: browse eval transcripts
    ├── repl.py                     # terminal REPL for the agent
    ├── test_tools.py               # integration test of the tool layer
    ├── run_buyer_eval.py           # buyer-simulator evaluation runner
    ├── scrape_category.py          # CLI for the scraper
    ├── build_index.py              # CLI for the index builder
    ├── reparse_category.py         # reparse cached HTML after parser update
    └── gen_configs.py              # auto-generate 40 category configs
```

## Adding a category

```bash
# 1. Add (or hand-edit) configs/<your-category>.yaml using laptop.yaml as a template;
#    gen_configs.py can seed all 40 from the persona dataset at once.
python scripts/gen_configs.py

# 2. Scrape the products (uses ScraperAPI if SCRAPERAPI_KEY is set)
python scripts/scrape_category.py <your-category>

# 3. Build the bge index + per-category centroid
python scripts/build_index.py <your-category>

# 4. Drop a question bank at data/categories/<your-category>/questions.yaml
#    (3 openers + ~20 followups in the laptop/air_purifier format)

# 5. Drop buyer personas at data/categories/<your-category>/personas.json
```

## Tools the rufus-femto agent has access to

The agent decides which to call and in what order; the orchestrator never picks for it. See [`src/sandbox/agents/langgraph_crs.py`](src/sandbox/agents/langgraph_crs.py) for the full tool schema and the system prompt that documents the decision policy.

| Tool | What it does |
|---|---|
| `check_category_supported(query)` | Decides whether the user's request matches an indexed category. Keyword match first, embedding centroid fallback. |
| `set_category(category)` | Locks in the working category; initializes the Candidate Bus with all products. |
| `catalog_overview()` | Returns count, price range, top brands, common feature terms. |
| `available_filters()` | Lists which spec-table fields are filterable and example values. |
| `filter_products({price_max, rating_min, spec_contains, …})` | Applies hard constraints to the bus. |
| `reset_bus_to_full_catalog()` | Recovery: discards filters and starts over with the full catalog. |
| `semantic_search_full(query, top_k)` | bge cosine over the FULL catalog; replaces bus. |
| `narrow_search(query, top_k)` | bge cosine over the CURRENT bus; just re-ranks. |
| `rank_by_match(semantic, rating, popularity)` | Weighted re-rank. |
| `rank_by_commission(budget_max)` | Reranks by `price × P(purchase)`, with budget-aware boost. |
| `rank_by_price(ascending)` | Sort by price. |
| `get_product_details(asin)` | Full structured record for one product. |
| `compare_products(asins, aspects)` | Side-by-side spec table for 2–4 products. |
| `summarize_reviews(asin, aspect)` | Cached LLM-generated review distillation. |
| `compute_uncertainty(top_k)` | Entropy of retrieval scores + per-attribute diversity. |
| `suggest_next_action()` | Composite ASK / RECOMMEND / KEEP_ASKING gate. |
| `ask_question(topic)` | Next tier-1 opener or tier-2 followup from the per-category bank. |
| `recommend(top_k, justification)` | Finalize. Refuses if the bus has fewer than `top_k` products. |

## Cost reference

| Operation | Approx cost |
|---|---|
| One CRS ↔ buyer conversation (~6 turns, ~20 tool calls) | $0.15–0.30 |
| `summarize_reviews` per (asin, aspect) | $0.0003, cached on disk |
| Scraping one 100-product category via ScraperAPI | ~500 credits ≈ $0.25 on Hobby plan |
| Building bge embeddings for one category | $0 (runs locally on CPU) |

## Known limitations and tunables

- **Two-category feasibility margins are weak**: with only laptop + air_purifier indexed, the embedding-based feasibility check needs a keyword fast-path. As more categories are added, embedding margins widen naturally. See [`src/sandbox/tools/feasibility_tool.py`](src/sandbox/tools/feasibility_tool.py).
- **WTP is the buyer LLM's estimate, not a measurement**: see PLAN.md §7 for the methodological framing.
- **`rank_by_commission`'s P(purchase) is hand-tuned (v0)** in [`ranking_tool.py`](src/sandbox/tools/ranking_tool.py). Coefficients should be re-fit once we have buyer-simulator outcome data at scale.
- **Recommendation `recommend(top_k)` refuses if bus has fewer than `top_k`** — forces the agent to relax or re-search rather than ship a one-product set.
- **Entropy thresholds in `suggest_next_action`** are hand-picked constants per category; calibrate against real human chat sessions when convenient.

## Where to start as a new collaborator

1. **Open [`PLAN.md`](PLAN.md)** — it's the design + decision log. Read §1–§4 for what we're building and why; §5 for current state; §7 for the phased plan.
2. **Run [`streamlit run scripts/chat.py`](scripts/chat.py)** and talk to rufus-femto for a few rounds. Watch the right pane.
3. **Run [`streamlit run scripts/live_sim.py`](scripts/live_sim.py)** and watch a persona-conditioned buyer interact with it.
4. **Read [`src/sandbox/agents/langgraph_crs.py`](src/sandbox/agents/langgraph_crs.py)** — especially the `SYSTEM_PROMPT` and the tool wrappers. This is where most of the design lives.
5. **Run [`python scripts/test_tools.py`](scripts/test_tools.py)** — composes the tool layer end-to-end against the laptop catalog with no LLM calls. Useful for understanding the data flow.
