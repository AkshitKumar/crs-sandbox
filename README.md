# crs-sandbox · rufus-femto

A sandbox for building and evaluating an **agentic conversational shopping assistant** (think mini Amazon Rufus) against a real scraped product catalog, with an LLM-based buyer simulator as the cheap stand-in for human evaluation.

The interactive demo and every batch arm use the same ReAct recommender.
Explicit policies constrain only its question and recommendation tools: `REC`
asks nothing and freezes a curated category slate, while `ATR(k)` asks exactly
`k` fixed, neutral questions before the agent supplies full and focused retrieval queries.
Local BGE/BM25 hybrid retrieval produces 15 candidates, and one model call selects and
briefly explains three. See
[EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md) before interpreting results.
The recommendation path performs no local training or fine-tuning and requires no
fitted relevance model; it uses a pinned pretrained BGE encoder for inference.
Its manifest records the catalog, index, prompts, configuration, question bank,
and model identifiers needed for replication. Its buyer and
recommender defaults pin `gpt-5-mini-2025-08-07` rather than the moving alias.

The shopping assistant — codenamed **rufus-femto** — is a LangGraph ReAct-style agent with 18 tools spanning catalog feasibility, hard-constraint filtering, semantic search, multi-objective ranking (match / commission / price), inspection, and an entropy-based decision policy that drives endogenous ask-vs-recommend transitions.

## What's in the box

| Layer | What | Where |
|---|---|---|
| **Catalog** | Real Amazon product data scraped via Playwright or ScraperAPI; spec tables (median 60 fields/product), prices, ratings, review excerpts | `data/categories/<cat>/products.jsonl` |
| **Tools** | Explicit modules for feasibility, filtering, semantic search, ranking, uncertainty, inspection, questions, and candidate-bus state | `src/sandbox/tools/` |
| **rufus-femto agent** | LangGraph ReAct loop wrapping the tool layer; per-conversation Candidate Bus state; entropy-driven ask-vs-recommend gate | `src/sandbox/agents/langgraph_crs.py` |
| **Buyer simulator** | LLM conditioned on a hidden persona; answers only what's asked; makes a structured PURCHASE / NO_PURCHASE + WTP decision when shown recommendations | `src/sandbox/agents/buyer.py` |
| **Recommendation pipeline** | Retrieves 15 products from the ReAct agent's query and validates one combined select-and-explain response; also produces non-mutating hidden checkpoints | `src/sandbox/agents/recommendation_pipeline.py` |
| **Sim orchestrator** | Pairs the buyer with the same CRS under adaptive or fixed tool gates; records a manifest and validated outcome | `src/sandbox/orchestrator/sim_conversation.py` |
| **Streamlit pages** | Three local UIs — see [Interactive pages](#interactive-pages) below | `scripts/chat.py`, `scripts/live_sim.py`, `scripts/visualize_eval.py` |
| **Eval CLI** | Runs N personas adaptively by default, or under explicit `rec`/`atr(k)` controls; writes ordered transcripts, summary, and run manifest | `scripts/run_buyer_eval.py` |

## Quick start

```bash
git clone <your-fork>
cd crs-sandbox

# Choose one install:
pip install -r requirements.txt       # full interactive/scraping stack
# OR:
pip install -r requirements-eval.txt  # controlled evaluation + local BGE inference

playwright install chromium      # full stack only; needed only if you re-scrape

# Set up secrets
cp .env.example .env
# edit .env to add OPENAI_API_KEY (and SCRAPERAPI_KEY if scraping new categories)

# No-provider structural check: catalogs, slates, indices, and CRS graph routing
python scripts/check_eval_setup.py

# One-call recommendation-operation check (real API; bypasses ReAct and buyer)
python scripts/check_recommendation_examples.py \
  --case laptop:laptop_001 --parallel 1

# One-person vertical slices through the real ReAct agent and buyer
python scripts/run_buyer_eval.py laptop --persona-ids laptop_001 --parallel 1 \
  --policy rec --fail-on-protocol-error
python scripts/run_buyer_eval.py laptop --persona-ids laptop_001 --parallel 1 \
  --policy atr --numquestions 1 --fail-on-protocol-error

# Three things you can do, in increasing order of "automated":
streamlit run scripts/chat.py        --server.port=8501   # talk to rufus-femto yourself
streamlit run scripts/live_sim.py    --server.port=8503   # watch buyer LLM talk to it
streamlit run scripts/visualize_eval.py --server.port=8502 # browse batch eval results

# Run the ordinary adaptive recommender in a batch:
python scripts/run_buyer_eval.py laptop --n-personas 10

# Or select an explicit controlled evaluation policy:
python scripts/run_buyer_eval.py laptop --n-personas 10 --policy rec
python scripts/run_buyer_eval.py laptop --n-personas 10 --policy atr --numquestions 3
python scripts/run_buyer_eval.py laptop --n-personas 10 --policy checkpoint_atr --numquestions 3

# Rerun all six documented revealed-prefix query/retrieval/selection diagnostics:
python scripts/check_recommendation_examples.py
```

The current catalog snapshots contain **laptop** (327 products) and
**air_purifier** (203 products). Rebuild their indexes whenever catalog contents
or the shared product-to-text format changes. To add more categories, see
[Adding a category](#adding-a-category) below.

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
              persona-conditioned                tool modules (feasibility,
              answers only what asked  ◀─ chat ─▶    filter, search, rank,
              makes purchase decision            inspect, uncertainty, questions)
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

The adaptive CRS can consult **entropy of retrieval scores** ([Frey et al. 2025](https://arxiv.org/abs/2509.06185)) through its uncertainty tools, but the ReAct model decides whether to call or override them. This is the batch default when `--policy` is omitted. Explicit `rec`, terminal `atr(k)`, and central-trajectory `checkpoint_atr(k)` runs retain the same agent while gating question count and terminality as described in [EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md).

## Repo layout

```
crs-sandbox/
├── README.md                       # this file
├── PLAN.md                         # design + decision log (living)
├── .env.example                    # copy to .env, fill in keys
├── .gitignore                      # ignores .env, raw HTML, embeddings, results
├── requirements.txt
├── configs/
│   └── <category>.yaml             # seed queries and catalog/index paths
├── data/
│   └── categories/<category>/
│       ├── products.jsonl          # scraped catalog (committed)
│       ├── personas.json           # LLM-generated buyer personas (committed)
│       ├── questions.yaml          # tier-1 openers + tier-2 followups (committed)
│       └── index/
│           ├── embeddings.npy      # bge-large-en-v1.5 (gitignored, regen-able)
│           ├── products.json       # ordered metadata (committed)
│           └── centroid.npy        # for check_category_supported (committed, regen-able)
├── src/sandbox/
│   ├── agents/
│   │   ├── langgraph_crs.py        # the rufus-femto agent (18 tool wrappers)
│   │   ├── recommendation_pipeline.py # hybrid recall + select-and-explain
│   │   └── buyer.py                # persona-conditioned buyer LLM
│   ├── tools/                      # the 8 tool modules the agent orchestrates
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
    ├── run_buyer_eval.py           # buyer-simulator evaluation runner
    ├── scrape_category.py          # CLI for the scraper
    ├── build_index.py              # CLI for the index builder
    ├── reparse_category.py         # reparse cached HTML after parser update
    └── gen_configs.py              # scaffold configs from a fixed category table
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

# 6. Curate exactly three representative ASINs in the config's default_slate.
#    REC and checkpoint 0 always show these exact products.
```

Curate the slate before looking at evaluation outcomes and without using
evaluation-persona preferences. Choose three valid, priced, non-duplicate
products that form a reasonable editorial introduction to the category. Include
at least one top-ten product by rating volume when that list contains a valid,
non-duplicative mainstream option; use the remaining slots to cover materially
different mainstream segments. Freeze their ASINs with the catalog/config
hashes recorded in the run manifest.

### Catalog records versus the embedding index

`products.jsonl` is the source catalog. Scraping appends newly discovered ASINs
with their raw parsed title, price, rating, rating count, specifications,
bullets, description, and review excerpts; it does not transform those records
into a second product set. Existing ASINs are skipped, so another scrape does
not refresh an old listing's price or rating count.

Index building reads those same records and converts each one to a shared text
view containing labelled title, brand, price, rating, specifications, selected
bullets, and a short description. Only that text view is embedded. The parallel
`index/products.json` is a snapshot used to keep embedding rows aligned with
ASINs. `index/index_manifest.json` binds the catalog and ordered ASIN hashes to
the pinned BGE revision, serialization version, embedding shape/dtype, and file
hashes. Evaluation validates this contract before scoring. Rebuild after a
scrape or a product-to-text change because existing vectors do not update
themselves:

```bash
python scripts/build_index.py <your-category>
```

The fixed `default_slate` is configuration rather than scraped or imputed data.
It changes only when a curator deliberately replaces its three ASINs.

## Tools the rufus-femto agent has access to

The agent decides which to call and in what order; the orchestrator never picks for it. See [`src/sandbox/agents/langgraph_crs.py`](src/sandbox/agents/langgraph_crs.py) for the full tool schema and the system prompt that documents the decision policy.

| Tool | What it does |
|---|---|
| `check_category_supported(query)` | Decides whether the user's request matches an indexed category. Keyword match first, embedding centroid fallback. |
| `set_category(category)` | Locks in the working category; initializes the Candidate Bus with all products. |
| `catalog_overview()` | Returns count, price range, top brands, common feature terms. |
| `available_filters()` | Lists which spec-table fields are filterable and example values. |
| `preview_filter({price_max, rating_min, spec_contains, …})` | Reports how many products a proposed filter would retain without mutating the bus. |
| `filter_products({price_max, rating_min, spec_contains, …})` | Applies hard constraints to the bus. |
| `reset_bus_to_full_catalog()` | Recovery: discards filters and starts over with the full catalog. |
| `semantic_search_full(query, top_k)` | bge cosine over the FULL catalog; replaces bus. |
| `narrow_search(query, top_k)` | bge cosine over the CURRENT bus; just re-ranks. |
| `rank_by_match(semantic, rating, popularity)` | Weighted re-rank. |
| `rank_by_commission(budget_max)` | Reranks by `price × P(purchase)`, with budget-aware boost. |
| `rank_by_price(ascending)` | Sort by price. |
| `get_product_details(asin)` | Full structured record for one product. |
| `compare_products(asins, aspects)` | Side-by-side spec table for 2–4 products. |
| `compute_uncertainty(top_k)` | Entropy of retrieval scores + per-attribute diversity. |
| `suggest_next_action()` | Composite ASK / RECOMMEND / KEEP_ASKING gate. |
| `ask_question(topic)` | Next tier-1 opener or tier-2 followup from the per-category bank. |
| `recommend(query, key_query)` | Build a 15-product round-robin pool from full-query BGE, full-query BM25, and focused-query BGE, then select and briefly explain exactly three in one validated call. |

The catalog tools themselves are local. The enclosing ReAct loop calls OpenAI
to decide which tools to use. For a normal recommendation, `recommend()` uses
local hybrid retrieval and one OpenAI select-and-explain call; the fixed REC slate
uses one explanation-only call. Buyer answers, hidden
checkpoint scores, and the terminal purchase decision are separate OpenAI
calls. BGE embedding/search is local inference over the prebuilt index.

## Cost reference

| Operation | Approx cost |
|---|---|
| Scraping one 100-product category via ScraperAPI | ~500 credits ≈ $0.25 on Hobby plan |
| Building bge embeddings for one category | $0 (runs locally on CPU) |

## Known limitations and tunables

- **The bounded shortlist is a recall assumption**: the LLM selector cannot recover a
  relevant product omitted by all three retrieval lanes. Production transcripts
  store the full query, key query, candidate identities, dense scores, and the
  lane that surfaced each candidate. New categories should repeat a recall audit
  before a primary experiment.
- **Checkpoint retrieval reuses agent state**: each personalized checkpoint uses
  the latest full and focused queries the ReAct agent supplied. If no full query
  exists, it uses the revealed ledger and records that fallback. No separate
  query-normalization model is involved.
- **Two-category feasibility margins are weak**: with only laptop + air_purifier indexed, the embedding-based feasibility check needs a keyword fast-path. As more categories are added, embedding margins widen naturally. See [`src/sandbox/tools/feasibility_tool.py`](src/sandbox/tools/feasibility_tool.py).
- **WTP is the buyer LLM's estimate, not a measurement**: see PLAN.md §7 for the methodological framing.
- **`rank_by_commission`'s P(purchase) is hand-tuned (v0)** in [`ranking_tool.py`](src/sandbox/tools/ranking_tool.py). Coefficients should be re-fit once we have buyer-simulator outcome data at scale.
- **Entropy thresholds in `suggest_next_action`** are hand-picked global defaults; calibrate against real human chat sessions when convenient.

## Where to start as a new collaborator

1. **Open [`PLAN.md`](PLAN.md)** — it's the design + decision log. Read §1–§4 for what we're building and why; §5 for current state; §7 for the phased plan.
2. **Run [`streamlit run scripts/chat.py`](scripts/chat.py)** and talk to rufus-femto for a few rounds. Watch the right pane.
3. **Run [`streamlit run scripts/live_sim.py`](scripts/live_sim.py)** and watch a persona-conditioned buyer interact with it.
4. **Read [`src/sandbox/agents/langgraph_crs.py`](src/sandbox/agents/langgraph_crs.py)** — especially the `SYSTEM_PROMPT` and the tool wrappers. This is where most of the design lives.
