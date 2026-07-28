# crs-sandbox · rufus-femto

A sandbox for building and evaluating an agentic conversational shopping
assistant against real scraped product catalogs, with a persona-conditioned LLM
buyer as a practical stand-in for early human evaluation.

The interactive demo and every evaluation arm use the same LangGraph ReAct
agent. Adaptive conversations let the model decide how to use the catalog
tools. Controlled runs constrain question count and recommendation timing:
`rec` asks nothing, `single_atr(k)` asks exactly `k` fixed questions, and
`checkpoint_atr(k)` adds hidden prefix-only recommendation snapshots to the
same visible dialogue.

Local hybrid retrieval combines full-query BGE, full-query BM25, and a focused
BGE query into 15 candidates. One structured model call selects and briefly
explains three products. The recommendation path uses a pinned pretrained BGE
encoder for inference; it performs no local training or fine-tuning.

## What's in the box

| Layer | What | Where |
|---|---|---|
| Catalog | Scraped Amazon products, personas, and question banks | `data/categories/<category>/` |
| Agent | ReAct loop and 18 catalog/recommendation tools | `src/sandbox/agents/langgraph_crs.py` |
| Recommendation pipeline | Hard-budget eligibility, hybrid retrieval, selection, and explanation | `src/sandbox/agents/recommendation_pipeline.py` |
| Buyer simulator | Hidden-persona answers and structured purchase decisions | `src/sandbox/agents/buyer.py` |
| Orchestrator | Runs one conversation and records its outcome | `src/sandbox/orchestrator/sim_conversation.py` |
| Evaluation CLI | Parallel persona runs, summaries, transcripts, and manifests | `scripts/run_buyer_eval.py` |
| Interactive UIs | Human chat, live simulation, and result browsing | `scripts/chat.py`, `scripts/live_sim.py`, `scripts/visualize_eval.py` |

The current catalog snapshots cover `laptop` and `air_purifier`.

## Quick start

```bash
git clone <your-fork>
cd crs-sandbox

uv venv --python 3.12
uv pip install -r requirements.txt

cp .env.example .env
# Add OPENAI_API_KEY. Add SCRAPERAPI_KEY only when scraping.

# Offline structural check
uv run python scripts/check_eval_setup.py

# One recommendation-pipeline API check
uv run python scripts/check_recommendation_examples.py \
  --case laptop:laptop_001 --parallel 1

# One-person controlled conversations
uv run python scripts/run_buyer_eval.py laptop \
  --persona-ids laptop_001 --parallel 1 --policy rec \
  --fail-on-protocol-error
uv run python scripts/run_buyer_eval.py laptop \
  --persona-ids laptop_001 --parallel 1 \
  --policy single_atr --numquestions 1 \
  --fail-on-protocol-error
```

Interactive surfaces:

```bash
uv run streamlit run scripts/chat.py --server.port=8501
uv run streamlit run scripts/visualize_eval.py --server.port=8502
uv run streamlit run scripts/live_sim.py --server.port=8503
```

Ordinary adaptive and controlled batches:

```bash
uv run python scripts/run_buyer_eval.py laptop --n-personas 10
uv run python scripts/run_buyer_eval.py laptop --n-personas 10 --policy rec
uv run python scripts/run_buyer_eval.py laptop --n-personas 10 \
  --policy single_atr --numquestions 3
uv run python scripts/run_buyer_eval.py laptop --n-personas 10 \
  --policy checkpoint_atr --numquestions 3
```

Outputs go to `results/local/eval_<category>_<timestamp>/` unless `--out-dir` is
provided. See [the controlled evaluation protocol](docs/EVALUATION_PROTOCOL.md)
for the state machine, checkpoint semantics, and reproducibility contract.

## Architecture

```text
buyer or buyer simulator
        │
        ▼
CRSAgentSession (ReAct)
        │
        ├── catalog, filter, search, rank, inspect, and uncertainty tools
        ├── per-conversation Candidate Bus
        └── recommendation pipeline
                ├── hard-budget eligibility
                ├── BGE + BM25 + focused BGE recall
                └── select-and-explain model call
        │
        ▼
transcript + outcome + run manifest
```

The entropy tools are advisory in adaptive conversations: the ReAct model may
consult or override them. Controlled policies enforce question and terminality
boundaries with tool guards and post-turn fallbacks.

## Repository layout

```text
configs/                    category configuration and curated REC slates
data/categories/            catalogs, personas, questions, and index metadata
docs/                       evaluation protocol, Bouchet runbook, decisions
runs/bouchet_day/           Bouchet sync, setup, Slurm, submit, and fetch helpers
scripts/                    CLIs, diagnostics, and Streamlit entry points
src/sandbox/                agent, simulator, retrieval, scraper, and tool code
results/                    local, cluster, and archived evaluation artifacts
```

## Adding or refreshing a category

1. Add `configs/<category>.yaml`.
2. Scrape products with `uv run python scripts/scrape_category.py <category>`.
3. Add `questions.yaml` and `personas.json`.
4. Curate exactly three valid ASINs in `default_slate` before examining outcomes.
5. Build the index with `uv run python scripts/build_index.py <category>`.
6. Run `uv run python scripts/check_eval_setup.py`.

`products.jsonl` is the source catalog. Index building creates a shared text
view and keeps embedding rows aligned through `index/products.json` and
`index/index_manifest.json`. Rebuild after scraping or changing the product
serialization.

## Important limitations

- The selector cannot recover a relevant product omitted by all retrieval lanes.
- Product snapshots, prices, ratings, and availability drift over time.
- The buyer's purchase decision and WTP are simulated judgments, not observed
  consumer behavior.
- `rank_by_commission` and the uncertainty thresholds are hand-tuned prototype
  mechanisms.
- With only two categories, semantic feasibility margins are weak and rely on a
  keyword fast path.

For the precise experiment contract, read
[docs/EVALUATION_PROTOCOL.md](docs/EVALUATION_PROTOCOL.md). For cluster
operation, read [docs/BOUCHET_RUNBOOK.md](docs/BOUCHET_RUNBOOK.md). Current
design rationale and caveats live in [docs/DECISIONS.md](docs/DECISIONS.md).
