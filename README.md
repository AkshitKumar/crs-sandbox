# Conversational recommender simulation

This repository evaluates an API-backed conversational recommender against
persona-conditioned API buyers for laptop and air-purifier catalogs. It supports
an adaptive agent and controlled ask-then-recommend (ATR) policies that measure
how recommendations change as more buyer answers become available.

No evaluation outcome is generated locally or deterministically. Buyer answers,
recommendations, purchases, and willingness to pay require a valid OpenAI API
key. Local code only controls the conversation, catalog retrieval, validation,
and aggregation.

## Setup

```bash
uv venv --python 3.12
uv pip install -r requirements.txt
cp .env.example .env
# Set OPENAI_API_KEY in .env
```

Build the local indices before the first run and after changing a catalog:

```bash
uv run python scripts/build_index.py laptop
uv run python scripts/build_index.py air_purifier
```

## Run evaluations

```bash
# Agent decides how many questions to ask
uv run python scripts/run_eval.py laptop --policy adaptive --n-personas 10

# Recommend immediately
uv run python scripts/run_eval.py laptop --policy rec --n-personas 10

# Ask exactly four configured questions, then recommend
uv run python scripts/run_eval.py laptop \
  --policy single_atr --numquestions 4 --n-personas 10

# One visible conversation plus hidden recommendation branches at depths 0...7
uv run python scripts/run_eval.py laptop \
  --policy branching_atr --numquestions 7 --n-personas 30

# Run only persona-file positions 51 through 100 (inclusive)
uv run python scripts/run_eval.py laptop \
  --policy branching_atr --numquestions 7 --persona-range 51-100
```

Persona ranges are 1-based positions in `personas.json`. `--persona-ids` remains
available for selecting specific IDs instead of a contiguous range.

Successful runs contain only `transcripts.jsonl` and `summary.json`. An
incomplete run may retain `transcripts.partial.jsonl` for recovery.

## Demo

```bash
uv run streamlit run scripts/demo.py --server.port=8501
```

The demo can run a simulation live, replay a saved conversation with timed
streaming, or compare saved run summaries.

## Recommendation behavior

The adaptive agent has five tools:

- `catalog_overview`
- `search_catalog`
- `get_products`
- `ask_question`
- `recommend`

Catalog exploration is stateless. `recommend()` always reads the exact visible
dialogue, derives a temporary focused query and optional explicit price ceiling,
runs fresh full-catalog hybrid retrieval, and adds the category's configured
default slate to the retrieved `k` candidates. After deduplication and budget
eligibility, an API model selects and explains the final three products.

`rec`, `single_atr`, `branching_atr`, and adaptive recommendations all use that
same operation.

## Repository layout

```text
configs/                    category configuration and default recall slate
data/categories/            products, personas, question banks, generated indices
src/sandbox/agents/         buyer and recommender behavior
src/sandbox/catalog.py      catalog loading and stateless hybrid retrieval
src/sandbox/simulation.py   one conversation and ATR branching
scripts/run_eval.py         parallel evaluation and aggregate output
scripts/demo.py             live simulation, replay, and comparison
scripts/scrape_category.py  catalog collection
```
