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

Catalog edits or changes to `src/sandbox/product_text.py` require rebuilding the
affected index. Before a Bouchet run, transfer the catalogs and matching index
manifests/embeddings together:

```bash
bash runs/bouchet_day/product_sync.sh
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
  --policy branching_atr --numquestions 7 --n-personas 30 --assortment-size 3

# The same branching evaluation with live abandonment decisions before questions
uv run python scripts/run_eval.py laptop \
  --policy branching_atr --numquestions 7 --n-personas 30 \
  --assortment-size 3 --endogenous-abandonment

# Run only persona-file positions 51 through 100 (inclusive)
uv run python scripts/run_eval.py laptop \
  --policy branching_atr --numquestions 7 --persona-range 51-100
```

Persona ranges are 1-based positions in `personas.json`. `--persona-ids` remains
available for selecting specific IDs instead of a contiguous range.

Successful ordinary runs contain only `transcripts.jsonl` and `summary.json`. An
incomplete run may retain `transcripts.partial.jsonl` for recovery.

The default local concurrency is four. Increase `--parallel` deliberately when
the host and API rate limits allow it; Bouchet's evaluation wrapper defaults to
20.

## Fixed question order

The current eight-question laptop sequence is use case, budget, RAM, screen
size, storage, battery, operating-system preference, and a final catch-all for
remaining must-haves or dealbreakers. The seven-question air-purifier sequence
is primary concern, budget, room size, filtration requirements, noise,
durability, and filter/maintenance cost. Fixed ATR policies use this exact YAML
order; adaptive conversations select from the same bank.

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

`recommend()` always reads the exact visible dialogue, derives a temporary
focused query and optional explicit price ceiling, runs fresh full-catalog hybrid
retrieval, and adds the category's configured default slate to the retrieved `k`
candidates. At each branching ATR checkpoint after the first, the immediately
preceding recommendations are also added to the reranker pool by default. The
pool is deduplicated, and prior products that violate a newly revealed hard price
ceiling are omitted. Eligible prior recommendations are identified to the
reranker so that it retains one when it remains among the best overall fits and
replaces it when newly revealed information makes another product meaningfully
better. No hidden purchase or willingness-to-pay information is used. An API
model then selects and explains the configured assortment (three products by
default).

`rec`, `single_atr`, `branching_atr`, and adaptive recommendations all use that
same operation.

The retrieval/index serializer and the final API evidence cards are deliberately
different. Retrieval text contains title, brand, price, average rating,
specifications, up to six feature bullets, and description; it does not contain
rating counts or review excerpts. Final selection and purchase calls share a
richer product-card renderer with rating count, up to ten feature bullets
(bounded to 5,500 characters), specifications, and up to three review excerpts.

## Counterfactual endogenous abandonment

A completed non-abandonment `branching_atr` run can be replayed through the
production abandonment evaluator without regenerating its answers,
recommendations, checkpoint purchases, or other API draws:

```bash
# Validate the source and show the maximum request count
uv run python scripts/run_counterfactual_abandonment.py \
  laptop_branching_atr7_k3_p1-100_JOBID --dry-run

# Apply abandonment sequentially within each persona, with 20 personas in flight
uv run python scripts/run_counterfactual_abandonment.py \
  laptop_branching_atr7_k3_p1-100_JOBID --parallel 20
```

At question `q`, the evaluator sees only the matching persona, the opener,
earlier question/answer pairs, and the current unanswered question. It stops at
the first abandonment. The derived directory contains `transcripts.jsonl`,
`abandonment_overlay.jsonl`, and a `crs-counterfactual-abandonment-v1`
`summary.json`, so it cannot be confused with a live run.

Normal summaries record hashes of the catalog, personas, questions, and the
buyer, recommender, simulation, and evaluation source files. Counterfactual
summaries additionally hash their source result files and abandonment prompt.

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
scripts/run_counterfactual_abandonment.py  derived abandonment replay
```
