# Conversational recommender simulation

This repository evaluates an API-backed conversational recommender against
persona-conditioned API buyers across six catalogs: laptops, air purifiers, bags,
briefcases, gloves, and can openers. It supports
an adaptive agent and controlled ask-then-recommend (ATR) policies that measure
how recommendations change as more buyer answers become available.

Buyer answers, purchase decisions, and product selection after elicitation use
the OpenAI API. The initial recommendation uses a configured fixed slate.
Catalog retrieval, validation, and aggregation run locally.

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

Both buyer and recommender models default to `gpt-6-luna`. Abandonment is off by
default; raw runs do not apply an abandonment overlay.

The default local concurrency is four. Increase `--parallel` deliberately when
the host and API rate limits allow it; Bouchet's evaluation wrapper defaults to
20.

## Fixed question order

The current seven-question laptop sequence is use case, budget, screen size,
combined memory/storage preferences, battery, operating-system preference, and
a final catch-all for remaining must-haves or dealbreakers.
The seven-question air-purifier sequence is primary concern, budget, room size,
filtration requirements, noise, durability, and the same final catch-all.
Filter/maintenance cost is question eight. Fixed ATR policies use this exact
YAML order. Adaptive conversations use the fixed openers before selecting
follow-up questions from the same bank.

## Demo

```bash
uv run streamlit run scripts/demo.py --server.port=8501
```

The demo can run a simulation live, replay a saved conversation with timed
streaming, or compare saved run summaries.

## Recommendation behavior

The recommender selects the three best fits for the shopper's expressed preferences.
It keeps earlier recommendations when new answers do not distinguish the options.

The adaptive agent has five tools:

- `catalog_overview`
- `search_catalog`
- `get_products`
- `ask_question`
- `recommend`

After elicitation, `recommend()` reads the visible dialogue and generates a
2–4 word product search query, plus a numeric target and ceiling only when
explicitly stated. The price floor is computed as `0.4 * target_price`; there is
no floor without a target. Fresh full-catalog hybrid retrieval returns up to
15 products by default. Eligible default-slate products and, for branching ATR,
the immediately preceding recommendations are added to the candidate pool.
All candidates must satisfy the current price floor and ceiling. The pool is
deduplicated before API selection of three products by default.

The continuity instruction is:

> Keep a [PREVIOUSLY RECOMMENDED] product if it is still among the best fits for
> what the shopper has said, and when the shopper's answers don't distinguish
> products, keep the earlier ones.

The recommender sees no private needs or hidden purchase outcomes. There is one
default best-fit recommendation prompt, with no conversion or revenue objective
switch. Immediate recommendations at checkpoint zero use the fixed default
slate directly; later recommendations use the operation above.

The retrieval/index serializer and the final API evidence cards are deliberately
different. Retrieval text contains title, brand, price, average rating,
specifications, up to six feature bullets, and description; it does not contain
rating counts or review excerpts. Final selection and purchase calls share a
richer product-card renderer with rating count, up to ten feature bullets
(bounded to 5,500 characters), specifications bounded to 2,200 characters, and
up to three review excerpts of 500 characters each. Truncated review excerpts
end in `...`. The scraper removes video-player controls and skips review text
that still contains video-player interface markers.

## Purchase evaluation

Each checkpoint is an independent purchase decision using the shopper's full
private needs, background, visible conversation, and offered product cards.
Earlier purchase decisions are not supplied to later checkpoints, and hidden
recommendations do not appear in the continuing buyer conversation.

After the offered product cards, the purchase instruction is:

```text
Decide as this shopper using their private needs and conversation. Ignore products above a hard budget. A product qualifies when its listing plausibly covers each top need; ordinary uses don't require strong evidence of ordinary features. Preferences and small misses can be traded off. Choose PURCHASE with the best qualifying product_number, or NO_PURCHASE if none qualifies.
```

## Bouchet: 100 personas, three repeats, two categories

On the local machine, sync code and matching catalog/index files:

```bash
bash runs/bouchet_day/sync_to_bouchet.sh
CATEGORIES="laptop air_purifier" bash runs/bouchet_day/product_sync.sh
```

On Bouchet, submit six raw branching runs. These jobs run sequentially, with
20 personas in flight within each job. Each category uses the same persona
positions 1–100 for three fresh repeats and checkpoints 0–7. No abandonment
calls or overlays are included.

```bash
bash <<'BASH'
set -euo pipefail
cd /nfs/roberts/project/pi_vh69/at2387/crs-sandbox

run_root="$PWD/results/final/results_$(date +%Y-%m-%d_%H%M%S)"
mkdir -p "$run_root" logs
previous_job=""

for repetition in 1 2 3; do
  for category in laptop air_purifier; do
    dependency_args=()
    if [[ -n "$previous_job" ]]; then
      dependency_args+=(--dependency="afterok:$previous_job")
    fi
    submitted=$(sbatch --parsable \
      "${dependency_args[@]}" \
      --job-name="${category}_100_r${repetition}" \
      --export="ALL,CATEGORY=${category},POLICY=branching_atr,NUMQUESTIONS=7,N_PERSONAS=100,PERSONA_RANGE=1-100,PERSONA_IDS=,PARALLEL=20,RETRIEVAL_K=15,ASSORTMENT_SIZE=3,MAX_TURNS=16,ENDOGENOUS_ABANDONMENT=0,FAIL_ON_PROTOCOL_ERROR=0,RUN_ROOT=${run_root}" \
      runs/bouchet_day/run_eval.sbatch)
    job_id="${submitted%%;*}"
    [[ "$job_id" =~ ^[0-9]+$ ]]
    printf 'Repeat %s | %s | job %s\n' "$repetition" "$category" "$job_id"
    printf '%s\t%s\t%s\n' "$repetition" "$category" "$job_id" >> "$run_root/jobs.tsv"
    previous_job="$job_id"
  done
done
printf 'Results: %s\n' "$run_root"
BASH
```

An unsuccessful job blocks later jobs in the dependency chain. After completion,
fetch results from the local machine:

```bash
bash runs/bouchet_day/fetch_final_results.sh
```

The separate `submit_final_postabandon.sh` helper also submits abandonment
replays; use the commands above for raw runs only.

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
