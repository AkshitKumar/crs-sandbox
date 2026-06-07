"""Phase B integration test: simulate the agent's tool-call sequence by hand,
against a real persona and the laptop catalog.

This verifies that all tools compose correctly into a sensible pipeline
before we wrap them in the LangGraph agent.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from sandbox.env import load_env
from sandbox.catalog import load_catalog, load_config  # noqa: E402
from sandbox.tools.candidate_bus import CandidateBus
from sandbox.tools.filter_tool import apply_filter, available_filters
from sandbox.tools.search_tool import semantic_search, narrow_search
from sandbox.tools.ranking_tool import rank_by_match, rank_by_commission, MatchWeights
from sandbox.tools.uncertainty_tool import compute_uncertainty, suggest_next_action
from sandbox.tools.inspect_tool import get_product_details, compare, catalog_overview
from sandbox.tools.feasibility_tool import check_category_supported
from sandbox.tools.review_tool import summarize_reviews  # only used at recommend time


def hr(title: str) -> None:
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")


def main() -> int:
    load_env()

    hr("STEP 0: User says 'I'm looking for a new computer for college'")
    feas = check_category_supported("I'm looking for a new computer for college")
    print(f"  best_category={feas['best_category']}  score={feas['best_score']:.3f}  supported={feas['supported']}")
    print(f"  ranked: {[(c, round(s, 3)) for c, s in feas['ranked']]}")
    category = feas["best_category"] or "laptop"

    hr(f"STEP 1: catalog_overview({category!r}) — orient the agent")
    ov = catalog_overview(category)
    print(f"  n={ov['n_products']}, price ${ov['price_min']}–${ov['price_max']} (median ${ov['price_median']}, p75 ${ov['price_p75']})")
    print(f"  top brands: {[b['brand'] for b in ov['top_brands'][:6]]}")

    hr("STEP 2: available_filters — what can we constrain on")
    af = available_filters(category)
    print(f"  filterable spec fields (top 8 by coverage):")
    for f, info in list(af["spec_fields"].items())[:8]:
        print(f"    {f:35s} cov={info['coverage_pct']:5.1f}%  examples: {info['example_values'][:3]}")

    hr("STEP 3: User adds 'mostly schoolwork plus gaming, around $1000, has to handle modern games'")
    bus = CandidateBus.full(category, list(load_catalog(category)))

    print(f"  3a. semantic_search to seed the bus...")
    bus = semantic_search(
        bus,
        "laptop for college student that handles schoolwork plus modern gaming, around $1000",
        top_k=40,
    )
    print(f"      bus.size: {bus.size()}")

    print(f"\n  3b. filter on hard constraints (price≤$1100, dedicated GPU, rating≥4.0)...")
    bus = apply_filter(
        bus,
        {"price_max": 1100, "rating_min": 4.0, "spec_contains": {"Graphics Description": "Dedicated"}},
    )
    print(f"      bus.size: {bus.size()}")

    print(f"\n  3c. compute_uncertainty after filter+search — should we recommend yet?")
    sig = compute_uncertainty(bus, top_k=min(10, bus.size()))
    print(f"      entropy_norm={sig.entropy_normalized:.3f}  score_gap={sig.score_gap:.3f}")
    print(f"      diversity attrs in top-K: {list(sig.top_k_diversity.items())[:5]}")

    action = suggest_next_action(bus, asks_so_far=2)
    print(f"      suggested action: {action['action']}  ({action['reason']})")

    hr("STEP 4: User adds 'lighter the better' — re-rank with narrow_search")
    bus = narrow_search(
        bus,
        "lightweight gaming laptop with dedicated GPU, around $1000",
        top_k=10,
    )
    print(f"  bus.size: {bus.size()}")

    hr("STEP 5: rank_by_match — weight semantic + rating")
    bus = rank_by_match(bus, MatchWeights(semantic=1.0, rating=0.3, popularity=0.1))
    cat = list(load_catalog(category))
    print(f"  Top 3 by match score:")
    for i, asin in enumerate(bus.top(3), 1):
        p = next(p for p in cat if p["asin"] == asin)
        print(f"    #{i}  ${p['price']:>6.0f}  ★{p['avg_rating']:.1f} ({p['num_reviews']:>4})  {p['title'][:75]}")

    hr("STEP 6: compare top 3 side-by-side")
    top3 = bus.top(3)
    cmp = compare(category, top3, aspects=[
        "title", "brand", "price", "avg_rating",
        "Screen Size", "RAM Memory Installed", "Hard-Drive Size",
        "Graphics Description", "Operating System", "Item Weight",
    ])
    print(f"  {'aspect':25s}  " + "  ".join(f"{'#'+str(i+1):<32s}" for i in range(len(top3))))
    for asp, row in cmp["table"].items():
        cells = "  ".join(f"{(str(v)[:30] if v is not None else '—'):<32s}" for v in row)
        print(f"  {asp:25s}  {cells}")

    hr("STEP 7: summarize_reviews on the top recommendation (caches result)")
    rev = summarize_reviews(category, top3[0], aspect=None)
    print(f"  sentiment_score: {rev['sentiment_score']}")
    print(f"  summary: {rev['summary']}")
    print(f"  evidence: {rev['evidence']}")

    hr("OFF-CATALOG sanity check")
    for q in ["power drill", "garden hose", "wireless headphones for running"]:
        f = check_category_supported(q)
        print(f"  query={q!r:40s}  best={f['best_category']!r:20s}  score={f['best_score']:.3f}  supported={f['supported']}")

    print("\n[done] All tools composed without errors.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
