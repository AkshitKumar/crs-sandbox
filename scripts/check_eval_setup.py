"""Validate evaluation inputs and graph wiring without making provider calls."""

from __future__ import annotations

import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from sandbox.agents.langgraph_crs import CRSAgentSession  # noqa: E402
from sandbox.catalog import load_catalog, load_config  # noqa: E402
from sandbox.elicitation_policy import make_policy  # noqa: E402
from sandbox.orchestrator.sim_conversation import SimConversation  # noqa: E402
from sandbox.tools.search_tool import _load_index  # noqa: E402


CATEGORIES = ("laptop", "air_purifier")


def _validate_category(category: str) -> None:
    config = load_config(category)
    catalog = list(load_catalog(category))
    by_asin = {product.get("asin"): product for product in catalog}
    if len(by_asin) != len(catalog) or None in by_asin:
        raise ValueError(f"{category}: catalog has missing or duplicate ASINs")

    slate = list(config.get("default_slate") or [])
    if len(slate) != 3 or len(set(slate)) != 3:
        raise ValueError(f"{category}: default_slate must contain three unique ASINs")
    missing = [asin for asin in slate if asin not in by_asin]
    if missing:
        raise ValueError(f"{category}: default_slate ASINs missing from catalog: {missing}")

    questions_path = REPO_ROOT / config["questions_path"]
    personas_path = questions_path.with_name("personas.json")
    if not questions_path.is_file() or not personas_path.is_file():
        raise FileNotFoundError(f"{category}: questions/personas input is missing")

    embeddings, indexed_products = _load_index(category)
    if len(indexed_products) != len(catalog) or embeddings.shape[0] != len(catalog):
        raise ValueError(f"{category}: dense index does not cover the current catalog")


def _validate_agent_wiring() -> None:
    # ChatOpenAI validates that a key exists when the graph is constructed, but
    # construction itself makes no request. Never replace a real key.
    os.environ.setdefault("OPENAI_API_KEY", "preflight-no-provider-call")

    adaptive = SimConversation(
        persona={"id": "preflight"},
        category="laptop",
        buyer=object(),
    )
    if not isinstance(adaptive.crs, CRSAgentSession) or adaptive.crs.category is not None:
        raise ValueError("adaptive simulation is not routed through an unset-category CRSAgentSession")

    for name, numquestions in (("rec", None), ("single_atr", 1), ("checkpoint_atr", 2)):
        policy = make_policy(name, numquestions)
        controlled = SimConversation(
            persona={"id": "preflight"},
            category="laptop",
            buyer=object(),
            elicitation_policy=policy,
        )
        if not isinstance(controlled.crs, CRSAgentSession):
            raise ValueError(f"{name}: simulation is not routed through CRSAgentSession")
        if controlled.crs.category != "laptop" or controlled.crs.elicitation_policy != policy:
            raise ValueError(f"{name}: category/policy was not installed on CRSAgentSession")
        controlled.crs._build_agent()


def main() -> int:
    for category in CATEGORIES:
        _validate_category(category)
        print(f"ok: {category} catalog, slate, questions, personas, and dense index")
    _validate_agent_wiring()
    print("ok: adaptive, REC, single_atr, and checkpoint_atr use CRSAgentSession")
    print("evaluation setup is internally consistent; no provider call was made")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
