"""CandidateBus — the in-memory candidate set shared across tool calls.

Borrowed from Microsoft InteRecAgent's pattern: each tool that touches the
candidate set restricts/reranks it and appends a one-line audit note. The
agent and the chat UI both read the note trail; the underlying ASINs are
stored compactly so the LLM doesn't have to keep huge lists in context.

Lifecycle within a conversation:
    bus = CandidateBus.full(category, catalog)        # all products
    bus = filter_tool.apply(bus, {"price_max": 1100}) # 167 → 84
    bus = semantic_search.apply(bus, query, top_k=20) # 84 → 20
    bus = rank_by_match.apply(bus, weights)           # reorder
    top3 = bus.top(3)                                 # what to show user
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class CandidateBus:
    category: str
    asins: list[str] = field(default_factory=list)
    # Audit trail of tool operations. One short line per call.
    notes: list[str] = field(default_factory=list)
    # Per-asin metadata (scores from the last ranking call, etc.). Optional.
    scores: dict[str, float] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def full(cls, category: str, catalog: list[dict[str, Any]]) -> "CandidateBus":
        """Initialize with every product in the catalog as a candidate."""
        return cls(
            category=category,
            asins=[p["asin"] for p in catalog if p.get("asin")],
            notes=[f"init: {len(catalog)} products"],
        )

    @classmethod
    def from_asins(cls, category: str, asins: list[str], note: str = "from_asins") -> "CandidateBus":
        return cls(category=category, asins=list(asins), notes=[f"{note}: {len(asins)} items"])

    # ------------------------------------------------------------------
    # Mutating ops (return self for chaining)
    # ------------------------------------------------------------------

    def restrict(self, kept_asins: list[str], note: str) -> "CandidateBus":
        """Narrow the bus to a subset. Preserves the order in `kept_asins`."""
        kept_set = set(kept_asins)
        before = len(self.asins)
        # Preserve incoming order, not current bus order.
        self.asins = [a for a in kept_asins if a in kept_set]
        self.notes.append(f"{note}: {before} → {len(self.asins)}")
        # Drop scores for asins no longer in bus.
        self.scores = {a: s for a, s in self.scores.items() if a in kept_set}
        return self

    def reorder(self, ordered_asins: list[str], scores: dict[str, float] | None, note: str) -> "CandidateBus":
        """Reorder bus by a new ranking. ASIN set must equal current asin set."""
        current = set(self.asins)
        new = set(ordered_asins)
        if current != new:
            # Don't error out — just intersect, since some scorers might drop items.
            ordered_asins = [a for a in ordered_asins if a in current]
        self.asins = ordered_asins
        if scores is not None:
            self.scores = {a: scores.get(a, 0.0) for a in ordered_asins}
        self.notes.append(f"{note}: reranked {len(self.asins)}")
        return self

    def annotate(self, note: str) -> "CandidateBus":
        """Append a free-form audit note without changing the bus contents."""
        self.notes.append(note)
        return self

    # ------------------------------------------------------------------
    # Read-only ops
    # ------------------------------------------------------------------

    def size(self) -> int:
        return len(self.asins)

    def top(self, k: int) -> list[str]:
        return self.asins[:k]

    def products(self, catalog: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Materialize full product records for current ASINs, preserving bus order."""
        by_asin = {p["asin"]: p for p in catalog if p.get("asin")}
        return [by_asin[a] for a in self.asins if a in by_asin]

    def to_dict(self) -> dict[str, Any]:
        """Serialize for chat-UI display and logging."""
        return asdict(self)
