"""compute_uncertainty + suggest_next_action — the ask-vs-recommend gate.

Following Frey et al. 2025 (arxiv 2509.06185): we treat the entropy of the
softmaxed retrieval score distribution as a numerical signal for how
ambiguous the current user need is over the candidate set.

    p_i = exp(score_i / τ) / Σ exp(score_j / τ)
    H   = -Σ p_i log p_i

Low H → a few products clearly dominate → agent should RECOMMEND.
High H → scores are spread → agent should ASK a clarifying question.

We also expose two side signals:
    - `top_k_diversity`: per-spec-attribute variance across the top-K
      candidates. If the top-K split on one attribute (e.g., screen size
      varies from 13" to 18"), that's the attribute to ask about next.
    - `score_gap`: gap between top-1 and the K-th score, normalized.

`suggest_next_action` composites these with the running ask budget into a
recommendation the agent can take or override.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np

from sandbox.catalog import load_catalog
from sandbox.tools.candidate_bus import CandidateBus


# Default temperature for the softmax. With L2-normalized bge cosine scores
# typically in [0.4, 0.8], τ=0.05 gives a usable spread between low and high
# entropy regimes. Calibrate per category if needed.
DEFAULT_TAU = 0.05


@dataclass
class UncertaintySignals:
    entropy: float
    entropy_max: float                  # log(K) — upper bound for top-K of size K
    entropy_normalized: float           # H / log(K) ∈ [0, 1]
    score_gap: float                    # (top1 - topK) / top1
    top_k_diversity: dict[str, int]     # attribute → distinct value count in top-K
    top_k_size: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "entropy": round(self.entropy, 4),
            "entropy_normalized": round(self.entropy_normalized, 4),
            "score_gap": round(self.score_gap, 4),
            "top_k_size": self.top_k_size,
            "top_k_diversity": dict(sorted(self.top_k_diversity.items(), key=lambda x: -x[1])[:8]),
        }


def compute_uncertainty(
    bus: CandidateBus,
    top_k: int = 10,
    tau: float = DEFAULT_TAU,
    diversity_fields: list[str] | None = None,
) -> UncertaintySignals:
    """Compute uncertainty signals over the current bus.

    Uses scores already in `bus.scores` (set by the most recent search/rank
    call). If scores are missing or all identical, entropy reaches its max.

    `diversity_fields`: which spec-table attributes to analyze across the
    top-K. If None, picks the 6 most-frequent fields in the catalog.
    """
    asins = bus.top(top_k)
    K = len(asins)
    if K == 0:
        return UncertaintySignals(0.0, 0.0, 0.0, 0.0, {}, 0)

    scores = np.array([bus.scores.get(a, 0.0) for a in asins], dtype=np.float64)
    # Softmax with temperature
    s = scores / max(tau, 1e-6)
    s = s - s.max()  # numerical stability
    p = np.exp(s)
    p = p / p.sum()
    # Entropy in nats
    eps = 1e-12
    H = float(-np.sum(p * np.log(p + eps)))
    H_max = float(np.log(K))
    H_norm = H / H_max if H_max > 0 else 0.0

    # Score gap: how much better is #1 than #K?
    top1, topk = float(scores[0]), float(scores[-1])
    score_gap = (top1 - topk) / max(abs(top1), 1e-6) if top1 != topk else 0.0

    # Diversity: distinct value count per spec field across top-K.
    catalog = list(load_catalog(bus.category))
    by_asin = {p["asin"]: p for p in catalog if p.get("asin")}
    if diversity_fields is None:
        # Pick fields populated in most of the top-K.
        field_pop: Counter[str] = Counter()
        for a in asins:
            for f in (by_asin.get(a, {}).get("spec_table") or {}):
                field_pop[f] += 1
        diversity_fields = [f for f, _ in field_pop.most_common(10)]
    diversity: dict[str, int] = {}
    for f in diversity_fields:
        vals = set()
        for a in asins:
            v = (by_asin.get(a, {}).get("spec_table") or {}).get(f)
            if v:
                vals.add(v.strip().lower())
        if len(vals) > 1:
            diversity[f] = len(vals)

    return UncertaintySignals(
        entropy=H,
        entropy_max=H_max,
        entropy_normalized=H_norm,
        score_gap=score_gap,
        top_k_diversity=diversity,
        top_k_size=K,
    )


# ---------------------------------------------------------------------------
# Composite policy: ASK / RECOMMEND / KEEP_ASKING
# ---------------------------------------------------------------------------


@dataclass
class ActionThresholds:
    h_recommend: float = 0.40   # H_normalized below this → recommend
    h_ask: float = 0.80         # H_normalized above this → ask
    min_asks_before_recommend: int = 2
    max_asks: int = 8


def suggest_next_action(
    bus: CandidateBus,
    asks_so_far: int,
    thresholds: ActionThresholds | None = None,
    top_k: int = 10,
    tau: float = DEFAULT_TAU,
) -> dict[str, Any]:
    """Composite ask-vs-recommend recommendation. Agent may follow or override."""
    thresholds = thresholds or ActionThresholds()
    signals = compute_uncertainty(bus, top_k=top_k, tau=tau)

    decision: str
    reason: str
    if asks_so_far < thresholds.min_asks_before_recommend:
        decision, reason = "ASK", f"need at least {thresholds.min_asks_before_recommend} asks (so far {asks_so_far})"
    elif asks_so_far >= thresholds.max_asks:
        decision, reason = "RECOMMEND", f"hit max asks ({thresholds.max_asks})"
    elif signals.entropy_normalized < thresholds.h_recommend:
        decision, reason = "RECOMMEND", f"low entropy ({signals.entropy_normalized:.2f} < {thresholds.h_recommend})"
    elif signals.entropy_normalized > thresholds.h_ask:
        decision, reason = "ASK", f"high entropy ({signals.entropy_normalized:.2f} > {thresholds.h_ask})"
    else:
        decision, reason = "KEEP_ASKING", f"borderline entropy ({signals.entropy_normalized:.2f})"

    return {
        "action": decision,
        "reason": reason,
        "signals": signals.to_dict(),
        "asks_so_far": asks_so_far,
    }
