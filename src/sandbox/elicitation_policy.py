"""Fixed elicitation policies for controlled CRS evaluation runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


PolicyName = Literal["rec", "atr", "checkpoint_atr"]


@dataclass(frozen=True)
class ElicitationPolicy:
    """Policy that fixes how many questions the CRS should ask before recommending."""

    name: PolicyName
    target_asks: int

    @property
    def label(self) -> str:
        if self.name == "rec":
            return "rec"
        if self.name == "checkpoint_atr":
            return f"checkpoint_atr{self.target_asks}"
        return f"atr{self.target_asks}"

    @property
    def allows_early_recommendations(self) -> bool:
        """Controlled policies never expose a recommendation before terminality."""
        return False

    @property
    def has_nonterminal_checkpoints(self) -> bool:
        """Whether to evaluate hidden recommendation snapshots at each prefix."""
        return self.name == "checkpoint_atr"


def make_policy(name: str, numquestions: int | None = None) -> ElicitationPolicy:
    """Build and validate an elicitation policy from CLI-style arguments."""
    if name == "rec":
        if numquestions is not None:
            raise ValueError("--numquestions is only valid with --policy atr or checkpoint_atr")
        return ElicitationPolicy(name="rec", target_asks=0)

    if name == "atr_recs":
        raise ValueError(
            "--policy atr_recs names the retired visible-card treatment; use --policy checkpoint_atr "
            "for hidden nonterminal recommendation checkpoints"
        )

    if name in {"atr", "checkpoint_atr"}:
        if numquestions is None:
            raise ValueError(f"--policy {name} requires --numquestions")
        if numquestions < 0:
            raise ValueError("--numquestions must be non-negative")
        return ElicitationPolicy(name=name, target_asks=numquestions)

    raise ValueError("--policy must be one of: rec, atr, checkpoint_atr")
