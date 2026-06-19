"""Fixed elicitation policies for controlled CRS evaluation runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


PolicyName = Literal["rec", "atr", "atr_recs"]


@dataclass(frozen=True)
class ElicitationPolicy:
    """Policy that fixes how many questions the CRS should ask before recommending."""

    name: PolicyName
    target_asks: int

    @property
    def label(self) -> str:
        if self.name == "rec":
            return "rec"
        if self.name == "atr_recs":
            return f"atr_recs{self.target_asks}"
        return f"atr{self.target_asks}"

    @property
    def allows_early_recommendations(self) -> bool:
        return self.name == "atr_recs"


def make_policy(name: str, numquestions: int | None = None) -> ElicitationPolicy:
    """Build and validate an elicitation policy from CLI-style arguments."""
    if name == "rec":
        if numquestions is not None:
            raise ValueError("--numquestions is only valid with --policy atr or atr_recs")
        return ElicitationPolicy(name="rec", target_asks=0)

    if name in {"atr", "atr_recs"}:
        if numquestions is None:
            raise ValueError(f"--policy {name} requires --numquestions")
        if numquestions < 0:
            raise ValueError("--numquestions must be non-negative")
        return ElicitationPolicy(name=name, target_asks=numquestions)

    raise ValueError("--policy must be one of: rec, atr, atr_recs")
