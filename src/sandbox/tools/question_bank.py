"""Per-category question bank exposed as a tool to the CRS agent.

Loaded from data/categories/<cat>/questions.yaml.

Contract:
    - Tier 1 (openers): asked in fixed order. The tool returns the next opener
      regardless of any `topic` argument until openers are exhausted.
    - Tier 2 (followups): topic-tagged pool. The CRS picks a topic; the tool
      returns an unasked question with that topic, or surfaces the list of
      uncovered topics if none specified.

This split gives clean ATR(k) semantics for early turns while letting the CRS
agent behave adaptively later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Question:
    id: str
    topic: str
    text: str
    tier: str  # "opener" or "followup"


@dataclass
class QuestionBank:
    openers: list[Question]
    followups_by_topic: dict[str, list[Question]]

    @classmethod
    def load(cls, path: Path) -> "QuestionBank":
        data: dict[str, Any] = yaml.safe_load(Path(path).read_text())
        openers = [
            Question(id=q["id"], topic=q["topic"], text=q["text"], tier="opener")
            for q in data.get("openers") or []
        ]
        followups_by_topic: dict[str, list[Question]] = {}
        for q in data.get("followups") or []:
            question = Question(
                id=q["id"], topic=q["topic"], text=q["text"], tier="followup"
            )
            followups_by_topic.setdefault(question.topic, []).append(question)
        return cls(openers=openers, followups_by_topic=followups_by_topic)

    def all_topics(self) -> list[str]:
        return list(self.followups_by_topic.keys())


@dataclass
class QuestionState:
    asked_ids: set[str] = field(default_factory=set)
    asked_topics: set[str] = field(default_factory=set)

    def mark_asked(self, question: Question) -> None:
        self.asked_ids.add(question.id)
        self.asked_topics.add(question.topic)


@dataclass
class QuestionTool:
    """Stateful wrapper around a bank for a single conversation."""

    bank: QuestionBank
    state: QuestionState = field(default_factory=QuestionState)

    # ------------------------------------------------------------------
    # Public surface (called by the CRS agent)
    # ------------------------------------------------------------------

    def ask(self, topic: str | None = None) -> dict[str, Any]:
        """Return the next question to ask the buyer.

        Returns a dict with:
            question_id: str         (the question's id)
            question_text: str       (what to say to the buyer)
            tier: "opener" | "followup"
            remaining_openers: int
            uncovered_topics: list[str]   (available topics for next followups)
        """
        # Openers are always next while any remain.
        opener = self._next_opener()
        if opener is not None:
            self.state.mark_asked(opener)
            return self._make_response(opener)

        # All openers done. Pick a followup.
        followup = self._next_followup(topic)
        if followup is None:
            return {
                "question_id": None,
                "question_text": None,
                "tier": None,
                "remaining_openers": 0,
                "uncovered_topics": self._uncovered_topics(),
                "note": "No questions remaining. Consider recommending.",
            }
        self.state.mark_asked(followup)
        return self._make_response(followup)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _next_opener(self) -> Question | None:
        for q in self.bank.openers:
            if q.id not in self.state.asked_ids:
                return q
        return None

    def _next_followup(self, topic: str | None) -> Question | None:
        if topic is not None:
            for q in self.bank.followups_by_topic.get(topic, []):
                if q.id not in self.state.asked_ids:
                    return q
            return None
        # No topic specified: prefer uncovered topics first, else fall through
        # to any remaining question.
        for t in self._uncovered_topics():
            for q in self.bank.followups_by_topic.get(t, []):
                if q.id not in self.state.asked_ids:
                    return q
        for qs in self.bank.followups_by_topic.values():
            for q in qs:
                if q.id not in self.state.asked_ids:
                    return q
        return None

    def _uncovered_topics(self) -> list[str]:
        return [
            t
            for t in self.bank.all_topics()
            if t not in self.state.asked_topics
            and any(q.id not in self.state.asked_ids for q in self.bank.followups_by_topic[t])
        ]

    def _make_response(self, q: Question) -> dict[str, Any]:
        return {
            "question_id": q.id,
            "question_text": q.text,
            "tier": q.tier,
            "remaining_openers": sum(
                1 for op in self.bank.openers if op.id not in self.state.asked_ids
            ),
            "uncovered_topics": self._uncovered_topics(),
        }
