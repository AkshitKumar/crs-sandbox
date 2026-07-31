"""Configured question sequence shared by adaptive and ATR conversations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import yaml

from sandbox.catalog import REPO_ROOT, load_config


@dataclass(frozen=True)
class Question:
    id: str
    topic: str
    text: str
    tier: str


@dataclass
class QuestionBank:
    category: str
    questions: list[Question]
    asked_ids: set[str] = field(default_factory=set)

    @classmethod
    def load(cls, category: str) -> "QuestionBank":
        path = REPO_ROOT / load_config(category)["questions_path"]
        data: dict[str, Any] = yaml.safe_load(path.read_text())
        questions = [
            Question(str(item["id"]), str(item["topic"]), str(item["text"]), tier)
            for key, tier in (("openers", "opener"), ("followups", "followup"))
            for item in (data.get(key) or [])
        ]
        return cls(category=category, questions=questions)

    def next(self, topic: str | None = None, *, fixed: bool = False) -> Question | None:
        remaining = [question for question in self.questions if question.id not in self.asked_ids]
        if not remaining:
            return None
        openers = [question for question in remaining if question.tier == "opener"]
        if openers:
            selected = openers[0]
        elif not fixed and topic:
            selected = next((question for question in remaining if question.topic == topic), remaining[0])
        else:
            selected = remaining[0]
        self.asked_ids.add(selected.id)
        return selected

    def topics(self) -> list[str]:
        return list(dict.fromkeys(question.topic for question in self.questions))
