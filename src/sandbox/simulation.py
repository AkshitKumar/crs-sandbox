"""One buyer/recommender conversation, including ATR counterfactual branches."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Iterator, Literal

from openai import OpenAI

from sandbox.agents.buyer import ABANDON_SENTINEL, BuyerAgent
from sandbox.catalog import category_with_article
from sandbox.agents.recommender import (
    RecommendationError,
    RecommendationResult,
    RecommendationService,
    RecommenderAgent,
    render_recommendations,
)
from sandbox.openai_responses import UsageTracker, make_client
from sandbox.questions import QuestionBank


PolicyName = Literal["adaptive", "rec", "single_atr", "branching_atr"]


@dataclass(frozen=True)
class Policy:
    name: PolicyName = "adaptive"
    numquestions: int = 0

    @classmethod
    def make(cls, name: str, numquestions: int | None = None) -> "Policy":
        if name not in {"adaptive", "rec", "single_atr", "branching_atr"}:
            raise ValueError("policy must be adaptive, rec, single_atr, or branching_atr")
        if name in {"adaptive", "rec"}:
            if numquestions not in {None, 0}:
                raise ValueError(f"{name} does not accept numquestions")
            return cls(name=name, numquestions=0)  # type: ignore[arg-type]
        if numquestions is None or numquestions < 0:
            raise ValueError(f"{name} requires a non-negative numquestions")
        return cls(name=name, numquestions=numquestions)  # type: ignore[arg-type]


@dataclass
class Outcome:
    persona_id: str
    category: str
    policy: str
    numquestions: int
    outcome: str
    turns_used: int
    asks: int
    purchased_asin: str | None = None
    wtp: float | None = None
    actual_price: float | None = None
    revenue: float = 0.0
    consumer_surplus: float = 0.0
    decision: dict[str, Any] | None = None
    dialogue: list[dict[str, str]] = field(default_factory=list)
    recommendation_result: dict[str, Any] | None = None
    question_ids: list[str] = field(default_factory=list)
    question_topics: list[str] = field(default_factory=list)
    tool_calls_per_turn: list[list[dict[str, Any]]] = field(default_factory=list)
    checkpoints: list[dict[str, Any]] = field(default_factory=list)
    engagement_decisions: list[dict[str, Any]] = field(default_factory=list)
    abandonment_reason: str | None = None
    error_code: str | None = None
    error: str | None = None
    api_usage: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["schema_version"] = "crs-transcript-v2"
        payload["recommendation_result"] = _compact_recommendation_result(
            payload.get("recommendation_result")
        )
        for checkpoint in payload.get("checkpoints") or []:
            if "recommendation_result" in checkpoint:
                checkpoint["recommendation_result"] = _compact_recommendation_result(
                    checkpoint.get("recommendation_result")
                )
        return payload


def _compact_recommendation_result(
    result: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Keep selection identity and rationale; catalog data is keyed by ASIN."""
    if result is None:
        return None
    compact = dict(result)
    compact["recommendations"] = [
        {
            "asin": product.get("asin"),
            "recommendation_explanation": product.get("recommendation_explanation"),
        }
        for product in result.get("recommendations") or []
    ]
    return compact


def _resolve_decision(
    decision: dict[str, Any], recommendations: list[dict[str, Any]]
) -> dict[str, Any]:
    if decision["decision"] == "NO_PURCHASE":
        return {
            "outcome": "NO_PURCHASE",
            "purchased_asin": None,
            "wtp": None,
            "actual_price": None,
            "revenue": 0.0,
            "consumer_surplus": 0.0,
        }
    number = decision["product_number"]
    product = recommendations[number - 1]
    price = product.get("price")
    if not isinstance(price, (int, float)):
        raise RecommendationError("PURCHASE_PRICE_MISSING", f"product {number} has no price")
    wtp = float(decision["willingness_to_pay"])
    actual_price = float(price)
    return {
        "outcome": "PURCHASE",
        "purchased_asin": product.get("asin"),
        "wtp": wtp,
        "actual_price": actual_price,
        "revenue": actual_price,
        "consumer_surplus": wtp - actual_price,
    }


@dataclass
class Simulation:
    persona: dict[str, Any]
    category: str
    policy: Policy = field(default_factory=Policy)
    max_turns: int = 16
    endogenous_abandonment: bool = False
    buyer_model: str = "gpt-5.6-luna"
    recommender_model: str = "gpt-5.6-luna"
    retrieval_limit: int = 15
    assortment_size: int = 3
    tracker: UsageTracker = field(default_factory=UsageTracker)
    client: OpenAI | None = None
    buyer: BuyerAgent | None = None
    recommender: RecommenderAgent | None = None

    def __post_init__(self) -> None:
        self.client = self.client or make_client()
        self.buyer = self.buyer or BuyerAgent(
            persona=self.persona,
            category=self.category,
            model=self.buyer_model,
            endogenous_abandonment=self.endogenous_abandonment,
            tracker=self.tracker,
            client=self.client,
        )
        if self.policy.name == "adaptive":
            self.recommender = self.recommender or RecommenderAgent(
                category=self.category,
                model=self.recommender_model,
                retrieval_limit=self.retrieval_limit,
                assortment_size=self.assortment_size,
                tracker=self.tracker,
                client=self.client,
            )
            self.service = self.recommender.service
            self.questions = self.recommender.questions
        else:
            self.service = RecommendationService(
                category=self.category,
                model=self.recommender_model,
                retrieval_limit=self.retrieval_limit,
                assortment_size=self.assortment_size,
                tracker=self.tracker,
                client=self.client,
            )
            self.questions = QuestionBank.load(self.category)
        if self.policy.numquestions > len(self.questions.questions):
            raise ValueError("policy requests more questions than the category defines")
        if self.policy.name != "adaptive" and self.max_turns < self.policy.numquestions + 1:
            raise ValueError("max_turns is too small for the controlled policy")
        self.dialogue: list[dict[str, str]] = []
        self.question_ids: list[str] = []
        self.question_topics: list[str] = []
        self.tool_calls: list[list[dict[str, Any]]] = []
        self.checkpoints: list[dict[str, Any]] = []
        self.engagement_decisions: list[dict[str, Any]] = []
        self.final_recommendation: RecommendationResult | None = None
        self.prior_recommendations: list[dict[str, Any]] = []

    def run(self) -> Outcome:
        outcome = None
        for event in self.run_iter():
            if event["type"] == "outcome":
                outcome = event["outcome"]
        assert outcome is not None
        return outcome

    def run_iter(self) -> Iterator[dict[str, Any]]:
        try:
            yield from self._run_iter()
        except Exception as exc:
            code = exc.code if isinstance(exc, RecommendationError) else type(exc).__name__.upper()
            yield {"type": "error", "error_code": code, "error": str(exc)}
            yield {
                "type": "outcome",
                "outcome": self._outcome(
                    outcome="PROTOCOL_ERROR",
                    error_code=code,
                    error=f"{type(exc).__name__}: {exc}",
                ),
            }

    def _run_iter(self) -> Iterator[dict[str, Any]]:
        opener = f"I'm looking for {category_with_article(self.category)}."
        self.dialogue.append({"role": "user", "content": opener})
        yield {"type": "buyer_message", "content": opener, "turn": 1}

        if self.policy.name == "adaptive":
            yield from self._run_adaptive(opener)
            return
        if self.policy.name == "rec":
            yield from self._finish_with_recommendation(self.service.recommend(self.dialogue))
            return
        if self.policy.name == "single_atr":
            for _ in range(self.policy.numquestions):
                abandoned = yield from self._ask_fixed()
                if abandoned:
                    return
            yield from self._finish_with_recommendation(self.service.recommend(self.dialogue))
            return
        yield from self._run_branching()

    def _run_adaptive(self, opener: str) -> Iterator[dict[str, Any]]:
        step = self.recommender.step(opener)
        self._adopt_adaptive_step(step)
        yield self._step_event(step)
        for _ in range(1, self.max_turns + 1):
            if step.recommendation is not None:
                yield from self._finish_with_recommendation(
                    step.recommendation,
                    recommendation_already_visible=True,
                )
                return
            abandonment_reason = self._abandonment_reason(step.reply)
            if abandonment_reason is not None:
                yield from self._finish_abandoned(
                    abandonment_reason,
                    turn=len(self.tool_calls) + 1,
                )
                return
            reply = self.buyer.respond(step.reply)
            self.dialogue.append({"role": "user", "content": reply})
            yield {"type": "buyer_message", "content": reply, "turn": len(self.tool_calls) + 1}
            if len(self.tool_calls) >= self.max_turns:
                break
            step = self.recommender.step(reply)
            self._adopt_adaptive_step(step)
            yield self._step_event(step)
        raise RecommendationError(
            "MAX_TURNS_WITHOUT_RECOMMENDATION",
            "adaptive recommender reached max_turns without recommending",
        )

    def _adopt_adaptive_step(self, step: Any) -> None:
        # RecommenderAgent owns the canonical adaptive transcript.
        self.dialogue = list(self.recommender.dialogue)
        self.tool_calls.append(step.tool_calls)
        if step.question_id:
            self.question_ids.append(step.question_id)
            self.question_topics.append(step.question_topic or "")
        if step.recommendation is not None:
            self.final_recommendation = step.recommendation

    def _step_event(self, step: Any) -> dict[str, Any]:
        return {
            "type": "recommender_message",
            "content": step.reply,
            "tool_calls": step.tool_calls,
            "turn": len(self.tool_calls),
        }

    def _abandonment_reason(self, question: str) -> str | None:
        if not self.endogenous_abandonment:
            return None
        record: dict[str, Any] = {
            "question_number": len(self.question_ids),
            "question_id": self.question_ids[-1] if self.question_ids else None,
            "question": question,
        }
        try:
            decision = self.buyer.decide_abandonment(question)
            record.update({"evaluation_status": "scored", **decision})
        except Exception as exc:
            decision = {
                "action": "ANSWER",
                "reason": (
                    "Defaulted to ANSWER because the abandonment decision "
                    "could not be evaluated."
                ),
            }
            record.update(
                {
                    "evaluation_status": "fallback_answer",
                    **decision,
                    "error_code": (
                        exc.code
                        if isinstance(exc, RecommendationError)
                        else type(exc).__name__.upper()
                    ),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        self.engagement_decisions.append(record)
        return decision["reason"] if decision["action"] == "ABANDON" else None

    def _finish_abandoned(
        self, reason: str, *, turn: int
    ) -> Iterator[dict[str, Any]]:
        reply = f"{ABANDON_SENTINEL} {reason}"
        self.dialogue.append({"role": "user", "content": reply})
        yield {"type": "buyer_message", "content": reply, "turn": turn}
        yield {"type": "abandoned", "reason": reason}
        yield {
            "type": "outcome",
            "outcome": self._outcome(outcome="ABANDONED", abandonment_reason=reason),
        }

    def _ask_fixed(self) -> Iterator[dict[str, Any] | bool]:
        question = self.questions.next(fixed=True)
        if question is None:
            raise RecommendationError("NO_QUESTIONS_REMAIN", "fixed ATR question bank exhausted")
        self.question_ids.append(question.id)
        self.question_topics.append(question.topic)
        self.dialogue.append({"role": "assistant", "content": question.text})
        self.tool_calls.append(
            [{"tool": "ask_question", "args": {"fixed": True}, "result": question.id}]
        )
        yield {
            "type": "recommender_message",
            "content": question.text,
            "tool_calls": self.tool_calls[-1],
            "turn": len(self.tool_calls),
        }
        abandonment_reason = self._abandonment_reason(question.text)
        if abandonment_reason is not None:
            yield from self._finish_abandoned(
                abandonment_reason,
                turn=len(self.tool_calls) + 1,
            )
            return True
        reply = self.buyer.respond(question.text)
        self.dialogue.append({"role": "user", "content": reply})
        yield {"type": "buyer_message", "content": reply, "turn": len(self.tool_calls) + 1}
        return False

    def _run_branching(self) -> Iterator[dict[str, Any]]:
        terminal_checkpoint: dict[str, Any] | None = None
        terminal_result: RecommendationResult | None = None
        for depth in range(self.policy.numquestions + 1):
            checkpoint, result = self._evaluate_checkpoint(depth)
            self.checkpoints.append(checkpoint)
            yield {"type": "checkpoint", "checkpoint": checkpoint, "depth": depth}
            if depth == self.policy.numquestions:
                terminal_checkpoint = checkpoint
                terminal_result = result
                break
            abandoned = yield from self._ask_fixed()
            if abandoned:
                return
        if (
            terminal_checkpoint is None
            or terminal_result is None
            or terminal_checkpoint["evaluation_status"] != "scored"
        ):
            code = (terminal_checkpoint or {}).get("error_code") or "TERMINAL_CHECKPOINT_FAILED"
            raise RecommendationError(code, (terminal_checkpoint or {}).get("error") or code)
        decision = terminal_checkpoint["decision"]
        resolved = {
            key: terminal_checkpoint[key]
            for key in (
                "outcome",
                "purchased_asin",
                "wtp",
                "actual_price",
                "revenue",
                "consumer_surplus",
            )
        }
        yield from self._finish_with_recommendation(
            terminal_result,
            existing_decision=decision,
            existing_resolution=resolved,
        )

    def _evaluate_checkpoint(
        self, depth: int
    ) -> tuple[dict[str, Any], RecommendationResult | None]:
        prefix_count = len(self.dialogue)
        prefix_hash = hashlib.sha256(
            json.dumps(self.dialogue, sort_keys=True).encode()
        ).hexdigest()
        try:
            result = self.service.recommend(
                list(self.dialogue),
                prior_recommendations=self.prior_recommendations,
            )
            # Carry recommendation identity only. Hidden purchase decisions and WTP
            # never affect the next checkpoint's candidate pool.
            self.prior_recommendations = [
                {"asin": product.get("asin")} for product in result.recommendations
            ]
            decision = self.buyer.evaluate_snapshot(result.recommendations)
            resolved = _resolve_decision(decision, result.recommendations)
            return (
                {
                    "depth": depth,
                    "dialogue_message_count": prefix_count,
                    "prefix_hash": prefix_hash,
                    "evaluation_status": "scored",
                    "recommendation_result": result.to_dict(),
                    "decision": decision,
                    **resolved,
                },
                result,
            )
        except Exception as exc:
            code = exc.code if isinstance(exc, RecommendationError) else type(exc).__name__.upper()
            return (
                {
                    "depth": depth,
                    "dialogue_message_count": prefix_count,
                    "prefix_hash": prefix_hash,
                    "evaluation_status": "protocol_error",
                    "error_code": code,
                    "error": f"{type(exc).__name__}: {exc}",
                },
                None,
            )

    def _finish_with_recommendation(
        self,
        result: RecommendationResult,
        *,
        recommendation_already_visible: bool = False,
        existing_decision: dict[str, Any] | None = None,
        existing_resolution: dict[str, Any] | None = None,
    ) -> Iterator[dict[str, Any]]:
        self.final_recommendation = result
        reply = render_recommendations(result.recommendations)
        if not recommendation_already_visible:
            self.dialogue.append({"role": "assistant", "content": reply})
            self.tool_calls.append(
                [{"tool": "recommend", "args": {}, "result": result.candidate_asins}]
            )
            yield {
                "type": "recommender_message",
                "content": reply,
                "tool_calls": self.tool_calls[-1],
                "turn": len(self.tool_calls),
            }
        yield {"type": "recommendations", "items": result.recommendations}
        decision = existing_decision or self.buyer.decide(result.recommendations)
        resolved = existing_resolution or _resolve_decision(decision, result.recommendations)
        yield {"type": "decision", "decision": decision, **resolved}
        yield {
            "type": "outcome",
            "outcome": self._outcome(
                decision=decision,
                recommendation=result,
                **resolved,
            ),
        }

    def _outcome(
        self,
        *,
        outcome: str,
        decision: dict[str, Any] | None = None,
        recommendation: RecommendationResult | None = None,
        purchased_asin: str | None = None,
        wtp: float | None = None,
        actual_price: float | None = None,
        revenue: float = 0.0,
        consumer_surplus: float = 0.0,
        abandonment_reason: str | None = None,
        error_code: str | None = None,
        error: str | None = None,
    ) -> Outcome:
        recommendation = recommendation or self.final_recommendation
        return Outcome(
            persona_id=str(self.persona.get("id") or ""),
            category=self.category,
            policy=self.policy.name,
            numquestions=self.policy.numquestions,
            outcome=outcome,
            turns_used=len(self.tool_calls),
            asks=len(self.question_ids),
            purchased_asin=purchased_asin,
            wtp=wtp,
            actual_price=actual_price,
            revenue=revenue,
            consumer_surplus=consumer_surplus,
            decision=decision,
            dialogue=list(self.dialogue),
            recommendation_result=(recommendation.to_dict() if recommendation else None),
            question_ids=list(self.question_ids),
            question_topics=list(self.question_topics),
            tool_calls_per_turn=list(self.tool_calls),
            checkpoints=list(self.checkpoints),
            engagement_decisions=list(self.engagement_decisions),
            abandonment_reason=abandonment_reason,
            error_code=error_code,
            error=error,
            api_usage=self.tracker.summary(),
        )
