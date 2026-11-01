"""Adaptive elicitation controller: preview, decide, choose a question, recommend.

Each turn the controller runs a fixed sequence of stateless structured calls:

1. ``candidate_preview`` shows what the recommender would choose from right now.
   Before any question is asked this is the configured default slate, which is
   exactly what ``recommend`` would show at that point. Afterwards it is the
   retrieval half of ``RecommendationService.recommend``: the same plan call and
   catalog retrieval, without the final selection.
2. ``decide_next_action`` returns ASK or RECOMMEND with the open uncertainties
   and a short reason.
3. ``choose_question`` (ASK only) picks one unasked question from the
   configured bank. With ``fixed_order`` set, questions come in bank (YAML)
   order without a model call, as in the ATR policies; the decider still
   chooses when to stop.
4. ``recommend`` is the unchanged shared recommendation service, so adaptive
   and ATR conversations differ only in which questions are asked and when.

Every call is rebuilt from the visible dialogue, so no tool output accumulates
across turns.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Callable

from openai import OpenAI

from sandbox.agents.recommender import (
    DEFAULT_MODEL,
    RecommendationError,
    RecommendationPlan,
    RecommendationResult,
    RecommendationService,
    dialogue_text,
    render_recommendations,
)
import numpy as np

from sandbox.catalog import _dense_scores, default_slate, load_catalog, retrieve_catalog
from sandbox.openai_responses import (
    UsageTracker,
    create_response,
    make_client,
    response_to_text,
)
from sandbox.questions import Question, QuestionBank


PREVIEW_SIZE = 8
PREVIEW_SPEC_LIMIT = 6
# Spec keys that rarely separate products; skipped so the few shown specs matter.
PREVIEW_SPEC_SKIP = ("color", "style", "pattern", "model name", "brand", "special feature")
STRUCTURED_ATTEMPTS = 2
STOP_RULES = ("swap_probability", "retrieval_swap", "confidence", "entropy", "semantic_entropy")
# Rule-specific defaults for stop_threshold (swap_probability uses ask_threshold).
STOP_THRESHOLDS = {
    "retrieval_swap": 0.1,
    "confidence": 8.0,
    "entropy": 0.6,
    "semantic_entropy": 0.5,
}
# Stall stop for the confidence and entropy rules: the last STALL_TURNS answers
# each moved the kth-best confidence by less than STALL_DELTA in either
# direction. A falling score is information (an answer exposed a constraint the
# picks miss), so it keeps the conversation going.
STALL_TURNS = 2
STALL_DELTA = 0.5
# semantic_entropy: softmax temperature over z-scored matches, top-k size, and the
# least entropy drop over the last STALL_TURNS answers that counts as progress.
SEMANTIC_TAU = 0.25
SEMANTIC_TOP_K = 10
SEMANTIC_MIN_DROP = 0.05
# Upper bound on follow-up questions for the numeric rules, beyond the openers.
MAX_FOLLOWUPS = 6
ANSWER_OPTIONS_PER_QUESTION = 3


@dataclass(frozen=True)
class AgentStep:
    reply: str
    tool_calls: list[dict[str, Any]]
    recommendation: RecommendationResult | None = None
    question_id: str | None = None
    question_topic: str | None = None


@dataclass(frozen=True)
class CandidatePreview:
    """Products the recommender would currently choose from."""

    source: str  # "default_slate" before any question, otherwise "retrieval"
    plan: RecommendationPlan | None
    eligible_count: int
    products: list[dict[str, Any]]
    default_slate_asins: list[str]
    pool_asins: list[str] = field(default_factory=list)  # full retrieval pool, before truncation

    def render(self) -> str:
        if self.source == "default_slate":
            header = (
                "The buyer has not stated any preferences yet. Recommending now "
                "shows this fixed default slate, chosen without knowing the buyer; the catalog "
                "has many other products:"
            )
        else:
            plan = self.plan
            limits = []
            if plan and plan.max_price is not None:
                limits.append(f"max ${plan.max_price:.0f}")
            if plan and plan.target_price is not None:
                limits.append(f"target ${plan.target_price:.0f}")
            prices = sorted(float(product["price"]) for product in self.products)
            header = (
                f"Current search: \"{plan.focus_query if plan else ''}\""
                f"{' (' + ', '.join(limits) + ')' if limits else ''}. "
                f"{self.eligible_count} products are within the price limits. "
                f"Top candidates (price ${prices[0]:.2f}-${prices[-1]:.2f}); "
                "[default] marks default-slate products that stay in the final pool:"
            )
        return header + "\n" + "\n".join(
            _preview_card(number, product, product["asin"] in self.default_slate_asins)
            for number, product in enumerate(self.products, start=1)
        )

    def log(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "plan": self.plan.to_dict() if self.plan else None,
            "eligible_count": self.eligible_count,
            "asins": [product["asin"] for product in self.products],
        }


@dataclass(frozen=True)
class ActionDecision:
    action: str  # "ASK" or "RECOMMEND"
    current_picks: list[str]  # ASINs the decider would recommend right now
    # Assessments at or above the ask threshold, most likely swap first:
    # {"question_id", "pick_at_risk" (ASIN), "swap_probability", "how"}.
    helpful_questions: list[dict[str, Any]]
    reason: str
    forced: bool = False
    assessments: list[dict[str, Any]] = field(default_factory=list)  # every estimate given
    metrics: dict[str, Any] = field(default_factory=dict)  # numeric stop-rule inputs

    @property
    def helpful_question_ids(self) -> list[str]:
        return [item["question_id"] for item in self.helpful_questions]


def _preview_card(number: int, product: dict[str, Any], is_default: bool) -> str:
    specs = "; ".join(
        f"{key}: {str(value)[:40]}"
        for key, value in [
            (key, value)
            for key, value in (product.get("spec_table") or {}).items()
            if not any(word in key.lower() for word in PREVIEW_SPEC_SKIP)
        ][:PREVIEW_SPEC_LIMIT]
    )
    rating = product.get("avg_rating")
    return (
        f"{number}. {'[default] ' if is_default else ''}{str(product.get('title') or '')[:110]}"
        f" — ${float(product['price']):.2f}, {rating if rating is not None else '?'} stars"
        f"{'. ' + specs if specs else ''}"
    )


def _question_list(questions: list[Question]) -> str:
    return "\n".join(f"- {question.id} ({question.topic}): {question.text}" for question in questions)


@dataclass
class AdaptiveController:
    """Adaptive policy for one conversation. The visible dialogue is its only user model."""

    category: str
    model: str = DEFAULT_MODEL
    retrieval_limit: int = 15
    assortment_size: int = 3
    # Ask questions in bank (YAML) order instead of choosing them, as ATR does.
    fixed_order: bool = False
    max_questions: int | None = None
    tracker: UsageTracker = field(default_factory=UsageTracker)
    client: OpenAI = field(default_factory=make_client)
    dialogue: list[dict[str, str]] = field(default_factory=list)
    questions: QuestionBank | None = None
    # Ask when some question's estimated chance of swapping a current pick reaches this.
    ask_threshold: float = 0.25
    # The previous decision's estimates, shown back so they are updated rather than redrawn.
    last_assessments: list[dict[str, Any]] = field(default_factory=list)
    # How to decide between asking and recommending; see STOP_RULES.
    stop_rule: str = "swap_probability"
    stop_threshold: float | None = None
    # Stop-rule metric after each non-forced decision, for the stall check.
    metric_history: list[float] = field(default_factory=list)
    # z-scored semantic match of every product to each answered question, by turn.
    answer_vectors: dict[int, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.stop_rule not in STOP_RULES:
            raise ValueError(f"stop_rule must be one of {', '.join(STOP_RULES)}")
        self.questions = self.questions or QuestionBank.load(self.category)
        self.service = RecommendationService(
            category=self.category,
            model=self.model,
            retrieval_limit=self.retrieval_limit,
            assortment_size=self.assortment_size,
            tracker=self.tracker,
            client=self.client,
        )

    def step(self, buyer_message: str) -> AgentStep:
        self.dialogue.append({"role": "user", "content": buyer_message})
        calls: list[dict[str, Any]] = []

        preview = self.candidate_preview()
        calls.append({"tool": "candidate_preview", "args": {}, "result": preview.log()})

        decision = self.decide_next_action(preview)
        calls.append(
            {
                "tool": "decide_next_action",
                "args": {"forced": decision.forced},
                "result": {
                    "action": decision.action,
                    "current_picks": decision.current_picks,
                    "stop_rule": self.stop_rule,
                    "threshold": self.threshold,
                    "metrics": decision.metrics,
                    "assessments": decision.assessments,
                    "reason": decision.reason,
                },
            }
        )

        if decision.action == "RECOMMEND":
            if preview.source == "default_slate":
                recommendation = self.service.default_recommendation()
            else:
                recommendation = self.service.recommend(self.dialogue)
            calls.append(
                {
                    "tool": "recommend",
                    "args": {"source": preview.source},
                    "result": recommendation.candidate_asins,
                }
            )
            step = AgentStep(
                reply=render_recommendations(recommendation.recommendations),
                tool_calls=calls,
                recommendation=recommendation,
            )
        else:
            question, call = self.choose_question(preview, decision)
            calls.append(call)
            step = AgentStep(
                reply=question.text,
                tool_calls=calls,
                question_id=question.id,
                question_topic=question.topic,
            )
        self.dialogue.append({"role": "assistant", "content": step.reply})
        return step

    # ── Step 1 ────────────────────────────────────────────────────────────────
    def candidate_preview(self) -> CandidatePreview:
        """Show the default slate before any question, otherwise the live candidate pool."""
        slate = default_slate(self.category)
        slate_asins = [product["asin"] for product in slate]
        if not self.questions.asked_ids:
            return CandidatePreview(
                source="default_slate",
                plan=None,
                eligible_count=len(slate),
                products=slate,
                default_slate_asins=slate_asins,
            )
        transcript = dialogue_text(self.dialogue)
        plan = self.service._plan(transcript)
        retrieval = retrieve_catalog(
            self.category,
            transcript,
            focus_query=plan.focus_query,
            max_price=plan.max_price,
            target_price=plan.target_price,
            min_price=plan.min_price,
            limit=self.retrieval_limit,
        )
        products = list(retrieval.products[:PREVIEW_SIZE])
        shown = {product["asin"] for product in products}
        # Mirror RecommendationService.recommend: default-slate products inside
        # the price limits are always part of the final selection pool.
        for product in slate:
            price = product.get("price")
            if not isinstance(price, (int, float)) or product["asin"] in shown:
                continue
            if plan.max_price is not None and float(price) > plan.max_price:
                continue
            if plan.min_price is not None and float(price) < plan.min_price:
                continue
            products.append(product)
        return CandidatePreview(
            source="retrieval",
            plan=plan,
            eligible_count=retrieval.eligible_count,
            products=products,
            default_slate_asins=slate_asins,
            pool_asins=[product["asin"] for product in retrieval.products],
        )

    # ── Step 2 ────────────────────────────────────────────────────────────────
    def decide_next_action(self, preview: CandidatePreview) -> ActionDecision:
        """Ask another question or recommend now."""
        asked = len(self.questions.asked_ids)
        remaining = self.questions.remaining()
        limit = self.max_questions if self.max_questions is not None else len(self.questions.questions)
        if not remaining or asked >= limit:
            return ActionDecision(
                action="RECOMMEND",
                current_picks=[],
                helpful_questions=[],
                reason="No questions remain." if not remaining else f"Reached the {limit}-question limit.",
                forced=True,
            )
        if not self.questions.asked_ids:
            # Recommending before any answer would only repeat the default slate.
            return ActionDecision(
                action="ASK",
                current_picks=[],
                helpful_questions=[],
                reason="Nothing is known about the buyer yet.",
                forced=True,
            )
        if self.stop_rule == "retrieval_swap":
            return self._decide_retrieval_swap(preview, remaining)
        if self.stop_rule in {"confidence", "entropy"}:
            return self._decide_scores(preview, remaining)
        if self.stop_rule == "semantic_entropy":
            return self._decide_semantic_entropy(preview, remaining)
        name = self.category.replace("_", " ")
        picks = min(self.assortment_size, len(preview.products))
        instructions = (
            f"You run a {name} shopping assistant and are deciding whether one more question is "
            f"worth asking before recommending {self.assortment_size} products.\n\n"
            "You see the conversation, the only questions you are able to ask, and a numbered "
            "preview of the products the recommender would currently choose from. You cannot ask "
            "anything outside that list, and asking cannot add products to the catalog.\n\n"
            f"1. current_picks: the {picks} preview products you would recommend right now.\n"
            "2. assessments: for each listed question that could matter, the probability "
            "(0 to 1) that this buyer's answer would lead you to replace one of current_picks "
            "with a different product from the catalog (the preview shows only the top "
            "candidates). Give the pick at risk and, in how, the answer that would cause the "
            "swap. Omit questions near zero. Calibrate:\n"
            "   - A basic specification most buyers in this category have firm needs on, that "
            "the buyer has not addressed and the picks differ on or could violate: often 0.3-0.6.\n"
            "   - A niche topic the conversation gives no hint of: 0.1 or less.\n"
            "   - A side feature (webcam, aesthetics, ports detail) the buyer has not raised as "
            "a priority: 0.15 or less, even if the use case loosely relates to it.\n"
            "   - A question on the same topic as one the buyer already answered vaguely or "
            "with 'not sure': near zero (asking again will get the same answer).\n"
            "   - A question whose answer is already stated or clearly implied: near zero.\n"
            "   - 'No preference' answers lower the probability for similar minor topics.\n"
            "   - Update your estimates from last turn using the buyer's latest answer.\n"
            "Keep reason to one or two sentences summarising where the uncertainty is."
        )
        asked_text = ", ".join(
            question.id for question in self.questions.questions if question.id in self.questions.asked_ids
        )
        previous = "; ".join(
            f"{item['question_id']} {item['swap_probability']:.2f}"
            for item in self.last_assessments
            if item["question_id"] not in self.questions.asked_ids
        )
        payload = (
            f"Conversation:\n{dialogue_text(self.dialogue)}\n\n"
            f"Questions asked so far ({asked}): {asked_text or 'none'}\n\n"
            f"Questions you can still ask:\n{_question_list(remaining)}\n\n"
            f"Your estimates last turn: {previous or 'none'}\n\n"
            f"Candidate preview:\n{preview.render()}"
        )
        ids = [question.id for question in remaining]
        numbers = list(range(1, len(preview.products) + 1))
        schema = {
            "type": "object",
            "properties": {
                "current_picks": {"type": "array", "items": {"type": "integer", "enum": numbers}},
                "assessments": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question_id": {"type": "string", "enum": ids},
                            "pick_at_risk": {"type": "integer", "enum": numbers},
                            "swap_probability": {"type": "number"},
                            "how": {"type": "string"},
                        },
                        "required": ["question_id", "pick_at_risk", "swap_probability", "how"],
                        "additionalProperties": False,
                    },
                },
                "reason": {"type": "string"},
            },
            "required": ["current_picks", "assessments", "reason"],
            "additionalProperties": False,
        }

        def valid(data: dict[str, Any]) -> bool:
            chosen = data.get("current_picks")
            assessments = data.get("assessments")
            if not (
                isinstance(chosen, list)
                and len(chosen) == picks
                and len(set(chosen)) == picks
                and all(number in numbers for number in chosen)
                and isinstance(assessments, list)
                and isinstance(data.get("reason"), str)
                and bool(data["reason"].strip())
            ):
                return False
            return all(
                isinstance(item, dict)
                and item.get("question_id") in ids
                and item.get("pick_at_risk") in chosen
                and isinstance(item.get("swap_probability"), (int, float))
                and 0 <= item["swap_probability"] <= 1
                and isinstance(item.get("how"), str)
                for item in assessments
            )

        data = self._structured(
            kind="adaptive_decision",
            name="next_action",
            instructions=instructions,
            payload=payload,
            schema=schema,
            valid=valid,
            error_code="INVALID_NEXT_ACTION",
        )
        asin = lambda number: preview.products[number - 1]["asin"]  # noqa: E731
        assessments: list[dict[str, Any]] = []
        for item in sorted(data["assessments"], key=lambda entry: -entry["swap_probability"]):
            if item["question_id"] not in {entry["question_id"] for entry in assessments}:
                assessments.append(
                    {
                        "question_id": item["question_id"],
                        "pick_at_risk": asin(item["pick_at_risk"]),
                        "swap_probability": round(float(item["swap_probability"]), 3),
                        "how": item["how"].strip(),
                    }
                )
        self.last_assessments = assessments
        # The code, not the model, applies the threshold, so it can be tuned or swept.
        helpful = [item for item in assessments if item["swap_probability"] >= self.ask_threshold]
        return ActionDecision(
            action="ASK" if helpful else "RECOMMEND",
            current_picks=[asin(number) for number in data["current_picks"]],
            helpful_questions=helpful,
            reason=data["reason"].strip(),
            assessments=assessments,
        )

    @property
    def threshold(self) -> float:
        if self.stop_rule == "swap_probability":
            return self.ask_threshold
        if self.stop_threshold is not None:
            return self.stop_threshold
        return STOP_THRESHOLDS[self.stop_rule]

    def _retrieve_pool(self, transcript: str, plan: RecommendationPlan | None) -> set[str]:
        retrieval = retrieve_catalog(
            self.category,
            transcript,
            focus_query=plan.focus_query if plan else None,
            max_price=plan.max_price if plan else None,
            target_price=plan.target_price if plan else None,
            min_price=plan.min_price if plan else None,
            limit=self.retrieval_limit,
        )
        return {product["asin"] for product in retrieval.products}

    def _decide_retrieval_swap(
        self, preview: CandidatePreview, remaining: list[Question]
    ) -> ActionDecision:
        """Ask the question whose likely answers most change the retrieved pool.

        One low-effort call writes answers this buyer might plausibly give to each
        remaining question. Each answer is appended to the transcript and local
        retrieval is rerun with the current plan's query and price limits.
        Turnover is the share of the current pool that drops out, minus the
        turnover a "no preference" answer causes, so the question wording alone
        does not count as information.
        """
        options = self._buyer_answer_options(preview, remaining)
        transcript = dialogue_text(self.dialogue)
        base = set(preview.pool_asins) or self._retrieve_pool(transcript, preview.plan)

        def turnover(question: Question, answer: str) -> float:
            pool = self._retrieve_pool(
                f"{transcript}\nRecommender: {question.text}\nBuyer: {answer}", preview.plan
            )
            return 1 - len(base & pool) / len(base)

        assessments: list[dict[str, Any]] = []
        for question in remaining:
            answers = options.get(question.id, [])
            if not answers:
                continue
            baseline = turnover(question, "I have no preference about that.")
            values = [turnover(question, answer) for answer in answers]
            assessments.append(
                {
                    "question_id": question.id,
                    "pool_turnover": round(max(0.0, sum(values) / len(values) - baseline), 3),
                    "baseline": round(baseline, 3),
                    "per_answer": [round(value, 3) for value in values],
                }
            )
        assessments.sort(key=lambda item: -item["pool_turnover"])
        best = assessments[0] if assessments else None
        ask = best is not None and best["pool_turnover"] >= self.threshold
        return ActionDecision(
            action="ASK" if ask else "RECOMMEND",
            current_picks=[],
            helpful_questions=(
                [
                    {
                        "question_id": best["question_id"],
                        "how": f"likely answers change {best['pool_turnover']:.0%} of the pool",
                    }
                ]
                if ask
                else []
            ),
            reason=(
                f"Highest net pool turnover is {best['pool_turnover']:.2f} ({best['question_id']})."
                if best
                else "No sample answers are available."
            ),
            assessments=assessments,
            metrics={"max_pool_turnover": best["pool_turnover"] if best else 0.0},
        )

    def _buyer_answer_options(
        self, preview: CandidatePreview, remaining: list[Question]
    ) -> dict[str, list[str]]:
        """Answers this buyer might plausibly give to each remaining question (one call)."""
        ids = [question.id for question in remaining]
        name = self.category.replace("_", " ")
        schema = {
            "type": "object",
            "properties": {
                "answers": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question_id": {"type": "string", "enum": ids},
                            "options": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["question_id", "options"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["answers"],
            "additionalProperties": False,
        }
        data = self._structured(
            kind="adaptive_answer_options",
            name="answer_options",
            instructions=(
                f"Given this {name} shopping conversation, write for each listed question "
                f"{ANSWER_OPTIONS_PER_QUESTION - 1} different answers this particular buyer might "
                "plausibly give, consistent with what they have said. Use 'I have no preference' "
                "when that is the most likely answer. Each answer under 20 words."
            ),
            payload=(
                f"Conversation:\n{dialogue_text(self.dialogue)}\n\n"
                f"Questions:\n{_question_list(remaining)}"
            ),
            schema=schema,
            valid=lambda data: isinstance(data.get("answers"), list)
            and all(
                isinstance(item, dict) and isinstance(item.get("options"), list)
                for item in data["answers"]
            ),
            error_code="INVALID_ANSWER_OPTIONS",
            effort="low",
        )
        return {
            item["question_id"]: [str(text).strip() for text in item["options"] if str(text).strip()]
            for item in data["answers"]
        }

    def _followup_cap_reached(self) -> bool:
        followups = sum(
            1
            for question in self.questions.questions
            if question.id in self.questions.asked_ids and question.tier != "opener"
        )
        return followups >= MAX_FOLLOWUPS

    def _decide_semantic_entropy(
        self, preview: CandidatePreview, remaining: list[Question]
    ) -> ActionDecision:
        """Recommend once a few in-budget products clearly match the buyer's answers.

        No model call for the decision. Each answered question (except budget,
        which the price filter handles) is embedded with its answer, matched
        against every product with the dense index, and z-scored across the
        catalog. The z-scores are summed and divided by sqrt(answers), restricted
        to products within the plan's price limits, and the normalised entropy
        of softmax(top-k / tau) measures how clearly the best matches stand out.
        """
        texts = {question.text: question for question in self.questions.questions}
        catalog = [product for product in load_catalog(self.category) if product.get("asin")]
        pending = self.dialogue
        for index in range(1, len(pending)):
            message, previous = pending[index], pending[index - 1]
            if message["role"] != "user" or previous["role"] != "assistant":
                continue
            question = texts.get(previous["content"])
            if question is None or question.topic == "budget" or index in self.answer_vectors:
                continue
            scores = _dense_scores(self.category, f"{previous['content']} {message['content']}")
            values = np.array([scores[product["asin"]] for product in catalog])
            self.answer_vectors[index] = (values - values.mean()) / (values.std() or 1.0)
        plan = preview.plan
        mask = np.array(
            [
                isinstance(product.get("price"), (int, float))
                and (plan is None or plan.max_price is None or product["price"] <= plan.max_price)
                and (plan is None or plan.min_price is None or product["price"] >= plan.min_price)
                for product in catalog
            ]
        )
        vectors = list(self.answer_vectors.values())
        if vectors and mask.sum() >= 2:
            combined = np.sum(vectors, axis=0) / math.sqrt(len(vectors))
            top = np.sort(combined[mask])[::-1][:SEMANTIC_TOP_K]
            logits = (top - top[0]) / SEMANTIC_TAU
            weights = np.exp(logits) / np.exp(logits).sum()
            entropy = float(-(weights * np.log(weights + 1e-12)).sum() / math.log(len(top)))
            order = np.argsort(-np.where(mask, combined, -np.inf))[: self.assortment_size]
            picks = [catalog[index]["asin"] for index in order]
            gap = float(top[0] - top[-1])
        else:
            entropy, picks, gap = 1.0, [], 0.0
        history = self.metric_history
        history.append(entropy)
        stalled = (
            len(history) > STALL_TURNS and history[-1] > history[-1 - STALL_TURNS] - SEMANTIC_MIN_DROP
        )
        ask = entropy > self.threshold and not stalled and not self._followup_cap_reached()
        metrics = {
            "semantic_entropy": round(entropy, 3),
            "z_gap_top_k": round(gap, 3),
            "answers_used": len(vectors),
            "eligible": int(mask.sum()),
            "stalled": stalled,
        }
        reason = (
            f"Semantic entropy {entropy:.2f} "
            + ("is above" if entropy > self.threshold else "is at or below")
            + f" {self.threshold:.2f}"
            + ("; the last answers stopped sharpening the matches." if stalled else ".")
        )
        return ActionDecision(
            action="ASK" if ask else "RECOMMEND",
            current_picks=picks,
            # The LLM chooser picks among all remaining questions.
            helpful_questions=[],
            reason=reason,
            metrics=metrics,
        )

    def _decide_scores(
        self, preview: CandidatePreview, remaining: list[Question]
    ) -> ActionDecision:
        """Score each preview product's chance of satisfying the buyer; stop on a numeric rule.

        confidence: recommend when the assortment_size-th best score reaches the threshold.
        entropy: recommend when the normalised entropy of softmax(scores) is at or below it.
        """
        name = self.category.replace("_", " ")
        instructions = (
            f"You run a {name} shopping assistant. For every numbered preview product, rate "
            "0-10 your confidence that it would satisfy this buyer. Consider what they have said "
            "and the important needs they have not stated yet: an unknown that could rule a "
            "product out lowers its score. Then name the one listed question whose answer would "
            "most raise your confidence in the best recommendations. Keep reason to one sentence."
        )
        payload = (
            f"Conversation:\n{dialogue_text(self.dialogue)}\n\n"
            f"Questions you can still ask:\n{_question_list(remaining)}\n\n"
            f"Candidate preview:\n{preview.render()}"
        )
        ids = [question.id for question in remaining]
        numbers = list(range(1, len(preview.products) + 1))
        schema = {
            "type": "object",
            "properties": {
                "scores": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "product_number": {"type": "integer", "enum": numbers},
                            "confidence": {"type": "number"},
                        },
                        "required": ["product_number", "confidence"],
                        "additionalProperties": False,
                    },
                },
                "best_question_id": {"type": "string", "enum": ids},
                "reason": {"type": "string"},
            },
            "required": ["scores", "best_question_id", "reason"],
            "additionalProperties": False,
        }

        def valid(data: dict[str, Any]) -> bool:
            scores = data.get("scores")
            return (
                isinstance(scores, list)
                and sorted(item.get("product_number") for item in scores if isinstance(item, dict))
                == numbers
                and all(
                    isinstance(item.get("confidence"), (int, float)) and 0 <= item["confidence"] <= 10
                    for item in scores
                )
                and data.get("best_question_id") in ids
                and isinstance(data.get("reason"), str)
            )

        data = self._structured(
            kind="adaptive_scores",
            name="product_scores",
            instructions=instructions,
            payload=payload,
            schema=schema,
            valid=valid,
            error_code="INVALID_PRODUCT_SCORES",
        )
        ranked = sorted(data["scores"], key=lambda item: -item["confidence"])
        values = [float(item["confidence"]) for item in ranked]
        k = min(self.assortment_size, len(values))
        weights = [math.exp(value - values[0]) for value in values]
        total = sum(weights)
        entropy = -sum(w / total * math.log(w / total) for w in weights if w > 0)
        metrics = {
            "kth_best_confidence": values[k - 1],
            "mean_top_confidence": round(sum(values[:k]) / k, 3),
            "normalized_entropy": round(entropy / math.log(len(values)), 3) if len(values) > 1 else 0.0,
            "top_margin": round(sum(values[:k]) / k - (sum(values[k:]) / len(values[k:]) if values[k:] else 0.0), 3),
        }
        history = self.metric_history
        history.append(metrics["kth_best_confidence"])
        stalled = len(history) > STALL_TURNS and all(
            abs(history[-index] - history[-index - 1]) < STALL_DELTA
            for index in range(1, STALL_TURNS + 1)
        )
        metrics["stalled"] = stalled
        if self.stop_rule == "confidence":
            ask = metrics["kth_best_confidence"] < self.threshold
        else:
            ask = metrics["normalized_entropy"] > self.threshold
        ask = ask and not stalled and not self._followup_cap_reached()
        asin = lambda number: preview.products[number - 1]["asin"]  # noqa: E731
        return ActionDecision(
            action="ASK" if ask else "RECOMMEND",
            current_picks=[asin(item["product_number"]) for item in ranked[:k]],
            helpful_questions=(
                [{"question_id": data["best_question_id"], "how": data["reason"].strip()}] if ask else []
            ),
            reason=data["reason"].strip(),
            assessments=[
                {"asin": asin(item["product_number"]), "confidence": item["confidence"]}
                for item in ranked
            ],
            metrics=metrics,
        )

    # ── Step 3 ────────────────────────────────────────────────────────────────
    def choose_question(
        self, preview: CandidatePreview, decision: ActionDecision
    ) -> tuple[Question, dict[str, Any]]:
        """Pick the unasked question that best separates the candidates."""
        remaining = self.questions.remaining()
        if self.fixed_order:
            question = self.questions.take(remaining[0].id)
            return question, {
                "tool": "choose_question",
                "args": {"fixed_order": True},
                "result": {"question_id": question.id, "reason": "next question in bank order"},
            }
        # The decider's shortlist; fall back to every remaining question.
        options = [
            question for question in remaining if question.id in decision.helpful_question_ids
        ] or remaining
        instructions = (
            "Choose the single question to ask the buyer next. Pick the one whose answer would "
            "best tell the strong candidates in the preview apart for this buyer.\n"
            "- Skip a question whose answer the buyer has already stated or clearly implied, "
            "including through an answer to a related question.\n"
            "- Prefer basic attributes the candidates differ on that the buyer has not mentioned "
            "(for example size, weight, or price trade-offs) over niche follow-ups.\n"
            "- If the buyer said they don't know something (for example the room size), don't "
            "ask a question that only rephrases it; move to a different attribute.\n"
            "Keep reason to one or two sentences."
        )
        notes = {item["question_id"]: item["how"] for item in decision.helpful_questions}
        payload = (
            f"Conversation:\n{dialogue_text(self.dialogue)}\n\n"
            f"Candidate preview:\n{preview.render()}\n\n"
            f"Questions you can ask (shortlisted as likely to change the picks):\n"
            + "\n".join(
                f"- {question.id} ({question.topic}): {question.text}"
                + (f" [could change picks: {notes[question.id]}]" if notes.get(question.id) else "")
                for question in options
            )
        )
        ids = [question.id for question in options]
        if len(options) == 1:
            question = self.questions.take(options[0].id)
            return question, {
                "tool": "choose_question",
                "args": {"fixed_order": False},
                "result": {"question_id": question.id, "reason": "only shortlisted question"},
            }
        schema = {
            "type": "object",
            "properties": {
                "question_id": {"type": "string", "enum": ids},
                "reason": {"type": "string"},
            },
            "required": ["question_id", "reason"],
            "additionalProperties": False,
        }
        data = self._structured(
            kind="adaptive_question_choice",
            name="question_choice",
            instructions=instructions,
            payload=payload,
            schema=schema,
            valid=lambda data: data.get("question_id") in ids
            and isinstance(data.get("reason"), str)
            and bool(data["reason"].strip()),
            error_code="INVALID_QUESTION_CHOICE",
        )
        question = self.questions.take(data["question_id"])
        return question, {
            "tool": "choose_question",
            "args": {"fixed_order": False},
            "result": {"question_id": question.id, "reason": data["reason"].strip()},
        }

    def _structured(
        self,
        *,
        kind: str,
        name: str,
        instructions: str,
        payload: str,
        schema: dict[str, Any],
        valid: Callable[[dict[str, Any]], bool],
        error_code: str,
        effort: str = "medium",
    ) -> dict[str, Any]:
        raw = ""
        for _ in range(STRUCTURED_ATTEMPTS):
            response = create_response(
                self.client,
                self.tracker,
                kind=kind,
                model=self.model,
                service_tier="default",
                instructions=instructions,
                input=payload,
                reasoning={"effort": effort},
                text={"format": {"type": "json_schema", "name": name, "strict": True, "schema": schema}},
            )
            raw = response_to_text(response) or ""
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict) and valid(data):
                return data
        raise RecommendationError(error_code, raw[:500])
