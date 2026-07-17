"""Simulated conversation orchestrator: pairs the buyer LLM with the
rufus-femto CRS agent and runs them until a recommendation is finalized,
the buyer abandons, or a hard turn cap is hit.

Used for cheap-eval: replace human shoppers with persona-conditioned LLM
buyers, run N simulations, collect outcomes for analysis.

Convention for the canonical transcript:
    role = "assistant"  → CRS spoke
    role = "user"       → Buyer spoke
The BuyerAgent maintains its own internal message history with the opposite
convention (CRS as "user", buyer as "assistant"), which is correct for the
buyer's own LLM calls.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Iterator, Optional

from sandbox.elicitation_policy import ElicitationPolicy

if TYPE_CHECKING:
    from sandbox.agents.buyer import BuyerAgent
    from sandbox.agents.langgraph_crs import CRSAgentSession


ABANDON_SENTINEL = "[ABANDON]"


@dataclass
class SimOutcome:
    """Structured result from one simulated conversation."""

    persona_id: str
    category: str
    outcome: str                                # PURCHASE | NO_PURCHASE | NO_FEASIBLE_MATCH | ABANDONED | PROTOCOL_ERROR
    turns_used: int                             # how many user-assistant exchanges happened
    asks: int                                   # clarifying questions the CRS asked
    purchased_asin: Optional[str] = None
    wtp: Optional[float] = None
    actual_price: Optional[float] = None
    consumer_surplus: Optional[float] = None    # wtp - actual_price if both present
    abandoned_at_turn: Optional[int] = None
    abandonment_type: Optional[str] = None       # "exogenous" | "endogenous"
    abandonment_reason: Optional[str] = None
    dialogue: list = field(default_factory=list)
    crs_recommendations: Optional[list] = None
    crs_tool_calls_per_turn: list = field(default_factory=list)
    buyer_decision_raw: Optional[dict] = None
    policy: Optional[str] = None
    numquestions: Optional[int] = None
    question_ids: list[str] = field(default_factory=list)
    recommendation_source: Optional[str] = None
    recommendation_validation_error: Optional[str] = None
    recommendation_selection_raw: Optional[str] = None
    recommendation_prose_raw: Optional[str] = None
    recommendation_product_numbers: list[int] = field(default_factory=list)
    retrieval_query: Optional[str] = None
    retrieval_key_query: Optional[str] = None
    retrieval_query_raw: Optional[str] = None
    retrieval_candidate_asins: list[str] = field(default_factory=list)
    retrieval_scores: dict[str, float] = field(default_factory=dict)
    retrieval_lane_sources: dict[str, list[str]] = field(default_factory=dict)
    eligible_count: Optional[int] = None
    protocol_version: Optional[str] = None
    recommendation_checkpoints: list[dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.policy is None:
            data.pop("policy")
            data.pop("numquestions")
        return data


@dataclass
class SimConversation:
    """One simulated buyer ↔ CRS conversation.

    `eta`: per-turn abandonment hazard (probability the buyer gives up before
    the next exchange). Set to 0.0 for noiseless eval; raise to model the
    paper's hazard-based attrition.
    """

    persona: dict[str, Any]
    category: str
    max_turns: int = 16
    eta: float = 0.0
    abandonment_seed: int = 0
    elicitation_policy: Optional[ElicitationPolicy] = None
    endogenous_abandonment: bool = False
    abandonment_instructions: str | None = None
    buyer_model: str = "gpt-5-mini-2025-08-07"
    recommender_model: str = "gpt-5-mini-2025-08-07"
    retrieval_limit: int = 15

    # Allow caller to pass pre-constructed agents for testing/customization.
    buyer: Optional["BuyerAgent"] = None
    crs: Optional["CRSAgentSession"] = None

    def __post_init__(self) -> None:
        if self.buyer is None:
            from sandbox.agents.buyer import BuyerAgent

            self.buyer = BuyerAgent(
                persona=self.persona,
                category=self.category,
                model=self.buyer_model,
                endogenous_abandonment=self.endogenous_abandonment,
                abandonment_instructions=self.abandonment_instructions,
            )
        if self.crs is None:
            from sandbox.agents.langgraph_crs import CRSAgentSession

            self.crs = CRSAgentSession(
                category=(self.category if self.elicitation_policy is not None else None),
                model=self.recommender_model,
                elicitation_policy=self.elicitation_policy,
                retrieval_limit=self.retrieval_limit,
            )
        self._rng = random.Random(self.abandonment_seed)

    # ------------------------------------------------------------------

    def _policy_fields(self) -> dict[str, Any]:
        if self.elicitation_policy is None:
            return {}
        return {
            "policy": self.elicitation_policy.name,
            "numquestions": self.elicitation_policy.target_asks,
        }

    @staticmethod
    def _protocol_fields(crs_out: dict[str, Any]) -> dict[str, Any]:
        """Copy recommendation/policy provenance from the CRS response."""
        return {
            "question_ids": list(crs_out.get("question_ids") or []),
            "recommendation_source": crs_out.get("recommendation_source"),
            "recommendation_validation_error": crs_out.get("recommendation_validation_error"),
            "recommendation_selection_raw": crs_out.get("recommendation_selection_raw"),
            "recommendation_prose_raw": crs_out.get("recommendation_prose_raw"),
            "recommendation_product_numbers": list(
                crs_out.get("recommendation_product_numbers") or []
            ),
            "retrieval_query": crs_out.get("retrieval_query"),
            "retrieval_key_query": crs_out.get("retrieval_key_query"),
            "retrieval_query_raw": crs_out.get("retrieval_query_raw"),
            "retrieval_candidate_asins": list(crs_out.get("retrieval_candidate_asins") or []),
            "retrieval_scores": dict(crs_out.get("retrieval_scores") or {}),
            "retrieval_lane_sources": {
                asin: list(sources)
                for asin, sources in (crs_out.get("retrieval_lane_sources") or {}).items()
            },
            "eligible_count": crs_out.get("eligible_count"),
            "protocol_version": crs_out.get("protocol_version"),
        }

    def _evaluate_new_checkpoints(
        self,
        crs_out: dict[str, Any],
        existing: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Score hidden snapshots without changing the active buyer conversation.

        Snapshot scorer failures are recorded on that checkpoint. They do not
        turn a completed terminal conversation into a protocol error.
        """
        records: list[dict[str, Any]] = []
        prior_current = next(
            (
                item.get("current_utility")
                for item in reversed(existing)
                if isinstance(item.get("current_utility"), (int, float))
            ),
            None,
        )
        prior_best = max(
            (
                float(item["best_observed_utility"])
                for item in existing
                if isinstance(item.get("best_observed_utility"), (int, float))
            ),
            default=None,
        )
        prior_best_purchase = any(
            item.get("best_observed_purchase") is True for item in existing
        )
        evaluator = getattr(self.buyer, "evaluate_snapshot", None)

        for snapshot in crs_out.get("new_checkpoints") or []:
            record = dict(snapshot)
            recs = snapshot.get("recommendations") or []
            if snapshot.get("terminal_status") == "NO_FEASIBLE_MATCH":
                record.update(
                    {
                        "evaluation_status": "not_scored_no_feasible_match",
                        "counterfactual_outcome": "NO_FEASIBLE_MATCH",
                        "current_purchase": False,
                        "best_observed_purchase": prior_best_purchase,
                        "current_utility": None,
                        "best_observed_utility": prior_best,
                        "degraded_from_previous": False,
                    }
                )
                records.append(record)
                continue
            if not callable(evaluator):
                record.update(
                    {
                        "evaluation_status": "unscored_no_checkpoint_evaluator",
                        "current_utility": None,
                        "best_observed_utility": prior_best,
                        "degraded_from_previous": False,
                    }
                )
                records.append(record)
                continue
            try:
                decision = evaluator(recs)
                resolved = self._resolve_decision(decision, recs)
            except Exception as exc:
                record.update(
                    {
                        "evaluation_status": "snapshot_evaluator_error",
                        "snapshot_evaluator_error": f"{type(exc).__name__}: {exc}",
                        "current_utility": None,
                        "best_observed_utility": prior_best,
                        "degraded_from_previous": False,
                    }
                )
                records.append(record)
                continue

            current_utility = (
                max(0.0, float(resolved["consumer_surplus"]))
                if resolved["outcome_label"] == "PURCHASE"
                and isinstance(resolved["consumer_surplus"], (int, float))
                else 0.0
            )
            current_purchase = resolved["outcome_label"] == "PURCHASE"
            best_observed = max(
                value for value in (prior_best, current_utility) if value is not None
            )
            record.update(
                {
                    "evaluation_status": "scored",
                    "counterfactual_decision_raw": resolved["decision"],
                    "counterfactual_outcome": resolved["outcome_label"],
                    "counterfactual_purchased_asin": resolved["purchased_asin"],
                    "counterfactual_wtp": resolved["wtp"],
                    "counterfactual_actual_price": resolved["actual_price"],
                    "counterfactual_consumer_surplus": resolved["consumer_surplus"],
                    "current_purchase": current_purchase,
                    "best_observed_purchase": prior_best_purchase or current_purchase,
                    "current_utility": current_utility,
                    "best_observed_utility": best_observed,
                    "degraded_from_previous": (
                        prior_current is not None and current_utility < prior_current
                    ),
                }
            )
            prior_current = current_utility
            prior_best = best_observed
            prior_best_purchase = prior_best_purchase or current_purchase
            records.append(record)
        return records

    def _recommendations_are_terminal(self, crs_out: dict[str, Any]) -> bool:
        if not crs_out.get("recommendations"):
            return False
        if self.elicitation_policy is None:
            return True
        if not self.elicitation_policy.allows_early_recommendations:
            return True
        return crs_out.get("asks_so_far", 0) >= self.elicitation_policy.target_asks

    def _build_opener(self) -> str:
        """First buyer utterance — generic, mimics how real shoppers start."""
        cat = self.category.replace("_", " ")
        return f"I'm looking for a {cat}."

    def _parse_endogenous_abandonment(self, reply: str) -> str | None:
        """Return the buyer's abandonment reason if the sentinel is present."""
        if not self.endogenous_abandonment:
            return None
        text = reply.strip()
        if not text.startswith(ABANDON_SENTINEL):
            return None
        reason = text[len(ABANDON_SENTINEL):].strip()
        return reason or "Buyer abandoned without a stated reason."

    def _coerce_product_number(self, value: Any) -> int | None:
        if type(value) is int:
            return value
        if isinstance(value, str) and value.strip().isdigit():
            return int(value.strip())
        return None

    def _coerce_money(self, value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        if isinstance(value, str):
            cleaned = value.strip().replace("$", "").replace(",", "")
            try:
                return float(cleaned)
            except ValueError:
                return None
        return None

    def _resolve_decision(self, decision: dict[str, Any], recs: list[dict[str, Any]]) -> dict[str, Any]:
        """Resolve buyer JSON into validated outcome fields.

        The buyer sees numbered products, not ASINs, so ``product_number`` is
        the only valid purchase identifier. Invalid choices and missing WTP are
        counted as no-purchase outcomes. WTP below price remains a purchase with
        negative surplus, which is directly identifiable from the recorded values.
        """
        resolved_decision = dict(decision)
        outcome_label = "NO_PURCHASE"
        purchased_asin = None
        actual_price = None
        consumer_surplus = None
        wtp = self._coerce_money(decision.get("willingness_to_pay"))
        if wtp is not None:
            resolved_decision["willingness_to_pay"] = wtp
        invalid_purchase = False

        if decision.get("decision") == "PURCHASE":
            chosen_asin = None
            product_number = self._coerce_product_number(decision.get("product_number"))
            if product_number is not None:
                resolved_decision["product_number"] = product_number
            if product_number is not None and 1 <= product_number <= len(recs):
                chosen_asin = recs[product_number - 1].get("asin")

            if not chosen_asin:
                invalid_purchase = True
            else:
                candidate_price = recs[product_number - 1].get("price")
                if candidate_price is None:
                    invalid_purchase = True
                else:
                    candidate_surplus = None
                    if wtp is None:
                        invalid_purchase = True
                    else:
                        try:
                            candidate_surplus = wtp - float(candidate_price)
                        except (TypeError, ValueError):
                            invalid_purchase = True
                    if not invalid_purchase:
                        outcome_label = "PURCHASE"
                        purchased_asin = chosen_asin
                        actual_price = candidate_price
                        consumer_surplus = candidate_surplus

            if invalid_purchase:
                resolved_decision["decision"] = "NO_PURCHASE"

        return {
            "decision": resolved_decision,
            "outcome_label": outcome_label,
            "purchased_asin": purchased_asin,
            "actual_price": actual_price,
            "wtp": wtp,
            "consumer_surplus": consumer_surplus,
        }

    def run(self) -> SimOutcome:
        """Run end-to-end and return the final outcome record.

        For live, step-by-step rendering use `run_iter()` instead — it yields
        events as each message / tool call / decision happens.
        """
        outcome: Optional[SimOutcome] = None
        for event in self.run_iter():
            if event.get("type") == "outcome":
                outcome = event["outcome"]
        assert outcome is not None, "run_iter must yield an 'outcome' event"
        return outcome

    def run_iter(self) -> Iterator[dict[str, Any]]:
        """Run the conversation and yield events as they happen.

        Event shapes:
            {"type": "buyer_message",  "content": str, "turn": int}
            {"type": "crs_message",    "content": str, "tool_calls": list, "bus_size": int, "turn": int}
            {"type": "recommendations","items": list}
            {"type": "decision",       "decision": dict}      # raw buyer JSON
            {"type": "abandoned",      "turn": int}
            {"type": "error",          "where": str, "error": str}
            {"type": "outcome",        "outcome": SimOutcome}  # always the final event
        """
        dialogue: list[dict[str, str]] = []
        tool_log: list[list[dict[str, Any]]] = []
        checkpoint_records: list[dict[str, Any]] = []
        ask_count = 0

        opener = self._build_opener()
        dialogue.append({"role": "user", "content": opener})
        yield {"type": "buyer_message", "content": opener, "turn": 1}

        # ---- CRS handles the opener ----
        try:
            crs_out = self.crs.chat(opener)
        except Exception as e:
            yield {"type": "error", "where": "crs.chat(opener)", "error": f"{type(e).__name__}: {e}"}
            yield {"type": "outcome", "outcome": SimOutcome(
                persona_id=self.persona.get("id", ""),
                category=self.category,
                outcome="PROTOCOL_ERROR",
                turns_used=0,
                asks=0,
                dialogue=dialogue,
                **self._policy_fields(),
                error=f"crs.chat(opener): {type(e).__name__}: {e}",
            )}
            return
        dialogue.append({"role": "assistant", "content": crs_out["reply"]})
        tool_log.append(crs_out["tool_calls_this_turn"])
        ask_count = crs_out.get("asks_so_far", 0)
        yield {
            "type": "crs_message",
            "content": crs_out["reply"],
            "tool_calls": crs_out["tool_calls_this_turn"],
            "bus_size": crs_out.get("bus_size"),
            "category": crs_out.get("category"),
            "turn": 1,
        }
        new_checkpoints = self._evaluate_new_checkpoints(crs_out, checkpoint_records)
        checkpoint_records.extend(new_checkpoints)
        for checkpoint in new_checkpoints:
            yield {"type": "checkpoint", "checkpoint": checkpoint, "turn": 0}

        if crs_out.get("terminal_status") == "NO_FEASIBLE_MATCH":
            yield {"type": "outcome", "outcome": SimOutcome(
                persona_id=self.persona.get("id", ""),
                category=self.category,
                outcome="NO_FEASIBLE_MATCH",
                turns_used=len(dialogue) // 2,
                asks=ask_count,
                dialogue=dialogue,
                crs_tool_calls_per_turn=tool_log,
                recommendation_checkpoints=checkpoint_records,
                **self._policy_fields(),
                **self._protocol_fields(crs_out),
            )}
            return

        # ---- main loop ----
        for turn in range(2, self.max_turns + 1):
            if crs_out.get("terminal_status") == "NO_FEASIBLE_MATCH":
                yield {"type": "outcome", "outcome": SimOutcome(
                    persona_id=self.persona.get("id", ""),
                    category=self.category,
                    outcome="NO_FEASIBLE_MATCH",
                    turns_used=len(dialogue) // 2,
                    asks=ask_count,
                    dialogue=dialogue,
                    crs_tool_calls_per_turn=tool_log,
                    recommendation_checkpoints=checkpoint_records,
                    **self._policy_fields(),
                    **self._protocol_fields(crs_out),
                )}
                return
            if crs_out.get("recommendations"):
                yield {"type": "recommendations", "items": crs_out["recommendations"]}
                if self._recommendations_are_terminal(crs_out):
                    break

            # Abandonment hazard.
            if self.eta > 0 and self._rng.random() < self.eta:
                yield {"type": "abandoned", "turn": turn}
                yield {"type": "outcome", "outcome": SimOutcome(
                    persona_id=self.persona.get("id", ""),
                    category=self.category,
                    outcome="ABANDONED",
                    turns_used=len(dialogue) // 2,
                    asks=ask_count,
                    abandoned_at_turn=turn,
                    abandonment_type="exogenous",
                    dialogue=dialogue,
                    crs_tool_calls_per_turn=tool_log,
                    recommendation_checkpoints=checkpoint_records,
                    **self._policy_fields(),
                    **self._protocol_fields(crs_out),
                )}
                return

            # Buyer responds.
            try:
                buyer_reply = self.buyer.respond(crs_out["reply"])
            except Exception as e:
                yield {"type": "error", "where": "buyer.respond", "error": f"{type(e).__name__}: {e}"}
                yield {"type": "outcome", "outcome": SimOutcome(
                    persona_id=self.persona.get("id", ""),
                    category=self.category,
                    outcome="PROTOCOL_ERROR",
                    turns_used=len(dialogue) // 2,
                    asks=ask_count,
                    dialogue=dialogue,
                    crs_tool_calls_per_turn=tool_log,
                    recommendation_checkpoints=checkpoint_records,
                    **self._policy_fields(),
                    error=f"buyer.respond: {type(e).__name__}: {e}",
                )}
                return
            dialogue.append({"role": "user", "content": buyer_reply})
            yield {"type": "buyer_message", "content": buyer_reply, "turn": turn}

            abandonment_reason = self._parse_endogenous_abandonment(buyer_reply)
            if abandonment_reason is not None:
                yield {
                    "type": "abandoned",
                    "turn": turn,
                    "reason": abandonment_reason,
                    "abandonment_type": "endogenous",
                }
                yield {"type": "outcome", "outcome": SimOutcome(
                    persona_id=self.persona.get("id", ""),
                    category=self.category,
                    outcome="ABANDONED",
                    turns_used=len(dialogue) // 2,
                    asks=ask_count,
                    abandoned_at_turn=turn,
                    abandonment_type="endogenous",
                    abandonment_reason=abandonment_reason,
                    dialogue=dialogue,
                    crs_tool_calls_per_turn=tool_log,
                    recommendation_checkpoints=checkpoint_records,
                    **self._policy_fields(),
                    **self._protocol_fields(crs_out),
                )}
                return

            # CRS handles.
            try:
                crs_out = self.crs.chat(buyer_reply)
            except Exception as e:
                yield {"type": "error", "where": "crs.chat", "error": f"{type(e).__name__}: {e}"}
                yield {"type": "outcome", "outcome": SimOutcome(
                    persona_id=self.persona.get("id", ""),
                    category=self.category,
                    outcome="PROTOCOL_ERROR",
                    turns_used=len(dialogue) // 2,
                    asks=ask_count,
                    dialogue=dialogue,
                    crs_tool_calls_per_turn=tool_log,
                    recommendation_checkpoints=checkpoint_records,
                    **self._policy_fields(),
                    error=f"crs.chat: {type(e).__name__}: {e}",
                )}
                return
            dialogue.append({"role": "assistant", "content": crs_out["reply"]})
            tool_log.append(crs_out["tool_calls_this_turn"])
            ask_count = crs_out.get("asks_so_far", ask_count)
            yield {
                "type": "crs_message",
                "content": crs_out["reply"],
                "tool_calls": crs_out["tool_calls_this_turn"],
                "bus_size": crs_out.get("bus_size"),
                "category": crs_out.get("category"),
                "turn": turn,
            }
            new_checkpoints = self._evaluate_new_checkpoints(crs_out, checkpoint_records)
            checkpoint_records.extend(new_checkpoints)
            for checkpoint in new_checkpoints:
                yield {"type": "checkpoint", "checkpoint": checkpoint, "turn": turn}

        # ---- terminal ----
        if not self._recommendations_are_terminal(crs_out):
            # Ran out of turns without a terminal recommendation.
            yield {"type": "outcome", "outcome": SimOutcome(
                persona_id=self.persona.get("id", ""),
                category=self.category,
                outcome="PROTOCOL_ERROR",
                turns_used=len(dialogue) // 2,
                asks=ask_count,
                dialogue=dialogue,
                crs_tool_calls_per_turn=tool_log,
                recommendation_checkpoints=checkpoint_records,
                **self._policy_fields(),
                **self._protocol_fields(crs_out),
                error="max_turns_reached_without_final_recommendation",
            )}
            return

        recs = crs_out["recommendations"]
        try:
            decision = self.buyer.decide(recs)
        except Exception as e:
            yield {"type": "error", "where": "buyer.decide", "error": f"{type(e).__name__}: {e}"}
            yield {"type": "outcome", "outcome": SimOutcome(
                persona_id=self.persona.get("id", ""),
                category=self.category,
                outcome="PROTOCOL_ERROR",
                turns_used=len(dialogue) // 2,
                asks=ask_count,
                crs_recommendations=recs,
                dialogue=dialogue,
                crs_tool_calls_per_turn=tool_log,
                recommendation_checkpoints=checkpoint_records,
                **self._policy_fields(),
                error=f"buyer.decide: {type(e).__name__}: {e}",
            )}
            return

        resolved = self._resolve_decision(decision, recs)
        reason = resolved["decision"].get("reasoning") or ""
        outcome_label = resolved["outcome_label"]
        dialogue.append({"role": "user", "content": f"{reason} [{outcome_label}]".strip()})
        yield {"type": "decision", **resolved}

        yield {"type": "outcome", "outcome": SimOutcome(
            persona_id=self.persona.get("id", ""),
            category=self.category,
            outcome=outcome_label,
            turns_used=len(dialogue) // 2,
            asks=ask_count,
            purchased_asin=resolved["purchased_asin"],
            wtp=resolved["wtp"],
            actual_price=resolved["actual_price"],
            consumer_surplus=resolved["consumer_surplus"],
            dialogue=dialogue,
            crs_recommendations=recs,
            crs_tool_calls_per_turn=tool_log,
            recommendation_checkpoints=checkpoint_records,
            buyer_decision_raw=resolved["decision"],
            **self._policy_fields(),
            **self._protocol_fields(crs_out),
        )}
