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
from typing import Any, Iterator, Optional

from sandbox.agents.buyer import BuyerAgent
from sandbox.agents.langgraph_crs import CRSAgentSession
from sandbox.elicitation_policy import ElicitationPolicy


@dataclass
class SimOutcome:
    """Structured result from one simulated conversation."""

    persona_id: str
    category: str
    outcome: str                                # "PURCHASE" | "NO_PURCHASE" | "ABANDONED" | "ERROR"
    turns_used: int                             # how many user-assistant exchanges happened
    asks: int                                   # clarifying questions the CRS asked
    purchased_asin: Optional[str] = None
    wtp: Optional[float] = None
    actual_price: Optional[float] = None
    consumer_surplus: Optional[float] = None    # wtp - actual_price if both present
    abandoned_at_turn: Optional[int] = None
    dialogue: list = field(default_factory=list)
    crs_recommendations: Optional[list] = None
    crs_tool_calls_per_turn: list = field(default_factory=list)
    buyer_decision_raw: Optional[dict] = None
    policy: Optional[str] = None
    numquestions: Optional[int] = None
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
    seed: int = 0
    elicitation_policy: Optional[ElicitationPolicy] = None

    # Allow caller to pass pre-constructed agents for testing/customization.
    buyer: Optional[BuyerAgent] = None
    crs: Optional[CRSAgentSession] = None

    def __post_init__(self) -> None:
        if self.buyer is None:
            self.buyer = BuyerAgent(persona=self.persona, category=self.category)
        if self.crs is None:
            self.crs = CRSAgentSession(elicitation_policy=self.elicitation_policy)
        self._rng = random.Random(self.seed)

    # ------------------------------------------------------------------

    def _policy_fields(self) -> dict[str, Any]:
        if self.elicitation_policy is None:
            return {}
        return {
            "policy": self.elicitation_policy.name,
            "numquestions": self.elicitation_policy.target_asks,
        }

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
                outcome="ERROR",
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

        # ---- main loop ----
        for turn in range(2, self.max_turns + 1):
            if crs_out.get("recommendations"):
                yield {"type": "recommendations", "items": crs_out["recommendations"]}
                if self._recommendations_are_terminal(crs_out):
                    break
                if self.elicitation_policy and self.elicitation_policy.allows_early_recommendations:
                    recs = crs_out["recommendations"]
                    try:
                        decision = self.buyer.decide(recs)
                    except Exception as e:
                        yield {"type": "error", "where": "buyer.decide", "error": f"{type(e).__name__}: {e}"}
                        yield {"type": "outcome", "outcome": SimOutcome(
                            persona_id=self.persona.get("id", ""),
                            category=self.category,
                            outcome="ERROR",
                            turns_used=len(dialogue) // 2,
                            asks=ask_count,
                            crs_recommendations=recs,
                            dialogue=dialogue,
                            crs_tool_calls_per_turn=tool_log,
                            **self._policy_fields(),
                            error=f"buyer.decide: {type(e).__name__}: {e}",
                        )}
                        return
                    if decision.get("decision") == "PURCHASE":
                        chosen = None
                        pn = decision.get("product_number")
                        if isinstance(pn, int) and 1 <= pn <= len(recs):
                            chosen = recs[pn - 1]["asin"]
                        else:
                            raw_asin = decision.get("asin")
                            if raw_asin and any(r.get("asin") == raw_asin for r in recs):
                                chosen = raw_asin
                        purchased_asin = chosen
                        actual_price = None
                        if purchased_asin:
                            for r in recs:
                                if r.get("asin") == purchased_asin:
                                    actual_price = r.get("price")
                                    break
                        wtp = decision.get("willingness_to_pay")
                        consumer_surplus = None
                        if wtp is not None and actual_price is not None:
                            try:
                                consumer_surplus = float(wtp) - float(actual_price)
                            except (TypeError, ValueError):
                                consumer_surplus = None

                        reason = decision.get("reasoning") or ""
                        dialogue.append({"role": "user", "content": f"{reason} [PURCHASE]".strip()})
                        yield {"type": "decision", "decision": decision, "outcome_label": "PURCHASE",
                               "purchased_asin": purchased_asin, "actual_price": actual_price,
                               "wtp": wtp, "consumer_surplus": consumer_surplus}
                        yield {"type": "outcome", "outcome": SimOutcome(
                            persona_id=self.persona.get("id", ""),
                            category=self.category,
                            outcome="PURCHASE",
                            turns_used=len(dialogue) // 2,
                            asks=ask_count,
                            purchased_asin=purchased_asin,
                            wtp=wtp,
                            actual_price=actual_price,
                            consumer_surplus=consumer_surplus,
                            dialogue=dialogue,
                            crs_recommendations=recs,
                            crs_tool_calls_per_turn=tool_log,
                            buyer_decision_raw=decision,
                            **self._policy_fields(),
                        )}
                        return

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
                    dialogue=dialogue,
                    crs_tool_calls_per_turn=tool_log,
                    **self._policy_fields(),
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
                    outcome="ERROR",
                    turns_used=len(dialogue) // 2,
                    asks=ask_count,
                    dialogue=dialogue,
                    crs_tool_calls_per_turn=tool_log,
                    **self._policy_fields(),
                    error=f"buyer.respond: {type(e).__name__}: {e}",
                )}
                return
            dialogue.append({"role": "user", "content": buyer_reply})
            yield {"type": "buyer_message", "content": buyer_reply, "turn": turn}

            # CRS handles.
            try:
                crs_out = self.crs.chat(buyer_reply)
            except Exception as e:
                yield {"type": "error", "where": "crs.chat", "error": f"{type(e).__name__}: {e}"}
                yield {"type": "outcome", "outcome": SimOutcome(
                    persona_id=self.persona.get("id", ""),
                    category=self.category,
                    outcome="ERROR",
                    turns_used=len(dialogue) // 2,
                    asks=ask_count,
                    dialogue=dialogue,
                    crs_tool_calls_per_turn=tool_log,
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

        # ---- terminal ----
        if not self._recommendations_are_terminal(crs_out):
            # Ran out of turns without a terminal recommendation.
            yield {"type": "outcome", "outcome": SimOutcome(
                persona_id=self.persona.get("id", ""),
                category=self.category,
                outcome="NO_PURCHASE",
                turns_used=len(dialogue) // 2,
                asks=ask_count,
                dialogue=dialogue,
                crs_tool_calls_per_turn=tool_log,
                **self._policy_fields(),
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
                outcome="ERROR",
                turns_used=len(dialogue) // 2,
                asks=ask_count,
                crs_recommendations=recs,
                dialogue=dialogue,
                crs_tool_calls_per_turn=tool_log,
                **self._policy_fields(),
                error=f"buyer.decide: {type(e).__name__}: {e}",
            )}
            return

        purchased_asin = None
        actual_price = None
        wtp = decision.get("willingness_to_pay")
        if decision.get("decision") == "PURCHASE":
            chosen = None
            pn = decision.get("product_number")
            if isinstance(pn, int) and 1 <= pn <= len(recs):
                chosen = recs[pn - 1]["asin"]
            else:
                raw_asin = decision.get("asin")
                if raw_asin and any(r.get("asin") == raw_asin for r in recs):
                    chosen = raw_asin
            purchased_asin = chosen
            if purchased_asin:
                for r in recs:
                    if r.get("asin") == purchased_asin:
                        actual_price = r.get("price")
                        break
            outcome_label = "PURCHASE"
        else:
            outcome_label = "NO_PURCHASE"

        consumer_surplus = None
        if outcome_label == "PURCHASE" and wtp is not None and actual_price is not None:
            try:
                consumer_surplus = float(wtp) - float(actual_price)
            except (TypeError, ValueError):
                consumer_surplus = None

        reason = decision.get("reasoning") or ""
        dialogue.append({"role": "user", "content": f"{reason} [{outcome_label}]".strip()})
        yield {"type": "decision", "decision": decision, "outcome_label": outcome_label,
               "purchased_asin": purchased_asin, "actual_price": actual_price,
               "wtp": wtp, "consumer_surplus": consumer_surplus}

        yield {"type": "outcome", "outcome": SimOutcome(
            persona_id=self.persona.get("id", ""),
            category=self.category,
            outcome=outcome_label,
            turns_used=len(dialogue) // 2,
            asks=ask_count,
            purchased_asin=purchased_asin,
            wtp=wtp,
            actual_price=actual_price,
            consumer_surplus=consumer_surplus,
            dialogue=dialogue,
            crs_recommendations=recs,
            crs_tool_calls_per_turn=tool_log,
            buyer_decision_raw=decision,
            **self._policy_fields(),
        )}
