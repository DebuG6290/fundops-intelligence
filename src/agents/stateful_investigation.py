"""Explicit, bounded live investigation over the existing deterministic tools.

The provider chooses one next action per turn and supplies short audit reasons.
Only tool outputs become current-case evidence. No hidden reasoning transcript is
stored, and historical cases are passed back solely as investigation analogies.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, model_validator

from src.agents.agent_loop import AgentRun
from src.agents.evidence import evidence_from_tool_result
from src.agents.evidence_challenge import EvidenceChallengeAgent
from src.agents.schemas import Hypothesis, InvestigationReport, ToolTrace
from src.agents.state import InvestigationState
from src.agents.tool_registry import ToolRegistry
from src.models.evidence import EvidenceItem, EvidenceSourceType


class InvestigationTurn(BaseModel):
    """One auditable planning decision, not a chain-of-thought transcript."""

    hypotheses: list[Hypothesis] = Field(default_factory=list, max_length=5)
    next_action: str | None = None
    action_arguments: dict[str, Any] = Field(default_factory=dict)
    decision_rationale: str = Field(min_length=1, max_length=300)
    stop: bool = False
    stopping_rationale: str = Field(default="", max_length=300)
    recommended_next_step: str = Field(default="Review the current records with a human operator.", min_length=1, max_length=300)

    @model_validator(mode="after")
    def unique_hypotheses(self) -> InvestigationTurn:
        ids = [item.hypothesis_id for item in self.hypotheses]
        if len(ids) != len(set(ids)):
            raise ValueError("Hypothesis IDs must be unique within a planning turn")
        return self


_PLANNING_PROMPT = """You are a fund-operations investigator, not an approver.
Return exactly one JSON object with hypotheses (maximum five, stable IDs),
next_action (one available tool name or null), action_arguments (object),
decision_rationale (one concise auditable sentence, not private reasoning),
stop (boolean), stopping_rationale, and recommended_next_step.
Shape: {"hypotheses":[{"hypothesis_id":"HYP-001","root_cause":"CAUSE:ID",
"rationale":"brief observable basis","confidence":0.5,
"required_evidence":["tool or source needed"],"uncertainty":"brief gap"}],
"next_action":"tool_name_or_null","action_arguments":{},
"decision_rationale":"one sentence","stop":false,
"stopping_rationale":"","recommended_next_step":"human review action"}.
Choose the next tool from current evidence and unresolved questions; never
follow a fixed tool order. Reassess hypotheses after every result. A prior case
is an analogy that may prioritize a check, never evidence of the current cause.
Only deterministic current-case tool outputs can support or contradict a cause.
If evidence contradicts the leader, select a different check or escalate.
If required primary evidence is absent, do not claim a resolved cause.
Stop only when current evidence is sufficient or no useful action remains.
Every outcome remains subject to human review. Never alter financial records.
"""


def timeline_event(stage: str, rationale: str, **details: Any) -> dict[str, Any]:
    """Small, serializable event for audit and UI; no raw model response."""
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "rationale": rationale[:300],
        "details": details,
    }


class StatefulInvestigationLoop:
    def __init__(self, provider: Any, tools: ToolRegistry, *, max_turns: int = 8) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be positive")
        self.provider = provider
        self.tools = tools
        self.max_turns = max_turns
        self.challenger = EvidenceChallengeAgent()

    def run(self, context: dict[str, Any], *, exception_id: str) -> AgentRun:
        run = AgentRun()
        state = InvestigationState(exception={"exception_id": exception_id})
        run.timeline.append(timeline_event(
            "initial_observation", "The detected exception and deterministic analytics opened the investigation.",
            exception_id=exception_id, observable_exception=context.get("exception", context),
        ))
        used_actions: set[str] = set()
        analogies: list[dict[str, Any]] = []
        previous_hypotheses: list[dict[str, Any]] = []
        recommendation = "Review the current records with a human operator."
        challenge_summary: dict[str, Any] | None = None
        try:
            for turn_number in range(1, self.max_turns + 1):
                response = self.provider.create_response(
                    system_instructions=_PLANNING_PROMPT,
                    context={
                        "initial_context": context,
                        "current_hypotheses": state.hypotheses,
                        "current_evidence": [_evidence_context(item) for item in state.evidence
                                             if item.source_type != EvidenceSourceType.HISTORICAL_CASE],
                        "historical_analogies": analogies,
                        "previous_challenge": challenge_summary,
                        "executed_actions": [step.tool_name for step in run.trace],
                        "available_tools": self.tools.definitions(),
                        "remaining_turns": self.max_turns - turn_number + 1,
                    },
                    tools=[],
                )
                self._record_usage(run)
                turn = self._parse_turn(getattr(response, "output_text", "") or "", run)
                recommendation = turn.recommended_next_step
                if turn.hypotheses:
                    state.hypotheses = [item.model_dump() for item in turn.hypotheses]
                    run.timeline.append(timeline_event(
                        "hypotheses" if not previous_hypotheses else "hypothesis_update",
                        turn.decision_rationale, turn=turn_number,
                        hypotheses=state.hypotheses,
                    ))
                    previous_hypotheses = list(state.hypotheses)
                if turn.stop:
                    challenge_summary = self._challenge(state, run, turn_number)
                    safe_to_stop = not (challenge_summary["contradiction_found"] or
                                        challenge_summary["ambiguity_found"] or
                                        challenge_summary["insufficient_evidence"])
                    run.timeline.append(timeline_event(
                        "stopping_decision", turn.stopping_rationale or turn.decision_rationale,
                        turn=turn_number, accepted=safe_to_stop or turn_number == self.max_turns,
                        challenge=challenge_summary,
                    ))
                    if safe_to_stop or turn_number == self.max_turns:
                        break
                    run.timeline.append(timeline_event(
                        "replan", "The evidence challenge requires another investigation action.",
                        turn=turn_number, reason=challenge_summary["recommendation"],
                    ))
                    continue
                if not turn.next_action:
                    run.timeline.append(timeline_event(
                        "replan", "No next action was selected; another action or an explicit stop is required.",
                        turn=turn_number,
                    ))
                    continue
                name = turn.next_action
                if name not in {item["name"] for item in self.tools.definitions()}:
                    raise ValueError(f"Investigator selected an unavailable tool: {name}")
                signature = json.dumps([name, turn.action_arguments], sort_keys=True, default=str)
                if signature in used_actions:
                    run.timeline.append(timeline_event(
                        "replan", "The same action and arguments were already executed; choose a different check.",
                        turn=turn_number, tool_name=name,
                    ))
                    continue
                used_actions.add(signature)
                run.timeline.append(timeline_event(
                    "action_selected", turn.decision_rationale,
                    turn=turn_number, tool_name=name, arguments=turn.action_arguments,
                ))
                result = self.tools.execute(name, turn.action_arguments)
                run.telemetry.record_tool_call()
                run.trace.append(ToolTrace(tool_name=name, arguments=turn.action_arguments, result=result))
                state.add_observation("tool_execution", {"tool_name": name}, name)
                if name == "compare_price_sources":
                    state.add_observation("price_source_check", result, name)
                run.timeline.append(timeline_event(
                    "tool_execution", "The deterministic tool completed without changing financial records.",
                    turn=turn_number, tool_name=name, record_count=len(result) if isinstance(result, list) else None,
                ))
                returned = evidence_from_tool_result(name, result, exception_id)
                state.evidence.extend(returned)
                run.evidence.extend(returned)
                if name == "search_historical_cases":
                    analogies = [_analogy_context(case) for case in result or []]
                    run.timeline.append(timeline_event(
                        "memory_retrieval", "Human-validated historical paths were retrieved as analogies, not current-case proof.",
                        turn=turn_number, case_ids=[case["case_id"] for case in analogies],
                        suggested_checks=[check for case in analogies for check in case["investigation_path"]],
                    ))
                run.timeline.append(timeline_event(
                    "evidence_returned", "The tool returned sourced records; historical cases remain analogies.",
                    turn=turn_number, tool_name=name,
                    evidence_ids=[item.evidence_id for item in returned],
                    source_types=[item.source_type.value for item in returned],
                ))
                challenge_summary = self._challenge(state, run, turn_number)
                if challenge_summary["contradiction_found"] or challenge_summary["insufficient_evidence"]:
                    run.timeline.append(timeline_event(
                        "replan", challenge_summary["recommendation"],
                        turn=turn_number, reason=challenge_summary["recommendation"],
                    ))
            else:
                run.timeline.append(timeline_event(
                    "stopping_decision", "Investigation turn limit reached; unresolved evidence must be escalated.",
                    accepted=True, bounded_limit=self.max_turns,
                ))
            # Recheck the *final* hypothesis state even when the model spent
            # every turn declining tools or repeating an action.
            challenge_summary = self._challenge(state, run, turn_number)
            run.report = _report_from_state(state, recommendation, challenge_summary)
            return run
        finally:
            run.telemetry.finish()

    def _challenge(self, state: InvestigationState, run: AgentRun, turn: int) -> dict[str, Any]:
        _link_evidence(state)
        challenge = asdict(self.challenger.review(state))
        run.timeline.append(timeline_event(
            "challenge_assessment", challenge["recommendation"], turn=turn,
            supporting_evidence_ids=challenge.get("supporting_evidence_ids", []),
            contradictory_evidence_ids=challenge.get("contradictory_evidence_ids", []),
            missing_evidence=challenge.get("missing_evidence", []),
            contradiction_found=challenge["contradiction_found"],
            ambiguity_found=challenge["ambiguity_found"],
            escalation_required=bool(challenge["contradiction_found"] or challenge["ambiguity_found"] or challenge["insufficient_evidence"]),
        ))
        return challenge

    def _record_usage(self, run: AgentRun) -> None:
        usage = getattr(self.provider, "last_call", None) or {}
        run.telemetry.provider = usage.get("provider", run.telemetry.provider)
        run.telemetry.model = usage.get("model", run.telemetry.model)
        if usage.get("request_id"):
            run.telemetry.request_ids.append(str(usage["request_id"]))
        for field in ("input_tokens", "output_tokens"):
            if usage.get(field) is not None:
                setattr(run.telemetry, field, (getattr(run.telemetry, field) or 0) + int(usage[field]))

    def _parse_turn(self, output: str, run: AgentRun) -> InvestigationTurn:
        try:
            return InvestigationTurn.model_validate_json(output)
        except Exception as exc:
            repair = getattr(self.provider, "repair_structured_response", None)
            if repair is None:
                raise ValueError("Investigator did not return a valid structured planning turn") from exc
            response = repair(invalid_output=output, validation_error=str(exc), system_instructions=_PLANNING_PROMPT)
            self._record_usage(run)
            try:
                return InvestigationTurn.model_validate_json(getattr(response, "output_text", "") or "")
            except Exception as repaired_exc:
                raise ValueError("Investigator did not return a valid structured planning turn") from repaired_exc


def _evidence_context(item: EvidenceItem) -> dict[str, Any]:
    return {"evidence_id": item.evidence_id, "source_type": item.source_type.value,
            "source_name": item.source_name, "claim": item.claim,
            "supports": item.supports, "contradicts": item.contradicts}


def _analogy_context(case: dict[str, Any]) -> dict[str, Any]:
    # Deliberately omit historical root_cause and resolution from model input.
    return {"case_id": str(case.get("case_id")), "symptoms": case.get("symptoms", []),
            "investigation_path": case.get("investigation_path", []),
            "useful_evidence": case.get("useful_evidence", []), "role": "analogy_only"}


def _link_evidence(state: InvestigationState) -> None:
    linked: list[EvidenceItem] = []
    for item in state.evidence:
        # Relationships are recomputed as hypotheses evolve. A contradiction
        # of an earlier leader must not cling to a reused hypothesis ID.
        updates: dict[str, Any] = {"supports_hypothesis": None, "contradicts_hypothesis": None}
        for hypothesis in state.hypotheses:
            root = hypothesis["root_cause"]
            for source_field, link_field in (("supports", "supports_hypothesis"), ("contradicts", "contradicts_hypothesis")):
                target = getattr(item, source_field)
                if target == root or (root.startswith("PRICE_EXCEPTION:") and target == "PRICE_EXCEPTION"
                                      and item.metadata.get("security_id") == root.split(":", 1)[1]):
                    updates[link_field] = hypothesis["hypothesis_id"]
        linked.append(item.model_copy(update=updates))
    state.evidence[:] = linked


def _report_from_state(state: InvestigationState, recommendation: str, challenge: dict[str, Any] | None) -> InvestigationReport:
    leading = state.hypotheses[0] if state.hypotheses else None
    root = leading["root_cause"] if leading else "UNKNOWN"
    confidence = float(leading["confidence"]) if leading else 0.0
    if challenge and (challenge["contradiction_found"] or challenge["ambiguity_found"] or challenge["insufficient_evidence"]):
        confidence = min(confidence, .35)
        recommendation = challenge["recommendation"]
    supports = [item for item in state.evidence if item.source_type != EvidenceSourceType.HISTORICAL_CASE
                and leading and (item.supports_hypothesis == leading["hypothesis_id"] or item.supports == root)]
    counters = [item for item in state.evidence if item.source_type != EvidenceSourceType.HISTORICAL_CASE
                and leading and (item.contradicts_hypothesis == leading["hypothesis_id"] or item.contradicts == root)]
    return InvestigationReport(
        probable_root_cause=root, confidence=confidence,
        observations=["Hypotheses were updated against current-case deterministic tool outputs."],
        supporting_evidence=[item.claim for item in supports],
        counter_evidence=[item.claim for item in counters],
        recommended_next_step=recommendation,
        human_review_required=True,
        hypotheses=[Hypothesis.model_validate(item) for item in state.hypotheses],
        cited_evidence_ids=[item.evidence_id for item in [*supports, *counters]],
    )

