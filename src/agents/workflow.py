from __future__ import annotations

from dataclasses import asdict
from typing import Any

from src.agents.corporate_action_agent import CorporateActionInvestigationAgent
from src.agents.agent_loop import AgentRun
from src.agents.evidence_challenge import EvidenceChallengeAgent, apply_challenge
from src.agents.resolution import ResolutionAgent
from src.agents.router import InvestigationRouter
from src.agents.rule_based_investigator import RuleBasedInvestigator
from src.agents.nav_agent import NavInvestigationAgent
from src.agents.schemas import InvestigationReport, ToolTrace
from src.agents.state import InvestigationState
from src.agents.transaction_agent import TransactionInvestigationAgent
from src.data.scenarios import InvestigationScenario
from src.data.scenarios_extra import (
    CorporateActionScenario,
    TransactionMismatchScenario,
)
from src.memory.cases import CaseMemory, HistoricalCase
from src.models.evidence import EvidenceItem
from src.agents.evidence import _json_safe
from src.review.models import HumanDecision, HumanReviewRecord, HumanReviewSubmission
from src.review.service import HumanReviewService
from src.tools.exception_tools import find_corporate_actions_tool, find_transaction_mismatches_tool
from src.tools.memory_tools import search_historical_cases_tool
from src.tools.nav_tools import calculate_nav_variance_tool, compare_price_sources_tool, get_fund_snapshot_tool, identify_top_contributors_tool
from src.tools.investigation_tools import check_transaction_activity_tool, check_corporate_actions_tool, check_security_mapping_tool, check_fx_context_tool


def _state_from_specialist_report(
    scenario: InvestigationScenario | TransactionMismatchScenario | CorporateActionScenario,
    report: Any,
    evidence: list[EvidenceItem] | None = None,
) -> InvestigationState:
    """Adapt a specialist report to the shared deterministic control state."""
    if isinstance(scenario, TransactionMismatchScenario):
        exception_type = "TRANSACTION_MISMATCH"
    elif isinstance(scenario, CorporateActionScenario):
        exception_type = "CORPORATE_ACTION"
    else:
        exception_type = "NAV_DISCREPANCY"
    state = InvestigationState(
        exception={
            "exception_id": scenario.exception_id,
            "exception_type": exception_type,
        },
        recommended_action=report.recommended_next_step,
        confidence=report.confidence,
    )

    for observation in report.observations or []:
        state.add_observation(
            "specialist_observation",
            {"claim": observation, "source_type": "SPECIALIST_OBSERVATION"},
            "specialist_report",
        )
    state.evidence.extend(evidence or [])

    for claim in report.supporting_evidence or []:
        state.add_observation(
            "reported_supporting_claim",
            {"claim": claim, "source_type": "SPECIALIST_OBSERVATION"},
            "specialist_report",
        )

    # Preserve ranked structured hypotheses where the specialist supplies them.
    hypotheses = getattr(report, "hypotheses", None) or []
    for hypothesis in hypotheses:
        values = (
            hypothesis.model_dump()
            if hasattr(hypothesis, "model_dump")
            else hypothesis
        )
        state.add_hypothesis(
            root_cause=values["root_cause"],
            rationale=values.get("rationale", ""),
            confidence=values["confidence"],
            hypothesis_id=values.get("hypothesis_id"),
            required_evidence=values.get("required_evidence", []),
            uncertainty=values.get("uncertainty", ""),
        )
    if not state.hypotheses and report.probable_root_cause:
        state.add_hypothesis(
            root_cause=report.probable_root_cause,
            rationale="",
            confidence=report.confidence,
        )

    for counter_evidence in report.counter_evidence or []:
        state.add_observation(
            "reported_counter_evidence",
            {"claim": counter_evidence, "source_type": "SPECIALIST_OBSERVATION"},
            "specialist_report",
        )
    return state


def _control_outputs(
    workflow: InvestigationWorkflow,
    state: InvestigationState,
    report_evidence_is_narrative: bool = False,
) -> dict[str, Any]:
    _bind_evidence_to_leading_hypothesis(state)
    challenge = workflow.challenge_agent.review(state)
    apply_challenge(state, challenge)
    resolution = workflow.resolution_agent.resolve(state)
    serialized_evidence = [item.model_dump(mode="json") for item in state.evidence]
    return {
        "challenge": asdict(challenge),
        "resolution": asdict(resolution),
        "status": state.status,
        "probable_root_cause": (
            state.hypotheses[0]["root_cause"] if state.hypotheses else None
        ),
        "hypotheses": list(state.hypotheses),
        "confidence": state.confidence,
        "evidence": serialized_evidence,
        "supporting_evidence_references": [
            item.evidence_id for item in state.evidence
            if state.hypotheses and _evidence_matches_target(
                item, "supports_hypothesis", "supports", state.hypotheses[0]
            )
        ],
        "counter_evidence_references": [
            item.evidence_id for item in state.evidence
            if state.hypotheses and _evidence_matches_target(
                item, "contradicts_hypothesis", "contradicts", state.hypotheses[0]
            )
        ],
        "human_review_required": True,
        "report_evidence_is_narrative": report_evidence_is_narrative,
    }


def _bind_evidence_to_leading_hypothesis(state: InvestigationState) -> None:
    """Resolve legacy root-cause evidence keys to this run's stable hypothesis ID."""
    if not state.hypotheses:
        return
    leading = state.hypotheses[0]
    root_cause = leading["root_cause"]
    bound: list[EvidenceItem] = []
    for item in state.evidence:
        updates: dict[str, Any] = {"exception_id": item.exception_id or state.exception.get("exception_id")}
        for legacy_field, hypothesis_field in (
            ("supports", "supports_hypothesis"),
            ("contradicts", "contradicts_hypothesis"),
        ):
            target = getattr(item, legacy_field)
            if not target:
                continue
            matches = target == root_cause
            # NAV baseline uses the category-only PRICE_EXCEPTION hypothesis;
            # specialist reports may instead identify the affected security.
            if root_cause == "PRICE_EXCEPTION" and target.startswith("PRICE_EXCEPTION:"):
                matches = True
            elif root_cause.startswith("PRICE_EXCEPTION:") and target == "PRICE_EXCEPTION":
                security_id = root_cause.split(":", 1)[1]
                matches = item.metadata.get("security_id") == security_id
            if matches:
                updates[hypothesis_field] = leading["hypothesis_id"]
        bound.append(item.model_copy(update=updates))
    state.evidence[:] = bound


def _evidence_matches_target(
    item: EvidenceItem,
    hypothesis_field: str,
    legacy_field: str,
    hypothesis: dict[str, Any],
) -> bool:
    hypothesis_id = hypothesis.get("hypothesis_id")
    evidence_hypothesis_id = getattr(item, hypothesis_field)
    if hypothesis_id and evidence_hypothesis_id:
        return evidence_hypothesis_id == hypothesis_id
    return bool(
        not evidence_hypothesis_id
        and hypothesis.get("root_cause")
        and getattr(item, legacy_field) == hypothesis["root_cause"]
    )


def _deterministic_transaction_run(scenario: TransactionMismatchScenario, memory: CaseMemory) -> AgentRun:
    """Reproducible demo path: hypothesis is derived from actual reconciliation rows."""
    from src.agents.evidence import evidence_from_tool_result

    records = find_transaction_mismatches_tool(scenario)
    evidence = evidence_from_tool_result("find_transaction_mismatches", records, scenario.exception_id)
    historical = search_historical_cases_tool(
        memory, query="transaction mismatch settlement quantity",
        exception_type="TRANSACTION_MISMATCH", top_k=3,
    )
    evidence.extend(evidence_from_tool_result(
        "search_historical_cases", historical, scenario.exception_id
    ))
    first = records[0] if records else None
    if first:
        transaction_id = first.get("transaction_id", "unknown")
        if first.get("_merge") == "left_only":
            root_cause = f"MISSING_TRANSACTION:{transaction_id}"
        elif first.get("_merge") == "right_only":
            root_cause = f"EXTRA_TRANSACTION:{transaction_id}"
        elif first.get("quantity_difference") not in (None, 0, 0.0):
            root_cause = f"TRANSACTION_QUANTITY_MISMATCH:{transaction_id}"
        else:
            root_cause = f"TRANSACTION_TYPE_MISMATCH:{transaction_id}"
        confidence = 0.9
        observations = [f"Deterministic reconciliation returned {len(records)} mismatched transaction record(s)."]
    else:
        root_cause, confidence = "UNKNOWN", 0.0
        observations = ["Deterministic reconciliation returned no mismatched transaction records."]
    report = InvestigationReport(
        probable_root_cause=root_cause, confidence=confidence, observations=observations,
        supporting_evidence=[], counter_evidence=[],
        recommended_next_step="Review the source transaction and settlement records.",
        human_review_required=True,
    )
    return AgentRun(
        report=report,
        trace=[
            ToolTrace(tool_name="find_transaction_mismatches", arguments={}, result=records),
            ToolTrace(tool_name="search_historical_cases", arguments={
                "query": "transaction mismatch settlement quantity",
                "exception_type": "TRANSACTION_MISMATCH",
            }, result=historical),
        ],
        evidence=evidence,
    )


def _deterministic_corporate_action_run(scenario: CorporateActionScenario, memory: CaseMemory) -> AgentRun:
    """Reproducible demo path: hypothesis is derived from actual action records."""
    from src.agents.evidence import evidence_from_tool_result

    records = find_corporate_actions_tool(scenario)
    evidence = evidence_from_tool_result("find_effective_corporate_actions", records, scenario.exception_id)
    historical = search_historical_cases_tool(
        memory, query="corporate action effective security event",
        exception_type="CORPORATE_ACTION", top_k=3,
    )
    evidence.extend(evidence_from_tool_result(
        "search_historical_cases", historical, scenario.exception_id
    ))
    first = records[0] if records else None
    if first:
        root_cause, confidence = f"CORPORATE_ACTION:{first.get('security_id')}", 0.9
        observations = [f"The deterministic action lookup returned {len(records)} effective record(s)."]
    else:
        root_cause, confidence = "UNKNOWN", 0.0
        observations = ["No effective corporate-action record was returned for this date."]
    report = InvestigationReport(
        probable_root_cause=root_cause, confidence=confidence, observations=observations,
        supporting_evidence=[], counter_evidence=[],
        recommended_next_step="Verify the event terms and operational position adjustment.",
        human_review_required=True,
    )
    return AgentRun(
        report=report,
        trace=[
            ToolTrace(tool_name="find_effective_corporate_actions", arguments={}, result=records),
            ToolTrace(tool_name="search_historical_cases", arguments={
                "query": "corporate action effective security event",
                "exception_type": "CORPORATE_ACTION",
            }, result=historical),
        ],
        evidence=evidence,
    )


def _memory_context_from_evidence(
    evidence: list[EvidenceItem],
    *,
    default_check_order: list[str] | None = None,
    actual_check_order: list[str] | None = None,
    influencing_case_ids: list[str] | None = None,
    relevant_checks: list[str] | None = None,
) -> dict[str, Any]:
    cases = []
    for item in evidence:
        if item.source_type.value != "HISTORICAL_CASE":
            continue
        meta = item.metadata
        if meta.get("human_validated") is not True:
            continue
        cases.append({key: meta[key] for key in (
            "case_id", "title", "historical_root_cause", "human_validated",
            "investigation_path", "useful_evidence",
        ) if key in meta})
    return {
        "retrieved_cases": cases,
        "retrieved_case_count": len(cases),
        "prior_investigation_paths": [case["investigation_path"] for case in cases if case.get("investigation_path")],
        "memory_influence": None,
        "relevant_case_ids": [
            case["case_id"] for case in cases
            if case.get("case_id") and any(
                check in (case.get("investigation_path") or [])
                for check in (relevant_checks or [])
            )
        ],
        "default_check_order": default_check_order or [],
        "actual_check_order": actual_check_order or [],
        "influencing_case_ids": influencing_case_ids or [],
    }


def _deterministic_adaptive_nav_run(scenario: InvestigationScenario, memory: CaseMemory):
    """Run an offline investigation; validated, relevant analogies may reorder checks."""
    from src.agents.agent_loop import AgentRun
    from src.agents.evidence import evidence_from_tool_result
    from src.agents.schemas import InvestigationReport

    calls: list[dict[str, Any]] = []
    evidence: list[EvidenceItem] = []
    def record(name: str, args: dict[str, Any], value: Any) -> Any:
        calls.append({"tool_name": name, "arguments": args, "result": value})
        return value

    nav_analytics = calculate_nav_variance_tool(scenario)
    evidence.extend(evidence_from_tool_result("calculate_nav_variance", nav_analytics, scenario.exception_id))
    record("calculate_nav_variance", {}, nav_analytics)
    snapshot = get_fund_snapshot_tool(scenario)
    evidence.extend(evidence_from_tool_result("get_fund_snapshot", snapshot, scenario.exception_id))
    record("get_fund_snapshot", {}, snapshot)
    # Search using observable symptoms/context, never the hidden answer key.
    query_terms = ["NAV variance"]
    if scenario.expected_prices is not None and scenario.calculated_prices is not None:
        query_terms.append("price vendor discrepancy")
    if scenario.expected_transactions is not None:
        query_terms.append("transaction position mismatch")
    if scenario.corporate_actions is not None and not scenario.corporate_actions.empty:
        query_terms.append("corporate action event records")
    query = " ".join(query_terms)
    candidates = search_historical_cases_tool(
        memory, query, "NAV_DISCREPANCY", 5
    )
    historical = [case for case in candidates if case.get("human_validated") is True]
    record(
        "search_historical_cases",
        {"query": query, "exception_type": "NAV_DISCREPANCY", "human_validated_only": True},
        historical,
    )
    evidence.extend(evidence_from_tool_result("search_historical_cases", historical, scenario.exception_id))

    default_order = ["check_corporate_actions", "check_transaction_activity"]
    choices = list(default_order)
    influencing_case_ids: list[str] = []
    available_checks = set()
    if scenario.expected_transactions is not None:
        available_checks.add("check_transaction_activity")
    if scenario.corporate_actions is not None and not scenario.corporate_actions.empty:
        available_checks.add("check_corporate_actions")
    for case in historical:
        path = case.get("investigation_path", [])
        suggested = list(dict.fromkeys(name for name in path if name in available_checks))
        if suggested:
            proposed_order = suggested + [name for name in default_order if name not in suggested]
            if proposed_order != default_order:
                choices = proposed_order
                influencing_case_ids = [str(case["case_id"])]
                break
    memory_order = {
        "default_check_order": default_order,
        "actual_check_order": list(choices),
        "influencing_case_ids": influencing_case_ids,
        "relevant_checks": [name for name in default_order if name in available_checks],
        "memory_influence": (
            f"{influencing_case_ids[0]} prioritized "
            f"{'transaction reconciliation' if choices[0] == 'check_transaction_activity' else 'corporate-action checks'} "
            f"before {'transaction reconciliation' if choices[1] == 'check_transaction_activity' else 'corporate-action checks'}."
            if influencing_case_ids else None
        ),
    }

    contributors = record("identify_top_contributors", {"top_n": 3}, identify_top_contributors_tool(scenario, 3))
    evidence.extend(evidence_from_tool_result("identify_top_contributors", contributors, scenario.exception_id))
    security_id = contributors[0]["security_id"] if contributors else None
    price = record("compare_price_sources", {"security_id": security_id}, compare_price_sources_tool(scenario, security_id) if security_id else {"found": False})
    evidence.extend(evidence_from_tool_result("compare_price_sources", price, scenario.exception_id))

    results: dict[str, Any] = {}
    for name in choices:
        tool = check_transaction_activity_tool if name == "check_transaction_activity" else check_corporate_actions_tool
        result = record(name, {"security_id": security_id}, tool(scenario, security_id))
        results[name] = result

    mapping = record("check_security_mapping", {"security_id": security_id or ""}, check_security_mapping_tool(scenario, security_id) if security_id else {"valid": False})
    fx = record("check_fx_context", {"security_id": security_id}, check_fx_context_tool(scenario, security_id))
    del mapping, fx

    if price.get("found") and abs(float(price.get("difference_pct", 0))) >= 10:
        root, confidence = f"PRICE_EXCEPTION:{security_id}", 0.9
        claim = f"Current price records show a {price['difference_pct']}% difference for {security_id}."
        evidence.append(EvidenceItem(source_type="PRICE_SOURCE", source_name=str(security_id), claim=claim, supports="PRICE_EXCEPTION", exception_id=scenario.exception_id, metadata=price))
        rationale = claim
    elif results["check_transaction_activity"].get("records"):
        row = results["check_transaction_activity"]["records"][0]
        tid = str(row.get("transaction_id"))
        root = f"MISSING_TRANSACTION:{tid}" if row.get("_merge") == "left_only" else f"TRANSACTION_QUANTITY_MISMATCH:{tid}"
        confidence = 0.88
        claim = f"Transaction reconciliation returned an unexplained record for {tid}: quantity difference {row.get('quantity_difference')} (expected {row.get('expected_quantity')}, actual {row.get('actual_quantity')})."
        evidence.append(EvidenceItem(source_type="TRANSACTION_RECORD", source_name=tid, claim=claim, supports=root, exception_id=scenario.exception_id, metadata=row))
        rationale = claim
    elif results["check_corporate_actions"].get("records") and scenario.expected_holdings is not None:
        action = results["check_corporate_actions"]["records"][0]
        sid = str(action["security_id"])
        root, confidence = f"CORPORATE_ACTION:{sid}", 0.88
        claim = f"An effective {action.get('action_type')} record exists for {sid}; the event is relevant to the current position break."
        evidence.append(EvidenceItem(source_type="CORPORATE_ACTION_RECORD", source_name=str(action.get("corporate_action_id", sid)), claim=claim, supports=root, exception_id=scenario.exception_id, metadata=action))
        rationale = claim
    else:
        root, confidence = "UNKNOWN", 0.25
        rationale = "Available current-case records do not sufficiently attribute the NAV break."

    report = InvestigationReport(probable_root_cause=root, confidence=confidence, observations=[rationale], supporting_evidence=[], counter_evidence=[], recommended_next_step="Review the cited records and obtain any missing primary evidence before taking operational action.", human_review_required=True)
    return AgentRun(report=report, trace=[ToolTrace(**item) for item in calls], evidence=evidence), memory_order


class InvestigationWorkflow:
    """
    Orchestration layer.

    Deterministic NAV workflow remains available as the baseline. Transaction
    and corporate-action routes use specialist agent loops when a provider is
    supplied, allowing direct comparison of routing and agentic investigation.
    """

    def __init__(
        self,
        memory: CaseMemory,
        review_service: HumanReviewService | None = None,
    ) -> None:
        self.router = InvestigationRouter()
        self.memory = memory
        self.investigator = RuleBasedInvestigator(memory)
        self.challenge_agent = EvidenceChallengeAgent()
        self.resolution_agent = ResolutionAgent()
        self.review_service = review_service or HumanReviewService()
        self.accepted_cases_by_review_id: dict[str, HistoricalCase] = {}

    def run_nav(
        self, scenario: InvestigationScenario, provider: Any | None = None,
    ) -> dict[str, Any]:
        route = self.router.route("NAV_DISCREPANCY")
        memory_order: dict[str, Any] = {}
        if provider is None:
            run, memory_order = _deterministic_adaptive_nav_run(scenario, self.memory)
            state = _state_from_specialist_report(scenario, run.report, run.evidence)
            state.exception = {**calculate_nav_variance_tool(scenario), **state.exception}
            for step in run.trace:
                if step.tool_name == "identify_top_contributors":
                    state.add_observation("top_contributors", step.result, step.tool_name)
                elif step.tool_name == "compare_price_sources":
                    state.add_observation("price_source_check", step.result, step.tool_name)
        else:
            run = NavInvestigationAgent(self.memory, provider=provider).investigate(scenario)
            if run.report is None:
                raise RuntimeError("NAV specialist did not return a valid investigation report")
            state = _state_from_specialist_report(scenario, run.report, run.evidence)
            state.exception = {
                **calculate_nav_variance_tool(scenario),
                **state.exception,
            }
        exception = {
            **state.exception,
            "exception_id": scenario.exception_id,
            "exception_type": "NAV_DISCREPANCY",
        }

        memory_context = _memory_context_from_evidence(
            state.evidence,
            default_check_order=memory_order.get("default_check_order"),
            actual_check_order=memory_order.get("actual_check_order"),
            influencing_case_ids=memory_order.get("influencing_case_ids"),
            relevant_checks=memory_order.get("relevant_checks"),
        )
        memory_context["memory_influence"] = memory_order.get("memory_influence")

        result = {
            "route": route,
            "exception_id": scenario.exception_id,
            "exception": exception,
            "observations": _json_safe(state.observations),
            "hypotheses": state.hypotheses,
            **_control_outputs(self, state),
            "memory_context": memory_context,
            "trace": [item.model_dump(mode="json") for item in run.trace],
        }
        if provider is not None:
            result.update({
                "exception_id": scenario.exception_id,
                "investigator_mode": "specialist_agent",
                "report": run.report.model_dump(),
                "trace": [item.model_dump() for item in run.trace],
                "telemetry": run.telemetry.as_dict(),
                "report_evidence_is_narrative": True,
            })
        return result

    def run_transaction(
        self,
        scenario: TransactionMismatchScenario,
        provider: Any | None = None,
    ) -> dict[str, Any]:
        route = self.router.route("TRANSACTION_MISMATCH")
        run = (
            _deterministic_transaction_run(scenario, self.memory)
            if provider is None
            else TransactionInvestigationAgent(self.memory, provider=provider).investigate(scenario)
        )
        report = run.report.model_dump()
        report["human_review_required"] = True
        state = _state_from_specialist_report(scenario, run.report, run.evidence)
        return {
            "route": route,
            "exception_id": scenario.exception_id,
            "exception": {"exception_id": scenario.exception_id, "exception_type": "TRANSACTION_MISMATCH"},
            "investigator_mode": "deterministic_demo" if provider is None else "specialist_agent",
            "known_root_cause": scenario.known_root_cause,
            "report": report,
            "trace": [item.model_dump() for item in run.trace],
            "telemetry": run.telemetry.as_dict(),
            **_control_outputs(self, state, report_evidence_is_narrative=True),
            "memory_context": _memory_context_from_evidence(state.evidence),
        }

    def run_corporate_action(
        self,
        scenario: CorporateActionScenario,
        provider: Any | None = None,
    ) -> dict[str, Any]:
        route = self.router.route("CORPORATE_ACTION")
        run = (
            _deterministic_corporate_action_run(scenario, self.memory)
            if provider is None
            else CorporateActionInvestigationAgent(self.memory, provider=provider).investigate(scenario)
        )
        report = run.report.model_dump()
        report["human_review_required"] = True
        state = _state_from_specialist_report(scenario, run.report, run.evidence)
        return {
            "route": route,
            "exception_id": scenario.exception_id,
            "exception": {"exception_id": scenario.exception_id, "exception_type": "CORPORATE_ACTION"},
            "investigator_mode": "deterministic_demo" if provider is None else "specialist_agent",
            "known_root_cause": scenario.known_root_cause,
            "report": report,
            "trace": [item.model_dump() for item in run.trace],
            "telemetry": run.telemetry.as_dict(),
            **_control_outputs(self, state, report_evidence_is_narrative=True),
            "memory_context": _memory_context_from_evidence(state.evidence),
        }

    def submit_human_review(
        self,
        investigation_result: dict[str, Any],
        human_decision: HumanDecision | str,
        reviewer_reason: str,
    ) -> HumanReviewRecord:
        """Record an explicit human decision; this method cannot edit finance data."""
        exception_id = investigation_result.get("exception_id") or (
            investigation_result.get("exception", {}).get("exception_id")
        )
        report = investigation_result.get("report", {})
        resolution = investigation_result.get("resolution", {})
        if not resolution.get("decision") or not resolution.get("rationale"):
            raise ValueError("Investigation result is missing its resolution recommendation")
        root_cause = investigation_result.get("probable_root_cause") or report.get(
            "probable_root_cause"
        )
        evidence = [
            EvidenceItem.model_validate(item)
            for item in investigation_result.get("evidence", [])
        ]
        leading_hypothesis = (investigation_result.get("hypotheses") or [None])[0]
        submission = HumanReviewSubmission(
            exception_id=exception_id,
            agent_root_cause=root_cause,
            agent_hypothesis_id=(leading_hypothesis or {}).get("hypothesis_id"),
            agent_confidence=investigation_result.get(
                "confidence", report.get("confidence", 0.0)
            ),
            agent_recommendation=(
                f"{resolution['decision']}: {resolution['rationale']}"
            ),
            supporting_evidence_references=tuple(
                item.evidence_id for item in evidence
                if (
                    _evidence_matches_target(item, "supports_hypothesis", "supports", leading_hypothesis)
                    if leading_hypothesis else bool(item.supports and item.supports == root_cause)
                )
            ),
            counter_evidence_references=tuple(
                item.evidence_id for item in evidence
                if (
                    _evidence_matches_target(item, "contradicts_hypothesis", "contradicts", leading_hypothesis)
                    if leading_hypothesis else bool(item.contradicts and item.contradicts == root_cause)
                )
            ),
            human_decision=human_decision,
            reviewer_reason=reviewer_reason,
            human_review_required=True,
        )
        record = self.review_service.submit_review(submission, evidence)
        if (
            record.human_decision is HumanDecision.ACCEPT
            and root_cause
            and root_cause not in {"UNKNOWN", "UNDETERMINED"}
            and record.agent_hypothesis_id
        ):
            exception = investigation_result.get("exception", {})
            exception_type = exception.get("exception_type") or (
                investigation_result.get("route", {}).get("exception_type")
                if isinstance(investigation_result.get("route"), dict) else None
            )
            exception_type = exception_type or "UNKNOWN_EXCEPTION"
            report = investigation_result.get("report", {})
            observations = investigation_result.get("observations") or report.get("observations", [])
            symptom_values = []
            for item in observations:
                symptom = (
                    item.get("name") or item.get("claim")
                    if isinstance(item, dict) else str(item)
                )
                if symptom:
                    symptom_values.append(str(symptom))
            symptoms = tuple(symptom_values)
            trace = investigation_result.get("trace", [])
            investigation_path: list[str] = []
            for step in trace:
                name = step.get("tool_name") if isinstance(step, dict) else None
                if name and (not investigation_path or investigation_path[-1] != name):
                    investigation_path.append(name)
            source_labels = {
                "DETERMINISTIC_ANALYTICS": "deterministic analytics",
                "PRICE_SOURCE": "price-source comparison",
                "TRANSACTION_RECORD": "transaction reconciliation",
                "CORPORATE_ACTION_RECORD": "corporate-action status",
                "HISTORICAL_CASE": "historical analogy",
                "TOOL_RESULT": "tool result",
            }
            useful_evidence = list(dict.fromkeys(
                source_labels[item.source_type.value]
                for item in evidence if item.source_type.value in source_labels
            ))
            title = f"Human-accepted {exception_type.replace('_', ' ').lower()} investigation: {root_cause}"
            case = HistoricalCase(
                case_id=self.memory.next_case_id(),
                exception_type=exception_type,
                title=title,
                symptoms=symptoms,
                root_cause=root_cause,
                resolution=record.agent_recommendation,
                evidence=tuple(evidence),
                human_validated=True,
                review_metadata={
                    "review_id": record.review_id,
                    "decision": record.human_decision.value,
                    "timestamp": record.timestamp.isoformat(),
                    "reviewer_reason": record.reviewer_reason,
                    "agent_recommendation": record.agent_recommendation,
                    "agent_confidence": record.agent_confidence,
                    "agent_hypothesis_id": record.agent_hypothesis_id,
                },
                investigation_path=tuple(investigation_path),
                useful_evidence=tuple(useful_evidence),
            )
            self.memory.add(case)
            self.accepted_cases_by_review_id[record.review_id] = case
        return record

