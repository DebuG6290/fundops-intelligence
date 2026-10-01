from __future__ import annotations

import os
from typing import Any
from datetime import datetime

import pandas as pd
import streamlit as st

from evaluation.benchmark import evaluate_rules_and_ml
from src.data.benchmark import generate_benchmark_dataset
from src.demo.workbench import DemoWorkbench
from src.llm.sarvam_provider import SarvamProvider
from src.review.models import HumanDecision


def _format_metric(value: Any) -> str:
    if value is None:
        return "Not evaluated"
    return f"{value:.1%}" if isinstance(value, (float, int)) else str(value)


def _label(value: Any) -> str:
    """Translate internal enum-like identifiers for analyst-facing labels."""
    text = str(value or "Not available")
    labels = {
        "NAV_DISCREPANCY": "NAV discrepancy",
        "TRANSACTION_MISMATCH": "Transaction mismatch",
        "CORPORATE_ACTION": "Corporate action",
        "READY_FOR_HUMAN_REVIEW": "Ready for human review",
        "ESCALATE": "Escalation required",
        "REVIEW_RECOMMENDATION": "Review recommendation",
        "INVESTIGATE_FURTHER": "Investigate further",
        "SPECIALIST_OBSERVATION": "Specialist observation",
        "DETERMINISTIC_ANALYTICS": "Deterministic analytics",
        "PRICE_SOURCE": "Pricing source",
        "HISTORICAL_CASE": "Historical analogy",
        "PRICE_EXCEPTION": "Pricing source discrepancy",
    }
    return labels.get(text, text.replace("_", " ").title())

st.set_page_config(page_title="FundOps Intelligence", page_icon="◈", layout="wide")
st.markdown("<style>.block-container{max-width:1440px;padding-top:1.2rem}[data-testid='stMetric']{background:#f5f7fa;border:1px solid #e4e8ef;border-radius:10px;padding:12px 14px}</style>", unsafe_allow_html=True)

if "fundops_workbench" not in st.session_state:
    st.session_state.fundops_workbench = DemoWorkbench()
if "investigation_result" not in st.session_state:
    st.session_state.investigation_result = None
if "review_records" not in st.session_state:
    st.session_state.review_records = []
if "benchmark_result" not in st.session_state:
    st.session_state.benchmark_result = None

workbench: DemoWorkbench = st.session_state.fundops_workbench
has_sarvam_key = bool(os.getenv("SARVAM_API_KEY"))
st.markdown("**FUND OPERATIONS · INVESTIGATION WORKBENCH**")
st.title("FundOps Intelligence")
st.caption("Synthetic decision-support prototype · Machines calculate. Agents investigate. Evidence constrains. Humans decide.")

with st.container(border=True):
    left, right, action = st.columns([2.2, 1.4, 1])
    with left:
        scenario_name = st.selectbox("Demo exception", [workbench.NAV, workbench.TRANSACTION, workbench.CORPORATE_ACTION, workbench.INSUFFICIENT])
    with right:
        modes = ["Reproducible demo"] + (["Live Sarvam"] if has_sarvam_key else [])
        mode = st.selectbox("Investigation mode", modes)
    with action:
        st.write("")
        run_clicked = st.button("Investigate", type="primary", use_container_width=True)
    if not has_sarvam_key:
        st.caption("Deterministic Demo is available offline. Configure SARVAM_API_KEY to enable Live Sarvam.")
    else:
        st.caption(f"Live Sarvam · model {os.getenv('SARVAM_MODEL', 'sarvam-105b')}")

if run_clicked:
    # Drop any previous case before a new attempt so a failed live request
    # cannot leave stale findings on screen as though they belonged to it.
    st.session_state.investigation_result = None
    st.session_state.review_records = []
    try:
        provider = SarvamProvider() if mode == "Live Sarvam" else None
        with st.spinner("Reviewing deterministic signals and operational evidence…"):
            st.session_state.investigation_result = workbench.investigate(scenario_name, provider)
            st.session_state.investigation_result["demo_run_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    except Exception as exc:
        st.error("The investigation could not be completed. Check the selected mode and provider configuration.")
        with st.expander("Technical details"):
            st.code(f"{type(exc).__name__}: {exc}")

result: dict[str, Any] | None = st.session_state.investigation_result
if result:
    exception = result.get("exception", {})
    report = result.get("report", {})
    hypotheses = result.get("hypotheses", [])
    leading = hypotheses[0] if hypotheses else {}
    root_cause = result.get("probable_root_cause") or report.get("probable_root_cause") or "No hypothesis"
    confidence = result.get("confidence", report.get("confidence", 0.0))
    challenge = result.get("challenge", {})
    resolution = result.get("resolution", {})
    exc_id = result.get("exception_id") or exception.get("exception_id", "—")
    evidence_items = result.get("evidence", [])
    supporting_count = sum(1 for item in evidence_items if item.get("supports_hypothesis") == leading.get("hypothesis_id") or item.get("supports") == root_cause)
    counter_count = sum(1 for item in evidence_items if item.get("contradicts_hypothesis") == leading.get("hypothesis_id") or item.get("contradicts") == root_cause)

    header = st.columns([2, 1, 1, 1])
    header[0].markdown(f"**Investigation**  \n`{exc_id}`")
    header[1].metric("Exception type", _label(exception.get("exception_type")))
    header[2].metric("Control status", _label(result.get("status")))
    header[3].metric("Mode", "Live Sarvam" if result.get("investigator_mode") == "specialist_agent" else "Deterministic Demo")

    overview, investigation, evidence_tab, decision_tab, audit_tab, eval_tab = st.tabs(["Overview", "Investigation", "Evidence", "Decision", "Audit & Memory", "Evaluation"])
    with overview:
        c1, c2 = st.columns([1.3, 1])
        with c1:
            st.subheader("Exception summary")
            st.write(f"**ID:** `{exc_id}`")
            st.write(f"**Type:** {_label(exception.get('exception_type'))}")
            st.write(f"**Current recommendation:** {_label(resolution.get('decision'))}")
            if exception.get("severity"):
                st.write(f"**Severity:** {_label(exception['severity'])}")
            if exception.get("financial_impact") is not None:
                st.write(f"**Financial impact:** {exception['financial_impact']:,.2f} (synthetic units)")
            st.write("Synthetic demonstration data; no live fund, holdings, or accounting records are modified.")
            if exception.get("variance_bps") is not None:
                st.write(f"**Measured NAV variance:** {exception['variance_bps']:.2f} bps")
            st.write("**What happened?** A fund-operation exception was identified and passed through deterministic analytics and evidence review.")
            st.write("**What is the system investigating?** Whether the leading hypothesis explains the observed records, and whether any primary evidence challenges it.")
            st.write(f"**Leading hypothesis:** {_label(root_cause)}" if leading else "**Leading hypothesis:** None; further investigation is required.")
            st.write(f"**Evidence:** {supporting_count} supporting · {counter_count} contradictory · {max(0, len(hypotheses) - 1)} competing hypotheses")
            st.write(f"**Next step:** {resolution.get('rationale') or report.get('recommended_next_step') or 'Obtain more evidence before deciding.'}")
        with c2:
            st.subheader("Control outcome")
            confidence_label = "High" if confidence >= .75 else "Moderate" if confidence >= .45 else "Low"
            st.metric("Investigator confidence", confidence_label, help="Qualitative band for an investigator score; it is not a calibrated probability.")
            with st.expander("Confidence score details"):
                st.write(f"Investigator score: {confidence:.2f}. This is not a calibrated probability.")
            st.metric("Evidence sufficiency", "Insufficient" if challenge.get("insufficient_evidence") else "Sufficient")
            if result.get("status") == "ESCALATE":
                st.warning(challenge.get("recommendation", "Escalation required."))
            else:
                st.success(challenge.get("recommendation", "Ready for human review."))

    with investigation:
        st.subheader("Hypotheses")
        st.caption("Confidence is an investigator score, not a calibrated probability.")
        if hypotheses:
            for index, hypothesis in enumerate(hypotheses, start=1):
                with st.expander(f"{index}. {_label(hypothesis.get('root_cause', 'Unspecified'))} · investigator score {float(hypothesis.get('confidence', 0)):.2f}", expanded=index == 1):
                    st.write(hypothesis.get("rationale") or "No rationale supplied.")
                    st.caption(f"Hypothesis ID: {hypothesis.get('hypothesis_id', 'not assigned')}")
                    required = hypothesis.get("required_evidence") or []
                    st.write("**Evidence needed:** " + (", ".join(required) if required else "Not specified"))
                    if hypothesis.get("uncertainty"):
                        st.write("**Uncertainty:** " + hypothesis["uncertainty"])
        else:
            st.info("No defensible hypothesis was produced; the case should be investigated further.")
        if report.get("observations"):
            st.subheader("Investigator observations")
            for observation in report["observations"]:
                st.write(f"• {observation}")
        for observation in result.get("observations", []):
            if observation.get("name") == "top_contributors":
                st.subheader("Deterministic NAV contributors")
                st.dataframe(pd.DataFrame(observation["value"]), use_container_width=True, hide_index=True)

    with evidence_tab:
        groups: dict[str, list[dict[str, Any]]] = {"Supporting current hypothesis": [], "Contradicting current hypothesis": [], "Context and historical analogy": []}
        for item in result.get("evidence", []):
            if item.get("supports_hypothesis") == leading.get("hypothesis_id") or item.get("supports") == root_cause:
                groups["Supporting current hypothesis"].append(item)
            elif item.get("contradicts_hypothesis") == leading.get("hypothesis_id") or item.get("contradicts") == root_cause:
                groups["Contradicting current hypothesis"].append(item)
            else:
                groups["Context and historical analogy"].append(item)
        for title, items in groups.items():
            st.subheader(title)
            if not items:
                st.caption("No items in this category.")
            for item in items:
                with st.expander(f"{item.get('source_name', 'Unknown source')} · {_label(item.get('source_type'))}"):
                    st.write(item.get("claim", "No claim text"))
                    st.code(item.get("evidence_id", "ID unavailable"), language=None)
                    st.caption(f"Exception: {item.get('exception_id', exc_id)} · Relationship: {_label(item.get('supports_hypothesis') and 'SUPPORTS' or item.get('contradicts_hypothesis') and 'CONTRADICTS' or 'CONTEXT')}")
                    if item.get("reliability") is not None:
                        st.caption(f"Source reliability: {item['reliability']:.0%}")
                    if st.checkbox("View provenance", key=f"provenance_{item.get('evidence_id', 'unknown')}"):
                        st.write(f"Source type: {_label(item.get('source_type'))}")
                        st.json(item.get("metadata") or {})

    with decision_tab:
        st.subheader("Your decision")
        st.write(f"**Agent recommendation:** {_label(resolution.get('decision'))}")
        st.write(resolution.get("rationale", "No rationale supplied."))
        st.warning("Human review is mandatory. This records an audit decision only; it never changes financial records.")
        st.caption("ACCEPT validates the proposed case and adds it to memory. REJECT records non-acceptance. INVESTIGATE FURTHER requests more evidence or escalation.")
        with st.form("human_review_form", clear_on_submit=True):
            decision = st.radio("Decision", [HumanDecision.ACCEPT.value, HumanDecision.REJECT.value, HumanDecision.INVESTIGATE_FURTHER.value], horizontal=True)
            reason = st.text_area("Reviewer comment (required)", placeholder="Explain the decision and evidence considered.")
            submitted = st.form_submit_button("Record human decision", type="primary")
        if submitted:
            if not reason.strip():
                st.error("Enter a reviewer comment before recording the decision.")
            else:
                try:
                    record = workbench.submit_review(result, decision, reason.strip())
                    st.session_state.review_records.append(record)
                    accepted = workbench.accepted_case(record.review_id)
                    if accepted:
                        st.success(f"Accepted case added to validated case memory: {accepted.case_id}")
                    elif record.human_decision is HumanDecision.REJECT:
                        st.success("Decision recorded. The case was not promoted to validated memory.")
                    else:
                        st.success("Decision recorded. Additional investigation is required; the case was not promoted to validated memory.")
                except Exception as exc:
                    st.error("Review was not recorded.")
                    with st.expander("Technical details"):
                        st.code(f"{type(exc).__name__}: {exc}")
        if st.session_state.review_records:
            latest_record = st.session_state.review_records[-1]
            if latest_record.exception_id == exc_id:
                accepted = workbench.accepted_case(latest_record.review_id)
                if latest_record.human_decision is HumanDecision.ACCEPT and accepted:
                    st.success(f"Human accepted this investigation. {accepted.case_id} is now in validated case memory.")
                elif latest_record.human_decision is HumanDecision.REJECT:
                    st.info("The human reviewer rejected the recommendation. No validated-memory write-back occurred.")
                else:
                    st.warning("The human reviewer requested further investigation. The case is not considered validated.")

    with audit_tab:
        accepted_cases = [workbench.accepted_case(record.review_id) for record in st.session_state.review_records]
        accepted_cases = [case for case in accepted_cases if case is not None]
        if accepted_cases:
            st.subheader("Recently validated")
            for case in reversed(accepted_cases):
                st.success(f"✓ {case.case_id} · {_label(case.exception_type)} · Human validated")
                with st.expander(f"Review the accepted case {case.case_id}"):
                    st.write(case.title)
                    st.write(f"Confirmed root cause: {_label(case.root_cause)}")
                    st.write(case.resolution)
                    st.caption(f"Added {case.review_metadata.get('timestamp', 'timestamp unavailable')}")
        st.subheader("Investigation timeline")
        if result.get("demo_run_at"):
            st.caption(f"Investigation completed · {result['demo_run_at']}")
        st.markdown("**System** · Deterministic analytics and evidence checks completed")
        st.markdown("**Control layer** · Hypothesis challenge and resolution recommendation completed")
        for record in st.session_state.review_records:
            st.markdown(f"**{record.timestamp.strftime('%H:%M:%S')}** · Human reviewer recorded **{_label(record.human_decision.value)}** · Review `{record.review_id}`")
            st.write(record.reviewer_reason)
            accepted = workbench.accepted_case(record.review_id)
            if accepted:
                st.caption(f"Validated memory write-back · {accepted.case_id} · explicitly human accepted")
        st.subheader("Historical memory search")
        query = st.text_input("Search prior cases", value="NAV variance price vendor discrepancy", key="memory_query")
        cases = workbench.search_memory(query)
        if cases:
            for case in cases:
                with st.expander(f"{case.case_id} · Historical analogy · {case.title}"):
                    st.write(f"Suggested prior cause: {case.root_cause}")
                    st.write(case.resolution)
        else:
            st.caption("No matching historical analogies.")

    with eval_tab:
        st.subheader("Offline baseline comparison")
        st.caption("Metrics are calculated on seeded synthetic holdout data, not claims about LLM quality.")
        live_eval = st.checkbox("Also run real Sarvam baselines (uses API quota)", disabled=not has_sarvam_key)
        llm_case_limit = st.slider("Maximum held-out cases per Sarvam approach", 1, 20, 5, disabled=not live_eval)
        confirm_live = st.checkbox("I understand this sends synthetic cases to Sarvam", disabled=not live_eval)
        if st.button("Run shared synthetic benchmark", key="run_eval"):
            if live_eval and not confirm_live:
                st.error("Confirm the live Sarvam evaluation before running it. No API calls were made.")
            else:
                with st.spinner("Generating one seeded holdout and evaluating the selected approaches…"):
                    dataset = generate_benchmark_dataset(seed=42, fund_count=5, security_count=30, position_count=100, price_count=160, transaction_count=120, corporate_action_count=40, fx_count=50, exception_count=480, historical_case_count=100)
                    if live_eval and confirm_live:
                        provider = SarvamProvider()
                        st.session_state.benchmark_result = evaluate_rules_and_ml(
                            dataset, seed=42, single_llm_provider=provider,
                            agentic_llm_provider=provider, max_llm_cases=llm_case_limit,
                        )
                    else:
                        st.session_state.benchmark_result = evaluate_rules_and_ml(dataset, seed=42)
        metrics = st.session_state.benchmark_result
        if metrics:
            rows = []
            for name, values in metrics["models"].items():
                rows.append({
                    "Approach": name.replace("_", " ").title(),
                    "Root-cause accuracy": _format_metric(values.get("root_cause_accuracy")) if isinstance(values, dict) else values,
                    "Top-3 recall": _format_metric(values.get("top3_hypothesis_recall")) if isinstance(values, dict) else values,
                    "Escalation accuracy": _format_metric(values.get("escalation_accuracy")) if isinstance(values, dict) else values,
                    "Recommendation accuracy": _format_metric(values.get("recommendation_accuracy")) if isinstance(values, dict) else "Not evaluated",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.caption(f"Stratified holdout: {metrics['evaluation_split']['test_cases']} cases; all approaches scored on the same {metrics['evaluation_split']['scored_cases']} cases. Unmeasured fields remain Not evaluated.")
            with st.expander("Evaluation methodology and per-difficulty results"):
                st.write("Features: " + ", ".join(metrics["features"]))
                st.json(metrics["by_difficulty"])
                detail_rows = []
                for name, values in metrics["models"].items():
                    if isinstance(values, dict):
                        detail_rows.append({
                            "Approach": name,
                            "Evidence precision": _format_metric(values.get("evidence_precision")),
                            "Evidence recall": _format_metric(values.get("evidence_recall")),
                            "Contradiction detection": _format_metric(values.get("contradiction_detection")),
                            "Mean latency (s)": f"{values['mean_latency_seconds']:.3f}" if values.get("mean_latency_seconds") is not None else "Not evaluated",
                            "Input tokens": values.get("input_tokens") if values.get("input_tokens") is not None else "Not evaluated",
                            "Output tokens": values.get("output_tokens") if values.get("output_tokens") is not None else "Not evaluated",
                            "Failure rate": _format_metric(values.get("failure_rate")),
                        })
                if detail_rows:
                    st.dataframe(pd.DataFrame(detail_rows), use_container_width=True, hide_index=True)
                if st.checkbox("Show provider request IDs and failure details", key="show_eval_provider_details"):
                    for name, values in metrics["models"].items():
                        if not isinstance(values, dict):
                            continue
                        st.write(f"**{name}**")
                        if values.get("request_ids"):
                            for request_id in values["request_ids"]:
                                st.code(request_id, language=None)
                        if values.get("failure_details"):
                            for failure in values["failure_details"]:
                                st.error(failure["error"])
                        elif values.get("failures", 0) == 0:
                            st.caption("No provider failures recorded.")
        else:
            st.info("Not evaluated yet. Run the offline baseline to create measured results; Sarvam quality is not inferred from demo behavior.")
else:
    st.info("Choose a synthetic exception and investigate to review hypotheses, evidence, challenge outcome, and a human decision.")
# Session-local review and benchmark state intentionally resets with the demo process.

