from __future__ import annotations

import os
from typing import Any

import pandas as pd
import streamlit as st

from evaluation.benchmark import evaluate_rules_and_ml
from src.data.benchmark import generate_benchmark_dataset
from src.demo.workbench import DemoWorkbench
from src.llm.sarvam_provider import SarvamProvider
from src.review.models import HumanDecision


def _format_metric(value: Any) -> str:
    return f"{value:.1%}" if isinstance(value, (float, int)) else str(value or "Not evaluated")

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

    header = st.columns([2, 1, 1, 1])
    header[0].markdown(f"**Investigation**  \n`{exc_id}`")
    header[1].metric("Exception type", exception.get("exception_type", "—").replace("_", " ").title())
    header[2].metric("Control status", result.get("status", "—").replace("_", " ").title())
    header[3].metric("Mode", "Live Sarvam" if result.get("investigator_mode") == "specialist_agent" else "Deterministic Demo")

    overview, investigation, evidence_tab, decision_tab, audit_tab, eval_tab = st.tabs(["Overview", "Investigation", "Evidence", "Decision", "Audit & Memory", "Evaluation"])
    with overview:
        c1, c2 = st.columns([1.3, 1])
        with c1:
            st.subheader("Exception summary")
            st.write(f"**ID:** `{exc_id}`")
            st.write(f"**Type:** {exception.get('exception_type', '—').replace('_', ' ').title()}")
            st.write(f"**Current recommendation:** {resolution.get('decision', '—').replace('_', ' ').title()}")
            st.write("Synthetic demonstration data; no live fund, holdings, or accounting records are modified.")
        with c2:
            st.subheader("Control outcome")
            st.metric("Confidence after challenge", f"{confidence:.0%}", help="Investigator confidence, not a calibrated probability.")
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
                with st.expander(f"{index}. {hypothesis.get('root_cause', 'Unspecified').replace('_', ' ')} · {float(hypothesis.get('confidence', 0)):.0%}", expanded=index == 1):
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
                with st.expander(f"{item.get('source_name', 'Unknown source')} · {item.get('source_type', 'SOURCE').replace('_', ' ').title()}"):
                    st.write(item.get("claim", "No claim text"))
                    st.caption(f"Evidence ID: {item.get('evidence_id', '—')} · Exception: {item.get('exception_id', exc_id)}")
                    if item.get("reliability") is not None:
                        st.caption(f"Source reliability: {item['reliability']:.0%}")
                    if item.get("metadata"):
                        st.write("**Provenance details**")
                        st.json(item["metadata"])

    with decision_tab:
        st.subheader("Your decision")
        st.write(f"**Agent recommendation:** {resolution.get('decision', '—').replace('_', ' ').title()}")
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
                    st.success(f"{record.human_decision.value.replace('_', ' ').title()} recorded by human reviewer.")
                    accepted = workbench.accepted_case(record.review_id)
                    if accepted:
                        st.success(f"Accepted case added to validated case memory: {accepted.case_id}")
                except Exception as exc:
                    st.error("Review was not recorded.")
                    with st.expander("Technical details"):
                        st.code(f"{type(exc).__name__}: {exc}")

    with audit_tab:
        st.subheader("Investigation timeline")
        st.markdown("**System** · Deterministic analytics and evidence checks completed")
        st.markdown("**Control layer** · Hypothesis challenge and resolution recommendation completed")
        for record in st.session_state.review_records:
            st.markdown(f"**{record.timestamp.strftime('%H:%M:%S')}** · Human reviewer recorded **{record.human_decision.value}** · Review `{record.review_id}`")
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
        if st.button("Run small synthetic baseline evaluation", key="run_eval"):
            with st.spinner("Generating synthetic holdout and evaluating rules and classical ML…"):
                dataset = generate_benchmark_dataset(seed=42, fund_count=5, security_count=30, position_count=100, price_count=160, transaction_count=120, corporate_action_count=40, fx_count=50, exception_count=480, historical_case_count=100)
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
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.caption(f"Held out {metrics['evaluation_split']['test_cases']} cases. Live Sarvam baselines remain Not evaluated until run with credentials.")
            with st.expander("Evaluation methodology and per-difficulty results"):
                st.write("Features: " + ", ".join(metrics["features"]))
                st.json(metrics["by_difficulty"])
        else:
            st.info("Not evaluated yet. Run the offline baseline to create measured results; Sarvam quality is not inferred from demo behavior.")
else:
    st.info("Choose a synthetic exception and investigate to review hypotheses, evidence, challenge outcome, and a human decision.")
# Session-local review and benchmark state intentionally resets with the demo process.

