from __future__ import annotations

import os
import json

import pandas as pd
import streamlit as st

from src.agents.openai_provider import OpenAIResponsesProvider
from src.demo.workbench import DemoWorkbench
from src.review.models import HumanDecision


def _evidence_table(items: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{
        "Evidence ID": item.get("evidence_id"),
        "Source": item.get("source_name"),
        "Source type": item.get("source_type"),
        "Claim": item.get("claim"),
        "Relationship": item.get("relationship_display"),
        "Reliability": item.get("reliability"),
        "Exception": item.get("exception_id"),
        "Provenance": json.dumps(item.get("metadata", {}), default=str),
    } for item in items])


st.set_page_config(page_title="FundOps Intelligence", page_icon="◈", layout="wide")
st.markdown(
    """
    <style>
    .block-container {max-width: 1280px; padding-top: 2rem;}
    [data-testid="stMetric"] {background: #f5f7fa; border: 1px solid #e4e8ef;
      border-radius: 10px; padding: 14px 16px;}
    .eyebrow {color:#52647a; font-size:.82rem; text-transform:uppercase;
      letter-spacing:.08em; font-weight:600;}
    </style>
    """,
    unsafe_allow_html=True,
)

if "fundops_workbench" not in st.session_state:
    st.session_state.fundops_workbench = DemoWorkbench()
if "investigation_result" not in st.session_state:
    st.session_state.investigation_result = None
if "review_records" not in st.session_state:
    st.session_state.review_records = []

workbench: DemoWorkbench = st.session_state.fundops_workbench
st.markdown('<div class="eyebrow">Operations Investigation Workbench</div>', unsafe_allow_html=True)
st.title("FundOps Intelligence")
st.caption(
    "Investigate fund-operation exceptions with deterministic analytics, sourced evidence, "
    "a challenge control, and an explicit human decision."
)

with st.container(border=True):
    left, right = st.columns([2, 1])
    with left:
        scenario_name = st.selectbox(
            "Investigation scenario",
            [workbench.NAV, workbench.TRANSACTION, workbench.CORPORATE_ACTION, workbench.INSUFFICIENT],
        )
    with right:
        live_enabled = bool(os.getenv("OPENAI_API_KEY"))
        mode = st.selectbox(
            "Investigator mode",
            ["Reproducible demo", "Live LLM specialist"],
            disabled=not live_enabled,
            help="The deterministic demo requires no network or API key.",
        )
    if not live_enabled:
        st.caption("Deterministic demo mode is available without an API key. Live specialist mode is disabled.")
    if st.button("Run investigation", type="primary", use_container_width=True):
        try:
            provider = OpenAIResponsesProvider() if mode == "Live LLM specialist" else None
            st.session_state.investigation_result = workbench.investigate(scenario_name, provider)
            st.session_state.review_records = []
        except Exception as exc:
            st.error(f"Investigation could not be completed: {exc}")

result = st.session_state.investigation_result
if result:
    exception = result.get("exception", {})
    report = result.get("report", {})
    root_cause = result.get("probable_root_cause") or report.get("probable_root_cause") or "No hypothesis"
    confidence = result.get("confidence", report.get("confidence", 0.0))
    mode_label = result.get("investigator_mode", "deterministic_baseline")

    st.divider()
    st.subheader("Exception")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Exception ID", result.get("exception_id") or exception.get("exception_id", "—"))
    metric_cols[1].metric("Type", exception.get("exception_type", "—").replace("_", " ").title())
    metric_cols[2].metric("Status", result.get("status", "—").replace("_", " ").title())
    metric_cols[3].metric("Investigator", mode_label.replace("_", " ").title())

    st.subheader("Investigation")
    summary_cols = st.columns([2, 1])
    summary_cols[0].markdown(f"**Probable root cause**\n\n`{root_cause}`")
    summary_cols[1].metric("Confidence after challenge", f"{confidence:.0%}")
    if report.get("observations"):
        for observation in report["observations"]:
            st.write(observation)
    for observation in result.get("observations", []):
        if observation.get("name") == "top_contributors":
            st.caption("Top NAV variance contributors — deterministic output")
            st.dataframe(pd.DataFrame(observation["value"]), use_container_width=True, hide_index=True)

    evidence = result.get("evidence", [])
    hypothesis_id = (result.get("hypotheses") or [{}])[0].get("hypothesis_id", "—")
    support, counter, context = [], [], []
    for item in evidence:
        relationship = "Context / historical analogy"
        if item.get("supports_hypothesis") == hypothesis_id:
            relationship = f"Supports {hypothesis_id}"
        elif item.get("contradicts_hypothesis") == hypothesis_id:
            relationship = f"Contradicts {hypothesis_id}"
        elif item.get("supports"):
            relationship = f"Legacy target: {item['supports']}"
        elif item.get("contradicts"):
            relationship = f"Legacy counter-target: {item['contradicts']}"
        row = {**item, "relationship_display": relationship}
        if relationship.startswith("Supports"):
            support.append(row)
        elif relationship.startswith("Contradicts"):
            counter.append(row)
        else:
            context.append(row)

    st.subheader("Evidence")
    if support:
        st.dataframe(_evidence_table(support), use_container_width=True, hide_index=True)
    else:
        st.info("No primary evidence is explicitly linked to the leading hypothesis.")
    with st.expander("Context and historical analogies", expanded=False):
        if context:
            st.dataframe(_evidence_table(context), use_container_width=True, hide_index=True)
        else:
            st.caption("No context-only evidence for this run.")

    st.subheader("Counter-evidence")
    if counter:
        st.dataframe(_evidence_table(counter), use_container_width=True, hide_index=True)
    else:
        st.caption("No structured counter-evidence was returned.")

    challenge = result.get("challenge", {})
    challenge_cols = st.columns(4)
    challenge_cols[0].metric("Evidence sufficient", "No" if challenge.get("insufficient_evidence") else "Yes")
    challenge_cols[1].metric("Contradiction", "Found" if challenge.get("contradiction_found") else "None")
    challenge_cols[2].metric("Ambiguity", "Found" if challenge.get("ambiguity_found") else "None")
    challenge_cols[3].metric("Control status", result.get("status", "—"))
    if result.get("status") == "ESCALATE":
        st.error(challenge.get("recommendation", "Escalation required."))
    else:
        st.success(challenge.get("recommendation", "Proceed to human review."))

    resolution = result.get("resolution", {})
    st.subheader("Resolution recommendation")
    st.write(f"**{resolution.get('decision', '—')}** — {resolution.get('rationale', '')}")
    st.warning("Human approval required. This workbench records decisions; it does not change NAV or accounting records.")

    st.subheader("Human review")
    with st.form("human_review_form", clear_on_submit=True):
        decision = st.radio(
            "Decision", [HumanDecision.ACCEPT.value, HumanDecision.REJECT.value,
                         HumanDecision.INVESTIGATE_FURTHER.value], horizontal=True,
        )
        reason = st.text_area("Reviewer comment (required)", placeholder="Explain the decision and evidence considered.")
        submitted = st.form_submit_button("Record human decision", type="primary")
    if submitted:
        if not reason.strip():
            st.error("Enter a reviewer comment before recording the decision.")
        else:
            try:
                record = workbench.submit_review(result, decision, reason.strip())
                st.session_state.review_records.append(record)
                st.success("Review recorded. The decision is separate from the agent recommendation.")
                accepted_case = workbench.accepted_case(record.review_id)
                if accepted_case:
                    st.success(f"Human-accepted case added to memory: {accepted_case.case_id}")
                elif decision == HumanDecision.ACCEPT.value:
                    st.info("The ACCEPT decision is audited, but no case was added because there was no specific hypothesis to validate.")
            except Exception as exc:
                st.error(f"Review was not recorded: {exc}")

    if st.session_state.review_records:
        st.subheader("Audit trail")
        for record in reversed(st.session_state.review_records):
            with st.expander(f"{record.human_decision.value} · {record.review_id}", expanded=True):
                st.json(record.model_dump(mode="json"))
                accepted_case = workbench.accepted_case(record.review_id)
                if accepted_case:
                    st.markdown("**Case memory write-back** · Human validated")
                    st.json(accepted_case.to_dict())

st.divider()
st.subheader("Historical case memory")
query = st.text_input("Search prior cases", value="NAV variance price vendor discrepancy")
memory_hits = workbench.search_memory(query)
if memory_hits:
    st.dataframe(pd.DataFrame([{
        "Case ID": case.case_id,
        "Exception type": case.exception_type,
        "Title": case.title,
        "Human validated": case.human_validated,
        "Root cause": case.root_cause,
    } for case in memory_hits]), use_container_width=True, hide_index=True)
else:
    st.caption("No matching historical cases.")
