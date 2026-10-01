from __future__ import annotations

import streamlit as st

from src.agents.workflow import InvestigationWorkflow
from src.data.scenarios import create_price_exception_scenario
from src.memory.cases import CaseMemory, seed_historical_cases


st.set_page_config(
    page_title="FundOps Intelligence",
    page_icon="📊",
    layout="wide",
)

st.title("FundOps Intelligence")
st.caption("Agentic AI for Fund Operations Exception Investigation")

st.info(
    "Prototype mode: deterministic investigation workflow. "
    "The LLM investigation layer will be enabled in a later milestone."
)

scenario = create_price_exception_scenario()
workflow = InvestigationWorkflow(
    CaseMemory(seed_historical_cases())
)

if st.button("Run NAV Exception Investigation", type="primary"):
    result = workflow.run_nav(scenario)

    st.subheader("1. Exception")
    st.json(result["exception"])

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("2. Investigation observations")
        for observation in result["observations"]:
            st.write(f"**{observation['name']}**")
            st.json(observation["value"])

    with col2:
        st.subheader("3. Hypotheses")
        for hypothesis in result["hypotheses"]:
            st.write(
                f"**{hypothesis['root_cause']}** "
                f"({hypothesis['confidence']:.0%})"
            )
            st.write(hypothesis["rationale"])

    st.subheader("4. Evidence challenge")
    st.json(result["challenge"])

    st.subheader("5. Resolution recommendation")
    st.json(result["resolution"])

    st.warning(
        "Human approval is required. The prototype does not modify NAV, "
        "positions, transactions, or accounting records."
    )
