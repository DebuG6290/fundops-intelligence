"""Streamlit-facing projection of structured investigation events."""

from __future__ import annotations

from typing import Any, Callable


EVENT_NAMES = {
    "initial_observation": "Initial observation", "memory_retrieval": "Memory retrieval",
    "hypotheses": "Initial hypotheses", "hypothesis_update": "Hypotheses updated",
    "action_selected": "Next action selected", "tool_execution": "Tool executed",
    "evidence_returned": "Evidence returned", "challenge_assessment": "Evidence challenged",
    "replan": "Investigation re-planned", "stopping_decision": "Stopping decision",
    "recommendation": "Recommendation for human review", "human_decision": "Human decision",
    "memory_write_back": "Validated memory write-back",
}


def render_investigation_timeline(ui: Any, events: list[dict[str, Any]], label: Callable[[Any], str]) -> None:
    """Show audit events and full details, but never a raw model transcript."""
    for index, event in enumerate(events, start=1):
        details = event.get("details") or {}
        timestamp = str(event.get("timestamp", ""))[11:19]
        stage = event.get("stage", "event")
        ui.markdown(f"**{timestamp} · {EVENT_NAMES.get(stage, label(stage))}**")
        ui.write(event.get("rationale") or "Event recorded.")
        if stage in {"hypotheses", "hypothesis_update"}:
            for hypothesis in details.get("hypotheses", []):
                ui.caption(
                    f"{hypothesis.get('hypothesis_id', 'Hypothesis')} · "
                    f"{label(hypothesis.get('root_cause'))} · "
                    f"investigator score {float(hypothesis.get('confidence', 0)):.2f}"
                )
                if hypothesis.get("uncertainty"):
                    ui.write(f"Uncertainty: {hypothesis['uncertainty']}")
        elif stage == "action_selected":
            ui.caption(f"Selected tool: {details.get('tool_name', 'Not available')}")
        elif stage == "evidence_returned":
            ui.caption("Evidence IDs: " + (", ".join(details.get("evidence_ids", [])) or "None"))
        elif stage == "challenge_assessment":
            ui.caption("Supporting: " + (", ".join(details.get("supporting_evidence_ids", [])) or "None"))
            ui.caption("Contradicting: " + (", ".join(details.get("contradictory_evidence_ids", [])) or "None"))
            ui.caption("Missing: " + (", ".join(details.get("missing_evidence", [])) or "None identified"))
        with ui.expander(f"Full event details · {index}"):
            ui.json(details)

