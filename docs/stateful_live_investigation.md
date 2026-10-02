# Stateful Live Sarvam investigation

The Streamlit **Live Sarvam** NAV route now uses `StatefulInvestigationLoop`.
The reproducible offline demo and semi-synthetic benchmark remain separate.
This sprint does not run a real provider evaluation.

Each bounded planning turn receives the current exception, current-case tool
evidence, hypotheses, prior challenge assessment, executed actions, and the
remaining turn budget. Sarvam returns a validated JSON decision: up to five
hypotheses, one next action with arguments, a concise audit rationale, and a
stopping proposal. The controller validates the selected tool, executes it
through the existing deterministic registry, converts only its outputs into
evidence, then runs the deterministic Evidence Challenger. Contradiction or
missing primary support triggers a re-plan; repeated actions cannot loop
indefinitely. The turn limit forces an escalated report when evidence remains
inadequate.

Historical memory is a selectable investigation tool. Operational retrieval
continues to require `human_validated=True`. The planner receives prior
symptoms and investigation paths as analogies but **not prior root causes**.
The prior root cause remains available for post-retrieval display/audit, and
historical evidence is excluded from current-case support and contradiction.

The final workflow runs the Evidence Challenger before the Resolution Agent.
The challenger reports supporting IDs, contradictory IDs, missing evidence,
ambiguity, and escalation. Resolution remains a recommendation requiring
human approval; only an explicit human `ACCEPT` writes a case to validated
operational memory. No tool or review service changes NAV, holdings, trades,
prices, or corporate-action records.

`timeline` contains timestamped structured events for initial observation,
hypotheses and updates, selected action, tool completion, returned evidence,
memory retrieval, challenge, re-plan, stopping decision, recommendation,
human decision, and accepted-memory write-back. Each model-supplied reason is
limited to a short field; raw responses and chain-of-thought are not stored in
the live timeline. Streamlit renders both the event sequence and hypothesis
evolution, with full event details available in expanders.

Limits: the planner can still make poor tool choices, and a source record is
not automatically sufficient to prove causality. The deterministic challenge
is conservative, not a calibrated probabilistic verifier. The loop currently
powers the Live Sarvam NAV cockpit; the legacy generic specialist API remains
for compatibility outside that cockpit. A live Sarvam quality/cost study is a
separate, explicitly authorized evaluation task.

