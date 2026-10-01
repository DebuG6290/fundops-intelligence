# Structured evidence and human review

## Evidence flow

`EvidenceItem` is the shared validated model in `src/models/evidence.py`.
It records an ID, source category and name, a claim derived from source data,
an optional `supports` or `contradicts` target, optional reliability, and
provenance metadata. Reliability remains unset when the repository has no
defensible calibration for a source.

Deterministic analytics and executed tools create current-case evidence from
their returned values. NAV evidence includes the calculated NAV variance,
security contribution output, price comparison, and fund snapshot. Transaction
and corporate-action evidence includes the record values returned by their
tools. The raw returned values are retained in metadata.

`InvestigationReport.supporting_evidence` and `counter_evidence` remain string
lists for compatibility with current providers. These are model-authored
narrative observations, not `EvidenceItem` records. The workflow does not use
them as factual evidence; a model-reported counter-claim can still trigger
conservative escalation. Specialist prose never overwrites tool provenance.
Workflow results mark those fields with `report_evidence_is_narrative: true`
for downstream consumers such as a future UI.

Historical memory results use source type `HISTORICAL_CASE`, carry the case ID,
and set `metadata.evidence_role` to `analogy`. They are not counted as primary
evidence and are not assigned support or contradiction relationships to a new
case. The challenge requires primary evidence supporting the leading
hypothesis; an analogy alone cannot pass. Tool output records support an
exception reference when they establish a record-level discrepancy.
Transaction quantity/type differences, effective corporate-action records,
and material NAV price differences can also target a matching typed
hypothesis. A relationship captures deterministic relevance; it does not turn
a historical analogy or unsupported model narrative into proof.

## Human review

`HumanReviewService` accepts an explicit `ACCEPT`, `REJECT`, or
`INVESTIGATE_FURTHER` decision and records the agent root cause, challenged
confidence and recommendation, evidence references, reviewer reason, and UTC
timestamp. Evidence references are checked against the investigation evidence
catalog and its direction. Records are immutable and history is append-only
for the lifetime of the process. The service receives no NAV, holdings, price,
transaction, or corporate-action write interface. A recommendation is never treated as an
accepted decision until a human explicitly submits one.

## Current limitations

- `InvestigationReport` represents one probable root cause. Competing
  hypotheses exist only in `InvestigationState`; the specialist report does not
  yet preserve multiple ranked hypotheses.
- Evidence provenance is only as strong as the current deterministic/tool
  outputs. Source identity is retained, but provider attestations, immutable
  source snapshots, calibrated reliability, and stable references into source
  systems are not available.
- Human review storage is in-memory and is lost on process restart.
- Accepted reviews are not written back into `CaseMemory`; no learning from
  reviews occurs yet.
- Evaluation still uses the existing small synthetic/metric harness; it does
  not measure evidence attribution quality, review outcomes, or calibration.
- Telemetry records tool-call counts and duration but does not yet emit
  structured evidence lineage or audit correlation IDs.
- The Streamlit UI has not been connected to structured evidence or review
  submission. The backend exposes the data and explicit submission method for
  that later integration.
