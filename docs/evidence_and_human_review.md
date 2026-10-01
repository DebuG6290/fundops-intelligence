# Structured evidence and human review

## Evidence flow

`EvidenceItem` is the shared validated model in `src/models/evidence.py`. It records an ID, exception ID, source category/name, source-derived claim, optional reliability, and provenance metadata. `supports_hypothesis` and `contradicts_hypothesis` point to stable IDs such as `HYP-001`. Legacy `supports`/`contradicts` root-cause fields remain readable during migration; new workflow challenge and review use explicit hypothesis links.

Deterministic analytics and executed tools create current-case evidence from their returned values. NAV evidence includes calculated variance, security contributions, price comparison, and fund snapshot. Transaction and corporate-action evidence includes the record values returned by their tools. Specialist narrative remains an observation, never factual evidence. Historical case results are tagged `HISTORICAL_CASE` with `evidence_role=analogy`; they cannot satisfy the primary-evidence requirement.

The workflow binds evidence from a deterministic tool result to the leading hypothesis only when the existing typed root-cause family aligns. A mismatch remains unlinked and challenge escalates. The relationship expresses traceable relevance, not certainty or truth. Reliability remains unset without calibration.

## Human review and learning gate

The service accepts `ACCEPT`, `REJECT`, or `INVESTIGATE_FURTHER` and records the investigated root cause/hypothesis, confidence, recommendation, evidence references, reviewer reason, and UTC timestamp. Each reference must exist, have the stated support/contradiction direction, link to the reviewed hypothesis/root cause, and belong to the same exception when that provenance is available.

Only an explicit human `ACCEPT` creates a `human_validated` case candidate in `CaseMemory`. The case preserves evidence and review metadata and can be retrieved through the existing keyword search. `REJECT` and `INVESTIGATE_FURTHER` create audit events only. Future retrieval still treats an accepted case as analogy, not proof. This service and memory do not mutate financial records.

## Reproducible demo

`src/demo/workbench.py` supplies deterministic synthetic NAV, transaction, corporate-action, and insufficient-evidence paths. Transaction and corporate-action demo reports derive from the actual deterministic tool results and are labeled `deterministic_demo`; live LLM mode remains optional. The Streamlit UI renders backend outputs and submits decisions through the review service.

## Current limitations and future work

- Reviews and cases are in-memory and disappear when the process restarts.
- Investigation reports still center on one probable root cause; richer ranked hypotheses are future work.
- Evidence lineage lacks immutable source snapshots, provider attestations, and cross-system IDs.
- Reliability and confidence are not calibrated; advanced attribution, calibration, agreement, latency, and cost evaluation remain future work.
- The deterministic demo investigator makes reproducible tool-derived proposals but does not measure live LLM quality.
- Production authentication, concurrency, durable persistence, and governed case versioning/retirement are intentionally deferred.
