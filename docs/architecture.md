# Architecture

## End-to-end workflow

```text
Synthetic fund data
  -> deterministic financial analytics
  -> exception and router
  -> specialist investigation
  -> structured evidence linked to hypothesis IDs
  -> evidence challenge
  -> resolution recommendation
  -> explicit human decision and audit record
  -> accepted case memory
  -> future retrieval as an analogy
```

Deterministic code owns financial arithmetic, thresholds, and measurable calculations. Specialists plan investigations, call tools, and propose hypotheses. Evidence is created from deterministic analytics and actual tool outputs; specialist prose is not upgraded to fact. The challenge requires primary evidence linked to the leading hypothesis and escalates for missing, unrelated, conflicting, or ambiguous evidence.

`EvidenceItem.exception_id` preserves case provenance. `supports_hypothesis` and `contradicts_hypothesis` use stable per-investigation identifiers such as `HYP-001`; root-cause strings are retained only as a compatibility representation for older records and are not the primary workflow relationship.

Human review records `ACCEPT`, `REJECT`, or `INVESTIGATE_FURTHER` in an append-only in-memory service. Only explicit `ACCEPT` writes a `human_validated` case candidate to memory. All decisions remain separate from the agent recommendation, and neither review nor memory has an interface to change NAV, holdings, transactions, prices, or corporate actions.

The Streamlit workbench is reproducible by default. NAV uses the deterministic baseline; transaction and corporate-action demo modes use actual deterministic tool results and are labeled as demo investigations. Live LLM specialist mode is optional when configured. No demo mode requires an LLM or network.

## Future Production Extensions

- Persistent review and case storage (for example, PostgreSQL)
- Vector or hybrid semantic retrieval beyond transparent keyword search
- Multiple ranked hypotheses in specialist reports
- Immutable source snapshots, provider attestations, and correlation IDs
- Authentication, authorization, persistence, and concurrency controls
- Advanced evidence attribution, calibration, review-agreement, latency, and cost evaluation
- Governed case lifecycle: versioning, retirement, and confidence decay

This is a capstone prototype. Reviews and cases are process-local, accepted cases are analogy candidates rather than independently verified ground truth, and the demo investigator is not a substitute for evaluating live LLM quality.
