# Capstone research methodology

## Research question

Under what conditions does an evidence-grounded agentic LLM architecture add
value over deterministic rules, classical ML, and a single-LLM baseline for
fund-operation exception investigation?

## Hypotheses

- H1: multi-step investigation may improve root-cause identification on
  ambiguous cases versus rules.
- H2: structured evidence grounding may improve evidence relevance.
- H3: challenge controls should improve escalation on insufficient or
  conflicting evidence.
- H4: the agentic approach may incur additional latency and token cost.

These are testable hypotheses, not reported results. The benchmark generator
separates observable feature rows from hidden evaluation labels. The current
offline runner calculates rule and Random Forest accuracy on the same
stratified held-out exception IDs. Live LLM and evidence-attribution metrics
remain Not evaluated until instrumented live experiments are run.

## Shared evaluation contract and leakage controls

`ObservableCase` is constructed from allowlisted analyst-visible feature
columns plus current evidence records, operational notes, and historical
analogies. It does not include ground truth, difficulty, injection mechanism,
secondary causes, or expected outcomes. The evaluator retains these in a
separate table and joins them only after each baseline has returned a result.
Rules and Random Forest use the same stratified held-out exception IDs. A
configured Sarvam run uses that same test split; the UI bounds live requests
to an explicitly declared maximum sample and reports failure rate and
available telemetry. A partial LLM sample must not be compared as though it
covered the full holdout.

The shared result includes predicted cause, optional ranked hypotheses and
evidence citations, recommendation/escalation when emitted, latency,
provider/model, token usage when reported, and an explicit error. Unsupported
metrics are null/Not evaluated, never zero. Cost is not estimated without a
dated, configured Sarvam price sheet.

## Difficulty design

Difficulty is evaluator-only metadata: easy, medium, hard, conflicting,
insufficient, multi-cause, or adversarial. Hard/adversarial cases damp a
causal numeric signal and inject a stronger unrelated signal; conflicting
cases expose source disagreement; insufficient cases remove current evidence;
multi-cause labels retain a secondary cause in ground truth. Generation is
seeded and on demand, with small-demo, 1k, 10k, and 100k exception presets.
No generated case files are committed.

This generator is a controlled research instrument, not a calibrated model of
real exception frequency or operational risk. Difficult-case features are
partly mechanism-scripted, current evidence relevance is coarse, and
contradiction scoring is a test of the injected benchmark rule rather than a
validated financial evidence standard.

## Limits

All generated records are synthetic. The tabular signals are intentionally
simple and currently do not model full ledger joins or every proposed noise
mechanism at record level. The current memory is keyword-based. Agent reports
can represent multiple hypotheses, but the specialist prompt/output behavior
and challenge linkage need broader empirical validation. Review/memory remain
in-memory and process-local.

The semantic-memory class is an injectable embedding boundary; no real
embedding provider is configured. The “agentic” benchmark baseline is one
tool-using Sarvam investigator plus deterministic evidence-link challenge
logic, not an LLM challenger, LLM resolution planner, or independently
orchestrated panel. Only live, recorded, bounded experiments can support
claims about LLM accuracy, cost, or latency.
