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

## Limits

All generated records are synthetic. The tabular signals are intentionally
simple and currently do not model full ledger joins or every proposed noise
mechanism at record level. The current memory is keyword-based. Agent reports
can represent multiple hypotheses, but the specialist prompt/output behavior
and challenge linkage need broader empirical validation. Review/memory remain
in-memory and process-local.
