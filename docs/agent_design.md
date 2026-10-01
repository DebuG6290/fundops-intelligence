# Agent Design

## Why the agent exists

The deterministic layer can detect and quantify an exception, but it does not
choose an investigation path when evidence is ambiguous. The Investigation
Agent handles that reasoning layer.

## Tool boundary

The model may call:
- compare_price_sources
- search_historical_cases

The model does not perform NAV arithmetic. Financial calculations remain in
Python.

## Current implementation

Live inference uses the official `sarvamai` SDK through a provider adapter
against stable Chat Completion V1. The rest of the application consumes a
provider-neutral function-call protocol. `SARVAM_API_KEY` is required only for
live mode; deterministic demo mode has no network or credential dependency.

The agent is expected to:
1. inspect deterministic exception context;
2. select investigative tools;
3. gather evidence;
4. form hypotheses;
5. distinguish evidence from inference;
6. propose multiple ranked hypotheses when appropriate;
7. produce a recommendation for human review.

The challenge and resolution controls remain deterministic. Hypotheses are
proposals, and any structured model report is validated with Pydantic before
it enters workflow state. A supplied report can still contain only one
hypothesis; multi-hypothesis behavior is a schema capability rather than a
claim that every live response provides alternatives.

The implementation should later be extended to a complete tool loop, structured
output validation, retries, trace logging, and evaluation.
