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

The first implementation uses the OpenAI Responses API through a provider
adapter. The provider is intentionally isolated so another model can be
benchmarked later.

The agent is expected to:
1. inspect deterministic exception context;
2. select investigative tools;
3. gather evidence;
4. form hypotheses;
5. distinguish evidence from inference;
6. produce a recommendation for human review.

The implementation should later be extended to a complete tool loop, structured
output validation, retries, trace logging, and evaluation.
