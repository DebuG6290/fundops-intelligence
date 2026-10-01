# Architecture

## Target workflow

Fund data -> Deterministic analytics -> Exception -> Router -> Specialist investigation -> Structured evidence -> Evidence challenge -> Resolution recommendation -> Explicit human decision -> Audit history

## Architectural boundary

Deterministic code owns arithmetic, thresholds and measurable financial calculations. Agents own investigation planning, tool selection, hypothesis generation and evidence synthesis.

The system should never ask an LLM to calculate NAV when deterministic code can do so.

Structured evidence is created from deterministic analytics and executed tool
results. Specialist report prose remains an observation and is not promoted to
financial evidence. Historical cases are labeled as analogies. Human decisions
are recorded in an append-only in-memory review service; they do not mutate
financial records or automatically update case memory.
