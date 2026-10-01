# Architecture

## Target workflow

Fund data -> Deterministic analytics -> Exception -> Router -> Specialist investigation -> Evidence challenge -> Resolution recommendation -> Human review -> Case memory

## Architectural boundary

Deterministic code owns arithmetic, thresholds and measurable financial calculations. Agents own investigation planning, tool selection, hypothesis generation and evidence synthesis.

The system should never ask an LLM to calculate NAV when deterministic code can do so.
