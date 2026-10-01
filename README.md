# FundOps Intelligence

Agentic AI for Fund Operations Exception Investigation.

## Objective
A pro-code decision-support platform that investigates fund-operation exceptions by combining deterministic financial analytics, specialized AI agents, historical case memory, evidence verification, and human review.

## Design principle
**Machines calculate. Agents investigate. Evidence constrains. Humans decide. The system remembers.**

## Initial MVP
- Synthetic fund-operations environment
- Deterministic NAV discrepancy detection and decomposition
- Tool-oriented investigation layer
- Historical case memory
- Agentic investigation workflow
- Human-in-the-loop review
- Evaluation against simpler baselines

## Important boundary
This project is a research/prototype system. It is not a fund accounting system, investment advisor, trading system, or autonomous financial decision-maker. It uses synthetic data and does not contain proprietary J.P. Morgan data.

## Repository structure
- `src/analytics`: deterministic financial calculations
- `src/data`: synthetic data generation
- `src/models`: domain models
- `src/agents`: agent implementations
- `src/tools`: tools exposed to agents
- `src/memory`: historical case retrieval
- `app`: demo interface
- `evaluation`: experimental evaluation
- `tests`: automated tests
- `docs`: architecture and methodology


## Current status

The prototype currently includes:
- Synthetic fund, holdings, pricing and transaction data
- Controlled NAV, transaction and corporate-action exception scenarios
- Deterministic NAV and exception analytics
- Structured historical case memory
- Rule-based investigation baseline
- Agent investigation state and orchestration
- Evidence challenge and human-review resolution
- Streamlit demonstration interface
- Provider-agnostic LLM investigation loop with function tools
- Structured investigation reports and agent telemetry
- Reproducible evaluation harness

### First agentic loop

```
Exception
   ↓
Deterministic context
   ↓
Investigation Agent
   ↓
Tool call
   ↓
Tool result
   ↓
Further tool call / reasoning
   ↓
Structured investigation report
   ↓
Evidence challenge
   ↓
Human review
```

The LLM is not responsible for financial arithmetic. Python owns calculations and thresholds; the agent owns investigation planning and evidence synthesis.
