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
