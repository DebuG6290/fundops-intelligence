# Project Plan

## Phase 1 - Deterministic foundation
1. Define synthetic fund-operation entities.
2. Generate realistic holdings, prices, transactions and NAV inputs.
3. Inject controlled exceptions with known ground truth.
4. Build NAV variance and contribution analysis.
5. Add tests.

## Phase 2 - Investigation intelligence
1. Expose analytics as typed tools.
2. Add historical case memory.
3. Build first Investigation Agent.
4. Add evidence and counter-evidence.

## Phase 3 - Agentic workflow
1. Router Agent.
2. Specialist investigation agents.
3. Evidence/Challenge Agent.
4. Resolution Agent.
5. Human review workflow.

## Phase 4 - Evaluation
Compare:
- Rule-based baseline
- Single-agent baseline
- RAG + LLM
- Multi-agent + tools + evidence

Metrics:
- Root-cause accuracy
- Evidence relevance
- Recommendation accuracy
- Investigation time
- Latency
- LLM cost
