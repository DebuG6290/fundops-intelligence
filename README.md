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

## Run the demo workbench

Install `requirements.txt`, then run:

```bash
streamlit run app/streamlit_app.py
```

The reproducible deterministic demo is the default and needs no LLM key. It
covers NAV, transaction mismatch, corporate action, and an insufficient
evidence escalation path. Optional Live Sarvam mode uses the stable Sarvam
Chat Completion V1 SDK adapter. Configure `SARVAM_API_KEY`; the default model
is `sarvam-105b`. Human review and case memory are process-local and reset
when Streamlit restarts. See [the demo flow](docs/demo_flow.md).

## Configure optional Live Sarvam mode

Copy `.env.example` to `.env` and add your own key locally. Never commit the
key. Without it, choose **Reproducible demo**; the app remains fully usable
offline. `SARVAM_TIMEOUT_SECONDS` and `SARVAM_MODEL` can be overridden.

## Synthetic benchmark and research comparison

`src.data.benchmark.generate_benchmark_dataset` builds deterministic
fund/security/position/price/transaction/corporate-action/FX records and
injected exceptions on demand. It can generate the documented scale (100
funds, 5,000 securities, 200,000 positions, 500,000 prices, 300,000
transactions, 20,000 corporate actions, 50,000 FX rows, 10,000 exceptions,
and 5,000 historical analogies) without storing a generated blob in Git.
Observable exception features are separate from the evaluation-only
`ground_truth` table. See the Evaluation tab or run
`python -m evaluation.run_benchmark` after installing dependencies.

The offline benchmark measures a transparent rule heuristic and a
Random-Forest tabular baseline on the same stratified holdout. Single-LLM and
agentic-LLM quality remain **Not evaluated** until actual Sarvam runs are
performed; no values are inferred from deterministic demo outcomes. The
research question is: *Under what conditions does an evidence-grounded
agentic LLM architecture add value over deterministic rules, classical ML,
and a single-LLM baseline for fund-operation exception investigation?*

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
- Sarvam Chat Completion V1 provider, optional live mode and usage telemetry
- Multi-hypothesis structured report extension
- Scalable on-demand synthetic benchmark and hidden evaluation labels
- Shared-holdout rule and classical ML baseline comparison
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
