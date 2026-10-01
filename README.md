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

The benchmark uses a shared result contract for rules, Random Forest, a single
Sarvam completion, and a tool-using Sarvam investigator with a deterministic
challenge check. Sarvam adapters run only when providers are explicitly
supplied; the UI requires an explicit quota confirmation and caps evaluations
to a small held-out sample. Metrics remain unavailable when an approach cannot
support them. No values are inferred from deterministic demo outcomes. The
research question is: *Under what conditions does an evidence-grounded
agentic LLM architecture add value over deterministic rules, classical ML,
and a single-LLM baseline for fund-operation exception investigation?*

Benchmark difficulty is evaluator-only metadata. Investigators receive an
allowlisted case view containing observable signals, current-case evidence,
operational notes, and historical analogies; labels, injection mechanism,
secondary causes, expected escalation/recommendation, and difficulty are
joined only after prediction. Presets include `small_demo`, `1k`, `10k`, and
`100k` exceptions. Hard/adversarial cases include damped causal signals and
stronger distractors. These are research fixtures, not claims about real fund
operations. Large presets are compute- and memory-intensive.

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

## Validation and limitations

The benchmark is synthetic. Evidence relevance and contradiction labels remain
coarse injection labels, not independently adjudicated financial records.
The single-LLM adapter makes one completion without tools; the agentic adapter
adds evidence/history tools and a deterministic challenge rule, not a separate
LLM challenger or resolution planner. Provider token totals are reported when
available; estimated cost remains unavailable until versioned Sarvam pricing
is configured. Semantic retrieval is an injected embedding interface only;
no embedding model is bundled. Multi-hypothesis reports support up to five
stable IDs, but evidence assessments are model-reported links checked against
observed evidence IDs, not independently verified semantic entailments.
Reviews and memory are process-local; only explicit human ACCEPT promotes a
case to analogy memory. This README does not claim model superiority.
