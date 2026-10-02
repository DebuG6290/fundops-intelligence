# Semi-synthetic public-data benchmark

This is a separate research track. The existing fully synthetic benchmark and
working Streamlit demo are unchanged.

## Inputs and provenance

Download an official quarterly N-PORT data set from the [SEC's Form N-PORT
data sets](https://www.sec.gov/data-research/sec-markets-data/form-n-port-data-sets),
extract it locally, and pass the directory containing `FUND_REPORTED_INFO.tsv`
and `FUND_REPORTED_HOLDING.tsv` (some releases use `.txt`). The loader retains
accession and holding IDs, filed balance/value/currency, and SHA-256 hashes of
both source files. It validates primary and foreign keys. Filed values are
as-reported data, **not** independently verified trade prices; implied unit
value is a derived ratio and is not used as external price truth. The SEC
warns that the data are as-filed and may contain inaccuracies. The dataset
must be reviewed for scope and amendments before research publication.

```bash
python -m evaluation.run_semi_synthetic --nport-dir /path/to/extracted/quarter --seed 42
```

The default constructs 100 historical cases with synthetic labels and a
120-case held-out comparison. No generated bulk data are committed. Each case
is anchored to a filed holding, while all exception signals, transaction or
corporate-action indicators, and root-cause labels are **controlled synthetic
injections**. N-PORT cannot establish internal transaction or corporate-action
exceptions. Neither the labels nor the injection mechanism are passed to
the investigator contract.

## Memory experiment

Source holding keys are split before case creation. The memory and evaluation
partitions therefore cannot share an accession/holding key. Holdings may be
sampled repeatedly *within* a partition; this small-PoC design limits
independence. The corpus includes pricing (25), quantity/transaction (25),
corporate action (20), mapping (10), FX (10), insufficient-evidence (5),
and conflicting-evidence (5)
cases. They are tagged `synthetic_ground_truth`, are **not** human validated,
and are never loaded into operational guidance memory.

The paired comparison scores the same holdout twice: transparent rules alone
and the same rules with keyword-retrieved historical analogies prioritizing
checks. Retrieval filters exception type before ranking and uses only
observable symptoms and useful-evidence terms. Historical root cause is
available after retrieval to suggest the next check, but it is never used in
relevance scoring. Cases are analogies, not current-case evidence.

Reported values are measured root-cause/first-check accuracy, check-order
coverage and position, retrieval rate, incorrect-analogy-first rate, and
latency. Check steps are a **ranking proxy**, not executed tool calls. Tokens,
cost, live Sarvam and human outcomes remain unavailable and are shown as
`null`/`Not evaluated`. This experiment does not establish that memory helps;
the measured OFF/ON difference may be zero or negative.

## Important limitations

- A small filing selection may overrepresent one fund or period; source
  holdings are public but the incident labels are not real operations cases.
- The rule baseline and memory-augmented rule are only a controlled first
  comparison. A live Sarvam OFF/ON trial requires provider calls and a fixed
  budget, separate prompt logs, token accounting, and repeated runs.
- Signal perturbations are not calibrated against a real incident population.
  The public filing does not validate the injected root cause.
- Held-out holding IDs prevent exact source-key overlap, not issuer, fund,
  regime, or temporal leakage. Stronger fund/date splits are needed for claims
  about generalization.
- No synthetic-label case is promoted to validated operational memory. Only
  the existing human-review workflow can make a case human validated.

