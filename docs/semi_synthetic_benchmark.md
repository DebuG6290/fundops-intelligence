# Semi-synthetic public-data benchmark

This is a separate research track. The existing fully synthetic benchmark and
working Streamlit demo are unchanged. The real-public-base benchmark below is
a data foundation only: no model or memory comparison was run to produce it.

## Empirical SEC 2025Q4 foundation

The benchmark uses the official [SEC 2025Q4 N-PORT bulk ZIP](https://www.sec.gov/files/dera/data/form-n-port-data-sets/2025q4_nport.zip), extracted locally. The full source profile and generated case files live outside Git; rerun the command below after extracting the SEC files. The three input SHA-256 hashes are:

| Table | SHA-256 |
| --- | --- |
| `FUND_REPORTED_INFO.tsv` | `3e44a2e58cf606c04e1646a7c76ed6c0148a32a501cd2794dedaecd592f26fa0` |
| `FUND_REPORTED_HOLDING.tsv` | `67cf1b735c657621a5a9ed7cc801a716d22164a82caf2c18a8a42d4ad9b91ef6` |
| `SUBMISSION.tsv` | `07ba04f7f143f9630f891de667d549462672d0fc359c5a7258f044492855f727` |

The complete bundle contains 13,328 fund-info filings, 12,556 distinct nonmissing series IDs, and 5,105,506 reported holdings. The broad arithmetic-usability filter (nonzero balance, positive reported value, finite numeric values, accession/holding IDs present) retains 4,929,373 holdings. There are zero duplicate accession/holding keys. Dominant asset categories are equity common (`EC`, 1,797,005), debt (`DBT`, 1,512,785), loans (`LON`, 881,010), and agency MBS (`ABS-MBS`, 449,150). USD dominates (4,193,912), but the source includes many other currencies. Filed balance median is 65,000; filed value median is 300,680.06. These describe as-filed data, not independently verified valuations.

The final benchmark recipe narrows to `EC`/`NS` holdings with nonmissing issuer name, CUSIP, currency, series and finite positive balance/value. It uses 700 September 30, 2025 series for synthetic-label historical memory and 700 October 31, 2025 series for held-out evaluation. This is 100 cases per each of seven exception families in each partition (1,400 cases total). The source has 4,810 and 2,665 eligible series in those periods respectively. One holding per series and distinct issuer CUSIP and normalized issuer name are selected deterministically from the seed. Measured partition overlap is zero for exact holdings, issuer identifiers/names, fund series, and filing periods. The 100 held-out cases per family support descriptive breakdowns while keeping the benchmark manageable; they do not deliver narrow confidence intervals or real incident prevalence.

```bash
python -m evaluation.build_nport_benchmark --nport-dir /path/to/extracted/2025q4 --quarter 2025q4 --seed 42 --per-family 100 --output-dir /local/benchmark-2025q4
```

This writes `manifest.json` with the complete source profile, filtering/split rules and overlap checks, plus separate `observable_exceptions.jsonl`, `observable_evidence.jsonl`, `hidden_ground_truth.jsonl`, and `synthetic_historical_cases.jsonl`. Keep the output files separate from the app's operational memory and do not commit the raw SEC bundle or generated bulk case files. Public evidence retains filed balance/value as raw strings, accession, holding ID, source filename/hash, currency, series and report date. Synthetic operational comparisons have a different source type. The hidden evaluator file retains the injected cause, difficulty, perturbation, expected investigation category, and relevant evidence IDs. Only the allowlisted observable contract reaches investigator adapters.

SEC N-PORT does **not** contain confirmed pricing incidents, trade breaks, corporate-action processing failures, security-mapping errors, or FX reconciliation cases. Every operational comparison, flag, and incident label here is controlled synthetic context. The filing-period and issuer split reduces some leakage but does not remove shared market regimes, manager families, or economic exposure. No case is human validated by this generator.

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

The older small-PoC path constructs 100 historical cases with synthetic labels and a
120-case held-out comparison. It is retained for compatibility, not the final empirical benchmark. No generated bulk data are committed. Each case
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
