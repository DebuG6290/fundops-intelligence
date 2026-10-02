"""Profile an extracted SEC N-PORT quarterly bundle without loading it all at once."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


IMPORTANT_FIELDS = (
    "ACCESSION_NUMBER", "HOLDING_ID", "ISSUER_NAME", "ISSUER_CUSIP",
    "BALANCE", "UNIT", "CURRENCY_CODE", "CURRENCY_VALUE", "ASSET_CAT",
)


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _distribution(values: list[np.ndarray]) -> dict[str, float | int | None]:
    if not values:
        return {key: None for key in ("min", "p25", "median", "p75", "p95", "max")}
    data = np.concatenate(values)
    return {
        "count": int(len(data)), "min": float(data.min()),
        "p25": float(np.quantile(data, .25)), "median": float(np.median(data)),
        "p75": float(np.quantile(data, .75)), "p95": float(np.quantile(data, .95)),
        "max": float(data.max()),
    }


def profile_nport(directory: str | Path, *, quarter: str) -> dict:
    """Return source-wide counts, quality checks, and exact numeric quantiles."""
    root = Path(directory)
    paths = {name: root / f"{name}.tsv" for name in (
        "FUND_REPORTED_INFO", "FUND_REPORTED_HOLDING", "SUBMISSION",
    )}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    info = pd.read_csv(paths["FUND_REPORTED_INFO"], sep="\t", dtype=str)
    submission = pd.read_csv(paths["SUBMISSION"], sep="\t", dtype=str)
    required_info = {"ACCESSION_NUMBER", "SERIES_ID", "SERIES_NAME"}
    required_submission = {"ACCESSION_NUMBER", "REPORT_DATE"}
    for table, required, name in ((info, required_info, "FUND_REPORTED_INFO"),
                                  (submission, required_submission, "SUBMISSION")):
        missing = required - set(table)
        if missing:
            raise ValueError(f"{name} missing {sorted(missing)}")
    missingness: Counter = Counter()
    assets: Counter = Counter()
    currencies: Counter = Counter()
    units: Counter = Counter()
    seen_keys: set[str] = set()
    holding_accessions: set[str] = set()
    balances: list[np.ndarray] = []
    values: list[np.ndarray] = []
    total = usable = duplicates = 0
    for chunk in pd.read_csv(paths["FUND_REPORTED_HOLDING"], sep="\t", dtype=str,
                             usecols=list(IMPORTANT_FIELDS), chunksize=100_000):
        total += len(chunk)
        missingness.update({field: int(chunk[field].isna().sum() + chunk[field].eq("").sum())
                            for field in IMPORTANT_FIELDS})
        assets.update(chunk.ASSET_CAT.fillna("<missing>").value_counts().to_dict())
        currencies.update(chunk.CURRENCY_CODE.fillna("<missing>").value_counts().to_dict())
        units.update(chunk.UNIT.fillna("<missing>").value_counts().to_dict())
        holding_accessions.update(chunk.ACCESSION_NUMBER.dropna())
        keys = chunk.ACCESSION_NUMBER.fillna("<missing>") + ":" + chunk.HOLDING_ID.fillna("<missing>")
        for key in keys:
            if key in seen_keys:
                duplicates += 1
            else:
                seen_keys.add(key)
        balance = pd.to_numeric(chunk.BALANCE, errors="coerce")
        value = pd.to_numeric(chunk.CURRENCY_VALUE, errors="coerce")
        balances.append(balance[np.isfinite(balance)].to_numpy(dtype="float64"))
        values.append(value[np.isfinite(value)].to_numpy(dtype="float64"))
        usable += int((balance.notna() & value.notna() & np.isfinite(balance) & np.isfinite(value)
                       & balance.ne(0) & value.gt(0) & chunk.ACCESSION_NUMBER.notna()
                       & chunk.HOLDING_ID.notna()).sum())
    info_accessions = set(info.ACCESSION_NUMBER.dropna())
    submission_accessions = set(submission.ACCESSION_NUMBER.dropna())
    return {
        "source": "SEC Form N-PORT quarterly bulk data (as-filed)",
        "quarter": quarter,
        "source_url": f"https://www.sec.gov/files/dera/data/form-n-port-data-sets/{quarter.lower()}_nport.zip",
        "files": {name: {"filename": path.name, "bytes": path.stat().st_size,
                         "sha256": _digest(path)} for name, path in paths.items()},
        "filings": {"fund_info_rows": int(len(info)), "unique_accessions": int(info.ACCESSION_NUMBER.nunique()),
                    "duplicate_accessions": int(info.ACCESSION_NUMBER.duplicated().sum()),
                    "duplicate_submission_accessions": int(submission.ACCESSION_NUMBER.duplicated().sum()),
                    "holdings_without_info": int(len(holding_accessions - info_accessions)),
                    "info_without_submission": int(len(info_accessions - submission_accessions))},
        "funds": {"unique_series_ids": int(info.SERIES_ID.nunique()),
                  "missing_series_ids": int(info.SERIES_ID.isna().sum()),
                  "unique_series_names": int(info.SERIES_NAME.nunique())},
        "periods": {str(k): int(v) for k, v in submission.REPORT_DATE.fillna("<missing>").value_counts().items()},
        "holdings": {"total": total, "usable_nonzero_balance_positive_value": usable,
                     "duplicate_accession_holding_keys": duplicates,
                     "distinct_accessions": len(holding_accessions),
                     "asset_categories": dict(assets.most_common()),
                     "currencies": dict(currencies.most_common()),
                     "units": dict(units.most_common()),
                     "missing_fields": dict(missingness),
                     "reported_balance": _distribution(balances),
                     "reported_value": _distribution(values)},
        "field_origin": {"public": list(IMPORTANT_FIELDS) + ["SERIES_ID", "SERIES_NAME", "REPORT_DATE"],
                         "synthetic": ["operational comparison balances and values", "exception signals",
                                       "injected incident labels", "difficulty", "ground truth"]},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nport-dir", required=True, type=Path)
    parser.add_argument("--quarter", required=True)
    args = parser.parse_args()
    print(json.dumps(profile_nport(args.nport_dir, quarter=args.quarter), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
