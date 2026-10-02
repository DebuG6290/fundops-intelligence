"""Normalize public SEC N-PORT quarterly bulk tables, without inventing trades.

The caller supplies files extracted from an official SEC N-PORT quarterly ZIP.
No download, proprietary feed, or synthetic row is hidden in this loader.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class NPortSnapshot:
    holdings: pd.DataFrame
    filings: pd.DataFrame
    source_sha256: dict[str, str]
    source_files: dict[str, str] | None = None


def load_nport_tables(directory: str | Path, *, accessions: list[str] | None = None,
                      holding_keys: set[str] | frozenset[str] | None = None) -> NPortSnapshot:
    """Read SEC TSVs and retain source accession/holding IDs for auditability.

    Reported holding ``CURRENCY_VALUE`` is a filed value, not an independently
    verified price. An implied unit value is computed only for nonzero balances.
    Missing values remain missing; derivatives and non-share units are not
    silently treated as equity positions.
    """
    root = Path(directory)
    names = ("FUND_REPORTED_INFO", "FUND_REPORTED_HOLDING")
    paths = {name: _find_table(root, name) for name in names}
    selected = set(accessions) if accessions is not None else None
    if holding_keys is not None and selected is None:
        raise ValueError("holding_keys requires selected accessions")
    raw = {"FUND_REPORTED_INFO": pd.read_csv(paths["FUND_REPORTED_INFO"], sep="\t", dtype=str)}
    holding_path = paths["FUND_REPORTED_HOLDING"]
    if selected is None:
        raw["FUND_REPORTED_HOLDING"] = pd.read_csv(holding_path, sep="\t", dtype=str)
    else:
        def selected_chunks():
            for chunk in pd.read_csv(holding_path, sep="\t", dtype=str, chunksize=100_000):
                chunk = chunk.loc[chunk.ACCESSION_NUMBER.isin(selected)]
                if holding_keys is not None:
                    keys = chunk.ACCESSION_NUMBER + ":" + chunk.HOLDING_ID
                    chunk = chunk.loc[keys.isin(holding_keys)]
                yield chunk
        chunks = selected_chunks()
        raw["FUND_REPORTED_HOLDING"] = pd.concat(chunks, ignore_index=True)
    for name, required in {
        "FUND_REPORTED_INFO": {"ACCESSION_NUMBER", "SERIES_NAME", "NET_ASSETS"},
        "FUND_REPORTED_HOLDING": {"ACCESSION_NUMBER", "HOLDING_ID", "BALANCE", "CURRENCY_VALUE", "CURRENCY_CODE"},
    }.items():
        missing = required - set(raw[name].columns)
        if missing:
            raise ValueError(f"{name} missing SEC fields: {sorted(missing)}")
    funds = raw["FUND_REPORTED_INFO"].copy()
    holdings = raw["FUND_REPORTED_HOLDING"].copy()
    if accessions is not None:
        funds = funds[funds.ACCESSION_NUMBER.isin(selected)]
        holdings = holdings[holdings.ACCESSION_NUMBER.isin(selected)]
    if funds.empty or holdings.empty:
        raise ValueError("No matching N-PORT fund and holding rows")
    if funds.ACCESSION_NUMBER.duplicated().any() or holdings.duplicated(["ACCESSION_NUMBER", "HOLDING_ID"]).any():
        raise ValueError("Duplicate N-PORT primary keys")
    if not set(holdings.ACCESSION_NUMBER).issubset(set(funds.ACCESSION_NUMBER)):
        raise ValueError("Holdings reference missing fund filings")
    if "SERIES_ID" not in funds:
        funds["SERIES_ID"] = pd.NA
    submission_path = root / "SUBMISSION.tsv"
    if submission_path.is_file():
        submission = pd.read_csv(submission_path, sep="\t", dtype=str,
                                 usecols=["ACCESSION_NUMBER", "REPORT_DATE"])
        if submission.ACCESSION_NUMBER.duplicated().any():
            raise ValueError("Duplicate SUBMISSION accession keys")
        funds = funds.merge(submission, on="ACCESSION_NUMBER", how="left", validate="one_to_one")
        paths["SUBMISSION"] = submission_path
    else:
        funds["REPORT_DATE"] = pd.NA
    funds = funds[["ACCESSION_NUMBER", "SERIES_NAME", "SERIES_ID", "REPORT_DATE", "NET_ASSETS"]].rename(columns={
        "ACCESSION_NUMBER": "accession_number", "SERIES_NAME": "fund_name",
        "SERIES_ID": "series_id", "REPORT_DATE": "report_date", "NET_ASSETS": "reported_net_assets",
    })
    funds["reported_net_assets"] = pd.to_numeric(funds.reported_net_assets, errors="coerce")
    holdings = holdings.rename(columns={
        "ACCESSION_NUMBER": "accession_number", "HOLDING_ID": "holding_id",
        "ISSUER_TITLE": "security_name", "ISSUER_CUSIP": "cusip",
        "ISSUER_NAME": "issuer_name",
        "BALANCE": "reported_balance", "UNIT": "unit", "CURRENCY_CODE": "currency",
        "CURRENCY_VALUE": "reported_value", "ASSET_CAT": "asset_category",
    })
    for optional in ("security_name", "cusip", "issuer_name", "unit", "asset_category"):
        if optional not in holdings:
            holdings[optional] = pd.NA
    holdings = holdings[["accession_number", "holding_id", "security_name", "issuer_name", "cusip", "reported_balance", "unit", "currency", "reported_value", "asset_category"]]
    holdings["filed_balance_raw"] = holdings.reported_balance.copy()
    holdings["filed_value_raw"] = holdings.reported_value.copy()
    holdings["reported_balance"] = pd.to_numeric(holdings.reported_balance, errors="coerce")
    holdings["reported_value"] = pd.to_numeric(holdings.reported_value, errors="coerce")
    holdings["implied_unit_value"] = holdings.reported_value.div(holdings.reported_balance.where(holdings.reported_balance.ne(0)))
    holdings = holdings.merge(funds[["accession_number", "series_id", "report_date"]],
                              on="accession_number", how="left", validate="many_to_one")
    digests = {name: _digest(path) for name, path in paths.items()}
    return NPortSnapshot(holdings.reset_index(drop=True), funds.reset_index(drop=True), digests,
                         {name: path.name for name, path in paths.items()})


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _find_table(directory: Path, name: str) -> Path:
    matches = [path for path in directory.iterdir() if path.is_file() and path.stem.upper() == name and path.suffix.lower() in {".tsv", ".txt"}]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {name}.tsv/.txt in {directory}")
    return matches[0]
