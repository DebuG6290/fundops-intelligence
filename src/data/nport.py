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


def load_nport_tables(directory: str | Path, *, accessions: list[str] | None = None) -> NPortSnapshot:
    """Read SEC TSVs and retain source accession/holding IDs for auditability.

    Reported holding ``CURRENCY_VALUE`` is a filed value, not an independently
    verified price. An implied unit value is computed only for nonzero balances.
    Missing values remain missing; derivatives and non-share units are not
    silently treated as equity positions.
    """
    root = Path(directory)
    names = ("FUND_REPORTED_INFO", "FUND_REPORTED_HOLDING")
    paths = {name: _find_table(root, name) for name in names}
    raw = {name: pd.read_csv(path, sep="\t", dtype=str, low_memory=False) for name, path in paths.items()}
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
        selected = set(accessions)
        funds = funds[funds.ACCESSION_NUMBER.isin(selected)]
        holdings = holdings[holdings.ACCESSION_NUMBER.isin(selected)]
    if funds.empty or holdings.empty:
        raise ValueError("No matching N-PORT fund and holding rows")
    if funds.ACCESSION_NUMBER.duplicated().any() or holdings.duplicated(["ACCESSION_NUMBER", "HOLDING_ID"]).any():
        raise ValueError("Duplicate N-PORT primary keys")
    if not set(holdings.ACCESSION_NUMBER).issubset(set(funds.ACCESSION_NUMBER)):
        raise ValueError("Holdings reference missing fund filings")
    funds = funds[["ACCESSION_NUMBER", "SERIES_NAME", "NET_ASSETS"]].rename(columns={
        "ACCESSION_NUMBER": "accession_number", "SERIES_NAME": "fund_name", "NET_ASSETS": "reported_net_assets",
    })
    funds["reported_net_assets"] = pd.to_numeric(funds.reported_net_assets, errors="coerce")
    holdings = holdings.rename(columns={
        "ACCESSION_NUMBER": "accession_number", "HOLDING_ID": "holding_id",
        "ISSUER_TITLE": "security_name", "ISSUER_CUSIP": "cusip",
        "BALANCE": "reported_balance", "UNIT": "unit", "CURRENCY_CODE": "currency",
        "CURRENCY_VALUE": "reported_value", "ASSET_CAT": "asset_category",
    })
    for optional in ("security_name", "cusip", "unit", "asset_category"):
        if optional not in holdings:
            holdings[optional] = pd.NA
    holdings = holdings[["accession_number", "holding_id", "security_name", "cusip", "reported_balance", "unit", "currency", "reported_value", "asset_category"]]
    holdings["reported_balance"] = pd.to_numeric(holdings.reported_balance, errors="coerce")
    holdings["reported_value"] = pd.to_numeric(holdings.reported_value, errors="coerce")
    holdings["implied_unit_value"] = holdings.reported_value.div(holdings.reported_balance.where(holdings.reported_balance.ne(0)))
    digests = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}
    return NPortSnapshot(holdings.reset_index(drop=True), funds.reset_index(drop=True), digests)


def _find_table(directory: Path, name: str) -> Path:
    matches = [path for path in directory.iterdir() if path.is_file() and path.stem.upper() == name and path.suffix.lower() in {".tsv", ".txt"}]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {name}.tsv/.txt in {directory}")
    return matches[0]

