"""Reproducibly select disjoint public holdings from an N-PORT bulk quarter."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class NPortSelection:
    memory_keys: frozenset[str]
    evaluation_keys: frozenset[str]
    memory_accessions: tuple[str, ...]
    evaluation_accessions: tuple[str, ...]
    available_series_by_period: dict[str, int]


def select_disjoint_holdings(
    directory: str | Path, *, seed: int, memory_period: str,
    evaluation_period: str, per_partition: int,
) -> NPortSelection:
    """Select one EC/NS position per series, with distinct series and CUSIPs.

    Filed data are not edited. The two named periods are disjoint, and a
    stable seeded hash chooses a candidate holding within each series.
    """
    if per_partition < 1 or memory_period == evaluation_period:
        raise ValueError("Need positive size and distinct filing periods")
    root = Path(directory)
    info = pd.read_csv(root / "FUND_REPORTED_INFO.tsv", sep="\t", dtype=str,
                       usecols=["ACCESSION_NUMBER", "SERIES_ID"])
    submission = pd.read_csv(root / "SUBMISSION.tsv", sep="\t", dtype=str,
                             usecols=["ACCESSION_NUMBER", "REPORT_DATE"])
    filings = info.merge(submission, on="ACCESSION_NUMBER", validate="one_to_one")
    filings = filings.loc[filings.REPORT_DATE.isin([memory_period, evaluation_period])
                          & filings.SERIES_ID.notna()]
    accessions = filings.set_index("ACCESSION_NUMBER")[["SERIES_ID", "REPORT_DATE"]].to_dict("index")
    best: dict[tuple[str, str], tuple[int, str, str, str, str]] = {}
    columns = ["ACCESSION_NUMBER", "HOLDING_ID", "ISSUER_CUSIP", "BALANCE", "CURRENCY_VALUE",
               "ISSUER_NAME", "CURRENCY_CODE", "ASSET_CAT", "UNIT"]
    for chunk in pd.read_csv(root / "FUND_REPORTED_HOLDING.tsv", sep="\t", dtype=str,
                             usecols=columns, chunksize=100_000):
        chunk = chunk.loc[chunk.ACCESSION_NUMBER.isin(accessions)
                          & chunk.ASSET_CAT.eq("EC") & chunk.UNIT.eq("NS")
                          & chunk.ISSUER_CUSIP.notna() & chunk.ISSUER_NAME.notna()
                          & chunk.CURRENCY_CODE.notna()]
        if chunk.empty:
            continue
        balance = pd.to_numeric(chunk.BALANCE, errors="coerce")
        value = pd.to_numeric(chunk.CURRENCY_VALUE, errors="coerce")
        chunk = chunk.loc[np.isfinite(balance) & np.isfinite(value) & balance.gt(0) & value.gt(0)]
        for row in chunk.itertuples(index=False):
            filing = accessions[row.ACCESSION_NUMBER]
            slot = (filing["REPORT_DATE"], filing["SERIES_ID"])
            holding_key = f"{row.ACCESSION_NUMBER}:{row.HOLDING_ID}"
            score = int.from_bytes(hashlib.blake2b(f"{seed}:{holding_key}".encode(), digest_size=8).digest(), "big")
            candidate = (score, holding_key, str(row.ISSUER_CUSIP),
                         str(row.ISSUER_NAME).strip().casefold(), row.ACCESSION_NUMBER)
            if slot not in best or candidate < best[slot]:
                best[slot] = candidate
    available = {period: sum(key[0] == period for key in best) for period in (memory_period, evaluation_period)}
    used_series: set[str] = set()
    used_issuers: set[str] = set()
    used_issuer_names: set[str] = set()

    def choose(period: str) -> list[tuple[int, str, str, str, str]]:
        candidates = sorted((record, series) for (date, series), record in best.items() if date == period)
        chosen = []
        for record, series in candidates:
            if series in used_series or record[2] in used_issuers or record[3] in used_issuer_names:
                continue
            chosen.append(record)
            used_series.add(series)
            used_issuers.add(record[2])
            used_issuer_names.add(record[3])
            if len(chosen) == per_partition:
                break
        if len(chosen) != per_partition:
            raise ValueError(f"Only {len(chosen)} issuer/series-distinct records available for {period}")
        return chosen

    memory = choose(memory_period)
    evaluation = choose(evaluation_period)
    return NPortSelection(
        frozenset(row[1] for row in memory), frozenset(row[1] for row in evaluation),
        tuple(row[4] for row in memory), tuple(row[4] for row in evaluation), available,
    )
