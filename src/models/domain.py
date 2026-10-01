from dataclasses import dataclass
from datetime import date
from typing import Optional


@dataclass(frozen=True)
class Security:
    security_id: str
    name: str
    currency: str
    asset_class: str


@dataclass(frozen=True)
class Holding:
    fund_id: str
    security_id: str
    quantity: float


@dataclass(frozen=True)
class Price:
    security_id: str
    price_date: date
    price: float
    source: str


@dataclass(frozen=True)
class ExceptionRecord:
    exception_id: str
    fund_id: str
    exception_type: str
    expected_nav: float
    calculated_nav: float
    materiality_bps: float
    known_root_cause: Optional[str] = None
