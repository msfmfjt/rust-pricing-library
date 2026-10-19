"""Explicit, immutable inputs for the daily European-option replay slice."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from datetime import date, datetime
from hashlib import sha256
import json
import math
from types import MappingProxyType
from typing import Mapping, Protocol


class BacktestError(ValueError):
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


def require(condition: bool, code: str, message: str) -> None:
    if not condition:
        raise BacktestError(code, message)


def finite(value: float, name: str, minimum: float | None = None) -> None:
    require(math.isfinite(value), "invalid_number", name)
    if minimum is not None:
        require(value >= minimum, "invalid_number", name)


def aware(value: datetime) -> None:
    require(isinstance(value, datetime) and value.utcoffset() is not None,
            "invalid_timestamp", "timestamps must include a timezone")


def checked_sum(values, code: str, context: str) -> float:
    try:
        total = math.fsum(values)
    except (OverflowError, ValueError) as error:
        raise BacktestError(code, context) from error
    require(math.isfinite(total), code, context)
    return total


def data(value):
    """Canonical JSON-compatible representation; no NaN/Infinity or implicit arrays."""
    if is_dataclass(value):
        return {f.name: data(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(k): data(v) for k, v in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [data(v) for v in value]
    return value


def canonical(value) -> str:
    return json.dumps(data(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value) -> str:
    return sha256(canonical(value).encode()).hexdigest()


@dataclass(frozen=True)
class Contract:
    contract_id: str
    expiry: date
    strike: float
    side: str = "call"
    multiplier: float = 1.0
    lot: float = 1.0
    listed_on: date = date.min
    underlying_id: int = 1
    currency_id: int = 2
    settlement_rule: str = "snapshot_spot"
    settlement_fixing_id: str | None = None
    payment_date: date | None = None

    def __post_init__(self):
        require(bool(self.contract_id) and self.contract_id != "__spot__",
                "invalid_contract", "contract_id is required and __spot__ is reserved")
        require(self.side in ("call", "put"), "unsupported_product", self.side)
        for name in ("strike", "multiplier", "lot"):
            finite(getattr(self, name), name)
            require(getattr(self, name) > 0, "invalid_contract", name)
        require(self.listed_on < self.expiry, "invalid_contract", "listing must precede expiry")
        require(self.payment_date is None or (isinstance(self.payment_date, date)
                and not isinstance(self.payment_date, datetime) and self.payment_date >= self.expiry),
                "invalid_payment_date", "payment must be on or after expiry")
        require(self.settlement_rule in ("snapshot_spot", "explicit_fixing")
                or (self.settlement_rule == "path_observations" and isinstance(self, PathContract)),
                "unsupported_settlement", self.settlement_rule)
        if self.settlement_rule == "explicit_fixing":
            require(isinstance(self.settlement_fixing_id, str) and bool(self.settlement_fixing_id.strip()),
                    "invalid_settlement", "explicit fixing requires a fixing ID")
        else:
            require(self.settlement_fixing_id is None, "invalid_settlement", "snapshot spot must not specify a fixing ID")

    def payoff(self, spot: float) -> float:
        return max((spot - self.strike) * (1 if self.side == "call" else -1), 0.0)

    @property
    def payment_day(self) -> date:
        return self.payment_date if self.payment_date is not None else self.expiry


@dataclass(frozen=True)
class PathObservation:
    observation_date: date
    fixing_id: str
    weight: float = 1.0

    def __post_init__(self):
        require(isinstance(self.observation_date, date) and not isinstance(self.observation_date, datetime),
                "invalid_path_observation", "calendar observation date required")
        require(isinstance(self.fixing_id, str) and bool(self.fixing_id.strip()),
                "invalid_path_observation", "fixing ID required")
        finite(self.weight, "observation weight")
        require(self.weight > 0, "invalid_path_observation", "positive weight required")


@dataclass(frozen=True, kw_only=True)
class PathContract(Contract):
    product_type: str
    observations: tuple[PathObservation, ...]
    settlement_rule: str = "path_observations"

    def __post_init__(self):
        super().__post_init__()
        require(self.product_type in ("arithmetic_asian", "fixed_lookback")
                and self.settlement_rule == "path_observations", "unsupported_product", "path observation settlement required")
        object.__setattr__(self, "observations", tuple(self.observations))
        require(bool(self.observations) and all(isinstance(o, PathObservation) for o in self.observations),
                "invalid_path_observation", "nonempty observation schedule required")
        require(all(a.observation_date < b.observation_date for a,b in zip(self.observations,self.observations[1:]))
                and len({o.fixing_id for o in self.observations}) == len(self.observations)
                and self.observations[-1].observation_date == self.expiry,
                "invalid_path_observation", "unique ordered dates and IDs ending at expiry required")
        require((self.product_type == "arithmetic_asian" and math.isclose(math.fsum(o.weight for o in self.observations), 1., rel_tol=0., abs_tol=1e-12))
                or (self.product_type == "fixed_lookback" and all(o.weight == 1. for o in self.observations)),
                "invalid_path_observation", "Asian weights must sum to one; Lookback weights must equal one")

    def payoff(self, spot: float) -> float:
        raise BacktestError("path_history_required", "path payoff requires the complete fixing schedule")

    def observed_payoff(self, values) -> float:
        require(len(values) == len(self.observations), "path_history_required", self.contract_id)
        for v in values: finite(v, "path fixing", 0.)
        underlying = (checked_sum((o.weight*v for o,v in zip(self.observations,values)), "nonfinite_settlement", self.contract_id)
                      if self.product_type == "arithmetic_asian" else (max(values) if self.side == "call" else min(values)))
        return max((underlying-self.strike)*(1 if self.side == "call" else -1), 0.)


@dataclass(frozen=True)
class Position:
    contract_id: str
    quantity: float
    entry_price: float
    entry_fee: float = 0.0

    def __post_init__(self):
        finite(self.quantity, "quantity")
        finite(self.entry_price, "entry_price", 0.0)
        finite(self.entry_fee, "entry_fee", 0.0)


@dataclass(frozen=True)
class Quote:
    bid: float
    ask: float
    observed_at: datetime
    available_at: datetime

    def __post_init__(self):
        finite(self.bid, "bid", 0.0)
        finite(self.ask, "ask", 0.0)
        aware(self.observed_at)
        aware(self.available_at)
        require(self.bid <= self.ask, "crossed_quote", "bid exceeds ask")
        require(self.observed_at <= self.available_at, "invalid_timestamp", "availability before observation")

    @property
    def mid(self) -> float:
        return self.bid / 2 + self.ask / 2

    def check(self, at: datetime, max_age: float) -> None:
        require(self.available_at <= at, "future_quote", "quote was not yet available")
        require(0 <= (at - self.observed_at).total_seconds() <= max_age,
                "stale_quote", "quote age exceeds the configured limit")


@dataclass(frozen=True)
class SettlementFixing:
    """A sourced underlying-price fixing, not an option payoff or tradable quote."""

    fixing_date: date
    value: float
    observed_at: datetime
    available_at: datetime
    source: str
    revision: str

    def __post_init__(self):
        require(isinstance(self.fixing_date, date) and not isinstance(self.fixing_date, datetime),
                "invalid_settlement_fixing", "fixing_date must be a calendar date")
        finite(self.value, "settlement_fixing", 0.0)
        aware(self.observed_at)
        aware(self.available_at)
        require(self.observed_at <= self.available_at, "invalid_timestamp", "fixing availability before observation")
        require(all(isinstance(v, str) and bool(v.strip()) for v in (self.source, self.revision)),
                "missing_provenance", "fixing source and revision required")

    def check_available(self, at: datetime) -> None:
        require(self.available_at <= at, "future_settlement_fixing", "fixing was not yet available")

    def check_expiry(self, expiry: date, at: datetime) -> None:
        self.check_available(at)
        require(self.fixing_date == expiry, "settlement_date_mismatch", "fixing date differs from expiry")
        require(self.observed_at.astimezone(at.tzinfo).date() == expiry,
                "settlement_date_mismatch", "fixing observation must be on expiry in the session timezone")


@dataclass(frozen=True)
class PaymentDiscount:
    """Point discount factor for a payment date, separate from cash funding."""

    valuation_date: date
    factor: float
    observed_at: datetime
    available_at: datetime
    source: str
    revision: str

    def __post_init__(self):
        require(isinstance(self.valuation_date, date) and not isinstance(self.valuation_date, datetime),
                "invalid_payment_discount", "valuation date required")
        finite(self.factor, "payment_discount")
        require(self.factor > 0, "invalid_payment_discount", "positive factor required")
        aware(self.observed_at)
        aware(self.available_at)
        require(self.observed_at <= self.available_at, "invalid_timestamp", "discount availability before observation")
        require(all(isinstance(v, str) and bool(v.strip()) for v in (self.source, self.revision)),
                "missing_provenance", "discount source and revision required")

    def check(self, at: datetime, payment_day: date, max_age: float) -> None:
        require(self.valuation_date == at.date(), "payment_discount_date_mismatch", "discount must be anchored to this valuation date")
        require(self.available_at <= at, "future_payment_discount", "discount was not yet available")
        require(0 <= (at - self.observed_at).total_seconds() <= max_age,
                "stale_payment_discount", "discount observation is too old")
        require(payment_day >= at.date(), "invalid_payment_discount", "payment discount date is in the past")
        require(payment_day != at.date() or self.factor == 1., "invalid_payment_discount", "same-day factor must equal one")


@dataclass(frozen=True)
class NumeraireObservation:
    """Dated value of an explicitly selected money-market account."""

    account_id: str
    currency_id: int
    valuation_date: date
    value: float
    observed_at: datetime
    available_at: datetime
    source: str
    revision: str

    def __post_init__(self):
        require(isinstance(self.account_id, str) and bool(self.account_id.strip()),
                "invalid_numeraire", "account ID required")
        require(isinstance(self.currency_id, int) and not isinstance(self.currency_id, bool),
                "invalid_numeraire", "integer currency ID required")
        require(isinstance(self.valuation_date, date) and not isinstance(self.valuation_date, datetime),
                "invalid_numeraire", "valuation date required")
        finite(self.value, "numeraire_value")
        require(self.value > 0, "invalid_numeraire", "positive account value required")
        aware(self.observed_at)
        aware(self.available_at)
        require(self.observed_at <= self.available_at, "invalid_timestamp", "account availability before observation")
        require(all(isinstance(v, str) and bool(v.strip()) for v in (self.source, self.revision)),
                "missing_provenance", "account source and revision required")

    def check(self, at: datetime, account_id: str | None, currency_id: int, max_age: float) -> None:
        require(account_id is None or self.account_id == account_id,
                "numeraire_account_mismatch", "account differs from configured numeraire")
        require(self.currency_id == currency_id, "numeraire_currency_mismatch", "account currency differs from portfolio")
        require(self.valuation_date == at.date(), "numeraire_date_mismatch", "account value must be dated to this session")
        require(self.available_at <= at, "future_numeraire", "account value was not yet available")
        require(0 <= (at - self.observed_at).total_seconds() <= max_age,
                "stale_numeraire", "account observation is too old")


@dataclass(frozen=True)
class ExternalCashflow:
    """Signed investor capital, booked only at an explicit replay snapshot."""

    flow_id: str
    effective_at: datetime
    amount: float
    currency_id: int
    available_at: datetime
    source: str
    revision: str

    def __post_init__(self):
        require(isinstance(self.flow_id, str) and bool(self.flow_id.strip()),
                "invalid_external_cashflow", "flow ID required")
        finite(self.amount, "external_cashflow_amount")
        require(self.amount != 0, "invalid_external_cashflow", "zero capital movement is not an event")
        require(isinstance(self.currency_id, int) and not isinstance(self.currency_id, bool),
                "invalid_external_cashflow", "integer currency ID required")
        aware(self.effective_at)
        aware(self.available_at)
        require(all(isinstance(v, str) and bool(v.strip()) for v in (self.source, self.revision)),
                "missing_provenance", "capital movement source and revision required")

    def check(self, at: datetime, currency_id: int) -> None:
        require(self.effective_at == at, "external_cashflow_time_mismatch",
                "capital movement must occur exactly at its supplied snapshot")
        require(self.available_at <= at, "future_external_cashflow", "capital movement was not yet available")
        require(self.currency_id == currency_id, "external_cashflow_currency_mismatch",
                "capital currency differs from portfolio")


@dataclass(frozen=True)
class RiskBasis:
    valuation_date: date
    maturity_dates: tuple[date, ...]
    log_moneyness: tuple[float, ...]
    method: str = "lv_paper_decomposition"
    coordinate: str = "canonical_f_iv"
    delta_convention: str = "frozen_local_variance_grid"
    unit: str = "currency_per_unit_absolute_volatility"

    def __post_init__(self):
        object.__setattr__(self, "maturity_dates", tuple(self.maturity_dates))
        object.__setattr__(self, "log_moneyness", tuple(self.log_moneyness))
        require(len(self.maturity_dates) >= 2 and len(self.log_moneyness) >= 2,
                "invalid_basis", "at least two nodes on each axis are required")
        require(all(self.valuation_date < t for t in self.maturity_dates), "invalid_basis", "nonpositive maturity")
        require(all(a < b for a, b in zip(self.maturity_dates, self.maturity_dates[1:])),
                "invalid_basis", "maturities must strictly increase")
        require(all(math.isfinite(x) for x in self.log_moneyness)
                and all(a < b for a, b in zip(self.log_moneyness, self.log_moneyness[1:])),
                "invalid_basis", "log-moneyness must be finite and strictly increasing")
        require(self.unit == "currency_per_unit_absolute_volatility", "unit_mismatch", self.unit)

    @property
    def size(self) -> int:
        return len(self.maturity_dates) * len(self.log_moneyness)

    @property
    def basis_id(self) -> str:
        return fingerprint(self)


@dataclass(frozen=True)
class Snapshot:
    snapshot_id: str
    at: datetime
    spot: Quote
    option_quotes: Mapping[str, Quote]
    basis: RiskBasis
    surface_available_at: datetime
    source: str
    pricing_requests: Mapping[str, str] | None = None
    lending_rate: float = 0.0
    borrowing_rate: float = 0.0
    stock_borrow_rate: float = 0.0
    dividend_per_share: float = 0.0
    settlement_fixings: Mapping[str, SettlementFixing] | None = None
    payment_discounts: Mapping[date, PaymentDiscount] | None = None
    numeraire: NumeraireObservation | None = None
    external_cashflows: tuple[ExternalCashflow, ...] = ()

    def __post_init__(self):
        aware(self.at)
        aware(self.surface_available_at)
        require(bool(self.snapshot_id) and bool(self.source), "missing_provenance", "snapshot ID and source required")
        require(self.basis.valuation_date == self.at.date(), "basis_mismatch", "valuation date differs")
        require(self.spot.mid > 0, "invalid_spot", "spot must be positive")
        for name in ("lending_rate", "borrowing_rate", "stock_borrow_rate", "dividend_per_share"):
            finite(getattr(self, name), name)
        require(self.stock_borrow_rate >= 0 and self.dividend_per_share >= 0,
                "invalid_carry", "negative stock borrow fee or dividend")
        object.__setattr__(self, "option_quotes", MappingProxyType(dict(self.option_quotes)))
        object.__setattr__(self, "pricing_requests", MappingProxyType(dict(self.pricing_requests or {})))
        fixings = dict(self.settlement_fixings or {})
        require(all(isinstance(k, str) and bool(k.strip()) and isinstance(v, SettlementFixing)
                    for k, v in fixings.items()), "invalid_settlement_fixing", "fixing IDs and SettlementFixing values required")
        object.__setattr__(self, "settlement_fixings", MappingProxyType(fixings))
        discounts = dict(self.payment_discounts or {})
        require(all(isinstance(k, date) and not isinstance(k, datetime) and isinstance(v, PaymentDiscount)
                    for k, v in discounts.items()), "invalid_payment_discount", "date keys and PaymentDiscount values required")
        object.__setattr__(self, "payment_discounts", MappingProxyType(discounts))
        require(self.numeraire is None or isinstance(self.numeraire, NumeraireObservation),
                "invalid_numeraire", "NumeraireObservation required")
        object.__setattr__(self, "external_cashflows", tuple(self.external_cashflows))
        require(all(isinstance(f, ExternalCashflow) for f in self.external_cashflows),
                "invalid_external_cashflow", "ExternalCashflow values required")


@dataclass(frozen=True)
class Risk:
    price: float
    delta: float
    buckets: tuple[float, ...]
    basis_id: str
    signed_residual: float = 0.0
    pre_projection: float | None = None
    price_standard_error: float | None = None
    edge_sensitivity_abs: float = 0.0
    provenance: str = ""
    bucket_standard_errors: tuple[float, ...] | None = None
    warning_codes: tuple[str, ...] = ()
    native_policy: str | None = None
    payment_date: date | None = None
    payment_scaling_factor: float = 1.0
    gamma: float | None = None
    gamma_method: str | None = None
    gamma_spot_bump: float | None = None
    reporting_iv: tuple[float, ...] | None = None
    reporting_forwards: tuple[float, ...] | None = None

    def __post_init__(self):
        object.__setattr__(self, "buckets", tuple(self.buckets))
        object.__setattr__(self, "warning_codes", tuple(self.warning_codes))
        if self.bucket_standard_errors is not None:
            object.__setattr__(self, "bucket_standard_errors", tuple(self.bucket_standard_errors))
            require(len(self.bucket_standard_errors) == len(self.buckets), "invalid_risk", "bucket SE shape")
            for se in self.bucket_standard_errors:
                finite(se, "bucket_standard_error", 0.0)
        for x in (self.price, self.delta, self.signed_residual, self.edge_sensitivity_abs, *self.buckets):
            finite(x, "risk")
        finite(self.payment_scaling_factor, "payment_scaling_factor")
        require(self.payment_scaling_factor > 0, "invalid_risk", "positive payment scaling required")
        require(self.payment_date is None or (isinstance(self.payment_date, date) and not isinstance(self.payment_date, datetime)),
                "invalid_risk", "payment date must be a calendar date")
        require(self.price >= 0 and self.edge_sensitivity_abs >= 0, "invalid_risk", "negative price or edge diagnostic")
        if self.price_standard_error is not None:
            finite(self.price_standard_error, "price_standard_error", 0.0)
        if self.gamma is not None:
            finite(self.gamma, "gamma")
        require(self.gamma_method is None or (isinstance(self.gamma_method, str) and bool(self.gamma_method.strip())),
                "invalid_risk", "gamma method must be a nonempty string")
        if self.gamma_spot_bump is not None:
            finite(self.gamma_spot_bump, "gamma_spot_bump")
            require(self.gamma_spot_bump > 0, "invalid_risk", "positive gamma bump required")
        require(self.gamma is not None or (self.gamma_method is None and self.gamma_spot_bump is None),
                "invalid_risk", "gamma metadata requires a calculated gamma")
        require((self.reporting_iv is None) == (self.reporting_forwards is None),
                "invalid_reporting_surface", "IV and forwards must be supplied together")
        if self.reporting_iv is not None:
            object.__setattr__(self, "reporting_iv", tuple(self.reporting_iv))
            object.__setattr__(self, "reporting_forwards", tuple(self.reporting_forwards))
            require(len(self.reporting_iv) == len(self.buckets) and len(self.reporting_forwards) >= 2
                    and len(self.reporting_iv) % len(self.reporting_forwards) == 0
                    and len(self.reporting_iv) // len(self.reporting_forwards) >= 2,
                    "invalid_reporting_surface", "reporting surface shape")
            require(all(math.isfinite(x) and x > 0 for x in (*self.reporting_iv, *self.reporting_forwards)),
                    "invalid_reporting_surface", "positive finite IV and forward levels required")
        if self.pre_projection is not None:
            finite(self.pre_projection, "pre_projection")
            expected = math.fsum(self.buckets) + self.signed_residual
            require(math.isclose(self.pre_projection, expected, rel_tol=1e-10, abs_tol=1e-10),
                    "projection_mismatch", "buckets plus residual do not reconcile")


class RiskProvider(Protocol):
    @property
    def identity(self) -> str: ...

    def evaluate(self, snapshot: Snapshot, contract: Contract) -> Risk: ...


@dataclass(frozen=True)
class BacktestConfig:
    strategy: str = "vega_kt"
    rebalance_every: int = 1
    initial_cash: float = 0.0
    cash_floor: float = -1_000_000.0
    option_position_limit: float = 100.0
    option_trade_limit: float = 100.0
    stock_position_limit: float = 100_000.0
    stock_trade_limit: float = 100_000.0
    stock_lot: float = 0.01
    allow_option_shorts: bool = True
    allow_stock_shorts: bool = True
    option_fee: float = 0.0
    stock_fee: float = 0.0
    slippage_bps: float = 0.0
    max_quote_age_seconds: float = 86400.0
    max_relative_spread: float = 1.0
    min_vega: float = 1e-8
    ridge: float = 1e-8
    turnover_penalty: float = 0.0
    solver_tolerance: float = 1e-9
    solver_iterations: int = 10_000
    max_abs_projection_residual: float = 1_000_000.0
    max_abs_edge_sensitivity: float = 1_000_000.0
    max_price_standard_error: float = 1_000_000.0
    max_bucket_standard_error: float = 1_000_000.0
    target_mark: str = "model"
    end_policy: str = "liquidate_hedges"
    allow_identical_hedge: bool = False
    numeraire_id: str | None = None
    constraint_mode: str = "execution_cash"
    gross_notional_limit: float | None = None
    market_shock_log_moneyness: tuple[float, ...] | None = None
    max_joint_bucket_standard_error: float | None = None

    def __post_init__(self):
        require(self.strategy in ("unhedged", "delta", "scalar_vega", "vega_kt"), "invalid_strategy", self.strategy)
        require(self.target_mark in ("model", "market"), "invalid_mark", self.target_mark)
        require(self.end_policy in ("mark_to_market", "liquidate_hedges"), "invalid_end_policy", self.end_policy)
        require(isinstance(self.rebalance_every, int) and self.rebalance_every > 0,
                "invalid_config", "rebalance_every")
        require(isinstance(self.solver_iterations, int) and self.solver_iterations > 0,
                "invalid_config", "solver_iterations")
        finite(self.initial_cash, "initial_cash")
        finite(self.cash_floor, "cash_floor")
        require(self.constraint_mode in ("execution_cash", "decision_cash_gross"), "invalid_config", "constraint_mode")
        if self.constraint_mode == "decision_cash_gross":
            require(type(self.gross_notional_limit) in (int, float), "invalid_config", "gross_notional_limit required")
            finite(self.gross_notional_limit, "gross_notional_limit", 0.)
        else:
            require(self.gross_notional_limit is None, "invalid_config", "gross cap requires decision_cash_gross mode")
        require(self.numeraire_id is None or (isinstance(self.numeraire_id, str) and bool(self.numeraire_id.strip())),
                "invalid_config", "numeraire_id must be a nonempty account ID or None")
        if self.max_joint_bucket_standard_error is not None:
            require(type(self.max_joint_bucket_standard_error) in (int, float),
                    "invalid_config", "joint bucket error limit must be a number")
            finite(self.max_joint_bucket_standard_error, "max_joint_bucket_standard_error", 0.)
        if self.market_shock_log_moneyness is not None:
            nodes = self.market_shock_log_moneyness
            require(isinstance(nodes, (tuple, list)) and len(nodes) >= 2
                    and all(type(x) in (int, float) and math.isfinite(x) for x in nodes)
                    and all(a < b for a, b in zip(nodes, nodes[1:])),
                    "invalid_config", "market shock nodes must be finite and strictly increasing")
            object.__setattr__(self, "market_shock_log_moneyness", tuple(nodes))
        non_numeric = {"strategy", "rebalance_every", "solver_iterations", "initial_cash", "cash_floor",
                       "allow_option_shorts", "allow_stock_shorts", "target_mark", "end_policy", "allow_identical_hedge",
                       "numeraire_id", "constraint_mode", "gross_notional_limit", "market_shock_log_moneyness",
                       "max_joint_bucket_standard_error"}
        for f in fields(self):
            if f.name not in non_numeric:
                finite(getattr(self, f.name), f.name, 0.0)
        require(self.stock_lot > 0 and self.solver_tolerance > 0, "invalid_config", "positive lot and tolerance required")
        require(self.slippage_bps < 10_000, "invalid_config", "slippage must be below 100%")
