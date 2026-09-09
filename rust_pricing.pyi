"""Typed Python facade for deterministic derivatives valuation."""

from collections.abc import Sequence
from datetime import date
from typing import Literal

DateLike = date | str
OptionSide = Literal["call", "put"]
SmileDynamics = Literal[
    "sticky_log_moneyness", "sticky_strike", "sticky_delta"
]

__version__: str


class ValidationIssue:
    @property
    def pointer(self) -> str: ...
    @property
    def phase(self) -> str: ...
    @property
    def code(self) -> str: ...
    @property
    def message(self) -> str: ...
    def to_dict(self) -> dict[str, str]: ...


class ValidationError(ValueError):
    issues: list[ValidationIssue]


class PricingError(RuntimeError): ...


class PricingWarning:
    @property
    def code(self) -> str: ...
    @property
    def message(self) -> str: ...


class Diagnostics:
    @property
    def master_seed(self) -> int: ...
    @property
    def estimator(self) -> str: ...
    @property
    def scramble_count(self) -> int | None: ...
    @property
    def policy_version(self) -> int: ...
    @property
    def worker_threads(self) -> int: ...
    @property
    def reduction_block_size(self) -> int: ...
    @property
    def aad_tile_policy_version(self) -> int: ...
    @property
    def aad_tile_capacity(self) -> int: ...
    @property
    def checkpoint_policy_version(self) -> int: ...
    @property
    def checkpoint_interval(self) -> int: ...
    @property
    def antithetic(self) -> bool: ...
    @property
    def discount_region(self) -> str: ...
    @property
    def dividend_region(self) -> str: ...
    @property
    def payoff_fingerprint(self) -> str: ...
    @property
    def delta_method(self) -> str | None: ...
    @property
    def gamma_method(self) -> str | None: ...
    @property
    def vega_method(self) -> str | None: ...
    @property
    def warnings(self) -> list[PricingWarning]: ...


class DiscountCurve:
    def __init__(
        self,
        curve_id: int,
        times: Sequence[float],
        discount_factors: Sequence[float],
    ) -> None: ...
    @property
    def curve_id(self) -> int: ...


class DividendEvent:
    @staticmethod
    def fixed_cash(event_id: int, ex_time: float, amount: float) -> DividendEvent: ...
    @staticmethod
    def proportional(event_id: int, ex_time: float, beta: float) -> DividendEvent: ...
    @staticmethod
    def fixed_cash_and_proportional(
        event_id: int, ex_time: float, fixed_cash: float, beta: float
    ) -> DividendEvent: ...
    @property
    def event_id(self) -> int: ...
    @property
    def ex_time(self) -> float: ...


class EssviSlice:
    def __init__(self, time: float, theta: float, psi: float, rho_psi: float) -> None: ...
    @property
    def time(self) -> float: ...
    @property
    def theta(self) -> float: ...
    @property
    def psi(self) -> float: ...
    @property
    def rho_psi(self) -> float: ...


class Product:
    @staticmethod
    def european_vanilla(
        underlying_id: int,
        currency_id: int,
        expiry: DateLike,
        strike: float,
        notional: float,
        side: OptionSide,
    ) -> Product: ...


class Market:
    @staticmethod
    def equity(
        currency_id: int,
        underlying_id: int,
        spot: float,
        discount_curve: DiscountCurve,
        dividend_curve: DiscountCurve,
        *,
        discrete_dividends: Sequence[DividendEvent] | None = None,
    ) -> Market: ...


class Model:
    @staticmethod
    def black_scholes(volatility: float) -> Model: ...
    @staticmethod
    def local_volatility_from_grid(
        time_nodes: Sequence[float],
        log_forward_moneyness_nodes: Sequence[float],
        local_variances: Sequence[float],
        floor: float,
        cap: float,
    ) -> Model: ...
    @staticmethod
    def local_volatility_from_essvi(
        slices: Sequence[EssviSlice],
        terminal_theta_slope: float,
        time_nodes: Sequence[float],
        log_forward_moneyness_nodes: Sequence[float],
        floor: float,
        cap: float,
    ) -> Model: ...
    @staticmethod
    def local_volatility_from_standard_ssvi_power_law(
        theta_times: Sequence[float],
        theta_values: Sequence[float],
        terminal_theta_slope: float,
        rho: float,
        eta: float,
        gamma: float,
        time_nodes: Sequence[float],
        log_forward_moneyness_nodes: Sequence[float],
        floor: float,
        cap: float,
    ) -> Model: ...
    @staticmethod
    def local_volatility_from_standard_ssvi_heston_like(
        theta_times: Sequence[float],
        theta_values: Sequence[float],
        terminal_theta_slope: float,
        rho: float,
        lambda_: float,
        time_nodes: Sequence[float],
        log_forward_moneyness_nodes: Sequence[float],
        floor: float,
        cap: float,
    ) -> Model: ...


class Engine:
    @staticmethod
    def pseudo_monte_carlo(
        master_seed: int,
        independent_sampling_units: int,
        *,
        antithetic: bool = False,
        brownian_bridge: bool = False,
    ) -> Engine: ...
    @staticmethod
    def randomized_quasi_monte_carlo(
        points_per_scramble: int,
        master_scramble_seed: int,
        *,
        scramble_count: int = 16,
        antithetic: bool = False,
        brownian_bridge: bool = True,
    ) -> Engine: ...


class RiskRequest:
    def __init__(
        self,
        *,
        delta: bool = False,
        gamma_relative_bump: float | None = None,
        gamma_absolute_bump: float | None = None,
        vega: bool = False,
        vega_kt_maturity_nodes: Sequence[DateLike] | None = None,
        vega_kt_log_forward_moneyness_nodes: Sequence[float] | None = None,
        vega_kt_relative_density_threshold: float | None = None,
        vega_kt_full_bucket_covariance: bool = False,
        smile_dynamics: SmileDynamics = "sticky_log_moneyness",
        checkpoint_interval: int | None = None,
        aad_tile_capacity: int | None = None,
    ) -> None: ...


class PricingRequest:
    def __init__(
        self,
        valuation_date: DateLike,
        product: Product,
        market: Market,
        model: Model,
        engine: Engine,
        risk: RiskRequest,
    ) -> None: ...
    @staticmethod
    def from_json(json: str) -> PricingRequest: ...
    def to_json(self) -> str: ...
    @property
    def fingerprint(self) -> str: ...


class PricingPlan:
    @staticmethod
    def compile(
        request: PricingRequest,
        *,
        worker_threads: int,
        reduction_block_size: int | None = None,
    ) -> PricingPlan: ...
    def evaluate(self) -> PricingResult: ...
    @property
    def request_fingerprint(self) -> str: ...
    @property
    def plan_fingerprint(self) -> str: ...
    @property
    def worker_threads(self) -> int: ...
    @property
    def reduction_block_size(self) -> int: ...


class PricingResult:
    @property
    def value(self) -> float: ...
    @property
    def standard_error(self) -> float: ...
    @property
    def confidence_interval(self) -> tuple[float, float]: ...
    @property
    def delta_raw(self) -> float | None: ...
    @property
    def delta_market_scaled(self) -> float | None: ...
    @property
    def gamma_raw(self) -> float | None: ...
    @property
    def gamma_market_scaled(self) -> float | None: ...
    @property
    def vega_raw(self) -> float | None: ...
    @property
    def vega_market_scaled(self) -> float | None: ...
    @property
    def sampling_variance(self) -> float: ...
    @property
    def estimator_variance(self) -> float: ...
    @property
    def independent_sampling_units(self) -> int: ...
    @property
    def evaluated_paths(self) -> int: ...
    @property
    def diagnostics(self) -> Diagnostics: ...
    @property
    def warnings(self) -> list[PricingWarning]: ...
    def to_json(self) -> str: ...


def version() -> str: ...
