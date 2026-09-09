"""Typed Python facade for deterministic derivatives valuation."""

from collections.abc import Sequence
from datetime import date
from typing import Literal

DateLike = date | str
OptionSide = Literal["call", "put"]
DigitalPayout = Literal["cash", "asset"]
BarrierDirection = Literal["up", "down"]
BarrierStyle = Literal["knock_in", "knock_out"]
SmileDynamics = Literal[
    "sticky_log_moneyness", "sticky_strike", "sticky_delta"
]
VegaKtCovarianceLayout = Literal[
    "price_and_bucket_variance_only", "full_bucket_matrix_row_major"
]
VegaKtUnit = Literal[
    "currency_per_unit_absolute_volatility", "currency_per_volatility_point"
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


class AsianObservation:
    @staticmethod
    def unknown(date: DateLike, weight: float) -> AsianObservation: ...
    @staticmethod
    def known(date: DateLike, weight: float, fixing: float) -> AsianObservation: ...
    @property
    def date(self) -> str: ...
    @property
    def weight(self) -> float: ...
    @property
    def fixing(self) -> float | None: ...


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
    @staticmethod
    def digital(
        underlying_id: int,
        currency_id: int,
        expiry: DateLike,
        strike: float,
        payout: float,
        side: OptionSide,
        payout_kind: DigitalPayout,
        *,
        payment_date: DateLike | None = None,
    ) -> Product: ...
    @staticmethod
    def barrier(
        underlying_id: int,
        currency_id: int,
        expiry: DateLike,
        strike: float,
        barrier: float,
        notional: float,
        side: OptionSide,
        direction: BarrierDirection,
        style: BarrierStyle,
        monitoring_dates: Sequence[DateLike],
        payment_date: DateLike,
        *,
        rebate: float | None = None,
    ) -> Product: ...
    @staticmethod
    def arithmetic_asian(
        underlying_id: int,
        currency_id: int,
        strike: float,
        notional: float,
        side: OptionSide,
        observations: Sequence[AsianObservation],
        payment_date: DateLike,
    ) -> Product: ...
    @staticmethod
    def fixed_lookback(
        underlying_id: int,
        currency_id: int,
        strike: float,
        notional: float,
        side: OptionSide,
        monitoring_dates: Sequence[DateLike],
        payment_date: DateLike,
        *,
        historical_extremum: float | None = None,
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
    def black_76(volatility: float) -> Model: ...
    @staticmethod
    def local_volatility_from_grid(
        time_nodes: Sequence[float],
        log_forward_moneyness_nodes: Sequence[float],
        local_variances: Sequence[float],
        floor: float,
        cap: float,
    ) -> Model: ...
    @staticmethod
    def local_volatility_from_grid_with_reporting_basis(
        time_nodes: Sequence[float],
        log_forward_moneyness_nodes: Sequence[float],
        local_variances: Sequence[float],
        floor: float,
        cap: float,
        reporting_maturity_nodes: Sequence[float],
        reporting_log_forward_moneyness_nodes: Sequence[float],
        reporting_implied_volatilities: Sequence[float],
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


class VegaKtCoordinate:
    @property
    def maturity(self) -> float: ...
    @property
    def log_moneyness(self) -> float: ...
    @property
    def implied_volatility(self) -> float: ...


class VegaKtBucketEstimate:
    @property
    def raw_mean(self) -> float: ...
    @property
    def market_scaled_mean(self) -> float: ...
    @property
    def sample_variance(self) -> float | None: ...
    @property
    def price_covariance(self) -> float | None: ...


class VegaKtReportingStats:
    @property
    def left_edge_count(self) -> int: ...
    @property
    def right_edge_count(self) -> int: ...
    @property
    def left_edge_sensitivity(self) -> float: ...
    @property
    def right_edge_sensitivity(self) -> float: ...


class VegaKtProjection:
    @property
    def scalar_vega(self) -> float: ...
    @property
    def signed_residual(self) -> float: ...
    @property
    def pre_projection(self) -> float: ...
    @property
    def reporting_stats(self) -> VegaKtReportingStats: ...


class VegaKtResidualDiagnostics:
    @property
    def active_domain_start_index(self) -> int: ...
    @property
    def active_domain_end_index(self) -> int: ...
    @property
    def active_domain_forward_index(self) -> int: ...
    @property
    def excluded_probability_mass(self) -> float: ...
    @property
    def signed_residual(self) -> float: ...
    @property
    def pre_projection(self) -> float: ...
    @property
    def reporting_stats(self) -> VegaKtReportingStats: ...


class VegaKtResult:
    @property
    def coordinates(self) -> list[VegaKtCoordinate]: ...
    @property
    def estimates(self) -> list[VegaKtBucketEstimate]: ...
    @property
    def raw_buckets(self) -> list[float]: ...
    @property
    def full_bucket_covariance(self) -> list[float | None] | None: ...
    @property
    def covariance_layout(self) -> VegaKtCovarianceLayout: ...
    @property
    def projection(self) -> VegaKtProjection: ...
    @property
    def residual_diagnostics(self) -> VegaKtResidualDiagnostics: ...
    @property
    def raw_unit(self) -> VegaKtUnit: ...
    @property
    def market_scaled_unit(self) -> VegaKtUnit: ...
    @property
    def policy_label(self) -> str: ...
    @property
    def truncation_order(self) -> str: ...


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
    def vega_kt(self) -> VegaKtResult | None: ...
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
