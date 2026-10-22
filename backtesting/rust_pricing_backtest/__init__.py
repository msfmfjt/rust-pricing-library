"""Daily VegaKT research backtests using the public rust_pricing engine."""

from .domain import BacktestConfig, BacktestError, Contract, ExternalCashflow, NumeraireObservation, PaymentDiscount, Position, Quote, Risk, RiskBasis, SettlementFixing, Snapshot
from .engine import BacktestResult, BacktestRunError, run_backtest, compare_strategies
from .rust_adapter import RustPricingProvider
from .domain import PathContract, PathObservation
from .task_adapter import PricingEvidence

__all__ = ["BacktestConfig", "BacktestError", "Contract", "ExternalCashflow", "Position", "Quote", "Risk", "RiskBasis", "PricingEvidence",
           "NumeraireObservation", "PaymentDiscount", "SettlementFixing", "Snapshot", "PathContract", "PathObservation", "BacktestResult", "BacktestRunError", "RustPricingProvider", "run_backtest", "compare_strategies"]
