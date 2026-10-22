"""Translate common pricing tasks/reports at the native backtest boundary."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Literal

from .domain import require

if TYPE_CHECKING:
    from rust_pricing import PricingResult, PricingTask, ValuationReport


@dataclass(frozen=True)
class PricingEvidence:
    """Replayable native evaluation, before the ledger's payment adjustment.

    ``ordinary_price_and_greeks`` excludes the separately accepted market-Gamma
    buckets and cross-instrument covariance. No compiled plan is retained.
    """

    task: PricingTask
    report: ValuationReport
    payment_date: date
    payment_scaling_factor: float
    scope: Literal['requested_risks', 'ordinary_price_and_greeks']


@dataclass(frozen=True)
class TaskValuation:
    task: PricingTask
    report: ValuationReport
    native: PricingResult

    def evidence(self, contract, *, payment_scale=1., scope='requested_risks'):
        return PricingEvidence(self.task, self.report, contract.payment_day, payment_scale, scope)


def evaluate_request(rp, request, *, worker_threads, reduction_block_size):
    """One common task evaluation; native-only warnings/diagnostics stay typed."""
    source = rp.PricingSpecification.from_request(request, worker_threads=worker_threads,
                                                  reduction_block_size=reduction_block_size)
    task = rp.PricingTask.from_specification(source)
    result = task.compile().evaluate()
    specification = result.native_result
    require(isinstance(specification, rp.SpecificationResult), 'unsupported_task_result',
            'the legacy request adapter requires a single-asset specification result')
    native = specification.native_result
    require(isinstance(native, rp.PricingResult), 'unsupported_task_result',
            'this ledger adapter requires the standard native result and its diagnostics')
    return TaskValuation(task, result.report, native)


def scalar_risk(report, name, *, underlying_id, unit, required=True):
    """Select a scalar without flattening asset axes or inferring unit scales."""
    blocks = [r for r in report.risks if r.name == name]
    if not blocks:
        require(not required, 'missing_risk', name)
        return None
    require(len(blocks) == 1, 'task_risk_shape', f'{name}: one scalar risk block required')
    block = blocks[0]
    require(block.underlying_ids == [underlying_id] and not block.axes and len(block.values) == 1,
            'task_risk_shape', f'{name}: expected one value for the contract underlying')
    require(block.unit == unit, 'unit_mismatch', f'{name}: {block.unit}')
    require(block.calibration == 'not_applicable', 'task_risk_convention',
            f'{name}: the current ledger requires the configured fixed-grid risk')
    return block


def ordinary_greeks(report, contract):
    """The current LV ledger's explicit currency, uncertainty and Greek contract."""
    require(report.currency_id == contract.currency_id, 'currency_mismatch', contract.contract_id)
    require(report.uncertainty_scope == 'pricing_sampling', 'task_risk_convention',
            'the current ledger requires ordinary pricing-sampling uncertainty')
    delta = scalar_risk(report, 'delta.raw', underlying_id=contract.underlying_id, unit='currency_per_spot')
    gamma = scalar_risk(report, 'gamma.raw', underlying_id=contract.underlying_id,
                        unit='currency_per_spot_squared', required=False)
    return delta, gamma
