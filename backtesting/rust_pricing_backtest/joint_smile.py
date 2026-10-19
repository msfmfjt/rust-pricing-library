"""Research-only coupled market-IV Gamma diagnostic for two time intervals.

The first IV is spatially flat. Terminal quotes define independent calibration
constraints; the dependent first-IV compensation is a readout, not another
constraint. No recovered Gamma or exotic price is used to choose the tangent.
"""
from __future__ import annotations

import json
import numpy as np
from .discrete_smile import TwoStepSmileCalibrator
from .domain import fingerprint


def evaluate_joint_market_gamma(payload, gamma_bin_boundaries, *, policy,
                                subdivisions=32768, relative_tolerance=1e-8,
                                worker_threads=1):
    import rust_pricing as rp
    calibrator = TwoStepSmileCalibrator(payload)
    tangent, tangent_audit = calibrator.joint_iv_tangent()
    plan = rp.PricingPlan.compile(
        rp.PricingRequest.from_json(json.dumps(calibrator.payload, allow_nan=False)),
        worker_threads=worker_threads, reduction_block_size=256)
    audit = plan.evaluate_vega_kt_joint_gamma_audit(
        (calibrator.spot*np.exp(calibrator.nodes)).tolist(), gamma_bin_boundaries,
        policy=rp.VegaKtGammaRecoveryPolicy(**policy),
        variance_conservation_directions=tangent[1:, 1:].T.tolist(),
        risk_directions=tangent.T.tolist(), subdivisions=subdivisions,
        relative_tolerance=relative_tolerance)
    return dict(document_kind='two_step_joint_market_iv_gamma', schema_version=1,
        diagnostic_only=True, status=audit['status'], estimate=audit['estimate'],
        request_hash=fingerprint(calibrator.payload),
        coordinate_order=tangent_audit['coordinate_order'], iv_unit='absolute_decimal',
        calibration_tangent=tangent.tolist(), tangent_audit=tangent_audit, audit=audit,
        full_market_iv_bucket_acceptance='not_established',
        limitations=['Exactly two intervals; one parallel first-maturity IV and terminal quote IVs.',
            'First-maturity individual strike shifts and general multistep smiles are unsupported.',
            'Minimum-norm annual-variance calibration extension is a model choice.',
            'Covariance is conditional on the frozen pilot choice; deterministic initial Gamma error is separate.'])
