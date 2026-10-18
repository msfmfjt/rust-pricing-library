"""Diagnostic three-step middle/terminal market Gamma; first marginal frozen."""
import json
import numpy as np
from .three_step_smile import ThreeStepTerminalSmileCalibrator
from .domain import fingerprint


def evaluate_middle_terminal_market_gamma(payload, gamma_bin_boundaries, *, policy,
        subdivisions=512, relative_tolerance=1e-8, worker_threads=1):
    import rust_pricing as rp
    calibrator = ThreeStepTerminalSmileCalibrator(payload)
    tangent, tangent_audit = calibrator.joint_iv_tangent()
    m = len(calibrator.nodes)
    plan = rp.PricingPlan.compile(rp.PricingRequest.from_json(json.dumps(
        calibrator.payload,allow_nan=False)),worker_threads=worker_threads,reduction_block_size=256)
    middle = plan.evaluate_vega_kt_middle_smile_gamma_audit(gamma_bin_boundaries,
        subdivisions=subdivisions,relative_tolerance=relative_tolerance)
    terminal = plan.evaluate_vega_kt_three_step_terminal_readout_audit(
        (calibrator.spot*np.exp(calibrator.nodes)).tolist(),
        policy=rp.VegaKtGammaRecoveryPolicy(**policy),
        variance_conservation_directions=tangent_audit['terminal_conservation_basis'],
        variance_risk_directions=tangent[m:].T.tolist())
    estimate = None
    status = ('middle_'+middle['status'] if middle['estimate'] is None else
              'terminal_'+terminal['status'] if terminal['estimate'] is None else
              'middle_terminal_market_gamma_passed')
    allowance = (np.sum(abs(tangent[:m]),axis=0)*middle['functional_tolerance']).tolist()
    if middle['estimate'] is not None and terminal['estimate'] is not None:
        fixed = tangent[:m].T@middle['estimate']
        estimate = [dict(value,mean=value['mean']+float(offset))
                    for value,offset in zip(terminal['estimate'],fixed)]
    return dict(document_kind='three_step_middle_terminal_market_gamma',schema_version=1,
        diagnostic_only=True,status=status,estimate=estimate,
        request_hash=fingerprint(calibrator.payload),first_variance_row='fixed',
        coordinate_order=tangent_audit['coordinate_order'],iv_unit='absolute_decimal',
        calibration_tangent=tangent.tolist(),tangent_audit=tangent_audit,
        middle_gamma_audit=middle,terminal_gamma_audit=terminal,
        deterministic_middle_functional_allowance=allowance,
        uncertainty_scope='Each SE is a same-path terminal readout; deterministic middle Gamma error is separate. No cross-quote covariance reported.',
        full_market_iv_bucket_acceptance='not_established',
        limitations=['Exactly three intervals; flat first marginal held fixed.',
            'Middle Gamma measures are conditional on first-state history and finite windows.',
            'Minimum-norm annual-variance calibration extension is a model choice.',
            'Refinement allowances are convergence diagnostics, not rigorous total error bounds.',
            'First-maturity Gamma and arbitrary step counts are unsupported.'])
