"""Three-step Gamma in all reachable IV coordinates; first IV is parallel only."""
import json
import numpy as np
from .three_step_smile import ThreeStepTerminalSmileCalibrator
from .domain import fingerprint


def evaluate_three_step_market_gamma(payload, gamma_bin_boundaries, *, policy,
        subdivisions=512, relative_tolerance=1e-8, worker_threads=1):
    import rust_pricing as rp
    calibrator=ThreeStepTerminalSmileCalibrator(payload)
    tangent, tangent_audit=calibrator.all_maturity_iv_tangent()
    m=len(calibrator.nodes)
    plan=rp.PricingPlan.compile(rp.PricingRequest.from_json(json.dumps(
        calibrator.payload,allow_nan=False)),worker_threads=worker_threads,reduction_block_size=256)
    initial=plan.evaluate_vega_kt_three_step_initial_gamma_audit(gamma_bin_boundaries,
        subdivisions=subdivisions,relative_tolerance=relative_tolerance)
    middle=plan.evaluate_vega_kt_middle_smile_gamma_audit(gamma_bin_boundaries,
        subdivisions=subdivisions,relative_tolerance=relative_tolerance)
    terminal=plan.evaluate_vega_kt_three_step_terminal_readout_audit(
        (calibrator.spot*np.exp(calibrator.nodes)).tolist(),
        policy=rp.VegaKtGammaRecoveryPolicy(**policy),
        variance_conservation_directions=tangent_audit['terminal_conservation_basis'],
        variance_risk_directions=tangent[1+m:].T.tolist())
    estimate=None;status='all_reachable_market_gamma_passed'
    for name,audit in [('initial',initial),('middle',middle),('terminal',terminal)]:
        if audit['estimate'] is None:
            status=name+'_'+audit['status'];break
    allowance=abs(tangent[0])*initial['functional_tolerance']+np.sum(abs(tangent[1:1+m]),axis=0)*middle['functional_tolerance']
    if status=='all_reachable_market_gamma_passed':
        fixed=tangent[0]*initial['estimate']+tangent[1:1+m].T@middle['estimate']
        estimate=[dict(value,mean=value['mean']+float(offset)) for value,offset in zip(terminal['estimate'],fixed)]
    return dict(document_kind='three_step_all_reachable_market_gamma',schema_version=1,
        diagnostic_only=True,status=status,estimate=estimate,request_hash=fingerprint(calibrator.payload),
        coordinate_order=tangent_audit['coordinate_order'],iv_unit='absolute_decimal',
        calibration_tangent=tangent.tolist(),tangent_audit=tangent_audit,
        initial_gamma_audit=initial,middle_gamma_audit=middle,terminal_gamma_audit=terminal,
        deterministic_functional_allowance=allowance.tolist(),
        uncertainty_scope='Same-path terminal readout SE conditional on pilot selection; deterministic initial and middle errors separate. No cross-quote covariance reported.',
        full_market_iv_bucket_acceptance='not_established',
        limitations=['Exactly three intervals; first-maturity IV is parallel only.',
            'Finite Gamma windows and refinement diagnostics are not rigorous total error bounds.',
            'Minimum-norm annual-variance calibration increments are a model choice.',
            'Arbitrary step counts, nonflat first maturity and production full-IV acceptance remain unsupported.'])
