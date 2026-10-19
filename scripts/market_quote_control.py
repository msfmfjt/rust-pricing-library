"""Diagnostic fixed-coefficient market-quote control for paired VegaKT samples."""
from copy import deepcopy
import math
from pathlib import Path
import numpy as np

from check_portfolio_holdout import sha
from rust_pricing_backtest.domain import require,fingerprint
from rust_pricing_backtest.discrete_smile import _black
from rust_pricing_backtest.multi_step_extension import MultiStepExtensionCalibrator
from rust_pricing_backtest.portfolio_market_risk import check_report,sampling_covariance


def apply_control(native, requests, policy):
    check_report(native,requests)
    ids=native['instrument_order'];control=policy['control_instrument'];index=ids.index(control)
    require(policy['coefficient_for_every_instrument']==1. and policy['coefficient_fitted_from_samples'] is False,
        'quote_control_policy','fixed unit coefficient required')
    require(native['extension']=='all_rows' and native['variance_basis'] is None,
        'quote_control_scope','all_rows calibration required')
    for cid,p in requests.items():
        product=p['product'];market=p['market']
        require(product['type']=='european_vanilla' and product['side']=={'type':'call'} and product['notional']==1.
            and product['expiry']==requests[control]['product']['expiry']
            and not market.get('discrete_dividends')
            and all(v==1. for name in ('discount_curve','dividend_curve') for v in market[name]['discount_factors']),
            'quote_control_scope','same-expiry unit calls and zero rates/dividends required')
        audit=native['audits'][cid]
        require(audit['status']=='path_gamma_chain_passed' and audit['recovery_passed'] and audit['validation_passed'],
            'quote_control_native','every original native gate remains mandatory')
    base=requests[control];c=MultiStepExtensionCalibrator(base,extension='all_rows')
    tangent,ta=c.all_maturity_iv_tangent()
    require(np.array_equal(tangent,np.array(native['calibration_tangent'])) and ta['iv_response_residual']<=1e-9,
        'quote_control_calibration','same checked market calibration tangent required')
    from datetime import date
    maturity=(date.fromisoformat(base['product']['expiry'])-date.fromisoformat(base['valuation_date'])).days/365.
    require(maturity==c.maturities[-1], 'quote_control_scope','terminal reporting maturity required')
    matches=np.flatnonzero(c.strikes==base['product']['strike'])
    require(len(matches)==1,'quote_control_scope','control must exactly equal a calibrated absolute quote strike')
    strike_index=int(matches[0]);coordinate=c.iv.size-len(c.strikes)+strike_index
    iv=c.iv[-1,strike_index];known=np.zeros(c.iv.size)
    known[coordinate]=float(_black(c.spot,c.strikes[strike_index],maturity*iv*iv)[1]*2*maturity*iv)
    raw=np.asarray(native['sampling_unit_values']);count,n,width=raw.shape;q=width-1
    require((n,q,count)==(len(ids),c.iv.size,native['independent_sampling_units'])
        and count>=2 and np.isfinite(raw).all(),'quote_control_samples','complete paired scramble samples required')
    adjusted=raw.copy();adjusted[:,:,1:]=raw[:,:,1:]-raw[:,index:index+1,1:]+known[None,None,:]
    transform=np.eye(n*width)
    for i in range(n):
        for k in range(1,width):transform[i*width+k,index*width+k]-=1.
    covariance=sampling_covariance(adjusted.reshape(count,-1))
    projection=transform@np.array(native['price_and_market_sample_covariance'])@transform.T
    projection_error=float(np.max(abs(covariance-projection)))
    require(np.allclose(covariance,projection,rtol=1e-9,atol=1e-8),
        'quote_control_covariance','control transform must reproduce direct paired-sample covariance')
    result=deepcopy(native);result.pop('report_hash')
    result['market_risk_estimator']='fixed_market_quote_control_v1'
    result['control_variate']=dict(policy=deepcopy(policy),control_instrument=control,coefficient=1.,
        expected_gradient=known.tolist(),coordinate=coordinate,
        source_native_report_hash=native['report_hash'],estimator_source_sha256=sha(Path(__file__)),
        iv_response_residual=ta['iv_response_residual'],covariance_projection_error=projection_error,
        expectation_scope='declared calibration quote identity; calibration/discretization error excluded from sampling SE',
        native_audits='unchanged raw Gamma/AD audits; controlled estimates stored separately')
    result['sampling_unit_values']=adjusted.tolist()
    for i,cid in enumerate(ids):
        result['instruments'][cid]['buckets']=adjusted[:,i,1:].mean(axis=0).tolist()
        result['instruments'][cid]['bucket_standard_errors']=(adjusted[:,i,1:].std(axis=0,ddof=1)/math.sqrt(count)).tolist()
    joint=sampling_covariance(adjusted[:,:,1:].reshape(count,-1))
    result['market_sample_covariance']=joint.tolist();result['market_mean_covariance']=(joint/count).tolist()
    result['price_and_market_sample_covariance']=covariance.tolist()
    weights=np.array([native['quantities'][cid] for cid in ids]);net=np.einsum('niq,i->nq',adjusted,weights)
    net_cov=sampling_covariance(net)/count
    result['net_price']=float(net[:,0].mean());result['net_buckets']=net[:,1:].mean(axis=0).tolist()
    result['net_price_and_market_mean_covariance']=net_cov.tolist()
    result['net_bucket_standard_errors']=np.sqrt(np.maximum(0,np.diag(net_cov)[1:])).tolist()
    require(np.array_equal(adjusted[:,:,0],raw[:,:,0]),'quote_control_price','price estimator must stay unchanged')
    result['report_hash']=fingerprint(result)
    return result
