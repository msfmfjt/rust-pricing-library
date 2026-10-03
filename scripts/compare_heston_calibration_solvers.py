"""Independent OPTIMIZER comparison on the same finite-grid pricing objective.

SciPy TRF uses the public residual/Jacobian evaluation, hence this is NOT an
independent model price reference. Production calibration never imports SciPy.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import least_squares
import rust_pricing as rp

P=dict(hurst=.1,initial_variance=.04,mean_reversion=.7,long_run_variance=.055,vol_of_vol=.18,correlation=-.65)
CFG=dict(time_steps=64,integration_intervals=256,cutoff=96.)
BOUNDS=[('initial_variance',.01,.1,.04),('mean_reversion',.1,2.,.7),('long_run_variance',.015,.12,.05),
        ('vol_of_vol',.03,.4,.2),('correlation',-.95,-.05,.5),('hurst',.02,.49,.2)]
def main():
    truth=rp.RoughVolatilityModel.rough_heston(**P);quotes=[]
    for t in [.25,.75,1.5]:
        plan=rp.HestonFourierPlan.compile(truth,t,**CFG)
        for k in [85.,100.,115.]:quotes.append(rp.HestonCalibrationQuote.create(t,100.,k,.97,plan.price(100.,k,.97).call,.1))
    initial=rp.RoughVolatilityModel.rough_heston(**(P|dict(initial_variance=.045,mean_reversion=.85,long_run_variance=.06,vol_of_vol=.21,correlation=-.55,hurst=.16)))
    problem=rp.HestonCalibrationProblem.compile(initial,quotes,[rp.HestonCalibrationVariable.create(*r) for r in BOUNDS],**CFG)
    ours=problem.calibrate(residual_tolerance=5e-5,max_evaluations=102)
    cache={}
    def evaluate(x):
        key=tuple(x)
        if key!=cache.get('key'):
            cache['value']=problem.evaluate(x);cache['key']=key
        return cache['value']
    independent=least_squares(lambda x:evaluate(x).scaled_residuals,problem.initial_parameters,
        jac=lambda x:evaluate(x).jacobian,bounds=([r[1] for r in BOUNDS],[r[2] for r in BOUNDS]),
        x_scale=[r[3] for r in BOUNDS],method='trf',gtol=1e-10,ftol=1e-12,xtol=1e-12,max_nfev=102)
    theirs=problem.evaluate(independent.x)
    assert ours.fit_achieved and independent.success
    assert max(map(abs,theirs.scaled_residuals))<5e-5
    gap=max(abs(a-b) for a,b in zip(ours.evaluation.model_prices,theirs.model_prices))
    assert gap<1e-5
    print('INDEPENDENT_SOLVER',dict(ours_evaluations=ours.evaluations,scipy_nfev=independent.nfev,
        ours_objective=ours.objective,scipy_objective=float(independent.cost),max_price_gap=gap,
        max_parameter_gap=float(np.max(np.abs(np.array(ours.parameters)-independent.x)))))
if __name__=='__main__':main()
