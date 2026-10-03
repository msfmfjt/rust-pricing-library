"""Synthetic price fit; real data must supply consistent F, D and expiry years.

Only initial variance, vol-of-vol and correlation are fitted here. All other
parameters stay fixed. Model prices are synthetic, not an independent oracle.
"""
import rust_pricing as rp

def main():
    parameters=dict(hurst=.1,initial_variance=.04,mean_reversion=.7,
                    long_run_variance=.055,vol_of_vol=.18,correlation=-.65)
    config=dict(time_steps=64,integration_intervals=256,cutoff=96.)
    truth=rp.RoughVolatilityModel.rough_heston(**parameters)
    quotes=[]
    for t in [.25,.75,1.5]:
        plan=rp.HestonFourierPlan.compile(truth,t,**config)
        for k in [85.,100.,115.]:
            price=plan.price(100.,k,.97).call
            quotes.append(rp.HestonCalibrationQuote.create(t,100.,k,.97,price,price_scale=1.))
    initial=rp.RoughVolatilityModel.rough_heston(**(parameters|dict(initial_variance=.045,vol_of_vol=.21,correlation=-.55)))
    variables=[rp.HestonCalibrationVariable.create(*row) for row in [
        ('initial_variance',.01,.1,.04),('vol_of_vol',.03,.4,.2),('correlation',-.95,-.05,.5)]]
    problem=rp.HestonCalibrationProblem.compile(initial,quotes,variables,**config)
    result=problem.calibrate(residual_tolerance=1e-5,max_evaluations=80)
    print('Termination:',result.termination,'Price fit:',result.fit_achieved)
    print(dict(zip(result.parameter_names,result.parameters)))
    print('Maximum scaled residual:',max(map(abs,result.evaluation.scaled_residuals)))
    assert result.fit_achieved
    # Reprice the fitted model on a finer grid. A small shift is a diagnostic,
    # not proof of continuum accuracy, parameter identification or market fit.
    for q in quotes:
        p=rp.HestonFourierPlan.compile(result.model,q.maturity,time_steps=128,
                                       integration_intervals=512,cutoff=128.)
        assert abs(p.price(q.forward,q.strike,q.discount).call-q.target_price)<.003

if __name__=='__main__':main()
