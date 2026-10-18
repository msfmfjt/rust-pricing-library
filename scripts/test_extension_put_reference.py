"""Direct payoff integration checks for the independent Put price oracle."""
import math
import unittest
import numpy as np
from scipy.integrate import quad
from extension_put_reference import conditional_prices,request,PRODUCTS
from multi_step_cases import model


class PutReferenceTests(unittest.TestCase):
    def test_final_shock_formulas_against_direct_payoff_integrals(self):
        for history in [[90.,105.,95.],[120.,110.,130.],[30.,250.,140.],[100.]*5]:
            n=len(history)+1;s=history[-1]
            for variance in [.01,.12]:
                actual=conditional_prices(np.array(history)[:,None],np.array([variance]))[0]
                for j,kind in enumerate(PRODUCTS):
                    def payoff(z):
                        last=s*math.exp(-variance/2+math.sqrt(variance)*z)
                        if kind=='put':value=max(100.-last,0.)
                        elif kind=='asian':value=max(100.-(sum(history)+last)/n,0.)
                        elif kind=='fixed_asian':value=max(100.-(98.+sum(history)+last)/(n+1),0.)
                        else:
                            fixed=[] if kind=='lookback' else [98.] if kind=='lookback_past' else [98.,90.]
                            value=max(100.-min([*history,last,*fixed]),0.)
                        return value*math.exp(-z*z/2)/math.sqrt(2*math.pi)
                    thresholds=[100.,n*100.-sum(history),(n+1)*100.-98.-sum(history),*history,98.,90.]
                    points=sorted(set((math.log(k/s)+variance/2)/math.sqrt(variance) for k in thresholds if k>0))
                    value,error=quad(payoff,-12.,12.,points=[x for x in points if -12<x<12],epsabs=1e-11,epsrel=1e-11)
                    with self.subTest(history=history,variance=variance,product=kind):
                        self.assertLess(abs(actual[j]-value),1e-10)

    def test_requests_preserve_markets_and_use_fixed_minima(self):
        from copy import deepcopy
        p=model(4);original=deepcopy(p)
        for kind in PRODUCTS:
            q=request(p,kind);self.assertEqual(q['product']['side'],{'type':'put'})
            self.assertEqual(q['model'],p['model']);self.assertEqual(q['market'],p['market'])
        q=request(p,'lookback_today')
        self.assertEqual(q['product']['valuation_date_fixing'],90.)
        self.assertEqual(q['product']['historical_extremum'],98.)
        self.assertEqual(p,original)


if __name__=='__main__':unittest.main()
