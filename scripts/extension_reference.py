"""Four independent conditional path prices sharing scrambled Sobol states.

No native price, AD, Gamma recovery or calibration derivative is used. The
formulas are the declared fixture payoffs, with the last lognormal shock
integrated analytically. This oracle covers call-side fixtures only.
"""
import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc
from multi_step_reference import black

PRODUCTS=('call','asian','fixed_asian','lookback')


def path_prices(payload,*,power=16,scrambles=8):
    g=payload['model']['local_variance_grid'];t=np.array(g['time_nodes']);nodes=np.array(g['log_forward_moneyness_nodes'])
    rows=np.array(g['values']).reshape(len(t),-1);n=len(t)-1;spot=payload['market']['spot'];batches=[]
    for seed in range(7001,7001+scrambles):
        normals=ndtri(qmc.Sobol(n-1,scramble=True,seed=seed).random_base2(power));x=np.zeros(len(normals));history=[]
        for row in range(n-1):
            variance=np.interp(x,nodes,rows[row])*(t[row+1]-t[row])
            x+=-.5*variance+np.sqrt(variance)*normals[:,row];history.append(spot*np.exp(x))
        s=history[-1];variance=np.interp(x,nodes,rows[n-1])*(t[n]-t[n-1]);total=np.sum(history,axis=0);maximum=np.max(history,axis=0)
        strikes=[np.full_like(s,100.),n*100.-total,(n+1)*100.-98.-total,np.maximum(maximum,100.)]
        factors=[1.,1/n,1/(n+1),1.];cash=[0.,0.,0.,np.maximum(maximum-100.,0.)];values=[]
        for strike,factor,constant in zip(strikes,factors,cash):
            positive=strike>0.;value=black(s,np.where(positive,strike,1.),variance)[0]
            values.append(np.mean(factor*np.where(positive,value,s-strike)+constant))
        batches.append(values)
    return np.asarray(batches)
