"""Independent validation: dense density quadrature and scrambled path integration.

Neither native adjoints nor the companion calibration Jacobian supplies the
reference risks. Sobol integration conditions the last payoff transition out.
"""
import math
import numpy as np
from scipy.special import ndtr,ndtri
from scipy.stats import qmc


def black(s,k,v):
    sd=np.sqrt(v);d1=np.log(s/k)/sd+sd/2
    return s*ndtr(d1)-k*ndtr(d1-sd),s*np.exp(-d1*d1/2)/(2*np.sqrt(2*np.pi)*sd)


def density_reference(p,order=96):
    g=p['model']['local_variance_grid'];t=np.array(g['time_nodes']);xnodes=np.array(g['log_forward_moneyness_nodes']);rows=np.array(g['values']).reshape(len(t),-1);spot=p['market']['spot']
    first=np.interp(0.,xnodes,rows[0])*t[1];total=first;states=[];z,w=np.polynomial.legendre.leggauss(order)
    for row in range(len(t)-2):
        if row==0:lo,hi=-first/2-12*np.sqrt(first),-first/2+12*np.sqrt(first)
        else:
            total+=np.max(rows[row])*(t[row+1]-t[row]);lo,hi=-total/2-12*np.sqrt(total),total/2+12*np.sqrt(total)
        cuts=np.r_[lo,xnodes[(lo<xnodes)&(xnodes<hi)],hi]
        y=((cuts[:-1]+cuts[1:])[:,None]/2+np.diff(cuts)[:,None]*z/2).ravel();dy=(np.diff(cuts)[:,None]*w/2).ravel()
        if row==0:density=np.exp(-.5*((y+first/2)/np.sqrt(first))**2)/np.sqrt(2*np.pi*first)
        else:
            v=np.interp(x,xnodes,rows[row])*(t[row+1]-t[row]);density=np.empty_like(y)
            for a in range(0,len(y),128):
                kernel=np.exp(-.5*((y[a:a+128,None]-x+v/2)/np.sqrt(v))**2)/np.sqrt(2*np.pi*v)
                density[a:a+128]=kernel@probabilities
        x,probabilities=y,dy*density;states.append((x.copy(),probabilities.copy()))
    s=spot*np.exp(x);hats=np.array([np.interp(x,xnodes,e) for e in np.eye(len(xnodes))]).T
    dt=t[-1]-t[-2];strikes=spot*np.exp(xnodes)
    _,d=black(s[:,None],strikes[None,:],(hats@rows[-2])[:,None]*dt)
    operator=(hats.T*probabilities)@(d*dt)
    return dict(states=states,operator=operator)


def path_reference(p,kind,*,power=16,scrambles=8):
    g=p['model']['local_variance_grid'];t=np.array(g['time_nodes']);nodes=np.array(g['log_forward_moneyness_nodes']);rows=np.array(g['values']).reshape(len(t),-1)
    count=len(t)-1;spot=p['market']['spot'];results=[]
    for seed in range(7001,7001+scrambles):
        normals=ndtri(qmc.Sobol(count-1,scramble=True,seed=seed).random_base2(power));x=np.zeros(len(normals));history=[]
        for row in range(count-1):
            v=np.interp(x,nodes,rows[row])*(t[row+1]-t[row]);x+=-.5*v+np.sqrt(v)*normals[:,row];history.append(spot*np.exp(x))
        s=history[-1];v=np.interp(x,nodes,rows[-2])*(t[-1]-t[-2])
        if kind=='call':strike=np.full(len(s),100.);factor=1.;constant=0.
        elif kind in ['asian','fixed_asian']:
            known=98. if kind=='fixed_asian' else 0.;divisor=count+(kind=='fixed_asian')
            strike=divisor*100-known-np.sum(history,axis=0);factor=1/divisor;constant=0.
        elif kind=='lookback':
            maximum=np.max(history,axis=0);strike=np.maximum(maximum,100.);factor=1.;constant=np.maximum(maximum-100.,0.)
        else:raise ValueError('unsupported payoff reference')
        valid=strike>0;safe=np.where(valid,strike,1.);calls,vegas=black(s,safe,v)
        values=factor*np.where(valid,calls,s-strike)+constant
        derivative=factor*np.where(valid,vegas,0.)*(t[-1]-t[-2])
        j=np.clip(np.searchsorted(nodes,x,side='right')-1,0,len(nodes)-2);h=np.clip((x-nodes[j])/(nodes[j+1]-nodes[j]),0.,1.)
        risk=(np.bincount(j,derivative*(1-h),minlength=len(nodes))+np.bincount(j+1,derivative*h,minlength=len(nodes)))/len(x)
        results.append(np.r_[np.mean(values),risk])
    a=np.array(results)
    return dict(method='independent_scrambled_Sobol_with_analytic_terminal_payoff',power=power,scrambles=scrambles,batches=a.tolist(),mean=a.mean(axis=0).tolist(),standard_error=(a.std(axis=0,ddof=1)/np.sqrt(scrambles)).tolist())
