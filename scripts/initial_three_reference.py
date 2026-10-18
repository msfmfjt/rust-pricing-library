"""Independent three-step initial/middle Gamma via analytic continuation Delta.

Uses analytic continuation Delta (including local-variance slope and moving
payoff strikes), rather than native slope-jump reconstruction or Gaussian score.
Only the four declared validation payoffs are covered; this is not product code.
"""
import math
import numpy as np
from scipy.special import ndtr
from check_three_step_gamma import rule


def reference(payload, kind, order=64, strike=100.):
    g = payload['model']['local_variance_grid']
    times = np.array(g['time_nodes'])
    nodes = np.array(g['log_forward_moneyness_nodes'])
    variances = np.array(g['values']).reshape(4,-1)
    spot = payload['market']['spot']
    v0 = variances[0,0]*times[1]
    sd0 = math.sqrt(v0)
    dt2, dt3 = np.diff(times)[1:]
    lo, hi = -v0/2-12*sd0, -v0/2+12*sd0
    x1, w1 = rule([lo,hi]+[x for x in [*nodes,math.log(strike/spot)] if lo<x<hi],order)
    outer_dx=w1.copy()
    w1 *= np.exp(-.5*((x1+v0/2)/sd0)**2)/(sd0*math.sqrt(2*math.pi))
    price = 0.
    middle, terminal = np.zeros(len(nodes)), np.zeros(len(nodes))
    gl = np.polynomial.legendre.leggauss(order)
    initial=0.;endpoint_deltas=[]
    x1=np.r_[x1,lo,hi];w1=np.r_[w1,0.,0.];outer_dx=np.r_[outer_dx,0.,0.]
    for index,(x,history_weight,outer_weight) in enumerate(zip(x1,w1,outer_dx)):
        s1 = spot*math.exp(x)
        v2 = np.interp(x,nodes,variances[1])*dt2
        sd2 = math.sqrt(v2)
        mu = x-v2/2
        low, high = mu-12*sd2, mu+12*sd2
        extra = []
        if kind=='lookback':
            extra = [math.log(max(strike,s1)/spot)]
        elif kind in ['asian','fixed_asian']:
            threshold = 3*strike-s1 if kind=='asian' else 4*strike-98-s1
            if threshold>0: extra = [math.log(threshold/spot)]
        cuts = np.array(sorted(set([low,high]+[y for y in [*nodes,*extra] if low<y<high])))
        z, w = gl
        y = ((cuts[:-1]+cuts[1:])[:,None]/2+np.diff(cuts)[:,None]*z/2).ravel()
        dx = (np.diff(cuts)[:,None]*w/2).ravel()

        def continuation(y):
            s2 = spot*np.exp(y)
            left = np.clip(np.searchsorted(nodes,y,side='right')-1,0,len(nodes)-2)
            annual_slope = (variances[2,left+1]-variances[2,left])/(nodes[left+1]-nodes[left])
            annual_slope = np.where((y>nodes[0])&(y<nodes[-1]),annual_slope,0.)
            sd3 = np.sqrt(np.interp(y,nodes,variances[2])*dt3)
            if kind=='call':
                k, kp, q, constant, cp = np.full_like(s2,strike),0.,1.,0.,0.
            elif kind in ['asian','fixed_asian']:
                k = (3*strike-s1-s2) if kind=='asian' else (4*strike-98-s1-s2)
                kp, q, constant, cp = -1.,(1/3 if kind=='asian' else .25),0.,0.
            elif kind=='lookback':
                k = np.maximum(max(strike,s1),s2)
                kp = (s2>max(strike,s1)).astype(float)
                q, constant, cp = 1.,np.maximum(np.maximum(s1,s2)-strike,0.),kp
            else:
                raise ValueError('unsupported reference payoff')
            valid = k>0
            d1 = np.zeros_like(k)
            d1[valid] = np.log(s2[valid]/k[valid])/sd3[valid]+sd3[valid]/2
            pdf = np.exp(-d1*d1/2)/math.sqrt(2*math.pi)
            calls = np.where(valid,s2*ndtr(d1)-k*ndtr(d1-sd3),s2-k)
            delta = np.where(valid,ndtr(d1)-kp*ndtr(d1-sd3)+pdf*annual_slope*dt3/(2*sd3),1-kp)
            b3 = np.where(valid,q*s2*pdf*dt3/(2*sd3),0.)
            if kind=='call':history_delta=np.zeros_like(s2)
            elif kind in ['asian','fixed_asian']:history_delta=q*np.where(valid,ndtr(d1-sd3),1.)
            else:history_delta=np.where((s1>strike)&(s1>s2),1-ndtr(d1-sd3),0.)
            return q*calls+constant,q*delta+cp,b3,left,history_delta

        values, delta, b3, left, history_delta = continuation(y)
        density = np.exp(-.5*((y-mu)/sd2)**2)/(sd2*math.sqrt(2*math.pi))
        weights = dx*density*history_weight
        price += float(weights@values)
        fraction = np.clip((y-nodes[left])/(nodes[left+1]-nodes[left]),0.,1.)
        terminal += np.bincount(left,weights*b3*(1-fraction),minlength=len(nodes))
        terminal += np.bincount(left+1,weights*b3*fraction,minlength=len(nodes))
        # Integral K(S2) dG'(S2) = [K G'] - integral K_x G' dx2.
        def kernel(y):
            d1 = (x-y)/sd2+sd2/2
            k = s1*np.exp(-d1*d1/2)/math.sqrt(2*math.pi)*dt2/(2*sd2)
            return k,k*d1/sd2
        _, kernel_derivative = kernel(y)
        ends = np.array([low,high])
        boundary = kernel(ends)[0]*continuation(ends)[1]
        risk = boundary[1]-boundary[0]-dx@(kernel_derivative*delta)
        j = int(np.clip(np.searchsorted(nodes,x,side='right')-1,0,len(nodes)-2))
        h = np.clip((x-nodes[j])/(nodes[j+1]-nodes[j]),0.,1.)
        middle[j] += history_weight*risk*(1-h)
        middle[j+1] += history_weight*risk*h
        # Differentiate the second transition with respect to its initial spot,
        # including v2(log S1), and differentiate the payoff's first fixing.
        slope=(variances[1,j+1]-variances[1,j])/(nodes[j+1]-nodes[j]) if nodes[0]<x<nodes[-1] else 0.
        normal=(y-mu)/sd2
        log_density_delta=(normal/sd2+slope*dt2*((normal*normal-1)/(2*v2)-normal/(2*sd2)))/s1
        gprime=float((dx*density)@(history_delta+values*log_density_delta))
        d1=-x/sd0+sd0/2
        initial_kernel=spot*np.exp(-d1*d1/2)/math.sqrt(2*math.pi)*times[1]/(2*sd0)
        initial-=outer_weight*initial_kernel*d1/sd0*gprime
        if index>=len(x1)-2:endpoint_deltas.append(initial_kernel*gprime)
    initial+=endpoint_deltas[1]-endpoint_deltas[0]
    return dict(price=price,initial_variance_sensitivity=initial,middle_variance_sensitivities=middle.tolist(),
                terminal_variance_sensitivities=terminal.tolist())
