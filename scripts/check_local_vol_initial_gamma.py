#!/usr/bin/env python3
"""Independent conditional price stencils and first-interval Vega references.

Decimal forward replay, two-pass moments, and Gaussian power-series integration.
No native continuation, finite-difference implementation, or reduction is used.
"""
import argparse
import json
from decimal import Decimal as D, localcontext
from pathlib import Path
from check_local_vol_step_adjoints import weights
from check_local_vol_gamma_numerics import cdf
from check_local_vol_step_recovery import stats

FIXTURE=Path(__file__).resolve().parents[1]/'fixtures/local-vol/initial-gamma-v0.1.json'


def payoff(case,first,shocks):
    times=list(map(D,case['times'])); grid_times=list(map(D,case['grid_times']))
    variances=list(map(D,case['row_variances']))
    states=[D(case['spot']),first]
    for i,shock in enumerate(shocks,1):
        ti,tw=weights(grid_times,times[i]);a=(1-tw)*variances[ti]+tw*variances[ti+1]
        dt=D(case['times'][i+1]-case['times'][i])
        states.append(states[-1]*(-a*dt/2+a.sqrt()*dt.sqrt()*D(shock)).exp())
    return D(case['discount'])*(D(case['quadratic'])*states[-1]**2+D(case['sextic'])*states[-1]**6
        +D(case['cross'])*states[1]*states[3]+D('.2')*states[1]+7)


def kernel(case,gamma):
    spot=D(case['spot']);v=D(case['row_variances'][0])*D(case['times'][1]);sd=v.sqrt()
    xs=list(map(D,case['strikes']));bounds=[D(0),*xs,D('Infinity')]
    probs=[];moments=[]
    for lo,hi in zip(bounds,bounds[1:]):
        zl=(lo/spot).ln()/sd-D('1.5')*sd if lo else D('-Infinity')
        zh=(hi/spot).ln()/sd-D('1.5')*sd
        probs.append(cdf(zh)-cdf(zl))
        moments.append(spot*(2*v).exp()*(cdf(zh-sd)-cdf(zl-sd)))
    left=gamma[0]*probs[0];right=gamma[-1]*probs[-1];value=left+right
    for i,(lo,hi) in enumerate(zip(xs,xs[1:])):
        slope=(gamma[i+1]-gamma[i])/(hi-lo)
        value+=(gamma[i]-slope*lo)*probs[i+1]+slope*moments[i+1]
    scale=spot**2*D(case['row_variances'][0]).sqrt()*D(case['times'][1])*v.exp()
    return value*scale,left*scale,right*scale,[float(probs[0]),float(probs[-1])]


def sample(case,shocks):
    coarse=[];fine=[];row=[]
    for strike in case['strikes']:
        h=strike*case['relative_bump']
        points=[strike-2*h,strike-h,strike-.5*h,strike,strike+.5*h,strike+h,strike+2*h]
        p=[payoff(case,D(x),shocks) for x in points]
        c=(16*(p[1]+p[5])-30*p[3]-p[0]-p[6])/(12*D(h)**2)
        f=(16*(p[2]+p[4])-30*p[3]-p[1]-p[5])/(12*D(.5*h)**2)
        coarse.append(c);fine.append(f);row.extend([c,f,f-c])
    cv,_,_,_=kernel(case,coarse);fv,left,right,masses=kernel(case,fine)
    return row+[cv,fv,fv-cv,left,right],masses


def artifact():
    case=dict(spot=100.,times=[0.,.1,.23,.4,.7,1.],grid_times=[0.,.4,1.],
              row_variances=[.04,.055,.035],strikes=[70.,85.,100.,115.,140.],relative_bump=.04,
              discount=.97,quadratic=.005,sextic=1e-10,cross=-.002)
    shocks=[[-.6,.8,-.2,.4],[.1,-.8,1.2,.6],[-.2,.1,-.7,-.5]]
    primary=[sample(case,g)[0] for g in shocks]
    mates=[sample(case,[-x for x in g])[0] for g in shocks]
    pairs=[[(a+b)/2 for a,b in zip(left,right)] for left,right in zip(primary,mates)]
    estimates=[[stats([s[i] for s in rows]) for i in range(len(rows[0]))] for rows in [primary,pairs]]
    return dict(fixture_version=1,reference='100-digit conditional forward price stencils; independent two-pass moments and Gaussian integration',
                relative_tolerance=1e-8,difference_absolute_tolerance=1e-9,
                case=case,shocks=shocks,samples=[[float(v) for v in row] for row in primary],
                estimates=estimates,kernel_tail_masses=sample(case,shocks[0])[1])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--write',action='store_true');args=p.parse_args()
    with localcontext() as context:
        context.prec=100
        payload=json.dumps(artifact(),indent=2,allow_nan=False)+'\n'
    if args.write:FIXTURE.write_text(payload,encoding='utf-8')
    elif FIXTURE.read_text(encoding='utf-8')!=payload:raise AssertionError('Initial Gamma references changed')
    print('Conditional Gamma and initial-Vega moments verified for three paths and three pairs')


if __name__=='__main__':main()
