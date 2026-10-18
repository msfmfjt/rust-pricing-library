#!/usr/bin/env python3
"""Independent 100-digit price differences -> Gamma -> finite-strike IV chain.

No native pricing, quadrature, AD or moment code is used. Gaussian probabilities
use a Decimal power series; moments use a two-pass independent reduction.
"""
import argparse
import json
from decimal import Decimal as D, localcontext
from pathlib import Path
from check_local_vol_gamma_numerics import cdf, PI
from check_local_vol_step_adjoints import weights
from check_local_vol_step_recovery import stats

FIXTURE = Path(__file__).resolve().parents[1] / 'fixtures/local-vol/vegakt-chain-v0.1.json'


def pdf(x):
    return (-x*x/2).exp()/(2*PI).sqrt()


def transition(s, v, ks, gs):
    sd=v.sqrt(); bounds=[D(0),*ks,D('Infinity')]
    probabilities=[]; moments=[]
    for lo,hi in zip(bounds,bounds[1:]):
        zl=(lo/s).ln()/sd-D('1.5')*sd if lo else D('-Infinity')
        zh=(hi/s).ln()/sd-D('1.5')*sd
        probabilities.append(cdf(zh)-cdf(zl))
        moments.append(s*(2*v).exp()*(cdf(zh-sd)-cdf(zl-sd)))
    tail=gs[0]*probabilities[0]+gs[-1]*probabilities[-1]
    total=tail
    for i,(lo,hi) in enumerate(zip(ks,ks[1:])):
        slope=(gs[i+1]-gs[i])/(hi-lo)
        total+=(gs[i]-slope*lo)*probabilities[i+1]+slope*moments[i+1]
    return total*v.exp(),tail*v.exp(),[probabilities[0]*gs[0]*v.exp(),probabilities[-1]*gs[-1]*v.exp()]


def price(s):
    return s*s/2+D('1e-10')*s**6


def path(c,z,vols,first=None):
    states=[D(c['spot'])] if first is None else [D(c['spot']),first]
    for i in range(0 if first is None else 1,len(z)):
        dt=D(c['times'][i+1]-c['times'][i])
        states.append(states[-1]*(-vols[i]**2*dt/2+vols[i]*dt.sqrt()*D(z[i])).exp())
    return states


def project(c,time,ks,gs,q):
    s=D(c['spot']);xs=list(map(D,c['report_nodes']));sigma=D(c['variance']).sqrt();time=D(time)
    knots=sorted(set(ks+[s*x.exp() for x in xs if ks[0]<s*x.exp()<ks[-1]]))
    out=[D(0)]*len(xs);edges=[D(0),D(0)]
    for lo,hi in zip(knots,knots[1:]):
        h=(hi-lo)/q
        for j in range(q):
            k=lo+(D(j)+D('.5'))*h
            i,w=weights(ks,k);density=(1-w)*gs[i]+w*gs[i+1]
            x=(k/s).ln();sd=sigma*time.sqrt();vega=s*pdf(-x/sd+sd/2)*time.sqrt()
            risk=-h*density*vega
            if x<xs[0]:edges[0]+=risk
            if x>xs[-1]:edges[1]+=risk
            i,w=weights(xs,x);out[i]+=(1-w)*risk;out[i+1]+=w*risk
    return out,edges


def sample(c,z,weak=False):
    sigma=D(c['variance']).sqrt();vols=[sigma]*len(z);s=path(c,z,vols)
    xnodes=list(map(D,c['nodes']));knodes=[D(c['spot'])*x.exp() for x in xnodes]
    coarse=[];fine=[]
    for strike in c['initial_strikes']:
        h=strike*c['relative_bump'];p=[price(path(c,z,vols,D(k))[-1]) for k in
            [strike-2*h,strike-h,strike-.5*h,strike,strike+.5*h,strike+h,strike+2*h]]
        coarse.append((16*(p[1]+p[5])-30*p[3]-p[0]-p[6])/(12*D(h)**2))
        fine.append((16*(p[2]+p[4])-30*p[3]-p[1]-p[5])/(12*D(.5*h)**2))
    gammas=[fine];strikes=[list(map(D,c['initial_strikes']))];native=[]
    for i in range(len(z)):
        h=D('1e-20');up=vols.copy();dn=vols.copy();up[i]+=h;dn[i]-=h
        derivative=(price(path(c,z,up)[-1])-price(path(c,z,dn)[-1]))/(2*h)
        native.append(derivative)
        if i==0:continue
        nodes=[D(0)]*len(xnodes);j,w=weights(xnodes,(s[i]/D(c['spot'])).ln())
        nodes[j]+=(1-w)*derivative;nodes[j+1]+=w*derivative
        t=D(c['times'][i]);dt=D(c['times'][i+1]-c['times'][i]);v=D(c['variance'])*t
        gs=[]
        if weak:
            from check_local_vol_density_mass import masses
            target_masses=masses(D(c['spot']),D(c['variance'])*D(c['times'][i+1]),xnodes)
        for j,(x,k) in enumerate(zip(xnodes,knodes)):
            area=((xnodes[j+1]-x) if j==0 else (x-xnodes[j-1]) if j==len(xnodes)-1 else (xnodes[j+1]-xnodes[j-1]))/2
            d=pdf((x+v/2)/v.sqrt())/(k*v.sqrt())
            gs.append(nodes[j]/(target_masses[j]*sigma*dt if weak else area*k**3*d*sigma*dt))
        gammas.append(gs);strikes.append(knodes)
    raw=[];hedges=[];bumpdiff=[];quaddiff=[];tails=[];edges=[]
    for i,(ks,gs) in enumerate(zip(strikes,gammas)):
        density=[-v for v in gs];tail=[D(0)]*len(ks)
        if i+1<len(gammas):
            v=D(c['variance'])*D(c['times'][i+2]-c['times'][i+1])
            for j,k in enumerate(ks):
                p,tail[j],_=transition(k,v,strikes[i+1],gammas[i+1]);density[j]+=p
        t=c['times'][i+1];q=c['subdivisions'];f,e=project(c,t,ks,density,2*q);r,_=project(c,t,ks,density,q);edges.extend(e)
        bd=project(c,t,ks,[b-a for a,b in zip(fine,coarse)],2*q)[0] if i==0 else [D(0)]*len(f)
        raw.extend(f);quaddiff.extend([a-b for a,b in zip(f,r)]);bumpdiff.extend(bd)
        for risk,x in zip(f,map(D,c['report_nodes'])):
            sd=sigma*D(t).sqrt();vega=D(c['spot'])*pdf(-x/sd+sd/2)*D(t).sqrt();hedges.append(-risk/vega)
        tails.append(sum(project(c,t,ks,tail,2*q)[0]))
    v=D(c['variance'])*D(c['times'][1]);scale=D(c['spot'])**2*sigma*D(c['times'][1])
    initial,_,initialtails=transition(D(c['spot']),v,strikes[0],fine)
    initial_coarse,_,_=transition(D(c['spot']),v,strikes[0],coarse)
    return raw+hedges+bumpdiff+quaddiff+[sum(raw),sum(bumpdiff),sum(quaddiff)]+[D(0)]*(len(z)-1)+[
        scale*initial-native[0],scale*(initial-initial_coarse),*[scale*v for v in initialtails]]+tails+edges


def artifact(weak=False):
    c=dict(spot=100.,times=[0.,.3,.6,1.],variance=.04,nodes=[-.25,-.12,0.,.14,.28],
        initial_strikes=[70.,88.,100.,117.,140.],report_nodes=[-.2,0.,.22],relative_bump=.04,subdivisions=2)
    shocks=[[-.6,.8,-.2],[.1,-.8,1.2],[-.2,.1,-.7]]
    primary=[sample(c,z,weak) for z in shocks];mates=[sample(c,[-g for g in z],weak) for z in shocks]
    pairs=[[(a+b)/2 for a,b in zip(x,y)] for x,y in zip(primary,mates)]
    return dict(fixture_version=1,reference='100-digit price differences, Gaussian power series, midpoint strike integration, two-pass moments'+('; next-time integrated hat density' if weak else ''),
        relative_tolerance=1e-8,small_difference_absolute_tolerance=1e-8,case=c,shocks=shocks,
        estimates=[[stats([s[i] for s in rows]) for i in range(len(rows[0]))] for rows in [primary,pairs]])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--write',action='store_true');a=p.parse_args()
    with localcontext() as ctx:
        ctx.prec=100;payload=json.dumps(artifact(),indent=2,allow_nan=False)+'\n'
    if a.write:FIXTURE.write_text(payload,encoding='utf-8')
    elif FIXTURE.read_text(encoding='utf-8')!=payload:raise AssertionError('VegaKT chain reference changed')
    print('Joint Gamma-to-IV moments verified: 54 channels, three paths and three antithetic pairs')

if __name__=='__main__':main()
