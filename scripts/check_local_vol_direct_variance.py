#!/usr/bin/env python3
"""Independent Decimal barrier prices for direct-node and transition LV risk.

Prices the product of bridge survival factors with trapezoidal node variance.
No native bridge reverse, path reverse, or risk statistics are used.
"""
import argparse
import json
from decimal import Decimal as D, localcontext
from pathlib import Path
from check_local_vol_step_adjoints import weights
from check_local_vol_step_recovery import stats

FIXTURE = Path(__file__).resolve().parents[1] / 'fixtures/local-vol/direct-variance-v0.1.json'


def price(case, kind=None, index=None, bump=D(0)):
    times, forwards, grid_times, xs, values = [list(map(D, case[k])) for k in
        ['times', 'forwards', 'grid_times', 'nodes', 'variances']]
    if kind == 'grid_variance':
        values[index] += bump

    def local(time, state, forward):
        ti, tw = weights(grid_times, time)
        xi, xw = weights(xs, (state/forward).ln())
        low = (1-xw)*values[ti*3+xi]+xw*values[ti*3+xi+1]
        high = (1-xw)*values[(ti+1)*3+xi]+xw*values[(ti+1)*3+xi+1]
        return (1-tw)*low+tw*high, xi, xw

    def hat(xi, xw, j):
        return (1-xw if j == xi else D(0)) + (xw if j == xi+1 else D(0))

    states = [D(case['initial'])]
    for i, shock in enumerate(map(D, case['shocks'])):
        a, xi, xw = local(times[i], states[-1], forwards[i])
        sigma = a.sqrt()
        if kind == 'transition_hat' and index[0] == i:
            sigma += bump*hat(xi,xw,index[1])
        if kind in ('parallel_vol','transition_parallel'): sigma += bump
        dt = times[i+1]-times[i]
        states.append(states[-1]*forwards[i+1]/forwards[i]*(-sigma*sigma*dt/2+sigma*dt.sqrt()*shock).exp())
    endpoint_variances = []
    for i, state in enumerate(states):
        a, xi, xw = local(times[i], state, forwards[i])
        sigma = a.sqrt()
        if kind == 'direct_hat' and index[0] == i:
            sigma += bump*hat(xi,xw,index[1])
        if kind == 'direct_row' and index == i: sigma += bump
        if kind == 'parallel_vol': sigma += bump
        endpoint_variances.append(sigma*sigma)
    barrier = D(case['barrier'])
    distances = [(barrier/s).ln() if case['direction'] == 'up' else (s/barrier).ln() for s in states]
    survival = D(1)
    for i in range(case['monitor_end']):
        if min(distances[i:i+2]) <= 0:
            survival = D(0)
            break
        integrated = (endpoint_variances[i]+endpoint_variances[i+1])*(times[i+1]-times[i])/2
        survival *= 1-(-2*distances[i]*distances[i+1]/integrated).exp()
    vanilla = D(case['quadratic'])*states[-1]**2
    rebate = D(case['rebate'])
    payoff = rebate+survival*(vanilla-rebate) if case['style'] == 'out' else vanilla+survival*(rebate-vanilla)
    return D(case['discount'])*payoff


def derivative(case, kind, index, h):
    p = lambda b: price(case,kind,index,b)
    return (8*(p(h)-p(-h))-(p(2*h)-p(-2*h)))/(12*h)


def reference(case, kind, index):
    coarse = derivative(case,kind,index,D('1e-6'))
    fine = derivative(case,kind,index,D('5e-7'))
    if abs(coarse-fine) > max(abs(fine),D('1e-70'))*D('1e-10'):
        raise AssertionError(f'Unconverged {case["id"]} {kind} {index}')
    return float(fine)


def artifact():
    cases = []
    configurations = [('up_out_nonflat','up','out',115.,5,False),
                      ('down_in_nonflat','down','in',85.,5,False),
                      ('partial_monitoring','up','out',115.,3,False),
                      ('flat_out','up','out',115.,5,True),
                      ('touched_out','up','out',101.,5,False)]
    for name,direction,style,barrier,end,flat in configurations:
        case = dict(id=name,initial=100.,times=[0.,.1,.23,.4,.7,1.],forwards=[100.]*6,
                    grid_times=[0.,.4,1.],nodes=[-.4,0.,.35],
                    variances=[.04]*9 if flat else [.07,.045,.065,.06,.05,.09,.05,.04,.08],
                    shocks=[.35,-.6,.8,-.2,.4],barrier=barrier,direction=direction,style=style,
                    monitor_end=end,quadratic=.005,rebate=3.,discount=.97)
        case['expected'] = dict(price=float(price(case)),
            transition_hat=[[reference(case,'transition_hat',(i,j)) for j in range(3)] for i in range(5)],
            direct_hat=[[reference(case,'direct_hat',(i,j)) for j in range(3)] for i in range(6)],
            grid_variance=[reference(case,'grid_variance',j) for j in range(9)],
            parallel_vol=reference(case,'parallel_vol',None))
        cases.append(case)
    primary_shocks = [[.35,-.6,.8,-.2,.4],[-.3,.1,-.8,1.2,.6],[1.3,-.2,.1,-.7,-.5],
                      [-1.,.4,.6,-.1,.2],[.2,.5,-.4,.7,-.1]]
    # Independent moments of whole-price derivatives, not sums of native risks.
    primary, mates = [], []
    for normal in primary_shocks:
        for sign,target in [(1,primary),(-1,mates)]:
            c=dict(cases[3],shocks=[sign*g for g in normal])
            quantities=[('parallel_vol',None),('transition_parallel',None)]+[('direct_row',i) for i in range(6)]
            row=[]
            for kind,index in quantities:
                reference(c,kind,index)  # Verify the two stencil widths.
                row.append(derivative(c,kind,index,D('5e-7')))
            target.append(row)
    paired=[[(a+b)/2 for a,b in zip(left,right)] for left,right in zip(primary,mates)]
    moments=[]
    for samples in [primary,paired]:
        moments.append([stats([sample[i] for sample in samples]) for i in range(8)])
    return dict(fixture_version=1,reference='100-digit Decimal forward bridge prices, two five-point bump widths',
                relative_tolerance=1e-9,absolute_tolerance=0.,cases=cases,
                statistics=dict(base_case_id='flat_out',primary_shocks=primary_shocks,moments=moments))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write',action='store_true')
    args=parser.parse_args()
    with localcontext() as context:
        context.prec=100
        payload=json.dumps(artifact(),indent=2,allow_nan=False)+'\n'
    if args.write: FIXTURE.write_text(payload,encoding='utf-8')
    elif FIXTURE.read_text(encoding='utf-8') != payload:
        raise AssertionError('Direct-variance references changed')
    print('Five barrier cases, 215 path derivatives and 80 statistical price derivatives verified')


if __name__ == '__main__': main()
