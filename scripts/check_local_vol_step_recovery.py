#!/usr/bin/env python3
"""Independent price-bump statistics and step-Gamma uncertainty references.

Uses Decimal forward prices, not the native reverse/statistics/recovery APIs.
The five antithetic pairs are five observations, not ten independent paths.
"""
import argparse
import json
from decimal import Decimal as D, localcontext
from pathlib import Path

from check_local_vol_gamma_numerics import PI
from check_local_vol_step_adjoints import derivative

FIXTURE=Path(__file__).resolve().parents[1]/"fixtures/local-vol/step-recovery-v0.1.json"


def stats(values):
    mean=sum(values)/len(values)
    variance=sum((v-mean)**2 for v in values)/(len(values)-1)
    return {"mean":float(mean),"sample_variance":float(variance),
            "standard_error":float((variance/len(values)).sqrt())}


def artifact():
    base={"initial":100.,"simulation_times":[0.,.1,.23,.4,.7,1.],"forwards":[100.]*6,
          "grid_times":[0.,.4,1.],"nodes":[-.4,0.,.35],"variances":[.04]*9,
          "quadratic_weights":[0.,.15,0.,-.08,0.,.25],"cross_weight":.1}
    shocks=[[.35,-.6,.8,-.2,.4],[-.3,.1,-.8,1.2,.6],[1.3,-.2,.1,-.7,-.5],
            [-1.,.4,.6,-.1,.2],[.2,.5,-.4,.7,-.1]]
    primary,mates=[],[]
    for path,normal in enumerate(shocks):
        for sign,target in [(1,primary),(-1,mates)]:
            case=dict(base,shocks=[sign*z for z in normal],id=f"{path}/{sign}")
            nodes=[]
            for step in range(5):
                for node in range(3):
                    coarse=derivative(case,"step_hat_volatility",(step,node),D("1e-6"))
                    fine=derivative(case,"step_hat_volatility",(step,node),D("5e-7"))
                    if abs(coarse-fine)>max(abs(fine),D("1e-70"))*D("1e-12"):
                        raise AssertionError("Unconverged price derivative")
                    nodes.append(fine)
            target.append(nodes)
    paired=[[(a+b)/2 for a,b in zip(left,right)] for left,right in zip(primary,mates)]
    results=[]
    for name,samples in [("independent_path",primary),("antithetic_pair",paired)]:
        rows=[]
        for step in range(5):
            row_samples=[sample[step*3:(step+1)*3] for sample in samples]
            node_stats=[stats([sample[j] for sample in row_samples]) for j in range(3)]
            row={"nodes":node_stats,"total":stats([sum(sample) for sample in row_samples])}
            if step:
                time=D(base["simulation_times"][step])
                dt=D(base["simulation_times"][step+1]-base["simulation_times"][step])
                xs=list(map(D,base["nodes"]))
                strikes=[D(100)*x.exp() for x in xs]
                sigma=D(.2)
                densities=[]
                for x,k in zip(xs,strikes):
                    d2=(-x-sigma*sigma*time/2)/(sigma*time.sqrt())
                    densities.append((-d2*d2/2).exp()/((2*PI).sqrt()*k*sigma*time.sqrt()))
                active=[j for j,d in enumerate(densities) if d>=D(.05)*max(densities)]
                areas=[(xs[1]-xs[0])/2,(xs[2]-xs[0])/2,(xs[2]-xs[1])/2]
                denominator=[a*k**3*d*D(.04).sqrt()*dt for a,k,d in zip(areas,strikes,densities)]
                row.update({
                    "active_indices":active,
                    "retained":stats([sum(sample[j] for j in active) for sample in row_samples]),
                    "excluded":stats([sum((sample[j] for j in range(3) if j not in active),D(0)) for sample in row_samples]),
                    "gamma":[stats([sample[j]/denominator[j] for sample in row_samples]) for j in active],
                })
            rows.append(row)
        results.append({"sampling_unit":name,"count":5,"rows":rows,
                        "total":stats([sum(sample) for sample in samples])})
    return {"fixture_version":1,"reference":"100-digit Decimal price bumps; independent two-pass moments; analytic flat density and hat areas",
            "relative_tolerance":1e-9,"absolute_tolerance":0.,"density_threshold":.05,
            "base":base,"primary_shocks":shocks,"estimates":results}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write",action="store_true")
    args=parser.parse_args()
    with localcontext() as context:
        context.prec=100
        payload=json.dumps(artifact(),indent=2,allow_nan=False)+'\n'
    if args.write: FIXTURE.write_text(payload,encoding="utf-8")
    elif FIXTURE.read_text(encoding="utf-8")!=payload:
        raise AssertionError("Step-recovery statistical references changed")
    print("Independent path and antithetic-pair step/Gamma statistics verified")


if __name__=="__main__": main()
