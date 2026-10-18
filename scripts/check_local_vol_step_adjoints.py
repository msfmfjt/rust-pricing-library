#!/usr/bin/env python3
"""Independent Decimal path-price bumps for step-local LV adjoints.

Replays the Log-Euler price in 100-digit arithmetic and applies five-point
differences to one interval's effective variance/volatility, one spatial hat
at that interval, or one original variance-grid node. No native reverse is used.
"""

import argparse
import json
from decimal import Decimal as D, localcontext
from pathlib import Path

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures/local-vol/step-adjoints-v0.1.json"


def weights(nodes, value):
    if value <= nodes[0]:
        return 0, D(0)
    if value >= nodes[-1]:
        return len(nodes)-2, D(1)
    index = next(i for i in range(len(nodes)-1) if nodes[i] <= value < nodes[i+1])
    return index, (value-nodes[index])/(nodes[index+1]-nodes[index])


def evaluate(case, kind=None, index=None, bump=D(0)):
    times, forwards = map(lambda k: list(map(D, case[k])), ["simulation_times", "forwards"])
    grid_times, nodes, values = map(lambda k: list(map(D, case[k])), ["grid_times", "nodes", "variances"])
    if kind == "grid_variance":
        values[index] += bump
    state = D(case["initial"])
    states = [state]
    for step, shock in enumerate(map(D, case["shocks"])):
        dt = times[step+1]-times[step]
        x = (state/forwards[step]).ln()
        ti, tw = weights(grid_times, times[step])
        xi, xw = weights(nodes, x)
        row = ti * len(nodes)
        low = (1-xw)*values[row+xi]+xw*values[row+xi+1]
        high = (1-xw)*values[row+len(nodes)+xi]+xw*values[row+len(nodes)+xi+1]
        variance = (1-tw)*low+tw*high
        if kind == "step_variance" and step == index:
            variance += bump
        sigma = variance.sqrt()
        if kind == "step_volatility" and step == index:
            sigma += bump
        if kind == "step_hat_volatility" and step == index[0]:
            hat = (1-xw if index[1] == xi else D(0)) + (xw if index[1] == xi+1 else D(0))
            sigma += bump*hat
        state *= forwards[step+1]/forwards[step] * (-sigma*sigma*dt/2+sigma*dt.sqrt()*shock).exp()
        states.append(state)
    price = sum(D(q)*s*s for q,s in zip(case["quadratic_weights"],states))
    price += D(case["cross_weight"])*states[2]*states[-1]
    return price, states


def derivative(case, kind, index, h):
    price = lambda b: evaluate(case, kind, index, b)[0]
    return (8*(price(h)-price(-h))-(price(2*h)-price(-2*h)))/(12*h)


def reference(case, kind, index):
    h = D("1e-6")
    coarse = derivative(case,kind,index,h)
    fine = derivative(case,kind,index,h/2)
    if abs(coarse-fine) > max(abs(fine), D("1e-70"))*D("1e-12"):
        raise AssertionError(f"Unconverged {case['id']} {kind} {index}")
    return float(fine)


def artifact():
    cases = []
    for name, initial, flat in [("nonflat_path_dependent",105.,False),
                                 ("flat_quadratic",105.,True),
                                 ("right_clamped",200.,False),
                                 ("left_clamped",50.,False)]:
        case = {
            "id":name, "initial":initial,
            "simulation_times":[0.,.1,.23,.4,.7,1.],
            "forwards":[100.,100.3,100.7,101.1,102.,103.],
            "grid_times":[0.,.4,1.], "nodes":[-.4,0.,.35],
            "variances":[.04]*9 if flat else [.075,.045,.065,.06,.05,.09,.05,.04,.08],
            "shocks":[.35,-.6,.8,-.2,.4],
            "quadratic_weights":[0.,0.,0.,0.,0.,.5] if flat else [0.,.15,0.,-.08,0.,.25],
            "cross_weight":0. if flat else .1,
        }
        price, states = evaluate(case)
        case["expected"] = {
            "price":float(price), "states":list(map(float,states)),
            "step_variance":[reference(case,"step_variance",i) for i in range(5)],
            "step_volatility":[reference(case,"step_volatility",i) for i in range(5)],
            "step_hat_volatility":[[reference(case,"step_hat_volatility",(i,j)) for j in range(3)] for i in range(5)],
            "grid_variance":[reference(case,"grid_variance",i) for i in range(9)],
        }
        cases.append(case)
    return {"fixture_version":1,
            "reference":"100-digit Decimal path prices; five-point differences at 1e-6 and 5e-7; exact binary64 inputs",
            "equation":"SSRN 4107770 equation (10), before time/space aggregation",
            "relative_tolerance":1e-9, "absolute_tolerance":0., "cases":cases}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write",action="store_true")
    args=parser.parse_args()
    with localcontext() as context:
        context.prec=100
        payload=json.dumps(artifact(),indent=2,allow_nan=False)+'\n'
    if args.write:
        FIXTURE.write_text(payload,encoding="utf-8")
    elif FIXTURE.read_text(encoding="utf-8") != payload:
        raise AssertionError("Step-adjoint price references changed")
    print("Four independent step-adjoint cases, 136 price derivatives verified")


if __name__ == "__main__":
    main()
