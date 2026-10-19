#!/usr/bin/env python3
"""Independent price-difference references for SSRN 4107770 equations (5)/(9).

120-digit Decimal prices, without native code or the total-Gamma kernel formula.
Gamma is integrated twice to construct a piecewise cubic terminal payoff. Its
lognormal expectation is differentiated by a five-point spot stencil, including
the spot dependence of the conditional variance. IV projection references bump
individual quoted IVs in Black-Scholes prices. Default checks, --write freezes.
"""

import argparse
import json
from decimal import Decimal as D, localcontext
from pathlib import Path

from check_local_vol_gamma_numerics import cdf

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures/local-vol/vegakt-paper-v0.1.json"


def payoff_price(spot, variance, strikes, gamma):
    """E[f(S')], where f'' is linear Gamma with constant tails; f(0)=f'(0)=0."""
    sd = variance.sqrt()
    bounds = [D(0), *strikes, D("Infinity")]
    price = D(0)
    value = derivative = D(0)
    for i, (lo, hi) in enumerate(zip(bounds, bounds[1:])):
        slope = D(0) if i in (0, len(strikes)) else (
            (gamma[i] - gamma[i - 1]) / (strikes[i] - strikes[i - 1])
        )
        intercept = gamma[0] if i == 0 else gamma[i - 1] - slope * lo
        linear = derivative - intercept * lo - slope * lo**2 / 2
        constant = value - linear * lo - intercept * lo**2 / 2 - slope * lo**3 / 6
        coeff = [constant, linear, intercept / 2, slope / 6]
        for power, coefficient in enumerate(coeff):
            zl = (lo.ln() - spot.ln() + (D(".5") - power) * variance) / sd if lo else D("-Infinity")
            zh = (hi.ln() - spot.ln() + (D(".5") - power) * variance) / sd
            moment = spot**power * (D(power * (power - 1)) * variance / 2).exp()
            price += coefficient * moment * (cdf(zh) - cdf(zl))
        if hi.is_finite():
            value = sum(c * hi**p for p, c in enumerate(coeff))
            derivative = linear + intercept * hi + slope * hi**2 / 2
    return price


def second_difference(function, x, h):
    return (-function(x + 2*h) + 16*function(x + h) - 30*function(x)
            + 16*function(x - h) - function(x - 2*h)) / (12*h*h)


def first_difference(function, h):
    return (function(-2*h) - 8*function(-h) + 8*function(h) - function(2*h)) / (12*h)


def call_price(spot, strike, sigma, time):
    sd = sigma * time.sqrt()
    d1 = (spot / strike).ln() / sd + sd / 2
    return spot * cdf(d1) - strike * cdf(d1 - sd)


def artifact():
    cases = [
        ("quadratic_sloped", 100., .04, .0004, .000002, [70., 130.], [1., 1.]),
        ("quadratic_negative_curvature", 100., .04, 0., -.0005, [70., 130.], [1., 1.]),
        ("broad_linear", 100., .04, -.0006, .00001, [50., 90., 160.], [1., -2., 3.]),
        ("narrow_hat", 100., .04, .0004, -.000003, [100., 100.000001, 100.000002], [0., 1., 0.]),
        ("right_tail_hat", 100., .04, -.0006, .000002, [640., 640.00001, 640.00002], [0., 1., 0.]),
        ("left_tail", 100., .04, .0002, .000003, [14., 15.5, 17.], [3., 1., 0.]),
        ("small_variance", 100., .0001, .000001, -.000000005, [98., 100., 102.], [1., 2., 4.]),
        ("signed_scale", 100., .04, .0004, -.000003, [90., 110.], [-2., -4.]),
    ]
    rows = []
    for name, spot, variance, first, second, strikes, gamma in cases:
        s, v, v1, v2 = map(D, (spot, variance, first, second))
        ks, gs = list(map(D, strikes)), list(map(D, gamma))
        def price(bumped):
            dx = bumped - s
            return payoff_price(bumped, v + v1*dx + v2*dx*dx/2, ks, gs)
        h = s * D("1e-6")
        coarse = second_difference(price, s, h)
        fine = second_difference(price, s, h/2)
        if abs(coarse - fine) > abs(fine) * D("1e-12"):
            raise AssertionError(f"Unconverged price stencil: {name}")
        rows.append({"id": name, "spot": spot, "variance": variance,
                     "variance_first": first, "variance_second": second,
                     "strikes": strikes, "gamma": gamma,
                     "spot_bump": float(h), "coarse_price_gamma": float(coarse),
                     "expected_price_gamma": float(fine)})
    times, nodes = [.5, 1.], [-.4, 0., .3]
    ivs = [.27, .21, .24, .29, .23, .26]
    projections = []
    for name, time, strikes, weights in [
        ("unit_call_at_node", 1., [100.], [-1.]),
        ("off_node_call", .5, [112.], [-1.]),
        ("signed_calls", 1., [87., 116.], [-1., 2.]),
        ("edge_calls", .5, [40., 190.], [-2., -1.]),
    ]:
        t, spot = D(time), D(100)
        time_index = times.index(time)
        xs, vol = list(map(D, nodes)), list(map(D, ivs))
        def price(quote_index, bump):
            total = D(0)
            for strike, weight in zip(map(D, strikes), map(D, weights)):
                x = (strike/spot).ln()
                if x <= xs[0]:
                    j, w = 0, D(0)
                elif x >= xs[-1]:
                    j, w = len(xs)-2, D(1)
                else:
                    j = next(i for i in range(len(xs)-1) if xs[i] <= x <= xs[i+1])
                    w = (x - xs[j])/(xs[j+1]-xs[j])
                index = time_index * len(xs) + j
                left = vol[index] + (bump if quote_index == index else 0)
                right = vol[index+1] + (bump if quote_index == index+1 else 0)
                # The position whose offsetting hedge has weight w is -w calls.
                total -= weight * call_price(spot, strike, (1-w)*left+w*right, t)
            return total
        coarse = [first_difference(lambda b: price(j, b), D("1e-5")) for j in range(len(ivs))]
        fine = [first_difference(lambda b: price(j, b), D("5e-6")) for j in range(len(ivs))]
        if any(abs(a-b) > max(abs(b), D("1e-50"))*D("1e-10") for a,b in zip(coarse,fine)):
            raise AssertionError(f"Unconverged IV stencil: {name}")
        projections.append({"id":name, "maturity":time, "spot":100., "strikes":strikes,
                            "hedge_weights":weights, "expected_raw_buckets":list(map(float,fine))})
    return {"fixture_version":1,
            "reference":"120-digit Decimal prices, five-point spot/quote-IV differences, exact binary64 inputs",
            "paper":"SSRN 4107770, May 2022, equations (4), (5), (9)",
            "relative_tolerance":1e-9, "absolute_tolerance":0.,
            "gamma_cases":rows,
            "projection_basis":{"maturities":times,"log_moneyness_nodes":nodes,"implied_volatilities":ivs},
            "projection_cases":projections}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    with localcontext() as context:
        context.prec = 120
        payload = json.dumps(artifact(), indent=2, allow_nan=False) + "\n"
    if args.write:
        FIXTURE.write_text(payload, encoding="utf-8")
    elif FIXTURE.read_text(encoding="utf-8") != payload:
        raise AssertionError("VegaKT paper price references changed")
    print("Eight total-Gamma price stencils and four quote-IV projection cases verified")


if __name__ == "__main__":
    main()
