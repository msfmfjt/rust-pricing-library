#!/usr/bin/env python3
"""Independent price-bump oracle for the positive-time Gamma recovery bridge."""

import argparse
import json
from decimal import Decimal as D, localcontext
from pathlib import Path

from check_local_vol_gamma_numerics import PI, cdf

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures/local-vol/gamma-recovery-v0.1.json"


def call(spot: D, strike: D, variance: D) -> D:
    sd = variance.sqrt()
    d1 = (spot.ln() - strike.ln() + variance / 2) / sd
    return spot * cdf(d1) - strike * cdf(d1 - sd)


def artifact() -> dict:
    forward, sigma, time, expiry, step, strike = 100.0, 0.25, 0.4, 1.0, 0.05, 105.0
    nodes = [-0.6, -0.27, -0.11, 0.0, 0.07, 0.19, 0.43, 0.7]
    cases = []
    with localcontext() as context:
        context.prec = 120
        f, vol, t, end, dt, payoff_strike = map(D, [forward, sigma, time, expiry, step, strike])
        xs = list(map(D, nodes))
        areas = [(xs[1] - xs[0]) / 2]
        areas += [(xs[i + 1] - xs[i - 1]) / 2 for i in range(1, len(xs) - 1)]
        areas += [(xs[-1] - xs[-2]) / 2]
        variance = vol * vol * (end - t)
        current_gamma = []
        for x in xs:
            k = f * x.exp()
            d1 = (k.ln() - payoff_strike.ln() + variance / 2) / variance.sqrt()
            current_gamma.append(float((-d1 * d1 / 2).exp() / (2 * PI).sqrt() / (k * variance.sqrt())))
        for bump in [0.0001, 0.00002]:
            h = D(bump)
            adjoints = []
            for x, area in zip(xs, areas):
                k = f * x.exp()
                d2 = (-x - vol * vol * t / 2) / (vol * t.sqrt())
                density = (-d2 * d2 / 2).exp() / ((2 * PI).sqrt() * k * vol * t.sqrt())
                up = variance + ((vol + h) ** 2 - vol * vol) * dt
                down = variance + ((vol - h) ** 2 - vol * vol) * dt
                conditional_vega = (call(k, payoff_strike, up) - call(k, payoff_strike, down)) / (2 * h)
                adjoints.append(float(area * k * density * conditional_vega))
            cases.append({"bump": bump, "integrated_local_vol_adjoints": adjoints})
    return {
        "fixture_version": 1,
        "reference": "120-digit Decimal conditional Black-Scholes price bumps; exact binary64 inputs",
        "scope": "One positive-time, mass-lumped local-volatility row; no vanilla weights or IV risk",
        "forward": forward, "volatility": sigma, "source_maturity": time,
        "option_expiry": expiry, "option_strike": strike, "delta_t": step,
        "log_moneyness_nodes": nodes, "relative_density_threshold": 1e-12,
        "current_time_gamma": current_gamma, "gamma_relative_tolerance": 1e-7,
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    payload = json.dumps(artifact(), indent=2, allow_nan=False) + "\n"
    if args.write:
        FIXTURE.write_text(payload, encoding="utf-8")
    elif FIXTURE.read_text(encoding="utf-8") != payload:
        raise AssertionError("Gamma recovery references changed")
    print("Two independent conditional-price bump references verified")


if __name__ == "__main__":
    main()
