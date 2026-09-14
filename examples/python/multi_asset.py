"""Two-asset Basket, Worst-of and Autocallable; price and asset-labelled risk."""
import math
import rust_pricing as rp

valuation = "2026-01-01"
expiry = "2027-01-01"
discount = rp.DiscountCurve(1, [0., 2.], [1., math.exp(-.06)])
dividend = rp.DiscountCurve(2, [0., 2.], [1., math.exp(-.02)])
markets = [rp.Market.equity(1, 1, 100., discount, dividend),
           rp.Market.equity(1, 2, 90., discount, dividend)]
models = [rp.Model.black_scholes(.2), rp.Model.black_scholes(.3)]
correlation = rp.CorrelationSchedule(
    [1, 2], [valuation, "2026-07-01"],
    [[[1., .3], [.3, 1.]], [[1., .6], [.6, 1.]]],
    symmetry_abs_tol=1e-12, diagonal_abs_tol=1e-12,
    psd_abs_tol=1e-12, psd_rel_tol=1e-12,
    zero_pivot_abs_tol=1e-12, zero_pivot_rel_tol=1e-12)
engine = rp.Engine.randomized_quasi_monte_carlo(4096, 702, antithetic=True)
products = {
    "Basket": rp.MultiAssetProduct.basket([1, 2], [.5, .5], [1., 1.], "call", 95., expiry, expiry),
    "Worst-of": rp.MultiAssetProduct.worst_of([1, 2], [100., 90.], "put", 1., expiry, expiry, notional=100., smoothing_half_width=.02),
    "Autocallable": rp.MultiAssetProduct.autocallable(
        [1, 2], [100., 90.],
        [rp.AutocallObservation(t, t, coupon_amount=4., coupon_level=.9, call_level=1.05)
         for t in ["2026-07-01", expiry]], expiry, expiry,
        notional=100., final_barrier=.7, memory=True,
        on_autocall="pay", on_maturity="forfeit", smoothing_half_width=.03),
}
for name, product in products.items():
    plan = rp.MultiAssetPlan.compile(valuation, product, markets, models, correlation, engine,
                                    maximum_step=.125, worker_threads=2)
    result = plan.evaluate_aad(gamma_relative_bump=.005)
    print(f"{name}: {result.value:.8f} (SE {result.standard_error:.8f})")
    print("  Asset   Delta       Vega / vol point")
    for risk in result.risks:
        print(f"  {risk.underlying_id:5d}   {risk.delta.value:.8f}  {risk.bs_vega_per_vol_point.value:.8f}")
    print("  Gamma:", [[e.value for e in row] for row in result.gamma])
