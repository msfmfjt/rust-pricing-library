"""Build, save and price a custom two-payment product without a new wrapper."""
from pathlib import Path
import json
import rust_pricing as rp

ROOT = Path(__file__).resolve().parents[2]


def coupon_and_capped_call():
    b = rp.SourceGraphBuilder()
    first = b.spot(1, "2027-03-04")
    last = b.spot(1, "2027-09-04")
    # Smooth the coupon trigger explicitly when pathwise Greeks are required.
    trigger = b.indicator(b.subtract(first, b.literal(95.)), smoothing_half_width=2.)
    coupon = b.multiply(b.literal(5.), trigger)
    call = b.maximum(b.subtract(last, b.literal(100.)), b.literal(0.))
    capped_call = b.minimum(call, b.literal(20.))
    return b.finish([coupon, capped_call], ["2027-03-06", "2027-09-06"],
                    currency_id=2, underlying_ids=[1])


def specification(graph):
    discount = rp.DiscountCurve(10, [0., 2.], [1., .95**2])
    dividend = rp.DiscountCurve(11, [0., 2.], [1., .98**2])
    request = rp.PricingRequest(
        "2026-09-04", rp.Product.from_graph(graph),
        rp.Market.equity(2, 1, 100., discount, dividend),
        rp.Model.black_scholes(.2),
        rp.Engine.pseudo_monte_carlo(7, 1024, antithetic=True),
        rp.RiskRequest(delta=True, vega=True),
    )
    return rp.PricingSpecification.from_request(request, worker_threads=2,
                                               reduction_block_size=128)


def main():
    graph = coupon_and_capped_call()
    replay = rp.GraphProduct.from_json(graph.to_json())
    assert replay.fingerprint == graph.fingerprint
    source = specification(replay)
    restored = rp.PricingSpecification.from_json(source.to_json())
    result = restored.compile().evaluate()
    assert result.value == source.compile().evaluate().value
    print(json.dumps(dict(value=result.value, standard_error=result.standard_error,
                          source_fingerprint=restored.fingerprint)))

    # The same graph document works with the existing multi-asset entry point.
    # Its market/model/correlation configuration is supplied separately.
    multi = rp.MultiAssetProduct.from_graph(replay)
    assert multi.to_graph().fingerprint == graph.fingerprint
    fixture = rp.GraphProduct.from_json(
        (ROOT / "fixtures/product-graph-v1/coupon_and_capped_call.golden.json").read_text())
    assert fixture.fingerprint == graph.fingerprint


if __name__ == "__main__":
    main()
