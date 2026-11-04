"""Build complete single-asset, dividend and basket tasks without JSON inputs."""
import rust_pricing as rp


def main():
    discount = rp.DiscountCurve(1, [0., 2.], [1., .96])
    repo = rp.DiscountCurve(2, [0., 2.], [1., .99])
    engine = rp.Engine.pseudo_monte_carlo(7, 128, antithetic=True)
    market = rp.Market.equity(1, 1, 100., discount, repo,
        discrete_dividends=[rp.DividendEvent.fixed_cash(1, .5, 2.)])
    product = rp.Product.european_vanilla(1, 1, '2027-01-01', 100., 1., 'call')
    source = rp.PricingSpecification('2026-01-01', product, market,
        rp.PricingModelSpec(rp.EquityModelSpec.standard(rp.Model.black_scholes(.2))),
        engine, rp.RiskRequest(), rp.PricingNumerics(worker_threads=1, reduction_block_size=32))
    dividends = rp.BuehlerDividendModel(mean_reversion=.5, equity_linkage=.2,
        dividend_volatility=.1, equity_dividend_correlation=.1)
    dividend_source = rp.StochasticDividendSpecification(source, dividends, maximum_step=.25)

    graph = rp.SourceGraphBuilder()
    average = graph.multiply(graph.literal(.5), graph.add(
        graph.spot(1, '2027-01-01'), graph.spot(2, '2027-01-01')))
    payoff = graph.maximum(graph.subtract(average, graph.literal(100.)),
                          graph.literal(0.), smoothing_half_width=1.)
    basket = graph.finish([payoff], ['2027-01-01'], currency_id=1, underlying_ids=[1, 2])
    correlations = rp.CorrelationSchedule([1, 2], ['2026-01-01'], [[[1., .2], [.2, 1.]]],
        symmetry_abs_tol=1e-12, diagonal_abs_tol=1e-12, psd_abs_tol=1e-12,
        psd_rel_tol=1e-12, zero_pivot_abs_tol=1e-12, zero_pivot_rel_tol=1e-12)
    basket_source = rp.MultiAssetSpecification('2026-01-01', basket,
        [market, rp.Market.equity(1, 2, 100., discount, repo)],
        rp.MultiAssetModelSpec([rp.Model.black_scholes(.2), rp.Model.black_scholes(.25)], correlations),
        engine, maximum_step=.25, worker_threads=1, reduction_block_size=32)

    for label, inputs, operation in [('single', source, 'evaluate'),
                                      ('dividend', dividend_source, 'aad'),
                                      ('basket', basket_source, 'aad')]:
        task = rp.PricingTask.from_source(inputs, operation=operation)
        report = task.compile().evaluate().report
        replay = rp.PricingTask.from_json(task.to_json())
        assert replay.fingerprint == task.fingerprint
        assert replay.compile().evaluate().report.to_json() == report.to_json()
        print(label, report.value, report.standard_error, len(report.risks), task.fingerprint)


if __name__ == '__main__':
    main()
