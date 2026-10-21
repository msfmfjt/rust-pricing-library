"""Typed construction matches independently frozen JSON sources and native reports."""
import copy
from datetime import date
import json
from pathlib import Path
import unittest

import rust_pricing as rp

ROOT = Path(__file__).resolve().parents[2]


def fixture(name, directory='task-v1'):
    return json.loads((ROOT / f'fixtures/{directory}/{name}.golden.json').read_text())


def curve(value):
    return rp.DiscountCurve(value['curve_id'], value['times'], value['discount_factors'])


def market(value):
    events = [rp.DividendEvent.fixed_cash(d['event_id'], d['ex_time'], d['quote']['amount'])
              for d in value.get('discrete_dividends', [])]
    return rp.Market.equity(value['currency_id'], value['underlying_id'], value['spot'],
                           curve(value['discount_curve']), curve(value['dividend_curve']),
                           discrete_dividends=events)


def engine(value):
    reduction = value['variance_reduction']
    if value['type'] == 'pseudo_monte_carlo':
        return rp.Engine.pseudo_monte_carlo(value['master_seed'], value['independent_sampling_units'],
                                          **reduction)
    return rp.Engine.randomized_quasi_monte_carlo(value['points_per_scramble'],
        value['master_scramble_seed'], scramble_count=value['scramble_count'], **reduction)


def local_model(grid):
    return rp.Model.local_volatility_from_grid(grid['time_nodes'], grid['log_forward_moneyness_nodes'],
                                    grid['values'], floor=grid['floor'], cap=grid['cap'])


def model(value):
    if value['type'] == 'black_scholes':
        return rp.Model.black_scholes(value['volatility'])
    return local_model(value['local_variance_grid'])


def particle_arguments(value):
    return dict(particle_count=value['particle_count'], calibration_seed=value['seed'],
                log_bandwidth=value['log_bandwidth'], minimum_effective_samples=value['minimum_effective_samples'],
                retain_reverse_trace=value['retain_reverse_trace'])


def factor(value):
    kwargs = {k: v for k, v in value.items() if k != 'type'}
    if value['type'] == 'bergomi':
        return rp.VolatilityFactorSpec.bergomi(**kwargs)
    if value['type'] == 'bergomi_two_factor':
        return rp.VolatilityFactorSpec.bergomi_two_factor(**kwargs)
    return rp.VolatilityFactorSpec.rough_bergomi(rp.RoughBergomiModel(
        value['hurst'], value['vol_of_vol'], equity_vol_correlation=value['correlation']))


def rate_model(value):
    return rp.HullWhiteModel(value['mean_reversion'], value['volatility_times'], value['volatilities'])


def target(value):
    grid = value['local_variance_grid']
    if 'market_iv' in value:
        iv = value['market_iv']
        return rp.HullWhiteLsvTarget.from_market_iv(iv['maturity_nodes'], iv['log_moneyness_nodes'],
            iv['implied_volatilities'], grid['time_nodes'], grid['log_forward_moneyness_nodes'],
            floor=grid['floor'], cap=grid['cap'])
    return rp.HullWhiteLsvTarget.from_grid(local_model(grid), value['log_densities'])


def equity_model(value):
    kind = value['type']
    if kind == 'standard':
        return rp.EquityModelSpec.standard(model(value['model']))
    if kind == 'stochastic_volatility':
        return rp.EquityModelSpec.stochastic_volatility(value['initial_volatility'], factor(value['factor']))
    if kind == 'local_stochastic_volatility':
        return rp.EquityModelSpec.local_stochastic_volatility(local_model(value['target']['local_variance_grid']),
            factor(value['factor']), rp.LsvParticleConfig(**particle_arguments(value['particles'])),
            forward_density=target(value['forward_density']) if 'forward_density' in value else None)
    rough = rp.RoughVolatilityModel.rough_heston(**{k: v for k, v in value['model'].items() if k != 'type'})
    if kind == 'rough_volatility':
        return rp.EquityModelSpec.rough_volatility(rough)
    return rp.EquityModelSpec.rough_local_stochastic_volatility(
        local_model(value['target']['local_variance_grid']), rough,
        rp.LsvParticleConfig(**particle_arguments(value['particles'])))


def single(value):
    p = value['product']
    product = rp.Product.european_vanilla(p['underlying_id'], p['currency_id'], p['expiry'],
                                         p['strike'], p['notional'], p['side']['type'])
    rates = value['model']['rates']
    r = (rp.RateModelSpec.deterministic() if rates['type'] == 'deterministic' else
         rp.RateModelSpec.hull_white(rate_model(rates), equity_rate_correlation=rates['equity_rate_correlation'],
                                     vol_rate_correlations=rates['vol_rate_correlations']))
    risk = value['risk']
    return rp.PricingSpecification(date.fromisoformat(value['valuation_date']), product, market(value['market']),
        rp.PricingModelSpec(equity_model(value['model']['equity']), rates=r), engine(value['engine']),
        rp.RiskRequest(delta=risk['delta'], vega=risk['vega'], smile_dynamics=risk['smile_dynamics']['type']),
        rp.PricingNumerics(**value['numerics']))


def correlation(value):
    return rp.CorrelationSchedule(value['underlying_ids'], [e['date'] for e in value['entries']],
                                  [e['matrix'] for e in value['entries']], **value['tolerances'])


def basket_product():
    # This is the pre-existing golden basket's node order, built without JSON.
    g = rp.SourceGraphBuilder()
    basket = g.literal(0.)
    for underlying in (1, 2):
        spot = g.spot(underlying, '2027-01-01')
        weight, initial = g.literal(.5), g.literal(1.)
        basket = g.add(basket, g.divide(g.multiply(weight, spot), initial))
    payoff = g.subtract(basket, g.literal(100.))
    payoff = g.maximum(payoff, g.literal(0.), smoothing_half_width=1.)
    payoff = g.multiply(g.literal(1.), payoff)
    return g.finish([payoff], ['2027-01-01'], currency_id=1, underlying_ids=[1, 2])


def basket_lsv(value):
    f = value['factor']
    cls = {'bergomi': rp.MultiAssetLsvConfig, 'bergomi_two_factor': rp.MultiAssetLsv2FactorConfig,
           'rough_bergomi': rp.MultiAssetRoughLsvConfig}[f['type']]
    return cls(**{k: v for k, v in f.items() if k != 'type'}, **particle_arguments(value['particles']))


def basket_model(value):
    hw = value.get('hull_white')
    hw_config = (rp.MultiAssetHullWhiteConfig(rate_model(hw), hw['rate_correlations'],
        lsv_targets=[target(t['value']) if t['type'] == 'present' else None for t in hw['lsv_targets']])
        if hw else None)
    lc = value.get('local_correlation')
    lc_config = (rp.LocalCorrelationConfig(basket_weights=lc['basket_weights'], target_model=local_model(lc['target']),
        second_correlations=correlation(lc['second_correlation']), **particle_arguments(lc['particles']),
        feasibility=lc['feasibility'], minimum_variance_span=lc['minimum_variance_span'],
        second_driver_correlations=lc.get('second_driver_correlations'),
        hull_white_target=target(lc['hull_white_target']) if 'hull_white_target' in lc else None) if lc else None)
    return rp.MultiAssetModelSpec([model(m) for m in value['models']], correlation(value['correlation']),
        lsv=[basket_lsv(c['value']) if c['type'] == 'present' else None for c in value['lsv']],
        driver_correlations=value.get('driver_correlations'), hull_white=hw_config, local_correlation=lc_config)


def basket(value, configuration=None):
    return rp.MultiAssetSpecification(value['valuation_date'], basket_product(), [market(m) for m in value['markets']],
        configuration if configuration is not None else basket_model(value), engine(value['engine']), **value['numerics'])


def dividend(value):
    return rp.StochasticDividendSpecification(single(value['equity']), rp.BuehlerDividendModel(**value['dividends']),
        maximum_step=value['maximum_step'], volatility_correlations=value['volatility_correlations'],
        rate_correlation=value.get('rate_correlation'))


class TypedPricingSourceTests(unittest.TestCase):
    def assert_task_matches(self, source, saved):
        op = saved['operation']
        typed = rp.PricingTask.from_source(source, operation=op['type'],
                                         gamma_relative_bump=op.get('gamma_relative_bump'))
        parsed = rp.PricingTask.from_json(json.dumps(saved))
        self.assertEqual(typed.to_json(), parsed.to_json())
        self.assertEqual(typed.fingerprint, parsed.fingerprint)
        a, b = typed.compile(), parsed.compile()
        self.assertEqual(a.plan_fingerprint, b.plan_fingerprint)
        self.assertEqual(a.evaluate().report.to_json(), b.evaluate().report.to_json())
        return typed

    def test_all_frozen_tasks_match_prices_risks_fingerprints_and_provenance(self):
        for name in ('standard_risk', 'multi_asset_bs', 'multi_asset_joint',
                     'stochastic_dividend_bs', 'stochastic_dividend_hw', 'stochastic_dividend_lsv'):
            with self.subTest(name=name):
                saved = fixture(name)
                v = saved['source']
                source = (single(v['specification']) if v['type'] == 'single_asset' else
                          basket(v) if v['type'] == 'multi_asset' else dividend(v))
                task = self.assert_task_matches(source, saved)
                if name == 'multi_asset_joint':
                    self.assertIn('market_iv', task.to_json())
                    self.assertIn('0.20000000000001', task.to_json())
                if name == 'standard_risk':
                    self.assertEqual(task.to_json(), rp.PricingTask.from_specification(source).to_json())

    def test_single_asset_model_compositions_match_legacy_source_fixtures(self):
        for name in ('black_scholes', 'bergomi_hull_white', 'bergomi_lsv', 'rough_heston'):
            with self.subTest(name=name):
                value = fixture(name, 'specification-v1')
                source = single(value)
                parsed = rp.PricingSpecification.from_json(json.dumps(value))
                self.assertEqual(source.to_json(), parsed.to_json())
                self.assertEqual(source.fingerprint, parsed.fingerprint)
                self.assertEqual(source.compile().plan_fingerprint, parsed.compile().plan_fingerprint)
                self.assertEqual(source.compile().evaluate().value, parsed.compile().evaluate().value)

    def test_early_exercise_and_graph_inputs_reuse_legacy_contracts(self):
        base = fixture('black_scholes', 'specification-v1')
        m, e, bs = market(base['market']), engine(base['engine']), rp.Model.black_scholes(.2)
        lsm = rp.LsmConfig(rp.Engine.pseudo_monte_carlo(99, 64), max_degree=2)
        american = rp.Product.american_vanilla(1, 2, '2027-09-04', 100., 1., 'put',
                                              ['2027-03-04', '2027-09-04'])
        g = rp.SourceGraphBuilder()
        cashflow = g.subtract(g.spot(1, '2027-09-04'), g.literal(90.))
        graph = rp.Product.from_graph(g.finish([cashflow], ['2027-09-04'], currency_id=2, underlying_ids=[1]))
        for product, training in ((american, lsm), (graph, None)):
            with self.subTest(lsm=training is not None):
                risk = rp.RiskRequest(delta=True)
                source = rp.PricingSpecification('2026-09-04', product, m,
                    rp.PricingModelSpec(rp.EquityModelSpec.standard(bs)), e, risk,
                    rp.PricingNumerics(worker_threads=1, reduction_block_size=16, lsm=training))
                legacy = rp.PricingSpecification.from_request(
                    rp.PricingRequest('2026-09-04', product, m, bs, e, risk, lsm=training),
                    worker_threads=1, reduction_block_size=16)
                self.assertEqual(source.to_json(), legacy.to_json())
                self.assertEqual(rp.PricingTask.from_source(source).compile().evaluate().report.to_json(),
                                 rp.PricingTask.from_specification(legacy).compile().evaluate().report.to_json())

    def test_two_factor_and_rough_dividends_cover_sv_and_lsv(self):
        factors = [dict(type='bergomi_two_factor', mean_reversions=[.4, 1.5], vol_of_vol=.2,
                       mixing_weight=.4, spot_correlations=[-.2, -.1], factor_correlation=.1),
                   dict(type='rough_bergomi', hurst=.2, vol_of_vol=.15, correlation=-.2)]
        for f in factors:
            for lsv in (False, True):
                with self.subTest(factor=f['type'], lsv=lsv):
                    saved = fixture('stochastic_dividend_lsv')
                    saved['operation'] = {'type': 'local_variance_risk' if lsv else 'aad'}
                    v = saved['source']
                    eq = v['equity']['model']['equity']
                    eq['factor'] = f
                    if not lsv:
                        v['equity']['model']['equity'] = dict(type='stochastic_volatility', initial_volatility=.2, factor=f)
                        v['equity']['numerics']['maximum_step'] = v['maximum_step']
                    v['volatility_correlations'] = [0., 0.] if f['type'] == 'bergomi_two_factor' else [0.]
                    self.assert_task_matches(dividend(v), saved)

    def test_hybrid_density_and_rough_family_lsv_preserve_inputs(self):
        base = fixture('bergomi_lsv', 'specification-v1')
        variants = []
        hybrid = copy.deepcopy(base)
        t = rp.HullWhiteLsvTarget.flat(.2, [0., .25, .5, .75, 1.], [-.6, 0., .6])
        eq = hybrid['model']['equity']
        eq['target']['local_variance_grid']['values'] = [0.2 * 0.2] * 15
        eq['forward_density'] = dict(local_variance_grid=copy.deepcopy(eq['target']['local_variance_grid']),
                                     log_densities=t.forward_log_densities)
        hybrid['model']['rates'] = dict(type='hull_white', mean_reversion=.1, volatility_times=[0.],
                                       volatilities=[.005], equity_rate_correlation=0., vol_rate_correlations=[0.])
        variants.append(hybrid)
        rough = copy.deepcopy(base)
        rough['model']['equity'] = dict(type='rough_local_stochastic_volatility',
            target=base['model']['equity']['target'], particles=base['model']['equity']['particles'],
            model=fixture('rough_heston', 'specification-v1')['model']['equity']['model'])
        variants.append(rough)
        for value in variants:
            with self.subTest(model=value['model']['equity']['type']):
                source = single(value)
                # Compare to the independently parsed source, including its numerical diagnostics.
                parsed = rp.PricingSpecification.from_json(json.dumps(value))
                saved = dict(document_kind='pricing_task', schema_version=1, operation={'type': 'evaluate'},
                             source=dict(type='single_asset', specification=json.loads(parsed.to_json())))
                self.assert_task_matches(source, saved)

    def test_basket_accepts_two_factor_and_rough_configs(self):
        for factor_value in (
            dict(type='bergomi_two_factor', mean_reversions=[.4, 1.5], vol_of_vol=.2,
                 mixing_weight=.4, spot_correlations=[-.2, -.1], factor_correlation=.1),
            dict(type='rough_bergomi', hurst=.2, vol_of_vol=.15, correlation=-.2),
        ):
            saved = fixture('multi_asset_joint')
            v = saved['source']
            saved['operation'] = {'type': 'evaluate'}
            v.pop('local_correlation')
            v.pop('driver_correlations')
            v['lsv'][0]['value']['factor'] = factor_value
            v['hull_white']['rate_correlations'] = [0.] * (4 if factor_value['type'] == 'bergomi_two_factor' else 3)
            with self.subTest(factor=factor_value['type']):
                self.assert_task_matches(basket(v), saved)

    def test_owned_inputs_frozen_configs_and_default_composition(self):
        saved = fixture('multi_asset_joint')
        v = copy.deepcopy(saved['source'])
        config = basket_model(v)
        # Nested input lists are copied before any source or task is built.
        v['driver_correlations'][0][0][1] = -999.
        v['hull_white']['rate_correlations'][0] = -999.
        v['hull_white']['lsv_targets'].clear()
        v['local_correlation']['second_driver_correlations'][0][0][1] = -999.
        v['lsv'].clear()
        self.assert_task_matches(basket(saved['source'], config), saved)
        plain = fixture('multi_asset_bs')
        v = plain['source']
        models = [model(m) for m in v['models']]
        default_config = rp.MultiAssetModelSpec(models, correlation(v['correlation']))
        models.clear()
        self.assert_task_matches(basket(v, default_config), plain)
        for obj in (config, default_config, single(fixture('black_scholes', 'specification-v1')),
                    rp.PricingNumerics(worker_threads=1), rp.LsvParticleConfig(particle_count=32,
                    calibration_seed=7, log_bandwidth=.5, minimum_effective_samples=2.),
                    rp.BuehlerDividendModel(**fixture('stochastic_dividend_bs')['source']['dividends'])):
            with self.subTest(cls=type(obj).__name__), self.assertRaises(AttributeError):
                obj.changed = True

    def test_rate_correlations_are_copied_and_default_targets_use_asset_count(self):
        saved = fixture('multi_asset_bs')
        v = saved['source']
        correlations = [.1, -.1]
        rates = rp.HullWhiteModel(.1, [0.], [.005])
        hw = rp.MultiAssetHullWhiteConfig(rates, correlations)
        correlations[0] = float('nan')
        config = rp.MultiAssetModelSpec([model(m) for m in v['models']], correlation(v['correlation']), hull_white=hw)
        v['hull_white'] = dict(mean_reversion=.1, volatility_times=[0.], volatilities=[.005],
                               rate_correlations=[.1, -.1], lsv_targets=[{'type':'absent'}, {'type':'absent'}])
        self.assert_task_matches(basket(v, config), saved)
        joint = fixture('multi_asset_joint')['source']
        hw = rp.MultiAssetHullWhiteConfig(rate_model(joint['hull_white']), [0., 0., 0.])
        config = rp.MultiAssetModelSpec([model(m) for m in joint['models']], correlation(joint['correlation']),
            lsv=[basket_lsv(joint['lsv'][0]['value']), None], hull_white=hw)
        task = rp.PricingTask.from_source(basket(joint, config))
        self.assertEqual(json.loads(task.to_json())['source']['hull_white']['lsv_targets'],
                         [{'type': 'absent'}, {'type': 'absent'}])
        with self.assertRaises(rp.PricingError):
            task.compile()  # The LSV target is required; absence must not silently calibrate one.
        # Single-asset rate arrays are also detached before source composition.
        base = fixture('bergomi_hull_white', 'specification-v1')
        corr = base['model']['rates']['vol_rate_correlations']
        r = rp.RateModelSpec.hull_white(rate_model(base['model']['rates']), equity_rate_correlation=.1,
                                      vol_rate_correlations=corr)
        corr[0] = float('nan')
        # Build a matching source directly, without using the modified fixture rate list.
        p = rp.Product.european_vanilla(1, 2, '2027-09-04', 100., 1., 'call')
        s = rp.PricingSpecification('2026-09-04', p, market(base['market']),
            rp.PricingModelSpec(equity_model(base['model']['equity']), rates=r), engine(base['engine']),
            rp.RiskRequest(), rp.PricingNumerics(**base['numerics']))
        self.assertEqual(s.to_json(), single(fixture('bergomi_hull_white', 'specification-v1')).to_json())

    def test_source_validation_and_capability_rejection_share_rust_metadata(self):
        for field, mutate in (
            ('model', lambda v: v['models'].pop()),
            ('market', lambda v: v['markets'].reverse()),
            ('maximum_step', lambda v: v['numerics'].update(maximum_step=0.)),
            ('model', lambda v: v.update(driver_correlations=[[[float('nan')]]])),
        ):
            value = fixture('multi_asset_bs')['source']
            mutate(value)
            with self.subTest(field=field), self.assertRaises(rp.PricingError) as caught:
                basket(value)
            self.assertEqual((caught.exception.stage, caught.exception.field), ('validation', field))
        value = fixture('stochastic_dividend_lsv')['source']
        value['volatility_correlations'] = []
        with self.assertRaises(rp.PricingError) as caught:
            dividend(value)
        self.assertEqual(caught.exception.field, 'volatility_correlations')
        value = fixture('stochastic_dividend_lsv')['source']
        value['equity']['model']['equity']['particles']['retain_reverse_trace'] = False
        source = dividend(value)
        with self.assertRaises(rp.PricingError) as caught:
            rp.PricingTask.from_source(source, operation='local_variance_risk')
        self.assertEqual((caught.exception.code, caught.exception.operation, caught.exception.reason),
                         ('unsupported_operation', 'local_variance_risk', 'reverse_trace_required'))

    def test_invalid_config_values_types_and_compile_only_covariance_checks(self):
        with self.assertRaises(rp.PricingError) as caught:
            rp.PricingNumerics(worker_threads=0)
        self.assertEqual((caught.exception.code, caught.exception.stage), ('invalid_execution_policy', 'validation'))
        with self.assertRaises(rp.PricingError):
            rp.LsvParticleConfig(particle_count=0, calibration_seed=1, log_bandwidth=.5, minimum_effective_samples=2.)
        with self.assertRaises(rp.PricingError) as caught:
            rp.EquityModelSpec.stochastic_volatility(float('nan'), rp.VolatilityFactorSpec.bergomi(.5,.2,-.2))
        self.assertEqual((caught.exception.code, caught.exception.stage), ('invalid_number', 'validation'))
        with self.assertRaises(rp.PricingError) as caught:
            rp.VolatilityFactorSpec.bergomi(-1., .2, -.3)
        self.assertEqual((caught.exception.code, caught.exception.field), ('invalid_number', 'bergomi_mean_reversion'))
        with self.assertRaises(rp.PricingError) as caught:
            rp.EquityModelSpec.local_stochastic_volatility(rp.Model.black_scholes(.2),
                rp.VolatilityFactorSpec.bergomi(.5,.2,-.2),
                rp.LsvParticleConfig(particle_count=32, calibration_seed=1, log_bandwidth=.5, minimum_effective_samples=2.))
        self.assertEqual(caught.exception.field, 'model.equity.target')
        for source in ({}, rp.Model.black_scholes(.2), None):
            with self.subTest(source=source), self.assertRaises(TypeError):
                rp.PricingTask.from_source(source)
        with self.assertRaises(TypeError):
            rp.PricingModelSpec(rp.Model.black_scholes(.2))
        saved = fixture('multi_asset_joint')
        saved['source']['driver_correlations'][0][1][2] = -.2
        typed = rp.PricingTask.from_source(basket(saved['source']))
        with self.assertRaises(rp.PricingError) as caught:
            typed.compile()
        self.assertEqual((caught.exception.stage, caught.exception.code), ('compile', 'invalid_correlation'))


if __name__ == '__main__':
    unittest.main()
