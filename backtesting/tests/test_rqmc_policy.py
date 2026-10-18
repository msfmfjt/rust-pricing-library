"""Sampling design is explicit; point count is never mistaken for sample size."""
from copy import deepcopy
import unittest
from unittest.mock import patch
from rust_pricing_backtest import BacktestError,RustPricingProvider
from rust_pricing_backtest.multi_step_provider import check_sampling,validation

ENGINE=dict(type='randomized_quasi_monte_carlo',points_per_scramble=256,scramble_count=16,
    master_scramble_seed=100003,variance_reduction=dict(antithetic=True,brownian_bridge=False))


class RqmcPolicyTests(unittest.TestCase):
    def test_explicit_opt_in_and_exact_design(self):
        for n in (4,6):
            check_sampling(ENGINE,n,True,'rqmc_256x16')
            with self.assertRaisesRegex(BacktestError,'multi_step_provider_scope'):check_sampling(ENGINE,n,True)
        for key,value in [('type','pseudo_monte_carlo'),('points_per_scramble',512),('points_per_scramble',256.),
                          ('scramble_count',8),('scramble_count',16.),('scramble_count',True),
                          ('variance_reduction',dict(antithetic=True,brownian_bridge=True)),
                          ('variance_reduction',dict(antithetic=False,brownian_bridge=False))]:
            with self.subTest(key=key,value=value),self.assertRaisesRegex(BacktestError,'rqmc_provider_scope'):
                check_sampling(ENGINE|{key:value},6,True,'rqmc_256x16')
        with self.assertRaisesRegex(BacktestError,'rqmc_provider_scope'):check_sampling(ENGINE,6,False,'rqmc_256x16')

    def test_pseudo_mc_limits_remain_unchanged(self):
        for n,minimum in ((4,4096),(6,2048)):
            mc=dict(type='pseudo_monte_carlo',independent_sampling_units=minimum,variance_reduction=dict(antithetic=True))
            check_sampling(mc,n,True)
            with self.assertRaisesRegex(BacktestError,'multi_step_provider_scope'):
                check_sampling(mc|{'independent_sampling_units':minimum-1},n,True)
            with self.assertRaisesRegex(BacktestError,'rqmc_provider_scope'):check_sampling(mc,n,True,'rqmc_256x16')

    def test_path_budget_is_determined_by_shape_not_seed_or_observed_error(self):
        for n in (4,6):
            for kind in ('european_vanilla','arithmetic_asian','fixed_lookback'):
                points=128 if n==6 and kind=='arithmetic_asian' else 256
                for seed in (100043,100049):
                    engine=ENGINE|{'points_per_scramble':points,'master_scramble_seed':seed}
                    check_sampling(engine,n,True,'rqmc_path_16_scrambles',kind)
                    with self.assertRaisesRegex(BacktestError,'rqmc_provider_scope'):
                        check_sampling(engine|{'points_per_scramble':128 if points==256 else 256},
                                       n,True,'rqmc_path_16_scrambles',kind)
        d=validation('multi_step_custom_basis_gamma',basis_policy='coupled_row_hats',sampling_policy='rqmc_path_16_scrambles')
        self.assertEqual(d['scramble_count'],16)
        self.assertEqual(d['points_per_scramble_by_shape'],{'default':256,'six_interval_arithmetic_asian':128})

    def test_reject_unchecked_calibration_modes_before_native_import(self):
        for kwargs in (dict(diagnostic_only=True),dict(numerical_scope='multi_step_gamma'),
                dict(numerical_scope='multi_step_extension_put_gamma',variance_extension='linear_time'),
                dict(numerical_scope='multi_step_custom_basis_gamma',variance_basis_policy='row_hats')):
            with self.subTest(kwargs=kwargs),self.assertRaisesRegex(BacktestError,'rqmc_provider_policy'):
                RustPricingProvider(**kwargs,sampling_policy='rqmc_256x16')

    def test_metadata_distinguishes_runtime_check_from_independent_scope(self):
        for scope,extension,basis in [('multi_step_extension_put_gamma','all_rows',None),
                                      ('multi_step_custom_basis_gamma',None,'coupled_row_hats')]:
            d=validation(scope,extension,basis,'rqmc_256x16')
            self.assertEqual(d['status'],'runtime_checked_multi_step_rqmc')
            self.assertEqual(d['sampling_unit'],'independent_scramble_mean')
            self.assertEqual((d['points_per_scramble'],d['scramble_count']),(256,16))
            self.assertNotIn('minimum_sampling_units_by_intervals',d)
            self.assertEqual(d['gamma_relative_standard_error_limit'],.25)
            self.assertEqual(d['relative_tolerance'],1e-6)


if __name__=='__main__':unittest.main()
