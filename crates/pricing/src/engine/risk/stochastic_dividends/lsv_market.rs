//! Exact market-input reverse for stochastic-dividend residual LSV.
//!
//! The residual LSV calibration is expressed in relative log-moneyness
//! log(F_res/F_res(0)). Spot, fixed-cash means and deterministic carry curves
//! only re-anchor F_res(0) and rebuild the affine physical-Spot coefficients;
//! the normalized f/Y dynamics and calibrated leverage values are unchanged.

use super::*;
use crate::engine::processes::stochastic_dividends::reverse::ReverseContext;

const METHOD: &str = "buehler-residual-lsv-scale-invariant-market-reverse-v1";

#[derive(Clone, Debug, PartialEq)]
pub struct StochasticDividendLsvMarketRisk {
    pub price: StochasticDividendPrice,
    pub delta: f64,
    pub delta_standard_error: f64,
    pub cash_times: Box<[f64]>,
    pub cash_mean_adjoints: Box<[f64]>,
    pub cash_mean_standard_errors: Box<[f64]>,
    pub discount_times: Box<[f64]>,
    pub discount_log_df_adjoints: Box<[f64]>,
    pub discount_log_df_standard_errors: Box<[f64]>,
    pub repo_spread_times: Box<[f64]>,
    pub repo_spread_log_df_adjoints: Box<[f64]>,
    pub repo_spread_log_df_standard_errors: Box<[f64]>,
    pub method: &'static str,
}

impl StochasticDividendLsvMarketRisk {
    #[must_use]
    pub fn discount_node_dv01(&self) -> Vec<f64> {
        self.discount_log_df_adjoints
            .iter()
            .zip(&self.discount_times)
            .map(|(adjoint, time)| -1e-4 * time * adjoint)
            .collect()
    }

    #[must_use]
    pub fn repo_spread_node_dv01(&self) -> Vec<f64> {
        self.repo_spread_log_df_adjoints
            .iter()
            .zip(&self.repo_spread_times)
            .map(|(adjoint, time)| -1e-4 * time * adjoint)
            .collect()
    }
}

impl StochasticDividendPricingPlan {
    /// Spot/cash/curve risk with residual-LSV re-anchoring and fixed requested
    /// Local-variance target. No finite bump and no calibration reverse trace are
    /// required because the normalized residual-equity calibration is homogeneous
    /// in funded residual equity.
    pub fn evaluate_lsv_market_risk(
        &self,
    ) -> Result<StochasticDividendLsvMarketRisk, MonteCarloError> {
        if self.lsv.is_none() || !self.path.is_lsv() {
            return Err(MonteCarloError::UnsupportedRiskForModel {
                model: "LSV market risk requires a stochastic-dividend residual LSV plan",
            });
        }
        if !self.risk_supported {
            return Err(MonteCarloError::UnsupportedRiskForModel {
                model: "stochastic-dividend LSV discontinuous payoff requires explicit smoothing",
            });
        }
        let surface = self
            .path
            .lsv_surface()
            .ok_or(MonteCarloError::UnsupportedRiskForModel {
                model: "LSV market risk requires a stochastic-dividend residual LSV surface",
            })?;
        if surface.initial_f().to_bits() != self.path.risky_spot().to_bits() {
            return Err(invalid("lsv_market_anchor").into());
        }

        let context = ReverseContext::new(&self.path, &self.market, self.payment_time)?;
        let cash_count = context.cash_times.len();
        let discount_count = context.discount_times.len();
        let repo_count = context.repo_spread_times.len();
        let width = 2 + cash_count + discount_count + repo_count;
        let executor = DeterministicExecutor::new(self.policy)?;
        let dimension = self.path.random_dimension();

        let (statistics, units, paths) = match self.engine {
            EngineConfig::PseudoMonteCarlo(config) => {
                let count = config.independent_sampling_units().get();
                let bridge = self.bridge(config.variance_reduction())?;
                let rng = Philox4x32::from_seed(config.master_seed());
                let stats = executor.try_map_reduce_statistics_vector(count, width, |p, out| {
                    let z = (0..dimension)
                        .map(|d| {
                            rng.standard_normal(RandomCoordinate::new(
                                p,
                                d,
                                RandomDomain::Valuation,
                            ))
                        })
                        .collect();
                    self.sample_lsv_market_risk(
                        &context,
                        z,
                        bridge.as_ref(),
                        config.variance_reduction().antithetic(),
                        out,
                    )
                })?;
                (stats, count, config.evaluated_paths())
            }
            EngineConfig::RandomizedQuasiMonteCarlo(config) => {
                let qmc = RqmcPlan::compile(config, dimension)?;
                let bridge = self.bridge(config.variance_reduction())?;
                let count = config.points_per_scramble().get();
                let mut means = Vec::with_capacity(config.scramble_count().get() as usize);
                for scramble in 0..config.scramble_count().get() {
                    let stats =
                        executor.try_map_reduce_statistics_vector(count, width, |p, out| {
                            let z = (0..dimension)
                                .map(|d| {
                                    let u = qmc
                                        .uniform(scramble, p, d)
                                        .map_err(|_| invalid("rqmc_uniform"))?;
                                    inverse_standard_normal(u).map_err(|_| invalid("rqmc_normal"))
                                })
                                .collect::<Result<Vec<_>, _>>()?;
                            self.sample_lsv_market_risk(
                                &context,
                                z,
                                bridge.as_ref(),
                                config.variance_reduction().antithetic(),
                                out,
                            )
                        })?;
                    means.push(
                        stats
                            .iter()
                            .map(|s| s.sum().total() / count as f64)
                            .collect::<Vec<_>>(),
                    );
                }
                let scrambles = u64::from(config.scramble_count().get());
                let stats = (0..width)
                    .map(|j| {
                        let values = means.iter().map(|v| v[j]).collect::<Vec<_>>();
                        DeterministicStatistics::from_ordered_values_two_pass(&values)
                    })
                    .collect::<Vec<_>>();
                (
                    stats,
                    scrambles,
                    u128::from(count)
                        * u128::from(scrambles)
                        * if config.variance_reduction().antithetic() {
                            2
                        } else {
                            1
                        },
                )
            }
        };

        let values = statistics
            .iter()
            .map(|statistics| statistics.sum().total() / units as f64)
            .collect::<Vec<_>>();
        let errors = statistics
            .iter()
            .map(|statistics| estimator_error(*statistics, units))
            .collect::<Result<Vec<_>, _>>()?;
        if values.iter().chain(&errors).any(|value| !value.is_finite()) {
            return Err(invalid("lsv_market_risk_estimator").into());
        }

        let cash_start = 2;
        let discount_start = cash_start + cash_count;
        let repo_start = discount_start + discount_count;
        Ok(StochasticDividendLsvMarketRisk {
            price: StochasticDividendPrice {
                value: values[0],
                standard_error: errors[0],
                independent_sampling_units: units,
                evaluated_paths: paths,
                plan_fingerprint: self.fingerprint,
                scheme: self.scheme(),
            },
            delta: values[1],
            delta_standard_error: errors[1],
            cash_times: context.cash_times.into_boxed_slice(),
            cash_mean_adjoints: values[cash_start..discount_start].into(),
            cash_mean_standard_errors: errors[cash_start..discount_start].into(),
            discount_times: context.discount_times.into_boxed_slice(),
            discount_log_df_adjoints: values[discount_start..repo_start].into(),
            discount_log_df_standard_errors: errors[discount_start..repo_start].into(),
            repo_spread_times: context.repo_spread_times.into_boxed_slice(),
            repo_spread_log_df_adjoints: values[repo_start..].into(),
            repo_spread_log_df_standard_errors: errors[repo_start..].into(),
            method: METHOD,
        })
    }

    fn sample_lsv_market_risk(
        &self,
        context: &ReverseContext,
        mut z: Vec<f64>,
        bridge: Option<&BrownianBridgePlan>,
        antithetic: bool,
        out: &mut [f64],
    ) -> Result<(), MonteCarloError> {
        let expected = 2
            + context.cash_times.len()
            + context.discount_times.len()
            + context.repo_spread_times.len();
        if out.len() != expected {
            return Err(invalid("lsv_market_risk_width").into());
        }
        if let Some(bridge) = bridge {
            let count = self.random_factor_count();
            for factor in 0..count {
                let input = z
                    .iter()
                    .skip(factor)
                    .step_by(count)
                    .copied()
                    .collect::<Vec<_>>();
                let output = bridge
                    .apply_one_factor(&input)
                    .map_err(|error| MonteCarloError::LocalVol(error.into()))?;
                for (step, value) in output.into_iter().enumerate() {
                    z[count * step + factor] = value;
                }
            }
        }

        out.fill(0.0);
        for &sign in if antithetic {
            &[1.0, -1.0][..]
        } else {
            &[1.0][..]
        } {
            let shocks = z.iter().map(|value| sign * value).collect::<Vec<_>>();
            let states = self.path.evolve_path(&shocks)?;
            let spots = self
                .path
                .nodes()
                .iter()
                .zip(&states)
                .map(|(node, state)| node.spots(*state))
                .collect::<Result<Vec<_>, _>>()?;
            let (payoff, seeds) = self
                .base
                .hybrid_spot_payoff_adjoints(self.path.times(), &spots)?;
            let reverse = context.market_reconstruction_pullback(&states, &seeds, payoff)?;
            let cash_count = context.cash_times.len();
            let discount_count = context.discount_times.len();
            let cash_source = 5;
            let discount_source = cash_source + cash_count;
            let repo_source = discount_source + discount_count;
            if reverse.len() != repo_source + context.repo_spread_times.len() {
                return Err(invalid("lsv_market_reverse_width").into());
            }

            out[0] += payoff;
            out[1] += reverse[0];
            let cash_target = 2;
            for (target, source) in out[cash_target..cash_target + cash_count]
                .iter_mut()
                .zip(&reverse[cash_source..discount_source])
            {
                *target += source;
            }
            let discount_target = cash_target + cash_count;
            for (target, source) in out[discount_target..discount_target + discount_count]
                .iter_mut()
                .zip(&reverse[discount_source..repo_source])
            {
                *target += source;
            }
            let repo_target = discount_target + discount_count;
            for (target, source) in out[repo_target..].iter_mut().zip(&reverse[repo_source..]) {
                *target += source;
            }
        }
        if antithetic {
            for value in out {
                *value *= 0.5;
            }
        }
        Ok(())
    }
}

fn estimator_error(
    statistics: DeterministicStatistics,
    units: u64,
) -> Result<f64, MonteCarloError> {
    let variance = statistics
        .moments()
        .sample_variance()
        .ok_or(MonteCarloError::InsufficientSamplingUnits { count: units })?;
    let error = (variance / units as f64).sqrt();
    if !error.is_finite() {
        return Err(invalid("lsv_market_standard_error").into());
    }
    Ok(error)
}
