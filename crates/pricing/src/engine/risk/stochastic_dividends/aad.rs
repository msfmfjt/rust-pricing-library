//! Payoff adjoint and analytic reverse of the finite positive-split algorithm.

use super::*;
use crate::engine::processes::stochastic_dividends::reverse::ReverseContext;

/// First-order derivatives at fixed dates, grid and smoothing.
/// The method and labels identify active Bergomi parameters and correlations.
/// No market-IV recalibration is implied.
#[derive(Clone, Debug, PartialEq)]
pub struct StochasticDividendAadRisk {
    pub price: StochasticDividendPrice,
    pub parameter_labels: Box<[String]>,
    pub derivatives: Box<[f64]>,
    /// MC: independent (possibly antithetic) units. RQMC: scramble means only.
    /// Neither includes time-discretization or parameter uncertainty.
    pub standard_errors: Box<[f64]>,
    pub cash_times: Box<[f64]>,
    pub discount_times: Box<[f64]>,
    pub repo_spread_times: Box<[f64]>,
    pub method: &'static str,
}
impl StochasticDividendAadRisk {
    #[must_use]
    pub fn delta(&self) -> f64 {
        self.derivatives[0]
    }
    /// dPrice/dsigma0, not market-IV Vega or VegaKT.
    #[must_use]
    pub fn initial_volatility_vega(&self) -> f64 {
        self.derivatives[1]
    }
    #[must_use]
    pub fn initial_volatility_vega_per_vol_point(&self) -> f64 {
        0.01 * self.derivatives[1]
    }
    #[must_use]
    pub fn dividend_volatility_vega_per_vol_point(&self) -> f64 {
        0.01 * self.derivatives[4]
    }
    #[must_use]
    pub fn cash_mean_adjoints(&self) -> &[f64] {
        &self.derivatives[5..5 + self.cash_times.len()]
    }
    #[must_use]
    pub fn discount_node_dv01(&self) -> Vec<f64> {
        let start = 5 + self.cash_times.len();
        self.derivatives[start..start + self.discount_times.len()]
            .iter()
            .zip(&self.discount_times)
            .map(|(d, t)| -1e-4 * t * d)
            .collect()
    }
    #[must_use]
    pub fn repo_spread_node_dv01(&self) -> Vec<f64> {
        let start = 5 + self.cash_times.len() + self.discount_times.len();
        self.derivatives[start..start + self.repo_spread_times.len()]
            .iter()
            .zip(&self.repo_spread_times)
            .map(|(d, t)| -1e-4 * t * d)
            .collect()
    }
}

#[derive(Clone, Copy)]
enum AadScope {
    Basic,
    Bergomi,
    Correlation,
}

impl StochasticDividendPricingPlan {
    /// Reverse of the actual discretized path, including stochastic future
    /// reserves, cash means, carry and payment discounting. Request Greeks remain
    /// rejected at construction; use this explicit method on a price-only plan.
    /// Unsmooth discontinuous payoffs fail before generating any paths.
    pub fn evaluate_aad(&self) -> Result<StochasticDividendAadRisk, MonteCarloError> {
        self.evaluate_aad_scope(AadScope::Basic)
    }

    /// Extend basic risk by 1F/2F Bergomi mean reversions, vol-of-vol and (2F)
    /// mixing weight. All correlations remain fixed. Integrated correlation
    /// pivots and weight variance must exceed 1e-10; otherwise fail before sampling.
    /// Existing evaluate_aad remains available at singular covariance boundaries.
    pub fn evaluate_bergomi_aad(&self) -> Result<StochasticDividendAadRisk, MonteCarloError> {
        self.evaluate_aad_scope(AadScope::Bergomi)
    }

    /// Append all raw Brownian-correlation partials, varying one symmetric
    /// off-diagonal pair while holding other entries fixed. BS includes basic
    /// AAD; Bergomi includes model-parameter AAD. Instantaneous and integrated
    /// correlation pivots must exceed 1e-10. No projection or market recalibration.
    pub fn evaluate_correlation_aad(&self) -> Result<StochasticDividendAadRisk, MonteCarloError> {
        self.evaluate_aad_scope(AadScope::Correlation)
    }

    fn evaluate_aad_scope(
        &self,
        scope: AadScope,
    ) -> Result<StochasticDividendAadRisk, MonteCarloError> {
        if !self.risk_supported {
            return Err(MonteCarloError::UnsupportedRiskForModel {
                model: "stochastic-dividend discontinuous payoff requires explicit smoothing",
            });
        }
        let mut context = ReverseContext::new(&self.path, &self.market, self.payment_time)?;
        match scope {
            AadScope::Basic => {}
            AadScope::Bergomi => context.enable_bergomi_parameters(&self.path)?,
            AadScope::Correlation => context.enable_correlations(&self.path)?,
        }
        let width = 1 + context.labels.len();
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
                    self.sample_aad(
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
                let mut means = Vec::new();
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
                            self.sample_aad(
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
                    .collect();
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
        let means = statistics
            .iter()
            .map(|s| s.sum().total() / units as f64)
            .collect::<Vec<_>>();
        let errors = statistics
            .iter()
            .map(|s| {
                let variance = s
                    .moments()
                    .sample_variance()
                    .ok_or(MonteCarloError::InsufficientSamplingUnits { count: units })?;
                Ok((variance / units as f64).sqrt())
            })
            .collect::<Result<Vec<_>, MonteCarloError>>()?;
        if means.iter().chain(&errors).any(|v| !v.is_finite()) {
            return Err(invalid("risk_estimator").into());
        }
        Ok(StochasticDividendAadRisk {
            price: StochasticDividendPrice {
                value: means[0],
                standard_error: errors[0],
                independent_sampling_units: units,
                evaluated_paths: paths,
                plan_fingerprint: self.fingerprint,
                scheme: self.scheme(),
            },
            parameter_labels: context.labels.into_boxed_slice(),
            derivatives: means[1..].into(),
            standard_errors: errors[1..].into(),
            cash_times: context.cash_times.into_boxed_slice(),
            discount_times: context.discount_times.into_boxed_slice(),
            repo_spread_times: context.repo_spread_times.into_boxed_slice(),
            method: match scope {
                AadScope::Basic => "buehler-split-payoff-reverse-fixed-correlation-v1",
                AadScope::Bergomi => "buehler-bergomi-parameter-reverse-fixed-correlation-v1",
                AadScope::Correlation => "buehler-joint-correlation-reverse-v1",
            },
        })
    }

    fn sample_aad(
        &self,
        context: &ReverseContext,
        mut z: Vec<f64>,
        bridge: Option<&BrownianBridgePlan>,
        antithetic: bool,
        out: &mut [f64],
    ) -> Result<(), MonteCarloError> {
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
                    .map_err(|e| MonteCarloError::LocalVol(e.into()))?;
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
            let shocks = z.iter().map(|v| sign * v).collect::<Vec<_>>();
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
            out[0] += payoff;
            let reverse = context.pullback(&self.path, &shocks, &states, &seeds, payoff)?;
            for (target, source) in out[1..].iter_mut().zip(reverse) {
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
