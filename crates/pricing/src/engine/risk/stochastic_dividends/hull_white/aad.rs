//! Basic HW dividend AAD. Stochastic rates are sampled exactly as for price;
//! rate-model parameters and driver correlations are fixed for this method.
use super::super::StochasticDividendAadRisk;
use super::*;
use crate::engine::processes::stochastic_dividends::hull_white::reverse::HullWhiteReverseContext;

impl StochasticDividendHullWhitePricingPlan {
    /// Reverse Spot, residual volatility, Buehler parameters, Q cash means and
    /// curve log-DF pillars. Includes full conditional funding and stochastic
    /// payment discounting. HW parameters/correlations and the time grid are fixed.
    /// Explicit smoothing is required for discontinuous contractual payoffs.
    pub fn evaluate_aad(&self) -> Result<StochasticDividendAadRisk, MonteCarloError> {
        if !self.risk_supported {
            return Err(MonteCarloError::UnsupportedRiskForModel {
                model: "HW stochastic-dividend discontinuous payoff requires explicit smoothing",
            });
        }
        let context = HullWhiteReverseContext::new(
            &self.path,
            &self.market,
            &self.rates,
            self.equity_rate_correlation,
            self.dividend_rate_correlation,
            self.payment_time,
        )?;
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
            method: "buehler-bs-hw-cash-payoff-reverse-fixed-rates-correlation-v1",
        })
    }

    fn sample_aad(
        &self,
        context: &HullWhiteReverseContext,
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
            let spots = states
                .iter()
                .enumerate()
                .map(|(index, state)| self.path.spots(index, *state))
                .collect::<Result<Vec<_>, _>>()?;
            let (payoff, seeds) = self
                .base
                .hybrid_spot_payoff_adjoints(self.path.times(), &spots)?;
            let terminal = states.last().ok_or(invalid("terminal_state"))?;
            let relative = self.payment_constant
                * (-terminal.integrated_rate_factor()
                    - self.payment_duration * terminal.rate_factor())
                .exp();
            positive(relative, "relative_payment_discount")?;
            let discounted_payoff = relative * payoff;
            out[0] += discounted_payoff;
            let seeds = seeds
                .iter()
                .map(|(post, pre)| (relative * post, relative * pre))
                .collect::<Vec<_>>();
            let reverse =
                context.pullback(&self.path, &shocks, &states, &seeds, discounted_payoff)?;
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
