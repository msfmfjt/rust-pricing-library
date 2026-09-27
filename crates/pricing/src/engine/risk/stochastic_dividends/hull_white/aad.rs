//! HW dividend AAD. Basic risk fixes rate parameters; the extended method
//! differentiates the covariance Cholesky and all conditional rate-dependent cash.
use super::super::StochasticDividendAadRisk;
use super::*;
use crate::engine::processes::stochastic_dividends::hull_white::correlation_sensitivity::HullWhiteCorrelationSensitivityContext;
use crate::engine::processes::stochastic_dividends::hull_white::rate_sensitivity::HullWhiteRateSensitivityContext;
use crate::engine::processes::stochastic_dividends::hull_white::reverse::HullWhiteReverseContext;

#[derive(Clone, Copy, PartialEq, Eq)]
enum AadScope {
    Basic,
    RateParameters,
    Correlations,
}

impl StochasticDividendHullWhitePricingPlan {
    /// Reverse Spot, residual volatility, Buehler parameters, Q cash means and
    /// curve log-DF pillars. Includes full conditional funding and stochastic
    /// payment discounting. HW parameters/correlations and the time grid are fixed.
    /// Explicit smoothing is required for discontinuous contractual payoffs.
    pub fn evaluate_aad(&self) -> Result<StochasticDividendAadRisk, MonteCarloError> {
        self.evaluate_aad_scope(AadScope::Basic)
    }

    /// Extend basic stochastic-dividend risk with the Hull--White mean
    /// reversion and each piecewise rate-volatility knot. The fixed-grid path,
    /// conditional cash claims, initial reserve, and delayed-payment discount
    /// are differentiated together. The simulated covariance must be full rank.
    pub fn evaluate_hull_white_aad(&self) -> Result<StochasticDividendAadRisk, MonteCarloError> {
        self.evaluate_aad_scope(AadScope::RateParameters)
    }

    /// Append raw equity/dividend, equity/rate, and dividend/rate correlation
    /// partials to `evaluate_hull_white_aad()`. Each direction changes one
    /// symmetric pair while holding all other inputs and normal coordinates
    /// fixed. Requires instantaneous correlation variance pivots and normalized
    /// simulated covariance Cholesky diagonals above 1e-10.
    pub fn evaluate_correlation_aad(&self) -> Result<StochasticDividendAadRisk, MonteCarloError> {
        self.evaluate_aad_scope(AadScope::Correlations)
    }

    fn evaluate_aad_scope(
        &self,
        scope: AadScope,
    ) -> Result<StochasticDividendAadRisk, MonteCarloError> {
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
        let correlation_context = if scope == AadScope::Correlations {
            Some(HullWhiteCorrelationSensitivityContext::new(
                &self.path,
                &self.rates,
                self.equity_rate_correlation,
                self.dividend_rate_correlation,
            )?)
        } else {
            None
        };
        let rate_context = if scope != AadScope::Basic {
            Some(HullWhiteRateSensitivityContext::new(
                &self.path,
                &self.rates,
                self.equity_rate_correlation,
                self.dividend_rate_correlation,
                self.payment_time,
            )?)
        } else {
            None
        };
        let rate_width = rate_context.as_ref().map_or(0, |c| c.labels().len());
        let correlation_width = correlation_context.as_ref().map_or(0, |c| c.labels().len());
        let width = 1 + context.labels.len() + rate_width + correlation_width;
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
                        rate_context.as_ref(),
                        correlation_context.as_ref(),
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
                                rate_context.as_ref(),
                                correlation_context.as_ref(),
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
        let mut parameter_labels = context.labels.clone();
        if let Some(rate_context) = &rate_context {
            parameter_labels.extend_from_slice(rate_context.labels());
        }
        if let Some(correlation_context) = &correlation_context {
            parameter_labels.extend(correlation_context.labels().iter().map(|s| (*s).to_owned()));
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
            parameter_labels: parameter_labels.into_boxed_slice(),
            derivatives: means[1..].into(),
            standard_errors: errors[1..].into(),
            cash_times: context.cash_times.into_boxed_slice(),
            discount_times: context.discount_times.into_boxed_slice(),
            repo_spread_times: context.repo_spread_times.into_boxed_slice(),
            method: match scope {
                AadScope::Basic => "buehler-bs-hw-cash-payoff-reverse-fixed-rates-correlation-v1",
                AadScope::RateParameters => {
                    "buehler-bs-hw-cash-payoff-forward-rate-parameter-adjoint-v1"
                }
                AadScope::Correlations => {
                    "buehler-bs-hw-cash-payoff-forward-correlation-adjoint-v1"
                }
            },
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn sample_aad(
        &self,
        context: &HullWhiteReverseContext,
        rate_context: Option<&HullWhiteRateSensitivityContext>,
        correlation_context: Option<&HullWhiteCorrelationSensitivityContext>,
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
            let rate_state_tangents = rate_context
                .map(|context| context.evolve_rate_tangents(&self.path, &shocks, &states))
                .transpose()?;
            let correlation_state_tangents = correlation_context
                .map(|context| context.evolve_tangents(&self.path, &shocks, &states))
                .transpose()?;
            let spots = states
                .iter()
                .enumerate()
                .map(|(index, state)| self.path.spots(index, *state))
                .collect::<Result<Vec<_>, _>>()?;
            let rate_spot_tangents = match (rate_context, rate_state_tangents.as_ref()) {
                (Some(context), Some(state_tangents)) => states
                    .iter()
                    .enumerate()
                    .map(|(index, state)| {
                        context.spot_derivatives(&self.path, index, *state, &state_tangents[index])
                    })
                    .collect::<Result<Vec<_>, _>>()?,
                _ => Vec::new(),
            };
            let (payoff, payoff_seeds) = self
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
            let discounted_seeds = payoff_seeds
                .iter()
                .map(|(post, pre)| (relative * post, relative * pre))
                .collect::<Vec<_>>();
            let reverse = context.pullback(
                &self.path,
                &shocks,
                &states,
                &discounted_seeds,
                discounted_payoff,
            )?;
            for (target, source) in out[1..].iter_mut().zip(reverse) {
                *target += source;
            }
            if let (Some(rate_context), Some(state_tangents)) =
                (rate_context, rate_state_tangents.as_ref())
            {
                let parameter_offset = 1 + context.labels.len();
                for p in 0..rate_context.labels().len() {
                    let payoff_tangent = payoff_seeds
                        .iter()
                        .zip(&rate_spot_tangents)
                        .map(|((post, pre), tangent)| (post + pre) * tangent[p])
                        .sum::<f64>();
                    let terminal_tangent = state_tangents
                        .last()
                        .ok_or(invalid("terminal_rate_tangent"))?[p];
                    let log_discount_tangent = rate_context.payment_log_constant_derivatives()[p]
                        - terminal_tangent.integrated_rate_factor
                        - rate_context.payment_duration_derivatives()[p] * terminal.rate_factor()
                        - self.payment_duration * terminal_tangent.rate_factor;
                    out[parameter_offset + p] +=
                        relative * payoff_tangent + discounted_payoff * log_discount_tangent;
                }
            }
            if let (Some(correlation_context), Some(state_tangents)) =
                (correlation_context, correlation_state_tangents.as_ref())
            {
                let parameter_offset =
                    1 + context.labels.len() + rate_context.map_or(0, |c| c.labels().len());
                let mut payoff_tangent = [0.0; 3];
                for (index, state) in states.iter().enumerate() {
                    let tangents = correlation_context.spot_derivatives(
                        &self.path,
                        index,
                        *state,
                        &state_tangents[index],
                    )?;
                    let (post_seed, pre_seed) = payoff_seeds[index];
                    for (value, (post, pre)) in payoff_tangent.iter_mut().zip(tangents) {
                        *value += post_seed * post + pre_seed * pre;
                    }
                }
                let terminal_tangents = state_tangents
                    .last()
                    .ok_or(invalid("terminal_correlation_tangent"))?;
                for (p, tangent) in terminal_tangents.iter().enumerate() {
                    // Rate-only bond constants/durations are correlation invariant.
                    let log_discount_tangent = -tangent.integrated_rate_factor
                        - self.payment_duration * tangent.rate_factor;
                    out[parameter_offset + p] +=
                        relative * payoff_tangent[p] + discounted_payoff * log_discount_tangent;
                }
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
