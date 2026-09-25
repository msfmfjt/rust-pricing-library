//! Full-recalibration finite-bump 2F Bergomi parameter risk for residual-equity LSV.

use super::*;
use crate::models::Bergomi2Factor;

const METHOD: &str = "buehler-residual-lsv-common-noise-full-recalibration-bergomi-2f-v1";
const WIDTH: usize = 5; // Price, k1, k2, vol-of-vol, mixing weight.

#[derive(Clone, Debug, PartialEq)]
pub struct StochasticDividendLsvBergomi2FactorRisk {
    pub price: StochasticDividendPrice,
    pub parameter_labels: Box<[String]>,
    pub derivatives: Box<[f64]>,
    pub standard_errors: Box<[f64]>,
    pub parameter_bumps: Box<[f64]>,
    pub method: &'static str,
}

impl StochasticDividendLsvBergomi2FactorRisk {
    #[must_use]
    pub fn mean_reversion_derivatives(&self) -> [f64; 2] {
        [self.derivatives[0], self.derivatives[1]]
    }

    #[must_use]
    pub fn vol_of_vol_derivative(&self) -> f64 {
        self.derivatives[2]
    }

    #[must_use]
    pub fn mixing_weight_derivative(&self) -> f64 {
        self.derivatives[3]
    }
}

struct Scenario {
    path: StochasticDividendPathPlan,
}

impl StochasticDividendPricingPlan {
    /// Central finite differences of 2F Bergomi model parameters with full
    /// particle recalibration for every bumped scenario.
    pub fn evaluate_lsv_bergomi_two_factor_parameter_risk(
        &self,
        mean_reversion_bumps: [f64; 2],
        vol_of_vol_bump: f64,
        mixing_weight_bump: f64,
    ) -> Result<StochasticDividendLsvBergomi2FactorRisk, MonteCarloError> {
        if !self.risk_supported {
            return Err(MonteCarloError::UnsupportedRiskForModel {
                model: "stochastic-dividend LSV discontinuous payoff requires explicit smoothing",
            });
        }
        let (calibration, dividend_volatility_correlations) = match self.lsv.as_ref() {
            Some(StochasticDividendLsvCalibration::Two {
                calibration,
                dividend_volatility_correlations,
                ..
            }) => (calibration, *dividend_volatility_correlations),
            Some(StochasticDividendLsvCalibration::One { .. }) => {
                return Err(MonteCarloError::UnsupportedRiskForModel {
                    model: "2F Bergomi LSV parameter risk requires a 2F plan",
                });
            }
            None => {
                return Err(MonteCarloError::UnsupportedRiskForModel {
                    model: "2F Bergomi LSV parameter risk requires a stochastic-dividend residual LSV plan",
                });
            }
        };

        let factor = calibration.factor();
        let k = factor.mean_reversions();
        validate_nonnegative_bump(
            k[0],
            mean_reversion_bumps[0],
            "lsv_bergomi_two_factor_mean_reversion_bump_0",
        )?;
        validate_nonnegative_bump(
            k[1],
            mean_reversion_bumps[1],
            "lsv_bergomi_two_factor_mean_reversion_bump_1",
        )?;
        validate_nonnegative_bump(
            factor.vol_of_vol(),
            vol_of_vol_bump,
            "lsv_bergomi_two_factor_vol_of_vol_bump",
        )?;
        validate_unit_interval_bump(
            factor.mixing_weight(),
            mixing_weight_bump,
            "lsv_bergomi_two_factor_mixing_weight_bump",
        )?;

        let spot_correlations = factor.spot_correlations();
        let factor_correlation = factor.factor_correlation();
        let executor = DeterministicExecutor::new(self.policy)?;
        let make = |mean_reversions: [f64; 2], vol_of_vol: f64, mixing_weight: f64| {
            Bergomi2Factor::new(
                mean_reversions,
                vol_of_vol,
                mixing_weight,
                spot_correlations,
                factor_correlation,
            )
            .map_err(|_| invalid("lsv_bergomi_two_factor_bumped_model"))
        };

        let scenarios = [
            self.recalibrated_two_factor_scenario(
                calibration,
                make(
                    [k[0] - mean_reversion_bumps[0], k[1]],
                    factor.vol_of_vol(),
                    factor.mixing_weight(),
                )?,
                dividend_volatility_correlations,
                &executor,
            )?,
            self.recalibrated_two_factor_scenario(
                calibration,
                make(
                    [k[0] + mean_reversion_bumps[0], k[1]],
                    factor.vol_of_vol(),
                    factor.mixing_weight(),
                )?,
                dividend_volatility_correlations,
                &executor,
            )?,
            self.recalibrated_two_factor_scenario(
                calibration,
                make(
                    [k[0], k[1] - mean_reversion_bumps[1]],
                    factor.vol_of_vol(),
                    factor.mixing_weight(),
                )?,
                dividend_volatility_correlations,
                &executor,
            )?,
            self.recalibrated_two_factor_scenario(
                calibration,
                make(
                    [k[0], k[1] + mean_reversion_bumps[1]],
                    factor.vol_of_vol(),
                    factor.mixing_weight(),
                )?,
                dividend_volatility_correlations,
                &executor,
            )?,
            self.recalibrated_two_factor_scenario(
                calibration,
                make(
                    k,
                    factor.vol_of_vol() - vol_of_vol_bump,
                    factor.mixing_weight(),
                )?,
                dividend_volatility_correlations,
                &executor,
            )?,
            self.recalibrated_two_factor_scenario(
                calibration,
                make(
                    k,
                    factor.vol_of_vol() + vol_of_vol_bump,
                    factor.mixing_weight(),
                )?,
                dividend_volatility_correlations,
                &executor,
            )?,
            self.recalibrated_two_factor_scenario(
                calibration,
                make(
                    k,
                    factor.vol_of_vol(),
                    factor.mixing_weight() - mixing_weight_bump,
                )?,
                dividend_volatility_correlations,
                &executor,
            )?,
            self.recalibrated_two_factor_scenario(
                calibration,
                make(
                    k,
                    factor.vol_of_vol(),
                    factor.mixing_weight() + mixing_weight_bump,
                )?,
                dividend_volatility_correlations,
                &executor,
            )?,
        ];
        let bumps = [
            mean_reversion_bumps[0],
            mean_reversion_bumps[1],
            vol_of_vol_bump,
            mixing_weight_bump,
        ];

        let dimension = self.path.random_dimension();
        let (statistics, units, paths) = match self.engine {
            EngineConfig::PseudoMonteCarlo(config) => {
                let count = config.independent_sampling_units().get();
                let bridge = self.bridge(config.variance_reduction())?;
                let rng = Philox4x32::from_seed(config.master_seed());
                let stats = executor.try_map_reduce_statistics_vector(count, WIDTH, |p, out| {
                    let z = (0..dimension)
                        .map(|d| {
                            rng.standard_normal(RandomCoordinate::new(
                                p,
                                d,
                                RandomDomain::Valuation,
                            ))
                        })
                        .collect();
                    self.sample_lsv_bergomi_two_factor_parameter_risk(
                        &scenarios,
                        bumps,
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
                        executor.try_map_reduce_statistics_vector(count, WIDTH, |p, out| {
                            let z = (0..dimension)
                                .map(|d| {
                                    let u = qmc
                                        .uniform(scramble, p, d)
                                        .map_err(|_| invalid("rqmc_uniform"))?;
                                    inverse_standard_normal(u).map_err(|_| invalid("rqmc_normal"))
                                })
                                .collect::<Result<Vec<_>, _>>()?;
                            self.sample_lsv_bergomi_two_factor_parameter_risk(
                                &scenarios,
                                bumps,
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
                let stats = (0..WIDTH)
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
            .map(|s| s.sum().total() / units as f64)
            .collect::<Vec<_>>();
        let errors = statistics
            .iter()
            .map(|s| estimator_error(*s, units))
            .collect::<Result<Vec<_>, _>>()?;
        if values.iter().chain(&errors).any(|value| !value.is_finite()) {
            return Err(invalid("lsv_bergomi_two_factor_parameter_risk_estimator").into());
        }

        Ok(StochasticDividendLsvBergomi2FactorRisk {
            price: StochasticDividendPrice {
                value: values[0],
                standard_error: errors[0],
                independent_sampling_units: units,
                evaluated_paths: paths,
                plan_fingerprint: self.fingerprint,
                scheme: self.scheme(),
            },
            parameter_labels: [
                "bergomi_mean_reversion[0]",
                "bergomi_mean_reversion[1]",
                "bergomi_vol_of_vol",
                "bergomi_mixing_weight",
            ]
            .map(str::to_owned)
            .into(),
            derivatives: values[1..].into(),
            standard_errors: errors[1..].into(),
            parameter_bumps: bumps.into(),
            method: METHOD,
        })
    }

    fn recalibrated_two_factor_scenario(
        &self,
        calibration: &CalibratedBergomiLsv<Bergomi2Factor>,
        factor: Bergomi2Factor,
        dividend_volatility_correlations: [f64; 2],
        executor: &DeterministicExecutor,
    ) -> Result<Scenario, MonteCarloError> {
        let bumped = calibrate_bergomi_lsv_parallel(
            calibration.target(),
            factor,
            self.path.risky_spot(),
            calibration.config().clone(),
            executor,
        )?;
        let path = self.path.clone().with_bergomi_two_factor_lsv(
            factor,
            dividend_volatility_correlations,
            bumped.surface().clone(),
        )?;
        Ok(Scenario { path })
    }

    #[allow(clippy::too_many_arguments)]
    fn sample_lsv_bergomi_two_factor_parameter_risk(
        &self,
        scenarios: &[Scenario; 8],
        bumps: [f64; 4],
        mut z: Vec<f64>,
        bridge: Option<&BrownianBridgePlan>,
        antithetic: bool,
        out: &mut [f64],
    ) -> Result<(), MonteCarloError> {
        if out.len() != WIDTH {
            return Err(invalid("lsv_bergomi_two_factor_parameter_risk_width").into());
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
            let shocks = z.iter().map(|value| sign * value).collect::<Vec<_>>();
            out[0] += self.discounted_payoff_for_two_factor_parameter_path(&self.path, &shocks)?;
            for parameter in 0..4 {
                let down = self.discounted_payoff_for_two_factor_parameter_path(
                    &scenarios[2 * parameter].path,
                    &shocks,
                )?;
                let up = self.discounted_payoff_for_two_factor_parameter_path(
                    &scenarios[2 * parameter + 1].path,
                    &shocks,
                )?;
                out[1 + parameter] += (up - down) / (2.0 * bumps[parameter]);
            }
        }
        if antithetic {
            for value in out {
                *value *= 0.5;
            }
        }
        Ok(())
    }

    fn discounted_payoff_for_two_factor_parameter_path(
        &self,
        path: &StochasticDividendPathPlan,
        shocks: &[f64],
    ) -> Result<f64, MonteCarloError> {
        let states = path.evolve_path(shocks)?;
        let spots = path
            .nodes()
            .iter()
            .zip(&states)
            .map(|(node, state)| node.spots(*state))
            .collect::<Result<Vec<_>, _>>()?;
        self.base.hybrid_spot_payoff(path.times(), &spots)
    }
}

fn validate_nonnegative_bump(
    parameter: f64,
    bump: f64,
    field: &'static str,
) -> Result<(), MonteCarloError> {
    if !parameter.is_finite()
        || !bump.is_finite()
        || bump <= 0.0
        || parameter - bump < 0.0
        || parameter - bump >= parameter
        || parameter + bump <= parameter
    {
        return Err(invalid(field).into());
    }
    Ok(())
}

fn validate_unit_interval_bump(
    parameter: f64,
    bump: f64,
    field: &'static str,
) -> Result<(), MonteCarloError> {
    if !parameter.is_finite()
        || !bump.is_finite()
        || bump <= 0.0
        || parameter - bump < 0.0
        || parameter + bump > 1.0
        || parameter - bump >= parameter
        || parameter + bump <= parameter
    {
        return Err(invalid(field).into());
    }
    Ok(())
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
        return Err(invalid("lsv_bergomi_two_factor_parameter_standard_error").into());
    }
    Ok(error)
}
