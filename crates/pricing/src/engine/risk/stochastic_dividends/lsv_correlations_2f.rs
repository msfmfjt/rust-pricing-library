//! Correlation risk for 2F Bergomi stochastic-dividend residual LSV.
//!
//! Spot/volatility and volatility-factor correlations belong to the marginal
//! residual-equity LSV calibration and require full particle recalibration.
//! Dividend/volatility correlations affect only the joint pricing Brownian
//! system and therefore reuse the calibrated leverage surface.

use super::*;
use crate::models::Bergomi2Factor;

const METHOD: &str = "buehler-residual-lsv-common-noise-bergomi-2f-correlation-v1";
const WIDTH: usize = 6; // Price plus five correlation derivatives.

#[derive(Clone, Debug, PartialEq)]
pub struct StochasticDividendLsvBergomi2FactorCorrelationRisk {
    pub price: StochasticDividendPrice,
    pub parameter_labels: Box<[String]>,
    pub derivatives: Box<[f64]>,
    pub standard_errors: Box<[f64]>,
    pub correlation_bumps: Box<[f64]>,
    pub method: &'static str,
}

impl StochasticDividendLsvBergomi2FactorCorrelationRisk {
    #[must_use]
    pub fn spot_volatility_correlation_derivatives(&self) -> [f64; 2] {
        [self.derivatives[0], self.derivatives[1]]
    }
    #[must_use]
    pub fn factor_correlation_derivative(&self) -> f64 {
        self.derivatives[2]
    }
    #[must_use]
    pub fn dividend_volatility_correlation_derivatives(&self) -> [f64; 2] {
        [self.derivatives[3], self.derivatives[4]]
    }
}

struct Scenario {
    path: StochasticDividendPathPlan,
}

impl StochasticDividendPricingPlan {
    pub fn evaluate_lsv_bergomi_two_factor_correlation_risk(
        &self,
        spot_volatility_correlation_bumps: [f64; 2],
        factor_correlation_bump: f64,
        dividend_volatility_correlation_bumps: [f64; 2],
    ) -> Result<StochasticDividendLsvBergomi2FactorCorrelationRisk, MonteCarloError> {
        if !self.risk_supported {
            return Err(MonteCarloError::UnsupportedRiskForModel {
                model: "stochastic-dividend LSV discontinuous payoff requires explicit smoothing",
            });
        }
        let (calibration, dividend_correlations) = match self.lsv.as_ref() {
            Some(StochasticDividendLsvCalibration::Two {
                calibration,
                dividend_volatility_correlations,
                ..
            }) => (calibration, *dividend_volatility_correlations),
            Some(StochasticDividendLsvCalibration::One { .. }) => {
                return Err(MonteCarloError::UnsupportedRiskForModel {
                    model: "2F Bergomi LSV correlation risk requires a 2F plan",
                });
            }
            None => {
                return Err(MonteCarloError::UnsupportedRiskForModel {
                    model: "2F Bergomi LSV correlation risk requires a stochastic-dividend residual LSV plan",
                });
            }
        };

        let factor = calibration.factor();
        let spot_correlations = factor.spot_correlations();
        validate_correlation_bump(
            spot_correlations[0],
            spot_volatility_correlation_bumps[0],
            "lsv_bergomi_two_factor_spot_volatility_correlation_bump_0",
        )?;
        validate_correlation_bump(
            spot_correlations[1],
            spot_volatility_correlation_bumps[1],
            "lsv_bergomi_two_factor_spot_volatility_correlation_bump_1",
        )?;
        validate_correlation_bump(
            factor.factor_correlation(),
            factor_correlation_bump,
            "lsv_bergomi_two_factor_factor_correlation_bump",
        )?;
        validate_correlation_bump(
            dividend_correlations[0],
            dividend_volatility_correlation_bumps[0],
            "lsv_bergomi_two_factor_dividend_volatility_correlation_bump_0",
        )?;
        validate_correlation_bump(
            dividend_correlations[1],
            dividend_volatility_correlation_bumps[1],
            "lsv_bergomi_two_factor_dividend_volatility_correlation_bump_1",
        )?;

        let executor = DeterministicExecutor::new(self.policy)?;
        let make = |spot: [f64; 2], factor_correlation: f64| {
            Bergomi2Factor::new(
                factor.mean_reversions(),
                factor.vol_of_vol(),
                factor.mixing_weight(),
                spot,
                factor_correlation,
            )
            .map_err(|_| invalid("lsv_bergomi_two_factor_bumped_correlation_model"))
        };

        let scenarios = [
            self.recalibrated_two_factor_correlation_scenario(
                calibration,
                make(
                    [
                        spot_correlations[0] - spot_volatility_correlation_bumps[0],
                        spot_correlations[1],
                    ],
                    factor.factor_correlation(),
                )?,
                dividend_correlations,
                &executor,
            )?,
            self.recalibrated_two_factor_correlation_scenario(
                calibration,
                make(
                    [
                        spot_correlations[0] + spot_volatility_correlation_bumps[0],
                        spot_correlations[1],
                    ],
                    factor.factor_correlation(),
                )?,
                dividend_correlations,
                &executor,
            )?,
            self.recalibrated_two_factor_correlation_scenario(
                calibration,
                make(
                    [
                        spot_correlations[0],
                        spot_correlations[1] - spot_volatility_correlation_bumps[1],
                    ],
                    factor.factor_correlation(),
                )?,
                dividend_correlations,
                &executor,
            )?,
            self.recalibrated_two_factor_correlation_scenario(
                calibration,
                make(
                    [
                        spot_correlations[0],
                        spot_correlations[1] + spot_volatility_correlation_bumps[1],
                    ],
                    factor.factor_correlation(),
                )?,
                dividend_correlations,
                &executor,
            )?,
            self.recalibrated_two_factor_correlation_scenario(
                calibration,
                make(
                    spot_correlations,
                    factor.factor_correlation() - factor_correlation_bump,
                )?,
                dividend_correlations,
                &executor,
            )?,
            self.recalibrated_two_factor_correlation_scenario(
                calibration,
                make(
                    spot_correlations,
                    factor.factor_correlation() + factor_correlation_bump,
                )?,
                dividend_correlations,
                &executor,
            )?,
            self.fixed_two_factor_correlation_scenario(
                calibration,
                factor,
                [
                    dividend_correlations[0] - dividend_volatility_correlation_bumps[0],
                    dividend_correlations[1],
                ],
            )?,
            self.fixed_two_factor_correlation_scenario(
                calibration,
                factor,
                [
                    dividend_correlations[0] + dividend_volatility_correlation_bumps[0],
                    dividend_correlations[1],
                ],
            )?,
            self.fixed_two_factor_correlation_scenario(
                calibration,
                factor,
                [
                    dividend_correlations[0],
                    dividend_correlations[1] - dividend_volatility_correlation_bumps[1],
                ],
            )?,
            self.fixed_two_factor_correlation_scenario(
                calibration,
                factor,
                [
                    dividend_correlations[0],
                    dividend_correlations[1] + dividend_volatility_correlation_bumps[1],
                ],
            )?,
        ];
        let bumps = [
            spot_volatility_correlation_bumps[0],
            spot_volatility_correlation_bumps[1],
            factor_correlation_bump,
            dividend_volatility_correlation_bumps[0],
            dividend_volatility_correlation_bumps[1],
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
                    self.sample_lsv_bergomi_two_factor_correlation_risk(
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
                            self.sample_lsv_bergomi_two_factor_correlation_risk(
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
        if values.iter().chain(&errors).any(|x| !x.is_finite()) {
            return Err(invalid("lsv_bergomi_two_factor_correlation_risk_estimator").into());
        }

        Ok(StochasticDividendLsvBergomi2FactorCorrelationRisk {
            price: StochasticDividendPrice {
                value: values[0],
                standard_error: errors[0],
                independent_sampling_units: units,
                evaluated_paths: paths,
                plan_fingerprint: self.fingerprint,
                scheme: self.scheme(),
            },
            parameter_labels: [
                "spot_volatility_correlation[0]",
                "spot_volatility_correlation[1]",
                "volatility_factor_correlation",
                "dividend_volatility_correlation[0]",
                "dividend_volatility_correlation[1]",
            ]
            .map(str::to_owned)
            .into(),
            derivatives: values[1..].into(),
            standard_errors: errors[1..].into(),
            correlation_bumps: bumps.into(),
            method: METHOD,
        })
    }

    fn recalibrated_two_factor_correlation_scenario(
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

    fn fixed_two_factor_correlation_scenario(
        &self,
        calibration: &CalibratedBergomiLsv<Bergomi2Factor>,
        factor: Bergomi2Factor,
        dividend_volatility_correlations: [f64; 2],
    ) -> Result<Scenario, MonteCarloError> {
        let path = self.path.clone().with_bergomi_two_factor_lsv(
            factor,
            dividend_volatility_correlations,
            calibration.surface().clone(),
        )?;
        Ok(Scenario { path })
    }

    #[allow(clippy::too_many_arguments)]
    fn sample_lsv_bergomi_two_factor_correlation_risk(
        &self,
        scenarios: &[Scenario; 10],
        bumps: [f64; 5],
        mut z: Vec<f64>,
        bridge: Option<&BrownianBridgePlan>,
        antithetic: bool,
        out: &mut [f64],
    ) -> Result<(), MonteCarloError> {
        if out.len() != WIDTH {
            return Err(invalid("lsv_bergomi_two_factor_correlation_risk_width").into());
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
            out[0] +=
                self.discounted_payoff_for_two_factor_correlation_path(&self.path, &shocks)?;
            for parameter in 0..5 {
                let down = self.discounted_payoff_for_two_factor_correlation_path(
                    &scenarios[2 * parameter].path,
                    &shocks,
                )?;
                let up = self.discounted_payoff_for_two_factor_correlation_path(
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

    fn discounted_payoff_for_two_factor_correlation_path(
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

fn validate_correlation_bump(
    correlation: f64,
    bump: f64,
    field: &'static str,
) -> Result<(), MonteCarloError> {
    if !correlation.is_finite()
        || !bump.is_finite()
        || bump <= 0.0
        || correlation - bump < -1.0
        || correlation + bump > 1.0
        || correlation - bump >= correlation
        || correlation + bump <= correlation
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
        return Err(invalid("lsv_bergomi_two_factor_correlation_standard_error").into());
    }
    Ok(error)
}
