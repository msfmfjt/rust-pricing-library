//! Full-recalibration finite-bump model-parameter risk for residual-equity LSV.
//!
//! This deliberately does not reuse fixed-leverage Bergomi AAD. Each scenario
//! reruns the finite-particle calibration with the same calibration seed, then
//! prices with common valuation normals.

use super::*;
use crate::models::Bergomi1Factor;

const METHOD: &str = "buehler-residual-lsv-common-noise-full-recalibration-bergomi-1f-v1";
const WIDTH: usize = 3; // Price, d/d mean reversion, d/d vol-of-vol.

#[derive(Clone, Debug, PartialEq)]
pub struct StochasticDividendLsvBergomiRisk {
    pub price: StochasticDividendPrice,
    pub parameter_labels: Box<[String]>,
    pub derivatives: Box<[f64]>,
    pub standard_errors: Box<[f64]>,
    pub parameter_bumps: Box<[f64]>,
    pub method: &'static str,
}

impl StochasticDividendLsvBergomiRisk {
    #[must_use]
    pub fn mean_reversion_derivative(&self) -> f64 {
        self.derivatives[0]
    }

    #[must_use]
    pub fn vol_of_vol_derivative(&self) -> f64 {
        self.derivatives[1]
    }
}

struct Scenario {
    path: StochasticDividendPathPlan,
}

impl StochasticDividendPricingPlan {
    /// Central finite differences of 1F Bergomi parameters with a full particle
    /// recalibration at each bump. Calibration seed, particle count, bandwidth,
    /// fallback policy, valuation random numbers, grid, dividend model and all
    /// correlations are held fixed.
    pub fn evaluate_lsv_bergomi_parameter_risk(
        &self,
        mean_reversion_bump: f64,
        vol_of_vol_bump: f64,
    ) -> Result<StochasticDividendLsvBergomiRisk, MonteCarloError> {
        if !self.risk_supported {
            return Err(MonteCarloError::UnsupportedRiskForModel {
                model: "stochastic-dividend LSV discontinuous payoff requires explicit smoothing",
            });
        }
        let (calibration, dividend_volatility_correlation) = match self.lsv.as_ref() {
            Some(StochasticDividendLsvCalibration::One {
                calibration,
                dividend_volatility_correlation,
                ..
            }) => (calibration, *dividend_volatility_correlation),
            Some(StochasticDividendLsvCalibration::Two { .. }) => {
                return Err(MonteCarloError::UnsupportedRiskForModel {
                    model: "1F Bergomi LSV parameter risk does not apply to a 2F plan",
                });
            }
            None => {
                return Err(MonteCarloError::UnsupportedRiskForModel {
                    model: "Bergomi LSV parameter risk requires a stochastic-dividend residual LSV plan",
                });
            }
        };

        let factor = calibration.factor();
        validate_parameter_bump(
            factor.mean_reversion(),
            mean_reversion_bump,
            "lsv_bergomi_mean_reversion_bump",
        )?;
        validate_parameter_bump(
            factor.vol_of_vol(),
            vol_of_vol_bump,
            "lsv_bergomi_vol_of_vol_bump",
        )?;

        let executor = DeterministicExecutor::new(self.policy)?;
        let scenarios = [
            self.recalibrated_one_factor_scenario(
                calibration,
                Bergomi1Factor::new(
                    factor.mean_reversion() - mean_reversion_bump,
                    factor.vol_of_vol(),
                    factor.correlation(),
                )
                .map_err(|_| invalid("lsv_bergomi_mean_reversion_down"))?,
                dividend_volatility_correlation,
                &executor,
            )?,
            self.recalibrated_one_factor_scenario(
                calibration,
                Bergomi1Factor::new(
                    factor.mean_reversion() + mean_reversion_bump,
                    factor.vol_of_vol(),
                    factor.correlation(),
                )
                .map_err(|_| invalid("lsv_bergomi_mean_reversion_up"))?,
                dividend_volatility_correlation,
                &executor,
            )?,
            self.recalibrated_one_factor_scenario(
                calibration,
                Bergomi1Factor::new(
                    factor.mean_reversion(),
                    factor.vol_of_vol() - vol_of_vol_bump,
                    factor.correlation(),
                )
                .map_err(|_| invalid("lsv_bergomi_vol_of_vol_down"))?,
                dividend_volatility_correlation,
                &executor,
            )?,
            self.recalibrated_one_factor_scenario(
                calibration,
                Bergomi1Factor::new(
                    factor.mean_reversion(),
                    factor.vol_of_vol() + vol_of_vol_bump,
                    factor.correlation(),
                )
                .map_err(|_| invalid("lsv_bergomi_vol_of_vol_up"))?,
                dividend_volatility_correlation,
                &executor,
            )?,
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
                    self.sample_lsv_bergomi_parameter_risk(
                        &scenarios,
                        [mean_reversion_bump, vol_of_vol_bump],
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
                            self.sample_lsv_bergomi_parameter_risk(
                                &scenarios,
                                [mean_reversion_bump, vol_of_vol_bump],
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
            return Err(invalid("lsv_bergomi_parameter_risk_estimator").into());
        }

        Ok(StochasticDividendLsvBergomiRisk {
            price: StochasticDividendPrice {
                value: values[0],
                standard_error: errors[0],
                independent_sampling_units: units,
                evaluated_paths: paths,
                plan_fingerprint: self.fingerprint,
                scheme: self.scheme(),
            },
            parameter_labels: ["bergomi_mean_reversion", "bergomi_vol_of_vol"]
                .map(str::to_owned)
                .into(),
            derivatives: values[1..].into(),
            standard_errors: errors[1..].into(),
            parameter_bumps: [mean_reversion_bump, vol_of_vol_bump].into(),
            method: METHOD,
        })
    }

    fn recalibrated_one_factor_scenario(
        &self,
        calibration: &CalibratedBergomiLsv<Bergomi1Factor>,
        factor: Bergomi1Factor,
        dividend_volatility_correlation: f64,
        executor: &DeterministicExecutor,
    ) -> Result<Scenario, MonteCarloError> {
        let bumped = calibrate_bergomi_lsv_parallel(
            calibration.target(),
            factor,
            self.path.risky_spot(),
            calibration.config().clone(),
            executor,
        )?;
        let path = self.path.clone().with_bergomi_lsv(
            factor,
            dividend_volatility_correlation,
            bumped.surface().clone(),
        )?;
        Ok(Scenario { path })
    }

    #[allow(clippy::too_many_arguments)]
    fn sample_lsv_bergomi_parameter_risk(
        &self,
        scenarios: &[Scenario; 4],
        bumps: [f64; 2],
        mut z: Vec<f64>,
        bridge: Option<&BrownianBridgePlan>,
        antithetic: bool,
        out: &mut [f64],
    ) -> Result<(), MonteCarloError> {
        if out.len() != WIDTH {
            return Err(invalid("lsv_bergomi_parameter_risk_width").into());
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
            out[0] += self.discounted_payoff_for_lsv_path(&self.path, &shocks)?;
            for parameter in 0..2 {
                let down =
                    self.discounted_payoff_for_lsv_path(&scenarios[2 * parameter].path, &shocks)?;
                let up = self
                    .discounted_payoff_for_lsv_path(&scenarios[2 * parameter + 1].path, &shocks)?;
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

    fn discounted_payoff_for_lsv_path(
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

fn validate_parameter_bump(
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
        return Err(invalid("lsv_bergomi_parameter_standard_error").into());
    }
    Ok(error)
}
