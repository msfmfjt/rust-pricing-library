//! Correlation risk for 1F Bergomi stochastic-dividend residual LSV.
//!
//! Equity/Bergomi correlation belongs to the marginal residual-equity LSV
//! calibration and therefore requires full particle recalibration. The
//! dividend/Bergomi correlation affects only the joint pricing Brownian system;
//! it leaves the marginal (F_res, A) calibration unchanged.

use super::*;
use crate::models::Bergomi1Factor;

const METHOD: &str = "buehler-residual-lsv-common-noise-bergomi-1f-correlation-v1";
const WIDTH: usize = 3; // Price, d/d rho_fV, d/d rho_DV.

#[derive(Clone, Debug, PartialEq)]
pub struct StochasticDividendLsvCorrelationRisk {
    pub price: StochasticDividendPrice,
    pub parameter_labels: Box<[String]>,
    pub derivatives: Box<[f64]>,
    pub standard_errors: Box<[f64]>,
    pub correlation_bumps: Box<[f64]>,
    pub method: &'static str,
}

impl StochasticDividendLsvCorrelationRisk {
    #[must_use]
    pub fn equity_volatility_correlation_derivative(&self) -> f64 {
        self.derivatives[0]
    }

    #[must_use]
    pub fn dividend_volatility_correlation_derivative(&self) -> f64 {
        self.derivatives[1]
    }
}

struct Scenario {
    path: StochasticDividendPathPlan,
}

impl StochasticDividendPricingPlan {
    /// Central finite-difference correlation risk for 1F Bergomi residual LSV.
    ///
    /// The equity/volatility correlation bump changes the Bergomi spot/vol
    /// correlation and fully reruns the finite-particle calibration. The
    /// dividend/volatility correlation bump changes only the joint pricing
    /// kernel because it is absent from the marginal residual-equity calibration.
    pub fn evaluate_lsv_correlation_risk(
        &self,
        equity_volatility_correlation_bump: f64,
        dividend_volatility_correlation_bump: f64,
    ) -> Result<StochasticDividendLsvCorrelationRisk, MonteCarloError> {
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
                    model: "1F Bergomi LSV correlation risk does not apply to a 2F plan",
                });
            }
            Some(StochasticDividendLsvCalibration::Rough { .. }) => {
                return Err(MonteCarloError::UnsupportedRiskForModel {
                    model: "1F Bergomi LSV correlation risk does not apply to a rough plan",
                });
            }
            None => {
                return Err(MonteCarloError::UnsupportedRiskForModel {
                    model: "Bergomi LSV correlation risk requires a stochastic-dividend residual LSV plan",
                });
            }
        };

        let factor = calibration.factor();
        validate_correlation_bump(
            factor.correlation(),
            equity_volatility_correlation_bump,
            "lsv_bergomi_equity_volatility_correlation_bump",
        )?;
        validate_correlation_bump(
            dividend_volatility_correlation,
            dividend_volatility_correlation_bump,
            "lsv_bergomi_dividend_volatility_correlation_bump",
        )?;

        let executor = DeterministicExecutor::new(self.policy)?;
        let equity_down = Bergomi1Factor::new(
            factor.mean_reversion(),
            factor.vol_of_vol(),
            factor.correlation() - equity_volatility_correlation_bump,
        )
        .map_err(|_| invalid("lsv_bergomi_equity_volatility_correlation_down"))?;
        let equity_up = Bergomi1Factor::new(
            factor.mean_reversion(),
            factor.vol_of_vol(),
            factor.correlation() + equity_volatility_correlation_bump,
        )
        .map_err(|_| invalid("lsv_bergomi_equity_volatility_correlation_up"))?;

        let scenarios = [
            self.recalibrated_correlation_scenario(
                calibration,
                equity_down,
                dividend_volatility_correlation,
                &executor,
            )?,
            self.recalibrated_correlation_scenario(
                calibration,
                equity_up,
                dividend_volatility_correlation,
                &executor,
            )?,
            self.fixed_calibration_correlation_scenario(
                calibration,
                factor,
                dividend_volatility_correlation - dividend_volatility_correlation_bump,
            )?,
            self.fixed_calibration_correlation_scenario(
                calibration,
                factor,
                dividend_volatility_correlation + dividend_volatility_correlation_bump,
            )?,
        ];

        let dimension = self.path.random_dimension();
        let bumps = [
            equity_volatility_correlation_bump,
            dividend_volatility_correlation_bump,
        ];
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
                    self.sample_lsv_correlation_risk(
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
                            self.sample_lsv_correlation_risk(
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
            return Err(invalid("lsv_correlation_risk_estimator").into());
        }

        Ok(StochasticDividendLsvCorrelationRisk {
            price: StochasticDividendPrice {
                value: values[0],
                standard_error: errors[0],
                independent_sampling_units: units,
                evaluated_paths: paths,
                plan_fingerprint: self.fingerprint,
                scheme: self.scheme(),
            },
            parameter_labels: [
                "equity_volatility_correlation",
                "dividend_volatility_correlation",
            ]
            .map(str::to_owned)
            .into(),
            derivatives: values[1..].into(),
            standard_errors: errors[1..].into(),
            correlation_bumps: bumps.into(),
            method: METHOD,
        })
    }

    fn recalibrated_correlation_scenario(
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

    fn fixed_calibration_correlation_scenario(
        &self,
        calibration: &CalibratedBergomiLsv<Bergomi1Factor>,
        factor: Bergomi1Factor,
        dividend_volatility_correlation: f64,
    ) -> Result<Scenario, MonteCarloError> {
        let path = self.path.clone().with_bergomi_lsv(
            factor,
            dividend_volatility_correlation,
            calibration.surface().clone(),
        )?;
        Ok(Scenario { path })
    }

    #[allow(clippy::too_many_arguments)]
    fn sample_lsv_correlation_risk(
        &self,
        scenarios: &[Scenario; 4],
        bumps: [f64; 2],
        mut z: Vec<f64>,
        bridge: Option<&BrownianBridgePlan>,
        antithetic: bool,
        out: &mut [f64],
    ) -> Result<(), MonteCarloError> {
        if out.len() != WIDTH {
            return Err(invalid("lsv_correlation_risk_width").into());
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
            out[0] += self.discounted_payoff_for_correlation_path(&self.path, &shocks)?;
            for correlation in 0..2 {
                let down = self.discounted_payoff_for_correlation_path(
                    &scenarios[2 * correlation].path,
                    &shocks,
                )?;
                let up = self.discounted_payoff_for_correlation_path(
                    &scenarios[2 * correlation + 1].path,
                    &shocks,
                )?;
                out[1 + correlation] += (up - down) / (2.0 * bumps[correlation]);
            }
        }
        if antithetic {
            for value in out {
                *value *= 0.5;
            }
        }
        Ok(())
    }

    fn discounted_payoff_for_correlation_path(
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
        return Err(invalid("lsv_correlation_standard_error").into());
    }
    Ok(error)
}
