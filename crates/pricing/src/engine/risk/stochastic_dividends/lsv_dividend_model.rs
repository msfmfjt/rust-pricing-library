//! Buehler dividend-model parameter risk for stochastic-dividend residual LSV.
//!
//! These parameters do not enter the marginal residual-equity LSV calibration.
//! The calibrated leverage surface is therefore reused exactly while the joint
//! stochastic-dividend pricing path is rebuilt for each common-noise bump.

use super::*;

const METHOD: &str = "buehler-residual-lsv-common-noise-fixed-calibration-dividend-model-v1";
const WIDTH: usize = 5; // Price plus four parameter derivatives.

#[derive(Clone, Debug, PartialEq)]
pub struct StochasticDividendLsvDividendModelRisk {
    pub price: StochasticDividendPrice,
    pub parameter_labels: Box<[String]>,
    pub derivatives: Box<[f64]>,
    pub standard_errors: Box<[f64]>,
    pub parameter_bumps: Box<[f64]>,
    pub method: &'static str,
}

impl StochasticDividendLsvDividendModelRisk {
    #[must_use]
    pub fn dividend_mean_reversion_derivative(&self) -> f64 {
        self.derivatives[0]
    }
    #[must_use]
    pub fn equity_linkage_derivative(&self) -> f64 {
        self.derivatives[1]
    }
    #[must_use]
    pub fn dividend_volatility_derivative(&self) -> f64 {
        self.derivatives[2]
    }
    #[must_use]
    pub fn equity_dividend_correlation_derivative(&self) -> f64 {
        self.derivatives[3]
    }
}

struct Scenario {
    path: StochasticDividendPathPlan,
}

impl StochasticDividendPricingPlan {
    /// Common-random central finite differences for Buehler dividend-model
    /// parameters. The residual-LSV calibration target, leverage values and all
    /// Bergomi parameters/correlations are held fixed.
    pub fn evaluate_lsv_dividend_model_risk(
        &self,
        dividend_mean_reversion_bump: f64,
        equity_linkage_bump: f64,
        dividend_volatility_bump: f64,
        equity_dividend_correlation_bump: f64,
    ) -> Result<StochasticDividendLsvDividendModelRisk, MonteCarloError> {
        if !self.risk_supported {
            return Err(MonteCarloError::UnsupportedRiskForModel {
                model: "stochastic-dividend LSV discontinuous payoff requires explicit smoothing",
            });
        }
        let lsv = self
            .lsv
            .as_ref()
            .ok_or(MonteCarloError::UnsupportedRiskForModel {
                model: "dividend-model LSV risk requires a stochastic-dividend residual LSV plan",
            })?;
        if !self.path.is_lsv() {
            return Err(MonteCarloError::UnsupportedRiskForModel {
                model: "dividend-model LSV risk requires a residual LSV pricing path",
            });
        }

        let model = self.path.model();
        validate_nonnegative_bump(
            model.mean_reversion(),
            dividend_mean_reversion_bump,
            "lsv_dividend_mean_reversion_bump",
        )?;
        validate_unit_interval_bump(
            model.equity_linkage(),
            equity_linkage_bump,
            "lsv_equity_linkage_bump",
        )?;
        validate_nonnegative_bump(
            model.dividend_volatility(),
            dividend_volatility_bump,
            "lsv_dividend_volatility_bump",
        )?;
        validate_correlation_bump(
            model.equity_dividend_correlation(),
            equity_dividend_correlation_bump,
            "lsv_equity_dividend_correlation_bump",
        )?;

        let bumped = |mean_reversion: f64,
                      equity_linkage: f64,
                      dividend_volatility: f64,
                      equity_dividend_correlation: f64| {
            BuehlerDividendModel::new(
                mean_reversion,
                equity_linkage,
                dividend_volatility,
                equity_dividend_correlation,
            )
            .map_err(|_| invalid("lsv_bumped_dividend_model"))
        };

        let scenarios = [
            self.fixed_calibration_dividend_model_scenario(
                lsv,
                bumped(
                    model.mean_reversion() - dividend_mean_reversion_bump,
                    model.equity_linkage(),
                    model.dividend_volatility(),
                    model.equity_dividend_correlation(),
                )?,
            )?,
            self.fixed_calibration_dividend_model_scenario(
                lsv,
                bumped(
                    model.mean_reversion() + dividend_mean_reversion_bump,
                    model.equity_linkage(),
                    model.dividend_volatility(),
                    model.equity_dividend_correlation(),
                )?,
            )?,
            self.fixed_calibration_dividend_model_scenario(
                lsv,
                bumped(
                    model.mean_reversion(),
                    model.equity_linkage() - equity_linkage_bump,
                    model.dividend_volatility(),
                    model.equity_dividend_correlation(),
                )?,
            )?,
            self.fixed_calibration_dividend_model_scenario(
                lsv,
                bumped(
                    model.mean_reversion(),
                    model.equity_linkage() + equity_linkage_bump,
                    model.dividend_volatility(),
                    model.equity_dividend_correlation(),
                )?,
            )?,
            self.fixed_calibration_dividend_model_scenario(
                lsv,
                bumped(
                    model.mean_reversion(),
                    model.equity_linkage(),
                    model.dividend_volatility() - dividend_volatility_bump,
                    model.equity_dividend_correlation(),
                )?,
            )?,
            self.fixed_calibration_dividend_model_scenario(
                lsv,
                bumped(
                    model.mean_reversion(),
                    model.equity_linkage(),
                    model.dividend_volatility() + dividend_volatility_bump,
                    model.equity_dividend_correlation(),
                )?,
            )?,
            self.fixed_calibration_dividend_model_scenario(
                lsv,
                bumped(
                    model.mean_reversion(),
                    model.equity_linkage(),
                    model.dividend_volatility(),
                    model.equity_dividend_correlation() - equity_dividend_correlation_bump,
                )?,
            )?,
            self.fixed_calibration_dividend_model_scenario(
                lsv,
                bumped(
                    model.mean_reversion(),
                    model.equity_linkage(),
                    model.dividend_volatility(),
                    model.equity_dividend_correlation() + equity_dividend_correlation_bump,
                )?,
            )?,
        ];
        let bumps = [
            dividend_mean_reversion_bump,
            equity_linkage_bump,
            dividend_volatility_bump,
            equity_dividend_correlation_bump,
        ];

        let executor = DeterministicExecutor::new(self.policy)?;
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
                    self.sample_lsv_dividend_model_risk(
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
                            self.sample_lsv_dividend_model_risk(
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
            .map(|statistics| statistics.sum().total() / units as f64)
            .collect::<Vec<_>>();
        let errors = statistics
            .iter()
            .map(|statistics| estimator_error(*statistics, units))
            .collect::<Result<Vec<_>, _>>()?;
        if values.iter().chain(&errors).any(|value| !value.is_finite()) {
            return Err(invalid("lsv_dividend_model_risk_estimator").into());
        }

        Ok(StochasticDividendLsvDividendModelRisk {
            price: StochasticDividendPrice {
                value: values[0],
                standard_error: errors[0],
                independent_sampling_units: units,
                evaluated_paths: paths,
                plan_fingerprint: self.fingerprint,
                scheme: self.scheme(),
            },
            parameter_labels: [
                "dividend_mean_reversion",
                "equity_linkage",
                "dividend_volatility",
                "equity_dividend_correlation",
            ]
            .map(str::to_owned)
            .into(),
            derivatives: values[1..].into(),
            standard_errors: errors[1..].into(),
            parameter_bumps: bumps.into(),
            method: METHOD,
        })
    }

    fn fixed_calibration_dividend_model_scenario(
        &self,
        lsv: &StochasticDividendLsvCalibration,
        model: BuehlerDividendModel,
    ) -> Result<Scenario, MonteCarloError> {
        let grid = LocalVolTimeGrid::compile(
            self.path.times().to_vec(),
            self.path.times()[self.path.times().len() - 1],
        )?;
        let base = StochasticDividendPathPlan::compile(&self.market, model, 0.0, &grid)?;
        let path = match lsv {
            StochasticDividendLsvCalibration::One {
                calibration,
                dividend_volatility_correlation,
                ..
            } => base.with_bergomi_lsv(
                calibration.factor(),
                *dividend_volatility_correlation,
                calibration.surface().clone(),
            )?,
            StochasticDividendLsvCalibration::Two {
                calibration,
                dividend_volatility_correlations,
                ..
            } => base.with_bergomi_two_factor_lsv(
                calibration.factor(),
                *dividend_volatility_correlations,
                calibration.surface().clone(),
            )?,
            StochasticDividendLsvCalibration::Rough {
                calibration,
                dividend_volatility_correlation,
                ..
            } => base.with_rough_bergomi_lsv(
                calibration.model(),
                *dividend_volatility_correlation,
                calibration.surface().clone(),
            )?,
        };
        if path.random_dimension() != self.path.random_dimension() {
            return Err(invalid("lsv_dividend_model_random_dimension").into());
        }
        Ok(Scenario { path })
    }

    #[allow(clippy::too_many_arguments)]
    fn sample_lsv_dividend_model_risk(
        &self,
        scenarios: &[Scenario; 8],
        bumps: [f64; 4],
        mut z: Vec<f64>,
        bridge: Option<&BrownianBridgePlan>,
        antithetic: bool,
        out: &mut [f64],
    ) -> Result<(), MonteCarloError> {
        if out.len() != WIDTH {
            return Err(invalid("lsv_dividend_model_risk_width").into());
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
            out[0] += self.discounted_payoff_for_dividend_model_path(&self.path, &shocks)?;
            for parameter in 0..4 {
                let down = self.discounted_payoff_for_dividend_model_path(
                    &scenarios[2 * parameter].path,
                    &shocks,
                )?;
                let up = self.discounted_payoff_for_dividend_model_path(
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

    fn discounted_payoff_for_dividend_model_path(
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
        return Err(invalid("lsv_dividend_model_standard_error").into());
    }
    Ok(error)
}
