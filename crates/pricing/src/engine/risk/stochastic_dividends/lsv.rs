//! Recalibration-aware Local-variance risk for residual-equity LSV.
//!
//! The pricing path first pulls physical-Spot payoff seeds through the Buehler
//! split to the squared-leverage surface. The existing finite-particle
//! calibration VJP then maps those seeds back to the requested residual-equity
//! Local-variance grid. Calibration randomness is fixed by its explicit seed.

use super::*;
use crate::mc::lsv::LsvError;
use crate::models::BergomiDynamics;

const METHOD: &str = "buehler-residual-lsv-path-and-discrete-particle-vjp-v1";

#[derive(Clone, Debug, PartialEq)]
pub struct StochasticDividendLocalVarianceRisk {
    pub price: StochasticDividendPrice,
    pub time_nodes: Box<[f64]>,
    pub log_moneyness_nodes: Box<[f64]>,
    /// dPrice / d residual-equity Dupire variance node, including particle
    /// recalibration on the finite calibration sample.
    pub node_adjoints: Box<[f64]>,
    /// RQMC scramble-level sampling errors. Calibration sampling uncertainty is
    /// not included because the calibration seed and particle cloud are fixed.
    pub standard_errors: Option<Box<[f64]>>,
    pub method: &'static str,
}

impl StochasticDividendLsvCalibration {
    fn validate_reverse(&self) -> Result<(), MonteCarloError> {
        let retained = match self {
            Self::One { calibration, .. } => calibration.config().retain_reverse_trace(),
            Self::Two { calibration, .. } => calibration.config().retain_reverse_trace(),
        };
        if retained {
            Ok(())
        } else {
            Err(LsvError::ReverseTraceNotRetained.into())
        }
    }

    fn original_target(&self) -> &LocalVarianceGrid {
        match self {
            Self::One {
                original_target, ..
            }
            | Self::Two {
                original_target, ..
            } => original_target,
        }
    }

    fn target_reverse(&self, leverage: &[f64]) -> Result<Vec<f64>, MonteCarloError> {
        match self {
            Self::One {
                calibration,
                original_target,
            } => target_reverse(calibration, original_target, leverage),
            Self::Two {
                calibration,
                original_target,
            } => target_reverse(calibration, original_target, leverage),
        }
    }
}

fn target_reverse<F: BergomiDynamics>(
    calibration: &CalibratedBergomiLsv<F>,
    original_target: &LocalVarianceGrid,
    leverage: &[f64],
) -> Result<Vec<f64>, MonteCarloError> {
    let refined = calibration.reverse_leverage(leverage)?;
    let m = original_target.log_moneyness_nodes().len();
    let mut original = vec![0.0; original_target.values().len()];
    for (r, &t) in calibration.target().time_nodes().iter().enumerate() {
        for (j, &x) in original_target.log_moneyness_nodes().iter().enumerate() {
            original_target.interpolate(t, x)?.transpose_accumulate(
                refined[r * m + j],
                &mut original,
                m,
            );
        }
    }
    Ok(original)
}

impl StochasticDividendPricingPlan {
    /// Sensitivity to the residual-equity Local-variance target used by the
    /// particle LSV calibration. This is deliberately separate from
    /// `evaluate_aad`: it includes leverage recalibration but does not report
    /// Spot, curve, cash-mean or Bergomi-parameter risk.
    pub fn evaluate_local_variance_risk(
        &self,
    ) -> Result<StochasticDividendLocalVarianceRisk, MonteCarloError> {
        let lsv = self
            .lsv
            .as_ref()
            .ok_or(MonteCarloError::UnsupportedRiskForModel {
                model: "local-variance risk requires a stochastic-dividend residual LSV plan",
            })?;
        if !self.risk_supported {
            return Err(MonteCarloError::UnsupportedRiskForModel {
                model: "stochastic-dividend LSV discontinuous payoff requires explicit smoothing",
            });
        }
        // Fail before generating valuation paths if the caller compiled a
        // price-only calibration without its reverse trace.
        lsv.validate_reverse()?;

        let leverage_count = self
            .path
            .lsv_surface()
            .ok_or(MonteCarloError::UnsupportedRiskForModel {
                model: "local-variance risk requires a stochastic-dividend residual LSV path",
            })?
            .squared_leverage()
            .len();
        let width = 1 + leverage_count;
        let executor = DeterministicExecutor::new(self.policy)?;
        let dimension = self.path.random_dimension();

        let (price_stats, units, paths, node_adjoints, node_errors) = match self.engine {
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
                    self.sample_lsv_local_variance_risk(
                        z,
                        bridge.as_ref(),
                        config.variance_reduction().antithetic(),
                        out,
                    )
                })?;
                let leverage = stats[1..]
                    .iter()
                    .map(|s| s.sum().total() / count as f64)
                    .collect::<Vec<_>>();
                (
                    stats[0],
                    count,
                    config.evaluated_paths(),
                    lsv.target_reverse(&leverage)?,
                    None,
                )
            }
            EngineConfig::RandomizedQuasiMonteCarlo(config) => {
                let qmc = RqmcPlan::compile(config, dimension)?;
                let bridge = self.bridge(config.variance_reduction())?;
                let count = config.points_per_scramble().get();
                let mut prices = Vec::with_capacity(config.scramble_count().get() as usize);
                let mut risks = Vec::with_capacity(config.scramble_count().get() as usize);
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
                            self.sample_lsv_local_variance_risk(
                                z,
                                bridge.as_ref(),
                                config.variance_reduction().antithetic(),
                                out,
                            )
                        })?;
                    prices.push(stats[0].sum().total() / count as f64);
                    let leverage = stats[1..]
                        .iter()
                        .map(|s| s.sum().total() / count as f64)
                        .collect::<Vec<_>>();
                    risks.push(lsv.target_reverse(&leverage)?);
                }
                let scrambles = u64::from(config.scramble_count().get());
                let mut means = Vec::with_capacity(lsv.original_target().values().len());
                let mut errors = Vec::with_capacity(lsv.original_target().values().len());
                for j in 0..lsv.original_target().values().len() {
                    let values = risks.iter().map(|r| r[j]).collect::<Vec<_>>();
                    let stat = DeterministicStatistics::from_ordered_values_two_pass(&values);
                    means.push(stat.sum().total() / scrambles as f64);
                    errors.push(estimator_error(stat, scrambles)?);
                }
                (
                    DeterministicStatistics::from_ordered_values_two_pass(&prices),
                    scrambles,
                    u128::from(count)
                        * u128::from(scrambles)
                        * if config.variance_reduction().antithetic() {
                            2
                        } else {
                            1
                        },
                    means,
                    Some(errors),
                )
            }
        };

        let value = price_stats.sum().total() / units as f64;
        let standard_error = estimator_error(price_stats, units)?;
        if !value.is_finite()
            || !standard_error.is_finite()
            || node_adjoints.iter().any(|x| !x.is_finite())
            || node_errors
                .as_ref()
                .is_some_and(|v| v.iter().any(|x| !x.is_finite()))
        {
            return Err(invalid("lsv_local_variance_risk_estimator").into());
        }

        let target = lsv.original_target();
        Ok(StochasticDividendLocalVarianceRisk {
            price: StochasticDividendPrice {
                value,
                standard_error,
                independent_sampling_units: units,
                evaluated_paths: paths,
                plan_fingerprint: self.fingerprint,
                scheme: self.scheme(),
            },
            time_nodes: target.time_nodes().into(),
            log_moneyness_nodes: target.log_moneyness_nodes().into(),
            node_adjoints: node_adjoints.into_boxed_slice(),
            standard_errors: node_errors.map(Vec::into_boxed_slice),
            method: METHOD,
        })
    }

    fn sample_lsv_local_variance_risk(
        &self,
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
            let leverage = self.path.lsv_leverage_pullback(&shocks, &states, &seeds)?;
            if out.len() != leverage.len() + 1 {
                return Err(invalid("lsv_local_variance_risk_width").into());
            }
            out[0] += payoff;
            for (target, source) in out[1..].iter_mut().zip(leverage) {
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
        return Err(invalid("lsv_local_variance_standard_error").into());
    }
    Ok(error)
}
