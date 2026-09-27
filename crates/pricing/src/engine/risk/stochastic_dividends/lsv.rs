//! Recalibration-aware Local-variance risk for residual-equity LSV.
//!
//! The pricing path first pulls physical-Spot payoff seeds through the Buehler
//! split to the squared-leverage surface. The existing finite-particle
//! calibration VJP then maps those seeds back to the requested residual-equity
//! Local-variance grid. Calibration randomness is fixed by its explicit seed.

use super::*;
use crate::VegaKtResult;
use crate::engine::processes::stochastic_dividends::reverse::ReverseContext;
use crate::mc::lsv::LsvError;
use crate::models::BergomiDynamics;
use crate::risk::{
    local_vega_density_from_node_adjoints, project_local_vega_nodes_to_reporting_iv,
    vega_kt_bucket_estimates, vega_kt_full_bucket_covariance, vega_kt_projection_from_parts,
    vega_kt_report,
};

const METHOD: &str = "buehler-residual-lsv-path-and-discrete-particle-vjp-v1";
const SPOT_METHOD: &str = "buehler-residual-lsv-scale-invariant-spot-reverse-v1";

#[derive(Clone, Debug, PartialEq)]
pub struct StochasticDividendLsvSpotRisk {
    pub price: StochasticDividendPrice,
    /// dPrice / d physical Spot with the residual-equity LSV surface
    /// re-anchored to the bumped funded residual equity.
    pub delta: f64,
    pub delta_standard_error: f64,
    pub method: &'static str,
}

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

struct LsvLocalVarianceSamples {
    price_stats: DeterministicStatistics,
    units: u64,
    paths: u128,
    node_adjoints: Vec<f64>,
    node_errors: Option<Vec<f64>>,
    /// Independent sampling units used for VegaKT covariance estimation:
    /// one aggregate pseudo-MC estimate or one row per RQMC scramble.
    price_samples: Vec<f64>,
    node_samples: Vec<Vec<f64>>,
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
    /// Physical-Spot Delta for residual-equity LSV. The particle calibration is
    /// homogeneous in its initial residual equity because both the calibration
    /// kernel and leverage lookup use log(F/F0). Recalibrating after a Spot bump
    /// therefore leaves the squared-leverage values unchanged while re-anchoring
    /// the surface initial_f. Normalized f/Y pricing states are consequently
    /// Spot-independent, and the exact finite-algorithm Delta is the Buehler
    /// reconstruction-coefficient reverse.
    pub fn evaluate_lsv_spot_risk(&self) -> Result<StochasticDividendLsvSpotRisk, MonteCarloError> {
        if self.lsv.is_none() || !self.path.is_lsv() {
            return Err(MonteCarloError::UnsupportedRiskForModel {
                model: "LSV Spot risk requires a stochastic-dividend residual LSV plan",
            });
        }
        let surface = self
            .path
            .lsv_surface()
            .ok_or(MonteCarloError::UnsupportedRiskForModel {
                model: "LSV Spot risk requires a stochastic-dividend residual LSV surface",
            })?;
        if surface.initial_f().to_bits() != self.path.risky_spot().to_bits() {
            return Err(invalid("lsv_spot_anchor").into());
        }
        if !self.risk_supported {
            return Err(MonteCarloError::UnsupportedRiskForModel {
                model: "stochastic-dividend LSV discontinuous payoff requires explicit smoothing",
            });
        }

        let context = ReverseContext::new(&self.path, &self.market, self.payment_time)?;
        let executor = DeterministicExecutor::new(self.policy)?;
        let dimension = self.path.random_dimension();
        let (price_stats, delta_stats, units, paths) = match self.engine {
            EngineConfig::PseudoMonteCarlo(config) => {
                let count = config.independent_sampling_units().get();
                let bridge = self.bridge(config.variance_reduction())?;
                let rng = Philox4x32::from_seed(config.master_seed());
                let stats = executor.try_map_reduce_statistics_vector(count, 2, |p, out| {
                    let z = (0..dimension)
                        .map(|d| {
                            rng.standard_normal(RandomCoordinate::new(
                                p,
                                d,
                                RandomDomain::Valuation,
                            ))
                        })
                        .collect();
                    self.sample_lsv_spot_risk(
                        &context,
                        z,
                        bridge.as_ref(),
                        config.variance_reduction().antithetic(),
                        out,
                    )
                })?;
                (stats[0], stats[1], count, config.evaluated_paths())
            }
            EngineConfig::RandomizedQuasiMonteCarlo(config) => {
                let qmc = RqmcPlan::compile(config, dimension)?;
                let bridge = self.bridge(config.variance_reduction())?;
                let count = config.points_per_scramble().get();
                let mut prices = Vec::with_capacity(config.scramble_count().get() as usize);
                let mut deltas = Vec::with_capacity(config.scramble_count().get() as usize);
                for scramble in 0..config.scramble_count().get() {
                    let stats = executor.try_map_reduce_statistics_vector(count, 2, |p, out| {
                        let z = (0..dimension)
                            .map(|d| {
                                let u = qmc
                                    .uniform(scramble, p, d)
                                    .map_err(|_| invalid("rqmc_uniform"))?;
                                inverse_standard_normal(u).map_err(|_| invalid("rqmc_normal"))
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        self.sample_lsv_spot_risk(
                            &context,
                            z,
                            bridge.as_ref(),
                            config.variance_reduction().antithetic(),
                            out,
                        )
                    })?;
                    prices.push(stats[0].sum().total() / count as f64);
                    deltas.push(stats[1].sum().total() / count as f64);
                }
                let scrambles = u64::from(config.scramble_count().get());
                (
                    DeterministicStatistics::from_ordered_values_two_pass(&prices),
                    DeterministicStatistics::from_ordered_values_two_pass(&deltas),
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

        let value = price_stats.sum().total() / units as f64;
        let standard_error = estimator_error(price_stats, units)?;
        let delta = delta_stats.sum().total() / units as f64;
        let delta_standard_error = estimator_error(delta_stats, units)?;
        if !value.is_finite()
            || !standard_error.is_finite()
            || !delta.is_finite()
            || !delta_standard_error.is_finite()
        {
            return Err(invalid("lsv_spot_risk_estimator").into());
        }
        Ok(StochasticDividendLsvSpotRisk {
            price: StochasticDividendPrice {
                value,
                standard_error,
                independent_sampling_units: units,
                evaluated_paths: paths,
                plan_fingerprint: self.fingerprint,
                scheme: self.scheme(),
            },
            delta,
            delta_standard_error,
            method: SPOT_METHOD,
        })
    }

    fn sample_lsv_spot_risk(
        &self,
        context: &ReverseContext,
        mut z: Vec<f64>,
        bridge: Option<&BrownianBridgePlan>,
        antithetic: bool,
        out: &mut [f64],
    ) -> Result<(), MonteCarloError> {
        if out.len() != 2 {
            return Err(invalid("lsv_spot_risk_width").into());
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
            out[1] += context.spot_pullback(&states, &seeds)?;
        }
        if antithetic {
            for value in out {
                *value *= 0.5;
            }
        }
        Ok(())
    }
}

impl StochasticDividendPricingPlan {
    /// Sensitivity to the residual-equity Local-variance target used by the
    /// particle LSV calibration. This is deliberately separate from
    /// `evaluate_aad`: it includes leverage recalibration but does not report
    /// Spot, curve, cash-mean or Bergomi-parameter risk.
    pub fn evaluate_local_variance_risk(
        &self,
    ) -> Result<StochasticDividendLocalVarianceRisk, MonteCarloError> {
        let samples = self.lsv_local_variance_samples()?;
        let value = samples.price_stats.sum().total() / samples.units as f64;
        let standard_error = estimator_error(samples.price_stats, samples.units)?;
        if !value.is_finite()
            || !standard_error.is_finite()
            || samples.node_adjoints.iter().any(|x| !x.is_finite())
            || samples
                .node_errors
                .as_ref()
                .is_some_and(|v| v.iter().any(|x| !x.is_finite()))
        {
            return Err(invalid("lsv_local_variance_risk_estimator").into());
        }

        let target = self
            .lsv
            .as_ref()
            .expect("LSV samples require an LSV calibration")
            .original_target();
        Ok(StochasticDividendLocalVarianceRisk {
            price: StochasticDividendPrice {
                value,
                standard_error,
                independent_sampling_units: samples.units,
                evaluated_paths: samples.paths,
                plan_fingerprint: self.fingerprint,
                scheme: self.scheme(),
            },
            time_nodes: target.time_nodes().into(),
            log_moneyness_nodes: target.log_moneyness_nodes().into(),
            node_adjoints: samples.node_adjoints.into_boxed_slice(),
            standard_errors: samples.node_errors.map(Vec::into_boxed_slice),
            method: METHOD,
        })
    }

    /// VegaKT of the residual-equity market-IV reporting surface. The reverse
    /// first includes finite-particle leverage recalibration, then converts
    /// Dupire-variance adjoints to Local-volatility adjoints and reuses the
    /// shared first-order reporting-IV projection.
    pub fn evaluate_vega_kt(&self) -> Result<VegaKtResult, MonteCarloError> {
        let vega_kt = self
            .base
            .local_volatility
            .as_ref()
            .and_then(|runtime| runtime.vega_kt.as_ref())
            .ok_or(MonteCarloError::MissingLocalVolatilityReportingBasis)?;
        let samples = self.lsv_local_variance_samples()?;
        let lsv = self
            .lsv
            .as_ref()
            .expect("LSV samples require an LSV calibration");
        let target = lsv.original_target();
        let x_count = target.log_moneyness_nodes().len();
        let bucket_count = vega_kt.basis.bucket_count();
        let mut raw_bucket_samples =
            Vec::with_capacity(samples.node_samples.len().saturating_mul(bucket_count));

        for node_sample in &samples.node_samples {
            let local_vol_adjoints = local_vol_node_adjoints(target, node_sample)?;
            let mut buckets = vec![0.0; bucket_count];
            for (time_index, maturity) in target.time_nodes().iter().copied().enumerate() {
                if maturity == 0.0 {
                    continue;
                }
                let row_start = time_index * x_count;
                let row_end = row_start + x_count;
                let density = local_vega_density_from_node_adjoints(
                    &local_vol_adjoints[row_start..row_end],
                    target.log_moneyness_nodes(),
                )?;
                let density_row = vega_kt
                    .density_rows
                    .iter()
                    .find(|row| row.maturity().get().to_bits() == maturity.to_bits())
                    .ok_or(MonteCarloError::MismatchedLocalVolatilityReportingBasis)?;
                let projection = project_local_vega_nodes_to_reporting_iv(
                    &vega_kt.basis,
                    maturity,
                    target.log_moneyness_nodes(),
                    &density,
                    density_row.active_domain(),
                )?;
                for (bucket, projected) in buckets.iter_mut().zip(projection.raw_buckets()) {
                    *bucket += *projected;
                }
            }
            raw_bucket_samples.extend(buckets);
        }

        let estimates =
            vega_kt_bucket_estimates(&samples.price_samples, &raw_bucket_samples, bucket_count)?;
        let raw_bucket_means = estimates
            .iter()
            .map(|estimate| estimate.raw_mean())
            .collect::<Vec<_>>();
        let local_vol_adjoints = local_vol_node_adjoints(target, &samples.node_adjoints)?;
        let mean_vega = local_vol_adjoints
            .iter()
            .copied()
            .collect::<pricing_numerics::NeumaierSum>()
            .total();
        let bucket_sum = raw_bucket_means
            .iter()
            .copied()
            .collect::<pricing_numerics::NeumaierSum>()
            .total();
        let projection = vega_kt_projection_from_parts(
            raw_bucket_means,
            mean_vega - bucket_sum,
            mean_vega,
            Default::default(),
        )?;
        let full_bucket_covariance = if vega_kt.full_bucket_covariance {
            Some(vega_kt_full_bucket_covariance(
                &raw_bucket_samples,
                bucket_count,
            )?)
        } else {
            None
        };
        let density_row = vega_kt
            .density_rows
            .last()
            .ok_or(MonteCarloError::MismatchedLocalVolatilityReportingBasis)?;
        Ok(VegaKtResult::try_from(&vega_kt_report(
            &vega_kt.basis,
            density_row,
            projection,
            estimates,
            full_bucket_covariance,
        )?)?)
    }

    fn lsv_local_variance_samples(&self) -> Result<LsvLocalVarianceSamples, MonteCarloError> {
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

        match self.engine {
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
                let node_adjoints = lsv.target_reverse(&leverage)?;
                Ok(LsvLocalVarianceSamples {
                    price_stats: stats[0],
                    units: count,
                    paths: config.evaluated_paths(),
                    price_samples: vec![stats[0].sum().total() / count as f64],
                    node_samples: vec![node_adjoints.clone()],
                    node_adjoints,
                    node_errors: None,
                })
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
                Ok(LsvLocalVarianceSamples {
                    price_stats: DeterministicStatistics::from_ordered_values_two_pass(&prices),
                    units: scrambles,
                    paths: u128::from(count)
                        * u128::from(scrambles)
                        * if config.variance_reduction().antithetic() {
                            2
                        } else {
                            1
                        },
                    price_samples: prices,
                    node_samples: risks,
                    node_adjoints: means,
                    node_errors: Some(errors),
                })
            }
        }
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

fn local_vol_node_adjoints(
    target: &LocalVarianceGrid,
    node_adjoints: &[f64],
) -> Result<Vec<f64>, MonteCarloError> {
    if node_adjoints.len() != target.values().len() {
        return Err(invalid("lsv_local_variance_node_count").into());
    }
    node_adjoints
        .iter()
        .zip(target.values())
        .map(|(adjoint, variance)| {
            let value = 2.0 * variance.sqrt() * adjoint;
            if value.is_finite() {
                Ok(value)
            } else {
                Err(invalid("lsv_local_volatility_adjoint").into())
            }
        })
        .collect()
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
