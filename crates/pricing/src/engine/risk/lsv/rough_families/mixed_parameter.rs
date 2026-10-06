//! Parameter risk at fixed Local Variance target, including particle recalibration.
use super::*;
use crate::engine::processes::rough_volatility::MixedBergomiVarianceRiskPlan;

#[derive(Clone, Debug, PartialEq)]
pub struct MixedBergomiLsvParameterRisk {
    pub price: LsvPrice,
    pub parameter_names: Box<[String]>,
    /// Total derivative at fixed target; not a stochastic-parameter refit.
    pub parameter_adjoints: Box<[f64]>,
    /// Direct valuation contribution holding squared leverage fixed.
    pub direct_adjoints: Box<[f64]>,
    /// Contribution from parameter-dependent particle recalibration.
    pub calibration_adjoints: Box<[f64]>,
    /// RQMC marginal SEs of TOTAL gradients, preserving direct/calibration
    /// covariance. None for pseudo MC. Conditional on one calibration sample.
    pub standard_errors: Option<Box<[f64]>>,
    pub method: &'static str,
    pub risk_fingerprint: Fingerprint,
}
struct ParameterReverse<'a> {
    plan: &'a RoughFamilyLsvPricingPlan,
    valuation: MixedBergomiVarianceRiskPlan,
    calibration: MixedBergomiVarianceRiskPlan,
}
impl ParameterReverse<'_> {
    fn sample(
        &self,
        z: &[f64],
        path: u64,
        anti: bool,
        out: &mut [f64],
    ) -> Result<(), MonteCarloError> {
        let core = &self.plan.core;
        let d = self.valuation.width();
        let signs = if anti { &[1.0, -1.0][..] } else { &[1.0][..] };
        let weight = 1.0 / signs.len() as f64;
        for &sign in signs {
            let shocks = z.iter().map(|a| a * sign).collect::<Vec<_>>();
            let path_record = core
                .path_plan
                .evolve_path(core.calibration.surface().initial_f(), &shocks)?;
            let (value, seeds) =
                core.base
                    .lsv_payoff(path_record.states(), PathIndex::new(path), true)?;
            let (leverage, variance) =
                path_record.reverse_variances(&seeds.expect("reverse requested"))?;
            let direct = self.valuation.reverse(&shocks, &variance)?;
            out[0] += weight * value;
            for (o, a) in out[1..1 + d].iter_mut().zip(direct) {
                *o += weight * a;
            }
            for (o, &a) in out[1 + d..].iter_mut().zip(&leverage.squared_leverage) {
                *o += weight * a;
            }
        }
        Ok(())
    }
    fn project(
        &self,
        stats: &[DeterministicStatistics],
        units: u64,
    ) -> Result<Vec<f64>, MonteCarloError> {
        let d = self.valuation.width();
        let mean = |s: &DeterministicStatistics| s.sum().total() / units as f64;
        let direct = stats[1..1 + d].iter().map(mean).collect::<Vec<_>>();
        let leverage = stats[1 + d..].iter().map(mean).collect::<Vec<_>>();
        let calibration = self
            .plan
            .core
            .calibration
            .mixed_bergomi_parameter_pullback(&leverage, &self.calibration)?;
        // Project BEFORE scramble variance reduction. The two contributions are
        // correlated through the valuation sample, despite fixed calibration.
        let mut result = direct
            .iter()
            .zip(&calibration)
            .map(|(a, b)| a + b)
            .collect::<Vec<_>>();
        result.extend(direct);
        result.extend(calibration);
        if result.iter().any(|a| !a.is_finite()) {
            return Err(LsvError::InvalidInput {
                field: "mixed_bergomi_lsv_risk_nonfinite",
                index: 0,
            }
            .into());
        }
        Ok(result)
    }
}
impl RoughFamilyLsvPricingPlan {
    /// Component eta and shared rho risk with Leverage recalibrated to the same
    /// relative Local Variance target. Hurst, weights and xi are fixed.
    /// Requires a retained reverse trace and |rho|<1. No parameter bumps.
    pub fn evaluate_mixed_bergomi_parameter_risk(
        &self,
    ) -> Result<MixedBergomiLsvParameterRisk, MonteCarloError> {
        let core = &self.core;
        core.calibration.validate_calibration_reverse()?;
        if !core.risk_supported {
            return Err(MonteCarloError::UnsupportedRiskForModel {
                model: "Mixed Bergomi LSV parameter risk requires pathwise payoff or explicit smoothing",
            });
        }
        let reverse = ParameterReverse {
            plan: self,
            valuation: MixedBergomiVarianceRiskPlan::compile(
                core.path_plan.model().clone(),
                core.path_plan.times().to_vec(),
            )?,
            calibration: MixedBergomiVarianceRiskPlan::compile(
                core.calibration.model().clone(),
                core.calibration.surface().times().to_vec(),
            )?,
        };
        let d = reverse.valuation.width();
        let width = 1 + d + core.calibration.surface().squared_leverage().len();
        let executor = DeterministicExecutor::new(core.policy)?;
        let (price_stats, units, paths, values, errors) = match core.engine {
            EngineConfig::PseudoMonteCarlo(c) => {
                let n = c.independent_sampling_units().get();
                let bridge = core.bridge(c.variance_reduction())?;
                let stats = executor.try_map_reduce_statistics_vector(n, width, |i, out| {
                    let z =
                        core.path_plan
                            .pseudo_shocks(c.master_seed(), i, RandomDomain::Valuation);
                    let z = core.apply_bridge(z, bridge.as_ref())?;
                    reverse.sample(&z, i, c.variance_reduction().antithetic(), out)
                })?;
                let risk = reverse.project(&stats, n)?;
                (stats[0], n, c.evaluated_paths(), risk, None)
            }
            EngineConfig::RandomizedQuasiMonteCarlo(c) => {
                let dim = core.path_plan.random_dimension();
                let qmc = RqmcPlan::compile(c, dim)?;
                let bridge = core.bridge(c.variance_reduction())?;
                let n = c.points_per_scramble().get();
                let count = c.scramble_count().get();
                let mut prices = Vec::with_capacity(count as usize);
                let mut risks = Vec::with_capacity(count as usize);
                for scramble in 0..count {
                    let stats = executor.try_map_reduce_statistics_vector(n, width, |i, out| {
                        let z = (0..dim)
                            .map(|j| {
                                let error = || LsvError::InvalidInput {
                                    field: "mixed_bergomi_lsv_rqmc_coordinate",
                                    index: j as usize,
                                };
                                let u = qmc.uniform(scramble, i, j).map_err(|_| error())?;
                                inverse_standard_normal(u).map_err(|_| error())
                            })
                            .collect::<Result<Vec<_>, LsvError>>()?;
                        let z = core.apply_bridge(z, bridge.as_ref())?;
                        reverse.sample(
                            &z,
                            u64::from(scramble) * n + i,
                            c.variance_reduction().antithetic(),
                            out,
                        )
                    })?;
                    prices.push(stats[0].sum().total() / n as f64);
                    risks.push(reverse.project(&stats, n)?);
                }
                let mut means = Vec::with_capacity(3 * d);
                let mut errors = Vec::with_capacity(d);
                for j in 0..3 * d {
                    let v = risks.iter().map(|r| r[j]).collect::<Vec<_>>();
                    let stat = DeterministicStatistics::from_ordered_values_two_pass(&v);
                    means.push(stat.sum().total() / f64::from(count));
                    if j < d {
                        errors.push(standard_error(stat, u64::from(count))?);
                    }
                }
                (
                    DeterministicStatistics::from_ordered_values_two_pass(&prices),
                    u64::from(count),
                    n as u128
                        * u128::from(count)
                        * if c.variance_reduction().antithetic() {
                            2
                        } else {
                            1
                        },
                    means,
                    Some(errors.into_boxed_slice()),
                )
            }
        };
        let names = reverse
            .valuation
            .path_plan()
            .mixed_bergomi_parameter_names()?;
        let method = "mixed-bergomi-lsv-fixed-target-parameter-particle-vjp-v1";
        let mut hash = blake3::Hasher::new();
        hash.update(method.as_bytes());
        hash.update(core.fingerprint.as_bytes());
        Ok(MixedBergomiLsvParameterRisk {
            price: LsvPrice {
                value: price_stats.sum().total() / units as f64,
                standard_error: standard_error(price_stats, units)?,
                independent_sampling_units: units,
                evaluated_paths: paths,
                calibration_seed: core.calibration.config().seed(),
                plan_fingerprint: core.fingerprint,
                scheme: CalibratedRoughFamilyLsv::SCHEME,
            },
            parameter_names: names.into_boxed_slice(),
            parameter_adjoints: values[..d].into(),
            direct_adjoints: values[d..2 * d].into(),
            calibration_adjoints: values[2 * d..].into(),
            standard_errors: errors,
            method,
            risk_fingerprint: Fingerprint::from_bytes(*hash.finalize().as_bytes()),
        })
    }
}
