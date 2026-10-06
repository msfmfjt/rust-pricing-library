//! Pure-SV Mixed rough Bergomi component eta and shared rho risk.
use super::delta::estimate;
use super::*;
use crate::engine::processes::rough_volatility::MixedBergomiMcHurstPlan;

#[derive(Clone, Debug, PartialEq)]
pub struct MixedBergomiMcParameterRisk {
    pub price: RoughVolatilityPrice,
    /// Component eta values, shared rho, and optionally Hurst; weights and xi fixed.
    pub parameter_names: Box<[String]>,
    pub parameter_adjoints: Box<[f64]>,
    /// Marginal sampling errors; these do not define the covariance of a basket.
    pub standard_errors: Box<[f64]>,
    pub method: &'static str,
    pub risk_fingerprint: Fingerprint,
}

impl RoughVolatilityPricingPlan {
    /// Reverse payoff, log-Euler asset, mixture exponentials and Gaussian coupling.
    /// Fix kernel/Hurst, mixture weights, xi, spot, curves, dividends and draws.
    /// This is not an LSV-parameter recalibration risk or Fourier derivative.
    pub fn evaluate_mixed_bergomi_parameter_risk(
        &self,
    ) -> Result<MixedBergomiMcParameterRisk, MonteCarloError> {
        self.mixed_bergomi_parameter_risk_impl(false)
    }

    /// Extend the existing eta/rho risk by Hurst at fixed weights and xi.
    /// At H=1/2 return its left derivative; no price bumps.
    pub fn evaluate_mixed_bergomi_parameter_risk_with_hurst(
        &self,
    ) -> Result<MixedBergomiMcParameterRisk, MonteCarloError> {
        self.mixed_bergomi_parameter_risk_impl(true)
    }

    fn mixed_bergomi_parameter_risk_impl(
        &self,
        include_hurst: bool,
    ) -> Result<MixedBergomiMcParameterRisk, MonteCarloError> {
        let mut names = self.path.mixed_bergomi_parameter_names()?;
        let hurst = if include_hurst {
            names.push("hurst".into());
            Some(MixedBergomiMcHurstPlan::compile(&self.path)?)
        } else {
            None
        };
        let d = names.len();
        if !self.risk_supported {
            return Err(MonteCarloError::UnsupportedRiskForModel {
                model: "Mixed Bergomi MC parameter risk requires pathwise payoff or explicit smoothing",
            });
        }
        let executor = DeterministicExecutor::new(self.policy)?;
        let (stats, units, paths) = match self.engine {
            EngineConfig::PseudoMonteCarlo(c) => {
                let n = c.independent_sampling_units().get();
                let bridge = self.bridge(c.variance_reduction())?;
                let stats = executor.try_map_reduce_statistics_vector(n, d + 1, |i, out| {
                    let z = self
                        .path
                        .pseudo_shocks(c.master_seed(), i, RandomDomain::Valuation);
                    self.mixed_bergomi_parameter_sample(
                        z,
                        bridge.as_ref(),
                        c.variance_reduction().antithetic(),
                        out,
                        hurst.as_ref(),
                    )
                })?;
                (stats, n, c.evaluated_paths())
            }
            EngineConfig::RandomizedQuasiMonteCarlo(c) => {
                let dimension = self.path.random_dimension();
                let qmc = RqmcPlan::compile(c, dimension)?;
                let bridge = self.bridge(c.variance_reduction())?;
                let n = c.points_per_scramble().get();
                let count = c.scramble_count().get();
                let mut means = vec![Vec::with_capacity(count as usize); d + 1];
                for scramble in 0..count {
                    let stats = executor.try_map_reduce_statistics_vector(n, d + 1, |i, out| {
                        let z = (0..dimension)
                            .map(|d| {
                                let u = qmc
                                    .uniform(scramble, i, d)
                                    .map_err(|_| invalid("rough_rqmc_uniform"))?;
                                inverse_standard_normal(u).map_err(|_| invalid("rough_rqmc_normal"))
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        self.mixed_bergomi_parameter_sample(
                            z,
                            bridge.as_ref(),
                            c.variance_reduction().antithetic(),
                            out,
                            hurst.as_ref(),
                        )
                    })?;
                    for j in 0..d + 1 {
                        means[j].push(stats[j].sum().total() / n as f64);
                    }
                }
                let stats = means
                    .iter()
                    .map(|m| DeterministicStatistics::from_ordered_values_two_pass(m))
                    .collect();
                (
                    stats,
                    u64::from(count),
                    n as u128
                        * count as u128
                        * if c.variance_reduction().antithetic() {
                            2
                        } else {
                            1
                        },
                )
            }
        };
        let (value, standard_error) = estimate(stats[0], units)?;
        let mut parameter_adjoints = vec![0.0; d];
        let mut standard_errors = vec![0.0; d];
        for j in 0..d {
            (parameter_adjoints[j], standard_errors[j]) = estimate(stats[j + 1], units)?;
        }
        let method = if include_hurst {
            "mixed-bergomi-mc-parameter-hurst-vjp-v1"
        } else {
            "mixed-bergomi-mc-fixed-kernel-parameter-vjp-v1"
        };
        let mut hash = blake3::Hasher::new();
        hash.update(method.as_bytes());
        hash.update(self.fingerprint.as_bytes());
        Ok(MixedBergomiMcParameterRisk {
            price: RoughVolatilityPrice {
                value,
                standard_error,
                independent_sampling_units: units,
                evaluated_paths: paths,
                plan_fingerprint: self.fingerprint,
                scheme: self.path.scheme(),
                calibration_method: None,
                calibration_seed: None,
                cash_dividend_model: Some(HULL_WHITE_CASH_DIVIDEND_MODEL),
            },
            parameter_names: names.into_boxed_slice(),
            parameter_adjoints: parameter_adjoints.into_boxed_slice(),
            standard_errors: standard_errors.into_boxed_slice(),
            method,
            risk_fingerprint: Fingerprint::from_bytes(*hash.finalize().as_bytes()),
        })
    }

    fn mixed_bergomi_parameter_sample(
        &self,
        mut normals: Vec<f64>,
        bridge: Option<&BrownianBridgePlan>,
        anti: bool,
        out: &mut [f64],
        hurst: Option<&MixedBergomiMcHurstPlan>,
    ) -> Result<(), MonteCarloError> {
        let n = self.time_nodes().len() - 1;
        if let Some(bridge) = bridge {
            for block in normals[..self.path.brownian_block_count() * n].chunks_exact_mut(n) {
                let mapped = bridge
                    .apply_one_factor(block)
                    .map_err(|e| MonteCarloError::LocalVol(e.into()))?;
                block.copy_from_slice(&mapped);
            }
        }
        let signs = if anti { &[1.0, -1.0][..] } else { &[1.0][..] };
        for &sign in signs {
            let z = normals.iter().map(|v| sign * v).collect::<Vec<_>>();
            let record = match hurst {
                Some(plan) => plan.evolve_path(self.initial_forward, &z)?,
                None => self
                    .path
                    .evolve_mixed_bergomi_parameter_path(self.initial_forward, &z)?,
            };
            let spots = self
                .observations
                .iter()
                .zip(&record.path().forwards)
                .map(|(o, &f)| {
                    let post = o.scale * f + o.reserve;
                    let pre = o.event.map(|e| (post + e.cash) / (1.0 - e.beta));
                    if !post.is_finite() || pre.is_some_and(|p| !p.is_finite()) {
                        return Err(invalid("rough_observation_overflow"));
                    }
                    Ok((post, pre))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let (price, seeds) = self
                .base
                .hybrid_spot_payoff_adjoints(self.time_nodes(), &spots)?;
            // Curves, cash and Spot are fixed. Only the affine observation slope
            // transmits these parameter adjoints; no market/escrow reverse term.
            let forward_seeds = seeds
                .iter()
                .zip(&self.observations)
                .map(|(&(post, pre), o)| {
                    o.scale * (post + o.event.map_or(0.0, |e| pre / (1.0 - e.beta)))
                })
                .collect::<Vec<_>>();
            let risk = record.reverse(&forward_seeds, &vec![0.0; n + 1])?;
            out[0] += price / signs.len() as f64;
            for j in 0..risk.parameters.len() {
                out[j + 1] += risk.parameters[j] / signs.len() as f64;
            }
        }
        Ok(())
    }
}
