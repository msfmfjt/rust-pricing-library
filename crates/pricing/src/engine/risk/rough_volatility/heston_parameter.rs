//! Pure-SV Heston scalar parameter risk through the full discrete variance history.
use super::delta::estimate;
use super::*;

#[derive(Clone, Debug, PartialEq)]
pub struct HestonMcParameterRisk {
    pub price: RoughVolatilityPrice,
    /// Natural units: v0, kappa, theta, nu, rho. Not IV Vega or refitted risk.
    pub parameter_adjoints: [f64; 5],
    /// Marginal sampling errors; these do not define the covariance of a basket.
    pub standard_errors: [f64; 5],
    pub method: &'static str,
    pub risk_fingerprint: Fingerprint,
}

impl RoughVolatilityPricingPlan {
    /// Reverse payoff, log-Euler asset, and the entire rough/lift variance
    /// recurrence. Fix kernel/Hurst, spot, curves, dividends and Gaussian draws.
    /// This is not an LSV-parameter recalibration risk or Fourier derivative.
    pub fn evaluate_heston_parameter_risk(&self) -> Result<HestonMcParameterRisk, MonteCarloError> {
        self.path.heston_parameter_domain()?;
        if !self.risk_supported {
            return Err(MonteCarloError::UnsupportedRiskForModel {
                model: "Heston MC parameter risk requires pathwise payoff or explicit smoothing",
            });
        }
        let executor = DeterministicExecutor::new(self.policy)?;
        let (stats, units, paths) = match self.engine {
            EngineConfig::PseudoMonteCarlo(c) => {
                let n = c.independent_sampling_units().get();
                let bridge = self.bridge(c.variance_reduction())?;
                let stats = executor.try_map_reduce_statistics_vector(n, 6, |i, out| {
                    let z = self
                        .path
                        .pseudo_shocks(c.master_seed(), i, RandomDomain::Valuation);
                    self.heston_parameter_sample(
                        z,
                        bridge.as_ref(),
                        c.variance_reduction().antithetic(),
                        out,
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
                let mut means: [Vec<f64>; 6] =
                    std::array::from_fn(|_| Vec::with_capacity(count as usize));
                for scramble in 0..count {
                    let stats = executor.try_map_reduce_statistics_vector(n, 6, |i, out| {
                        let z = (0..dimension)
                            .map(|d| {
                                let u = qmc
                                    .uniform(scramble, i, d)
                                    .map_err(|_| invalid("rough_rqmc_uniform"))?;
                                inverse_standard_normal(u).map_err(|_| invalid("rough_rqmc_normal"))
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        self.heston_parameter_sample(
                            z,
                            bridge.as_ref(),
                            c.variance_reduction().antithetic(),
                            out,
                        )
                    })?;
                    for j in 0..6 {
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
        let mut parameter_adjoints = [0.0; 5];
        let mut standard_errors = [0.0; 5];
        for j in 0..5 {
            (parameter_adjoints[j], standard_errors[j]) = estimate(stats[j + 1], units)?;
        }
        let method = "heston-mc-fixed-kernel-parameter-vjp-v1";
        let mut hash = blake3::Hasher::new();
        hash.update(method.as_bytes());
        hash.update(self.fingerprint.as_bytes());
        Ok(HestonMcParameterRisk {
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
            parameter_adjoints,
            standard_errors,
            method,
            risk_fingerprint: Fingerprint::from_bytes(*hash.finalize().as_bytes()),
        })
    }

    fn heston_parameter_sample(
        &self,
        mut normals: Vec<f64>,
        bridge: Option<&BrownianBridgePlan>,
        anti: bool,
        out: &mut [f64],
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
            let record = self
                .path
                .evolve_heston_parameter_path(self.initial_forward, &z)?;
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
            for j in 0..5 {
                out[j + 1] += risk.parameters[j] / signs.len() as f64;
            }
        }
        Ok(())
    }
}
