//! Pure-SV Rough Heston Hurst risk through kernel loadings and the variance history.
use super::delta::estimate;
use super::*;
use crate::engine::processes::rough_volatility::RoughHestonMcHurstPlan;

#[derive(Clone, Debug, PartialEq)]
pub struct RoughHestonMcHurstRisk {
    pub price: RoughVolatilityPrice,
    /// Per one absolute H unit, not IV Vega or recalibrated LSV risk.
    pub hurst_sensitivity: f64,
    /// Sampling error only, excluding discretization and truncation bias.
    pub standard_error: f64,
    pub method: &'static str,
    pub risk_fingerprint: Fingerprint,
}

impl RoughVolatilityPricingPlan {
    /// Reverse payoff, the full variance recurrence, and every H-dependent
    /// kernel loading. Fix five scalar parameters, Spot, curves, dividends and
    /// Gaussian draws. H=1/2 returns the left derivative. Pure Rough Heston only;
    /// this is not a Fourier derivative or a recalibrated LSV sensitivity.
    pub fn evaluate_hurst_risk(&self) -> Result<RoughHestonMcHurstRisk, MonteCarloError> {
        let hurst_plan = RoughHestonMcHurstPlan::compile(&self.path)?;
        if !self.risk_supported {
            return Err(MonteCarloError::UnsupportedRiskForModel {
                model: "Rough Heston MC Hurst risk requires pathwise payoff or explicit smoothing",
            });
        }
        let executor = DeterministicExecutor::new(self.policy)?;
        let (stats, units, paths) = match self.engine {
            EngineConfig::PseudoMonteCarlo(c) => {
                let n = c.independent_sampling_units().get();
                let bridge = self.bridge(c.variance_reduction())?;
                let stats = executor.try_map_reduce_statistics_vector(n, 2, |i, out| {
                    let z = self
                        .path
                        .pseudo_shocks(c.master_seed(), i, RandomDomain::Valuation);
                    self.hurst_sample(
                        &hurst_plan,
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
                let mut means: [Vec<f64>; 2] =
                    std::array::from_fn(|_| Vec::with_capacity(count as usize));
                for scramble in 0..count {
                    let stats = executor.try_map_reduce_statistics_vector(n, 2, |i, out| {
                        let z = (0..dimension)
                            .map(|d| {
                                let u = qmc
                                    .uniform(scramble, i, d)
                                    .map_err(|_| invalid("rough_rqmc_uniform"))?;
                                inverse_standard_normal(u).map_err(|_| invalid("rough_rqmc_normal"))
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        self.hurst_sample(
                            &hurst_plan,
                            z,
                            bridge.as_ref(),
                            c.variance_reduction().antithetic(),
                            out,
                        )
                    })?;
                    for j in 0..2 {
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
        let (hurst_sensitivity, hurst_error) = estimate(stats[1], units)?;
        let method = "rough-heston-mc-hurst-kernel-vjp-v1";
        let mut hash = blake3::Hasher::new();
        hash.update(method.as_bytes());
        hash.update(self.fingerprint.as_bytes());
        Ok(RoughHestonMcHurstRisk {
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
            hurst_sensitivity,
            standard_error: hurst_error,
            method,
            risk_fingerprint: Fingerprint::from_bytes(*hash.finalize().as_bytes()),
        })
    }

    fn hurst_sample(
        &self,
        hurst_plan: &RoughHestonMcHurstPlan,
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
            let record = hurst_plan.evolve_path(self.initial_forward, &z)?;
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
            out[1] += risk.hurst / signs.len() as f64;
        }
        Ok(())
    }
}
