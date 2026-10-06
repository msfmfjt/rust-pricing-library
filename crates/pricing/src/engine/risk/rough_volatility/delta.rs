//! Fixed-model physical Spot Delta via payoff, path and escrow reverse.
use super::*;

#[derive(Clone, Debug, PartialEq)]
pub struct RoughVolatilityDelta {
    pub price: RoughVolatilityPrice,
    pub delta: f64,
    pub delta_standard_error: f64,
    pub method: &'static str,
}
fn estimate(s: DeterministicStatistics, n: u64) -> Result<(f64, f64), MonteCarloError> {
    let v = s
        .moments()
        .sample_variance()
        .ok_or(MonteCarloError::InsufficientSamplingUnits { count: n })?;
    let mean = s.sum().total() / n as f64;
    let se = (v / n as f64).sqrt();
    if !mean.is_finite() || !se.is_finite() {
        return Err(invalid("rough_delta_estimator").into());
    }
    Ok((mean, se))
}
impl RoughVolatilityPricingPlan {
    /// Physical Spot Delta, keeping all rough-model parameters, rate/repo curves,
    /// and cash/proportional dividend amounts fixed. No bumps or recalibration.
    /// Discontinuous payoffs need an explicit supported smoothing contract.
    pub fn evaluate_delta(&self) -> Result<RoughVolatilityDelta, MonteCarloError> {
        if !self.risk_supported {
            return Err(MonteCarloError::UnsupportedRiskForModel {
                model: "rough MC Delta requires a pathwise payoff or explicit smoothing",
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
                    self.delta_sample(z, bridge.as_ref(), c.variance_reduction().antithetic(), out)
                })?;
                (stats, n, c.evaluated_paths())
            }
            EngineConfig::RandomizedQuasiMonteCarlo(c) => {
                let dimension = self.path.random_dimension();
                let qmc = RqmcPlan::compile(c, dimension)?;
                let bridge = self.bridge(c.variance_reduction())?;
                let n = c.points_per_scramble().get();
                let count = c.scramble_count().get();
                let mut means = [
                    Vec::with_capacity(count as usize),
                    Vec::with_capacity(count as usize),
                ];
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
                        self.delta_sample(
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
        let (delta, delta_standard_error) = estimate(stats[1], units)?;
        Ok(RoughVolatilityDelta {
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
            delta,
            delta_standard_error,
            method: "rough-fixed-model-spot-delta-discrete-vjp-v1",
        })
    }
    fn delta_sample(
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
            let record = self.path.evolve_recorded_path(self.initial_forward, &z)?;
            let spots = self
                .observations
                .iter()
                .zip(&record.path().forwards)
                .map(|(o, &f)| {
                    let post = o.scale * f + o.reserve;
                    let pre = o.event.map(|e| (post + e.cash) / (1.0 - e.beta));
                    if !post.is_finite() || pre.is_some_and(|s| !s.is_finite()) {
                        return Err(invalid("rough_observation_overflow"));
                    }
                    Ok((post, pre))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let (price, seeds) = self
                .base
                .hybrid_spot_payoff_adjoints(self.time_nodes(), &spots)?;
            let mut market_seeds = self.dividends.zero_adjoints();
            let mut forward_seeds = Vec::with_capacity(n + 1);
            for (j, (&(post, pre), &f)) in seeds.iter().zip(&record.path().forwards).enumerate() {
                forward_seeds.push(self.dividends.nodes()[j].reverse_spots(
                    f,
                    0.0,
                    post,
                    pre,
                    &mut market_seeds[j],
                )?);
            }
            let delta = record.reverse_initial_forward(&forward_seeds)?
                + self.dividends.reverse_market(&market_seeds)?.spot;
            if !delta.is_finite() {
                return Err(invalid("rough_spot_delta").into());
            }
            out[0] += price / signs.len() as f64;
            out[1] += delta / signs.len() as f64;
        }
        Ok(())
    }
}
