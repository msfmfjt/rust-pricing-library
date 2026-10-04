//! Common-random-number physical Spot second differences, not second-order AAD.
use crate::mc::{
    BrownianBridgePlan, DeterministicExecutor, DeterministicStatistics, EngineConfig,
    ExecutionPolicy, RqmcPlan, inverse_standard_normal,
};
use crate::models::rough_volatility::invalid;
use crate::{Fingerprint, MonteCarloError};

/// Finite-bump Gamma and a paired half-bump diagnostic. All Greeks are per
/// physical Spot currency unit squared. SEs exclude bump and time-grid bias.
#[derive(Clone, Debug, PartialEq)]
pub struct RoughGammaBump<P> {
    pub price: P,
    /// Central second difference with the explicitly supplied absolute bump.
    pub gamma: f64,
    pub gamma_standard_error: f64,
    pub half_bump_gamma: f64,
    pub half_bump_standard_error: f64,
    /// Signed Gamma(h/2) - Gamma(h), not a bound on the remaining bias.
    pub bump_difference: f64,
    /// Paired SE retaining covariance between both second differences.
    pub bump_difference_standard_error: f64,
    pub spot_bump: f64,
    pub payoff_evaluations: u128,
    pub convention: &'static str,
    pub method: &'static str,
    pub risk_fingerprint: Fingerprint,
}

#[derive(Clone, Copy)]
pub(super) struct BumpGrid {
    pub spot: f64,
    pub residual: f64,
    pub h: f64,
}
impl BumpGrid {
    pub fn new(spot: f64, residual: f64, h: f64) -> Result<Self, MonteCarloError> {
        // This excludes numerically meaningless widths; it is not an accuracy
        // prescription. Both physical Spot and residual equity must stay positive.
        if !spot.is_finite()
            || spot <= 0.0
            || !residual.is_finite()
            || residual <= 0.0
            || !h.is_finite()
            || h <= 0.0
            || h / spot < 1e-5
            || h >= spot
            || h >= residual
            || !(spot + h).is_finite()
            || !(h * h).is_finite()
            || !(4.0 / (h * h)).is_finite()
            || !(spot / residual).is_finite()
        {
            return Err(invalid("rough_gamma_spot_bump_domain").into());
        }
        Ok(Self { spot, residual, h })
    }
    pub fn shifts(self) -> [f64; 5] {
        [0.0, self.h, -self.h, 0.5 * self.h, -0.5 * self.h]
    }
    pub fn channels(self, prices: [f64; 5]) -> Result<[f64; 4], MonteCarloError> {
        let central =
            |up: f64, dn: f64, width: f64| ((up - prices[0]) + (dn - prices[0])) / (width * width);
        let full = central(prices[1], prices[2], self.h);
        let half = central(prices[3], prices[4], 0.5 * self.h);
        let out = [prices[0], full, half, half - full];
        if prices.iter().chain(out.iter()).any(|v| !v.is_finite()) {
            return Err(invalid("rough_gamma_nonfinite_sample").into());
        }
        Ok(out)
    }
}

pub(super) struct GammaSampling<'a> {
    pub engine: EngineConfig,
    pub policy: ExecutionPolicy,
    pub times: &'a [f64],
    pub brownian_blocks: usize,
    pub dimension: u32,
}
pub(super) struct GammaRun {
    pub stats: Vec<DeterministicStatistics>,
    pub units: u64,
    pub paths: u128,
}
impl GammaRun {
    pub fn moments(&self, channel: usize) -> Result<(f64, f64), MonteCarloError> {
        let s = self.stats[channel];
        let variance = s
            .moments()
            .sample_variance()
            .ok_or(MonteCarloError::InsufficientSamplingUnits { count: self.units })?;
        let mean = s.sum().total() / self.units as f64;
        let se = (variance / self.units as f64).sqrt();
        if !mean.is_finite() || !se.is_finite() {
            return Err(invalid("rough_gamma_estimator").into());
        }
        Ok((mean, se))
    }
    pub fn finish<P>(
        &self,
        price: P,
        bump: BumpGrid,
        fingerprint: Fingerprint,
        convention: &'static str,
    ) -> Result<RoughGammaBump<P>, MonteCarloError> {
        let (gamma, gamma_standard_error) = self.moments(1)?;
        let (half_bump_gamma, half_bump_standard_error) = self.moments(2)?;
        let (bump_difference, bump_difference_standard_error) = self.moments(3)?;
        let method = "rough-physical-spot-central-crn-gamma-v1";
        let mut hash = blake3::Hasher::new();
        hash.update(method.as_bytes());
        hash.update(&[0]);
        hash.update(fingerprint.as_bytes());
        hash.update(convention.as_bytes());
        hash.update(&bump.h.to_le_bytes());
        Ok(RoughGammaBump {
            price,
            gamma,
            gamma_standard_error,
            half_bump_gamma,
            half_bump_standard_error,
            bump_difference,
            bump_difference_standard_error,
            spot_bump: bump.h,
            payoff_evaluations: 5 * self.paths,
            convention,
            method,
            risk_fingerprint: Fingerprint::from_bytes(*hash.finalize().as_bytes()),
        })
    }
}
impl GammaSampling<'_> {
    /// `values` prices five physical scenarios using the SAME transformed shocks.
    /// Antithetic averaging happens before variance estimation; RQMC variance is
    /// taken across scramble means, never across individual Sobol points.
    pub fn evaluate<S, V>(
        &self,
        bump: BumpGrid,
        shocks: S,
        values: V,
    ) -> Result<GammaRun, MonteCarloError>
    where
        S: Fn(u64, u64) -> Vec<f64> + Sync,
        V: Fn(&[f64], u64) -> Result<[f64; 5], MonteCarloError> + Sync,
    {
        let executor = DeterministicExecutor::new(self.policy)?;
        let n = self.times.len() - 1;
        let make_bridge = |enabled| -> Result<_, MonteCarloError> {
            if enabled {
                Ok(Some(
                    BrownianBridgePlan::compile(self.times.to_vec(), 1)
                        .map_err(|e| MonteCarloError::LocalVol(e.into()))?,
                ))
            } else {
                Ok(None)
            }
        };
        let sample = |mut z: Vec<f64>,
                      path,
                      anti,
                      bridge: Option<&BrownianBridgePlan>,
                      out: &mut [f64]|
         -> Result<(), MonteCarloError> {
            if let Some(b) = bridge {
                for block in z[..self.brownian_blocks * n].chunks_exact_mut(n) {
                    let mapped = b
                        .apply_one_factor(block)
                        .map_err(|e| MonteCarloError::LocalVol(e.into()))?;
                    block.copy_from_slice(&mapped);
                }
            }
            let signs = if anti { &[1.0, -1.0][..] } else { &[1.0][..] };
            for &sign in signs {
                let zz = z.iter().map(|z| z * sign).collect::<Vec<_>>();
                let channels = bump.channels(values(&zz, path)?)?;
                for j in 0..4 {
                    out[j] += channels[j] / signs.len() as f64;
                }
            }
            Ok(())
        };
        match self.engine {
            EngineConfig::PseudoMonteCarlo(c) => {
                let units = c.independent_sampling_units().get();
                let vr = c.variance_reduction();
                let bridge = make_bridge(vr.brownian_bridge())?;
                let stats = executor.try_map_reduce_statistics_vector(units, 4, |i, out| {
                    sample(
                        shocks(c.master_seed(), i),
                        i,
                        vr.antithetic(),
                        bridge.as_ref(),
                        out,
                    )
                })?;
                Ok(GammaRun {
                    stats,
                    units,
                    paths: c.evaluated_paths(),
                })
            }
            EngineConfig::RandomizedQuasiMonteCarlo(c) => {
                let qmc = RqmcPlan::compile(c, self.dimension)?;
                let vr = c.variance_reduction();
                let bridge = make_bridge(vr.brownian_bridge())?;
                let count = c.points_per_scramble().get();
                let units = u64::from(c.scramble_count().get());
                let mut means: [Vec<f64>; 4] =
                    std::array::from_fn(|_| Vec::with_capacity(units as usize));
                for scramble in 0..c.scramble_count().get() {
                    let stats = executor.try_map_reduce_statistics_vector(count, 4, |i, out| {
                        let z = (0..self.dimension)
                            .map(|d| {
                                let u = qmc
                                    .uniform(scramble, i, d)
                                    .map_err(|_| invalid("rough_gamma_rqmc"))?;
                                inverse_standard_normal(u)
                                    .map_err(|_| invalid("rough_gamma_normal"))
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        sample(
                            z,
                            u64::from(scramble) * count + i,
                            vr.antithetic(),
                            bridge.as_ref(),
                            out,
                        )
                    })?;
                    for j in 0..4 {
                        means[j].push(stats[j].sum().total() / count as f64);
                    }
                }
                let stats = means
                    .iter()
                    .map(|m| DeterministicStatistics::from_ordered_values_two_pass(m))
                    .collect();
                Ok(GammaRun {
                    stats,
                    units,
                    paths: u128::from(count)
                        * u128::from(units)
                        * if vr.antithetic() { 2 } else { 1 },
                })
            }
        }
    }
}
