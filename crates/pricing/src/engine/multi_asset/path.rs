use super::lsv_kernels::LsvPath;
use super::*;
use crate::core::DayCountConvention;
use crate::mc::{
    LocalVolPath, Philox4x32, RandomCoordinate, RandomDomain, inverse_standard_normal,
};
use crate::multi_asset::MultiAssetError as E;
use crate::product::PayoffEvaluation;
#[cfg(test)]
mod sampling_contracts;

pub(super) struct AssetPath {
    pub spots: Vec<f64>,
    pub pre_spots: Vec<f64>,
    pub spot_derivatives: Vec<f64>,
    pub pre_spot_derivatives: Vec<f64>,
    pub bs_vega: Vec<f64>,
    pub local: Option<LocalVolPath>,
    pub lsv: Option<LsvPath>,
    pub hw: Option<super::hull_white::HwAssetPath>,
    pub local_correlation: Option<std::sync::Arc<super::local_correlation::LocalCorrelationPath>>,
}
impl MultiAssetPricingPlan {
    pub(super) fn shocks(&self, scramble: Option<u32>, point: u64) -> Result<Vec<Vec<f64>>, E> {
        let n = self.random_factor_count();
        let steps = self.times.len() - 1;
        // Put every factor's terminal bridge normal in the first n Sobol
        // dimensions (requirements section 5.1). Factor-major blocks can have poor
        // low-dimensional projections even for a terminal basket payoff.
        let bridge_rank_major = self.qmc.is_some() && self.bridge.is_some();
        let mut independent = vec![vec![0.0; steps]; n];
        for (factor, values) in independent.iter_mut().enumerate() {
            for (step, value) in values.iter_mut().enumerate() {
                let dimension = u32::try_from(if bridge_rank_major {
                    step * n + factor
                } else {
                    factor * steps + step
                })
                .map_err(E::numerical)?;
                *value = match self.engine {
                    EngineConfig::PseudoMonteCarlo(c) => {
                        Philox4x32::from_seed(c.master_seed()).standard_normal(
                            RandomCoordinate::new(point, dimension, RandomDomain::Valuation),
                        )
                    }
                    EngineConfig::RandomizedQuasiMonteCarlo(_) => {
                        let q = self.qmc.as_ref().expect("compiled QMC");
                        let u = q
                            .uniform(scramble.expect("QMC scramble"), point, dimension)
                            .map_err(E::numerical)?;
                        inverse_standard_normal(u).map_err(E::numerical)?
                    }
                };
            }
            if let Some(bridge) = &self.bridge {
                *values = bridge.apply_one_factor(values).map_err(E::numerical)?;
            }
        }
        if self.local_correlation.is_some() {
            return Ok(independent);
        }
        let mut correlated = vec![vec![0.0; steps]; n];
        for step in 0..steps {
            if let Some(d) = &self.lsv_drivers
                && let Some(permutations) = &d.permutations
            {
                let order = &permutations[step];
                let l = d.intervals[step].lower();
                for (i, &row) in order.iter().enumerate() {
                    let mut value = 0.0;
                    for k in 0..=i {
                        value += l[i * n + k] * independent[order[k]][step];
                    }
                    correlated[row][step] = value * d.scales[step][row];
                }
                continue;
            }
            let l = if let Some(d) = &self.lsv_drivers {
                d.intervals[step].lower()
            } else {
                self.correlation.entries()[self.interval_correlations[step]]
                    .1
                    .lower()
            };
            for (i, row) in correlated.iter_mut().enumerate() {
                let mut value = 0.0;
                for k in 0..=i {
                    value += l[i * n + k] * independent[k][step];
                }
                row[step] = value * self.lsv_drivers.as_ref().map_or(1.0, |d| d.scales[step][i]);
            }
        }
        Ok(correlated)
    }
    pub(super) fn paths(&self, shocks: &[Vec<f64>]) -> Result<Vec<AssetPath>, E> {
        if self.local_correlation.is_some() {
            return self.local_correlation_paths(shocks);
        }
        if self.hull_white.is_some() {
            return self.hw_paths(shocks);
        }
        self.assets
            .iter()
            .zip(shocks)
            .enumerate()
            .map(|(asset, (a, z))| {
                let spot = a.forward.spot().get();
                let mut vega = vec![0.0; self.times.len()];
                let (f, local, lsv) = if let Some(lsv) = &a.lsv {
                    let drivers = self.lsv_drivers.as_ref().expect("LSV drivers");
                    let index = self.assets.len()
                        + drivers
                            .asset_indices
                            .iter()
                            .position(|i| *i == asset)
                            .expect("LSV asset");
                    let path = lsv
                        .process
                        .evolve(z, &shocks[index..index + lsv.calibration.factor_count()])
                        .map_err(E::numerical)?;
                    let f = path
                        .states()
                        .iter()
                        .zip(a.process.forward_normalizers())
                        .map(|(m, forward)| m * forward)
                        .collect();
                    (f, None, Some(path))
                } else {
                    let (f, local) = match &a.model {
                        ModelSpec::BlackScholes(model) => {
                            let vol = model.volatility().get();
                            let mut f = vec![spot];
                            let mut score = 0.0;
                            let forwards = a.process.forward_normalizers();
                            for (step, &z) in z.iter().enumerate() {
                                let dt = self.times[step + 1] - self.times[step];
                                let next = f[step]
                                    * (forwards[step + 1] / forwards[step])
                                    * (-0.5 * vol * vol * dt + vol * dt.sqrt() * z).exp();
                                if !next.is_finite() || next <= 0.0 {
                                    return Err(E::Invalid(
                                        "nonpositive or nonfinite continuous equity state",
                                    ));
                                }
                                score += dt.sqrt() * z - vol * dt;
                                f.push(next);
                                vega[step + 1] = next * score;
                            }
                            (f, None)
                        }
                        ModelSpec::LocalVolatility(model) => {
                            let path = a
                                .process
                                .evolve_path(model.local_variance_grid(), spot, z)
                                .map_err(E::numerical)?;
                            (path.states().to_vec(), Some(path))
                        }
                        _ => unreachable!("validated model"),
                    };
                    (f, local, None)
                };
                let spots: Vec<_> = f
                    .iter()
                    .zip(&a.coordinates)
                    .map(|(&f, c)| c.reconstruct_spot(a.forward.spot(), f))
                    .collect();
                let pre_spots: Vec<_> = f
                    .iter()
                    .zip(&a.pre_coordinates)
                    .map(|(&f, c)| c.reconstruct_spot(a.forward.spot(), f))
                    .collect();
                if spots
                    .iter()
                    .chain(&pre_spots)
                    .any(|s| !s.is_finite() || *s <= 0.0)
                {
                    return Err(E::Invalid(
                        "nonpositive or nonfinite physical spot; no path flooring or resampling",
                    ));
                }
                // Under fixed log-forward grid coordinates, f and F scale together with Spot.
                // Paid cash is held fixed: d(A*S0)/dS0=0, and dS/dS0=B*f/S0.
                let spot_derivatives = f
                    .iter()
                    .zip(&a.coordinates)
                    .map(|(f, c)| c.b() * f / spot)
                    .collect();
                let pre_spot_derivatives = f
                    .iter()
                    .zip(&a.pre_coordinates)
                    .map(|(f, c)| c.b() * f / spot)
                    .collect();
                Ok(AssetPath {
                    spots,
                    pre_spots,
                    spot_derivatives,
                    pre_spot_derivatives,
                    bs_vega: vega,
                    local,
                    lsv,
                    hw: None,
                    local_correlation: None,
                })
            })
            .collect()
    }
    pub(super) fn indices(&self, u: UnderlyingId, date: Date) -> Option<(usize, usize)> {
        let asset = self
            .assets
            .iter()
            .position(|a| a.forward.underlying() == u)?;
        let t = DayCountConvention::Act365F.year_fraction(self.valuation_date, date);
        let node = self.times.binary_search_by(|x| x.total_cmp(&t)).ok()?;
        Some((asset, node))
    }
    pub(super) fn observe(
        &self,
        paths: &[AssetPath],
        u: UnderlyingId,
        date: Date,
        pre: bool,
        bump: Option<(usize, f64)>,
    ) -> Option<f64> {
        let (asset, node) = self.indices(u, date)?;
        let p = &paths[asset];
        let (spot, slope) = if pre {
            (p.pre_spots[node], p.pre_spot_derivatives[node])
        } else {
            (p.spots[node], p.spot_derivatives[node])
        };
        Some(
            spot + bump
                .filter(|(i, _)| *i == asset)
                .map_or(0.0, |(_, h)| h * slope),
        )
    }
    pub(super) fn payoff_adjoints(
        &self,
        paths: &[AssetPath],
        bump: Option<(usize, f64)>,
    ) -> Result<PayoffEvaluation, E> {
        if let Some((asset, h)) = bump {
            let p = &paths[asset];
            for (values, slopes) in [
                (&p.spots, &p.spot_derivatives),
                (&p.pre_spots, &p.pre_spot_derivatives),
            ] {
                if values.iter().zip(slopes).any(|(v, s)| {
                    let b = v + h * s;
                    !b.is_finite() || b <= 0.0
                }) {
                    return Err(E::Invalid(
                        "Spot bump produces a nonpositive or nonfinite physical spot",
                    ));
                }
            }
        }
        if self.hull_white.is_some() {
            return self.hw_payoff_adjoints(paths, bump);
        }
        Ok(self.payoff.evaluate_single_with_observation_adjoints(
            |u, d| self.observe(paths, u, d, false, bump),
            |u, d| self.observe(paths, u, d, true, bump),
        )?)
    }
    pub(super) fn delta(&self, paths: &[AssetPath], payoff: &PayoffEvaluation) -> Vec<f64> {
        let mut delta = vec![0.0; paths.len()];
        for (pre, u, d, value) in payoff
            .terminal_adjoints
            .iter()
            .map(|s| (false, s.underlying, s.observation_date, s.value))
            .chain(
                payoff
                    .pre_dividend_adjoints
                    .iter()
                    .map(|s| (true, s.underlying, s.observation_date, s.value)),
            )
        {
            let (i, j) = self.indices(u, d).expect("validated observation");
            delta[i] += value
                * if pre {
                    paths[i].pre_spot_derivatives[j]
                } else {
                    paths[i].spot_derivatives[j]
                };
        }
        delta
    }
}
