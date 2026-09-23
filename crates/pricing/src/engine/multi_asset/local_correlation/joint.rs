//! Exact joint Gaussian endpoint transitions. Mixing the entire transition keeps
//! every spot/own-volatility/rate marginal block fixed, including OU integrals.
use super::super::lsv::LsvDrivers;
use super::*;
use crate::mc::hull_white::HullWhiteLsvTarget;
use crate::mc::lsv::LsvLeverageSurface;
use crate::models::HullWhite1Factor;
use pricing_numerics::NeumaierSum;

#[derive(Clone, Debug)]
pub(super) struct Joint {
    pub assets: Vec<Asset>,
    pub configs: Vec<Option<MultiAssetBergomiLsvConfig>>,
    pub drivers: [LsvDrivers; 2],
    pub width: usize,
    pub offsets: Vec<usize>,
    pub rate: Option<usize>,
    pub rates: Option<HullWhite1Factor>,
    pub target: Option<HullWhiteLsvTarget>,
    pub risk_target: Option<HullWhiteLsvTarget>,
    pub rough_weights: Vec<Vec<Vec<f64>>>,
    pub rough_variances: Vec<Vec<f64>>,
    pub near: Vec<Option<usize>>,
}
impl Joint {
    /// Unit-forward escrow quote coordinate, rate diffusion, and deterministic
    /// reserve shift. Simulation history continues to hold residual equity.
    pub fn quote_state(&self, i: usize, row: usize, states: &[f64]) -> Result<[f64; 3], E> {
        if let Some(hw) = &self.assets[i].hw {
            let spot = hw.dividends.initial_spot();
            let node = &hw.dividends.nodes()[row];
            let (f, zeta) = node
                .target_state(states[i] * spot, states[self.rate.unwrap()])
                .map_err(E::numerical)?;
            Ok([
                f / spot,
                zeta / spot,
                node.deterministic_reserve() / node.scale() / spot,
            ])
        } else {
            Ok([states[i], 0.0, 0.0])
        }
    }
    pub fn compile(
        plan: &MultiAssetPricingPlan,
        config: &LocalCorrelationConfig,
        extensions: LocalCorrelationExtensions,
        configs: &[Option<MultiAssetBergomiLsvConfig>],
    ) -> Result<Option<Self>, E> {
        if plan.hull_white.is_none() && extensions.hull_white_target.is_some() {
            return Err(E::Invalid(
                "basket paired target requires shared Hull-White",
            ));
        }
        if plan.lsv_drivers.is_none() {
            if extensions.second_driver_correlations.is_some() {
                return Err(E::Invalid(
                    "full second driver endpoint requires LSV or Hull-White",
                ));
            }
            return Ok(None);
        }
        let (second, rates, target, risk_target) = if let Some(hw) = &plan.hull_white {
            let original = extensions.hull_white_target.ok_or(E::Invalid(
                "local correlation with HW requires a paired basket variance/density target",
            ))?;
            if original.grid() != &config.target {
                return Err(E::Invalid(
                    "paired basket target and local correlation target grid must match",
                ));
            }
            let refined = super::super::hull_white::refine_target(&original, &plan.times)?;
            let risk_target = if original.market_iv_surface().is_some() {
                refined.clone()
            } else {
                original
            };
            (
                super::super::hull_white::compile_drivers(
                    &config.second_correlation,
                    &plan.times,
                    &plan.interval_correlations,
                    configs,
                    extensions.second_driver_correlations,
                    &hw.config,
                )?,
                Some(hw.config.rate_model.clone()),
                Some(refined),
                Some(risk_target),
            )
        } else {
            (
                LsvDrivers::compile(
                    &config.second_correlation,
                    &plan.times,
                    &plan.interval_correlations,
                    configs,
                    extensions.second_driver_correlations,
                )?
                .expect("LSV drivers"),
                None,
                None,
                None,
            )
        };
        let mut drivers = [plan.lsv_drivers.as_ref().unwrap().clone(), second];
        // Deterministic-rate driver compiler represents spot shocks as normals;
        // use actual Brownian increments throughout this joint simulator.
        if rates.is_none() {
            for d in &mut drivers {
                for (row, scales) in d.scales.iter_mut().enumerate() {
                    for s in &mut scales[..plan.assets.len()] {
                        *s *= (plan.times[row + 1] - plan.times[row]).sqrt();
                    }
                }
            }
        }
        let width = drivers[0].scales[0].len();
        let mut offset = plan.assets.len();
        let offsets = configs
            .iter()
            .map(|c| {
                let o = offset;
                offset += c.as_ref().map_or(0, |c| c.factor_count());
                o
            })
            .collect();
        let rate = rates.as_ref().map(|_| offset);
        let mut near = vec![None; plan.assets.len()];
        let mut rough_weights = vec![Vec::new(); plan.assets.len()];
        let mut rough_variances = vec![Vec::new(); plan.assets.len()];
        for (k, &i) in drivers[0].rough_asset_indices.iter().enumerate() {
            near[i] = Some(offset + 2 + k);
            let model = configs[i].as_ref().unwrap().rough().unwrap();
            let mut variances = Vec::new();
            for (r, &t) in plan.times.iter().enumerate() {
                let mut sum = NeumaierSum::new();
                if r > 0 {
                    sum.add((t - plan.times[r - 1]).powf(2.0 * model.hurst()));
                }
                let weights = (0..r.saturating_sub(1))
                    .map(|j| {
                        let w = model
                            .average_kernel(t - plan.times[j + 1], t - plan.times[j])
                            .map_err(E::numerical)?;
                        sum.add(w * w * (plan.times[j + 1] - plan.times[j]));
                        Ok(w)
                    })
                    .collect::<Result<Vec<_>, E>>()?;
                rough_weights[i].push(weights);
                variances.push(sum.total());
            }
            rough_variances[i] = variances;
        }
        Ok(Some(Self {
            assets: plan.assets.clone(),
            configs: configs.to_vec(),
            drivers,
            width,
            offsets,
            rate,
            rates,
            target,
            risk_target,
            rough_weights,
            rough_variances,
            near,
        }))
    }
    pub fn surface(&self, i: usize) -> Option<&LsvLeverageSurface> {
        let a = &self.assets[i];
        a.lsv
            .as_ref()
            .map(|l| l.calibration.surface())
            .or_else(|| a.hw.as_ref()?.calibration.as_ref().map(|c| &c.surface))
    }
    pub fn parameter_counts(&self) -> Vec<usize> {
        self.assets
            .iter()
            .enumerate()
            .map(|(i, a)| {
                if let Some(hw) = &a.hw {
                    return hw.parameter_count();
                }
                self.surface(i).map_or_else(
                    || match &a.model {
                        ModelSpec::LocalVolatility(l) => l.local_variance_grid().values().len(),
                        _ => 1,
                    },
                    |s| s.squared_leverage().len(),
                )
            })
            .collect()
    }
    pub fn log_multiplier(&self, i: usize, row: usize, history: &[Vec<f64>]) -> f64 {
        let v = self.offsets[i];
        match &self.configs[i] {
            None => 0.0,
            Some(MultiAssetBergomiLsvConfig::OneFactor(c)) => {
                c.factor.vol_of_vol() * history[row][v]
            }
            Some(MultiAssetBergomiLsvConfig::TwoFactor(c)) => {
                let w = c.factor.normalized_weights();
                c.factor.vol_of_vol() * (w[0] * history[row][v] + w[1] * history[row][v + 1])
            }
            Some(MultiAssetBergomiLsvConfig::Rough(c)) => {
                let mut sum = NeumaierSum::new();
                if row > 0 {
                    sum.add(if c.factor.hurst() == 0.5 {
                        history[row][v] - history[row - 1][v]
                    } else {
                        history[row][self.near[i].unwrap()]
                    });
                }
                for (k, &w) in self.rough_weights[i][row].iter().enumerate() {
                    sum.add(w * (history[k + 1][v] - history[k][v]));
                }
                0.5 * c.factor.vol_of_vol()
                    * (sum.total() - 0.5 * c.factor.vol_of_vol() * self.rough_variances[i][row])
            }
        }
    }
    pub fn endpoint_noises(&self, row: usize, z: &[f64]) -> Vec<[f64; 2]> {
        let mut values = vec![[0.0; 2]; self.width];
        for e in 0..2 {
            let d = &self.drivers[e];
            let l = d.intervals[row].lower();
            for k in 0..self.width {
                let external = d.permutations.as_ref().map_or(k, |p| p[row][k]);
                values[external][e] = (0..=k)
                    .map(|j| {
                        let source = d.permutations.as_ref().map_or(j, |p| p[row][j]);
                        l[k * self.width + j] * z[e * self.width + source]
                    })
                    .sum::<f64>()
                    * d.scales[row][external];
            }
        }
        values
    }
}
impl LocalCorrelationCalibration {
    pub(super) fn joint_basket(&self, row: usize, states: &[f64]) -> Result<f64, E> {
        let j = self.joint.as_ref().unwrap();
        self.config
            .basket_weights
            .iter()
            .enumerate()
            .map(|(i, w)| Ok(w * j.quote_state(i, row, states)?[0]))
            .sum()
    }
    pub(super) fn joint_basket_shift(&self, row: usize) -> f64 {
        let j = self.joint.as_ref().unwrap();
        self.config
            .basket_weights
            .iter()
            .zip(&j.assets)
            .map(|(w, a)| {
                a.hw.as_ref().map_or(0.0, |hw| {
                    let node = &hw.dividends.nodes()[row];
                    w * node.deterministic_reserve() / node.scale() / hw.dividends.initial_spot()
                })
            })
            .sum()
    }
    pub fn driver_correlation_at(&self, time: f64, log_basket: f64) -> Result<Vec<Vec<f64>>, E> {
        self.correlation_at(time, log_basket)?; // common coverage validation
        let Some(joint) = &self.joint else {
            return self.correlation_at(time, log_basket);
        };
        let row = self.times.partition_point(|t| *t <= time).saturating_sub(1);
        let lambda = self.lookup(row, log_basket).value;
        let d = &joint.drivers;
        let entry = self.entries[row];
        let n = d[0].entries[entry].dimension();
        Ok((0..n)
            .map(|i| {
                (0..n)
                    .map(|j| {
                        (1.0 - lambda) * d[0].entries[entry].canonical()[i * n + j]
                            + lambda * d[1].entries[entry].canonical()[i * n + j]
                    })
                    .collect()
            })
            .collect())
    }
    pub(super) fn joint_sigma(&self, i: usize, row: usize, history: &[Vec<f64>]) -> Result<f64, E> {
        let joint = self.joint.as_ref().unwrap();
        if let Some(s) = joint.surface(i) {
            let f = if let Some(hw) = &joint.assets[i].hw {
                hw.dividends.nodes()[row]
                    .target_state(
                        history[row][i] * hw.dividends.initial_spot(),
                        history[row][joint.rate.unwrap()],
                    )
                    .map_err(E::numerical)?
                    .0
            } else {
                history[row][i]
            };
            let l = crate::engine::processes::hull_white::reverse::lookup(s, row, f);
            Ok(l.value.sqrt() * joint.log_multiplier(i, row, history).exp())
        } else {
            self.sigma(i, row, history[row][i])
        }
    }
    pub(super) fn joint_moments(&self, row: usize, history: &[Vec<f64>]) -> Result<[f64; 2], E> {
        let n = self.models.len();
        let states = &history[row];
        let b = self.joint_basket(row, states)?;
        let a =
            (0..n)
                .map(|i| {
                    Ok(self.config.basket_weights[i]
                        * states[i]
                        * self.joint_sigma(i, row, history)?
                        / b)
                })
                .collect::<Result<Vec<_>, E>>()?;
        let joint = self.joint.as_ref().unwrap();
        let mut rate_loading = 0.0;
        for (i, w) in self.config.basket_weights.iter().enumerate() {
            rate_loading += w * joint.quote_state(i, row, states)?[1] / b;
        }
        Ok(std::array::from_fn(|e| {
            let equity_variance: f64 = (0..n)
                .map(|i| {
                    (0..n)
                        .map(|j| {
                            a[i] * a[j]
                                * self.endpoints[self.entries[row]][e].canonical()[i * n + j]
                        })
                        .sum::<f64>()
                })
                .sum();
            let cross: f64 = joint.rate.map_or(0.0, |r| {
                let corr = &joint.drivers[e].entries[self.entries[row]];
                (0..n)
                    .map(|i| a[i] * corr.canonical()[i * corr.dimension() + r])
                    .sum()
            });
            equity_variance + 2.0 * rate_loading * cross + rate_loading * rate_loading
        }))
    }
    pub(super) fn joint_step(
        &self,
        row: usize,
        history: &[Vec<f64>],
        z: &[f64],
    ) -> Result<Vec<f64>, E> {
        let j = self.joint.as_ref().unwrap();
        let s = &history[row];
        let mut next = s.clone();
        let lambda = self.lookup(row, self.joint_basket(row, s)?.ln()).value;
        let noise: Vec<_> = j
            .endpoint_noises(row, z)
            .iter()
            .map(|v| (1.0 - lambda).sqrt() * v[0] + lambda.sqrt() * v[1])
            .collect();
        let dt = self.times[row + 1] - self.times[row];
        let (integral, shift) = if let Some(r) = j.rate {
            let k = &j.assets[0].hw.as_ref().unwrap().process.kernels[row];
            next[r] = k.transition.rate_decay * s[r] + noise[r];
            let integral = k.transition.integral_loading * s[r] + noise[r + 1];
            next[r + 1] = s[r + 1] + integral;
            (integral, k.integrated_shift)
        } else {
            (0.0, 0.0)
        };
        for i in 0..self.models.len() {
            let sigma = self.joint_sigma(i, row, history)?;
            next[i] = s[i] * (integral + shift - 0.5 * sigma * sigma * dt + sigma * noise[i]).exp();
            if let Some(c) = &j.configs[i] {
                for (k, f) in c.components().iter().enumerate() {
                    let v = j.offsets[i] + k;
                    next[v] = (-f.mean_reversion() * dt).exp() * s[v] + noise[v];
                }
                if let Some(near) = j.near[i] {
                    next[near] = noise[near];
                }
            }
        }
        if next.iter().any(|x| !x.is_finite())
            || next[..self.models.len()].iter().any(|x| *x <= 0.0)
        {
            return Err(E::Invalid(
                "nonfinite/nonpositive joint local correlation state",
            ));
        }
        Ok(next)
    }
    pub(super) fn initial_joint_state(&self) -> Vec<f64> {
        let mut s = vec![0.0; self.joint.as_ref().unwrap().width];
        s[..self.models.len()].fill(1.0);
        s
    }
    pub(super) fn evolve_joint(&self, independent: &[Vec<f64>]) -> Result<LocalCorrelationPath, E> {
        let j = self.joint.as_ref().unwrap();
        let nt = self.times.len();
        if independent.len() != 2 * j.width
            || independent
                .iter()
                .any(|v| v.len() != nt - 1 || v.iter().any(|x| !x.is_finite()))
        {
            return Err(E::Invalid(
                "joint local correlation shock dimensions/values",
            ));
        }
        let mut states = vec![self.initial_joint_state()];
        let mut boundary_counts = vec![0.0; self.models.len()];
        for row in 0..nt - 1 {
            for (i, count) in boundary_counts.iter_mut().enumerate() {
                let xs = if let Some(s) = j.surface(i) {
                    Some(s.log_nodes())
                } else if let ModelSpec::LocalVolatility(m) = &self.models[i] {
                    Some(m.local_variance_grid().log_moneyness_nodes())
                } else {
                    None
                };
                if let Some(xs) = xs {
                    let x = j.quote_state(i, row, &states[row])?[0].ln();
                    if x < xs[0] || x > xs[xs.len() - 1] {
                        *count += 1.0;
                    }
                }
            }
            let z = independent.iter().map(|v| v[row]).collect::<Vec<_>>();
            states.push(self.joint_step(row, &states, &z)?);
        }
        Ok(LocalCorrelationPath {
            states,
            independent: independent.to_vec(),
            boundary_counts,
        })
    }
}
