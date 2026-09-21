use super::super::evaluate::Layout;
use super::super::path::AssetPath;
use super::*;
use crate::mc::DeterministicStatistics;
use std::sync::Arc;

impl MultiAssetPricingPlan {
    pub fn local_correlation_calibration(&self) -> Option<&LocalCorrelationCalibration> {
        self.local_correlation.as_deref()
    }
    pub(in crate::engine::multi_asset) fn local_correlation_paths(
        &self,
        shocks: &[Vec<f64>],
    ) -> Result<Vec<AssetPath>, E> {
        let calibration = self.local_correlation.as_ref().expect("local correlation");
        let path = Arc::new(calibration.evolve(shocks)?);
        if self.hull_white.is_some() {
            return self.joint_hw_paths(path);
        }
        self.assets.iter().enumerate().map(|(i, a)| {
            let initial = a.forward.spot().get();
            let f: Vec<_> = path.states.iter().zip(a.process.forward_normalizers())
                .map(|(m, forward)| m[i] * forward).collect();
            let reconstruct = |coordinates: &[crate::market::AffineDividendCoordinate]| {
                f.iter().zip(coordinates).map(|(&f, c)| c.reconstruct_spot(a.forward.spot(), f)).collect::<Vec<_>>()
            };
            let spots = reconstruct(&a.coordinates);
            let pre_spots = reconstruct(&a.pre_coordinates);
            if spots.iter().chain(&pre_spots).any(|s| !s.is_finite() || *s <= 0.0) {
                return Err(E::Invalid(
                    "nonpositive/nonfinite local-correlation physical spot; no flooring or resampling",
                ));
            }
            Ok(AssetPath {
                spots, pre_spots,
                spot_derivatives: f.iter().zip(&a.coordinates).map(|(f, c)| c.spot_scale()*f/initial).collect(),
                pre_spot_derivatives: f.iter().zip(&a.pre_coordinates).map(|(f, c)| c.spot_scale()*f/initial).collect(),
                bs_vega: vec![0.0; self.times.len()], local: None, lsv: None, hw: None,
                local_correlation: Some(Arc::clone(&path)),
            })
        }).collect()
    }
    pub(in crate::engine::multi_asset) fn local_correlation_path_risk(
        &self,
        paths: &[AssetPath],
        seeds: &[Vec<f64>],
        layout: &Layout,
        output: &mut [f64],
    ) -> Result<(), E> {
        let c = self.local_correlation.as_ref().expect("local correlation");
        let normalized: Vec<Vec<f64>> = seeds
            .iter()
            .zip(&self.assets)
            .map(|(s, a)| {
                s.iter()
                    .zip(a.process.forward_normalizers())
                    .map(|(s, f)| s * f)
                    .collect()
            })
            .collect();
        let (direct, mixing) = c.reverse_path(
            paths[0].local_correlation.as_ref().expect("coupled path"),
            &normalized,
        )?;
        for (values, &(offset, count)) in direct.iter().zip(&layout.vega) {
            output[offset..offset + count].copy_from_slice(values);
        }
        let (offset, count) = layout.local_correlation;
        output[offset..offset + count].copy_from_slice(&mixing);
        Ok(())
    }
    pub(in crate::engine::multi_asset) fn local_correlation_target_means(
        &self,
        stats: &[DeterministicStatistics],
        units: u64,
        layout: &Layout,
        risk: bool,
    ) -> Result<Option<Vec<f64>>, E> {
        let Some(c) = self.local_correlation.as_ref().filter(|_| risk) else {
            return Ok(None);
        };
        let (offset, count) = layout.local_correlation;
        let mean: Vec<_> = stats[offset..offset + count]
            .iter()
            .map(|s| s.sum().total() / units as f64)
            .collect();
        let (mut assets, basket) = c.reverse_calibration(&mean)?;
        for (values, &(o, n)) in assets.iter_mut().zip(&layout.vega) {
            for (value, stat) in values.iter_mut().zip(&stats[o..o + n]) {
                *value += stat.sum().total() / units as f64;
            }
        }
        for (i, values) in assets.iter_mut().enumerate() {
            let a = &self.assets[i];
            if a.has_lsv() {
                *values = if let Some(hw) = &a.hw {
                    hw.target_reverse(values)?
                } else {
                    a.lsv.as_ref().unwrap().target_reverse(values)?
                };
            } else if a.hw.is_some() {
                values.truncate(1);
            }
        }
        Ok(Some(
            basket
                .into_iter()
                .chain(assets.into_iter().flatten())
                .collect(),
        ))
    }
    pub(in crate::engine::multi_asset) fn make_local_correlation_risk(
        &self,
        values: Vec<f64>,
        errors: Option<Vec<f64>>,
    ) -> LocalCorrelationRisk {
        let c = self.local_correlation.as_ref().expect("local correlation");
        if c.joint.is_some() {
            return self.make_joint_correlation_risk(values, errors);
        }
        let nb = c.config.target.values().len();
        let split = |v: &[f64]| {
            let mut cursor = nb;
            c.parameter_counts()
                .into_iter()
                .map(|n| {
                    let r = v[cursor..cursor + n].to_vec();
                    cursor += n;
                    r
                })
                .collect()
        };
        LocalCorrelationRisk {
            basket_time_nodes: c.config.target.time_nodes().to_vec(),
            basket_log_nodes: c.log_nodes().to_vec(),
            basket_variance_adjoints: values[..nb].to_vec(),
            basket_standard_errors: errors.as_ref().map(|e| e[..nb].to_vec()),
            asset_adjoints: split(&values),
            asset_standard_errors: errors.as_ref().map(|e| split(e)),
            asset_time_nodes: c
                .models
                .iter()
                .map(|m| {
                    if let ModelSpec::LocalVolatility(lv) = m {
                        lv.local_variance_grid().time_nodes().to_vec()
                    } else {
                        Vec::new()
                    }
                })
                .collect(),
            asset_log_nodes: c
                .models
                .iter()
                .map(|m| {
                    if let ModelSpec::LocalVolatility(lv) = m {
                        lv.local_variance_grid().log_moneyness_nodes().to_vec()
                    } else {
                        Vec::new()
                    }
                })
                .collect(),
            basket_hull_white: None,
            asset_hull_white: vec![None; self.assets.len()],
            method: "local-correlation-finite-particle-reverse-v1",
        }
    }
}

impl LocalCorrelationCalibration {
    pub(in crate::engine::multi_asset) fn validate_reverse(&self) -> Result<(), E> {
        if !self.retains_reverse_trace() {
            return Err(E::Invalid(
                "local correlation AAD requires retain_reverse_trace=true",
            ));
        }
        let used = (self.times.len() - 1) * self.log_nodes().len();
        if self.diagnostics[..used]
            .iter()
            .any(|d| !d.unidentifiable && (d.raw_mixing == 0.0 || d.raw_mixing == 1.0))
        {
            return Err(E::Invalid(
                "local correlation AAD is undefined at an exact projection active-set transition",
            ));
        }
        Ok(())
    }
    pub(in crate::engine::multi_asset) fn fingerprint(&self, h: &mut blake3::Hasher) {
        use super::super::compile::floats;
        h.update(b"local-correlation-normalized-basket-particle-quartic-psd-mixture-v1");
        floats(h, &self.config.basket_weights);
        floats(h, self.config.target.time_nodes());
        floats(h, self.log_nodes());
        floats(h, self.config.target.values());
        floats(
            h,
            &[
                self.config.target.floor(),
                self.config.target.cap(),
                self.config.minimum_variance_span,
                self.config.particles.log_bandwidth(),
                self.config.particles.minimum_effective_samples(),
            ],
        );
        h.update(&(self.config.particles.particle_count() as u64).to_be_bytes());
        h.update(&self.config.particles.seed().to_be_bytes());
        h.update(&[
            u8::from(self.config.particles.retain_reverse_trace()),
            u8::from(self.config.feasibility == LocalCorrelationFeasibility::ProjectAndReport),
        ]);
        for (_, endpoint) in self.config.second_correlation.entries() {
            floats(h, endpoint.raw());
            floats(h, endpoint.canonical());
        }
        if let Some(j) = &self.joint {
            h.update(b"joint-lsv-hw-local-correlation-v1");
            for d in &j.drivers[1].entries {
                floats(h, d.raw());
                floats(h, d.canonical());
            }
            if let Some(t) = &j.target {
                floats(h, t.grid().time_nodes());
                floats(h, t.grid().values());
                floats(h, t.log_densities());
                if let Some(iv) = t.market_iv_surface() {
                    floats(h, iv.maturity_nodes());
                    floats(h, iv.log_moneyness_nodes());
                    floats(h, iv.implied_volatilities());
                }
            }
        }
        floats(h, &self.mixing);
        for d in &self.diagnostics {
            h.update(&(d.source_node as u64).to_be_bytes());
            h.update(&[u8::from(d.projected), u8::from(d.unidentifiable)]);
        }
    }
}
