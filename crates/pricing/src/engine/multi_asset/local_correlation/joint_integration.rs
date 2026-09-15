use super::super::{evaluate::Layout, hull_white::HwAssetPath, path::AssetPath};
use super::*;
use crate::mc::hull_white::{HullWhiteLsvTarget, HybridState};
use crate::models::hull_white_dividends::transpose_log_curve;
use crate::product::PayoffEvaluation;
use std::sync::Arc;

impl MultiAssetPricingPlan {
    pub(super) fn joint_hw_paths(
        &self,
        path: Arc<LocalCorrelationPath>,
    ) -> Result<Vec<AssetPath>, E> {
        let cal = self.local_correlation.as_ref().unwrap();
        let j = cal.joint.as_ref().unwrap();
        let r = j.rate.unwrap();
        self.assets
            .iter()
            .enumerate()
            .map(|(i, a)| {
                let hw = a.hw.as_ref().unwrap();
                let spot = a.forward.spot().get();
                let states: Vec<_> = path
                    .states
                    .iter()
                    .enumerate()
                    .map(|(row, s)| HybridState {
                        normalized_equity: s[i],
                        volatility_factor: j.log_multiplier(i, row, &path.states),
                        rate_factor: s[r],
                        integrated_rate_factor: s[r + 1],
                    })
                    .collect();
                let physical_states: Vec<_> = states
                    .iter()
                    .map(|s| HybridState {
                        normalized_equity: s.normalized_equity * spot,
                        ..*s
                    })
                    .collect();
                let affine = hw.affine.record(&physical_states).map_err(E::numerical)?;
                Ok(AssetPath {
                    spots: affine.spots.iter().map(|(s, _)| *s).collect(),
                    pre_spots: affine.spots.iter().map(|(s, p)| p.unwrap_or(*s)).collect(),
                    spot_derivatives: states
                        .iter()
                        .enumerate()
                        .map(|(k, s)| hw.affine.scale(k, false) * s.normalized_equity)
                        .collect(),
                    pre_spot_derivatives: states
                        .iter()
                        .enumerate()
                        .map(|(k, s)| hw.affine.scale(k, true) * s.normalized_equity)
                        .collect(),
                    bs_vega: Vec::new(),
                    local: None,
                    lsv: None,
                    local_correlation: Some(Arc::clone(&path)),
                    hw: Some(HwAssetPath {
                        states,
                        physical_states,
                        equity_increments: Vec::new(),
                        affine,
                    }),
                })
            })
            .collect()
    }
    pub(in crate::engine::multi_asset) fn joint_hw_path_risk(
        &self,
        paths: &[AssetPath],
        payoff: &PayoffEvaluation,
        layout: &Layout,
        output: &mut [f64],
    ) -> Result<(), E> {
        let cal = self.local_correlation.as_ref().unwrap();
        let joint = cal.joint.as_ref().unwrap();
        let r = joint.rate.unwrap();
        let nt = self.times.len();
        let mut physical = vec![vec![(0.0, 0.0); nt]; paths.len()];
        for (pre, u, d, v) in payoff
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
            let (i, k) = self.indices(u, d).unwrap();
            if pre {
                physical[i][k].1 += v;
            } else {
                physical[i][k].0 += v;
            }
        }
        let mut seeds = vec![vec![0.0; nt]; joint.width];
        let (doff, dcount) = layout.discount;
        for (i, (a, p)) in self.assets.iter().zip(paths).enumerate() {
            let hw = a.hw.as_ref().unwrap();
            let path = p.hw.as_ref().unwrap();
            let adj = hw
                .affine
                .reverse(&path.affine, &path.physical_states, &physical[i], 1.0)
                .map_err(E::numerical)?;
            for (seed, bar) in seeds[i].iter_mut().zip(&adj.equity) {
                *seed += a.forward.spot().get() * bar;
            }
            for (seed, bar) in seeds[r + 1].iter_mut().zip(&adj.integrated_rate) {
                *seed += bar;
            }
            for (k, v) in adj.discount_log_df.iter().enumerate() {
                output[doff + k] += v;
            }
            let (offset, count) = layout.dividends[i];
            output[offset..offset + count].copy_from_slice(&adj.dividend_log_df);
        }
        let hw = self.hull_white.as_ref().unwrap();
        let states = &paths[0].hw.as_ref().unwrap().states;
        for c in &hw.cashflows {
            let value = c.payoff.evaluate_with_pre_dividend_spots(
                |u, d| self.observe(paths, u, d, false, None),
                |u, d| self.observe(paths, u, d, true, None),
            )?[0];
            let pv = value * hw.discount(c, states)?;
            seeds[r + 1][c.observation_node] -= pv;
            seeds[r][c.observation_node] -= c.rate_loading * pv;
            transpose_log_curve(
                self.assets[0].forward.discount_curve(),
                c.payment_time,
                pv,
                &mut output[doff..doff + dcount],
            )
            .map_err(E::numerical)?;
        }
        let (direct, mixing) =
            cal.reverse_joint_path(paths[0].local_correlation.as_ref().unwrap(), &seeds)?;
        for (v, &(o, n)) in direct.iter().zip(&layout.vega) {
            output[o..o + n].copy_from_slice(v);
        }
        let (o, n) = layout.local_correlation;
        output[o..o + n].copy_from_slice(&mixing);
        Ok(())
    }
}

pub(super) fn target_count(t: &HullWhiteLsvTarget) -> usize {
    2 * t.grid().values().len()
        + t.market_iv_surface()
            .map_or(0, |iv| iv.implied_volatilities().len() + 1)
}
pub(super) fn paired_risk(
    t: &HullWhiteLsvTarget,
    v: &[f64],
    errors: Option<&[f64]>,
) -> MultiAssetHullWhiteLsvRisk {
    let n = t.grid().values().len();
    let iv = t.market_iv_surface();
    let nq = iv.map_or(0, |iv| iv.implied_volatilities().len());
    MultiAssetHullWhiteLsvRisk {
        forward_log_density_adjoints: v[n..2 * n].to_vec(),
        density_standard_errors: errors.map(|e| e[n..2 * n].to_vec()),
        vega_kt_raw: iv.map(|_| v[2 * n..2 * n + nq].to_vec()),
        vega_kt_market_scaled: iv.map(|_| v[2 * n..2 * n + nq].iter().map(|x| 0.01 * x).collect()),
        vega_kt_standard_errors: iv.and_then(|_| errors.map(|e| e[2 * n..2 * n + nq].to_vec())),
        vega_kt_maturity_nodes: iv.map_or_else(Vec::new, |v| v.maturity_nodes().to_vec()),
        vega_kt_log_moneyness_nodes: iv.map_or_else(Vec::new, |v| v.log_moneyness_nodes().to_vec()),
        parallel_vega: iv.map(|_| v[2 * n + nq]),
        parallel_vega_standard_error: iv.and_then(|_| errors.map(|e| e[2 * n + nq])),
        method: "local-correlation-joint-discounted-particle-reverse-v1",
    }
}
impl MultiAssetPricingPlan {
    pub(super) fn make_joint_correlation_risk(
        &self,
        values: Vec<f64>,
        errors: Option<Vec<f64>>,
    ) -> LocalCorrelationRisk {
        use joint_integration::{paired_risk, target_count};
        let c = self.local_correlation.as_ref().unwrap();
        let j = c.joint.as_ref().unwrap();
        let bt = j.risk_target.as_ref();
        let grid = bt.map_or(&c.config.target, |t| t.grid());
        let nb = grid.values().len();
        let mut cursor = bt.map_or(nb, target_count);
        let mut asset_adjoints = Vec::new();
        let mut asset_errors = Vec::new();
        let mut asset_time_nodes = Vec::new();
        let mut asset_log_nodes = Vec::new();
        let mut asset_hull_white = Vec::new();
        for a in &self.assets {
            let paired = a.hw.as_ref().and_then(|h| h.target.as_ref());
            let g = paired.map(|t| t.grid()).or_else(|| {
                if let ModelSpec::LocalVolatility(lv) = &a.model {
                    Some(lv.local_variance_grid())
                } else {
                    None
                }
            });
            let n = g.map_or(1, |g| g.values().len());
            let count = paired.map_or(n, target_count);
            asset_adjoints.push(values[cursor..cursor + n].to_vec());
            asset_errors.push(
                errors
                    .as_ref()
                    .map(|e| e[cursor..cursor + n].to_vec())
                    .unwrap_or_default(),
            );
            asset_time_nodes.push(g.map_or_else(Vec::new, |g| g.time_nodes().to_vec()));
            asset_log_nodes.push(g.map_or_else(Vec::new, |g| g.log_moneyness_nodes().to_vec()));
            asset_hull_white.push(paired.map(|t| {
                paired_risk(
                    t,
                    &values[cursor..cursor + count],
                    errors.as_ref().map(|e| &e[cursor..cursor + count]),
                )
            }));
            cursor += count;
        }
        debug_assert_eq!(cursor, values.len());
        LocalCorrelationRisk {
            basket_time_nodes: grid.time_nodes().to_vec(),
            basket_log_nodes: grid.log_moneyness_nodes().to_vec(),
            basket_variance_adjoints: values[..nb].to_vec(),
            basket_standard_errors: errors.as_ref().map(|e| e[..nb].to_vec()),
            basket_hull_white: bt.map(|t| {
                paired_risk(
                    t,
                    &values[..target_count(t)],
                    errors.as_ref().map(|e| &e[..target_count(t)]),
                )
            }),
            asset_adjoints,
            asset_standard_errors: errors.map(|_| asset_errors),
            asset_time_nodes,
            asset_log_nodes,
            asset_hull_white,
            method: "local-correlation-joint-discounted-particle-reverse-v1",
        }
    }
}
