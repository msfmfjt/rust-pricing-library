use super::super::{evaluate::Layout, path::AssetPath};
use super::*;
use crate::models::hull_white_dividends::transpose_log_curve;
use crate::product::PayoffEvaluation;

impl MultiAssetPricingPlan {
    /// Compile the linear transpose into the small public Spot/curve report.
    /// Applying it before statistical reduction preserves covariance for both
    /// pseudo-MC and RQMC. Particle calibration is never reversed per path.
    pub(in crate::engine::multi_asset) fn hw_market_weights(
        &self,
        layout: &Layout,
    ) -> Result<&MarketWeights, E> {
        if let Some(weights) = self.market_weights.get() {
            return Ok(weights);
        }
        let mut result = Vec::new();
        let ranges = layout
            .vega
            .iter()
            .copied()
            .chain(std::iter::once(layout.local_correlation));
        for (start, count) in ranges {
            for source in start..start + count {
                let mut parameters: Vec<_> = self
                    .assets
                    .iter()
                    .map(|a| vec![0.0; a.hw.as_ref().unwrap().parameter_count()])
                    .collect();
                if source >= layout.local_correlation.0
                    && source < layout.local_correlation.0 + layout.local_correlation.1
                {
                    let mut mixing = vec![0.0; layout.local_correlation.1];
                    mixing[source - layout.local_correlation.0] = 1.0;
                    parameters = self
                        .local_correlation
                        .as_ref()
                        .unwrap()
                        .reverse_calibration(&mixing)?
                        .0;
                } else {
                    for (values, &(o, n)) in parameters.iter_mut().zip(&layout.vega) {
                        if source >= o && source < o + n {
                            values[source - o] = 1.0;
                        }
                    }
                }
                let mut weights = Vec::new();
                for (i, (a, values)) in self.assets.iter().zip(parameters).enumerate() {
                    if values.iter().all(|v| *v == 0.0) {
                        continue;
                    }
                    let market = a.hw.as_ref().unwrap().market_reverse(&values)?;
                    weights.push((1 + i, market.spot));
                    weights.extend(
                        market
                            .discount_log_df
                            .iter()
                            .enumerate()
                            .map(|(j, v)| (layout.discount.0 + j, *v)),
                    );
                    weights.extend(
                        market
                            .dividend_log_df
                            .iter()
                            .enumerate()
                            .map(|(j, v)| (layout.dividends[i].0 + j, *v)),
                    );
                }
                weights.retain(|(_, w)| *w != 0.0);
                if !weights.is_empty() {
                    result.push((source, weights));
                }
            }
        }
        let _ = self.market_weights.set(result);
        Ok(self
            .market_weights
            .get()
            .expect("initialized market transpose"))
    }
    pub(in crate::engine::multi_asset) fn recalibrated_gamma(&self) -> bool {
        self.hull_white.is_some()
            && self.assets.iter().any(|a| {
                a.forward
                    .discrete_dividends()
                    .is_some_and(|d| d.initial_reserve() > 0.0)
            })
            && (self.local_correlation.is_some() || self.assets.iter().any(Asset::has_lsv))
    }
    pub(in crate::engine::multi_asset) fn hw_spot_bump(
        &self,
        asset: usize,
        bump: f64,
    ) -> Result<Self, E> {
        let mut plan = self.clone();
        plan.market_weights = std::sync::Arc::new(std::sync::OnceLock::new());
        let a = &mut plan.assets[asset];
        let spot = crate::core::PositiveF64::new(a.forward.spot().get() + bump, "bumped Spot")
            .map_err(E::numerical)?;
        a.forward = a.forward.with_spot(spot)?;
        let old = a.hw.as_ref().unwrap();
        a.hw = Some(std::sync::Arc::new(HwAsset::compile(
            &a.forward,
            &a.model,
            a.process.time_grid(),
            old.config.clone(),
            &plan.hull_white.as_ref().unwrap().config,
            asset,
            old.driver_offset,
        )?));
        if let Some(cal) = &self.local_correlation {
            plan.local_correlation = Some(std::sync::Arc::new(cal.recalibrate_hw_assets(&plan)?));
        }
        Ok(plan)
    }
    pub(in crate::engine::multi_asset) fn make_target_risks(
        &self,
        asset: usize,
        values: Vec<f64>,
        errors: Option<Vec<f64>>,
    ) -> (
        Option<MultiAssetLsvRisk>,
        Option<MultiAssetHullWhiteLsvRisk>,
    ) {
        let a = &self.assets[asset];
        let Some(hw) = &a.hw else {
            return (
                Some(a.lsv.as_ref().expect("LSV").risk(values, errors)),
                None,
            );
        };
        let target = hw.target.as_ref().expect("HW target");
        let g = target.grid();
        let n = g.values().len();
        let iv = target.market_iv_surface();
        let nq = iv.map_or(0, |iv| iv.implied_volatilities().len());
        let variance = MultiAssetLsvRisk {
            time_nodes: g.time_nodes().to_vec(),
            log_moneyness_nodes: g.log_moneyness_nodes().to_vec(),
            node_adjoints: values[..n].to_vec(),
            standard_errors: errors.as_ref().map(|v| v[..n].to_vec()),
            method: "hw-discounted-paired-calibration-vjp-v1",
        };
        let risk = MultiAssetHullWhiteLsvRisk {
            forward_log_density_adjoints: values[n..2 * n].to_vec(),
            density_standard_errors: errors.as_ref().map(|v| v[n..2 * n].to_vec()),
            vega_kt_raw: iv.map(|_| values[2 * n..2 * n + nq].to_vec()),
            vega_kt_market_scaled: iv
                .map(|_| values[2 * n..2 * n + nq].iter().map(|v| 0.01 * v).collect()),
            vega_kt_standard_errors: iv
                .and_then(|_| errors.as_ref().map(|v| v[2 * n..2 * n + nq].to_vec())),
            vega_kt_maturity_nodes: iv.map_or_else(Vec::new, |v| v.maturity_nodes().to_vec()),
            vega_kt_log_moneyness_nodes: iv
                .map_or_else(Vec::new, |v| v.log_moneyness_nodes().to_vec()),
            parallel_vega: iv.map(|_| values[2 * n + nq]),
            parallel_vega_standard_error: iv.and_then(|_| errors.as_ref().map(|v| v[2 * n + nq])),
            method: "hw-discounted-paired-calibration-vjp-v1",
        };
        (Some(variance), Some(risk))
    }

    pub(in crate::engine::multi_asset) fn hw_path_risk(
        &self,
        paths: &[AssetPath],
        payoff: &PayoffEvaluation,
        layout: &Layout,
        values: &mut [f64],
    ) -> Result<(), E> {
        let mut seeds = vec![vec![(0.0, 0.0); self.times.len()]; paths.len()];
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
            let (i, j) = self.indices(u, d).expect("validated observation");
            if pre {
                seeds[i][j].1 += v;
            } else {
                seeds[i][j].0 += v;
            }
        }
        let (discount_offset, discount_count) = layout.discount;
        for (i, (a, p)) in self.assets.iter().zip(paths).enumerate() {
            let hw = a.hw.as_ref().expect("HW asset");
            let path = p.hw.as_ref().expect("HW path");
            let (offset, count) = layout.vega[i];
            let parameters = &mut values[offset..offset + count];
            let (equity, _) = hw.observation_reverse(&path.states, &seeds[i], parameters)?;
            let adj = hw.reverse(path, &equity)?;
            let nv = hw.volatility_parameter_count();
            if hw.calibration.is_some() {
                parameters[..nv].copy_from_slice(&adj.squared_leverage);
            } else {
                parameters[0] = adj.bs_volatility.expect("BS sigma risk");
            }
            parameters[nv] += adj.initial_spot;
            if let Some(nodes) = adj.dividends {
                hw.add_dividend_parameters(&nodes, parameters);
            }
            values[1 + i] = 0.0;
        }
        let hw = self.hull_white.as_ref().expect("HW context");
        let states = &paths[0].hw.as_ref().expect("HW path").states;
        for cashflow in &hw.cashflows {
            let payoff = cashflow.payoff.evaluate_with_pre_dividend_spots(
                |u, d| self.observe(paths, u, d, false, None),
                |u, d| self.observe(paths, u, d, true, None),
            )?[0];
            let pv = payoff * hw.discount(cashflow, states)?;
            transpose_log_curve(
                self.assets[0].forward.discount_curve(),
                cashflow.payment_time,
                pv,
                &mut values[discount_offset..discount_offset + discount_count],
            )
            .map_err(E::numerical)?;
        }
        Ok(())
    }
}
