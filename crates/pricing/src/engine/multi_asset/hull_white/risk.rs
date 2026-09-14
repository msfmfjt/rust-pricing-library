use super::super::{evaluate::Layout, path::AssetPath};
use super::*;
use crate::models::hull_white_dividends::transpose_log_curve;
use crate::product::PayoffEvaluation;

impl MultiAssetPricingPlan {
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
            let affine = hw
                .affine
                .reverse(&path.affine, &path.physical_states, &seeds[i], 1.0)
                .map_err(E::numerical)?;
            let spot = a.forward.spot().get();
            let unit_seeds: Vec<_> = affine.equity.iter().map(|v| spot * v).collect();
            let adj = hw.reverse(path, &unit_seeds)?;
            let (offset, count) = layout.vega[i];
            if hw.calibration.is_some() {
                values[offset..offset + count].copy_from_slice(&adj.squared_leverage);
            } else {
                values[offset] = adj.bs_volatility.expect("BS sigma risk");
            }
            for (j, v) in affine.discount_log_df.iter().enumerate() {
                values[discount_offset + j] += v;
            }
            let (offset, count) = layout.dividends[i];
            values[offset..offset + count].copy_from_slice(&affine.dividend_log_df);
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
