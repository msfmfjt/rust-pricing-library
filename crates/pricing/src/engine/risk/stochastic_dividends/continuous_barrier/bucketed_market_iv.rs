//! Selected retained residual-forward IV quotes, rebuilt and recalibrated one at a time.
use super::sampling::BumpStatistics;
use super::*;

const METHOD: &str =
    "buehler-rough-residual-lsv-continuous-bridge-bucketed-market-iv-recalibrated-crn-v1";

/// Finite quote-IV risk per unit absolute volatility in the residual-forward
/// coordinate. Selected quotes move independently; rows preserve requested order.
/// Columns follow half/base/double bumps or the two adjacent gaps. Each scenario
/// rebuilds original-grid Dupire variance and fully recalibrates leverage.
#[derive(Clone, Debug, PartialEq)]
pub struct StochasticDividendContinuousBarrierBucketedMarketIvRisk {
    pub price: StochasticDividendPrice,
    /// Unique zero-based indices: maturity_index * quote_log_moneyness_nodes.len() + x_index.
    pub quote_indices: Vec<usize>,
    /// Retained quote axes and row-major IV, not the original/refined variance grid.
    pub quote_maturity_nodes: Vec<f64>,
    pub quote_log_moneyness_nodes: Vec<f64>,
    pub implied_volatilities: Vec<f64>,
    pub implied_volatility_bumps: [f64; 3],
    pub vega_estimates: Vec<[f64; 3]>,
    pub vega_standard_errors: Vec<[f64; 3]>,
    pub bump_differences: Vec<[f64; 2]>,
    pub bump_difference_standard_errors: Vec<[f64; 2]>,
    /// Sum of the selected finite node risks, paired before reduction.
    /// At a finite bump this need not equal a simultaneous parallel shift.
    pub sum_vega_estimates: [f64; 3],
    pub sum_vega_standard_errors: [f64; 3],
    pub sum_bump_differences: [f64; 2],
    pub sum_bump_difference_standard_errors: [f64; 2],
    /// Includes the base valuation; excludes calibration particle paths.
    pub scenario_evaluated_paths: u128,
    pub payoff_evaluations: u128,
    pub recalibration_count: usize,
    pub risk_fingerprint: Fingerprint,
    pub method: &'static str,
}
impl StochasticDividendContinuousBarrierBucketedMarketIvRisk {
    #[must_use]
    pub const fn interpolation(&self) -> &'static str {
        crate::market::MARKET_IV_INTERPOLATION
    }

    #[must_use]
    pub const fn uncertainty_scope(&self) -> &'static str {
        "pricing_only_fixed_calibration_seed_grid_bridge_and_quote_interpolation"
    }
}

impl StochasticDividendContinuousBarrierPlan {
    /// Shift each selected quote alone by +/- h/2, h and 2h. Selection must be
    /// nonempty, unique and in range. Validate all shifted quote surfaces and
    /// original Dupire grids before calibration, rejecting nonpositive or
    /// unrepresentable quotes, sampled arbitrage and any variance repair.
    /// Six calibrations and six valuation paths per selected quote, plus base.
    /// Sum errors include cross-quote sampling covariance. No SSVI refit occurs.
    pub fn evaluate_bucketed_market_iv_risk(
        &self,
        implied_volatility_bump: f64,
        quote_indices: &[usize],
    ) -> Result<StochasticDividendContinuousBarrierBucketedMarketIvRisk, MonteCarloError> {
        let surface = self
            .market_iv_surface
            .as_ref()
            .ok_or_else(|| invalid("continuous_market_iv_source_not_retained"))?;
        let mut seen = std::collections::BTreeSet::new();
        if quote_indices.is_empty()
            || quote_indices
                .iter()
                .any(|&i| i >= surface.implied_volatilities().len() || !seen.insert(i))
        {
            return Err(invalid("continuous_market_iv_quote_indices").into());
        }
        let bumps = [
            0.5 * implied_volatility_bump,
            implied_volatility_bump,
            2.0 * implied_volatility_bump,
        ];
        let count = quote_indices.len();
        let selections = quote_indices.iter().copied().map(Some).collect::<Vec<_>>();
        let scenarios = self.market_iv_scenarios_for_quotes(&bumps, &selections)?;
        let node_width = 1 + 5 * count;
        let BumpStatistics {
            values,
            errors,
            units,
            paths,
        } = self.bucketed_volatility_statistics(&scenarios, &bumps)?;
        let triples = |v: &[f64]| {
            (0..count)
                .map(|b| [v[1 + 5 * b], v[2 + 5 * b], v[3 + 5 * b]])
                .collect()
        };
        let gaps = |v: &[f64]| (0..count).map(|b| [v[4 + 5 * b], v[5 + 5 * b]]).collect();
        let mut hash = blake3::Hasher::new();
        hash.update(METHOD.as_bytes());
        hash.update(self.plan_fingerprint().as_bytes());
        hash.update(&(count as u64).to_le_bytes());
        for &i in quote_indices {
            hash.update(&(i as u64).to_le_bytes());
        }
        for h in bumps {
            hash.update(&h.to_bits().to_le_bytes());
        }
        let recalibration_count = scenarios.len();
        let scenario_evaluated_paths = (recalibration_count as u128 + 1) * paths;
        Ok(StochasticDividendContinuousBarrierBucketedMarketIvRisk {
            price: StochasticDividendPrice {
                value: values[0],
                standard_error: errors[0],
                independent_sampling_units: units,
                evaluated_paths: paths,
                plan_fingerprint: self.plan_fingerprint(),
                scheme: SCHEME,
            },
            quote_indices: quote_indices.to_vec(),
            quote_maturity_nodes: surface.maturity_nodes().to_vec(),
            quote_log_moneyness_nodes: surface.log_moneyness_nodes().to_vec(),
            implied_volatilities: surface.implied_volatilities().to_vec(),
            implied_volatility_bumps: bumps,
            vega_estimates: triples(&values),
            vega_standard_errors: triples(&errors),
            bump_differences: gaps(&values),
            bump_difference_standard_errors: gaps(&errors),
            sum_vega_estimates: values[node_width..node_width + 3].try_into().unwrap(),
            sum_vega_standard_errors: errors[node_width..node_width + 3].try_into().unwrap(),
            sum_bump_differences: values[node_width + 3..].try_into().unwrap(),
            sum_bump_difference_standard_errors: errors[node_width + 3..].try_into().unwrap(),
            scenario_evaluated_paths,
            payoff_evaluations: scenario_evaluated_paths,
            recalibration_count,
            risk_fingerprint: Fingerprint::from_bytes(*hash.finalize().as_bytes()),
            method: METHOD,
        })
    }
}
