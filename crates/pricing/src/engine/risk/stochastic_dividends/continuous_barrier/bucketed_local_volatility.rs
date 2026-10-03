//! Selected original-target Local-volatility nodes, recalibrated one at a time.
use super::sampling::BumpStatistics;
use super::*;

const METHOD: &str =
    "buehler-rough-residual-lsv-continuous-bridge-bucketed-local-vol-recalibrated-crn-v1";

/// Finite-bump risk per unit absolute residual Local volatility. Each selected
/// original node is shifted alone, before refinement and full recalibration.
/// Rows follow the requested node order; columns follow the half/base/double
/// ladder (or the two adjacent gaps). These are not market-IV/VegaKT buckets.
#[derive(Clone, Debug, PartialEq)]
pub struct StochasticDividendContinuousBarrierBucketedLocalVolatilityRisk {
    pub price: StochasticDividendPrice,
    /// Unique zero-based indices: time_index * log_moneyness_nodes.len() + x_index.
    pub node_indices: Vec<usize>,
    /// Original target grid, not the refined execution grid.
    pub time_nodes: Vec<f64>,
    pub log_moneyness_nodes: Vec<f64>,
    pub local_volatility_bumps: [f64; 3],
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
impl StochasticDividendContinuousBarrierBucketedLocalVolatilityRisk {
    #[must_use]
    pub const fn uncertainty_scope(&self) -> &'static str {
        "pricing_only_fixed_calibration_seed_grid_bridge_and_bump"
    }
}

impl StochasticDividendContinuousBarrierPlan {
    /// Recalibrate six scenarios per selected original target node using common
    /// calibration and valuation seeds. Selection must be nonempty, unique and
    /// in range. Every shifted node must remain positive, representably changed
    /// and within the original floor/cap. All targets validate before calibration.
    /// Cost is six full calibrations and six valuation paths per node, plus the
    /// base path. Errors for sums include cross-node sampling covariance.
    pub fn evaluate_bucketed_local_volatility_risk(
        &self,
        local_volatility_bump: f64,
        node_indices: &[usize],
    ) -> Result<StochasticDividendContinuousBarrierBucketedLocalVolatilityRisk, MonteCarloError>
    {
        let target = self.original_local_variance_target()?;
        let mut seen = std::collections::BTreeSet::new();
        if node_indices.is_empty()
            || node_indices
                .iter()
                .any(|&i| i >= target.values().len() || !seen.insert(i))
        {
            return Err(invalid("continuous_local_volatility_node_indices").into());
        }
        let bumps = [
            0.5 * local_volatility_bump,
            local_volatility_bump,
            2.0 * local_volatility_bump,
        ];
        let count = node_indices.len();
        let selections = node_indices.iter().copied().map(Some).collect::<Vec<_>>();
        let scenarios = self.local_volatility_scenarios_for_nodes(&bumps, &selections)?;
        let node_width = 1 + 5 * count;
        let BumpStatistics {
            values,
            errors,
            units,
            paths,
        } = self.bump_statistics(node_width + 5, |z, bridge, antithetic, out| {
            let (nodes, sums) = out.split_at_mut(node_width);
            self.local_volatility_sample(&scenarios, &bumps, z, bridge, antithetic, nodes)?;
            for (j, sum) in sums.iter_mut().enumerate() {
                *sum = nodes[1..]
                    .as_chunks::<5>()
                    .0
                    .iter()
                    .map(|row| row[j])
                    .collect::<pricing_numerics::NeumaierSum>()
                    .total();
            }
            Ok(())
        })?;
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
        for &i in node_indices {
            hash.update(&(i as u64).to_le_bytes());
        }
        for h in bumps {
            hash.update(&h.to_bits().to_le_bytes());
        }
        let recalibration_count = scenarios.len();
        let scenario_evaluated_paths = (recalibration_count as u128 + 1) * paths;
        Ok(
            StochasticDividendContinuousBarrierBucketedLocalVolatilityRisk {
                price: StochasticDividendPrice {
                    value: values[0],
                    standard_error: errors[0],
                    independent_sampling_units: units,
                    evaluated_paths: paths,
                    plan_fingerprint: self.plan_fingerprint(),
                    scheme: SCHEME,
                },
                node_indices: node_indices.to_vec(),
                time_nodes: target.time_nodes().to_vec(),
                log_moneyness_nodes: target.log_moneyness_nodes().to_vec(),
                local_volatility_bumps: bumps,
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
            },
        )
    }
}
