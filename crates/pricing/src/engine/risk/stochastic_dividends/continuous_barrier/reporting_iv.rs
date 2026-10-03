//! Reporting-only projection of recalibrated finite Local-volatility node risks.
//! This does not bump quoted IVs or differentiate a surface/Dupire calibration.
use super::sampling::BumpStatistics;
use super::*;
use crate::engine::plan::simulation::ReportingIvSurface;
use crate::risk::{
    AnalyticCallDensityRow, ReportingIvBasis, analytic_call_density_row_from_surface,
    local_vega_density_from_node_adjoints, project_local_vega_nodes_to_reporting_iv,
};
use pricing_numerics::NeumaierSum;

const METHOD: &str = "buehler-rough-residual-lsv-continuous-bridge-reporting-iv-projection-crn-v1";
const POLICY: &str = "local_vega_density_reporting_iv_projection_v1";

/// The existing residual-LSV density/reporting-basis map applied to finite node
/// price differences. It is a reporting approximation, not quoted-IV bump Vega
/// or the full equation-(3)-(11) VegaKT operator. Rows are reporting buckets in
/// maturity-major order; columns are half/base/double Local-volatility bumps.
#[derive(Clone, Debug, PartialEq)]
pub struct StochasticDividendContinuousBarrierReportingIvRisk {
    pub price: StochasticDividendPrice,
    pub local_volatility_bumps: [f64; 3],
    pub reporting_maturity_nodes: Vec<f64>,
    pub reporting_log_moneyness_nodes: Vec<f64>,
    pub reporting_implied_volatilities: Vec<f64>,
    pub bucket_estimates: Vec<[f64; 3]>,
    pub bucket_standard_errors: Vec<[f64; 3]>,
    pub bump_differences: Vec<[f64; 2]>,
    pub bump_difference_standard_errors: Vec<[f64; 2]>,
    /// Sum of original finite node risks, before density normalization/projection.
    pub pre_projection_estimates: [f64; 3],
    pub pre_projection_standard_errors: [f64; 3],
    pub projected_sum_estimates: [f64; 3],
    pub projected_sum_standard_errors: [f64; 3],
    /// Pre-projection minus projected sum, including time zero, excluded nodes
    /// and the change from node sensitivities to density weights.
    pub residual_estimates: [f64; 3],
    pub residual_standard_errors: [f64; 3],
    /// Density diagnostics on each strictly positive ORIGINAL target time/row.
    pub positive_target_time_nodes: Vec<f64>,
    pub target_log_moneyness_nodes: Vec<f64>,
    pub active_domain_start_indices: Vec<usize>,
    pub active_domain_end_indices: Vec<usize>,
    pub excluded_probability_masses: Vec<f64>,
    pub relative_density_threshold: f64,
    pub scenario_evaluated_paths: u128,
    pub payoff_evaluations: u128,
    pub recalibration_count: usize,
    pub risk_fingerprint: Fingerprint,
    pub method: &'static str,
}
impl StochasticDividendContinuousBarrierReportingIvRisk {
    #[must_use]
    pub const fn projection_policy(&self) -> &'static str {
        POLICY
    }
    #[must_use]
    pub const fn uncertainty_scope(&self) -> &'static str {
        "pricing_only_fixed_calibration_seed_grid_bridge_bump_and_projection"
    }
}

pub(super) struct ReportingProjection {
    pub basis: ReportingIvBasis,
    pub density_rows: Vec<AnalyticCallDensityRow>,
    // Each original node's contribution to all reporting buckets.
    pub node_weights: Vec<Vec<f64>>,
}
impl StochasticDividendContinuousBarrierPlan {
    /// Project all original finite Local-volatility node risks onto the retained
    /// reporting-IV basis. The price request must contain a reporting basis, but
    /// generic Greek flags remain unsupported. Threshold is explicit in (0,1].
    /// Six recalibrations per original node; no surface rebootstrap/IV bump occurs.
    /// The map and active domains stay fixed across the Local-volatility ladder.
    pub fn evaluate_reporting_iv_projection(
        &self,
        local_volatility_bump: f64,
        relative_density_threshold: f64,
    ) -> Result<StochasticDividendContinuousBarrierReportingIvRisk, MonteCarloError> {
        // Validate the entire map before any scenario calibration or valuation.
        let projection = self.reporting_projection(relative_density_threshold)?;
        let bumps = [
            0.5 * local_volatility_bump,
            local_volatility_bump,
            2.0 * local_volatility_bump,
        ];
        let node_count = projection.node_weights.len();
        let selections = (0..node_count).map(Some).collect::<Vec<_>>();
        let scenarios = self.local_volatility_scenarios_for_nodes(&bumps, &selections)?;
        let bucket_count = projection.basis.bucket_count();
        let aggregate_offset = 1 + 5 * bucket_count;
        let BumpStatistics {
            values,
            errors,
            units,
            paths,
        } = self.bump_statistics(aggregate_offset + 9, |z, bridge, antithetic, out| {
            let mut nodes = vec![0.0; 1 + 5 * node_count];
            self.local_volatility_sample(&scenarios, &bumps, z, bridge, antithetic, &mut nodes)?;
            projection.apply(&nodes, out);
            Ok(())
        })?;
        let triples = |v: &[f64]| {
            (0..bucket_count)
                .map(|b| [v[1 + 5 * b], v[2 + 5 * b], v[3 + 5 * b]])
                .collect()
        };
        let gaps = |v: &[f64]| {
            (0..bucket_count)
                .map(|b| [v[4 + 5 * b], v[5 + 5 * b]])
                .collect()
        };
        let aggregate = |v: &[f64], i: usize| -> [f64; 3] {
            std::array::from_fn(|j| v[aggregate_offset + 3 * i + j])
        };
        let mut hash = blake3::Hasher::new();
        hash.update(METHOD.as_bytes());
        hash.update(POLICY.as_bytes());
        hash.update(self.plan_fingerprint().as_bytes());
        hash.update(&relative_density_threshold.to_bits().to_le_bytes());
        for h in bumps {
            hash.update(&h.to_bits().to_le_bytes());
        }
        let recalibration_count = scenarios.len();
        let scenario_evaluated_paths = (recalibration_count as u128 + 1) * paths;
        Ok(StochasticDividendContinuousBarrierReportingIvRisk {
            price: StochasticDividendPrice {
                value: values[0],
                standard_error: errors[0],
                independent_sampling_units: units,
                evaluated_paths: paths,
                plan_fingerprint: self.plan_fingerprint(),
                scheme: SCHEME,
            },
            local_volatility_bumps: bumps,
            reporting_maturity_nodes: projection.basis.maturity_nodes().to_vec(),
            reporting_log_moneyness_nodes: projection.basis.log_moneyness_nodes().to_vec(),
            reporting_implied_volatilities: projection.basis.implied_volatilities().to_vec(),
            bucket_estimates: triples(&values),
            bucket_standard_errors: triples(&errors),
            bump_differences: gaps(&values),
            bump_difference_standard_errors: gaps(&errors),
            pre_projection_estimates: aggregate(&values, 0),
            pre_projection_standard_errors: aggregate(&errors, 0),
            projected_sum_estimates: aggregate(&values, 1),
            projected_sum_standard_errors: aggregate(&errors, 1),
            residual_estimates: aggregate(&values, 2),
            residual_standard_errors: aggregate(&errors, 2),
            positive_target_time_nodes: projection
                .density_rows
                .iter()
                .map(|r| r.maturity().get())
                .collect(),
            active_domain_start_indices: projection
                .density_rows
                .iter()
                .map(|r| r.active_domain().start_index())
                .collect(),
            active_domain_end_indices: projection
                .density_rows
                .iter()
                .map(|r| r.active_domain().end_index())
                .collect(),
            excluded_probability_masses: projection
                .density_rows
                .iter()
                .map(|r| r.excluded_probability_mass())
                .collect(),
            target_log_moneyness_nodes: self
                .original_local_variance_target()?
                .log_moneyness_nodes()
                .to_vec(),
            relative_density_threshold,
            scenario_evaluated_paths,
            payoff_evaluations: scenario_evaluated_paths,
            recalibration_count,
            risk_fingerprint: Fingerprint::from_bytes(*hash.finalize().as_bytes()),
            method: METHOD,
        })
    }

    pub(super) fn reporting_projection(
        &self,
        threshold: f64,
    ) -> Result<ReportingProjection, MonteCarloError> {
        let retained = self
            .reporting_iv_basis
            .as_ref()
            .ok_or(MonteCarloError::MissingLocalVolatilityReportingBasis)?;
        let basis = ReportingIvBasis::new(
            retained.maturity_nodes().to_vec(),
            retained.log_forward_moneyness_nodes().to_vec(),
            retained.implied_volatilities().to_vec(),
        )?;
        let surface = ReportingIvSurface::new(retained);
        let target = self.original_local_variance_target()?;
        let x = target.log_moneyness_nodes();
        let mut node_weights = vec![vec![0.0; basis.bucket_count()]; target.values().len()];
        let mut density_rows = Vec::new();
        for (t, &maturity) in target.time_nodes().iter().enumerate() {
            if maturity == 0.0 {
                continue;
            }
            // Normalized residual forward: density ratios and excluded mass are
            // scale invariant; there is no physical-Spot smile remapping here.
            let row = analytic_call_density_row_from_surface(
                &surface,
                maturity,
                1.0,
                x.to_vec(),
                threshold,
            )?;
            for i in 0..x.len() {
                let mut unit = vec![0.0; x.len()];
                unit[i] = 1.0;
                let density = local_vega_density_from_node_adjoints(&unit, x)?;
                let projected = project_local_vega_nodes_to_reporting_iv(
                    &basis,
                    maturity,
                    x,
                    &density,
                    row.active_domain(),
                )?;
                node_weights[t * x.len() + i] = projected.raw_buckets().to_vec();
            }
            density_rows.push(row);
        }
        Ok(ReportingProjection {
            basis,
            density_rows,
            node_weights,
        })
    }
}
impl ReportingProjection {
    pub fn apply(&self, nodes: &[f64], out: &mut [f64]) {
        out.fill(0.0);
        out[0] = nodes[0];
        let buckets = self.basis.bucket_count();
        for b in 0..buckets {
            for j in 0..3 {
                out[1 + 5 * b + j] = self
                    .node_weights
                    .iter()
                    .enumerate()
                    .map(|(i, w)| w[b] * nodes[1 + 5 * i + j])
                    .collect::<NeumaierSum>()
                    .total();
            }
            out[4 + 5 * b] = out[1 + 5 * b] - out[2 + 5 * b];
            out[5 + 5 * b] = out[2 + 5 * b] - out[3 + 5 * b];
        }
        let a = 1 + 5 * buckets;
        for j in 0..3 {
            out[a + j] = nodes[1..]
                .as_chunks::<5>()
                .0
                .iter()
                .map(|r| r[j])
                .collect::<NeumaierSum>()
                .total();
            out[a + 3 + j] = (0..buckets)
                .map(|b| out[1 + 5 * b + j])
                .collect::<NeumaierSum>()
                .total();
            out[a + 6 + j] = out[a + j] - out[a + 3 + j];
        }
    }
}
