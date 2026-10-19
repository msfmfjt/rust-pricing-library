//! Audit the provisional local-node allocation; this is not paper VegaKT.

use pricing_numerics::NeumaierSum;

use crate::MonteCarloError;
use crate::engine::plan::simulation::{LocalVolPathwise, LocalVolRuntime};
use crate::risk::{ReportingIvProjectionStats, project_local_vega_nodes_to_reporting_iv};

pub(in crate::engine) const LOCAL_NODE_ALLOCATION_LABEL: &str =
    "local_vol_node_allocation_diagnostic_v1";

#[derive(Clone, Copy, Debug, Default)]
pub(in crate::engine) struct LocalVolAllocationAudit {
    /// Independently summed atom IV risk, used only by the terminal-measure policy.
    pub measure_pre_projection: f64,
    pub initial_time: f64,
    pub density_excluded: f64,
    pub reporting_stats: ReportingIvProjectionStats,
}

/// Average sensitivities but count all evaluated path/row edge assignments.
/// The same reduction applies to antithetic pairs and RQMC scramble means.
#[derive(Default)]
pub(in crate::engine) struct LocalVolAllocationAuditSum {
    measure_pre_projection: NeumaierSum,
    initial_time: NeumaierSum,
    density_excluded: NeumaierSum,
    left_edge: NeumaierSum,
    right_edge: NeumaierSum,
    left_count: u64,
    right_count: u64,
}

impl LocalVolAllocationAuditSum {
    pub fn add(&mut self, audit: LocalVolAllocationAudit) {
        self.measure_pre_projection
            .add(audit.measure_pre_projection);
        self.initial_time.add(audit.initial_time);
        self.density_excluded.add(audit.density_excluded);
        self.left_edge
            .add(audit.reporting_stats.left_edge_sensitivity);
        self.right_edge
            .add(audit.reporting_stats.right_edge_sensitivity);
        self.left_count = self
            .left_count
            .saturating_add(audit.reporting_stats.left_edge_count);
        self.right_count = self
            .right_count
            .saturating_add(audit.reporting_stats.right_edge_count);
    }

    pub fn mean(self, count: u64) -> LocalVolAllocationAudit {
        assert!(count > 0, "allocation audit reduction must be nonempty");
        LocalVolAllocationAudit {
            measure_pre_projection: self.measure_pre_projection.total() / count as f64,
            initial_time: self.initial_time.total() / count as f64,
            density_excluded: self.density_excluded.total() / count as f64,
            reporting_stats: ReportingIvProjectionStats {
                left_edge_count: self.left_count,
                right_edge_count: self.right_count,
                left_edge_sensitivity: self.left_edge.total() / count as f64,
                right_edge_sensitivity: self.right_edge.total() / count as f64,
            },
        }
    }
}

/// Reduce paired path sensitivities and their audit in the risk layer.
pub(in crate::engine) fn average_local_vol_pathwise(
    primary: LocalVolPathwise,
    mate: LocalVolPathwise,
) -> Result<LocalVolPathwise, MonteCarloError> {
    let values =
        std::array::from_fn(|component| (primary.values[component] + mate.values[component]) * 0.5);
    let raw_buckets = match (primary.raw_buckets, mate.raw_buckets) {
        (Some(primary), Some(mate)) => {
            if primary.len() != mate.len() {
                return Err(MonteCarloError::MismatchedLocalVolatilityReportingBasis);
            }
            Some(
                primary
                    .into_iter()
                    .zip(mate)
                    .map(|(left, right)| (left + right) * 0.5)
                    .collect(),
            )
        }
        (None, None) => None,
        _ => return Err(MonteCarloError::MismatchedLocalVolatilityReportingBasis),
    };
    let allocation_audit = match (primary.allocation_audit, mate.allocation_audit) {
        (Some(left), Some(right)) => {
            let mut sum = LocalVolAllocationAuditSum::default();
            sum.add(left);
            sum.add(right);
            Some(sum.mean(2))
        }
        (None, None) => None,
        _ => return Err(MonteCarloError::MismatchedLocalVolatilityReportingBasis),
    };
    Ok(LocalVolPathwise {
        values,
        raw_buckets,
        allocation_audit,
    })
}

pub(in crate::engine) fn allocate_local_vol_nodes(
    runtime: &LocalVolRuntime,
    adjoints: &[f64],
) -> Result<(Vec<f64>, LocalVolAllocationAudit), MonteCarloError> {
    let kt = runtime
        .vega_kt
        .as_ref()
        .ok_or(MonteCarloError::MissingLocalVolatilityReportingBasis)?;
    if adjoints.len() != runtime.grid.values().len() {
        return Err(MonteCarloError::MismatchedLocalVolatilityReportingBasis);
    }
    let x = runtime.grid.log_moneyness_nodes();
    let mut buckets = vec![NeumaierSum::new(); kt.basis.bucket_count()];
    let mut audit = LocalVolAllocationAuditSum::default();
    for (time, row) in runtime
        .grid
        .time_nodes()
        .iter()
        .copied()
        .zip(adjoints.chunks_exact(x.len()))
    {
        if time == 0.0 {
            audit.add(LocalVolAllocationAudit {
                initial_time: row.iter().copied().collect::<NeumaierSum>().total(),
                ..Default::default()
            });
            continue;
        }
        let density = kt
            .density_rows
            .iter()
            .find(|row| row.maturity().get().to_bits() == time.to_bits())
            .ok_or(MonteCarloError::MismatchedLocalVolatilityReportingBasis)?;
        // The routine consumes integrated node adjoints, never A_i / hat_area_i.
        let projection = project_local_vega_nodes_to_reporting_iv(
            &kt.basis,
            time,
            x,
            row,
            density.active_domain(),
        )?;
        for (bucket, contribution) in buckets.iter_mut().zip(projection.raw_buckets()) {
            bucket.add(*contribution);
        }
        audit.add(LocalVolAllocationAudit {
            density_excluded: projection.signed_residual(),
            reporting_stats: projection.reporting_stats(),
            ..Default::default()
        });
    }
    // Sum over maturities; the path/sample reductions happen in the caller.
    Ok((
        buckets.into_iter().map(NeumaierSum::total).collect(),
        audit.mean(1),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audit_reduction_averages_sensitivities_but_counts_actual_paths() {
        let mut pair = LocalVolAllocationAuditSum::default();
        for sign in [-1.0, 2.0] {
            pair.add(LocalVolAllocationAudit {
                measure_pre_projection: 7.0 * sign,
                initial_time: 4.0 * sign,
                density_excluded: -2.0 * sign,
                reporting_stats: ReportingIvProjectionStats {
                    left_edge_count: 1,
                    right_edge_count: 2,
                    left_edge_sensitivity: 6.0 * sign,
                    right_edge_sensitivity: -8.0 * sign,
                },
            });
        }
        let pair_mean = pair.mean(2);
        assert_eq!(pair_mean.initial_time, 2.0);
        assert_eq!(pair_mean.density_excluded, -1.0);
        assert_eq!(pair_mean.reporting_stats.left_edge_count, 2);
        assert_eq!(pair_mean.reporting_stats.right_edge_count, 4);
        assert_eq!(pair_mean.reporting_stats.left_edge_sensitivity, 3.0);
        assert_eq!(pair_mean.reporting_stats.right_edge_sensitivity, -4.0);
        let mut scramble = LocalVolAllocationAuditSum::default();
        scramble.add(pair_mean);
        scramble.add(pair_mean);
        let mean = scramble.mean(2);
        assert_eq!(mean.initial_time, pair_mean.initial_time);
        assert_eq!(mean.reporting_stats.left_edge_count, 4);
        assert_eq!(mean.reporting_stats.right_edge_count, 8);
    }
}
