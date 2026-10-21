//! Copy retained native diagnostics. No calibration, pricing, or acceptance policy.
use crate::bass_lv::{BassCalibrationDiagnostics, BassSurfaceDiagnostics};
use crate::mc::hull_white::HullWhiteCalibrationDiagnostics;
use crate::mc::lsv::LsvCalibrationRowDiagnostics;
use crate::multi_asset::LocalCorrelationCalibration;
use crate::{
    CalibrationKind as K, CalibrationMetric, DiagnosticUnit as U, ReportAxis, ReportedCalibration,
};

type Column<T> = (&'static str, U, &'static str, fn(&T) -> f64);

pub(super) fn multi_asset(
    plan: &crate::multi_asset::MultiAssetPricingPlan,
    source: &crate::MultiAssetSpecification,
) -> Vec<ReportedCalibration> {
    use crate::multi_asset::MultiAssetBergomiLsvConfig as Config;
    let one = plan.lsv_calibrations();
    let two = plan.lsv_two_factor_calibrations();
    let hw = plan.hull_white_lsv_calibrations();
    let ids: Vec<_> = source
        .markets()
        .iter()
        .map(|m| m.forward().underlying().get())
        .collect();
    let mut blocks = Vec::new();
    for (i, &id) in ids.iter().enumerate() {
        if let Some(c) = hw[i] {
            let seed = match source.model().lsv[i].as_ref().expect("HW LSV config") {
                Config::OneFactor(c) => c.particles.seed(),
                Config::TwoFactor(c) => c.particles.seed(),
                Config::Rough(c) => c.particles.seed(),
            };
            blocks.push(hull_white(id, seed, &c.diagnostics));
        } else if let Some(c) = one[i] {
            blocks.push(lsv(id, c.config().seed(), c.diagnostics()));
        } else if let Some(c) = two[i] {
            blocks.push(lsv(id, c.config().seed(), c.diagnostics()));
        }
    }
    if let Some(c) = plan.local_correlation_calibration() {
        // The joint simulator uses both marginal surfaces and basket calibration.
        // Keep their asset IDs, calibration seeds and ESS populations separate.
        blocks.push(local_correlation(&ids, c));
    }
    blocks
}

fn table<T>(
    kind: K,
    ids: &[u32],
    seed: Option<u64>,
    axes: Vec<ReportAxis>,
    rows: &[T],
    columns: &[Column<T>],
) -> ReportedCalibration {
    ReportedCalibration {
        kind,
        underlying_ids: ids.to_vec(),
        seed,
        axes,
        metrics: columns
            .iter()
            .map(|&(name, unit, convention, get)| CalibrationMetric {
                name: name.into(),
                values: rows.iter().map(get).collect(),
                unit,
                convention: convention.into(),
            })
            .collect(),
    }
}
fn axis(name: &str, values: Vec<f64>) -> ReportAxis {
    ReportAxis {
        name: name.into(),
        values,
    }
}

pub(crate) fn lsv(
    id: u32,
    seed: u64,
    rows: &[LsvCalibrationRowDiagnostics],
) -> ReportedCalibration {
    table(
        K::ParticleLsv,
        &[id],
        Some(seed),
        vec![axis("time", rows.iter().map(|r| r.time).collect())],
        rows,
        &[
            (
                "minimum_effective_samples",
                U::Dimensionless,
                "Native minimum kernel ESS over supported donor nodes; excludes unsupported nodes before fallback.",
                |r| r.minimum_effective_samples,
            ),
            (
                "extrapolated_nodes",
                U::Count,
                "Native number of tail or interior nodes assigned a supported donor in each calibration row.",
                |r| r.extrapolated_nodes as f64,
            ),
            (
                "particle_mean_f",
                U::EquityCoordinate,
                "Native mean of the calibrated equity coordinate, including residual-equity coordinates for stochastic dividends; not a pricing estimate.",
                |r| r.particle_mean_f,
            ),
        ],
    )
}

pub(crate) fn hull_white(
    id: u32,
    seed: u64,
    rows: &[HullWhiteCalibrationDiagnostics],
) -> ReportedCalibration {
    table(
        K::HullWhiteLsv,
        &[id],
        Some(seed),
        vec![axis("time", rows.iter().map(|r| r.time).collect())],
        rows,
        &[
            (
                "minimum_effective_samples",
                U::Dimensionless,
                "Native minimum discounted-kernel ESS, with the native analytic initial-row convention.",
                |r| r.minimum_effective_samples,
            ),
            (
                "fallback_nodes",
                U::Count,
                "Native number of nodes using fallback in each calibration row.",
                |r| r.fallback_nodes as f64,
            ),
            (
                "mean_relative_discount",
                U::Dimensionless,
                "Native mean of the stochastic discount factor relative to the initial discount curve.",
                |r| r.mean_relative_discount,
            ),
            (
                "mean_discounted_normalized_equity",
                U::EquityCoordinate,
                "Native mean of relative discount times normalized equity; retains the native equity scale.",
                |r| r.mean_discounted_normalized_equity,
            ),
            (
                "maximum_rate_correction",
                U::Variance,
                "Native maximum absolute rate correction across each row's target nodes.",
                |r| r.maximum_rate_correction,
            ),
        ],
    )
}

pub(crate) fn local_correlation(
    ids: &[u32],
    c: &LocalCorrelationCalibration,
) -> ReportedCalibration {
    table(
        K::LocalCorrelation,
        ids,
        Some(c.config().particles.seed()),
        vec![
            axis("time", c.time_nodes().to_vec()),
            axis("log_basket", c.log_nodes().to_vec()),
        ],
        c.diagnostics(),
        &[
            (
                "endpoint_variance_0",
                U::Variance,
                "Native conditional variance at the first correlation endpoint, before rate correction.",
                |r| r.endpoint_variances[0],
            ),
            (
                "endpoint_variance_1",
                U::Variance,
                "Native conditional variance at the second correlation endpoint, before rate correction.",
                |r| r.endpoint_variances[1],
            ),
            (
                "target_variance",
                U::Variance,
                "Native effective relative basket target variance.",
                |r| r.target_variance,
            ),
            (
                "attained_variance",
                U::Variance,
                "Native attained basket variance after projection, including rate correction.",
                |r| r.attained_variance,
            ),
            (
                "rate_correction",
                U::Variance,
                "Native stochastic-rate correction included in attained variance.",
                |r| r.rate_correction,
            ),
            (
                "raw_mixing",
                U::Dimensionless,
                "Native unprojected correlation mixing coefficient; may be outside [0,1].",
                |r| r.raw_mixing,
            ),
            (
                "effective_samples",
                U::Dimensionless,
                "Native kernel ESS at the original node before donor fallback; not comparable to donor-only LSV minima.",
                |r| r.effective_samples,
            ),
            (
                "source_node",
                U::Count,
                "Zero-based donor index on this row's log_basket axis.",
                |r| r.source_node as f64,
            ),
            (
                "fallback",
                U::Indicator,
                "One when the native node uses a different donor, zero otherwise.",
                |r| f64::from(r.fallback),
            ),
            (
                "projected",
                U::Indicator,
                "One when native feasibility handling projected the mixing coefficient, zero otherwise.",
                |r| f64::from(r.projected),
            ),
            (
                "unidentifiable",
                U::Indicator,
                "One when the native endpoint variance span is unidentifiable, zero otherwise.",
                |r| f64::from(r.unidentifiable),
            ),
        ],
    )
}

pub(crate) fn bass(id: u32, rows: &[BassCalibrationDiagnostics]) -> ReportedCalibration {
    table(
        K::BassFixedPoint,
        &[id],
        None,
        vec![axis("end_time", rows.iter().map(|r| r.end_time).collect())],
        rows,
        &[
            (
                "start_time",
                U::Time,
                "Native interval start time in years.",
                |r| r.start_time,
            ),
            (
                "iterations",
                U::Count,
                "Native fixed-point iteration count per interval.",
                |r| r.iterations as f64,
            ),
            (
                "cdf_residual",
                U::Dimensionless,
                "Native unshifted infinity norm of A F - F on the Brownian grid.",
                |r| r.cdf_residual,
            ),
            (
                "marginal_cdf_error",
                U::Dimensionless,
                "Native propagated marginal CDF error at input knots and nine quantiles; includes interpolation error.",
                |r| r.marginal_cdf_error,
            ),
            (
                "boundary_tail_probability",
                U::Dimensionless,
                "Native Brownian-grid boundary tail probability.",
                |r| r.boundary_tail_probability,
            ),
            (
                "brownian_min",
                U::BrownianCoordinate,
                "Native lower Brownian-grid boundary.",
                |r| r.brownian_min,
            ),
            (
                "brownian_max",
                U::BrownianCoordinate,
                "Native upper Brownian-grid boundary.",
                |r| r.brownian_max,
            ),
            (
                "grid_spacing",
                U::BrownianCoordinate,
                "Native Brownian-grid spacing.",
                |r| r.grid_spacing,
            ),
        ],
    )
}

pub(crate) fn bass_projection(id: u32, rows: &[BassSurfaceDiagnostics]) -> ReportedCalibration {
    table(
        K::BassSurfaceProjection,
        &[id],
        None,
        vec![axis(
            "marginal_index",
            (0..rows.len()).map(|i| i as f64).collect(),
        )],
        rows,
        &[
            (
                "lower_tail_probability",
                U::Dimensionless,
                "Native lower truncated tail probability.",
                |r| r.lower_tail_probability,
            ),
            (
                "upper_tail_probability",
                U::Dimensionless,
                "Native upper truncated tail probability.",
                |r| r.upper_tail_probability,
            ),
            (
                "retained_probability",
                U::Dimensionless,
                "Native probability retained by the projected marginal.",
                |r| r.retained_probability,
            ),
            (
                "unscaled_mean",
                U::EquityCoordinate,
                "Native marginal mean before mean normalization.",
                |r| r.unscaled_mean,
            ),
            (
                "mean_scale",
                U::Dimensionless,
                "Native multiplicative mean normalization.",
                |r| r.mean_scale,
            ),
            (
                "max_call_price_error",
                U::EquityCoordinate,
                "Native maximum absolute source call error at supplied nodes in the martingale quote coordinate.",
                |r| r.max_call_price_error,
            ),
            (
                "retained_nodes",
                U::Count,
                "Native number of retained marginal nodes.",
                |r| r.retained_nodes as f64,
            ),
        ],
    )
}
