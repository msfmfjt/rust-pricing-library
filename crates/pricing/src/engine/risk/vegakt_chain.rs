//! Joint conditional-initial / equation-(11) / equation-(4) / equation-(9) diagnostic.
use super::vegakt_steps::moment;
type Matrix = Vec<Vec<f64>>;
use crate::MonteCarloError;
use crate::core::{FiniteF64, PositiveF64};
use crate::mc::DeterministicExecutor;
use crate::risk::{ChainLayout, ChainRow};
use crate::risk::{
    ReportingIvBasis, RiskConfigError, VEGA_KT_CHAIN_LABEL, VegaKtChainEstimate, VegaKtChainPlan,
    VegaKtChainRowAudit, VegaKtChainSample, VegaKtInitialGammaPlan, VegaKtStepRecoveryPlan,
    VegaKtStepSamplingUnit, integrate_piecewise_linear_local_gamma_transition,
    local_gamma_transition_cells, project_vanilla_hedge_measure_to_reporting_iv,
    recover_equation11_gamma_row,
};
use pricing_numerics::{NeumaierSum, standard_normal_pdf};
use std::sync::Arc;

fn invalid(quantity: &'static str) -> RiskConfigError {
    RiskConfigError::VegaKtChainInput { quantity }
}
fn finite(v: f64) -> Result<f64, RiskConfigError> {
    Ok(FiniteF64::new(v, "vega_kt_chain")?.get())
}
fn sum(v: impl IntoIterator<Item = f64>) -> Result<f64, RiskConfigError> {
    finite(v.into_iter().collect::<NeumaierSum>().total())
}
fn apply(matrix: &[Vec<f64>], values: &[f64]) -> Result<Vec<f64>, RiskConfigError> {
    matrix
        .iter()
        .map(|row| sum(row.iter().zip(values).map(|(a, b)| a * b)))
        .collect()
}
fn bounds(k: &[f64]) -> [f64; 2] {
    [k[0], k[k.len() - 1]]
}
fn consistent(a: f64, b: f64) -> bool {
    a.is_finite() && b.is_finite() && (a - b).abs() <= 1e-10 * a.abs().max(b.abs())
}

impl VegaKtChainPlan {
    /// Compile a diagnostic for spatially flat variance, constant forward and
    /// state-only discounted payoff. Reporting expiries must equal every positive
    /// simulation node. Densities and IV must match the discrete lognormal law.
    /// `subdivisions` is the coarse midpoint count per strike/reporting-knot cell;
    /// the estimate uses twice this count and reports their signed difference.
    pub fn new(
        initial: VegaKtInitialGammaPlan,
        steps: VegaKtStepRecoveryPlan,
        basis: ReportingIvBasis,
        subdivisions: usize,
    ) -> Result<Self, RiskConfigError> {
        if initial.simulation_plan() != steps.simulation_plan()
            || initial.variance_grid() != steps.variance_grid()
        {
            return Err(invalid("initial and positive-time model plans differ"));
        }
        if subdivisions == 0 || subdivisions.checked_mul(2).is_none() {
            return Err(invalid(
                "positive representable quadrature subdivisions required",
            ));
        }
        let times = initial.simulation_plan().time_grid().nodes();
        if basis.maturity_nodes() != &times[1..] {
            return Err(invalid(
                "reporting expiries must equal positive simulation nodes",
            ));
        }
        let spot = initial.simulation_plan().forward_normalizers()[0];
        let width = steps.variance_grid().log_moneyness_nodes().len();
        let bw = basis.log_moneyness_nodes().len();
        let mut cumulative = 0.;
        let mut strikes = vec![initial.strike_nodes().to_vec()];
        for i in 0..times.len() - 1 {
            cumulative += steps.layout.variances[i * width] * (times[i + 1] - times[i]);
            let sigma = (cumulative / times[i + 1]).sqrt();
            if basis.implied_volatilities()[i * bw..(i + 1) * bw]
                .iter()
                .any(|&v| !consistent(v, sigma))
            {
                return Err(invalid(
                    "reporting IV differs from discrete lognormal model",
                ));
            }
            if i < steps.layout.densities.len() {
                let d = &steps.layout.densities[i];
                for (&x, &density) in d.log_moneyness_nodes().iter().zip(d.call_densities()) {
                    let expected = standard_normal_pdf((x + 0.5 * cumulative) / cumulative.sqrt())
                        / (spot * x.exp() * cumulative.sqrt());
                    if !consistent(density, expected) {
                        return Err(invalid(
                            "recovery density differs from discrete lognormal model",
                        ));
                    }
                }
                let row = recover_equation11_gamma_row(
                    d,
                    &vec![0.; width],
                    &steps.layout.variances[(i + 1) * width..(i + 2) * width],
                    times[i + 2] - times[i + 1],
                )?;
                if row.strike_nodes().len() < 2 {
                    return Err(invalid(
                        "at least two active strikes required in every Gamma row",
                    ));
                }
                strikes.push(row.strike_nodes().to_vec());
            }
        }
        let mut rows = Vec::new();
        for (i, k) in strikes.iter().enumerate() {
            let (fine, reporting_edges) =
                projection_matrix(&basis, times[i + 1], spot, k, 2 * subdivisions)?;
            let (coarse, _) = projection_matrix(&basis, times[i + 1], spot, k, subdivisions)?;
            let mut propagation = Vec::new();
            let mut tails = Vec::new();
            if i + 1 < strikes.len() {
                let next = &strikes[i + 1];
                let v = steps.layout.variances[(i + 1) * width] * (times[i + 2] - times[i + 1]);
                for &s in k {
                    let mut row = Vec::new();
                    for j in 0..next.len() {
                        let mut unit = vec![0.; next.len()];
                        unit[j] = 1.;
                        row.push(integrate_piecewise_linear_local_gamma_transition(
                            s, v, next, &unit,
                        )?);
                    }
                    propagation.push(row);
                    let cells = local_gamma_transition_cells(s, v, next)?;
                    let mut tail = vec![0.; next.len()];
                    tail[0] = finite(v.exp() * cells[0].probability())?;
                    tail[next.len() - 1] = finite(v.exp() * cells[cells.len() - 1].probability())?;
                    tails.push(tail);
                }
            }
            let nodal_vegas = basis
                .log_moneyness_nodes()
                .iter()
                .enumerate()
                .map(|(j, &x)| {
                    let sd = basis.implied_volatilities()[i * bw + j] * times[i + 1].sqrt();
                    Ok(PositiveF64::new(
                        spot * standard_normal_pdf(-x / sd + 0.5 * sd) * times[i + 1].sqrt(),
                        "vega_kt_chain_nodal_vega",
                    )?
                    .get())
                })
                .collect::<Result<Vec<_>, RiskConfigError>>()?;
            rows.push(ChainRow {
                audit: VegaKtChainRowAudit {
                    maturity: times[i + 1],
                    strike_bounds: bounds(k),
                    next_gamma_bounds: strikes.get(i + 1).map(|k| bounds(k)),
                    recovery_source_maturity: (i > 0).then(|| times[i]),
                    recovery_density_maturity: (i > 0).then(|| times[i]),
                    excluded_density_probability: (i > 0)
                        .then(|| steps.layout.densities[i - 1].excluded_probability_mass()),
                },
                fine,
                coarse,
                reporting_edges,
                propagation,
                tails,
                nodal_vegas,
            });
        }
        // Four bucket vectors, three totals, N-1 excluded rows, four initial
        // audits, N propagated-tail audits, and 2N reporting-edge audits.
        let components = basis
            .bucket_count()
            .checked_mul(4)
            .and_then(|n| n.checked_add(6))
            .and_then(|n| rows.len().checked_mul(4).and_then(|m| n.checked_add(m)))
            .ok_or(RiskConfigError::VegaKtStepShapeOverflow)?;
        Ok(Self {
            layout: Arc::new(ChainLayout {
                initial,
                steps,
                basis,
                rows,
                subdivisions,
                components,
                integrated_denominators: None,
            }),
        })
    }
    /// Weak Gamma recovery: divide native integrated adjoints by
    /// sigma*dt*E[S_next^2 h_j(log(S_next/S0))]. Equation (3), integrated over
    /// the source state, transports S^2*d to the NEXT state exactly. Replacing
    /// conditional source hats by next-state hats is the temporal approximation.
    /// Integrated masses avoid division by an underresolved point density.
    /// This is not the pointwise equation-(11) formula or exact kernel inversion.
    pub fn with_integrated_density_recovery(mut self) -> Result<Self, RiskConfigError> {
        let l = Arc::make_mut(&mut self.layout);
        let times = l.steps.simulation_plan().time_grid().nodes();
        let width = l.steps.variance_grid().log_moneyness_nodes().len();
        let spot = l.steps.simulation_plan().forward_normalizers()[0];
        let mut cumulative = 0.;
        let mut rows = Vec::new();
        for i in 0..times.len() - 2 {
            cumulative += l.steps.layout.variances[i * width] * (times[i + 1] - times[i]);
            let sigma = l.steps.layout.variances[(i + 1) * width].sqrt();
            let dt = times[i + 2] - times[i + 1];
            rows.push(
                crate::risk::lognormal_spot_squared_hat_masses(
                    spot,
                    cumulative + sigma * sigma * dt,
                    l.steps.variance_grid().log_moneyness_nodes(),
                )?
                .into_iter()
                .map(|m| finite(m * sigma * dt))
                .collect::<Result<Vec<_>, _>>()?,
            );
        }
        for row in l.rows.iter_mut().skip(1) {
            row.audit.recovery_density_maturity = Some(row.audit.maturity);
        }
        l.integrated_denominators = Some(rows);
        Ok(self)
    }

    /// A single independent draw. Full-path AD and conditional initial repricing
    /// share the same shocks and payoff callback. Return discounted price and its
    /// state partials holding variance fixed. Explicit variance/bridge/payoff-policy
    /// dependence is unsupported, not inferred to be zero. Callback correctness
    /// remains the caller's responsibility. RQMC points are not independent draws.
    pub fn sample_from_shocks<F>(
        &self,
        shocks: &[f64],
        payoff: F,
    ) -> Result<VegaKtChainSample, MonteCarloError>
    where
        F: Fn(&[f64]) -> Result<(f64, Vec<f64>), MonteCarloError>,
    {
        let l = &self.layout;
        let path = l.steps.simulation_plan().evolve_path(
            l.steps.variance_grid(),
            l.steps.simulation_plan().forward_normalizers()[0],
            shocks,
        )?;
        let (price, seeds) = payoff(path.states())?;
        finite(price)?;
        let step = l.steps.sample_from_path(&path, &seeds)?;
        let initial = l.initial.sample_from_shocks(&shocks[1..], |states| {
            let (price, seeds) = payoff(states)?;
            if seeds.len() != states.len() {
                return Err(invalid("payoff state partial length").into());
            }
            for v in seeds {
                finite(v)?;
            }
            Ok(price)
        })?;
        let width = l.steps.variance_grid().log_moneyness_nodes().len();
        let times = l.steps.simulation_plan().time_grid().nodes();
        let mut gammas = vec![
            initial.values[..3 * l.initial.strike_nodes().len()]
                .as_chunks::<3>()
                .0
                .iter()
                .map(|v| v[1])
                .collect::<Vec<_>>(),
        ];
        let bump: Vec<_> = initial.values[..3 * l.initial.strike_nodes().len()]
            .as_chunks::<3>()
            .0
            .iter()
            .map(|v| v[2])
            .collect();
        let mut excluded = Vec::new();
        for i in 1..times.len() - 1 {
            if let Some(denominators) = &l.integrated_denominators {
                let active = l.steps.layout.densities[i - 1].active_domain();
                let mut values = Vec::new();
                let mut dropped = Vec::new();
                for (j, &a) in step.integrated_node_adjoints()[i * width..(i + 1) * width]
                    .iter()
                    .enumerate()
                {
                    if active.contains_index(j) {
                        let d = PositiveF64::new(
                            denominators[i - 1][j],
                            "integrated_gamma_denominator",
                        )
                        .map_err(RiskConfigError::from)?
                        .get();
                        values.push(finite(a / d)?);
                    } else {
                        dropped.push(a);
                    }
                }
                excluded.push(sum(dropped)?);
                gammas.push(values);
            } else {
                let gamma = recover_equation11_gamma_row(
                    &l.steps.layout.densities[i - 1],
                    &step.integrated_node_adjoints()[i * width..(i + 1) * width],
                    &l.steps.layout.variances[i * width..(i + 1) * width],
                    times[i + 1] - times[i],
                )?;
                excluded.push(gamma.excluded_local_vega());
                gammas.push(gamma.gamma_values().to_vec());
            }
        }
        let mut raw = Vec::new();
        let mut hedge = Vec::new();
        let mut bump_diff = Vec::new();
        let mut quad_diff = Vec::new();
        let mut tail_iv = Vec::new();
        let mut edge_iv = Vec::new();
        for (i, row) in l.rows.iter().enumerate() {
            let mut density: Vec<_> = gammas[i].iter().map(|v| -v).collect();
            if let Some(next) = gammas.get(i + 1) {
                for (v, p) in density.iter_mut().zip(apply(&row.propagation, next)?) {
                    *v = finite(*v + p)?;
                }
                tail_iv.push(sum(apply(&row.fine, &apply(&row.tails, next)?)?)?);
            } else {
                tail_iv.push(0.);
            }
            edge_iv.extend(apply(&row.reporting_edges, &density)?);
            let fine = apply(&row.fine, &density)?;
            let coarse = apply(&row.coarse, &density)?;
            let bd = if i == 0 {
                apply(&row.fine, &bump.iter().map(|v| -v).collect::<Vec<_>>())?
            } else {
                vec![0.; fine.len()]
            };
            for j in 0..fine.len() {
                raw.push(fine[j]);
                hedge.push(finite(-fine[j] / row.nodal_vegas[j])?);
                bump_diff.push(bd[j]);
                quad_diff.push(finite(fine[j] - coarse[j])?);
            }
        }
        let totals = [
            sum(raw.iter().copied())?,
            sum(bump_diff.iter().copied())?,
            sum(quad_diff.iter().copied())?,
        ];
        let n = 3 * l.initial.strike_nodes().len();
        let native_initial = sum(step.integrated_node_adjoints()[..width].iter().copied())?;
        let mut values = raw;
        values.extend(hedge);
        values.extend(bump_diff);
        values.extend(quad_diff);
        values.extend(totals);
        values.extend(excluded);
        values.extend([
            finite(initial.values[n + 1] - native_initial)?,
            initial.values[n + 2],
            initial.values[n + 3],
            initial.values[n + 4],
        ]);
        values.extend(tail_iv);
        values.extend(edge_iv);
        debug_assert_eq!(values.len(), l.components);
        Ok(VegaKtChainSample {
            layout: Arc::clone(l),
            values: values.into_boxed_slice(),
            sampling_unit: VegaKtStepSamplingUnit::IndependentPath,
        })
    }
    /// Reduce already composed independent draws (or pair means). Scalar and
    /// diagnostic differences retain their joint covariance through sample sums.
    pub fn estimate<F>(
        &self,
        executor: &DeterministicExecutor,
        count: u64,
        sampling_unit: VegaKtStepSamplingUnit,
        evaluate: F,
    ) -> Result<VegaKtChainEstimate, MonteCarloError>
    where
        F: Fn(u64) -> Result<VegaKtChainSample, MonteCarloError> + Send + Sync,
    {
        if count == 0 {
            return Err(RiskConfigError::VegaKtStepNoSamplingUnits.into());
        }
        let stats = executor.try_map_reduce_statistics_vector(
            count,
            self.layout.components,
            |i, out| -> Result<(), MonteCarloError> {
                let sample = evaluate(i)?;
                if !Arc::ptr_eq(&sample.layout, &self.layout) {
                    return Err(invalid("sample origin").into());
                }
                if sample.sampling_unit != sampling_unit {
                    return Err(RiskConfigError::VegaKtStepSamplingUnitMismatch.into());
                }
                out.copy_from_slice(&sample.values);
                Ok(())
            },
        )?;
        let m = stats
            .into_iter()
            .map(|s| moment(s, count))
            .collect::<Result<Vec<_>, _>>()?;
        let b = self.layout.basis.bucket_count();
        let n = self.layout.rows.len();
        let a = 4 * b;
        Ok(VegaKtChainEstimate {
            policy_label: if self.layout.integrated_denominators.is_some() {
                "flat_variance_weak_gamma_chain_diagnostic_v1"
            } else {
                VEGA_KT_CHAIN_LABEL
            },
            reporting_basis: self.layout.basis.clone(),
            independent_sampling_units: count,
            sampling_unit,
            coarse_subdivisions: self.layout.subdivisions,
            rows: self.layout.rows.iter().map(|r| r.audit.clone()).collect(),
            position_iv_buckets: m[..b].to_vec(),
            offsetting_hedge_quantities: m[b..2 * b].to_vec(),
            initial_bump_bucket_differences: m[2 * b..3 * b].to_vec(),
            quadrature_bucket_differences: m[3 * b..4 * b].to_vec(),
            total_position_iv_vega: m[a],
            total_initial_bump_difference: m[a + 1],
            total_quadrature_difference: m[a + 2],
            excluded_local_vegas: m[a + 3..a + n + 2].to_vec(),
            initial_reconstruction_difference: m[a + n + 2],
            initial_bump_vega_difference: m[a + n + 3],
            initial_tail_vegas: [m[a + n + 4], m[a + n + 5]],
            propagated_tail_iv_vegas: m[a + n + 6..a + 2 * n + 6].to_vec(),
            reporting_edge_iv_vegas: m[a + 2 * n + 6..]
                .as_chunks::<2>()
                .0
                .iter()
                .map(|v| [v[0], v[1]])
                .collect(),
        })
    }
}
impl VegaKtChainSample {
    /// Caller must supply opposite shocks for the same draw; average before moments.
    pub fn antithetic_pair(primary: Self, mate: Self) -> Result<Self, RiskConfigError> {
        if !Arc::ptr_eq(&primary.layout, &mate.layout) {
            return Err(invalid("antithetic sample origin"));
        }
        if primary.sampling_unit != VegaKtStepSamplingUnit::IndependentPath
            || mate.sampling_unit != VegaKtStepSamplingUnit::IndependentPath
        {
            return Err(RiskConfigError::VegaKtStepSamplingUnitMismatch);
        }
        let mut values = primary.values;
        for (a, b) in values.iter_mut().zip(mate.values.iter()) {
            *a = finite(0.5 * *a + 0.5 * b)?;
        }
        Ok(Self {
            layout: primary.layout,
            values,
            sampling_unit: VegaKtStepSamplingUnit::AntitheticPair,
        })
    }
}

/// Linear finite-interval projection: split at every Gamma and IV-hat knot,
/// then midpoint-integrate in STRIKE, interpolating hedge density linearly.
fn projection_matrix(
    basis: &ReportingIvBasis,
    time: f64,
    spot: f64,
    k: &[f64],
    q: usize,
) -> Result<(Matrix, Matrix), RiskConfigError> {
    let mut knots = k.to_vec();
    for &x in basis.log_moneyness_nodes() {
        let s = finite(spot * x.exp())?;
        if s > k[0] && s < k[k.len() - 1] {
            knots.push(s);
        }
    }
    knots.sort_by(f64::total_cmp);
    knots.dedup();
    let size = (knots.len() - 1)
        .checked_mul(q)
        .ok_or(RiskConfigError::VegaKtStepShapeOverflow)?;
    // Bound diagnostic compilation memory rather than attempting an unbounded allocation.
    if size > 1_000_000 {
        return Err(invalid("quadrature point budget exceeded"));
    }
    let mut points = Vec::with_capacity(size);
    let mut weights = Vec::with_capacity(size);
    let mut hats = Vec::with_capacity(size);
    for cell in knots.windows(2) {
        let h = (cell[1] - cell[0]) / q as f64;
        for j in 0..q {
            let s = finite(cell[0] + (j as f64 + 0.5) * h)?;
            // Distinct knots computed by exp/log can be a few ULPs apart.
            // A rounded midpoint at an endpoint remains a valid rule for the
            // continuous interpolants; its positive cell measure is retained.
            if s < cell[0] || s > cell[1] || h <= 0. {
                return Err(invalid("unrepresentable midpoint cell"));
            }
            let lower = k
                .partition_point(|&v| v <= s)
                .saturating_sub(1)
                .min(k.len() - 2);
            let w = (s - k[lower]) / (k[lower + 1] - k[lower]);
            points.push(s);
            weights.push(h);
            hats.push((lower, w));
        }
    }
    let b = basis.log_moneyness_nodes().len();
    let t = basis
        .maturity_nodes()
        .iter()
        .position(|&v| v == time)
        .expect("validated expiry");
    let mut matrix = vec![vec![0.; k.len()]; b];
    let mut edges = vec![vec![0.; k.len()]; 2];
    for j in 0..k.len() {
        let measure: Vec<_> = hats
            .iter()
            .zip(&weights)
            .map(|(&(i, w), &h)| {
                if j == i {
                    h * (1. - w)
                } else if j == i + 1 {
                    h * w
                } else {
                    0.
                }
            })
            .collect();
        let p =
            project_vanilla_hedge_measure_to_reporting_iv(basis, time, spot, &points, &measure)?;
        let stats = p.position_risk().reporting_stats();
        edges[0][j] = stats.left_edge_sensitivity();
        edges[1][j] = stats.right_edge_sensitivity();
        for (r, m) in matrix.iter_mut().enumerate() {
            m[j] = p.position_risk().raw_buckets()[t * b + r];
        }
    }
    Ok((matrix, edges))
}
