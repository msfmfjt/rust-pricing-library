//! Same-path terminal-measure IV risk, with ordinary paired/scramble statistics.
use super::local_vol_allocation::LocalVolAllocationAudit;
use super::vegakt_measure::{invalid, log_ratio};
use crate::MonteCarloError;
use crate::core::{FiniteF64, PositiveF64};
use crate::engine::plan::simulation::SimulationPlan;
use crate::risk::{ReportingIvProjectionStats, RiskConfigError};
use pricing_numerics::NeumaierSum;

pub(in crate::engine) const TERMINAL_PATHWISE_LABEL: &str =
    "flat_variance_terminal_measure_pathwise_v1";

#[derive(Clone, Debug)]
struct AtomProjection {
    strike: f64,
    mass: f64,
    weights: Vec<(usize, f64)>,
    left: bool,
    right: bool,
}

#[derive(Clone, Debug)]
pub(in crate::engine) struct TerminalVegaKtPathwise {
    atoms: Vec<AtomProjection>,
    bucket_count: usize,
    shock_weights: Vec<f64>,
    drift_derivative: f64,
}

impl SimulationPlan {
    pub(in crate::engine) fn compile_terminal_vega_kt(
        &self,
    ) -> Result<Option<TerminalVegaKtPathwise>, MonteCarloError> {
        let Some(model) = self.terminal_measure_model()? else {
            return Ok(None);
        };
        let lv = self.local_volatility.as_ref().expect("validated model");
        let kt = lv.vega_kt.as_ref().expect("validated reporting basis");
        let times = lv.plan.time_grid().nodes();
        let total = *model.cumulative.last().expect("positive maturity");
        let sigma = PositiveF64::new((total / self.time).sqrt(), "terminal_pathwise_iv")
            .map_err(RiskConfigError::from)?
            .get();
        let shock_weights = model
            .variances
            .iter()
            .zip(times.windows(2))
            .map(|(&v, t)| (v * (t[1] - t[0])).sqrt() / sigma)
            .collect();
        let mut atoms = Vec::new();
        for atom in model.payoff.gamma().atoms() {
            let mut weights = vec![0.; kt.basis.bucket_count()];
            let mut stats = ReportingIvProjectionStats::default();
            kt.basis.project_weight(
                self.time,
                log_ratio(atom.strike(), self.spot),
                1.,
                &mut weights,
                &mut stats,
            )?;
            atoms.push(AtomProjection {
                strike: atom.strike(),
                mass: atom.mass(),
                weights: weights
                    .into_iter()
                    .enumerate()
                    .filter(|(_, w)| *w != 0.)
                    .collect(),
                left: stats.left_edge_count != 0,
                right: stats.right_edge_count != 0,
            });
        }
        Ok(Some(TerminalVegaKtPathwise {
            atoms,
            bucket_count: kt.basis.bucket_count(),
            shock_weights,
            drift_derivative: -total / sigma,
        }))
    }
}

impl TerminalVegaKtPathwise {
    /// Scale all interval vols by sigma_IV'/sigma_IV while keeping the same
    /// native Gaussian increments. d log(S_T)/d sigma_IV =
    /// -V/sigma_IV + sum sigma_i*sqrt(dt_i)*Z_i/sigma_IV.
    /// Each signed terminal call mass contributes q*1(S_T>K)*dS_T/dsigma_IV.
    /// The affine payoff remainder has exactly zero IV risk under zero carry.
    /// These are actual per-path samples, not rescaled means or residual fits.
    pub(in crate::engine) fn sample(
        &self,
        terminal_spot: f64,
        shocks: &[f64],
    ) -> Result<(Vec<f64>, LocalVolAllocationAudit), RiskConfigError> {
        PositiveF64::new(terminal_spot, "terminal_pathwise_spot")?;
        if shocks.len() != self.shock_weights.len() {
            return Err(invalid("terminal pathwise shock count mismatch"));
        }
        let mut derivative = NeumaierSum::new();
        derivative.add(self.drift_derivative);
        for (&w, &z) in self.shock_weights.iter().zip(shocks) {
            derivative.add(FiniteF64::new(w * z, "terminal_pathwise_shock_derivative")?.get());
        }
        let ds = FiniteF64::new(
            terminal_spot * derivative.total(),
            "terminal_pathwise_spot_derivative",
        )?
        .get();
        let mut buckets = vec![NeumaierSum::new(); self.bucket_count];
        let mut total = NeumaierSum::new();
        let mut left = NeumaierSum::new();
        let mut right = NeumaierSum::new();
        let mut audit = LocalVolAllocationAudit::default();
        for atom in &self.atoms {
            let active = if terminal_spot > atom.strike {
                1.
            } else if terminal_spot == atom.strike {
                0.5
            } else {
                0.
            };
            let risk =
                FiniteF64::new(atom.mass * active * ds, "terminal_pathwise_atom_risk")?.get();
            total.add(risk);
            for &(j, w) in &atom.weights {
                buckets[j].add(w * risk);
            }
            if atom.left {
                audit.reporting_stats.left_edge_count += 1;
                left.add(risk);
            }
            if atom.right {
                audit.reporting_stats.right_edge_count += 1;
                right.add(risk);
            }
        }
        audit.measure_pre_projection =
            FiniteF64::new(total.total(), "terminal_pathwise_pre_projection")?.get();
        audit.reporting_stats.left_edge_sensitivity = left.total();
        audit.reporting_stats.right_edge_sensitivity = right.total();
        let buckets = buckets
            .into_iter()
            .map(|s| Ok(FiniteF64::new(s.total(), "terminal_pathwise_bucket")?.get()))
            .collect::<Result<Vec<_>, RiskConfigError>>()?;
        Ok((buckets, audit))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signed_spread_samples_match_payoff_bumps_and_keep_independent_total() {
        let plan = TerminalVegaKtPathwise {
            atoms: vec![
                AtomProjection {
                    strike: 95.,
                    mass: 2.,
                    weights: vec![(0, 0.3), (1, 0.7)],
                    left: false,
                    right: false,
                },
                AtomProjection {
                    strike: 105.,
                    mass: -1.5,
                    weights: vec![(2, 1.)],
                    left: false,
                    right: true,
                },
            ],
            bucket_count: 3,
            shock_weights: vec![1.],
            drift_derivative: -0.2,
        };
        for z in [-1., 0., 1., 2.] {
            let terminal = |sigma: f64| 100. * (-0.5 * sigma * sigma + sigma * z).exp();
            let s = terminal(0.2);
            let h = 1e-6;
            let call_bump = |k: f64| {
                ((terminal(0.2 + h) - k).max(0.) - (terminal(0.2 - h) - k).max(0.)) / (2. * h)
            };
            let a = 2. * call_bump(95.);
            let b = -1.5 * call_bump(105.);
            let (buckets, audit) = plan.sample(s, &[z]).unwrap();
            for (actual, expected) in buckets.iter().zip([0.3 * a, 0.7 * a, b]) {
                assert!((actual - expected).abs() < 5e-8);
            }
            assert!((audit.measure_pre_projection - a - b).abs() < 5e-8);
            assert!((audit.reporting_stats.right_edge_sensitivity - b).abs() < 5e-8);
            assert_eq!(audit.initial_time, 0.);
            assert_eq!(audit.density_excluded, 0.);
            assert!(
                crate::risk::vega_kt_projection_from_parts(
                    buckets.clone(),
                    0.,
                    audit.measure_pre_projection,
                    audit.reporting_stats
                )
                .is_ok()
            );
            let mut corrupted = buckets;
            corrupted[0] += 1.;
            assert!(
                crate::risk::vega_kt_projection_from_parts(
                    corrupted,
                    0.,
                    audit.measure_pre_projection,
                    audit.reporting_stats
                )
                .is_err()
            );
        }
        assert!(plan.sample(100., &[]).is_err());
        assert!(plan.sample(100., &[f64::NAN]).is_err());
        assert!(plan.sample(0., &[0.]).is_err());
    }
}
