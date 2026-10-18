//! Additional conditional price information for the singular initial boundary.
//! No initial density division and no residual-driven normalization are used.
use super::vegakt_steps::moment;
use crate::MonteCarloError;
use crate::core::{FiniteF64, PositiveF64};
use crate::market::LocalVarianceGrid;
use crate::mc::{DeterministicExecutor, LocalVolLogEulerPlan};
use crate::risk::{
    Equation11GammaRow, INITIAL_GAMMA_CONDITIONAL_FD_LABEL, InitialGammaLayout, RiskConfigError,
    VegaKtGammaBumpEstimate, VegaKtInitialGammaEstimate, VegaKtInitialGammaPlan,
    VegaKtInitialGammaSample, VegaKtStepMoment, VegaKtStepSamplingUnit,
    integrate_local_gamma_total_variance_transition,
    integrate_piecewise_linear_local_gamma_transition, local_gamma_transition_cells,
};
use std::sync::Arc;

fn finite(v: f64) -> Result<f64, RiskConfigError> {
    Ok(FiniteF64::new(v, "vega_kt_initial_gamma")?.get())
}
fn invalid(quantity: &'static str) -> RiskConfigError {
    RiskConfigError::VegaKtInitialGammaInput { quantity }
}

impl VegaKtInitialGammaPlan {
    /// First-boundary diagnostic, with at least one later stochastic interval.
    /// Spatially nonflat variance is rejected: piecewise-linear variance knots
    /// do not supply the smooth second derivatives needed for this first slice.
    pub fn new(
        simulation: LocalVolLogEulerPlan,
        grid: LocalVarianceGrid,
        strikes: Vec<f64>,
        relative_bump: f64,
    ) -> Result<Self, RiskConfigError> {
        if simulation.time_grid().step_count() < 2 {
            return Err(invalid(
                "no continuation interval; terminal Dirac requires a measure",
            ));
        }
        let spot = simulation.forward_normalizers()[0];
        if simulation.forward_normalizers().iter().any(|&f| f != spot) {
            return Err(invalid("constant forward required"));
        }
        if grid.time_nodes()[0] != 0.
            || grid.time_nodes().last() < simulation.time_grid().nodes().last()
        {
            return Err(invalid("variance-grid time coverage"));
        }
        let width = grid.log_moneyness_nodes().len();
        if grid
            .values()
            .chunks_exact(width)
            .any(|row| row.iter().any(|&v| v != row[0]))
        {
            return Err(invalid("spatially flat variance required"));
        }
        if !relative_bump.is_finite() || relative_bump <= 0. || relative_bump >= 0.25 {
            return Err(invalid("relative bump must be in (0, 0.25)"));
        }
        let dt = simulation.time_grid().nodes()[1];
        let variance = grid.interpolate(0., 0.)?.value;
        let first_variance = PositiveF64::new(variance * dt, "vega_kt_first_step_variance")?.get();
        let cells = local_gamma_transition_cells(spot, first_variance, &strikes)?;
        let kernel_tail_masses = [cells[0].probability(), cells[cells.len() - 1].probability()];
        let vega_scale = finite(spot * spot * variance.sqrt() * dt)?;
        let mut stencils = Vec::with_capacity(strikes.len());
        let mut bumps = Vec::with_capacity(strikes.len());
        for &strike in &strikes {
            let h = PositiveF64::new(strike * relative_bump, "vega_kt_initial_gamma_bump")?.get();
            let stencil = [
                strike - 2. * h,
                strike - h,
                strike - 0.5 * h,
                strike,
                strike + 0.5 * h,
                strike + h,
                strike + 2. * h,
            ];
            if stencil.iter().any(|v| !v.is_finite() || *v <= 0.)
                || stencil.windows(2).any(|p| p[0] >= p[1])
                || !(12. * h * h).is_finite()
                || h * h == 0.
            {
                return Err(invalid("unrepresentable positive bump stencil"));
            }
            stencils.push(stencil);
            bumps.push(h);
        }
        let components = strikes
            .len()
            .checked_mul(3)
            .and_then(|n| n.checked_add(5))
            .ok_or(RiskConfigError::VegaKtStepShapeOverflow)?;
        Ok(Self {
            layout: Arc::new(InitialGammaLayout {
                simulation,
                grid,
                strikes: strikes.into_boxed_slice(),
                stencils: stencils.into_boxed_slice(),
                bumps: bumps.into_boxed_slice(),
                relative_bump,
                first_variance,
                vega_scale,
                kernel_tail_masses,
                components,
            }),
        })
    }
    #[must_use]
    pub fn simulation_plan(&self) -> &LocalVolLogEulerPlan {
        &self.layout.simulation
    }
    #[must_use]
    pub fn variance_grid(&self) -> &LocalVarianceGrid {
        &self.layout.grid
    }
    #[must_use]
    pub fn strike_nodes(&self) -> &[f64] {
        &self.layout.strikes
    }
    /// `remaining_shocks` corresponds exactly to intervals 1..N; reuse it at
    /// every strike/stencil point. The callback must be a deterministic discounted
    /// payoff of the returned original-index states, with no explicit variance,
    /// barrier bridge, dividend reconstruction, or fitted-policy dependence.
    pub fn sample_from_shocks<F>(
        &self,
        remaining_shocks: &[f64],
        payoff: F,
    ) -> Result<VegaKtInitialGammaSample, MonteCarloError>
    where
        F: Fn(&[f64]) -> Result<f64, MonteCarloError>,
    {
        let l = &self.layout;
        let spot = l.simulation.forward_normalizers()[0];
        let mut values = vec![0.; l.components];
        let mut coarse = Vec::with_capacity(l.strikes.len());
        let mut fine = Vec::with_capacity(l.strikes.len());
        for (i, stencil) in l.stencils.iter().enumerate() {
            let mut prices = [0.; 7];
            for (j, &state) in stencil.iter().enumerate() {
                let path = l.simulation.continue_states_from_first_node(
                    &l.grid,
                    spot,
                    state,
                    remaining_shocks,
                )?;
                prices[j] = finite(payoff(&path)?)?;
            }
            let h = l.bumps[i];
            let c = second(prices[0], prices[1], prices[3], prices[5], prices[6], h)?;
            let f = second(
                prices[1],
                prices[2],
                prices[3],
                prices[4],
                prices[5],
                h * 0.5,
            )?;
            values[3 * i..3 * i + 3].copy_from_slice(&[c, f, finite(f - c)?]);
            coarse.push(c);
            fine.push(f);
        }
        let n = 3 * l.strikes.len();
        let coarse_vega = finite(
            l.vega_scale
                * integrate_piecewise_linear_local_gamma_transition(
                    spot,
                    l.first_variance,
                    &l.strikes,
                    &coarse,
                )?,
        )?;
        let fine_vega = finite(
            l.vega_scale
                * integrate_piecewise_linear_local_gamma_transition(
                    spot,
                    l.first_variance,
                    &l.strikes,
                    &fine,
                )?,
        )?;
        values[n..n + 3].copy_from_slice(&[
            coarse_vega,
            fine_vega,
            finite(fine_vega - coarse_vega)?,
        ]);
        let tail_scale = finite(l.vega_scale * l.first_variance.exp())?;
        values[n + 3] = finite(tail_scale * l.kernel_tail_masses[0] * fine[0])?;
        values[n + 4] = finite(tail_scale * l.kernel_tail_masses[1] * fine[fine.len() - 1])?;
        Ok(VegaKtInitialGammaSample {
            layout: Arc::clone(l),
            values: values.into_boxed_slice(),
            sampling_unit: VegaKtStepSamplingUnit::IndependentPath,
        })
    }
    /// Fixed-block moments of paths or antithetic pair means. RQMC points are
    /// not independent observations. Differences and reconstructed initial Vega
    /// are separate per-sample channels, preserving stencil/node covariances.
    pub fn estimate<F>(
        &self,
        executor: &DeterministicExecutor,
        count: u64,
        sampling_unit: VegaKtStepSamplingUnit,
        evaluate: F,
    ) -> Result<VegaKtInitialGammaEstimate, MonteCarloError>
    where
        F: Fn(u64) -> Result<VegaKtInitialGammaSample, MonteCarloError> + Send + Sync,
    {
        if count == 0 {
            return Err(RiskConfigError::VegaKtStepNoSamplingUnits.into());
        }
        let statistics = executor.try_map_reduce_statistics_vector(
            count,
            self.layout.components,
            |i, out| {
                let sample = evaluate(i)?;
                if !Arc::ptr_eq(&sample.layout, &self.layout) {
                    return Err(MonteCarloError::from(invalid("sample origin")));
                }
                if sample.sampling_unit != sampling_unit {
                    return Err(RiskConfigError::VegaKtStepSamplingUnitMismatch.into());
                }
                out.copy_from_slice(&sample.values);
                Ok(())
            },
        )?;
        let m = statistics
            .into_iter()
            .map(|s| moment(s, count))
            .collect::<Result<Vec<_>, _>>()?;
        let bump = |i| VegaKtGammaBumpEstimate {
            coarse: m[i],
            fine: m[i + 1],
            difference: m[i + 2],
        };
        let n = self.layout.strikes.len();
        Ok(VegaKtInitialGammaEstimate {
            maturity: self.layout.simulation.time_grid().nodes()[1],
            strikes: self.layout.strikes.clone(),
            relative_bump: self.layout.relative_bump,
            nodes: (0..n).map(|i| bump(3 * i)).collect(),
            first_step_vega: bump(3 * n),
            kernel_tail_masses: self.layout.kernel_tail_masses,
            tail_vegas: [m[3 * n + 3], m[3 * n + 4]],
            count,
            sampling_unit,
        })
    }
}
fn second(mm: f64, m: f64, c: f64, p: f64, pp: f64, h: f64) -> Result<f64, RiskConfigError> {
    // Center before combining to reduce cancellation of the payoff's level.
    finite((16. * ((p - c) + (m - c)) - ((pp - c) + (mm - c))) / (12. * h * h))
}
impl VegaKtInitialGammaSample {
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
impl VegaKtGammaBumpEstimate {
    #[must_use]
    pub const fn coarse(self) -> VegaKtStepMoment {
        self.coarse
    }
    #[must_use]
    pub const fn fine(self) -> VegaKtStepMoment {
        self.fine
    }
    /// Signed fine minus coarse, with its own correlated sampling uncertainty.
    /// This is a bump-sensitivity diagnostic, not a rigorous error bound.
    #[must_use]
    pub const fn difference(self) -> VegaKtStepMoment {
        self.difference
    }
}
impl VegaKtInitialGammaEstimate {
    #[must_use]
    pub const fn maturity(&self) -> f64 {
        self.maturity
    }
    #[must_use]
    pub fn strike_nodes(&self) -> &[f64] {
        &self.strikes
    }
    #[must_use]
    pub const fn relative_bump(&self) -> f64 {
        self.relative_bump
    }
    #[must_use]
    pub fn nodes(&self) -> &[VegaKtGammaBumpEstimate] {
        &self.nodes
    }
    #[must_use]
    pub const fn first_step_vega(&self) -> VegaKtGammaBumpEstimate {
        self.first_step_vega
    }
    /// Masses of the normalized tilted Gamma kernel, not the model state density.
    #[must_use]
    pub const fn kernel_tail_masses(&self) -> [f64; 2] {
        self.kernel_tail_masses
    }
    /// Fine-stencil contributions under constant Gamma extrapolation outside
    /// the supplied strike range. They are not bounds on unknown tail error.
    #[must_use]
    pub const fn tail_vegas(&self) -> [VegaKtStepMoment; 2] {
        self.tail_vegas
    }
    #[must_use]
    pub const fn independent_sampling_units(&self) -> u64 {
        self.count
    }
    #[must_use]
    pub const fn sampling_unit(&self) -> VegaKtStepSamplingUnit {
        self.sampling_unit
    }
    #[must_use]
    pub const fn policy_label(&self) -> &'static str {
        INITIAL_GAMMA_CONDITIONAL_FD_LABEL
    }
    /// Connect the fine initial row at t1 to equation (4)'s next positive-time
    /// row. Returns hedge density, without claiming a hedge standard error or
    /// validating the next row's recovery/discretization error.
    pub fn vanilla_hedge_density_from_next(
        &self,
        next: &Equation11GammaRow,
        jets: &[[f64; 3]],
    ) -> Result<Vec<f64>, RiskConfigError> {
        if self.maturity != next.source_maturity().get() {
            return Err(RiskConfigError::VegaKtGammaRowTimeMismatch {
                expected_bits: self.maturity.to_bits(),
                actual_bits: next.source_maturity().get().to_bits(),
            });
        }
        if jets.len() != self.strikes.len() {
            return Err(RiskConfigError::VegaKtRecoveryLengthMismatch {
                quantity: "initial row variance jets",
                expected: self.strikes.len(),
                actual: jets.len(),
            });
        }
        self.strikes
            .iter()
            .zip(&self.nodes)
            .zip(jets)
            .map(|((&spot, node), &[v, v1, v2])| {
                finite(
                    integrate_local_gamma_total_variance_transition(
                        spot,
                        v,
                        v1,
                        v2,
                        next.strike_nodes(),
                        next.gamma_values(),
                    )? - node.fine.mean(),
                )
            })
            .collect()
    }
}
