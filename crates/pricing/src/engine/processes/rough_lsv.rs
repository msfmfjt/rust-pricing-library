//! Rough Bergomi LSV with deterministic rates: the standalone counterpart of
//! the Hull–White rough-LSV engine, with the same nonuniform kappa=1 Volterra
//! hybrid scheme and discrete-variance centring.
//!
//! Each step draws three factor-major Gaussian blocks: spot normals, variance
//! normals orthogonal to spot, and near-cell residual normals. The variance
//! driver's Brownian increment is `sqrt(dt)*(rho*z_spot + sqrt(1-rho^2)*z_vol)`.
//! The newest-cell integral `J` is its regression on that increment plus an
//! independent residual, which reproduces `Var(J)=dt^(2H)` and
//! `Cov(J,dW_v)=sqrt(2H)*dt^(H+1/2)/(H+1/2)` exactly. At H=1/2 the residual is
//! zero and `J` equals the increment. The stock step uses the variance at the
//! left node, so no future driver noise enters its volatility.

use std::sync::Arc;

use super::lsv::{
    LsvError, LsvLeverageSurface, LsvPathAdjoints, Step, advance, length, lsv_shocks_with_factors,
    step_rows, valid,
};
use crate::mc::{LocalVolTimeGrid, RandomDomain};
use crate::models::{HistoryInnovations, RoughBergomi};
use pricing_numerics::NeumaierSum;

pub const ROUGH_BERGOMI_LSV_SCHEME: &str = "rough-bergomi-lsv-hybrid-kappa1-log-euler-v1";

/// Gaussian blocks per step: spot, orthogonal variance and near-cell residual.
pub(in crate::engine) const ROUGH_RANDOM_BLOCKS: usize = 3;

#[derive(Clone, Copy, Debug)]
struct StepLoading {
    sqrt_dt: f64,
    spot: f64,
    orthogonal: f64,
    near_vol: f64,
    near_residual: f64,
}

/// Volterra weights, discrete variances and per-step innovation loadings for
/// one time grid. Shared by paths so reverse passes need not copy O(N^2) data.
#[derive(Clone, Debug)]
pub(in crate::engine) struct RoughKernel {
    model: RoughBergomi,
    weights: Box<[Box<[f64]>]>,
    variances: Box<[f64]>,
    loadings: Box<[StepLoading]>,
}

impl RoughKernel {
    pub(in crate::engine) fn compile(model: RoughBergomi, times: &[f64]) -> Result<Self, LsvError> {
        let (weights, variances) = model.volterra_weights(times)?;
        let hurst = model.hurst();
        let p = hurst + 0.5;
        let rho = model.correlation();
        let orthogonal = (1.0 - rho * rho).max(0.0).sqrt();
        // 2H <= (H+1/2)^2, so the residual fraction is nonnegative; the clamp
        // only removes roundoff. It is exactly zero at H=1/2.
        let residual_fraction = (1.0 - 2.0 * hurst / (p * p)).max(0.0);
        let loadings = times
            .windows(2)
            .map(|w| {
                let dt = w[1] - w[0];
                StepLoading {
                    sqrt_dt: dt.sqrt(),
                    spot: rho,
                    orthogonal,
                    near_vol: (2.0 * hurst).sqrt() * dt.powf(hurst - 0.5) / p,
                    near_residual: (dt.powf(2.0 * hurst) * residual_fraction).sqrt(),
                }
            })
            .collect::<Vec<_>>();
        for (j, l) in loadings.iter().enumerate() {
            for v in [l.near_vol, l.near_residual] {
                valid(v, "rough_near_loading", j, false)?;
            }
        }
        Ok(Self {
            model,
            weights: weights.into_boxed_slice(),
            variances: variances.into_boxed_slice(),
            loadings: loadings.into_boxed_slice(),
        })
    }

    pub(in crate::engine) fn steps(&self) -> usize {
        self.loadings.len()
    }

    /// Variance-driver Brownian increment and newest-cell integral of step j.
    pub(in crate::engine) fn innovations(
        &self,
        j: usize,
        z_spot: f64,
        z_vol: f64,
        z_near: f64,
    ) -> (f64, f64) {
        let l = self.loadings[j];
        let dw = l.sqrt_dt * (l.spot * z_spot + l.orthogonal * z_vol);
        (dw, l.near_vol * dw + l.near_residual * z_near)
    }

    /// `eta*Y_i` at every node, where `Y_i = X_i - eta*V_i/2`, so the squared
    /// variance multiplier is `exp(eta*Y_i)` with unit mean on this grid.
    pub(in crate::engine) fn prepare_history(
        &self,
        history: HistoryInnovations<'_>,
        out: &mut Vec<f64>,
    ) -> Result<(), LsvError> {
        let dw = history.increments;
        let near = history.near_cell;
        length("rough increments", self.steps(), dw.len())?;
        length("rough near-cell integrals", self.steps(), near.len())?;
        let eta = self.model.vol_of_vol();
        out.clear();
        out.push(0.0);
        for i in 1..self.variances.len() {
            let mut x = NeumaierSum::new();
            x.add(near[i - 1]);
            for (&w, &d) in self.weights[i].iter().zip(dw) {
                x.add(w * d);
            }
            let value = eta * (x.total() - 0.5 * eta * self.variances[i]);
            valid(value, "rough_log_variance_multiplier", i, false)?;
            out.push(value);
        }
        Ok(())
    }
}

/// Path plan for calibrated rough-LSV. Mirrors `BergomiLsvPlan`.
#[derive(Clone, Debug)]
pub struct RoughBergomiLsvPlan {
    model: RoughBergomi,
    surface: LsvLeverageSurface,
    times: Box<[f64]>,
    rows: Box<[usize]>,
    kernel: Arc<RoughKernel>,
}

/// One recorded rough-LSV path, for the pathwise reverse.
#[derive(Clone, Debug)]
pub struct RoughBergomiLsvPath {
    states: Box<[f64]>,
    steps: Box<[Step]>,
    kernel: Arc<RoughKernel>,
    value_count: usize,
}

impl RoughBergomiLsvPlan {
    /// The execution grid may insert payoff and dividend observations, but must
    /// contain every leverage knot inside its horizon. The Volterra kernel is
    /// compiled on the execution grid.
    pub fn new(
        model: RoughBergomi,
        surface: LsvLeverageSurface,
        time_grid: &LocalVolTimeGrid,
    ) -> Result<Self, LsvError> {
        let times = time_grid.nodes();
        let rows = step_rows(&surface, times)?;
        let kernel = RoughKernel::compile(model, times)?;
        Ok(Self {
            model,
            surface,
            times: times.into(),
            rows,
            kernel: Arc::new(kernel),
        })
    }
    #[must_use]
    pub fn surface(&self) -> &LsvLeverageSurface {
        &self.surface
    }
    #[must_use]
    pub fn times(&self) -> &[f64] {
        &self.times
    }
    #[must_use]
    pub const fn model(&self) -> RoughBergomi {
        self.model
    }
    /// Discrete driver variances `V_i` at the time nodes.
    #[must_use]
    pub fn driver_variances(&self) -> &[f64] {
        &self.kernel.variances
    }

    /// Factor-major layout: n spot normals, n orthogonal variance normals, then
    /// n near-cell residual normals.
    pub fn pseudo_shocks(
        &self,
        seed: u64,
        path: u64,
        domain: RandomDomain,
    ) -> Result<Vec<f64>, LsvError> {
        lsv_shocks_with_factors(seed, path, self.kernel.steps(), domain, ROUGH_RANDOM_BLOCKS)
    }

    fn log_multipliers(&self, initial_f: f64, shocks: &[f64]) -> Result<Vec<f64>, LsvError> {
        let n = self.kernel.steps();
        length("shocks", ROUGH_RANDOM_BLOCKS * n, shocks.len())?;
        for (i, &z) in shocks.iter().enumerate() {
            valid(z, "shock", i, false)?;
        }
        valid(initial_f, "initial_f", 0, true)?;
        let mut dw = Vec::with_capacity(n);
        let mut near = Vec::with_capacity(n);
        for j in 0..n {
            let (d, q) = self
                .kernel
                .innovations(j, shocks[j], shocks[n + j], shocks[2 * n + j]);
            dw.push(d);
            near.push(q);
        }
        let mut out = Vec::with_capacity(n + 1);
        self.kernel.prepare_history(
            HistoryInnovations {
                increments: &dw,
                near_cell: &near,
            },
            &mut out,
        )?;
        Ok(out)
    }

    /// Price-only evolution. Writes the same f states as `evolve_path` for the
    /// same shocks, bit for bit, without the reverse-mode step records.
    pub fn evolve_states(
        &self,
        initial_f: f64,
        shocks: &[f64],
        states: &mut Vec<f64>,
    ) -> Result<(), LsvError> {
        let log_multipliers = self.log_multipliers(initial_f, shocks)?;
        states.clear();
        states.push(initial_f);
        let mut f = initial_f;
        let mut cell = 0;
        for (i, &m) in log_multipliers[..self.kernel.steps()].iter().enumerate() {
            let dt = self.times[i + 1] - self.times[i];
            let lookup = self.surface.lookup_row_from(
                self.rows[i],
                (f / self.surface.initial_f()).ln(),
                &mut cell,
            );
            f = advance(f, lookup.value * m.exp(), dt, shocks[i], i, 0)?;
            states.push(f);
        }
        Ok(())
    }

    pub fn evolve_path(
        &self,
        initial_f: f64,
        shocks: &[f64],
    ) -> Result<RoughBergomiLsvPath, LsvError> {
        let log_multipliers = self.log_multipliers(initial_f, shocks)?;
        let n = self.kernel.steps();
        let mut states = Vec::with_capacity(n + 1);
        let mut steps = Vec::with_capacity(n);
        states.push(initial_f);
        let mut f = initial_f;
        let mut cell = 0;
        for (i, &m) in log_multipliers[..n].iter().enumerate() {
            let dt = self.times[i + 1] - self.times[i];
            let lookup = self.surface.lookup_row_from(
                self.rows[i],
                (f / self.surface.initial_f()).ln(),
                &mut cell,
            );
            let multiplier_squared = m.exp();
            let variance = lookup.value * multiplier_squared;
            f = advance(f, variance, dt, shocks[i], i, 0)?;
            steps.push(Step {
                lookup,
                multiplier_squared,
                variance,
                z: shocks[i],
                dt,
            });
            states.push(f);
        }
        Ok(RoughBergomiLsvPath {
            states: states.into_boxed_slice(),
            steps: steps.into_boxed_slice(),
            kernel: Arc::clone(&self.kernel),
            value_count: self.surface.squared_leverage().len(),
        })
    }
}

impl RoughBergomiLsvPath {
    #[must_use]
    pub fn states(&self) -> &[f64] {
        &self.states
    }
    /// Squared variance multipliers `a_i^2` used by each step.
    #[must_use]
    pub fn multipliers_squared(&self) -> Vec<f64> {
        self.steps.iter().map(|s| s.multiplier_squared).collect()
    }

    /// Adjoints of the seeded states with respect to squared leverage, the
    /// initial f and all three shock blocks. `orthogonal_shocks` holds the
    /// orthogonal-variance block followed by the near-cell residual block.
    pub fn reverse(&self, state_seeds: &[f64]) -> Result<LsvPathAdjoints<()>, LsvError> {
        length("state_seeds", self.states.len(), state_seeds.len())?;
        for (i, &s) in state_seeds.iter().enumerate() {
            valid(s, "state_seed", i, false)?;
        }
        let kernel = &self.kernel;
        let n = self.steps.len();
        let eta = kernel.model.vol_of_vol();
        let mut fbar = state_seeds[n];
        let mut values = vec![0.0; self.value_count];
        let mut spot = vec![0.0; n];
        // Adjoint of the Volterra state X_i at the left node of each step.
        let mut xbar = vec![0.0; n];
        for i in (0..n).rev() {
            let c = self.steps[i];
            let exponent_bar = fbar * self.states[i + 1];
            let vbar = exponent_bar * (-0.5 * c.dt + c.dt.sqrt() * c.z / (2.0 * c.variance.sqrt()));
            let leverage_bar = vbar * c.multiplier_squared;
            c.lookup.transpose(leverage_bar, &mut values);
            // variance = leverage * exp(eta*Y_i), and dY_i/dX_i = 1.
            xbar[i] = vbar * c.variance * eta;
            spot[i] = exponent_bar * c.variance.sqrt() * c.dt.sqrt();
            fbar = state_seeds[i]
                + fbar * self.states[i + 1] / self.states[i]
                + leverage_bar * c.lookup.derivative_log_f / self.states[i];
        }
        // X_i = J_(i-1) + sum_(j<=i-2) w_ij dW_j; X_0 is the constant zero.
        let mut orth = vec![0.0; 2 * n];
        for j in 0..n {
            let near_bar = if j + 1 < n { xbar[j + 1] } else { 0.0 };
            let mut dw_bar = NeumaierSum::new();
            for (i, &x) in xbar.iter().enumerate().skip(j + 2) {
                dw_bar.add(kernel.weights[i][j] * x);
            }
            let l = kernel.loadings[j];
            let dw_bar = dw_bar.total() + l.near_vol * near_bar;
            spot[j] += l.sqrt_dt * l.spot * dw_bar;
            orth[j] = l.sqrt_dt * l.orthogonal * dw_bar;
            orth[n + j] = l.near_residual * near_bar;
        }
        for (i, &v) in values
            .iter()
            .chain(spot.iter())
            .chain(orth.iter())
            .chain([fbar].iter())
            .enumerate()
        {
            valid(v, "path_adjoint", i, false)?;
        }
        Ok(LsvPathAdjoints {
            initial_f: fbar,
            initial_factor: (),
            squared_leverage: values.into_boxed_slice(),
            spot_shocks: spot.into_boxed_slice(),
            orthogonal_shocks: orth.into_boxed_slice(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{HullWhite1Factor, HybridCorrelation};

    fn close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() <= 1e-12 * (1.0 + expected.abs()),
            "actual={actual:.15e}, expected={expected:.15e}"
        );
    }

    // The three-block law must equal the Hull–White hybrid law restricted to
    // (dW_S, dW_v, J) when rates are deterministic.
    #[test]
    fn near_cell_law_matches_the_hybrid_covariance_at_zero_rate_volatility() {
        let rates = HullWhite1Factor::new(0.1, vec![0.0], vec![0.0]).unwrap();
        let times = [0.0, 0.01, 0.2, 0.25, 1.0];
        for (hurst, rho) in [(0.07, -0.7), (0.3, 0.4), (0.5, -1.0), (0.5, 0.2)] {
            let model = RoughBergomi::new(hurst, 1.3, rho).unwrap();
            let kernel = RoughKernel::compile(model, &times).unwrap();
            let correlation = HybridCorrelation::new(rho, 0.0, 0.0).unwrap();
            for (j, w) in times.windows(2).enumerate() {
                let c = model
                    .hybrid_covariance(&rates, w[0], w[1], correlation)
                    .unwrap();
                let l = kernel.loadings[j];
                let dt = w[1] - w[0];
                close(l.sqrt_dt * l.sqrt_dt, c[0][0]);
                close(
                    l.sqrt_dt * l.sqrt_dt * (l.spot * l.spot + l.orthogonal * l.orthogonal),
                    c[1][1],
                );
                close(l.sqrt_dt * l.sqrt_dt * l.spot, c[1][0]);
                close(
                    l.near_vol * l.near_vol * dt + l.near_residual * l.near_residual,
                    c[4][4],
                );
                close(l.near_vol * dt, c[4][1]);
                close(l.near_vol * l.spot * dt, c[4][0]);
            }
            if hurst == 0.5 {
                assert!(
                    kernel
                        .loadings
                        .iter()
                        .all(|l| l.near_residual == 0.0 && l.near_vol == 1.0)
                );
            }
        }
    }
}
