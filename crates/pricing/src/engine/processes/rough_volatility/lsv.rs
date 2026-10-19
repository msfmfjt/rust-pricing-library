//! Fixed-variance-driver LSV and its initial-state / leverage-node pullback.
use super::*;
use crate::engine::processes::lsv::{
    LsvError, LsvLeverageSurface, Step, advance, length, step_rows, valid,
};
use crate::mc::LocalVolTimeGrid;

pub(in crate::engine) fn validate_model(model: &RoughVolatilityModel) -> Result<(), LsvError> {
    if matches!(model, RoughVolatilityModel::RoughSabr(m) if m.beta != 1.0) {
        return Err(LsvError::InvalidInput {
            field: "rough_family_lsv_requires_beta_one",
            index: 0,
        });
    }
    Ok(())
}

#[derive(Clone, Debug)]
pub struct RoughFamilyLsvPlan {
    driver: RoughVolatilityPathPlan,
    surface: LsvLeverageSurface,
    rows: Box<[usize]>,
}
#[derive(Clone, Debug)]
pub struct RoughFamilyLsvPath {
    states: Box<[f64]>,
    steps: Box<[Step]>,
    value_count: usize,
}
#[derive(Clone, Debug, PartialEq)]
pub struct RoughFamilyLsvAdjoints {
    /// Initial positive martingale coordinate, leverage axes/anchor fixed.
    pub initial_forward: f64,
    /// Natural d(output)/d(L^2) nodes, not d/dL or market-IV Vega.
    pub squared_leverage: Box<[f64]>,
}
impl RoughFamilyLsvPlan {
    pub fn new(
        model: RoughVolatilityModel,
        surface: LsvLeverageSurface,
        grid: &LocalVolTimeGrid,
    ) -> Result<Self, LsvError> {
        validate_model(&model)?;
        let driver = RoughVolatilityPathPlan::compile(model, grid.nodes().to_vec())?;
        let rows = step_rows(&surface, driver.time_nodes())?;
        Ok(Self {
            driver,
            surface,
            rows,
        })
    }
    #[must_use]
    pub fn surface(&self) -> &LsvLeverageSurface {
        &self.surface
    }
    #[must_use]
    pub fn times(&self) -> &[f64] {
        self.driver.time_nodes()
    }
    #[must_use]
    pub fn model(&self) -> &RoughVolatilityModel {
        self.driver.model()
    }
    #[must_use]
    pub fn random_dimension(&self) -> u32 {
        self.driver.random_dimension()
    }
    #[must_use]
    pub fn pseudo_shocks(&self, seed: u64, path: u64, domain: RandomDomain) -> Vec<f64> {
        self.driver.pseudo_shocks(seed, path, domain)
    }
    pub fn evolve_path(
        &self,
        initial_forward: f64,
        normals: &[f64],
    ) -> Result<RoughFamilyLsvPath, LsvError> {
        valid(initial_forward, "initial_forward", 0, true)?;
        let history = self.driver.variance_history(normals)?;
        let n = self.times().len() - 1;
        let mut states = Vec::with_capacity(n + 1);
        let mut steps = Vec::with_capacity(n);
        states.push(initial_forward);
        let mut f = initial_forward;
        for (j, &normal) in normals.iter().enumerate().take(n) {
            let dt = self.times()[j + 1] - self.times()[j];
            let lookup = self
                .surface
                .lookup_row(self.rows[j], (f / self.surface.initial_f()).ln());
            let multiplier_squared = history.variances[j];
            let variance = lookup.value * multiplier_squared;
            f = if multiplier_squared == 0.0 {
                f
            } else {
                advance(f, variance, dt, normal, j, 0)?
            };
            states.push(f);
            steps.push(Step {
                lookup,
                multiplier_squared,
                variance,
                z: normal,
                dt,
            });
        }
        Ok(RoughFamilyLsvPath {
            states: states.into_boxed_slice(),
            steps: steps.into_boxed_slice(),
            value_count: self.surface.squared_leverage().len(),
        })
    }
    /// A fixed-anchor Spot move crosses a spatial interpolation kink when a
    /// recorded state is exactly on a node with unequal one-sided slopes.
    /// Do not report the arbitrarily selected branch as a two-sided Delta.
    pub(in crate::engine) fn validate_spot_delta_path(
        &self,
        path: &RoughFamilyLsvPath,
    ) -> Result<(), LsvError> {
        let xs = self.surface.log_nodes();
        let m = xs.len();
        for (j, step) in path.steps.iter().enumerate() {
            if step.multiplier_squared == 0.0 {
                continue;
            }
            let x = (path.states[j] / self.surface.initial_f()).ln();
            if let Ok(i) = xs.binary_search_by(|a| a.partial_cmp(&x).expect("finite axes")) {
                let row =
                    &self.surface.squared_leverage()[self.rows[j] * m..(self.rows[j] + 1) * m];
                let left = if i == 0 {
                    0.0
                } else {
                    (row[i] - row[i - 1]) / (xs[i] - xs[i - 1])
                };
                let right = if i + 1 == m {
                    0.0
                } else {
                    (row[i + 1] - row[i]) / (xs[i + 1] - xs[i])
                };
                let scale = left.abs().max(right.abs()).max(f64::MIN_POSITIVE);
                if (left - right).abs() > 64.0 * f64::EPSILON * scale {
                    return Err(LsvError::InvalidInput {
                        field: "rough_lsv_delta_spatial_kink",
                        index: j,
                    });
                }
            }
        }
        Ok(())
    }
    pub fn evolve_states(
        &self,
        initial_forward: f64,
        normals: &[f64],
        states: &mut Vec<f64>,
    ) -> Result<(), LsvError> {
        // Same primal operations as the recorded path, without retaining a tape.
        valid(initial_forward, "initial_forward", 0, true)?;
        let history = self.driver.variance_history(normals)?;
        states.clear();
        states.push(initial_forward);
        let mut f = initial_forward;
        for (j, &normal) in normals.iter().enumerate().take(self.times().len() - 1) {
            let lookup = self
                .surface
                .lookup_row(self.rows[j], (f / self.surface.initial_f()).ln());
            f = if history.variances[j] == 0.0 {
                f
            } else {
                advance(
                    f,
                    lookup.value * history.variances[j],
                    self.times()[j + 1] - self.times()[j],
                    normal,
                    j,
                    0,
                )?
            };
            states.push(f);
        }
        Ok(())
    }
}
impl RoughFamilyLsvPath {
    #[must_use]
    pub fn states(&self) -> &[f64] {
        &self.states
    }
    pub fn reverse(&self, seeds: &[f64]) -> Result<RoughFamilyLsvAdjoints, LsvError> {
        self.reverse_impl(seeds, None)
    }
    /// Also seed the exogenous diffusion variance at every time node.
    pub(in crate::engine) fn reverse_variances(
        &self,
        seeds: &[f64],
    ) -> Result<(RoughFamilyLsvAdjoints, Vec<f64>), LsvError> {
        let mut variance = vec![0.0; self.states.len()];
        let adjoints = self.reverse_impl(seeds, Some(&mut variance))?;
        Ok((adjoints, variance))
    }
    fn reverse_impl(
        &self,
        seeds: &[f64],
        mut variance: Option<&mut [f64]>,
    ) -> Result<RoughFamilyLsvAdjoints, LsvError> {
        length("state_seeds", self.states.len(), seeds.len())?;
        for (i, &v) in seeds.iter().enumerate() {
            valid(v, "state_seed", i, false)?;
        }
        let n = self.steps.len();
        let mut bar = seeds[n];
        let mut values = vec![0.0; self.value_count];
        for j in (0..n).rev() {
            let s = self.steps[j];
            let lb = if s.multiplier_squared == 0.0 {
                0.0
            } else {
                bar * self.states[j + 1]
                    * (-0.5 * s.dt + s.dt.sqrt() * s.z / (2.0 * s.variance.sqrt()))
                    * s.multiplier_squared
            };
            if let Some(v) = variance.as_deref_mut() {
                // A negative raw Heston state is locally stuck at diffusion
                // variance zero. Exact raw zero is rejected by the driver VJP.
                v[j] = if s.multiplier_squared == 0.0 {
                    0.0
                } else {
                    bar * self.states[j + 1]
                        * (-0.5 * s.dt + s.dt.sqrt() * s.z / (2.0 * s.variance.sqrt()))
                        * s.lookup.value
                };
                valid(v[j], "rough_lsv_variance_adjoint", j, false)?;
            }
            s.lookup.transpose(lb, &mut values);
            bar = seeds[j]
                + bar * self.states[j + 1] / self.states[j]
                + lb * s.lookup.derivative_log_f / self.states[j];
        }
        for (i, &v) in values.iter().chain([bar].iter()).enumerate() {
            valid(v, "rough_lsv_adjoint", i, false)?;
        }
        Ok(RoughFamilyLsvAdjoints {
            initial_forward: bar,
            squared_leverage: values.into_boxed_slice(),
        })
    }
}
