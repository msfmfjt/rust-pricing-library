//! Checkpointed reverse of the realized hybrid algorithm at fixed HW/Bergomi
//! parameters. Initial curves, Spot and paired target samples remain active.

use super::*;

use crate::models::hull_white_dividends::HullWhiteDividendNodeAdjoints;

pub const HULL_WHITE_AAD_METHOD: &str = "equity-hw-discrete-particle-vjp-v1";

#[derive(Clone, Debug)]
pub struct HullWhitePathAdjoints {
    pub initial_spot: f64,
    /// Legacy slot: BS sigma, or the pure rough model's initial sigma0.
    pub bs_volatility: Option<f64>,
    pub squared_leverage: Vec<f64>,
    pub dividends: Option<Vec<HullWhiteDividendNodeAdjoints>>,
}

#[derive(Clone, Debug)]
pub struct HullWhiteCalibrationAdjoints {
    pub initial_spot: f64,
    /// Effective relative local-variance nodes, holding the paired density fixed.
    pub local_variance: Vec<f64>,
    /// Samples K*p_F^T(K), not the logarithm of those samples.
    pub forward_log_density: Vec<f64>,
    pub dividends: Option<Vec<HullWhiteDividendNodeAdjoints>>,
}

/// Immutable pricing checkpoint. The borrowed plan fixes the model and grid.
pub struct HullWhiteRecordedPath<'a> {
    pub(in crate::engine) plan: &'a HullWhiteEquityPlan,
    pub(in crate::engine) spot: f64,
    pub(in crate::engine) shocks: Vec<f64>,
    pub(in crate::engine) states: Vec<HybridState>,
}

impl HullWhiteEquityPlan {
    pub fn record_path(
        &self,
        spot: f64,
        shocks: &[f64],
    ) -> Result<HullWhiteRecordedPath<'_>, HullWhiteMcError> {
        Ok(HullWhiteRecordedPath {
            plan: self,
            spot,
            shocks: shocks.to_vec(),
            states: self.evolve_path(spot, shocks)?,
        })
    }
}

#[derive(Clone, Copy)]
pub(in crate::engine) struct Lookup {
    pub(in crate::engine) left: usize,
    pub(in crate::engine) weight: f64,
    pub(in crate::engine) value: f64,
    pub(in crate::engine) slope: f64,
}

pub(in crate::engine) fn lookup(surface: &LsvLeverageSurface, row: usize, f: f64) -> Lookup {
    let xs = surface.log_nodes();
    let x = (f / surface.initial_f()).ln();
    let j = xs
        .partition_point(|v| *v <= x)
        .saturating_sub(1)
        .min(xs.len() - 2);
    let weight = ((x - xs[j]) / (xs[j + 1] - xs[j])).clamp(0.0, 1.0);
    let left = row * xs.len() + j;
    let values = surface.squared_leverage();
    Lookup {
        left,
        weight,
        value: (1.0 - weight) * values[left] + weight * values[left + 1],
        slope: if x <= xs[0] || x >= xs[xs.len() - 1] {
            0.0
        } else {
            (values[left + 1] - values[left]) / (xs[j + 1] - xs[j])
        },
    }
}

pub(in crate::engine) fn transpose_lookup(v: Lookup, seed: f64, bars: &mut [f64]) {
    bars[v.left] += (1.0 - v.weight) * seed;
    bars[v.left + 1] += v.weight * seed;
}

pub(in crate::engine) fn target_reverse(
    dividends: Option<&HullWhiteDividendPlan>,
    r: usize,
    state: HybridState,
    f_bar: f64,
    zeta_bar: f64,
    bars: &mut Option<Vec<HullWhiteDividendNodeAdjoints>>,
) -> Result<f64, HullWhiteMcError> {
    match (dividends, bars) {
        (Some(d), Some(bars)) => Ok(d.nodes()[r].reverse_target(
            state.normalized_equity,
            state.rate_factor,
            f_bar,
            zeta_bar,
            &mut bars[r],
        )?),
        (None, None) => Ok(f_bar),
        _ => Err(invalid("dividend_adjoint_shape", r)),
    }
}

impl HullWhiteRecordedPath<'_> {
    #[must_use]
    pub fn states(&self) -> &[HybridState] {
        &self.states
    }

    /// Seeds are on normalized residual equity. Reverse the physical dividend
    /// observations separately. Rate/factor parameters and normal draws are fixed.
    pub fn reverse(&self, state_seeds: &[f64]) -> Result<HullWhitePathAdjoints, HullWhiteMcError> {
        if state_seeds.len() != self.states.len() {
            return Err(invalid("hybrid_state_seed_shape", state_seeds.len()));
        }
        for &v in state_seeds {
            hw_valid(v, "hybrid_state_seed", 0, false)?;
        }
        let plan = self.plan;
        let n = plan.kernels.len();
        let mut out = HullWhitePathAdjoints {
            initial_spot: 0.0,
            bs_volatility: plan.volatility.direct_volatility().map(|_| 0.0),
            squared_leverage: plan
                .volatility
                .lsv()
                .map_or_else(Vec::new, |(_, leverage)| {
                    vec![0.0; leverage.squared_leverage().len()]
                }),
            dividends: plan
                .dividends
                .as_ref()
                .map(HullWhiteDividendPlan::zero_adjoints),
        };
        let mut bar = state_seeds[n];
        for i in (0..n).rev() {
            let state = self.states[i];
            let next = self.states[i + 1].normalized_equity;
            let dt = plan.kernels[i].dt;
            let dw = plan.kernels[i].loading[0][0] * self.shocks[i];
            let exponent_bar = bar * next;
            let direct = bar * next / state.normalized_equity;
            let feedback = if let Some(sigma) = plan.volatility.direct_volatility() {
                let a2 =
                    (2.0 * plan.volatility.log_vol_coefficient() * state.volatility_factor).exp();
                *out.bs_volatility.as_mut().unwrap() +=
                    exponent_bar * (-sigma * a2 * dt + a2.sqrt() * dw);
                0.0
            } else {
                let (factor, leverage) = plan.volatility.lsv().expect("LSV variant");
                let f = target_state(plan.dividends.as_ref(), i, state)?.0;
                let row = leverage
                    .times()
                    .partition_point(|t| *t <= plan.times[i])
                    .saturating_sub(1);
                let l = lookup(leverage, row, f);
                let a2 = (2.0 * factor.vol_of_vol() * state.volatility_factor).exp();
                let v = l.value * a2;
                let lbar = exponent_bar * (-0.5 * dt + dw / (2.0 * v.sqrt())) * a2;
                transpose_lookup(l, lbar, &mut out.squared_leverage);
                out.initial_spot -= lbar * l.slope / self.spot;
                target_reverse(
                    plan.dividends.as_ref(),
                    i,
                    state,
                    lbar * l.slope / f,
                    0.0,
                    &mut out.dividends,
                )?
            };
            bar = state_seeds[i] + direct + feedback;
        }
        out.initial_spot += bar;
        for &v in std::iter::once(&out.initial_spot)
            .chain(out.bs_volatility.iter())
            .chain(&out.squared_leverage)
        {
            hw_valid(v, "hybrid_path_adjoint", 0, false)?;
        }
        Ok(out)
    }
}
