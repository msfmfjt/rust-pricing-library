//! Checkpointed reverse of the realized hybrid algorithm at fixed HW/Bergomi
//! parameters. Initial curves, Spot and paired target samples remain active.

use super::*;
use pricing_models::hull_white_dividends::HullWhiteDividendNodeAdjoints;

pub const HULL_WHITE_AAD_METHOD: &str = "equity-hw-discrete-particle-vjp-v1";

#[derive(Clone, Debug)]
pub(super) struct TraceRow {
    pub(super) states: Box<[HybridState]>,
    pub(super) weight_sums: Box<[f64]>,
    pub(super) donors: Box<[usize]>,
    pub(super) analytic: bool,
}

#[derive(Clone, Debug)]
pub(super) struct CalibrationTrace {
    pub(super) rows: Box<[TraceRow]>,
    pub(super) target: HullWhiteLsvTarget,
    pub(super) factor: HybridVolatilityFactor,
    pub(super) rates: HullWhite1Factor,
    pub(super) correlation: HybridCorrelation,
    pub(super) config: LsvParticleConfig,
    pub(super) dividends: Option<HullWhiteDividendPlan>,
    pub(super) kernels: Box<[Kernel]>,
    pub(super) primal_fingerprint: [u8; 32],
}

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
    plan: &'a HullWhiteEquityPlan,
    spot: f64,
    shocks: Vec<f64>,
    states: Vec<HybridState>,
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
struct Lookup {
    left: usize,
    weight: f64,
    value: f64,
    slope: f64,
}
fn lookup(surface: &LsvLeverageSurface, row: usize, f: f64) -> Lookup {
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
fn transpose_lookup(v: Lookup, seed: f64, bars: &mut [f64]) {
    bars[v.left] += (1.0 - v.weight) * seed;
    bars[v.left + 1] += v.weight * seed;
}
fn target_reverse(
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

impl CalibratedHullWhiteLsv {
    pub(super) fn reverse_primal_fingerprint(&self) -> [u8; 32] {
        let mut hash = blake3::Hasher::new();
        hash.update(b"hybrid-calibration-reverse-primal/v1\0");
        hash.update(&self.surface.initial_f().to_bits().to_be_bytes());
        for values in [
            self.surface.times(),
            self.surface.log_nodes(),
            self.surface.squared_leverage(),
            &self.conditional_second_moments,
            &self.rate_corrections,
            &self.conditional_cross_moments,
            &self.conditional_rate_variances,
        ] {
            hash.update(&(values.len() as u64).to_be_bytes());
            for &v in values {
                hash.update(&v.to_bits().to_be_bytes());
            }
        }
        *hash.finalize().as_bytes()
    }
    #[must_use]
    pub fn retains_reverse_trace(&self) -> bool {
        self.reverse_trace.is_some()
    }

    /// Differentiate the realized finite calibration with support, donor and
    /// digital-indicator decisions fixed. The density and variance arrays are
    /// separate active inputs; a smile shock must contract both sets of adjoints.
    pub fn reverse_leverage(
        &self,
        leverage_adjoints: &[f64],
    ) -> Result<HullWhiteCalibrationAdjoints, HullWhiteMcError> {
        let trace = self
            .reverse_trace
            .as_ref()
            .ok_or(LsvError::ReverseTraceNotRetained)?;
        if self.reverse_primal_fingerprint() != trace.primal_fingerprint {
            return Err(invalid("calibration_changed_after_trace", 0));
        }
        let count = self.surface.squared_leverage().len();
        if leverage_adjoints.len() != count {
            return Err(invalid(
                "hybrid_leverage_seed_shape",
                leverage_adjoints.len(),
            ));
        }
        for &v in leverage_adjoints {
            hw_valid(v, "hybrid_leverage_seed", 0, false)?;
        }
        let spot = self.surface.initial_f();
        let xs = self.surface.log_nodes();
        let m = xs.len();
        let nt = trace.rows.len();
        let np = trace.config.particle_count();
        let mut lbar = leverage_adjoints.to_vec();
        let mut state_bar = vec![0.0; np];
        let mut out = HullWhiteCalibrationAdjoints {
            initial_spot: 0.0,
            local_variance: vec![0.0; count],
            forward_log_density: vec![0.0; count],
            dividends: trace
                .dividends
                .as_ref()
                .map(HullWhiteDividendPlan::zero_adjoints),
        };
        let rng = Philox4x32::from_seed(trace.config.seed());
        for r in (0..nt).rev() {
            let row = &trace.rows[r];
            let t = self.surface.times()[r];
            if r + 1 < nt {
                let kernel = &trace.kernels[r];
                for (p, bar) in state_bar.iter_mut().enumerate() {
                    let state = row.states[p];
                    let next = trace.rows[r + 1].states[p].normalized_equity;
                    let f = target_state(trace.dividends.as_ref(), r, state)?.0;
                    let l = lookup(&self.surface, r, f);
                    let a2 = (2.0 * trace.factor.vol_of_vol() * state.volatility_factor).exp();
                    let v = l.value * a2;
                    let dw = kernel.loading[0][0]
                        * rng.standard_normal(RandomCoordinate::new(
                            p as u64,
                            r as u32,
                            RandomDomain::LsvCalibration,
                        ));
                    let seed = *bar * next * (-0.5 * kernel.dt + dw / (2.0 * v.sqrt())) * a2;
                    transpose_lookup(l, seed, &mut lbar);
                    out.initial_spot -= seed * l.slope / spot;
                    *bar = *bar * next / state.normalized_equity
                        + target_reverse(
                            trace.dividends.as_ref(),
                            r,
                            state,
                            seed * l.slope / f,
                            0.0,
                            &mut out.dividends,
                        )?;
                }
            }
            // Donor moments and donor drift correction are shared, but each
            // destination cell retains its own effective target variance.
            let mut moment_bars = vec![[0.0; 4]; m];
            for j in 0..m {
                let index = r * m + j;
                let seed = lbar[index];
                if seed == 0.0 {
                    continue;
                }
                let l2 = self.surface.squared_leverage()[index];
                let l = l2.sqrt();
                let a = self.conditional_second_moments[index];
                let b = trace.correlation.equity_rate * self.conditional_cross_moments[index];
                let denominator = a * l + b;
                if !denominator.is_finite() || denominator <= 0.0 {
                    return Err(invalid("singular_leverage_reverse", index));
                }
                let vbar = seed * l / denominator;
                out.local_variance[index] += vbar;
                let donor = &mut moment_bars[row.donors[j]];
                donor[0] -= vbar * l2;
                donor[1] -= vbar * 2.0 * trace.correlation.equity_rate * l;
                donor[2] -= vbar;
                donor[3] -= vbar;
            }
            let half_v = 0.5 * trace.rates.integrated_variance(t)?;
            for (j, bars) in moment_bars.iter().enumerate() {
                if bars.iter().all(|v| *v == 0.0) {
                    continue;
                }
                let index = r * m + j;
                let strike = spot * xs[j].exp();
                if row.analytic {
                    if r == 0 {
                        let zeta = target_state(trace.dividends.as_ref(), 0, row.states[0])?.1;
                        let ratio = zeta / strike;
                        let ratio_bar = bars[1] + 2.0 * bars[2] * ratio;
                        out.initial_spot -= ratio_bar * ratio / spot;
                        target_reverse(
                            trace.dividends.as_ref(),
                            0,
                            row.states[0],
                            0.0,
                            ratio_bar / strike,
                            &mut out.dividends,
                        )?;
                    }
                    continue;
                }
                let density = trace.target.log_densities()[index];
                let correction = self.rate_corrections[index];
                if density > 0.0 {
                    out.forward_log_density[index] -= bars[3] * correction / density;
                }
                if let (Some(dividends), Some(dividend_bars)) =
                    (&trace.dividends, &mut out.dividends)
                {
                    let node = &dividends.nodes()[r];
                    let h = node.deterministic_reserve() / node.scale();
                    let base_correction = correction / (1.0 + h / strike);
                    let hbar = bars[3] * base_correction / strike;
                    dividend_bars[r].deterministic_reserve += hbar / node.scale();
                    dividend_bars[r].scale -= hbar * h / node.scale();
                    out.initial_spot -= hbar * h / spot;
                }
                let width = trace.config.log_bandwidth();
                let total = row.weight_sums[j];
                if !total.is_finite() || total <= 0.0 {
                    return Err(invalid("unsupported_reverse_donor", index));
                }
                for (p, state) in row.states.iter().copied().enumerate() {
                    let (f, zeta) = target_state(trace.dividends.as_ref(), r, state)?;
                    let u = ((f / spot).ln() - xs[j]) / width;
                    if u.abs() >= 1.0 {
                        continue;
                    }
                    let discount = (-state.integrated_rate_factor - half_v).exp();
                    let weight = (1.0 - u * u).powi(2) * discount / total;
                    let a2 = (2.0 * trace.factor.vol_of_vol() * state.volatility_factor).exp();
                    let a = a2.sqrt();
                    let eta = a * state.normalized_equity / f;
                    let rate_ratio = zeta / f;
                    let eta_bar = weight * (2.0 * bars[0] * eta + bars[1] * rate_ratio);
                    let rate_bar = weight * (bars[1] * eta + 2.0 * bars[2] * rate_ratio);
                    let log_f_bar = -4.0 * u * (1.0 - u * u) / width * discount / total
                        * (bars[0] * (eta * eta - self.conditional_second_moments[index])
                            + bars[1] * (eta * rate_ratio - self.conditional_cross_moments[index])
                            + bars[2]
                                * (rate_ratio * rate_ratio
                                    - self.conditional_rate_variances[index]));
                    let f_bar = (log_f_bar - eta_bar * eta - rate_bar * rate_ratio) / f;
                    out.initial_spot -= log_f_bar / spot;
                    state_bar[p] += eta_bar * a / f
                        + target_reverse(
                            trace.dividends.as_ref(),
                            r,
                            state,
                            f_bar,
                            rate_bar / f,
                            &mut out.dividends,
                        )?;
                }
            }
        }
        let mut initial = NeumaierSum::new();
        for v in state_bar {
            initial.add(v);
        }
        out.initial_spot += initial.total();
        for &v in std::iter::once(&out.initial_spot)
            .chain(&out.local_variance)
            .chain(&out.forward_log_density)
        {
            hw_valid(v, "hybrid_calibration_adjoint", 0, false)?;
        }
        Ok(out)
    }
}
