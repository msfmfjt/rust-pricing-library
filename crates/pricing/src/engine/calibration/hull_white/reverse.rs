//! Reverse of the adjacent discounted calibration.
use super::*;
use crate::engine::calibration::capabilities::CalibrationReverse;
use crate::engine::processes::hull_white::reverse::{lookup, target_reverse, transpose_lookup};

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

impl CalibrationReverse for CalibratedHullWhiteLsv {
    type Adjoints = HullWhiteCalibrationAdjoints;
    type Error = HullWhiteMcError;
    fn validate_calibration_reverse(&self) -> Result<(), HullWhiteMcError> {
        let trace = self
            .reverse_trace
            .as_ref()
            .ok_or(LsvError::ReverseTraceNotRetained)?;
        if self.reverse_primal_fingerprint() != trace.primal_fingerprint {
            return Err(invalid("calibration_changed_after_trace", 0));
        }
        Ok(())
    }
    fn calibration_pullback(&self, seeds: &[f64]) -> Result<Self::Adjoints, HullWhiteMcError> {
        self.reverse_leverage(seeds)
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
        self.validate_calibration_reverse()?;
        let trace = self
            .reverse_trace
            .as_ref()
            .expect("validated calibration trace");
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
            let rate_discount = GaussianDiscount::new(trace.rates.integrated_variance(t)?);
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
                    let discount = rate_discount.relative_discount(state.integrated_rate_factor);
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
