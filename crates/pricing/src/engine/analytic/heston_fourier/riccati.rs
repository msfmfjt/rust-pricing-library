//! Implicit linear product integration for the fractional Riccati equation,
//! and exponential linear product integration for the finite-factor ODEs.
//! Both integrate the SAME continuous-time affine model as their path counterpart,
//! but neither reproduces its finite-grid full-truncation Monte Carlo law.

use super::{FourierError, HestonFourierConfig};
use crate::rough_volatility::{RoughHeston, RoughVolatilityModel};
use pricing_numerics::{Complex64 as C, digamma_half_to_two, gamma_half_to_two};

#[derive(Clone, Debug)]
enum Kernel {
    Power {
        initial: Vec<f64>,
        interior: Vec<f64>,
        endpoint: f64,
    },
    Lift {
        weights: Vec<f64>,
        decay: Vec<f64>,
        left: Vec<f64>,
        right: Vec<f64>,
    },
}
#[derive(Clone, Debug)]
pub(super) struct RiccatiPlan {
    parameters: RoughHeston,
    kernel: Kernel,
    maturity: f64,
    steps: usize,
}
impl RiccatiPlan {
    pub(super) fn new(
        model: &RoughVolatilityModel,
        maturity: f64,
        cfg: HestonFourierConfig,
    ) -> Result<Self, FourierError> {
        let n = cfg.time_steps();
        let dt = maturity / n as f64;
        let (parameters, kernel) = match model {
            RoughVolatilityModel::RoughHeston(p) => {
                let alpha = p.hurst() + 0.5;
                let gamma = gamma_half_to_two(alpha)
                    .ok_or(FourierError::NumericalFailure("Gamma(alpha)"))?;
                let endpoint = dt.powf(alpha) / (alpha * (alpha + 1.0) * gamma);
                let mut initial = vec![0.0; n + 1];
                let mut interior = vec![0.0; n + 1];
                // Positive hat-function integrals. Direct powers are adequate at
                // this bounded grid size; independent quadrature tests guard them.
                for l in 1..=n {
                    let x = l as f64;
                    initial[l] = endpoint
                        * ((x - 1.0).powf(alpha + 1.0) - (x - 1.0 - alpha) * x.powf(alpha));
                    interior[l] = endpoint
                        * ((x + 1.0).powf(alpha + 1.0) - 2.0 * x.powf(alpha + 1.0)
                            + (x - 1.0).powf(alpha + 1.0));
                }
                (
                    p.clone(),
                    Kernel::Power {
                        initial,
                        interior,
                        endpoint,
                    },
                )
            }
            RoughVolatilityModel::LiftedHeston(p) => {
                let mut decay = Vec::new();
                let mut left = Vec::new();
                let mut right = Vec::new();
                for &rate in p.rates() {
                    let r = rate * dt;
                    if !r.is_finite() {
                        return Err(FourierError::NumericalFailure("lift rate * dt"));
                    }
                    let (a, b) = exponential_weights(r);
                    decay.push((-r).exp());
                    left.push(dt * a);
                    right.push(dt * b);
                }
                (
                    p.heston_parameters().clone(),
                    Kernel::Lift {
                        weights: p.weights().to_vec(),
                        decay,
                        left,
                        right,
                    },
                )
            }
            _ => return Err(FourierError::UnsupportedModel),
        };
        Ok(Self {
            parameters,
            kernel,
            maturity,
            steps: n,
        })
    }
    pub(super) fn is_zero_variance(&self) -> bool {
        self.maturity == 0.0
            || (self.parameters.initial_variance() == 0.0
                && self.parameters.mean_reversion() * self.parameters.long_run_variance() == 0.0)
    }
    pub(super) fn control_variance(&self) -> f64 {
        self.parameters.initial_variance() * self.maturity
    }
    pub(super) fn maturity(&self) -> f64 {
        self.maturity
    }
    pub(super) fn parameter_work(&self) -> u64 {
        let per_node = match &self.kernel {
            Kernel::Power { .. } => (self.steps as u64).pow(2),
            Kernel::Lift { weights, .. } => self.steps as u64 * weights.len() as u64,
        };
        // One primal and five tangent directions; guard before allocations.
        6 * per_node
    }
    pub(super) fn log_transform(&self, z: C) -> Result<C, FourierError> {
        self.solve::<false, false>(z, None).map(|(value, _)| value)
    }
    pub(super) fn log_transform_derivatives(&self, z: C) -> Result<(C, [C; 5]), FourierError> {
        if self.maturity > 0.0 && self.control_variance() <= 0.0 {
            return Err(FourierError::InvalidInput(
                "parameter risk requires positive initial/control variance at positive maturity",
            ));
        }
        self.solve::<true, false>(z, None)
    }
    pub(super) fn hurst_work(&self) -> u64 {
        // Primal history, tangent history, and differentiated-weight history.
        3 * (self.steps as u64).pow(2)
    }
    pub(super) fn hurst_weights(&self) -> Result<PowerHurstWeights, FourierError> {
        if !matches!(self.kernel, Kernel::Power { .. }) {
            return Err(FourierError::InvalidInput(
                "Hurst risk requires Rough Heston, not a fixed finite lift",
            ));
        }
        PowerHurstWeights::new(self.parameters.hurst(), self.maturity, self.steps)
    }
    pub(super) fn log_transform_hurst_derivative(
        &self,
        z: C,
        weights: &PowerHurstWeights,
    ) -> Result<(C, C), FourierError> {
        // Lane zero holds ONLY Hurst in this distinct const instantiation.
        self.solve::<false, true>(z, Some(weights))
            .map(|(v, d)| (v, d[0]))
    }
    // Preserve the primal operation order. The const-false instantiation performs
    // no tangent allocations or arithmetic; both routes use the same root.
    fn solve<const RISK: bool, const HURST: bool>(
        &self,
        z: C,
        hurst_weights: Option<&PowerHurstWeights>,
    ) -> Result<(C, [C; 5]), FourierError> {
        if !z.is_finite() || !(0.0..=1.0).contains(&z.re) || z.im.abs() > 1e6 {
            return Err(FourierError::InvalidInput(
                "exponent: 0<=real<=1, |imag|<=1e6, finite",
            ));
        }
        let p = &self.parameters;
        if self.maturity == 0.0
            || z == C::ZERO
            || z == C::ONE
            || (p.initial_variance() == 0.0 && p.mean_reversion() * p.long_run_variance() == 0.0)
        {
            return Ok((C::ZERO, [C::ZERO; 5]));
        }
        let c = (z * z - z) * 0.5;
        let b = z * (p.correlation() * p.vol_of_vol()) - C::new(p.mean_reversion(), 0.0);
        let a = 0.5 * p.vol_of_vol() * p.vol_of_vol();
        if !a.is_finite() || !b.is_finite() || !self.control_variance().is_finite() {
            return Err(FourierError::NumericalFailure("model coefficient overflow"));
        }
        let rhs = |psi: C| c + b * psi + psi * psi * a;
        let dt = self.maturity / self.steps as f64;
        let mut psi = C::ZERO;
        let mut f = c;
        let mut exponent = C::ZERO;
        let mut history = vec![c];
        let mut factors = match &self.kernel {
            Kernel::Lift { weights, .. } => vec![C::ZERO; weights.len()],
            _ => Vec::new(),
        };
        let mut dpsi = [C::ZERO; 5];
        let mut df = [C::ZERO; 5];
        let mut dexponent = [C::ZERO; 5];
        let mut dhistory = if RISK { vec![df] } else { Vec::new() };
        let mut dfactors = if RISK {
            vec![df; factors.len()]
        } else {
            Vec::new()
        };
        let mut hurst_history = if HURST { vec![C::ZERO] } else { Vec::new() };
        let mut hurst_psi = C::ZERO;
        let mut hurst_f = C::ZERO;
        let mut hurst_exponent = C::ZERO;
        for n in 1..=self.steps {
            let (past, end) = match &self.kernel {
                Kernel::Power {
                    initial,
                    interior,
                    endpoint,
                } => {
                    let mut sum = c * initial[n];
                    for j in 1..n {
                        sum = sum + history[j] * interior[n - j];
                    }
                    (sum, *endpoint)
                }
                Kernel::Lift {
                    weights,
                    decay,
                    left,
                    right,
                } => {
                    let mut sum = C::ZERO;
                    let mut end = 0.0;
                    for k in 0..weights.len() {
                        sum = sum + (factors[k] * decay[k] + f * left[k]) * weights[k];
                        end += weights[k] * right[k];
                    }
                    (sum, end)
                }
            };
            // y = past + end*(c+b*y+a*y^2). Use the small quadratic root,
            // rationalized to remain continuous as a -> 0; no Newton iteration.
            let linear = C::ONE - b * end;
            let d = past + c * end;
            let root = if a == 0.0 {
                d / linear
            } else {
                let discriminant = linear * linear - d * (4.0 * end * a);
                let sq = discriminant.sqrt();
                let denom = linear + sq;
                if denom.abs() == 0.0 {
                    return Err(FourierError::NumericalFailure(
                        "quadratic branch denominator",
                    ));
                }
                d * 2.0 / denom
            };
            let next_f = rhs(root);
            let residual = root - past - next_f * end;
            if !root.is_finite()
                || !next_f.is_finite()
                || residual.abs() > 1e-9 * (1.0 + root.abs() + past.abs())
            {
                return Err(FourierError::NumericalFailure(
                    "Riccati residual/non-finite; refine grid",
                ));
            }
            if RISK {
                // Kernel is held fixed: no Hurst or lift-weight/rate derivative.
                let mut dpast = [C::ZERO; 5];
                match &self.kernel {
                    Kernel::Power { interior, .. } => {
                        for j in 1..n {
                            for (q, value) in dpast.iter_mut().enumerate() {
                                *value = *value + dhistory[j][q] * interior[n - j];
                            }
                        }
                    }
                    Kernel::Lift {
                        weights,
                        decay,
                        left,
                        ..
                    } => {
                        for k in 0..weights.len() {
                            for (q, value) in dpast.iter_mut().enumerate() {
                                *value = *value
                                    + (dfactors[k][q] * decay[k] + df[q] * left[k]) * weights[k];
                            }
                        }
                    }
                }
                let explicit = [
                    C::ZERO,
                    -root,
                    C::ZERO,
                    z * p.correlation() * root + root * root * p.vol_of_vol(),
                    z * p.vol_of_vol() * root,
                ];
                let slope = b + root * (2.0 * a);
                let denominator = C::ONE - slope * end;
                if !denominator.is_finite()
                    || denominator.abs() <= 64.0 * f64::EPSILON * (1.0 + (slope * end).abs())
                {
                    return Err(FourierError::NumericalFailure("singular Riccati tangent"));
                }
                let mut next_dpsi = [C::ZERO; 5];
                let mut next_df = [C::ZERO; 5];
                for q in 0..5 {
                    next_dpsi[q] = (dpast[q] + explicit[q] * end) / denominator;
                    next_df[q] = explicit[q] + slope * next_dpsi[q];
                    let residual = next_dpsi[q] - dpast[q] - next_df[q] * end;
                    if !next_df[q].is_finite()
                        || !next_dpsi[q].is_finite()
                        || residual.abs() > 1e-9 * (1.0 + next_dpsi[q].abs() + dpast[q].abs())
                    {
                        return Err(FourierError::NumericalFailure(
                            "Riccati tangent residual/non-finite",
                        ));
                    }
                    let direct_v0 = if q == 0 { f + next_f } else { C::ZERO };
                    let immigration = match q {
                        1 => p.long_run_variance(),
                        2 => p.mean_reversion(),
                        _ => 0.0,
                    };
                    dexponent[q] = dexponent[q]
                        + ((df[q] + next_df[q]) * p.initial_variance()
                            + (dpsi[q] + next_dpsi[q])
                                * (p.mean_reversion() * p.long_run_variance())
                            + direct_v0
                            + (psi + root) * immigration)
                            * (0.5 * dt);
                }
                if let Kernel::Lift {
                    decay, left, right, ..
                } = &self.kernel
                {
                    for k in 0..dfactors.len() {
                        for q in 0..5 {
                            dfactors[k][q] =
                                dfactors[k][q] * decay[k] + df[q] * left[k] + next_df[q] * right[k];
                        }
                    }
                } else {
                    dhistory.push(next_df);
                }
                dpsi = next_dpsi;
                df = next_df;
            }
            if HURST {
                let w =
                    hurst_weights.ok_or(FourierError::NumericalFailure("missing Hurst weights"))?;
                let Kernel::Power { interior, .. } = &self.kernel else {
                    return Err(FourierError::InvalidInput(
                        "Hurst risk requires Rough Heston",
                    ));
                };
                // Differentiate ALL weights, including the newest-cell endpoint.
                let mut dpast = c * w.initial[n];
                for j in 1..n {
                    dpast =
                        dpast + hurst_history[j] * interior[n - j] + history[j] * w.interior[n - j];
                }
                let slope = b + root * (2.0 * a);
                let denominator = C::ONE - slope * end;
                if !denominator.is_finite()
                    || denominator.abs() <= 64.0 * f64::EPSILON * (1.0 + (slope * end).abs())
                {
                    return Err(FourierError::NumericalFailure("singular Hurst tangent"));
                }
                let next_hpsi = (dpast + next_f * w.endpoint) / denominator;
                let next_hf = slope * next_hpsi;
                let residual = next_hpsi - dpast - next_f * w.endpoint - next_hf * end;
                if !next_hpsi.is_finite()
                    || !next_hf.is_finite()
                    || residual.abs() > 1e-9 * (1.0 + next_hpsi.abs() + dpast.abs())
                {
                    return Err(FourierError::NumericalFailure(
                        "Hurst tangent residual/non-finite",
                    ));
                }
                // v0, kappa and theta are fixed: only R and psi depend on H.
                hurst_exponent = hurst_exponent
                    + ((hurst_f + next_hf) * p.initial_variance()
                        + (hurst_psi + next_hpsi) * (p.mean_reversion() * p.long_run_variance()))
                        * (0.5 * dt);
                hurst_psi = next_hpsi;
                hurst_f = next_hf;
                hurst_history.push(next_hf);
            }
            if let Kernel::Lift {
                decay, left, right, ..
            } = &self.kernel
            {
                for k in 0..factors.len() {
                    factors[k] = factors[k] * decay[k] + f * left[k] + next_f * right[k];
                }
            }
            // log M = v0 * integral R + kappa*theta * integral psi.
            // The v0 contribution is NOT v0*psi(T) for H<1/2.
            exponent = exponent
                + ((f + next_f) * p.initial_variance()
                    + (psi + root) * (p.mean_reversion() * p.long_run_variance()))
                    * (0.5 * dt);
            psi = root;
            f = next_f;
            if matches!(self.kernel, Kernel::Power { .. }) {
                history.push(f);
            }
        }
        // A necessary moment-strip check, not a convergence certificate.
        if !exponent.is_finite() || exponent.re > 1e-8 {
            return Err(FourierError::NumericalFailure(
                "transform moment-strip check; refine grid",
            ));
        }
        if RISK && dexponent.iter().any(|x| !x.is_finite()) {
            return Err(FourierError::NumericalFailure(
                "non-finite log-transform derivative",
            ));
        }
        if HURST {
            if !hurst_exponent.is_finite() {
                return Err(FourierError::NumericalFailure(
                    "non-finite Hurst log-transform derivative",
                ));
            }
            dexponent[0] = hurst_exponent;
        }
        Ok((exponent, dexponent))
    }
}

/// Analytic H derivatives of the mathematical product-integration weights.
/// Stored separately so existing price/fixed-parameter plans do not allocate them.
#[derive(Clone, Debug)]
pub(super) struct PowerHurstWeights {
    initial: Vec<f64>,
    interior: Vec<f64>,
    endpoint: f64,
}
impl PowerHurstWeights {
    fn new(h: f64, t: f64, n: usize) -> Result<Self, FourierError> {
        let mut result = Self {
            initial: vec![0.0; n + 1],
            interior: vec![0.0; n + 1],
            endpoint: 0.0,
        };
        if t == 0.0 {
            return Ok(result);
        }
        let alpha = h + 0.5;
        let beta = alpha + 1.0;
        let dt = t / n as f64;
        if dt <= 0.0 {
            return Err(FourierError::NumericalFailure("Hurst time step underflow"));
        }
        let gamma =
            gamma_half_to_two(alpha).ok_or(FourierError::NumericalFailure("Hurst Gamma"))?;
        let psi =
            digamma_half_to_two(alpha).ok_or(FourierError::NumericalFailure("Hurst digamma"))?;
        let endpoint = dt.powf(alpha) / (alpha * beta * gamma);
        let log_derivative = dt.ln() - psi - 1.0 / alpha - 1.0 / beta;
        result.endpoint = endpoint * log_derivative;
        for l in 1..=n {
            let (a, da, b, db) = if l == 1 {
                let power = 2.0_f64.powf(beta);
                (alpha, 1.0, power - 2.0, power * 2.0_f64.ln())
            } else {
                // Binomial tails of (1 +/- 1/l)^beta. Start at k=2 rather
                // than subtract large near-equal powers. Derivative recurrence
                // does not divide by beta-k: remains valid at beta=2 (H=.5).
                // 64 terms at |1/l|<=.5; no data-dependent convergence exit.
                let u = 1.0 / l as f64;
                let mut c = beta;
                let mut dc = 1.0;
                let mut power = u;
                let (mut a, mut da, mut b, mut db) = (0.0, 0.0, 0.0, 0.0);
                for k in 2..=64 {
                    let factor = (beta - k as f64 + 1.0) / k as f64;
                    dc = dc * factor + c / k as f64;
                    c *= factor;
                    power *= u;
                    let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
                    a += sign * c * power;
                    da += sign * dc * power;
                    if k % 2 == 0 {
                        b += 2.0 * c * power;
                        db += 2.0 * dc * power;
                    }
                }
                let x = l as f64;
                let xp = x.powf(beta);
                (
                    xp * a,
                    xp * (da + a * x.ln()),
                    xp * b,
                    xp * (db + b * x.ln()),
                )
            };
            result.initial[l] = endpoint * (log_derivative * a + da);
            result.interior[l] = endpoint * (log_derivative * b + db);
        }
        if result
            .initial
            .iter()
            .chain(result.interior.iter())
            .chain([&result.endpoint])
            .any(|x| !x.is_finite())
        {
            return Err(FourierError::NumericalFailure("non-finite Hurst weights"));
        }
        Ok(result)
    }
}

// Exact left/right linear-interpolant integrals on [0,1] against exp(-r*t).
fn exponential_weights(r: f64) -> (f64, f64) {
    if r < 0.01 {
        let mut term = 1.0;
        let mut a = 0.0;
        let mut b = 0.0;
        for k in 0..12 {
            a += term / (k + 2) as f64;
            b += term / ((k + 1) * (k + 2)) as f64;
            term *= -r / (k + 1) as f64;
        }
        (a, b)
    } else {
        let e = (-r).exp();
        (
            ((-(-r).exp_m1()) / r - e) / r,
            (1.0 + (-r).exp_m1() / r) / r,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn hurst_hat_weights_match_independent_improper_integrals() {
        let reference: serde_json::Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/rough-volatility/hurst-risk.json"
        )))
        .unwrap();
        for row in reference["kernels"].as_array().unwrap() {
            let n = row["steps"].as_u64().unwrap() as usize;
            let l = row["lag"].as_u64().unwrap() as usize;
            let w = PowerHurstWeights::new(row["hurst"].as_f64().unwrap(), 1.0, n).unwrap();
            for (actual, expected) in [w.initial[l], w.interior[l], w.endpoint]
                .iter()
                .zip(row["derivatives"].as_array().unwrap())
            {
                assert!(
                    (actual - expected.as_f64().unwrap()).abs() < 3e-13,
                    "{row}: {actual}"
                );
            }
        }
        for h in [0.01, 0.1, 0.3, 0.5] {
            let w = PowerHurstWeights::new(h, 1.0, 8192).unwrap();
            for n in [1, 2, 128, 8192] {
                let alpha = h + 0.5;
                let t = n as f64 / 8192.0;
                let mass = t.powf(alpha) / (alpha * gamma_half_to_two(alpha).unwrap());
                let derivative =
                    mass * (t.ln() - digamma_half_to_two(alpha).unwrap() - 1.0 / alpha);
                let actual = w.initial[n] + w.endpoint + w.interior[1..n].iter().sum::<f64>();
                assert!(
                    (actual - derivative).abs() < 3e-12,
                    "{h} {n}: {actual} {derivative}"
                );
            }
        }
    }
    #[test]
    fn exponential_hat_weights_and_power_mass() {
        for r in [0.0, 1e-9, 0.009, 0.01, 1.0, 100.0, 1e8] {
            let (a, b) = exponential_weights(r);
            let mass = if r == 0.0 { 1.0 } else { -(-r).exp_m1() / r };
            assert!(a >= 0.0 && b >= 0.0);
            assert!((a + b - mass).abs() < 2e-14 * mass);
        }
        for h in [0.01, 0.1, 0.3, 0.5] {
            let p = RoughHeston::new(h, 0.04, 0.7, 0.05, 0.2, -0.6).unwrap();
            let plan = RiccatiPlan::new(
                &p.into(),
                1.0,
                HestonFourierConfig::new(128, 128, 64.0).unwrap(),
            )
            .unwrap();
            if let Kernel::Power {
                initial,
                interior,
                endpoint,
            } = plan.kernel
            {
                for n in [1, 2, 8, 64, 128] {
                    let mass = initial[n] + interior[1..n].iter().sum::<f64>() + endpoint;
                    let target = (n as f64 / 128.0).powf(h + 0.5)
                        / ((h + 0.5) * gamma_half_to_two(h + 0.5).unwrap());
                    assert!((mass - target).abs() < 2e-12);
                }
            }
        }
    }
}
