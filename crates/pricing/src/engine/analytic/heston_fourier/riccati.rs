//! Implicit linear product integration for the fractional Riccati equation,
//! and exponential linear product integration for the finite-factor ODEs.
//! Both integrate the SAME continuous-time affine model as their path counterpart,
//! but neither reproduces its finite-grid full-truncation Monte Carlo law.

use super::{FourierError, HestonFourierConfig};
use crate::rough_volatility::{RoughHeston, RoughVolatilityModel};
use pricing_numerics::{Complex64 as C, gamma_half_to_two};

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
    pub(super) fn control_variance(&self) -> f64 {
        self.parameters.initial_variance() * self.maturity
    }
    pub(super) fn log_transform(&self, z: C) -> Result<C, FourierError> {
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
            return Ok(C::ZERO);
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
        Ok(exponent)
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
