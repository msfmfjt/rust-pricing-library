//! Implicit product integration of h=K*R(z,h). No MC process/RNG helpers.
use super::FourierError;
use crate::models::{RoughHeston, RoughVolatilityModel};
use pricing_numerics::{Complex64 as C, gamma_half_to_two};

type Result<T> = std::result::Result<T, FourierError>;

#[derive(Clone, Debug)]
pub(super) enum Kernel {
    Fractional {
        left: Vec<f64>,
        interior: Vec<f64>,
        implicit: f64,
    },
    Lifted {
        weights: Vec<f64>,
        decay: Vec<f64>,
        left: Vec<f64>,
        right: Vec<f64>,
        implicit: f64,
    },
}

// Exact integrals of exponential kernel against left/right linear hats.
fn exponential_hats(rate: f64, dt: f64) -> (f64, f64, f64) {
    let a = rate * dt;
    if a == 0.0 {
        return (1.0, dt * 0.5, dt * 0.5);
    }
    if a < 0.01 {
        let mut term = 1.0;
        let mut left = 0.0;
        let mut right = 0.0;
        for m in 0..12 {
            left += term / (m + 2) as f64;
            right += term / ((m + 1) * (m + 2)) as f64;
            term *= -a / (m + 1) as f64;
        }
        ((-a).exp(), dt * left, dt * right)
    } else {
        let decay = (-a).exp();
        // Compute in 1/rate rather than a*a; the stiff a=+inf limit is valid.
        let left = if a.is_infinite() {
            0.0
        } else {
            ((1.0 - decay) / a - decay) / rate
        };
        let right = (1.0 - (1.0 - decay) / a) / rate;
        (decay, left, right)
    }
}

impl Kernel {
    pub(super) fn new(model: &RoughVolatilityModel, maturity: f64, steps: usize) -> Result<Self> {
        let dt = maturity / steps as f64;
        let kernel = match model {
            RoughVolatilityModel::RoughHeston(m) => {
                let alpha = m.hurst() + 0.5;
                if alpha == 1.0 {
                    return Ok(Self::Lifted {
                        weights: vec![1.0],
                        decay: vec![1.0],
                        left: vec![0.5 * dt],
                        right: vec![0.5 * dt],
                        implicit: 0.5 * dt,
                    });
                }
                let scale = dt.powf(alpha)
                    / gamma_half_to_two(alpha)
                        .ok_or(FourierError::Numerical("fractional Gamma"))?;
                let mut left = Vec::with_capacity(steps);
                let mut right = Vec::with_capacity(steps);
                for lag in 0..steps {
                    let r = lag as f64;
                    let (d0, d1) = if lag == 0 {
                        (1.0 / alpha, 1.0 / (alpha + 1.0))
                    } else {
                        let log = (1.0 / r).ln_1p();
                        (
                            r.powf(alpha) * (alpha * log).exp_m1() / alpha,
                            r.powf(alpha + 1.0) * ((alpha + 1.0) * log).exp_m1() / (alpha + 1.0),
                        )
                    };
                    left.push(scale * (d1 - r * d0));
                    right.push(scale * ((r + 1.0) * d0 - d1));
                }
                let interior = (1..steps).map(|lag| left[lag - 1] + right[lag]).collect();
                Self::Fractional {
                    implicit: right[0],
                    left,
                    interior,
                }
            }
            RoughVolatilityModel::LiftedHeston(m) => {
                let mut decay = Vec::new();
                let mut left = Vec::new();
                let mut right = Vec::new();
                for &rate in m.rates() {
                    let (d, l, r) = exponential_hats(rate, dt);
                    decay.push(d);
                    left.push(l);
                    right.push(r);
                }
                let implicit = m.weights().iter().zip(&right).map(|(w, r)| w * r).sum();
                Self::Lifted {
                    weights: m.weights().to_vec(),
                    decay,
                    left,
                    right,
                    implicit,
                }
            }
            _ => return Err(FourierError::UnsupportedModel),
        };
        let (coefficients, implicit) = match &kernel {
            Self::Fractional {
                left,
                interior,
                implicit,
            } => (
                left.iter().chain(interior).copied().collect::<Vec<_>>(),
                *implicit,
            ),
            Self::Lifted {
                left,
                right,
                implicit,
                ..
            } => (left.iter().chain(right).copied().collect(), *implicit),
        };
        if !implicit.is_finite()
            || implicit < 0.0
            || coefficients.iter().any(|x| !x.is_finite() || *x < 0.0)
        {
            return Err(FourierError::Numerical("kernel coefficients"));
        }
        Ok(kernel)
    }

    pub(super) fn transform(&self, m: &RoughHeston, dt: f64, steps: usize, z: C) -> Result<C> {
        if dt == 0.0 || z == C::ZERO || z == C::ONE {
            return Ok(C::ONE);
        }
        let constant = (z * z - z) * 0.5;
        let linear = z * (m.correlation() * m.vol_of_vol()) - C::new(m.mean_reversion(), 0.0);
        let quadratic = 0.5 * m.vol_of_vol() * m.vol_of_vol();
        if !constant.is_finite() || !linear.is_finite() || !quadratic.is_finite() {
            return Err(FourierError::Numerical("Riccati coefficients"));
        }
        let rhs = |h: C| constant + linear * h + (h * h) * quadratic;
        let mut q = vec![constant];
        let mut previous_h = C::ZERO;
        let mut integral_h = C::ZERO;
        let mut integral_q = C::ZERO;
        let mut factors = match self {
            Self::Lifted { weights, .. } => vec![C::ZERO; weights.len()],
            _ => Vec::new(),
        };
        for i in 1..=steps {
            let (history, b) = match self {
                Self::Fractional {
                    left,
                    interior,
                    implicit,
                } => {
                    let mut history = q[0] * left[i - 1];
                    for j in 1..i {
                        history = history + q[j] * interior[i - j - 1];
                    }
                    (history, *implicit)
                }
                Self::Lifted {
                    weights,
                    decay,
                    left,
                    implicit,
                    ..
                } => {
                    let mut history = C::ZERO;
                    for k in 0..factors.len() {
                        factors[k] = factors[k] * decay[k] + q[i - 1] * left[k];
                        history = history + factors[k] * weights[k];
                    }
                    (history, *implicit)
                }
            };
            let bb = C::ONE - linear * b;
            let cc = history + constant * b;
            let aa = b * quadratic;
            let h = if aa == 0.0 {
                cc / bb
            } else {
                let root = (bb * bb - cc * (4.0 * aa)).sqrt();
                // Select the principal (continuous small-step) solution, but
                // evaluate either algebraic form to avoid cancellation.
                if (bb + root).norm() >= (bb - root).norm() {
                    cc * 2.0 / (bb + root)
                } else {
                    (bb - root) / (2.0 * aa)
                }
            };
            let current_q = rhs(h);
            let residual = h - history - current_q * b;
            if !h.is_finite()
                || !current_q.is_finite()
                || residual.norm() > 2e-10 * (1.0 + h.norm() + history.norm())
            {
                return Err(FourierError::RiccatiFailure {
                    step: i,
                    frequency: z.im,
                });
            }
            if let Self::Lifted { right, .. } = self {
                for (factor, &r) in factors.iter_mut().zip(right) {
                    *factor = *factor + current_q * r;
                }
            }
            integral_h = integral_h + (previous_h + h) * (0.5 * dt);
            integral_q = integral_q + (q[i - 1] + current_q) * (0.5 * dt);
            previous_h = h;
            q.push(current_q);
        }
        // Fubini: log M = v0 * integral R + kappa*theta * integral h.
        // This also holds for the finite kernel (NOT a full-truncated MC law).
        let log = integral_q * m.initial_variance()
            + integral_h * (m.mean_reversion() * m.long_run_variance());
        if !log.is_finite() || log.re > 1e-8 {
            return Err(FourierError::Numerical(
                "transform outside martingale strip bound",
            ));
        }
        let value = log.exp();
        if !value.is_finite() {
            return Err(FourierError::Numerical("transform exponent"));
        }
        Ok(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn hats_integrate_constant_and_linear_functions() {
        for rate in [0.0, 1e-12, 0.01, 2.0, 1e8, 1e300] {
            let dt = 0.3;
            let (d, l, r) = exponential_hats(rate, dt);
            let integral = if rate == 0.0 {
                dt
            } else {
                -(-rate * dt).exp_m1() / rate
            };
            assert!((l + r - integral).abs() < 1e-13 * integral);
            assert!(d >= 0.0 && l >= 0.0 && r >= 0.0);
        }
        let m = RoughHeston::new(0.1, 0.04, 0.7, 0.05, 0.18, -0.6).unwrap();
        let Kernel::Fractional {
            left,
            interior,
            implicit,
        } = Kernel::new(&m.into(), 1.0, 32).unwrap()
        else {
            panic!()
        };
        for n in 1..=32 {
            let sum = left[n - 1] + interior[..n - 1].iter().sum::<f64>() + implicit;
            let expected = (n as f64 / 32.0).powf(0.6) / (0.6 * gamma_half_to_two(0.6).unwrap());
            assert!((sum - expected).abs() < 2e-12);
        }
    }
}
