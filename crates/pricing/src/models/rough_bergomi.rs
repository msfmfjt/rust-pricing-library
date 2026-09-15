//! Riemann--Liouville rough Bergomi kernel and its joint near-cell/HW law.

use crate::models::hull_white::{b, hw_valid};
use crate::models::{HullWhite1Factor, HullWhiteError, HybridCorrelation};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RoughBergomi {
    hurst: f64,
    vol_of_vol: f64,
    correlation: f64,
}
impl RoughBergomi {
    /// eta is the coefficient in log *variance*: v=xi0 exp(eta X-eta^2 Var(X)/2).
    /// H=1/2 is included as the Brownian boundary; rough inputs have 0<H<1/2.
    pub fn new(hurst: f64, vol_of_vol: f64, correlation: f64) -> Result<Self, HullWhiteError> {
        if !hurst.is_finite() || hurst <= 0.0 || hurst > 0.5 {
            return Err(invalid("rough_hurst"));
        }
        hw_valid(vol_of_vol, "rough_vol_of_vol", 0, true)?;
        if !correlation.is_finite() || correlation.abs() > 1.0 {
            return Err(invalid("rough_correlation"));
        }
        Ok(Self {
            hurst,
            vol_of_vol,
            correlation,
        })
    }
    #[must_use]
    pub const fn hurst(self) -> f64 {
        self.hurst
    }
    #[must_use]
    pub const fn vol_of_vol(self) -> f64 {
        self.vol_of_vol
    }
    #[must_use]
    pub const fn correlation(self) -> f64 {
        self.correlation
    }

    /// Mean of sqrt(2H)*u^(H-1/2) over a lag interval; the optimal L2 constant.
    pub fn average_kernel(self, lower: f64, upper: f64) -> Result<f64, HullWhiteError> {
        if !lower.is_finite() || lower < 0.0 || !upper.is_finite() || upper <= lower {
            return Err(invalid("rough_lag_interval"));
        }
        let p = self.hurst + 0.5;
        let value =
            (2.0 * self.hurst).sqrt() * power_difference(lower, upper, p) / (p * (upper - lower));
        hw_valid(value, "rough_kernel_weight", 0, true)?;
        Ok(value)
    }

    /// Covariance of a near-cell power integral and an OU innovation driven by
    /// the same Brownian motion. Multiply by the specified Brownian correlation
    /// for different drivers; k=0 gives the covariance with a Brownian increment.
    pub fn near_ou_covariance(self, mean_reversion: f64, dt: f64) -> Result<f64, HullWhiteError> {
        hw_valid(mean_reversion, "rough_cross_mean_reversion", 0, true)?;
        if !dt.is_finite() || dt <= 0.0 {
            return Err(invalid("rough_cross_time_step"));
        }
        if self.hurst == 0.5 {
            return Ok(b(mean_reversion, dt));
        }
        let p = self.hurst + 0.5;
        let scale = (2.0 * self.hurst).sqrt();
        let value = if mean_reversion == 0.0 {
            scale * dt.powf(p) / p
        } else {
            scale * weighted_rate_integrals(p, mean_reversion, 0.0, dt)?.0
        };
        hw_valid(value, "rough_near_ou_covariance", 0, true)?;
        Ok(value)
    }

    /// Same-cell power integrals with different H, before multiplying by the
    /// correlation of their Brownian drivers. No Markovian approximation is used.
    pub fn near_near_covariance(self, other: Self, dt: f64) -> Result<f64, HullWhiteError> {
        if !dt.is_finite() || dt <= 0.0 {
            return Err(invalid("rough_cross_time_step"));
        }
        let value = if self.hurst == other.hurst {
            dt.powf(2.0 * self.hurst)
        } else {
            let ratio = self.hurst.min(other.hurst) / self.hurst.max(other.hurst);
            2.0 * ratio.sqrt() / (1.0 + ratio) * dt.powf(self.hurst + other.hurst)
        };
        hw_valid(value, "rough_near_near_covariance", 0, true)?;
        Ok(value)
    }

    /// Variables: (dW_S, dW_vol, OU_rate, integrated_OU_rate, J_near), where
    /// J_near=int_start^end sqrt(2H)*(end-s)^(H-1/2) dW_vol(s).
    /// All rate volatility breakpoints inside the interval are included.
    pub fn hybrid_covariance(
        self,
        rates: &HullWhite1Factor,
        start: f64,
        end: f64,
        correlation: HybridCorrelation,
    ) -> Result<[[f64; 5]; 5], HullWhiteError> {
        let base = rates.transition(start, end, 0.0, correlation)?;
        if end <= start || correlation.equity_vol != self.correlation {
            return Err(invalid("rough_transition_interval_or_correlation"));
        }
        let h = end - start;
        let mut c = [[0.0; 5]; 5];
        for (row, old) in c.iter_mut().zip(base.covariance) {
            row[..4].copy_from_slice(&old);
        }
        if self.hurst == 0.5 {
            c[4] = c[1];
            for row in c.iter_mut().take(4) {
                row[4] = row[1];
            }
            c[4][4] = h;
            return Ok(c);
        }
        let p = self.hurst + 0.5;
        let scale = (2.0 * self.hurst).sqrt();
        let cov = scale * h.powf(p) / p;
        c[4][0] = correlation.equity_vol * cov;
        c[4][1] = cov;
        c[4][4] = h.powf(2.0 * self.hurst);
        for (i, &left) in rates.volatility_times().iter().enumerate() {
            let lo = start.max(left);
            let hi = end.min(rates.volatility_times().get(i + 1).copied().unwrap_or(end));
            if hi <= lo {
                continue;
            }
            let (e, integral) =
                weighted_rate_integrals(p, rates.mean_reversion(), end - hi, end - lo)?;
            let loading = correlation.vol_rate * rates.volatilities()[i] * scale;
            c[4][2] += loading * e;
            c[4][3] += loading * integral;
        }
        let near = c[4];
        for (row, v) in c.iter_mut().take(4).zip(near) {
            row[4] = v;
        }
        for &v in c.iter().flatten() {
            hw_valid(v, "rough_transition_covariance", 0, false)?;
        }
        Ok(c)
    }
}

fn invalid(field: &'static str) -> HullWhiteError {
    HullWhiteError::InvalidInput { field, index: 0 }
}
fn power_difference(lo: f64, hi: f64, p: f64) -> f64 {
    if lo == 0.0 {
        hi.powf(p)
    } else {
        hi.powf(p) * (-(p * ((lo - hi) / hi).ln_1p()).exp_m1())
    }
}

// Integral of u^(p-1) times exp(-a*u) and B(a,u). A small-a power series
// avoids both the singular endpoint and cancellation in the Ho--Lee limit.
fn weighted_rate_integrals(p: f64, a: f64, lo: f64, hi: f64) -> Result<(f64, f64), HullWhiteError> {
    let z = a * hi;
    if z <= 1.0 {
        let r = lo / hi;
        let mut ce = 1.0;
        let mut cb = 1.0;
        let mut e = 0.0;
        let mut integral = 0.0;
        for n in 0..48 {
            let q = p + n as f64;
            let de = ce * power_difference(r, 1.0, q) / q;
            let db = cb * power_difference(r, 1.0, q + 1.0) / (q + 1.0);
            e += de;
            integral += db;
            if n > 2 && de.abs() + db.abs() < 1e-17 * (e.abs() + integral.abs()) {
                break;
            }
            ce *= -z / (n + 1) as f64;
            cb *= -z / (n + 2) as f64;
        }
        return Ok((hi.powf(p) * e, hi.powf(p + 1.0) * integral));
    }
    // y=(u/hi)^p removes the power singularity. Endpoint-aware quadrature
    // detects narrow large-a boundary layers; unresolved integrals are errors.
    let left = (lo / hi).powf(p);
    let exp = |y: f64| (-a * hi * y.powf(1.0 / p)).exp();
    let bond = |y: f64| b(a, hi * y.powf(1.0 / p));
    let scale = hi.powf(p) / p;
    Ok((
        scale * integrate(&exp, left, 1.0)?,
        scale * integrate(&bond, left, 1.0)?,
    ))
}
fn integrate(f: &impl Fn(f64) -> f64, lo: f64, hi: f64) -> Result<f64, HullWhiteError> {
    let values = [f(lo), f(0.5 * (lo + hi)), f(hi)];
    let tol = 2e-14 * (hi - lo) * values.iter().copied().fold(0.0, f64::max);
    adaptive(f, lo, hi, values, tol, 40)
}
fn adaptive(
    f: &impl Fn(f64) -> f64,
    lo: f64,
    hi: f64,
    v: [f64; 3],
    tol: f64,
    depth: u32,
) -> Result<f64, HullWhiteError> {
    let mid = 0.5 * (lo + hi);
    let l = f(0.5 * (lo + mid));
    let r = f(0.5 * (mid + hi));
    let coarse = (hi - lo) * (v[0] + 4.0 * v[1] + v[2]) / 6.0;
    let fine = (hi - lo) * (v[0] + 4.0 * l + 2.0 * v[1] + 4.0 * r + v[2]) / 12.0;
    if (fine - coarse).abs() <= 15.0 * tol {
        return Ok(fine + (fine - coarse) / 15.0);
    }
    if depth == 0 || mid == lo || mid == hi {
        return Err(invalid("rough_covariance_quadrature_resolution"));
    }
    Ok(adaptive(f, lo, mid, [v[0], l, v[1]], tol * 0.5, depth - 1)?
        + adaptive(f, mid, hi, [v[1], r, v[2]], tol * 0.5, depth - 1)?)
}
