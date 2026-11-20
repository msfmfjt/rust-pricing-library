//! Undiscounted Black option value for positive forward and strike.

use pricing_numerics::{standard_normal_cdf, standard_normal_pdf};

pub(crate) fn log_forward_over_strike(forward: f64, strike: f64) -> f64 {
    let relative = (forward - strike) / strike;
    if relative.abs() < 0.5 {
        return relative.ln_1p();
    }
    let ratio = forward / strike;
    if ratio.is_finite() && ratio > 0.0 {
        ratio.ln()
    } else {
        forward.ln() - strike.ln()
    }
}

pub(crate) fn black_price(forward: f64, strike: f64, log_fk: f64, sd: f64, call: bool) -> f64 {
    let intrinsic = if call {
        forward - strike
    } else {
        strike - forward
    }
    .max(0.0);
    if sd == 0.0 {
        return intrinsic;
    }
    let abs_log_fk = log_fk.abs();
    let q = abs_log_fk / sd;
    let near = sd <= 1.0 / 256.0;
    let lower = q - 0.5 * sd;
    if lower >= 7.0 || (near && lower >= 2.0) {
        return intrinsic + tail_time_value(forward.min(strike), lower, sd);
    }
    if near {
        return intrinsic + small_time_value(forward.min(strike), abs_log_fk, q, sd);
    }
    let d1 = log_fk / sd + 0.5 * sd;
    let d2 = d1 - sd;
    if call {
        forward * standard_normal_cdf(d1) - strike * standard_normal_cdf(d2)
    } else {
        strike * standard_normal_cdf(-d2) - forward * standard_normal_cdf(-d1)
    }
}

// Integrate Black vega from zero to sd, writing u for the fraction of sd.
// The common time value is sqrt(F*K)*sd*integral_0^1 phi(q/u)*exp(-h*u^2) du.
// Keeping three exponential terms has relative truncation error <= exp(h)*h^3/6.
fn small_time_value(low: f64, abs_log_fk: f64, q: f64, sd: f64) -> f64 {
    let q2 = q * q;
    let h = sd * sd / 8.0;
    // J_n = integral_0^1 u^(2n)*phi(q/u)/phi(q) du.
    let j0 = 1.0 - q * standard_normal_cdf(-q) / standard_normal_pdf(q);
    let j1 = (1.0 - q2 * j0) / 3.0;
    let j2 = (1.0 - q2 * j1) / 5.0;
    let weight = j0 - h * (j1 - 0.5 * h * j2);
    (low * sd) * (0.5 * (abs_log_fk - q2)).exp() * (standard_normal_pdf(0.0) * weight)
}

// For the OTM side let a = |log(F/K)|/sd - sd/2 and b = a+sd.
// Time value = min(F,K)*phi(a)*(R(a)-R(b)), with R(x)=Phi(-x)/phi(x).
// Propagate the divided difference through Laplace's continued fraction
// (DLMF 7.9.1), so even a+sd == a in binary64 does not erase time value.
fn tail_time_value(low: f64, a: f64, sd: f64) -> f64 {
    if a >= 56.0 {
        // Even MAX * Phi(-56) is below half the minimum binary64 subnormal.
        return 0.0;
    }
    let b = a + sd;
    let (mut left, mut right, mut difference) = (0.0, 0.0, 0.0);
    let terms = if a >= 7.0 { 32 } else { 128 };
    for n in (1..=terms).rev() {
        let da = a + left;
        let db = b + right;
        let n = f64::from(n);
        difference = n * (1.0 - difference) / (da * db);
        left = n / da;
        right = n / db;
    }
    let slope = (1.0 - difference) / ((a + left) * (b + right));
    // A large forward may keep the price representable after phi(a) underflows.
    let half_exp = (-0.25 * a * a).exp();
    (low * sd) * half_exp * (standard_normal_pdf(0.0) * slope) * half_exp
}
