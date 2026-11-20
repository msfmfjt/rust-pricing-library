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
    if sd <= 1.0 / 256.0 {
        return intrinsic + small_time_value(forward.min(strike), log_fk.abs(), sd);
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
fn small_time_value(low: f64, abs_log_fk: f64, sd: f64) -> f64 {
    let q = abs_log_fk / sd;
    if q >= 56.0 {
        // Even MAX * sd * phi(56) * exp(56*sd/2) rounds to zero.
        return 0.0;
    }
    let q2 = q * q;
    let h = sd * sd / 8.0;
    // J_n = integral_0^1 u^(2n)*phi(q/u)/phi(q) du.
    // J_0 = 1 - q*Phi(-q)/phi(q). Use the Laplace continued fraction for
    // the Mills ratio in the tails to avoid subtracting near equals (DLMF 7.9.1).
    let j0 = if q < 2.0 {
        1.0 - q * standard_normal_cdf(-q) / standard_normal_pdf(q)
    } else {
        let mut tail = 0.0;
        for n in (1..=128).rev() {
            tail = f64::from(n) / (q + tail);
        }
        tail / (q + tail)
    };
    let j1 = (1.0 - q2 * j0) / 3.0;
    let j2 = (1.0 - q2 * j1) / 5.0;
    let weight = j0 - h * (j1 - 0.5 * h * j2);
    // Split the Gaussian exponential so a large forward can rescue a
    // representable result even when phi(q) alone would underflow.
    let half_exp = (0.25 * (abs_log_fk - q2)).exp();
    (low * sd) * half_exp * (standard_normal_pdf(0.0) * weight) * half_exp
}
