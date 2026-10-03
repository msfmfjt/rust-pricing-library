//! Numerical support for fractional Gaussian models (no financial dependencies).

/// Gamma on [1/2, 2], evaluated by the Lanczos approximation.
/// The restricted domain avoids reflection, poles and overflow ambiguity.
#[must_use]
pub fn gamma_half_to_two(x: f64) -> Option<f64> {
    if !x.is_finite() || !(0.5..=2.0).contains(&x) {
        return None;
    }
    if x == 1.0 || x == 2.0 {
        return Some(1.0);
    }
    let coefficients = [
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    let z = x - 1.0;
    let mut sum = 0.999_999_999_999_809_9;
    for (i, c) in coefficients.iter().enumerate() {
        sum += c / (z + i as f64 + 1.0);
    }
    let t = z + 7.5;
    Some((2.0 * std::f64::consts::PI).sqrt() * t.powf(z + 0.5) * (-t).exp() * sum)
}

/// Digamma on [1/2, 2]. Recurrence to x>=12 followed by the Bernoulli
/// asymptotic expansion through x^-16 (NIST DLMF 5.5.2, 5.11.2).
/// Invalid arguments return None; no reflection or pole handling is implied.
#[must_use]
pub fn digamma_half_to_two(mut x: f64) -> Option<f64> {
    if !x.is_finite() || !(0.5..=2.0).contains(&x) {
        return None;
    }
    let mut shift = crate::NeumaierSum::new();
    while x < 12.0 {
        shift.add(-1.0 / x);
        x += 1.0;
    }
    let y = 1.0 / (x * x);
    let coefficients = [
        -1.0 / 12.0,
        1.0 / 120.0,
        -1.0 / 252.0,
        1.0 / 240.0,
        -1.0 / 132.0,
        691.0 / 32760.0,
        -1.0 / 12.0,
        3617.0 / 8160.0,
    ];
    let mut series = 0.0;
    for c in coefficients.iter().rev() {
        series = (series + c) * y;
    }
    Some(shift.total() + x.ln() - 0.5 / x + series)
}

/// Normalized covariance of stationary fractional OU, at dimensionless lag
/// z=kappa*|t|. Evaluates a nonoscillatory second-difference integral. Returns
/// None on invalid input or failure of the adaptive quadrature; no covariance
/// clipping or diagonal jitter is applied. Negative long-lag values are valid.
///
/// The absolute quadrature target is 2e-12 before division by 2*Gamma(2H+1).
/// The exponential tail is cut at 64; for 0<2H<=1 its absolute omitted integral
/// is bounded by 2*exp(-64)*(65), far below this target.
pub fn fractional_ou_correlation(hurst: f64, lag: f64) -> Option<f64> {
    if !hurst.is_finite() || hurst <= 0.0 || hurst > 0.5 || !lag.is_finite() || lag < 0.0 {
        return None;
    }
    if lag == 0.0 {
        return Some(1.0);
    }
    if hurst == 0.5 {
        return Some((-lag).exp());
    }
    let q = 2.0 * hurst;
    let integrand = |u: f64| {
        let difference = if u < lag {
            let ratio = u / lag;
            lag.powf(q) * ((q * ratio.ln_1p()).exp_m1() + (q * (-ratio).ln_1p()).exp_m1())
        } else {
            (lag + u).powf(q) + (u - lag).powf(q) - 2.0 * lag.powf(q)
        };
        (-u).exp() * difference
    };
    let integrate_segment = |a: f64, b: f64| {
        // sin^2 transformation regularizes the cusp and clusters endpoints.
        let transformed = |s: f64| {
            if s == 0.0 || s == 1.0 {
                return 0.0;
            }
            let angle = 0.5 * std::f64::consts::PI * s;
            let u = a + (b - a) * angle.sin().powi(2);
            let jacobian = (b - a) * 0.5 * std::f64::consts::PI * (std::f64::consts::PI * s).sin();
            integrand(u) * jacobian
        };
        adaptive_simpson(&transformed, 0.0, 1.0, 1e-12)
    };
    let integral = if lag < 64.0 {
        integrate_segment(0.0, lag)? + integrate_segment(lag, 64.0)?
    } else {
        integrate_segment(0.0, 64.0)?
    };
    let result = integral / (2.0 * gamma_half_to_two(q + 1.0)?);
    if result.is_finite() && result.abs() <= 1.0 + 1e-11 {
        Some(result)
    } else {
        None
    }
}

#[derive(Clone, Debug)]
struct SimpsonPanel {
    interval: [f64; 2],
    values: [f64; 5],
    estimate: f64,
    error: f64,
    depth: u32,
    serial: u64,
}
impl PartialEq for SimpsonPanel {
    fn eq(&self, other: &Self) -> bool {
        self.serial == other.serial && self.error.total_cmp(&other.error).is_eq()
    }
}
impl Eq for SimpsonPanel {}
impl PartialOrd for SimpsonPanel {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for SimpsonPanel {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.error
            .total_cmp(&other.error)
            .then_with(|| other.serial.cmp(&self.serial))
    }
}
impl SimpsonPanel {
    fn new(
        f: &impl Fn(f64) -> f64,
        interval: [f64; 2],
        values: [f64; 3],
        depth: u32,
        serial: u64,
    ) -> Option<Self> {
        let [a, b] = interval;
        let [fa, fm, fb] = values;
        let m = (a + b) * 0.5;
        let fl = f((a + m) * 0.5);
        let fr = f((m + b) * 0.5);
        let whole = (b - a) * (fa + 4.0 * fm + fb) / 6.0;
        let left = (m - a) * (fa + 4.0 * fl + fm) / 6.0;
        let right = (b - m) * (fm + 4.0 * fr + fb) / 6.0;
        let correction = (left + right - whole) / 15.0;
        let estimate = left + right + correction;
        if !estimate.is_finite() || !correction.is_finite() {
            return None;
        }
        Some(Self {
            interval,
            values: [fa, fl, fm, fr, fb],
            estimate,
            error: correction.abs(),
            depth,
            serial,
        })
    }
}

// Refine the panel with the largest error until the GLOBAL error estimate is
// small. A recursively halved tolerance needlessly demands sub-roundoff local
// accuracy at the fractional cusp when H is very small.
fn adaptive_simpson(f: &impl Fn(f64) -> f64, a: f64, b: f64, tolerance: f64) -> Option<f64> {
    if a == b {
        return Some(0.0);
    }
    let first = SimpsonPanel::new(f, [a, b], [f(a), f((a + b) * 0.5), f(b)], 0, 0)?;
    let mut estimate = crate::NeumaierSum::new();
    let mut error = crate::NeumaierSum::new();
    estimate.add(first.estimate);
    error.add(first.error);
    let mut heap = std::collections::BinaryHeap::from([first]);
    let mut serial = 1;
    while error.total() > tolerance {
        if heap.len() >= 16_384 {
            return None;
        }
        let panel = heap.pop()?;
        if panel.depth >= 40 {
            return None;
        }
        let [a, b] = panel.interval;
        let [fa, fl, fm, fr, fb] = panel.values;
        let middle = (a + b) * 0.5;
        if middle == a || middle == b {
            return None;
        }
        let left = SimpsonPanel::new(f, [a, middle], [fa, fl, fm], panel.depth + 1, serial)?;
        let right = SimpsonPanel::new(f, [middle, b], [fm, fr, fb], panel.depth + 1, serial + 1)?;
        serial += 2;
        estimate.add(-panel.estimate);
        estimate.add(left.estimate);
        estimate.add(right.estimate);
        error.add(-panel.error);
        error.add(left.error);
        error.add(right.error);
        heap.push(left);
        heap.push(right);
    }
    Some(estimate.total())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digamma_constants_recurrence_and_domain() {
        let euler = 0.577_215_664_901_532_9;
        for (x, expected) in [
            (0.5, -euler - 2.0 * 2.0_f64.ln()),
            (1.0, -euler),
            (1.5, 2.0 - euler - 2.0 * 2.0_f64.ln()),
            (2.0, 1.0 - euler),
        ] {
            assert!((digamma_half_to_two(x).unwrap() - expected).abs() < 3e-15);
        }
        for x in [0.51, 0.6, 0.75, 0.9] {
            assert!(
                (digamma_half_to_two(x + 1.0).unwrap() - digamma_half_to_two(x).unwrap() - 1.0 / x)
                    .abs()
                    < 3e-15
            );
        }
        for x in [0.49, 2.01, f64::NAN, f64::INFINITY] {
            assert_eq!(digamma_half_to_two(x), None);
        }
    }
    #[test]
    fn gamma_boundaries_and_recurrence() {
        assert_eq!(gamma_half_to_two(1.0), Some(1.0));
        assert_eq!(gamma_half_to_two(2.0), Some(1.0));
        let root_pi = std::f64::consts::PI.sqrt();
        assert!((gamma_half_to_two(0.5).unwrap() - root_pi).abs() < 3e-15);
        assert!((gamma_half_to_two(1.5).unwrap() - root_pi / 2.0).abs() < 3e-15);
        for x in [0.51, 0.6, 0.75, 0.9] {
            assert!(
                (gamma_half_to_two(x + 1.0).unwrap() - x * gamma_half_to_two(x).unwrap()).abs()
                    < 4e-15
            );
        }
        assert_eq!(gamma_half_to_two(f64::NAN), None);
        assert_eq!(gamma_half_to_two(0.0), None);
    }

    #[test]
    fn fractional_ou_brownian_boundary_and_negative_tail() {
        for lag in [0.0, 0.01, 0.1, 1.0, 3.0, 10.0] {
            assert_eq!(fractional_ou_correlation(0.5, lag), Some((-lag).exp()));
        }
        assert_eq!(fractional_ou_correlation(0.1, 0.0), Some(1.0));
        assert!(fractional_ou_correlation(0.1, 10.0).unwrap() < 0.0);
        assert_eq!(fractional_ou_correlation(0.0, 1.0), None);
        assert_eq!(fractional_ou_correlation(0.1, -1.0), None);
    }
}
