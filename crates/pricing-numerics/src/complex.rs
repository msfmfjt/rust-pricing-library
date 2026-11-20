//! Small double-precision complex arithmetic for deterministic numerical solvers.

use std::ops::{Add, Div, Mul, Neg, Sub};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Complex64 {
    pub re: f64,
    pub im: f64,
}
impl Complex64 {
    pub const ZERO: Self = Self::new(0.0, 0.0);
    pub const ONE: Self = Self::new(1.0, 0.0);
    #[must_use]
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }
    #[must_use]
    pub fn abs(self) -> f64 {
        self.re.hypot(self.im)
    }
    #[must_use]
    pub fn is_finite(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }
    #[must_use]
    pub fn conj(self) -> Self {
        Self::new(self.re, -self.im)
    }
    #[must_use]
    pub fn exp(self) -> Self {
        let (s, c) = self.im.sin_cos();
        Self::new(self.re.exp() * c, self.re.exp() * s)
    }
    /// Principal square root. Scaling avoids intermediate overflow for finite inputs.
    #[must_use]
    pub fn sqrt(self) -> Self {
        let scale = self.re.abs().max(self.im.abs());
        if scale == 0.0 {
            return Self::new(0.0, self.im);
        }
        let x = self.re / scale;
        let y = self.im / scale;
        let t = ((x.abs() + x.hypot(y)) * 0.5).sqrt() * scale.sqrt();
        if self.re >= 0.0 {
            Self::new(t, 0.5 * (self.im / t))
        } else {
            Self::new(0.5 * (self.im.abs() / t), t.copysign(self.im))
        }
    }
}
impl Add for Complex64 {
    type Output = Self;
    fn add(self, b: Self) -> Self {
        Self::new(self.re + b.re, self.im + b.im)
    }
}
impl Sub for Complex64 {
    type Output = Self;
    fn sub(self, b: Self) -> Self {
        Self::new(self.re - b.re, self.im - b.im)
    }
}
impl Neg for Complex64 {
    type Output = Self;
    fn neg(self) -> Self {
        Self::new(-self.re, -self.im)
    }
}
impl Mul for Complex64 {
    type Output = Self;
    fn mul(self, b: Self) -> Self {
        Self::new(
            self.re * b.re - self.im * b.im,
            self.re * b.im + self.im * b.re,
        )
    }
}
impl Mul<f64> for Complex64 {
    type Output = Self;
    fn mul(self, b: f64) -> Self {
        Self::new(self.re * b, self.im * b)
    }
}
impl Div<f64> for Complex64 {
    type Output = Self;
    fn div(self, b: f64) -> Self {
        Self::new(self.re / b, self.im / b)
    }
}
impl Div for Complex64 {
    type Output = Self;
    fn div(self, b: Self) -> Self {
        if !self.is_finite() || !b.is_finite() || b == Self::ZERO {
            // Preserve the IEEE propagation of the original scaled formula.
            let scale = b.re.abs().max(b.im.abs());
            let x = b.re / scale;
            let y = b.im / scale;
            let d = x * x + y * y;
            let a = self / scale;
            return Self::new((a.re * x + a.im * y) / d, (a.im * x - a.re * y) / d);
        }
        if b.im == 0.0 {
            return self / b.re;
        }
        if b.re == 0.0 {
            return Self::new(self.im / b.im, -self.re / b.im);
        }
        let (a, v, c, d) = (self.re, self.im, b.re, b.im);
        // In this range all nonzero products, cancellation residuals and the
        // normalized quotients fit in binary64 without intermediate scaling.
        let moderate =
            |x: f64| x == 0.0 || (power_of_two(-200)..=power_of_two(200)).contains(&x.abs());
        if [a, v, c, d].into_iter().all(moderate) {
            let denom = c.mul_add(c, d * d);
            return Self::new(dot(a, c, v, d) / denom, dot(v, c, -a, d) / denom);
        }
        let denominator = scaled_dot(c, c, d, d);
        Self::new(
            scaled_dot(a, c, v, d).quotient(denominator),
            scaled_dot(v, c, -a, d).quotient(denominator),
        )
    }
}

// Recover the low part of the rounded product before a possible cancellation.
fn dot(a: f64, b: f64, c: f64, d: f64) -> f64 {
    let cd = c * d;
    a.mul_add(b, cd) + c.mul_add(d, -cd)
}

const fn power_of_two(exponent: i32) -> f64 {
    f64::from_bits(((exponent + 1023) as u64) << 52)
}

// x = mantissa * 2^exponent, with 1 <= |mantissa| < 2 for finite nonzero x.
fn split_binary(mut x: f64) -> (f64, i32) {
    if x == 0.0 {
        return (x, 0);
    }
    let mut adjustment = 0;
    if x.abs() < f64::MIN_POSITIVE {
        x *= power_of_two(54);
        adjustment = -54;
    }
    let bits = x.to_bits();
    let exponent = ((bits >> 52) & 0x7ff) as i32 - 1023 + adjustment;
    let mantissa = (bits & !(0x7ff_u64 << 52)) | (1023_u64 << 52);
    (f64::from_bits(mantissa), exponent)
}

#[derive(Clone, Copy)]
struct ScaledDot {
    high: f64,
    low: f64,
    exponent: i32,
}

impl ScaledDot {
    fn quotient(self, divisor: Self) -> f64 {
        let q = self.high / divisor.high;
        let residual = (-q).mul_add(divisor.high, self.high) + (self.low - q * divisor.low);
        scale_binary(
            q + residual / divisor.high,
            self.exponent - divisor.exponent,
        )
    }
}

// Error-free transform for the sum of two finite, range-normalized numbers.
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let sum = a + b;
    let b_rounded = sum - a;
    (sum, (a - (sum - b_rounded)) + (b - b_rounded))
}

// Align the products, rather than scaling the entire numerator or denominator.
// This retains a small component even when the other quotient component overflows.
fn scaled_dot(a: f64, b: f64, c: f64, d: f64) -> ScaledDot {
    let (a, ea) = split_binary(a);
    let (b, eb) = split_binary(b);
    let (c, ec) = split_binary(c);
    let (d, ed) = split_binary(d);
    let (e1, e2) = (ea + eb, ec + ed);
    let exponent = if a == 0.0 || b == 0.0 {
        e2
    } else if c == 0.0 || d == 0.0 {
        e1
    } else {
        e1.max(e2)
    };
    // Differences beyond 120 bits cannot cancel a leading 106-bit product.
    let align = |x: f64, e: i32| {
        if x == 0.0 || e - exponent < -120 {
            0.0
        } else {
            x * power_of_two(e - exponent)
        }
    };
    let a = if b == 0.0 { 0.0 } else { align(a, e1) };
    let c = if d == 0.0 { 0.0 } else { align(c, e2) };
    let p = a * b;
    let q = c * d;
    let (sum, error) = two_sum(p, q);
    let (high, low) = two_sum(sum, error + (a.mul_add(b, -p) + c.mul_add(d, -q)));
    ScaledDot {
        high,
        low,
        exponent,
    }
}

fn scale_binary(value: f64, shift: i32) -> f64 {
    if value == 0.0 {
        return value;
    }
    let (mantissa, exponent) = split_binary(value);
    match exponent + shift {
        e if e > 1023 => f64::INFINITY.copysign(mantissa),
        e if e >= -1022 => mantissa * power_of_two(e),
        e if e >= -1075 => (mantissa * f64::MIN_POSITIVE) * power_of_two(e + 1022),
        _ => 0.0_f64.copysign(mantissa),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn division_recovers_representable_components_across_the_exponent_range() {
        let z = Complex64::new;
        let max = f64::MAX;
        let tiny = f64::from_bits(1);
        let large = 2.0_f64.powi(1000);
        for (a, b, expected) in [
            (z(max, max), z(1.0, 1.0), z(max, 0.0)),
            (z(max, -max), z(1.0, 1.0), z(0.0, -max)),
            (z(tiny, tiny), z(tiny, tiny), z(1.0, 0.0)),
            (z(max, tiny), z(1.0, 0.0), z(max, tiny)),
            (z(max, tiny), z(0.0, 1.0), z(tiny, -max)),
            (
                z(large, 0.0),
                z(large, 2.0_f64.powi(-70)),
                z(1.0, -f64::from_bits(16)),
            ),
            (z(max, max), z(0.5, 0.5), z(f64::INFINITY, 0.0)),
            (z(max, max), z(max, max), z(1.0, 0.0)),
        ] {
            assert_eq!(a / b, expected, "{a:?} / {b:?}");
        }
    }

    #[test]
    fn roots_division_and_conjugation() {
        for x in [-1e200, -3.0, 0.0, 3.0, 1e200] {
            for y in [-1e200, -2.0, 0.0, 2.0, 1e200] {
                let z = Complex64::new(x, y);
                let r = z.sqrt();
                assert!(r.is_finite() && r.re >= 0.0);
                assert!((r * r - z).abs() <= 1e-14 * z.abs().max(1.0));
                if z.abs() > 0.0 {
                    assert!((z / z - Complex64::ONE).abs() < 1e-15);
                }
                assert_eq!(z.conj().conj(), z);
            }
        }
        let z = Complex64::new(-2.0, 0.7);
        assert!((z.exp() * (-z).exp() - Complex64::ONE).abs() < 1e-15);
        assert_eq!(Complex64::new(-4.0, -0.0).sqrt(), Complex64::new(0.0, -2.0));
    }
}
