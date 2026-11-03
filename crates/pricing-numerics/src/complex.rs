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
        let scale = b.re.abs().max(b.im.abs());
        let x = b.re / scale;
        let y = b.im / scale;
        let d = x * x + y * y;
        let a = self / scale;
        Self::new((a.re * x + a.im * y) / d, (a.im * x - a.re * y) / d)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arithmetic_matches_independent_cartesian_values() {
        let a = Complex64::new(3.0, 4.0);
        let b = Complex64::new(-2.0, 5.0);
        assert_eq!(a.abs(), 5.0);
        assert_eq!(a + b, Complex64::new(1.0, 9.0));
        assert_eq!(a - b, Complex64::new(5.0, -1.0));
        assert_eq!(-a, Complex64::new(-3.0, -4.0));
        assert_eq!(a.conj(), Complex64::new(3.0, -4.0));
        assert_eq!(a * b, Complex64::new(-26.0, 7.0));
        assert_eq!(a * -2.0, Complex64::new(-6.0, -8.0));
        assert_eq!(a / 2.0, Complex64::new(1.5, 2.0));
        assert_eq!(
            Complex64::new(8.0, 6.0) / Complex64::new(2.0, -1.0),
            Complex64::new(2.0, 4.0)
        );
        let exponential = Complex64::new(2.0_f64.ln(), std::f64::consts::FRAC_PI_2).exp();
        assert!(exponential.re.abs() < 5e-16);
        assert!((exponential.im - 2.0).abs() < 5e-16);
    }

    #[test]
    fn finiteness_requires_both_components_to_be_finite() {
        assert!(Complex64::ZERO.is_finite());
        assert!(Complex64::new(f64::MAX, -f64::MAX).is_finite());
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(!Complex64::new(invalid, 1.0).is_finite());
            assert!(!Complex64::new(1.0, invalid).is_finite());
            assert!(!Complex64::new(invalid, invalid).is_finite());
        }
    }

    #[test]
    fn principal_roots_choose_the_correct_branch_and_preserve_signed_zero() {
        for (z, expected) in [
            (Complex64::new(3.0, 4.0), Complex64::new(2.0, 1.0)),
            (Complex64::new(3.0, -4.0), Complex64::new(2.0, -1.0)),
            (Complex64::new(-3.0, 4.0), Complex64::new(1.0, 2.0)),
            (Complex64::new(-3.0, -4.0), Complex64::new(1.0, -2.0)),
        ] {
            let actual = z.sqrt();
            assert!((actual.re - expected.re).abs() < 5e-16);
            assert!((actual.im - expected.im).abs() < 5e-16);
        }
        for imaginary in [0.0, -0.0] {
            let zero = Complex64::new(0.0, imaginary).sqrt();
            assert_eq!(zero.re.to_bits(), 0.0_f64.to_bits());
            assert_eq!(zero.im.to_bits(), imaginary.to_bits());
            let negative_real = Complex64::new(-4.0, imaginary).sqrt();
            assert_eq!(negative_real.re.to_bits(), 0.0_f64.to_bits());
            assert_eq!(
                negative_real.im.to_bits(),
                2.0_f64.copysign(imaginary).to_bits()
            );
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
