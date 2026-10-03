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
