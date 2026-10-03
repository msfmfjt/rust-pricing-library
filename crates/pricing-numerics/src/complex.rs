//! Small complex arithmetic foundation, with scaled division and principal square root.
//! Arithmetic follows IEEE-754; callers decide how to handle non-finite results.
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
    pub fn norm(self) -> f64 {
        self.re.hypot(self.im)
    }
    #[must_use]
    pub const fn conjugate(self) -> Self {
        Self::new(self.re, -self.im)
    }
    #[must_use]
    pub fn is_finite(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }
    #[must_use]
    pub fn exp(self) -> Self {
        let scale = self.re.exp();
        let (sin, cos) = self.im.sin_cos();
        Self::new(scale * cos, scale * sin)
    }
    /// Principal root, with the sign of the imaginary part preserved on the cut.
    #[must_use]
    pub fn sqrt(self) -> Self {
        if self.re == 0.0 && self.im == 0.0 {
            return Self::new(0.0, self.im);
        }
        let t = (self.norm() * 0.5 + self.re.abs() * 0.5).sqrt();
        if self.re >= 0.0 {
            Self::new(t, self.im / (2.0 * t))
        } else {
            Self::new(self.im.abs() / (2.0 * t), t.copysign(self.im))
        }
    }
}
impl Add for Complex64 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::new(self.re + rhs.re, self.im + rhs.im)
    }
}
impl Sub for Complex64 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.re - rhs.re, self.im - rhs.im)
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
    fn mul(self, rhs: Self) -> Self {
        Self::new(
            self.re * rhs.re - self.im * rhs.im,
            self.re * rhs.im + self.im * rhs.re,
        )
    }
}
impl Mul<f64> for Complex64 {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        Self::new(self.re * rhs, self.im * rhs)
    }
}
impl Div<f64> for Complex64 {
    type Output = Self;
    fn div(self, rhs: f64) -> Self {
        Self::new(self.re / rhs, self.im / rhs)
    }
}
impl Div for Complex64 {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        // Scale both operands by the denominator's largest component. Unlike the
        // unscaled Smith denominator, this handles (1e308+i1e308)/(1e308+i1e308).
        let scale = rhs.re.abs().max(rhs.im.abs());
        let ar = self.re / scale;
        let ai = self.im / scale;
        let br = rhs.re / scale;
        let bi = rhs.im / scale;
        let denominator = br * br + bi * bi;
        Self::new(
            (ar * br + ai * bi) / denominator,
            (ai * br - ar * bi) / denominator,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::Complex64 as C;
    #[test]
    fn complex_arithmetic_and_roots() {
        for z in [
            C::new(3.0, 4.0),
            C::new(-3.0, 4.0),
            C::new(-3.0, -4.0),
            C::new(0.0, 4.0),
            C::ZERO,
        ] {
            let r = z.sqrt();
            assert!((r * r - z).norm() < 2e-14);
            assert!(r.re >= 0.0);
        }
        assert_eq!(C::new(-4.0, -0.0).sqrt(), C::new(0.0, -2.0));
        assert_eq!(C::new(1.0, 2.0) * C::new(3.0, -4.0), C::new(11.0, 2.0));
        assert!((C::new(1.0, 2.0) / C::new(3.0, -4.0) - C::new(-0.2, 0.4)).norm() < 1e-15);
        assert!((C::new(0.0, std::f64::consts::PI).exp() + C::ONE).norm() < 2e-15);
    }
    #[test]
    fn division_avoids_squared_denominator_overflow_and_underflow() {
        for x in [1e-300, 1e300, 1e308] {
            let z = C::new(x, x);
            assert_eq!(z / z, C::ONE);
        }
        let z = C::new(1e300, -1e300);
        let root = z.sqrt();
        assert!(root.is_finite());
        assert!((root * root - z).norm() / z.norm() < 1e-14);
    }
}
