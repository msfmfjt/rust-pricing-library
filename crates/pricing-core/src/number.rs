use std::fmt;

use crate::CoreError;

/// A finite IEEE-754 binary64 value. Exact bits, including negative zero, are preserved.
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct FiniteF64(f64);

impl FiniteF64 {
    pub fn new(value: f64, field: &'static str) -> Result<Self, CoreError> {
        if value.is_finite() {
            Ok(Self(value))
        } else {
            Err(CoreError::NonFiniteNumber {
                field,
                bits: value.to_bits(),
            })
        }
    }

    #[must_use]
    pub const fn get(self) -> f64 {
        self.0
    }

    #[must_use]
    pub const fn to_bits(self) -> u64 {
        self.0.to_bits()
    }
}

impl PartialEq for FiniteF64 {
    fn eq(&self, other: &Self) -> bool {
        self.to_bits() == other.to_bits()
    }
}

impl Eq for FiniteF64 {}

impl fmt::Display for FiniteF64 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

/// A finite value greater than zero.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(transparent)]
pub struct PositiveF64(FiniteF64);

impl PositiveF64 {
    pub fn new(value: f64, field: &'static str) -> Result<Self, CoreError> {
        let finite = FiniteF64::new(value, field)?;
        if value > 0.0 {
            Ok(Self(finite))
        } else {
            Err(CoreError::NumberNotPositive {
                field,
                bits: value.to_bits(),
            })
        }
    }

    #[must_use]
    pub const fn get(self) -> f64 {
        self.0.get()
    }
}

/// A finite value greater than or equal to zero.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(transparent)]
pub struct NonNegativeF64(FiniteF64);

impl NonNegativeF64 {
    pub fn new(value: f64, field: &'static str) -> Result<Self, CoreError> {
        let finite = FiniteF64::new(value, field)?;
        if value >= 0.0 {
            Ok(Self(finite))
        } else {
            Err(CoreError::NumberNegative {
                field,
                bits: value.to_bits(),
            })
        }
    }

    #[must_use]
    pub const fn get(self) -> f64 {
        self.0.get()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finite_value_preserves_signed_zero() {
        let positive = FiniteF64::new(0.0, "value").expect("finite");
        let negative = FiniteF64::new(-0.0, "value").expect("finite");
        assert_ne!(positive, negative);
        assert_eq!(negative.to_bits(), (-0.0_f64).to_bits());
    }

    #[test]
    fn wrappers_enforce_their_domains() {
        assert!(FiniteF64::new(f64::NAN, "value").is_err());
        assert!(FiniteF64::new(f64::INFINITY, "value").is_err());
        assert!(PositiveF64::new(0.0, "strike").is_err());
        assert!(NonNegativeF64::new(-0.0, "dividend").is_ok());
        assert!(NonNegativeF64::new(-1.0, "dividend").is_err());
    }
}
