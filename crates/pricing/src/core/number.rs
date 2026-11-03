use std::fmt;

use crate::core::CoreError;

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

    #[test]
    fn finite_values_round_trip_and_equality_is_reflexive() {
        let smallest = f64::from_bits(1);
        for value in [
            -f64::MAX,
            -2.5,
            -smallest,
            -0.0,
            0.0,
            smallest,
            2.5,
            f64::MAX,
        ] {
            let wrapped = FiniteF64::new(value, "value").unwrap();
            assert_eq!(wrapped.get().to_bits(), value.to_bits());
            assert_eq!(wrapped.to_bits(), value.to_bits());
            assert_eq!(wrapped, FiniteF64::new(value, "other_field").unwrap());
        }
        assert_ne!(
            FiniteF64::new(2.5, "x").unwrap(),
            FiniteF64::new(3.5, "x").unwrap()
        );
        assert_eq!(FiniteF64::new(-2.5, "x").unwrap().to_string(), "-2.5");
    }

    #[test]
    fn constrained_values_preserve_valid_inputs_and_signed_zero() {
        for value in [f64::from_bits(1), f64::MIN_POSITIVE, 2.5, f64::MAX] {
            assert_eq!(
                PositiveF64::new(value, "strike").unwrap().get().to_bits(),
                value.to_bits()
            );
            assert_eq!(
                NonNegativeF64::new(value, "cash").unwrap().get().to_bits(),
                value.to_bits()
            );
        }
        for zero in [0.0_f64, -0.0] {
            assert!(PositiveF64::new(zero, "strike").is_err());
            assert_eq!(
                NonNegativeF64::new(zero, "cash").unwrap().get().to_bits(),
                zero.to_bits()
            );
        }
        for value in [-f64::from_bits(1), -2.5, -f64::MAX] {
            assert!(PositiveF64::new(value, "strike").is_err());
            assert!(NonNegativeF64::new(value, "cash").is_err());
        }
    }

    #[test]
    fn nonfinite_inputs_preserve_error_field_and_original_bits() {
        for value in [
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::from_bits(0x7ff8_0000_0000_0042),
        ] {
            for error in [
                FiniteF64::new(value, "input").unwrap_err(),
                PositiveF64::new(value, "input").unwrap_err(),
                NonNegativeF64::new(value, "input").unwrap_err(),
            ] {
                assert!(
                    matches!(error, CoreError::NonFiniteNumber { field: "input", bits }
                    if bits == value.to_bits())
                );
            }
        }
    }
}
