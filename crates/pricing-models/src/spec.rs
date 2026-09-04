use pricing_core::{CoreError, NonNegativeF64};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BlackScholesSpec {
    volatility: NonNegativeF64,
}

impl BlackScholesSpec {
    pub fn new(volatility: f64) -> Result<Self, CoreError> {
        Ok(Self {
            volatility: NonNegativeF64::new(volatility, "volatility")?,
        })
    }

    #[must_use]
    pub const fn volatility(self) -> NonNegativeF64 {
        self.volatility
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ModelSpec {
    BlackScholes(BlackScholesSpec),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn black_scholes_allows_zero_volatility_limit() {
        assert_eq!(
            BlackScholesSpec::new(0.0)
                .expect("zero-volatility limit")
                .volatility()
                .get(),
            0.0
        );
        assert!(BlackScholesSpec::new(-0.01).is_err());
        assert!(BlackScholesSpec::new(f64::NAN).is_err());
    }
}
