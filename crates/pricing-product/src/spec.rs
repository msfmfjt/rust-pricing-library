use pricing_core::{CoreError, CurrencyId, Date, PositiveF64, UnderlyingId};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum OptionSide {
    Call,
    Put,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EuropeanVanillaSpec {
    underlying: UnderlyingId,
    currency: CurrencyId,
    expiry: Date,
    strike: PositiveF64,
    notional: PositiveF64,
    side: OptionSide,
}

impl EuropeanVanillaSpec {
    pub fn new(
        underlying: UnderlyingId,
        currency: CurrencyId,
        expiry: Date,
        strike: f64,
        notional: f64,
        side: OptionSide,
    ) -> Result<Self, CoreError> {
        Ok(Self {
            underlying,
            currency,
            expiry,
            strike: PositiveF64::new(strike, "strike")?,
            notional: PositiveF64::new(notional, "notional")?,
            side,
        })
    }

    #[must_use]
    pub const fn underlying(&self) -> UnderlyingId {
        self.underlying
    }

    #[must_use]
    pub const fn currency(&self) -> CurrencyId {
        self.currency
    }

    #[must_use]
    pub const fn expiry(&self) -> Date {
        self.expiry
    }

    #[must_use]
    pub const fn strike(&self) -> PositiveF64 {
        self.strike
    }

    #[must_use]
    pub const fn notional(&self) -> PositiveF64 {
        self.notional
    }

    #[must_use]
    pub const fn side(&self) -> OptionSide {
        self.side
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ProductSpec {
    EuropeanVanilla(EuropeanVanillaSpec),
}

impl ProductSpec {
    #[must_use]
    pub const fn underlying(&self) -> UnderlyingId {
        match self {
            Self::EuropeanVanilla(spec) => spec.underlying(),
        }
    }

    #[must_use]
    pub const fn currency(&self) -> CurrencyId {
        match self {
            Self::EuropeanVanilla(spec) => spec.currency(),
        }
    }

    #[must_use]
    pub const fn expiry(&self) -> Date {
        match self {
            Self::EuropeanVanilla(spec) => spec.expiry(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn european_contract_requires_positive_strike_and_notional() {
        let expiry = "2027-09-04".parse().expect("valid date");
        let valid = EuropeanVanillaSpec::new(
            UnderlyingId::new(1),
            CurrencyId::new(2),
            expiry,
            100.0,
            1_000_000.0,
            OptionSide::Call,
        )
        .expect("valid contract");
        assert_eq!(valid.strike().get(), 100.0);
        assert_eq!(valid.side(), OptionSide::Call);
        assert!(EuropeanVanillaSpec::new(
            UnderlyingId::new(1),
            CurrencyId::new(2),
            expiry,
            0.0,
            1.0,
            OptionSide::Put,
        )
        .is_err());
    }
}
