use pricing_core::CurrencyId;

use crate::EquityForward;

#[derive(Clone, Debug, PartialEq)]
pub struct EquityMarket {
    currency: CurrencyId,
    forward: EquityForward,
}

#[derive(Clone, Debug, PartialEq)]
pub enum MarketContext {
    Equity(EquityMarket),
}

impl MarketContext {
    #[must_use]
    pub const fn currency(&self) -> CurrencyId {
        match self {
            Self::Equity(market) => market.currency(),
        }
    }

    #[must_use]
    pub const fn equity(&self) -> &EquityMarket {
        match self {
            Self::Equity(market) => market,
        }
    }
}

impl EquityMarket {
    #[must_use]
    pub const fn new(currency: CurrencyId, forward: EquityForward) -> Self {
        Self { currency, forward }
    }

    #[must_use]
    pub const fn currency(&self) -> CurrencyId {
        self.currency
    }

    #[must_use]
    pub const fn forward(&self) -> &EquityForward {
        &self.forward
    }
}
