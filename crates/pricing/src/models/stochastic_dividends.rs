//! Buehler's mean-reverting discrete cash-dividend factor.
//!
//! Expected cash amounts belong to the market schedule. This model specifies
//! their dynamics, not a continuous dividend yield or a repo spread.

use std::{error::Error, fmt};

pub const STOCHASTIC_DIVIDEND_SCHEME: &str = "buehler-cash-positive-split-v1";

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BuehlerDividendModel {
    pub(crate) mean_reversion: f64,
    pub(crate) equity_linkage: f64,
    pub(crate) dividend_volatility: f64,
    pub(crate) equity_dividend_correlation: f64,
}

impl BuehlerDividendModel {
    /// dY = kappa*(alpha*f + 1-alpha-Y)*dt + nu*Y*dW_D.
    /// f is a positive normalized martingale; the correlation is with dW_f,
    /// not with the reconstructed physical stock (which also depends on Y).
    pub fn new(
        mean_reversion: f64,
        equity_linkage: f64,
        dividend_volatility: f64,
        equity_dividend_correlation: f64,
    ) -> Result<Self, StochasticDividendError> {
        nonnegative(mean_reversion, "mean_reversion")?;
        nonnegative(dividend_volatility, "dividend_volatility")?;
        if !equity_linkage.is_finite() || !(0.0..=1.0).contains(&equity_linkage) {
            return Err(invalid("equity_linkage"));
        }
        if !equity_dividend_correlation.is_finite()
            || !(-1.0..=1.0).contains(&equity_dividend_correlation)
        {
            return Err(invalid("equity_dividend_correlation"));
        }
        Ok(Self {
            mean_reversion,
            equity_linkage,
            dividend_volatility,
            equity_dividend_correlation,
        })
    }
    #[must_use]
    pub const fn mean_reversion(self) -> f64 {
        self.mean_reversion
    }
    #[must_use]
    pub const fn equity_linkage(self) -> f64 {
        self.equity_linkage
    }
    #[must_use]
    pub const fn dividend_volatility(self) -> f64 {
        self.dividend_volatility
    }
    #[must_use]
    pub const fn equity_dividend_correlation(self) -> f64 {
        self.equity_dividend_correlation
    }
}

/// Strictly positive normalized equity and dividend states. Initial values are 1.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BuehlerDividendState {
    pub(crate) equity: f64,
    pub(crate) dividend: f64,
}
impl BuehlerDividendState {
    pub fn new(equity: f64, dividend: f64) -> Result<Self, StochasticDividendError> {
        positive(equity, "equity_state")?;
        positive(dividend, "dividend_state")?;
        Ok(Self { equity, dividend })
    }
    #[must_use]
    pub const fn initial() -> Self {
        Self {
            equity: 1.0,
            dividend: 1.0,
        }
    }
    #[must_use]
    pub const fn equity(self) -> f64 {
        self.equity
    }
    #[must_use]
    pub const fn dividend(self) -> f64 {
        self.dividend
    }
}

#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum StochasticDividendError {
    InvalidInput { field: &'static str },
    Unsupported { feature: &'static str },
}
impl fmt::Display for StochasticDividendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput { field } => write!(f, "invalid stochastic-dividend {field}"),
            Self::Unsupported { feature } => {
                write!(f, "stochastic dividends do not support {feature}")
            }
        }
    }
}
impl Error for StochasticDividendError {}

pub(crate) fn invalid(field: &'static str) -> StochasticDividendError {
    StochasticDividendError::InvalidInput { field }
}
pub(crate) fn nonnegative(x: f64, field: &'static str) -> Result<(), StochasticDividendError> {
    if !x.is_finite() || x < 0.0 {
        Err(invalid(field))
    } else {
        Ok(())
    }
}
pub(crate) fn positive(x: f64, field: &'static str) -> Result<(), StochasticDividendError> {
    if !x.is_finite() || x <= 0.0 {
        Err(invalid(field))
    } else {
        Ok(())
    }
}
