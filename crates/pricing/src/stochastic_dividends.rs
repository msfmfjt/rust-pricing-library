//! Buehler stochastic discrete cash dividends with carry-funded escrowed equity.
//!
//! [`StochasticDividendPricingPlan::compile_bs`] is a separate price-only
//! deterministic-rate entry point. The `compile_bergomi` and
//! `compile_bergomi_two_factor` factories add pure stochastic volatility with
//! explicit dividend/volatility correlations. Calibration and
//! AAD are not silently reused. See the stochastic-dividends model reference.

pub use crate::engine::processes::stochastic_dividends::{
    StochasticDividendNode, StochasticDividendPathPlan,
};
pub use crate::engine::risk::stochastic_dividends::{
    StochasticDividendPrice, StochasticDividendPricingPlan,
};
pub use crate::models::{
    BuehlerDividendModel, BuehlerDividendState, STOCHASTIC_DIVIDEND_SCHEME, StochasticDividendError,
};
