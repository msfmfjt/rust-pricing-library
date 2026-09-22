//! Buehler stochastic discrete cash dividends with carry-funded escrowed equity.
//!
//! [`StochasticDividendPricingPlan::compile_bs`] is a separate price-only
//! deterministic-rate entry point. Existing model combinations, calibration and
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
