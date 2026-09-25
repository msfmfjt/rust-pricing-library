//! Buehler stochastic discrete cash dividends with carry-funded escrowed equity.
//!
//! [`StochasticDividendPricingPlan::compile_bs`] is a separate price-only
//! deterministic-rate entry point. The `compile_bergomi` and
//! `compile_bergomi_two_factor` factories add pure stochastic volatility with
//! explicit dividend/volatility correlations. Calibration and
//! AAD are not silently reused. `evaluate_aad` uses a dedicated split-path
//! reverse and the shared payoff adjoint. See the model reference for fixed inputs.

pub use crate::engine::processes::stochastic_dividends::{
    StochasticDividendNode, StochasticDividendPathPlan,
};
pub use crate::engine::risk::stochastic_dividends::{
    StochasticDividendAadRisk, StochasticDividendGammaRisk, StochasticDividendLocalVarianceRisk,
    StochasticDividendLsvSpotRisk, StochasticDividendPrice, StochasticDividendPricingPlan,
};
pub use crate::models::{
    BuehlerDividendModel, BuehlerDividendState, STOCHASTIC_DIVIDEND_SCHEME, StochasticDividendError,
};

/// Constant-volatility stochastic dividends coupled to stochastic rates.
pub use crate::engine::processes::stochastic_dividends::hull_white::{
    STOCHASTIC_DIVIDEND_HW_SCHEME, StochasticDividendHullWhitePathPlan,
    StochasticDividendHullWhiteState,
};
pub use crate::engine::risk::stochastic_dividends::hull_white::StochasticDividendHullWhitePricingPlan;
