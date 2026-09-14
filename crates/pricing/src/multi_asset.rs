//! Experimental multi-underlying, single-currency pricing with deterministic rates.
//! BS, Local Volatility and Bergomi LSV share correlated drivers and the payoff tape.
//! The stable single-asset JSON request is unchanged; see `docs/multi-asset-v0.1.md`.
pub use crate::engine::multi_asset::{
    MultiAssetBergomiLsvConfig, MultiAssetHullWhiteConfig, MultiAssetHullWhiteCurveRisk,
    MultiAssetHullWhiteLsvRisk, MultiAssetLsv2FactorConfig, MultiAssetLsvConfig, MultiAssetLsvRisk,
    MultiAssetPrice, MultiAssetPricingPlan, MultiAssetRisk, MultiAssetRiskConfig,
};
pub use crate::market::{CorrelationTermStructure, CorrelationToleranceConfig};
pub use crate::product::multi_asset::{
    AutocallObservation, AutocallSpec, BasketComponent, MemoryTermination, MultiAssetProduct,
    WorstOfComponent,
};
use crate::{core::CoreError, market::MarketError, product::GraphError};
use std::{error::Error, fmt};

#[derive(Debug)]
pub enum MultiAssetError {
    Invalid(&'static str),
    Core(CoreError),
    Market(MarketError),
    Graph(GraphError),
    Correlation(pricing_numerics::CorrelationError),
    Numerical(String),
}
impl fmt::Display for MultiAssetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(s) => write!(f, "multi-asset: {s}"),
            Self::Core(e) => e.fmt(f),
            Self::Market(e) => e.fmt(f),
            Self::Graph(e) => e.fmt(f),
            Self::Correlation(e) => e.fmt(f),
            Self::Numerical(s) => write!(f, "multi-asset numerical error: {s}"),
        }
    }
}
impl Error for MultiAssetError {}
macro_rules! conversion {
    ($source:ty, $variant:ident) => {
        impl From<$source> for MultiAssetError {
            fn from(e: $source) -> Self {
                Self::$variant(e)
            }
        }
    };
}
conversion!(CoreError, Core);
conversion!(MarketError, Market);
conversion!(GraphError, Graph);
conversion!(pricing_numerics::CorrelationError, Correlation);
impl MultiAssetError {
    pub(crate) fn numerical(e: impl fmt::Display) -> Self {
        Self::Numerical(e.to_string())
    }
}
