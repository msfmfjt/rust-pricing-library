//! Experimental multi-underlying, single-currency pricing with deterministic rates.
//! BS, Local Volatility and Bergomi LSV share correlated drivers and the payoff tape.
//! The stable single-asset JSON request is unchanged; see `docs/models/multi-asset.md`.
pub use crate::engine::multi_asset::{
    LocalCorrelationCalibration, LocalCorrelationConfig, LocalCorrelationExtensions,
    LocalCorrelationFeasibility, LocalCorrelationNodeDiagnostics, LocalCorrelationRisk,
    MultiAssetBergomiLsvConfig, MultiAssetHullWhiteConfig, MultiAssetHullWhiteCurveRisk,
    MultiAssetHullWhiteLsvRisk, MultiAssetLsv2FactorConfig, MultiAssetLsvConfig, MultiAssetLsvRisk,
    MultiAssetPrice, MultiAssetPricingPlan, MultiAssetRisk, MultiAssetRiskConfig,
    MultiAssetRoughLsvConfig,
};
pub use crate::market::{CorrelationTermStructure, CorrelationToleranceConfig};
pub use crate::product::multi_asset::{
    AutocallObservation, AutocallSpec, BasketComponent, MemoryTermination, MultiAssetProduct,
    WorstOfComponent,
};
use crate::{core::CoreError, market::MarketError, product::GraphError};
use std::{error::Error, fmt};

#[derive(Debug)]
#[non_exhaustive]
pub enum MultiAssetError {
    Invalid(&'static str),
    Core(CoreError),
    Market(MarketError),
    Graph(GraphError),
    Correlation(pricing_numerics::CorrelationError),
    /// A legacy numerical diagnostic without an underlying typed error.
    Numerical(String),
    /// A numerical failure whose concrete cause remains available via `Error::source`.
    NumericalSource(Box<dyn Error + Send + Sync>),
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
            Self::NumericalSource(e) => write!(f, "multi-asset numerical error: {e}"),
        }
    }
}
impl Error for MultiAssetError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Core(e) => Some(e),
            Self::Market(e) => Some(e),
            Self::Graph(e) => Some(e),
            Self::Correlation(e) => Some(e),
            Self::NumericalSource(e) => Some(e.as_ref()),
            Self::Invalid(_) | Self::Numerical(_) => None,
        }
    }
}
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
    pub(crate) fn numerical(e: impl Error + Send + Sync + 'static) -> Self {
        Self::NumericalSource(Box::new(e))
    }
}
