//! Immutable inputs for the native single-asset, basket and dividend adapters.
use std::sync::Arc;

use crate::core::Date;
use crate::market::{CorrelationTermStructure, EquityMarket};
use crate::mc::{EngineConfig, ExecutionPolicy};
use crate::models::{BuehlerDividendModel, ModelSpec};
use crate::multi_asset::{
    LocalCorrelationConfig, LocalCorrelationExtensions, MultiAssetBergomiLsvConfig,
    MultiAssetHullWhiteConfig,
};
use crate::product::GraphProduct;
use crate::{EquityModelSpec, PricingSpecification, RateModelSpec, SpecificationValidationError};

/// Operations have distinct numerical contracts. `Evaluate` retains the risks
/// already requested by a standard single-asset specification.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum PricingOperation {
    Evaluate,
    Aad { gamma_relative_bump: Option<f64> },
    LocalVarianceRisk,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CapabilityStatus {
    Supported,
    Conditional,
    Unsupported,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum CapabilityReason {
    CompilationRequired,
    NativePreflightRequired,
    DedicatedRiskApi,
    ModelCombination,
    DiscontinuousPayoff,
    ReverseTraceRequired,
    InvalidGammaBump,
    InvalidSource,
}

#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct PricingCapability {
    pub operation: PricingOperation,
    pub status: CapabilityStatus,
    pub reason: Option<CapabilityReason>,
    pub detail: String,
}

impl PricingCapability {
    pub(crate) fn new(
        operation: PricingOperation,
        status: CapabilityStatus,
        reason: Option<CapabilityReason>,
        detail: impl Into<String>,
    ) -> Self {
        Self {
            operation,
            status,
            reason,
            detail: detail.into(),
        }
    }
}

/// Settings consumed by the existing mixed-factor basket compiler. Every
/// calibration input is retained; compiled surfaces and workspaces are not.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct MultiAssetModelSpec {
    pub models: Vec<ModelSpec>,
    pub correlation: CorrelationTermStructure,
    pub lsv: Vec<Option<MultiAssetBergomiLsvConfig>>,
    pub driver_correlations: Option<Vec<Vec<Vec<f64>>>>,
    pub hull_white: Option<MultiAssetHullWhiteConfig>,
    pub local_correlation: Option<(LocalCorrelationConfig, LocalCorrelationExtensions)>,
}
impl MultiAssetModelSpec {
    #[must_use]
    pub fn new(models: Vec<ModelSpec>, correlation: CorrelationTermStructure) -> Self {
        Self {
            lsv: vec![None; models.len()],
            models,
            correlation,
            driver_correlations: None,
            hull_white: None,
            local_correlation: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct MultiAssetSpecification {
    valuation_date: Date,
    product: GraphProduct,
    markets: Vec<EquityMarket>,
    model: MultiAssetModelSpec,
    engine: EngineConfig,
    execution: ExecutionPolicy,
    maximum_step: f64,
}
impl MultiAssetSpecification {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        valuation_date: Date,
        product: GraphProduct,
        markets: Vec<EquityMarket>,
        model: MultiAssetModelSpec,
        engine: EngineConfig,
        execution: ExecutionPolicy,
        maximum_step: f64,
    ) -> Result<Self, SpecificationValidationError> {
        positive_step(maximum_step)?;
        let order: Vec<_> = markets.iter().map(|m| m.forward().underlying()).collect();
        if markets.is_empty()
            || markets.len() != model.models.len()
            || markets.len() != model.lsv.len()
        {
            return Err(invalid(
                "model",
                "market, model and LSV counts must agree and be nonzero",
            ));
        }
        if order != product.underlyings() || order != model.correlation.underlyings() {
            return Err(invalid(
                "market",
                "product, market and correlation underlying order must match",
            ));
        }
        if markets.iter().any(|m| {
            m.currency() != product.currency()
                || m.forward().discount_curve() != markets[0].forward().discount_curve()
        }) {
            return Err(invalid(
                "market",
                "all assets must share the product currency and discount curve",
            ));
        }
        if product.payment_dates().iter().any(|d| *d < valuation_date)
            || product
                .observations()
                .iter()
                .any(|(_, d)| *d < valuation_date)
        {
            return Err(invalid(
                "product",
                "past observations must be literal fixings and payments cannot precede valuation",
            ));
        }
        for (m, lsv) in model.models.iter().zip(&model.lsv) {
            if !matches!(
                m,
                ModelSpec::BlackScholes(_) | ModelSpec::LocalVolatility(_)
            ) || (lsv.is_some() && !matches!(m, ModelSpec::LocalVolatility(_)))
            {
                return Err(invalid(
                    "model",
                    "basket marginals support BS or LV; LSV requires an LV target",
                ));
            }
            if matches!(lsv, Some(MultiAssetBergomiLsvConfig::Rough(_)))
                && model.hull_white.is_none()
            {
                return Err(invalid(
                    "model",
                    "rough basket LSV requires the shared HW adapter",
                ));
            }
        }
        // Public native configuration records contain raw floats. Keep NaN and
        // infinity out of immutable source documents; grid-dependent covariance
        // and calibration acceptance still belong to the existing compiler.
        let finite_matrices = |matrices: &Option<Vec<Vec<Vec<f64>>>>| {
            matrices
                .as_ref()
                .is_none_or(|ms| ms.iter().flatten().flatten().all(|x| x.is_finite()))
        };
        if !finite_matrices(&model.driver_correlations)
            || model
                .hull_white
                .as_ref()
                .is_some_and(|hw| hw.rate_correlations.iter().any(|x| !valid_correlation(*x)))
            || model.local_correlation.as_ref().is_some_and(|(c, e)| {
                c.basket_weights.iter().any(|x| !x.is_finite() || *x <= 0.0)
                    || !c.minimum_variance_span.is_finite()
                    || c.minimum_variance_span <= 0.0
                    || !finite_matrices(&e.second_driver_correlations)
            })
        {
            return Err(invalid(
                "model",
                "nonfinite or invalid correlation/calibration inputs",
            ));
        }
        Ok(Self {
            valuation_date,
            product,
            markets,
            model,
            engine,
            execution,
            maximum_step,
        })
    }
    #[must_use]
    pub fn valuation_date(&self) -> Date {
        self.valuation_date
    }
    #[must_use]
    pub fn product(&self) -> &GraphProduct {
        &self.product
    }
    #[must_use]
    pub fn markets(&self) -> &[EquityMarket] {
        &self.markets
    }
    #[must_use]
    pub fn model(&self) -> &MultiAssetModelSpec {
        &self.model
    }
    #[must_use]
    pub fn engine(&self) -> EngineConfig {
        self.engine
    }
    #[must_use]
    pub fn execution(&self) -> ExecutionPolicy {
        self.execution
    }
    #[must_use]
    pub fn maximum_step(&self) -> f64 {
        self.maximum_step
    }
}

/// Adds the stochastic reserve to a complete equity/rate specification.
/// `maximum_step` is the dividend split grid limit, also used by direct SV/HW.
#[derive(Clone, Debug, PartialEq)]
pub struct StochasticDividendSpecification {
    equity: PricingSpecification,
    dividends: BuehlerDividendModel,
    volatility_correlations: Vec<f64>,
    rate_correlation: Option<f64>,
    maximum_step: f64,
}
impl StochasticDividendSpecification {
    pub fn new(
        equity: PricingSpecification,
        dividends: BuehlerDividendModel,
        volatility_correlations: Vec<f64>,
        rate_correlation: Option<f64>,
        maximum_step: f64,
    ) -> Result<Self, SpecificationValidationError> {
        positive_step(maximum_step)?;
        let r = equity.risk();
        if r.delta()
            || r.gamma().is_some()
            || r.vega()
            || r.vega_kt().is_some()
            || r.checkpoint_interval().is_some()
            || r.aad_tile_capacity().is_some()
            || r.payoff_smoothing_width_ladder().is_some()
            || equity.numerics().lsm.is_some()
        {
            return Err(invalid(
                "equity.risk",
                "dividend sources require price-only inputs; select an explicit source operation",
            ));
        }
        if equity
            .numerics()
            .maximum_step
            .is_some_and(|v| v != maximum_step)
        {
            return Err(invalid(
                "maximum_step",
                "equity and dividend execution grids must use the same maximum step",
            ));
        }
        let factors = match &equity.model().equity {
            EquityModelSpec::Standard(ModelSpec::BlackScholes(_)) => 0,
            EquityModelSpec::StochasticVolatility { factor, .. }
            | EquityModelSpec::LocalStochasticVolatility { factor, .. } => match factor {
                crate::VolatilityFactorSpec::BergomiTwoFactor(_) => 2,
                _ => 1,
            },
            _ => {
                return Err(invalid(
                    "equity.model",
                    "stochastic dividends support BS and 1F/2F/rough Bergomi SV/LSV",
                ));
            }
        };
        if volatility_correlations.len() != factors
            || volatility_correlations
                .iter()
                .any(|x| !valid_correlation(*x))
        {
            return Err(invalid(
                "volatility_correlations",
                "one finite correlation in [-1,1] is required per volatility factor",
            ));
        }
        match (&equity.model().rates, rate_correlation) {
            (RateModelSpec::Deterministic, None) => {}
            (RateModelSpec::HullWhite { .. }, Some(x)) if factors == 0 && valid_correlation(x) => {}
            _ => {
                return Err(invalid(
                    "rate_correlation",
                    "HW stochastic dividends require BS and an explicit dividend/rate correlation; deterministic rates use none",
                ));
            }
        }
        Ok(Self {
            equity,
            dividends,
            volatility_correlations,
            rate_correlation,
            maximum_step,
        })
    }
    #[must_use]
    pub fn equity(&self) -> &PricingSpecification {
        &self.equity
    }
    #[must_use]
    pub fn dividends(&self) -> BuehlerDividendModel {
        self.dividends
    }
    #[must_use]
    pub fn volatility_correlations(&self) -> &[f64] {
        &self.volatility_correlations
    }
    #[must_use]
    pub fn rate_correlation(&self) -> Option<f64> {
        self.rate_correlation
    }
    #[must_use]
    pub fn maximum_step(&self) -> f64 {
        self.maximum_step
    }
}

#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum PricingSource {
    SingleAsset(PricingSpecification),
    MultiAsset(MultiAssetSpecification),
    StochasticDividend(StochasticDividendSpecification),
}
impl From<PricingSpecification> for PricingSource {
    fn from(s: PricingSpecification) -> Self {
        Self::SingleAsset(s)
    }
}
impl From<MultiAssetSpecification> for PricingSource {
    fn from(s: MultiAssetSpecification) -> Self {
        Self::MultiAsset(s)
    }
}
impl From<StochasticDividendSpecification> for PricingSource {
    fn from(s: StochasticDividendSpecification) -> Self {
        Self::StochasticDividend(s)
    }
}

/// A replayable calculation includes the operation, not just the price inputs.
#[derive(Clone, Debug)]
pub struct PricingTask {
    source: Arc<PricingSource>,
    operation: PricingOperation,
}
impl PricingTask {
    pub fn new(
        source: impl Into<PricingSource>,
        operation: PricingOperation,
    ) -> Result<Self, SpecificationValidationError> {
        Self::from_shared_source(Arc::new(source.into()), operation)
    }
    /// Select another operation while sharing the immutable source inputs.
    /// Compilation remains explicit; use the compiled task's `with_operation`
    /// to reuse an existing native plan and its calibration.
    pub fn with_operation(
        &self,
        operation: PricingOperation,
    ) -> Result<Self, SpecificationValidationError> {
        Self::from_shared_source(Arc::clone(&self.source), operation)
    }
    fn from_shared_source(
        source: Arc<PricingSource>,
        operation: PricingOperation,
    ) -> Result<Self, SpecificationValidationError> {
        let capability = source.capability(operation);
        if capability.status == CapabilityStatus::Unsupported {
            return Err(SpecificationValidationError::UnsupportedOperation {
                reason: capability.reason,
                detail: capability.detail,
            });
        }
        Ok(Self { source, operation })
    }
    #[must_use]
    pub fn source(&self) -> &PricingSource {
        &self.source
    }
    #[must_use]
    pub fn operation(&self) -> PricingOperation {
        self.operation
    }
}

pub(crate) fn invalid(field: &'static str, message: impl ToString) -> SpecificationValidationError {
    SpecificationValidationError::Invalid {
        field,
        message: message.to_string(),
    }
}
fn positive_step(step: f64) -> Result<(), SpecificationValidationError> {
    if !step.is_finite() || step <= 0.0 {
        return Err(invalid(
            "maximum_step",
            "a finite positive step is required",
        ));
    }
    Ok(())
}
fn valid_correlation(x: f64) -> bool {
    x.is_finite() && x.abs() <= 1.0
}
