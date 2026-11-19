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
use crate::{
    EquityModelSpec, MeasureKind, MeasureMethodKind, PricingSpecification, RateModelSpec,
    SpecificationValidationError,
};

/// Operations have distinct numerical contracts. `Evaluate` retains the risks
/// already requested by a standard single-asset specification.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum PricingOperation {
    Evaluate,
    Aad {
        gamma_relative_bump: Option<f64>,
    },
    LocalVarianceRisk,
    /// An explicit list of measures. The nested risk request of a standard
    /// single-asset source must select nothing, so risk is chosen in one place.
    Measures(crate::MeasureSet),
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
    /// The backend has no estimator of this measure by the requested method.
    NoNativeEstimator,
    /// A `Measures` task whose nested risk request also selects risk.
    AmbiguousRiskSelection,
    /// The stated bump calibration does not fit the source: a calibrated
    /// source needs `Fixed` or `Recalibrated`, any other `NotApplicable`.
    BumpCalibrationMismatch,
    /// Holding a calibration fixed across a bump needs a backend hook that
    /// does not exist yet.
    FixedCalibrationUnavailable,
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

/// The source compositions that share one set of exposed task operations.
/// Capability queries, task construction and the published operation matrix
/// all read this single classification.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum SourceFamily {
    /// BS, Black-76, Local Volatility or Bass-LV at deterministic rates.
    StandardSingleAsset,
    /// One-/two-factor or rough Bergomi stochastic volatility.
    StochasticVolatility,
    /// Deterministic-rate one-/two-factor or rough Bergomi LSV.
    LocalStochasticVolatility,
    /// Deterministic-rate LSV on a rough-volatility family (rough Heston,
    /// lifted Heston, quadratic rough Heston, mixed rough Bergomi, rough
    /// SABR, RFSV). Separate because it has native Spot estimators.
    RoughFamilyLsv,
    /// Any single-asset equity model with Hull-White rates.
    SingleAssetHullWhite,
    /// A direct rough-volatility family without a local-volatility target.
    DirectRoughVolatility,
    /// Baskets, including mixed LSV, shared Hull-White and Local Correlation.
    MultiAsset,
    /// Stochastic cash dividends without a Bergomi LSV residual equity.
    StochasticDividend,
    /// Stochastic cash dividends with a Bergomi LSV residual equity.
    StochasticDividendLsv,
}

impl SourceFamily {
    pub const ALL: [Self; 9] = [
        Self::StandardSingleAsset,
        Self::StochasticVolatility,
        Self::LocalStochasticVolatility,
        Self::RoughFamilyLsv,
        Self::SingleAssetHullWhite,
        Self::DirectRoughVolatility,
        Self::MultiAsset,
        Self::StochasticDividend,
        Self::StochasticDividendLsv,
    ];

    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::StandardSingleAsset => "standard_single_asset",
            Self::StochasticVolatility => "stochastic_volatility",
            Self::LocalStochasticVolatility => "local_stochastic_volatility",
            Self::RoughFamilyLsv => "rough_family_lsv",
            Self::SingleAssetHullWhite => "single_asset_hull_white",
            Self::DirectRoughVolatility => "direct_rough_volatility",
            Self::MultiAsset => "multi_asset",
            Self::StochasticDividend => "stochastic_dividend",
            Self::StochasticDividendLsv => "stochastic_dividend_lsv",
        }
    }

    /// Whether the common task exposes `operation` for this family at all.
    /// A true answer still leaves the payoff, reverse-trace, compilation and
    /// native preflight conditions reported by `PricingSource::capability`.
    /// `Measures` is exposed everywhere because a price is always available;
    /// each listed measure is then judged by `measure_methods`.
    #[must_use]
    pub fn exposes(self, operation: &PricingOperation) -> bool {
        match operation {
            PricingOperation::Evaluate | PricingOperation::Measures(_) => true,
            PricingOperation::Aad {
                gamma_relative_bump: Some(_),
            } => matches!(self, Self::MultiAsset),
            PricingOperation::Aad {
                gamma_relative_bump: None,
            } => matches!(
                self,
                Self::StochasticVolatility
                    | Self::SingleAssetHullWhite
                    | Self::MultiAsset
                    | Self::StochasticDividend
            ),
            PricingOperation::LocalVarianceRisk => matches!(
                self,
                Self::LocalStochasticVolatility
                    | Self::RoughFamilyLsv
                    | Self::StochasticDividendLsv
            ),
        }
    }

    /// The methods by which the `Measures` operation can produce `measure`
    /// for this family. Empty means the measure has no adapter here yet.
    #[must_use]
    pub const fn measure_methods(self, measure: MeasureKind) -> &'static [MeasureMethodKind] {
        use MeasureMethodKind as M;
        match (self, measure) {
            (_, MeasureKind::Price) => &[M::Native],
            (Self::StandardSingleAsset, MeasureKind::SpotGamma) => {
                &[M::NativeSpotBump, M::BumpAndRevalue]
            }
            (
                Self::StandardSingleAsset,
                MeasureKind::SpotDelta | MeasureKind::ParallelMarketIvVega,
            ) => &[M::Native, M::BumpAndRevalue],
            (Self::StandardSingleAsset, MeasureKind::VegaKt) => &[M::Native],
            (Self::RoughFamilyLsv, MeasureKind::SpotDelta) => {
                &[M::NativeCalibrated, M::BumpAndRevalue]
            }
            (Self::RoughFamilyLsv, MeasureKind::SpotGamma) => {
                &[M::NativeSpotBumpCalibrated, M::BumpAndRevalue]
            }
            // An LSV target with a reproducing reporting IV basis: the market
            // IVs move the target, and the calibration follows.
            (Self::RoughFamilyLsv, MeasureKind::ParallelMarketIvVega) => {
                &[M::NativeCalibrated, M::BumpAndRevalue]
            }
            (Self::LocalStochasticVolatility, MeasureKind::ParallelMarketIvVega) => {
                &[M::BumpAndRevalue]
            }
            // Fixed-target parameter risk through the particle recalibration.
            (Self::RoughFamilyLsv, MeasureKind::ModelParameters) => &[M::NativeCalibrated],
            // Fixed-model pathwise Delta and explicitly bumped Gamma; nothing
            // is calibrated, so no treatment is stated.
            (Self::DirectRoughVolatility, MeasureKind::SpotDelta) => {
                &[M::Native, M::BumpAndRevalue]
            }
            (Self::DirectRoughVolatility, MeasureKind::SpotGamma) => {
                &[M::NativeSpotBump, M::BumpAndRevalue]
            }
            (Self::DirectRoughVolatility, MeasureKind::ModelParameters) => &[M::Native],
            // Stochastic dividends: AAD Delta and a bump of AAD Deltas; the
            // residual-equity LSV estimators are scale-invariant
            // (recalibrated) only.
            // Raw Brownian-correlation partials of the joint reverse.
            (
                Self::StochasticDividend,
                MeasureKind::Correlations
                | MeasureKind::ModelParameters
                | MeasureKind::DividendCash
                | MeasureKind::DiscountCurve,
            ) => &[M::Native],
            // Cash means and curves re-anchor the residual-equity LSV like Spot.
            (
                Self::StochasticDividendLsv,
                MeasureKind::DividendCash | MeasureKind::DiscountCurve,
            ) => &[M::NativeCalibrated],
            (Self::StochasticDividend, MeasureKind::SpotDelta) => &[M::Native, M::BumpAndRevalue],
            (Self::StochasticDividend, MeasureKind::SpotGamma) => {
                &[M::NativeSpotBump, M::BumpAndRevalue]
            }
            (Self::StochasticDividendLsv, MeasureKind::SpotDelta) => {
                &[M::NativeCalibrated, M::BumpAndRevalue]
            }
            (Self::StochasticDividendLsv, MeasureKind::SpotGamma) => {
                &[M::NativeSpotBumpCalibrated, M::BumpAndRevalue]
            }
            // Baskets: the AAD Delta per asset and the bump of AAD Deltas,
            // stated without a treatment on a basket without a calibration
            // step and as recalibrated on an LSV or Local Correlation one.
            (Self::MultiAsset, MeasureKind::SpotDelta) => {
                &[M::Native, M::NativeCalibrated, M::BumpAndRevalue]
            }
            // A basket's Spot correlations have no native derivative.
            (Self::MultiAsset, MeasureKind::Correlations) => &[M::BumpAndRevalue],
            // The shared Hull-White initial curve, in the same AAD run.
            (Self::MultiAsset, MeasureKind::DiscountCurve) => &[M::Native, M::NativeCalibrated],
            (Self::MultiAsset, MeasureKind::CrossGamma) => {
                &[M::NativeSpotBump, M::NativeSpotBumpCalibrated]
            }
            // Every source reprices under a moved Spot.
            (_, MeasureKind::SpotDelta | MeasureKind::SpotGamma) => &[M::BumpAndRevalue],
            // A Black-Scholes equity with Hull-White rates has one volatility.
            (Self::SingleAssetHullWhite, MeasureKind::ParallelMarketIvVega) => &[M::BumpAndRevalue],
            _ => &[],
        }
    }
}

/// Renders which task operations each source family exposes. The task guide
/// embeds this table and a test fails when the two differ.
#[must_use]
pub fn operation_matrix_markdown() -> String {
    let columns: [(&str, PricingOperation); 4] = [
        ("`evaluate`", PricingOperation::Evaluate),
        (
            "`aad`",
            PricingOperation::Aad {
                gamma_relative_bump: None,
            },
        ),
        (
            "`aad` with `gamma_relative_bump`",
            PricingOperation::Aad {
                gamma_relative_bump: Some(0.01),
            },
        ),
        ("`local_variance_risk`", PricingOperation::LocalVarianceRisk),
    ];
    let mut table = String::from("| Source family |");
    for (label, _) in &columns {
        table.push_str(&format!(" {label} |"));
    }
    table.push_str("\n| --- |");
    for _ in &columns {
        table.push_str(" --- |");
    }
    table.push('\n');
    for family in SourceFamily::ALL {
        table.push_str(&format!("| `{}` |", family.as_str()));
        for (_, operation) in &columns {
            table.push_str(if family.exposes(operation) {
                " exposed |"
            } else {
                " not exposed |"
            });
        }
        table.push('\n');
    }
    table
}

/// Renders which measures each source family produces through the `Measures`
/// operation, and by which method. Embedded in the task guide like the
/// operation matrix.
#[must_use]
pub fn measure_matrix_markdown() -> String {
    let mut table = String::from("| Source family |");
    for measure in MeasureKind::ALL {
        table.push_str(&format!(" `{}` |", measure.as_str()));
    }
    table.push_str("\n| --- |");
    for _ in MeasureKind::ALL {
        table.push_str(" --- |");
    }
    table.push('\n');
    for family in SourceFamily::ALL {
        table.push_str(&format!("| `{}` |", family.as_str()));
        for measure in MeasureKind::ALL {
            let methods = family.measure_methods(measure);
            if methods.is_empty() {
                table.push_str(" not exposed |");
            } else {
                let names: Vec<_> = methods
                    .iter()
                    .map(|m| format!("`{}`", m.as_str()))
                    .collect();
                table.push_str(&format!(" {} |", names.join(", ")));
            }
        }
        table.push('\n');
    }
    table
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
impl PricingSource {
    /// Classifies the composition without compiling or calibrating it.
    #[must_use]
    pub fn family(&self) -> SourceFamily {
        match self {
            Self::SingleAsset(s) => {
                if matches!(s.model().rates, RateModelSpec::HullWhite { .. }) {
                    return SourceFamily::SingleAssetHullWhite;
                }
                match &s.model().equity {
                    EquityModelSpec::StochasticVolatility { .. } => {
                        SourceFamily::StochasticVolatility
                    }
                    EquityModelSpec::LocalStochasticVolatility { .. } => {
                        SourceFamily::LocalStochasticVolatility
                    }
                    EquityModelSpec::RoughLocalStochasticVolatility { .. } => {
                        SourceFamily::RoughFamilyLsv
                    }
                    EquityModelSpec::RoughVolatility(_) => SourceFamily::DirectRoughVolatility,
                    EquityModelSpec::Standard(_) => SourceFamily::StandardSingleAsset,
                }
            }
            Self::MultiAsset(_) => SourceFamily::MultiAsset,
            Self::StochasticDividend(s) => match &s.equity().model().equity {
                EquityModelSpec::LocalStochasticVolatility { .. } => {
                    SourceFamily::StochasticDividendLsv
                }
                _ => SourceFamily::StochasticDividend,
            },
        }
    }
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
        let capability = source.capability(operation.clone());
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
    /// The selected operation. Cloning a measure list shares its storage.
    #[must_use]
    pub fn operation(&self) -> PricingOperation {
        self.operation.clone()
    }
    /// The requested measures, or `None` for the three legacy operations.
    #[must_use]
    pub fn measures(&self) -> Option<&crate::MeasureSet> {
        match &self.operation {
            PricingOperation::Measures(set) => Some(set),
            _ => None,
        }
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
