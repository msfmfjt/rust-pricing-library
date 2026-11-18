//! Measures name a quantity by its financial meaning, not by the algorithm
//! that produces it. See ADR 0043.
use std::sync::Arc;

use crate::core::PositiveF64;
use crate::risk::{SpotBump, VegaKtConfig};
use crate::{CapabilityReason, CapabilityStatus, SpecificationValidationError};

/// A quantity a task can ask for. Later phases add variants.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum Measure {
    /// Always part of a result, whether or not it is listed.
    Price,
    SpotDelta,
    SpotGamma,
    /// Parallel shift of the market implied-volatility surface. For a
    /// constant-volatility model this is its single volatility.
    ParallelMarketIvVega,
    /// Bucketed quote-node Vega on the stated grid.
    VegaKt(VegaKtConfig),
    /// Derivatives with respect to the stated native model parameters, one
    /// report row per parameter in the stated order. Not a Vega: a model's
    /// volatility inputs are parameters here, compared across models only
    /// through the market-IV measures.
    ModelParameters(ModelParameterSelection),
    /// Derivatives with respect to the stated Brownian correlations, one
    /// report row per correlation in the stated order. Each varies one
    /// symmetric pair of the driver correlation matrix, holding the others.
    Correlations(CorrelationSelection),
}

/// A native model parameter, identified by its role in the source's model.
/// The same identifier names the same model input on every model that has
/// it; a model without it refuses it.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum ModelParameterId {
    /// Initial instantaneous variance (Heston family).
    InitialVariance,
    /// Initial state of the feedback variable (quadratic rough Heston).
    InitialState,
    MeanReversion,
    /// Long-run variance (Heston family).
    LongRunVariance,
    /// Volatility of the volatility driver.
    VolOfVol,
    /// Volatility of one component of a mixed rough Bergomi driver.
    ComponentVolOfVol {
        component: u16,
    },
    /// Spot-volatility correlation.
    Correlation,
    /// Hurst exponent of the kernel (left derivative at one half).
    Hurst,
    /// Quadratic coefficient of the feedback map (quadratic rough Heston).
    Quadratic,
    /// Shift of the feedback map (quadratic rough Heston).
    Shift,
    /// Variance floor of the feedback map (quadratic rough Heston).
    VarianceFloor,
    /// Initial volatility of the residual equity (stochastic dividends).
    InitialVolatility,
    /// Mean reversion of the Buehler dividend factor.
    DividendMeanReversion,
    /// Equity linkage of the Buehler dividend split.
    EquityLinkage,
    /// Volatility of the Buehler dividend factor.
    DividendVolatility,
    /// Mean reversion of one Bergomi variance factor.
    FactorMeanReversion {
        factor: u16,
    },
    /// Mixing weight of a two-factor Bergomi variance.
    MixingWeight,
    /// Hull-White rate mean reversion.
    RateMeanReversion,
    /// One knot of the Hull-White piecewise-constant rate volatility.
    RateVolatility {
        knot: u16,
    },
}

impl ModelParameterId {
    /// The wire name. A component parameter also carries its index.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::InitialVariance => "initial_variance",
            Self::InitialState => "initial_state",
            Self::MeanReversion => "mean_reversion",
            Self::LongRunVariance => "long_run_variance",
            Self::VolOfVol => "vol_of_vol",
            Self::ComponentVolOfVol { .. } => "component_vol_of_vol",
            Self::Correlation => "correlation",
            Self::Hurst => "hurst",
            Self::Quadratic => "quadratic",
            Self::Shift => "shift",
            Self::VarianceFloor => "variance_floor",
            Self::InitialVolatility => "initial_volatility",
            Self::DividendMeanReversion => "dividend_mean_reversion",
            Self::EquityLinkage => "equity_linkage",
            Self::DividendVolatility => "dividend_volatility",
            Self::FactorMeanReversion { .. } => "factor_mean_reversion",
            Self::MixingWeight => "mixing_weight",
            Self::RateMeanReversion => "rate_mean_reversion",
            Self::RateVolatility { .. } => "rate_volatility",
        }
    }
    /// The scalar parameter of a wire name; `None` for an indexed name or
    /// an unknown one.
    #[must_use]
    pub fn scalar(name: &str) -> Option<Self> {
        Some(match name {
            "initial_variance" => Self::InitialVariance,
            "initial_state" => Self::InitialState,
            "mean_reversion" => Self::MeanReversion,
            "long_run_variance" => Self::LongRunVariance,
            "vol_of_vol" => Self::VolOfVol,
            "correlation" => Self::Correlation,
            "hurst" => Self::Hurst,
            "quadratic" => Self::Quadratic,
            "shift" => Self::Shift,
            "variance_floor" => Self::VarianceFloor,
            "initial_volatility" => Self::InitialVolatility,
            "dividend_mean_reversion" => Self::DividendMeanReversion,
            "equity_linkage" => Self::EquityLinkage,
            "dividend_volatility" => Self::DividendVolatility,
            "mixing_weight" => Self::MixingWeight,
            "rate_mean_reversion" => Self::RateMeanReversion,
            _ => return None,
        })
    }
    /// The index of an indexed parameter (component, factor or knot).
    #[must_use]
    pub const fn index(self) -> Option<u16> {
        match self {
            Self::ComponentVolOfVol { component: i }
            | Self::FactorMeanReversion { factor: i }
            | Self::RateVolatility { knot: i } => Some(i),
            _ => None,
        }
    }
    /// A stable label: the wire name, with the index for an indexed
    /// parameter (`component_vol_of_vol.1`, `rate_volatility.0`).
    #[must_use]
    pub fn label(self) -> String {
        match self.index() {
            Some(i) => format!("{}.{i}", self.as_str()),
            None => self.as_str().to_owned(),
        }
    }
}

impl ModelParameterId {
    /// The parameter a `label` names; `None` for an unknown label.
    #[must_use]
    pub fn from_label(label: &str) -> Option<Self> {
        if let Some((name, index)) = label.split_once('.') {
            let index = index.parse().ok()?;
            return match name {
                "component_vol_of_vol" => Some(Self::ComponentVolOfVol { component: index }),
                "factor_mean_reversion" => Some(Self::FactorMeanReversion { factor: index }),
                "rate_volatility" => Some(Self::RateVolatility { knot: index }),
                _ => None,
            };
        }
        Self::scalar(label)
    }
}

/// A non-empty, duplicate-free, ordered list of model parameters.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelParameterSelection(Arc<[ModelParameterId]>);

impl ModelParameterSelection {
    pub fn new(parameters: Vec<ModelParameterId>) -> Result<Self, SpecificationValidationError> {
        let invalid = |message: String| SpecificationValidationError::Invalid {
            field: "operation.measures.parameters",
            message,
        };
        if parameters.is_empty() {
            return Err(invalid("state at least one model parameter".into()));
        }
        for (i, p) in parameters.iter().enumerate() {
            if parameters[..i].contains(p) {
                return Err(invalid(format!("{} is listed more than once", p.label())));
            }
        }
        Ok(Self(parameters.into()))
    }
    #[must_use]
    pub fn parameters(&self) -> &[ModelParameterId] {
        &self.0
    }
}

/// A Brownian correlation between two drivers of a source, identified by
/// the roles of the drivers. The same identifier names the same pair on
/// every source that has it; a source without it refuses it.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum CorrelationId {
    /// Residual equity and the stochastic dividend driver.
    EquityDividend,
    /// Residual equity and the short rate.
    EquityRate,
    /// Stochastic dividend driver and the short rate.
    DividendRate,
    /// Spot and one volatility factor.
    SpotVolatility { factor: u16 },
    /// Stochastic dividend driver and one volatility factor.
    DividendVolatility { factor: u16 },
    /// The two factors of a two-factor volatility.
    VolatilityFactors,
}

impl CorrelationId {
    /// The wire name. A factor correlation also carries its index.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::EquityDividend => "equity_dividend",
            Self::EquityRate => "equity_rate",
            Self::DividendRate => "dividend_rate",
            Self::SpotVolatility { .. } => "spot_volatility",
            Self::DividendVolatility { .. } => "dividend_volatility",
            Self::VolatilityFactors => "volatility_factors",
        }
    }
    /// A stable label: the wire name, with the factor index for a factor
    /// correlation (`spot_volatility.0`).
    #[must_use]
    pub fn label(self) -> String {
        match self {
            Self::SpotVolatility { factor } | Self::DividendVolatility { factor } => {
                format!("{}.{factor}", self.as_str())
            }
            _ => self.as_str().to_owned(),
        }
    }
    /// The correlation a `label` names; `None` for an unknown label.
    #[must_use]
    pub fn from_label(label: &str) -> Option<Self> {
        if let Some((name, index)) = label.split_once('.') {
            let factor = index.parse().ok()?;
            return match name {
                "spot_volatility" => Some(Self::SpotVolatility { factor }),
                "dividend_volatility" => Some(Self::DividendVolatility { factor }),
                _ => None,
            };
        }
        Some(match label {
            "equity_dividend" => Self::EquityDividend,
            "equity_rate" => Self::EquityRate,
            "dividend_rate" => Self::DividendRate,
            "volatility_factors" => Self::VolatilityFactors,
            _ => return None,
        })
    }
}

/// A non-empty, duplicate-free, ordered list of correlations.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CorrelationSelection(Arc<[CorrelationId]>);

impl CorrelationSelection {
    pub fn new(correlations: Vec<CorrelationId>) -> Result<Self, SpecificationValidationError> {
        let invalid = |message: String| SpecificationValidationError::Invalid {
            field: "operation.measures.correlations",
            message,
        };
        if correlations.is_empty() {
            return Err(invalid("state at least one correlation".into()));
        }
        for (i, c) in correlations.iter().enumerate() {
            if correlations[..i].contains(c) {
                return Err(invalid(format!("{} is listed more than once", c.label())));
            }
        }
        Ok(Self(correlations.into()))
    }
    #[must_use]
    pub fn correlations(&self) -> &[CorrelationId] {
        &self.0
    }
}

/// Identity of a measure without its configuration. Used for discovery,
/// duplicate detection and report lookup.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum MeasureKind {
    Price,
    SpotDelta,
    SpotGamma,
    ParallelMarketIvVega,
    VegaKt,
    ModelParameters,
    Correlations,
}

impl MeasureKind {
    pub const ALL: [Self; 7] = [
        Self::Price,
        Self::SpotDelta,
        Self::SpotGamma,
        Self::ParallelMarketIvVega,
        Self::VegaKt,
        Self::ModelParameters,
        Self::Correlations,
    ];

    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Price => "price",
            Self::SpotDelta => "spot_delta",
            Self::SpotGamma => "spot_gamma",
            Self::ParallelMarketIvVega => "parallel_market_iv_vega",
            Self::VegaKt => "vega_kt",
            Self::ModelParameters => "model_parameters",
            Self::Correlations => "correlations",
        }
    }
}

impl Measure {
    #[must_use]
    pub const fn kind(&self) -> MeasureKind {
        match self {
            Self::Price => MeasureKind::Price,
            Self::SpotDelta => MeasureKind::SpotDelta,
            Self::SpotGamma => MeasureKind::SpotGamma,
            Self::ParallelMarketIvVega => MeasureKind::ParallelMarketIvVega,
            Self::VegaKt(_) => MeasureKind::VegaKt,
            Self::ModelParameters(_) => MeasureKind::ModelParameters,
            Self::Correlations(_) => MeasureKind::Correlations,
        }
    }
}

/// How a measure is estimated. The caller always states it; a backend never
/// substitutes one method for another.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum MeasureMethod {
    /// The backend's own estimator, with no caller-chosen bump.
    Native,
    /// The backend's own second-order estimator: a central difference of its
    /// native Deltas on common random numbers. The bump is part of the request.
    NativeSpotBump { bump: SpotBump },
    /// The backend's own estimator on a calibrated source, under the stated
    /// treatment of the calibration: `Fixed` holds the realized leverage,
    /// `Recalibrated` recalibrates it. `NotApplicable` is refused.
    NativeCalibrated { calibration: BumpCalibration },
    /// The backend's own second-order Spot estimator on a calibrated source,
    /// with the caller's bump and the stated calibration treatment.
    NativeSpotBumpCalibrated {
        bump: SpotBump,
        calibration: BumpCalibration,
    },
    /// Central difference of complete revaluations: the source is rebuilt
    /// with the input moved up and down and each side is priced on the task's
    /// own random coordinates. Only ever chosen by the caller. No sampling
    /// error is reported for it yet.
    BumpAndRevalue {
        size: BumpSize,
        calibration: BumpCalibration,
    },
}

/// The half-width of a central bump: an amount of the input, or a fraction of
/// its current value.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BumpSize {
    Absolute(PositiveF64),
    Relative(PositiveF64),
}

impl BumpSize {
    pub fn absolute(value: f64) -> Result<Self, crate::core::CoreError> {
        PositiveF64::new(value, "bump_size").map(Self::Absolute)
    }
    pub fn relative(value: f64) -> Result<Self, crate::core::CoreError> {
        PositiveF64::new(value, "bump_size").map(Self::Relative)
    }
    /// The half-width in units of an input whose current value is `level`.
    #[must_use]
    pub fn amount(self, level: f64) -> f64 {
        match self {
            Self::Absolute(h) => h.get(),
            Self::Relative(h) => h.get() * level,
        }
    }
}

/// What happens to a source's calibration on the bumped sides. There is no
/// default: the caller states it, and it must fit the source.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum BumpCalibration {
    /// The source has no calibration step.
    NotApplicable,
    /// Reuse the calibration of the unbumped source on both sides.
    Fixed,
    /// Recalibrate on each bumped side with the same calibration seeds.
    Recalibrated,
}

impl BumpCalibration {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::NotApplicable => "not_applicable",
            Self::Fixed => "fixed",
            Self::Recalibrated => "recalibrated",
        }
    }
}

/// What a task's bump-and-revalue requests cost beyond the base evaluation.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct BumpCost {
    /// Calibrations run to build the bumped sides.
    pub calibrations: u64,
    /// Complete price evaluations of bumped sides.
    pub evaluations: u64,
}

/// Identity of a method without its parameters.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum MeasureMethodKind {
    Native,
    NativeSpotBump,
    NativeCalibrated,
    NativeSpotBumpCalibrated,
    BumpAndRevalue,
}

impl MeasureMethodKind {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Native => "native",
            Self::NativeSpotBump => "native_spot_bump",
            Self::NativeCalibrated => "native_calibrated",
            Self::NativeSpotBumpCalibrated => "native_spot_bump_calibrated",
            Self::BumpAndRevalue => "bump_and_revalue",
        }
    }
}

impl MeasureMethod {
    #[must_use]
    pub const fn kind(self) -> MeasureMethodKind {
        match self {
            Self::Native => MeasureMethodKind::Native,
            Self::NativeSpotBump { .. } => MeasureMethodKind::NativeSpotBump,
            Self::NativeCalibrated { .. } => MeasureMethodKind::NativeCalibrated,
            Self::NativeSpotBumpCalibrated { .. } => MeasureMethodKind::NativeSpotBumpCalibrated,
            Self::BumpAndRevalue { .. } => MeasureMethodKind::BumpAndRevalue,
        }
    }
    /// The calibration treatment the caller stated, for the methods that
    /// carry one.
    #[must_use]
    pub const fn calibration(self) -> Option<BumpCalibration> {
        match self {
            Self::Native | Self::NativeSpotBump { .. } => None,
            Self::NativeCalibrated { calibration }
            | Self::NativeSpotBumpCalibrated { calibration, .. }
            | Self::BumpAndRevalue { calibration, .. } => Some(calibration),
        }
    }
    /// Whether the method is evaluated above the backend by revaluation.
    #[must_use]
    pub const fn is_revaluation(self) -> bool {
        matches!(self, Self::BumpAndRevalue { .. })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MeasureRequest {
    pub measure: Measure,
    pub method: MeasureMethod,
}

impl MeasureRequest {
    #[must_use]
    pub const fn new(measure: Measure, method: MeasureMethod) -> Self {
        Self { measure, method }
    }

    /// `measure` with the backend's own estimator.
    #[must_use]
    pub const fn native(measure: Measure) -> Self {
        Self::new(measure, MeasureMethod::Native)
    }

    /// Spot Gamma as a central difference of native Deltas with `bump`.
    #[must_use]
    pub const fn spot_gamma(bump: SpotBump) -> Self {
        Self::new(Measure::SpotGamma, MeasureMethod::NativeSpotBump { bump })
    }
}

/// An ordered list of requests with at most one request per measure. Order is
/// part of the task fingerprint and has no effect on the numbers.
#[derive(Clone, Debug, PartialEq)]
pub struct MeasureSet(Arc<[MeasureRequest]>);

impl MeasureSet {
    pub fn new(requests: Vec<MeasureRequest>) -> Result<Self, SpecificationValidationError> {
        for (i, request) in requests.iter().enumerate() {
            if requests[..i]
                .iter()
                .any(|earlier| earlier.measure.kind() == request.measure.kind())
            {
                return Err(SpecificationValidationError::Invalid {
                    field: "operation.measures",
                    message: format!(
                        "measure {} is requested more than once",
                        request.measure.kind().as_str()
                    ),
                });
            }
        }
        Ok(Self(requests.into()))
    }

    #[must_use]
    pub fn requests(&self) -> &[MeasureRequest] {
        &self.0
    }

    /// The requests the backend itself answers.
    #[must_use]
    pub fn native_requests(&self) -> Vec<MeasureRequest> {
        self.0
            .iter()
            .filter(|r| !r.method.is_revaluation())
            .cloned()
            .collect()
    }
    /// The requests answered by revaluation above the backend.
    #[must_use]
    pub fn revaluation_requests(&self) -> Vec<MeasureRequest> {
        self.0
            .iter()
            .filter(|r| r.method.is_revaluation())
            .cloned()
            .collect()
    }

    #[must_use]
    pub fn get(&self, kind: MeasureKind) -> Option<&MeasureRequest> {
        self.0.iter().find(|r| r.measure.kind() == kind)
    }
}

/// The answer to "can this source produce this measure by this method".
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct MeasureCapability {
    pub request: MeasureRequest,
    pub status: CapabilityStatus,
    pub reason: Option<CapabilityReason>,
    pub detail: String,
}

impl MeasureCapability {
    pub(crate) fn new(
        request: &MeasureRequest,
        status: CapabilityStatus,
        reason: Option<CapabilityReason>,
        detail: impl Into<String>,
    ) -> Self {
        Self {
            request: request.clone(),
            status,
            reason,
            detail: detail.into(),
        }
    }
}

/// One discovery row: a measure the source does not statically refuse, with
/// the methods that apply to it.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct AvailableMeasure {
    pub measure: MeasureKind,
    pub methods: Vec<MeasureMethodKind>,
}
