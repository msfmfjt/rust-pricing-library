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
        }
    }
    /// The scalar parameter of a wire name; `None` for a component name or
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
            _ => return None,
        })
    }
    /// A stable label: the wire name, with the component index for a
    /// component parameter (`component_vol_of_vol.1`).
    #[must_use]
    pub fn label(self) -> String {
        match self {
            Self::ComponentVolOfVol { component } => {
                format!("{}.{component}", self.as_str())
            }
            _ => self.as_str().to_owned(),
        }
    }
}

impl ModelParameterId {
    /// The parameter a `label` names; `None` for an unknown label.
    #[must_use]
    pub fn from_label(label: &str) -> Option<Self> {
        if let Some(index) = label.strip_prefix("component_vol_of_vol.") {
            return index
                .parse()
                .ok()
                .map(|component| Self::ComponentVolOfVol { component });
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
}

impl MeasureKind {
    pub const ALL: [Self; 6] = [
        Self::Price,
        Self::SpotDelta,
        Self::SpotGamma,
        Self::ParallelMarketIvVega,
        Self::VegaKt,
        Self::ModelParameters,
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
