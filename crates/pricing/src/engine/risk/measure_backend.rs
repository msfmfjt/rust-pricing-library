//! The adapter every native plan implements for the `Measures` operation.
//! Adapters call existing evaluations; they add no estimator (ADR 0043).
use crate::MonteCarloError;
use crate::engine::risk::stochastic_dividends::DividendMeasureSelection;
use crate::lsv::{RoughFamilyLsvDeltaConvention, RoughFamilyLsvPricingPlan};
use crate::market::MarketIvSurface;
use crate::mc::lsv::LsvError;
use crate::multi_asset::MultiAssetPricingPlan;
use crate::risk::{GammaConfig, SpotBump};
use crate::rough_volatility::RoughVolatilityPricingPlan;
use crate::stochastic_dividends::{
    StochasticDividendHullWhitePricingPlan, StochasticDividendPricingPlan,
};
use crate::{
    BumpCalibration, CapabilityReason, CapabilityStatus, CompiledPricingSpecification,
    CorrelationId, CorrelationSelection, Measure, MeasureCapability, MeasureKind, MeasureMethod,
    MeasureRequest, ModelParameterId, ModelParameterSelection, PricingTaskError, PricingTaskOutput,
};

pub(crate) trait MeasureBackend {
    /// Whether the realized plan produces `request` as it stands. Never
    /// compiles, calibrates or runs paths.
    fn preflight(&self, request: &MeasureRequest) -> MeasureCapability;

    /// One native evaluation for the whole list. The caller has checked the
    /// list with `validate_measures`.
    fn evaluate_measures(
        &self,
        requests: &[MeasureRequest],
    ) -> Result<PricingTaskOutput, PricingTaskError>;

    /// The list is valid when every request passes preflight. Backends whose
    /// plan fixes a selection also require that nothing else was compiled in.
    fn validate_measures(&self, requests: &[MeasureRequest]) -> Result<(), PricingTaskError> {
        match requests
            .iter()
            .map(|request| self.preflight(request))
            .find(|capability| capability.status != CapabilityStatus::Supported)
        {
            None => Ok(()),
            Some(refused) => Err(unsupported(refused.reason, refused.detail)),
        }
    }
}

fn unsupported(reason: Option<CapabilityReason>, detail: String) -> PricingTaskError {
    PricingTaskError::Validation(crate::SpecificationValidationError::UnsupportedOperation {
        reason,
        detail,
    })
}

fn supported(request: &MeasureRequest) -> MeasureCapability {
    MeasureCapability::new(
        request,
        CapabilityStatus::Supported,
        None,
        "compiled source and native preflight passed; numerical evaluation can still fail",
    )
}

/// A plan that prices and has no measure adapter for anything else yet.
fn price_only(request: &MeasureRequest) -> MeasureCapability {
    if request.measure == Measure::Price && request.method == MeasureMethod::Native {
        return supported(request);
    }
    MeasureCapability::new(
        request,
        CapabilityStatus::Unsupported,
        Some(CapabilityReason::DedicatedRiskApi),
        format!(
            "{} has no measure adapter on this plan yet",
            request.measure.kind().as_str()
        ),
    )
}

/// The rough-family LSV Spot convention a stated calibration selects.
fn convention(calibration: BumpCalibration) -> Option<RoughFamilyLsvDeltaConvention> {
    match calibration {
        BumpCalibration::Fixed => Some(RoughFamilyLsvDeltaConvention::FrozenLeverage),
        BumpCalibration::Recalibrated => {
            Some(RoughFamilyLsvDeltaConvention::StickyRelativeLocalVariance)
        }
        BumpCalibration::NotApplicable => None,
    }
}

/// Rough-family LSV: native Spot Delta and Gamma under the stated
/// calibration treatment, and parallel market-IV Vega through the
/// recalibration, as the dedicated APIs compute them.
fn rough_family_lsv_preflight(
    plan: &RoughFamilyLsvPricingPlan,
    market_iv: Option<&MarketIvSurface>,
    request: &MeasureRequest,
) -> MeasureCapability {
    let refused = |reason, detail: String| {
        MeasureCapability::new(request, CapabilityStatus::Unsupported, Some(reason), detail)
    };
    let calibration = match (&request.measure, request.method) {
        (Measure::Price, MeasureMethod::Native) => return supported(request),
        (Measure::SpotDelta, MeasureMethod::NativeCalibrated { calibration })
        | (Measure::SpotGamma, MeasureMethod::NativeSpotBumpCalibrated { calibration, .. }) => {
            calibration
        }
        (Measure::ParallelMarketIvVega, MeasureMethod::NativeCalibrated { calibration }) => {
            if calibration != BumpCalibration::Recalibrated {
                return refused(
                    CapabilityReason::BumpCalibrationMismatch,
                    "a market-IV move reaches LSV prices only through the calibration; state recalibrated".into(),
                );
            }
            let Some(surface) = market_iv else {
                return refused(
                    CapabilityReason::ModelCombination,
                    "the local-variance target retains no reporting IV basis that reproduces it"
                        .into(),
                );
            };
            return match plan.market_iv_fingerprint(surface) {
                Ok(_) => supported(request),
                Err(MonteCarloError::Lsv(LsvError::ReverseTraceNotRetained)) => refused(
                    CapabilityReason::ReverseTraceRequired,
                    "native market-IV Vega differentiates through the calibration and requires retain_reverse_trace=true".into(),
                ),
                Err(MonteCarloError::UnsupportedRiskForModel { model }) => {
                    refused(CapabilityReason::DiscontinuousPayoff, model.into())
                }
                Err(e) => refused(CapabilityReason::ModelCombination, e.to_string()),
            };
        }
        (Measure::ModelParameters(selection), MeasureMethod::NativeCalibrated { calibration }) => {
            return parameter_preflight(plan, selection, calibration, request);
        }
        _ => return price_only(request),
    };
    if convention(calibration).is_none() {
        return refused(
            CapabilityReason::BumpCalibrationMismatch,
            "this source is calibrated; state fixed (frozen leverage) or recalibrated (sticky relative local variance)".into(),
        );
    }
    if !plan.supports_spot_risk() {
        return refused(
            CapabilityReason::DiscontinuousPayoff,
            "native Spot risk requires a differentiable payoff or explicit smoothing".into(),
        );
    }
    supported(request)
}

/// Rough-family LSV parameter risk: recalibrated only, on a model with a
/// native estimator, for parameters that model has.
fn parameter_preflight(
    plan: &RoughFamilyLsvPricingPlan,
    selection: &ModelParameterSelection,
    calibration: BumpCalibration,
    request: &MeasureRequest,
) -> MeasureCapability {
    let refused = |reason, detail: String| {
        MeasureCapability::new(request, CapabilityStatus::Unsupported, Some(reason), detail)
    };
    if calibration != BumpCalibration::Recalibrated {
        return refused(
            CapabilityReason::BumpCalibrationMismatch,
            PARAMETER_CALIBRATION.into(),
        );
    }
    let Some((available, hurst)) = plan.native_parameters() else {
        return refused(
            CapabilityReason::NoNativeEstimator,
            "this rough-family model has no native parameter risk".into(),
        );
    };
    if let Some(detail) = missing_parameter(selection, &available, hurst) {
        return refused(CapabilityReason::ModelCombination, detail);
    }
    match plan.parameter_preflight() {
        Ok(()) => supported(request),
        Err(MonteCarloError::Lsv(LsvError::ReverseTraceNotRetained)) => refused(
            CapabilityReason::ReverseTraceRequired,
            PARAMETER_TRACE.into(),
        ),
        Err(MonteCarloError::UnsupportedRiskForModel { model }) => {
            refused(CapabilityReason::DiscontinuousPayoff, model.into())
        }
        Err(e) => refused(CapabilityReason::ModelCombination, e.to_string()),
    }
}

pub(crate) const PARAMETER_CALIBRATION: &str = "the native parameter risk differentiates through the particle recalibration at the fixed target; state recalibrated (its fixed-leverage part is in the native result, without an error)";
pub(crate) const PARAMETER_TRACE: &str = "native parameter risk differentiates through the calibration and requires retain_reverse_trace=true";

/// The detail of the first requested parameter the model lacks.
pub(crate) fn missing_parameter(
    selection: &ModelParameterSelection,
    available: &[ModelParameterId],
    hurst: bool,
) -> Option<String> {
    let names = || {
        available
            .iter()
            .map(|p| p.label())
            .chain(hurst.then(|| ModelParameterId::Hurst.label()))
            .collect::<Vec<_>>()
            .join(", ")
    };
    selection
        .parameters()
        .iter()
        .find(|&&p| !(available.contains(&p) || (hurst && p == ModelParameterId::Hurst)))
        .map(|p| {
            format!(
                "this model has no native parameter {}; available: {}",
                p.label(),
                names()
            )
        })
}

/// `None` when the list asks for no native risk measure.
fn rough_family_lsv_evaluate(
    plan: &RoughFamilyLsvPricingPlan,
    market_iv: Option<&MarketIvSurface>,
    requests: &[MeasureRequest],
) -> Result<Option<PricingTaskOutput>, PricingTaskError> {
    let (mut delta, mut gamma, mut vega, mut parameters) = (None, None, None, None);
    for request in requests {
        match (&request.measure, request.method) {
            (Measure::SpotDelta, MeasureMethod::NativeCalibrated { calibration }) => {
                delta = convention(calibration);
            }
            (Measure::SpotGamma, MeasureMethod::NativeSpotBumpCalibrated { bump, calibration }) => {
                let amount = spot_amount(bump, plan.spot());
                gamma = convention(calibration).map(|c| (amount, c));
            }
            (Measure::ParallelMarketIvVega, MeasureMethod::NativeCalibrated { .. }) => {
                vega = market_iv;
            }
            (Measure::ModelParameters(selection), MeasureMethod::NativeCalibrated { .. }) => {
                parameters = Some(selection.parameters().contains(&ModelParameterId::Hurst));
            }
            _ => {}
        }
    }
    if delta.is_none() && gamma.is_none() && vega.is_none() && parameters.is_none() {
        return Ok(None);
    }
    Ok(Some(PricingTaskOutput::RoughFamilyLsvMeasureRisk(
        Box::new(plan.evaluate_measure_risk(delta, gamma, vega, parameters)?),
    )))
}

/// The absolute physical Spot amount a native Gamma bump resolves to.
fn spot_amount(bump: SpotBump, spot: f64) -> f64 {
    match bump {
        SpotBump::Absolute(h) => h.get(),
        SpotBump::Relative(h) => h.get() * spot,
    }
}

pub(crate) const DIRECT_PARAMETERS_MISSING: &str = "has no native fixed-model parameter risk";

/// Direct rough volatility: the fixed-model Spot Delta, the explicitly
/// bumped Spot Gamma and the fixed-model parameter risks, as the dedicated
/// APIs compute them. Nothing is calibrated, so no treatment is stated.
fn rough_volatility_preflight(
    plan: &RoughVolatilityPricingPlan,
    request: &MeasureRequest,
) -> MeasureCapability {
    let refused = |reason, detail: String| {
        MeasureCapability::new(request, CapabilityStatus::Unsupported, Some(reason), detail)
    };
    match (&request.measure, request.method) {
        (Measure::Price, MeasureMethod::Native) => return supported(request),
        (Measure::SpotDelta, MeasureMethod::Native)
        | (Measure::SpotGamma, MeasureMethod::NativeSpotBump { .. }) => {}
        (Measure::ModelParameters(selection), MeasureMethod::Native) => {
            let model = plan.rough_model();
            let Some((available, hurst)) = model.native_parameters() else {
                return refused(
                    CapabilityReason::NoNativeEstimator,
                    format!("{} {DIRECT_PARAMETERS_MISSING}", model.name()),
                );
            };
            if let Some(detail) = missing_parameter(selection, &available, hurst) {
                return refused(CapabilityReason::ModelCombination, detail);
            }
        }
        _ => return price_only(request),
    }
    if !plan.supports_spot_risk() {
        return refused(
            CapabilityReason::DiscontinuousPayoff,
            "native risk requires a differentiable payoff or explicit smoothing".into(),
        );
    }
    supported(request)
}

/// `None` when the list asks for no native risk measure.
fn rough_volatility_evaluate(
    plan: &RoughVolatilityPricingPlan,
    requests: &[MeasureRequest],
) -> Result<Option<PricingTaskOutput>, PricingTaskError> {
    let (mut delta, mut gamma, mut parameters) = (false, None, None);
    for request in requests {
        match (&request.measure, request.method) {
            (Measure::SpotDelta, MeasureMethod::Native) => delta = true,
            (Measure::SpotGamma, MeasureMethod::NativeSpotBump { bump }) => {
                gamma = Some(spot_amount(bump, plan.spot()));
            }
            (Measure::ModelParameters(selection), MeasureMethod::Native) => {
                parameters = Some(selection.parameters());
            }
            _ => {}
        }
    }
    if !delta && gamma.is_none() && parameters.is_none() {
        return Ok(None);
    }
    Ok(Some(PricingTaskOutput::RoughVolatilityMeasureRisk(
        Box::new(plan.evaluate_measure_risk(delta, gamma, parameters)?),
    )))
}

impl MeasureBackend for CompiledPricingSpecification {
    fn preflight(&self, request: &MeasureRequest) -> MeasureCapability {
        if let Some(plan) = self.rough_family_lsv() {
            return rough_family_lsv_preflight(plan, self.market_iv(), request);
        }
        if let Some(plan) = self.rough_volatility() {
            return rough_volatility_preflight(plan, request);
        }
        let Some(risk) = self.standard_risk() else {
            return price_only(request);
        };
        let compiled = match (&request.measure, request.method) {
            (Measure::Price, MeasureMethod::Native) => true,
            (Measure::SpotDelta, MeasureMethod::Native) => risk.delta(),
            (Measure::SpotGamma, MeasureMethod::NativeSpotBump { bump }) => {
                risk.gamma() == Some(GammaConfig::new(bump))
            }
            (Measure::ParallelMarketIvVega, MeasureMethod::Native) => risk.vega(),
            (Measure::VegaKt(config), MeasureMethod::Native) => risk.vega_kt() == Some(config),
            _ => false,
        };
        if compiled {
            supported(request)
        } else {
            MeasureCapability::new(
                request,
                CapabilityStatus::Unsupported,
                Some(CapabilityReason::NativePreflightRequired),
                format!(
                    "{} by {} was not selected when this standard plan was compiled; compile a task that requests it",
                    request.measure.kind().as_str(),
                    request.method.kind().as_str()
                ),
            )
        }
    }

    fn validate_measures(&self, requests: &[MeasureRequest]) -> Result<(), PricingTaskError> {
        if let Some(refused) = requests
            .iter()
            .map(|request| self.preflight(request))
            .find(|capability| capability.status != CapabilityStatus::Supported)
        {
            return Err(unsupported(refused.reason, refused.detail));
        }
        // The plan must hold nothing beyond the list, or the result would
        // carry Greeks the task did not ask for.
        if let Some(risk) = self.standard_risk() {
            let listed = |kind| requests.iter().any(|r| r.measure.kind() == kind);
            if risk.delta() != listed(MeasureKind::SpotDelta)
                || risk.gamma().is_some() != listed(MeasureKind::SpotGamma)
                || risk.vega() != listed(MeasureKind::ParallelMarketIvVega)
                || risk.vega_kt().is_some() != listed(MeasureKind::VegaKt)
            {
                return Err(super::specification::reselected());
            }
        }
        Ok(())
    }

    fn evaluate_measures(
        &self,
        requests: &[MeasureRequest],
    ) -> Result<PricingTaskOutput, PricingTaskError> {
        if let Some(plan) = self.rough_family_lsv()
            && let Some(output) = rough_family_lsv_evaluate(plan, self.market_iv(), requests)?
        {
            return Ok(output);
        }
        if let Some(plan) = self.rough_volatility()
            && let Some(output) = rough_volatility_evaluate(plan, requests)?
        {
            return Ok(output);
        }
        Ok(PricingTaskOutput::SingleAsset(self.evaluate()?))
    }
}

pub(crate) const BASKET_FIXED: &str = "basket LSV marginals are calibrated in forward-normalized coordinates, where the native AAD is the recalibrated (sticky-moneyness) derivative; there is no frozen-leverage estimator, so state recalibrated";
pub(crate) const BASKET_CALIBRATED: &str =
    "this basket is calibrated (LSV or Local Correlation); state recalibrated";
pub(crate) const BASKET_DETERMINISTIC_RATES: &str = "the basket AAD differentiates the discount curve only under shared Hull-White rates; a deterministic-rate basket has no native discount-curve risk";
pub(crate) const BASKET_UNCALIBRATED: &str =
    "this basket has no calibration step; omit the calibration treatment";

/// Basket AAD Delta and cross Gamma: the method a basket's calibration
/// admits and the relative bump the native Gamma takes. `None` when the
/// request passes, or is not one of these measures.
pub(crate) fn basket_refusal(
    calibrated: bool,
    request: &MeasureRequest,
) -> Option<(CapabilityReason, String)> {
    let (calibration, bump) = match (&request.measure, request.method) {
        (Measure::SpotDelta | Measure::DiscountCurve, MeasureMethod::Native) => (None, None),
        (
            Measure::SpotDelta | Measure::DiscountCurve,
            MeasureMethod::NativeCalibrated { calibration },
        ) => (Some(calibration), None),
        (Measure::CrossGamma, MeasureMethod::NativeSpotBump { bump }) => (None, Some(bump)),
        (Measure::CrossGamma, MeasureMethod::NativeSpotBumpCalibrated { bump, calibration }) => {
            (Some(calibration), Some(bump))
        }
        _ => return None,
    };
    match (calibrated, calibration) {
        (false, None) | (true, Some(BumpCalibration::Recalibrated)) => {}
        (true, Some(BumpCalibration::Fixed)) => {
            return Some((
                CapabilityReason::FixedCalibrationUnavailable,
                BASKET_FIXED.into(),
            ));
        }
        (true, _) => {
            return Some((
                CapabilityReason::BumpCalibrationMismatch,
                BASKET_CALIBRATED.into(),
            ));
        }
        (false, Some(_)) => {
            return Some((
                CapabilityReason::BumpCalibrationMismatch,
                BASKET_UNCALIBRATED.into(),
            ));
        }
    }
    match bump {
        None => None,
        Some(SpotBump::Relative(h)) if h.get() < 1.0 => None,
        Some(SpotBump::Relative(_)) => Some((
            CapabilityReason::InvalidGammaBump,
            "basket Gamma relative bump must lie in (0,1)".into(),
        )),
        Some(SpotBump::Absolute(_)) => Some((
            CapabilityReason::ModelCombination,
            "basket cross Gamma bumps every Spot by the same relative amount; state a relative bump".into(),
        )),
    }
}

/// The AAD configuration a validated list asks for; `None` when it asks for
/// no native risk.
fn basket_selection(
    requests: &[MeasureRequest],
) -> Option<crate::multi_asset::MultiAssetRiskConfig> {
    let mut config = None;
    for request in requests {
        match (&request.measure, request.method) {
            (
                Measure::SpotDelta | Measure::DiscountCurve,
                MeasureMethod::Native | MeasureMethod::NativeCalibrated { .. },
            ) => {
                config.get_or_insert_with(crate::multi_asset::MultiAssetRiskConfig::default);
            }
            (
                Measure::CrossGamma,
                MeasureMethod::NativeSpotBump {
                    bump: SpotBump::Relative(h),
                }
                | MeasureMethod::NativeSpotBumpCalibrated {
                    bump: SpotBump::Relative(h),
                    ..
                },
            ) => {
                config
                    .get_or_insert_with(crate::multi_asset::MultiAssetRiskConfig::default)
                    .gamma_relative_bump = Some(h.get());
            }
            _ => {}
        }
    }
    config
}

/// Baskets: the native AAD Delta of every asset and the central bump of
/// AAD Deltas, as `evaluate_aad` computes them.
impl MeasureBackend for MultiAssetPricingPlan {
    fn preflight(&self, request: &MeasureRequest) -> MeasureCapability {
        let refused = |reason, detail: String| {
            MeasureCapability::new(request, CapabilityStatus::Unsupported, Some(reason), detail)
        };
        let config = match basket_selection(std::slice::from_ref(request)) {
            Some(config) => config,
            None => return price_only(request),
        };
        if let Some((reason, detail)) = basket_refusal(self.is_calibrated(), request) {
            return refused(reason, detail);
        }
        if request.measure == Measure::DiscountCurve && !self.has_hull_white() {
            return refused(
                CapabilityReason::NoNativeEstimator,
                BASKET_DETERMINISTIC_RATES.into(),
            );
        }
        match self.validate_aad(config) {
            Ok(()) => supported(request),
            Err(crate::multi_asset::MultiAssetError::Invalid(detail)) => refused(
                if detail.contains("retain_reverse_trace") {
                    CapabilityReason::ReverseTraceRequired
                } else if detail.contains("discontinuous") {
                    CapabilityReason::DiscontinuousPayoff
                } else if detail.contains("Gamma") {
                    CapabilityReason::InvalidGammaBump
                } else {
                    CapabilityReason::ModelCombination
                },
                detail.into(),
            ),
            Err(e) => refused(CapabilityReason::ModelCombination, e.to_string()),
        }
    }
    fn evaluate_measures(
        &self,
        requests: &[MeasureRequest],
    ) -> Result<PricingTaskOutput, PricingTaskError> {
        let price = match basket_selection(requests) {
            Some(config) => self.evaluate_aad(config)?,
            None => self.evaluate()?,
        };
        Ok(PricingTaskOutput::MultiAsset(Box::new(price)))
    }
}

pub(crate) const DIVIDEND_LSV_FIXED: &str = "the residual-equity LSV Spot estimators re-anchor the calibrated surface (recalibrated by scale invariance); there is no frozen-leverage estimator, so state recalibrated, or use bump_and_revalue with fixed";
pub(crate) const DIVIDEND_LSV_CALIBRATION: &str = "this source is calibrated; state recalibrated (the scale-invariant residual-equity LSV estimator)";

/// What a stochastic-dividend source offers the measures facade.
struct DividendSource {
    calibrated: bool,
    pathwise: bool,
    correlations: Vec<CorrelationId>,
    parameters: Vec<ModelParameterId>,
    cash_dividends: usize,
}

/// Stochastic dividends: AAD Delta, the bump of AAD Deltas, correlation and
/// model-parameter AAD, or on a residual-equity LSV the scale-invariant
/// (recalibrated) Delta and Gamma.
fn stochastic_dividend_preflight(
    source: &DividendSource,
    request: &MeasureRequest,
) -> MeasureCapability {
    let DividendSource {
        calibrated,
        pathwise,
        ref correlations,
        ref parameters,
        cash_dividends,
    } = *source;
    let refused = |reason, detail: String| {
        MeasureCapability::new(request, CapabilityStatus::Unsupported, Some(reason), detail)
    };
    match (calibrated, &request.measure, request.method) {
        (_, Measure::Price, MeasureMethod::Native) => return supported(request),
        (
            false,
            Measure::SpotDelta | Measure::DividendCash | Measure::DiscountCurve,
            MeasureMethod::Native,
        )
        | (false, Measure::SpotGamma, MeasureMethod::NativeSpotBump { .. }) => {}
        (false, Measure::Correlations(selection), MeasureMethod::Native) => {
            if let Some(detail) = missing_correlation(selection, correlations) {
                return refused(CapabilityReason::ModelCombination, detail);
            }
        }
        (false, Measure::ModelParameters(selection), MeasureMethod::Native) => {
            if let Some(detail) = missing_parameter(selection, parameters, false) {
                return refused(CapabilityReason::ModelCombination, detail);
            }
        }
        (
            true,
            Measure::SpotDelta | Measure::DividendCash | Measure::DiscountCurve,
            MeasureMethod::NativeCalibrated { calibration },
        )
        | (true, Measure::SpotGamma, MeasureMethod::NativeSpotBumpCalibrated { calibration, .. }) => {
            match calibration {
                BumpCalibration::Recalibrated => {}
                BumpCalibration::Fixed => {
                    return refused(
                        CapabilityReason::NoNativeEstimator,
                        DIVIDEND_LSV_FIXED.into(),
                    );
                }
                BumpCalibration::NotApplicable => {
                    return refused(
                        CapabilityReason::BumpCalibrationMismatch,
                        DIVIDEND_LSV_CALIBRATION.into(),
                    );
                }
            }
        }
        _ => return price_only(request),
    }
    if request.measure == Measure::DividendCash && cash_dividends == 0 {
        return refused(CapabilityReason::ModelCombination, NO_CASH_DIVIDENDS.into());
    }
    if !pathwise {
        return refused(
            CapabilityReason::DiscontinuousPayoff,
            "native risk requires a differentiable payoff or explicit smoothing".into(),
        );
    }
    supported(request)
}

pub(crate) const NO_CASH_DIVIDENDS: &str =
    "the market schedules no cash dividends, so there is no cash-dividend risk";

/// The detail of the first requested correlation the source lacks.
pub(crate) fn missing_correlation(
    selection: &CorrelationSelection,
    available: &[CorrelationId],
) -> Option<String> {
    selection
        .correlations()
        .iter()
        .find(|c| !available.contains(c))
        .map(|c| {
            format!(
                "this source has no correlation {}; available: {}",
                c.label(),
                available
                    .iter()
                    .map(|a| a.label())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })
}

/// The native selection of a validated list.
fn dividend_selection(requests: &[MeasureRequest]) -> DividendMeasureSelection<'_> {
    let mut selection = DividendMeasureSelection::default();
    for request in requests {
        match (&request.measure, request.method) {
            (Measure::Correlations(_), MeasureMethod::Native) => selection.correlations = true,
            (Measure::ModelParameters(listed), MeasureMethod::Native) => {
                selection.parameters = Some(listed.parameters());
            }
            (
                Measure::SpotDelta,
                MeasureMethod::Native | MeasureMethod::NativeCalibrated { .. },
            ) => selection.delta = true,
            (
                Measure::DividendCash,
                MeasureMethod::Native | MeasureMethod::NativeCalibrated { .. },
            ) => selection.dividend_cash = true,
            (
                Measure::DiscountCurve,
                MeasureMethod::Native | MeasureMethod::NativeCalibrated { .. },
            ) => selection.discount_curve = true,
            (
                Measure::SpotGamma,
                MeasureMethod::NativeSpotBump { bump }
                | MeasureMethod::NativeSpotBumpCalibrated { bump, .. },
            ) => selection.gamma = Some(GammaConfig::new(bump)),
            _ => {}
        }
    }
    selection
}

impl MeasureBackend for StochasticDividendPricingPlan {
    fn preflight(&self, request: &MeasureRequest) -> MeasureCapability {
        let source = DividendSource {
            calibrated: self.is_calibrated(),
            pathwise: self.supports_spot_risk(),
            correlations: self.correlation_ids(),
            parameters: self.parameter_ids(),
            cash_dividends: self.cash_dividend_count(),
        };
        stochastic_dividend_preflight(&source, request)
    }
    fn evaluate_measures(
        &self,
        requests: &[MeasureRequest],
    ) -> Result<PricingTaskOutput, PricingTaskError> {
        let selection = dividend_selection(requests);
        if selection.is_empty() {
            return Ok(PricingTaskOutput::StochasticDividend(self.evaluate()?));
        }
        Ok(PricingTaskOutput::StochasticDividendMeasureRisk(Box::new(
            self.evaluate_measure_risk(selection)?,
        )))
    }
}

impl MeasureBackend for StochasticDividendHullWhitePricingPlan {
    fn preflight(&self, request: &MeasureRequest) -> MeasureCapability {
        let source = DividendSource {
            calibrated: false,
            pathwise: self.supports_spot_risk(),
            correlations: self.correlation_ids(),
            parameters: self.parameter_ids(),
            cash_dividends: self.cash_dividend_count(),
        };
        stochastic_dividend_preflight(&source, request)
    }
    fn evaluate_measures(
        &self,
        requests: &[MeasureRequest],
    ) -> Result<PricingTaskOutput, PricingTaskError> {
        let selection = dividend_selection(requests);
        if selection.is_empty() {
            return Ok(PricingTaskOutput::StochasticDividend(self.evaluate()?));
        }
        Ok(PricingTaskOutput::StochasticDividendMeasureRisk(Box::new(
            self.evaluate_measure_risk(selection)?,
        )))
    }
}

#[cfg(test)]
mod rough_family_lsv_tests {
    use serde_json::{Value, json};

    use crate::lsv::RoughFamilyLsvPricingPlan;
    use crate::risk::SpotBump;
    use crate::*;

    fn source(rqmc: bool, payoff: Option<Value>) -> PricingSpecification {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/specification-v1/bergomi_lsv.golden.json");
        let mut v: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        let values = (0..15)
            .map(|i| 0.04 + 0.006 * f64::from(2 - i % 3) + 0.001 * f64::from(i / 3))
            .collect::<Vec<_>>();
        v["model"]["equity"] = json!({
            "type":"rough_local_stochastic_volatility",
            "target":{"local_variance_grid":{
                "time_nodes":[0.0,0.25,0.5,0.75,1.0],
                "log_forward_moneyness_nodes":[-0.6,0.0,0.6],
                "shape":[5,3],"values":values,"floor":1e-8,"cap":4.0}},
            "model":{"type":"rough_heston","hurst":0.2,"initial_variance":0.04,
                "mean_reversion":0.7,"long_run_variance":0.055,"vol_of_vol":0.15,
                "correlation":-0.6},
            "particles":{"particle_count":64,"seed":429,"log_bandwidth":0.4,
                "minimum_effective_samples":2.0,"retain_reverse_trace":false}});
        if rqmc {
            v["engine"] = json!({"type":"randomized_quasi_monte_carlo",
                "points_per_scramble":32,"scramble_count":8,"master_scramble_seed":819,
                "variance_reduction":{"antithetic":true,"brownian_bridge":true}});
        } else {
            v["engine"]["independent_sampling_units"] = json!(256);
        }
        if let Some(product) = payoff {
            v["product"] = product;
        }
        parse_specification_json(&serde_json::to_vec(&v).unwrap(), JsonLimits::DEFAULT).unwrap()
    }

    fn native(s: &PricingSpecification) -> RoughFamilyLsvPricingPlan {
        let EquityModelSpec::RoughLocalStochasticVolatility {
            model, particles, ..
        } = &s.model().equity
        else {
            unreachable!()
        };
        RoughFamilyLsvPricingPlan::compile(
            s.carrier_request(),
            model.clone(),
            particles.clone(),
            s.numerics().execution_policy,
        )
        .unwrap()
    }

    fn delta(calibration: BumpCalibration) -> MeasureRequest {
        MeasureRequest::new(
            Measure::SpotDelta,
            MeasureMethod::NativeCalibrated { calibration },
        )
    }
    fn gamma(calibration: BumpCalibration) -> MeasureRequest {
        MeasureRequest::new(
            Measure::SpotGamma,
            MeasureMethod::NativeSpotBumpCalibrated {
                bump: SpotBump::relative(0.01).unwrap(),
                calibration,
            },
        )
    }
    fn task(s: &PricingSpecification, requests: Vec<MeasureRequest>) -> PricingTask {
        PricingTask::new(
            s.clone(),
            PricingOperation::Measures(MeasureSet::new(requests).unwrap()),
        )
        .unwrap()
    }

    /// Each row is the dedicated API's estimate and error, bit for bit, under
    /// both conventions and both samplers; the price is the plan's price.
    #[test]
    fn spot_rows_equal_the_dedicated_apis() {
        for rqmc in [false, true] {
            let s = source(rqmc, None);
            let plan = native(&s);
            let price = plan.evaluate().unwrap();
            for calibration in [BumpCalibration::Fixed, BumpCalibration::Recalibrated] {
                let (d, g, treatment) = match calibration {
                    BumpCalibration::Fixed => (
                        plan.evaluate_frozen_leverage_delta().unwrap(),
                        plan.evaluate_frozen_leverage_gamma_bump(1.0).unwrap(),
                        CalibrationTreatment::FixedCalibration,
                    ),
                    _ => (
                        plan.evaluate_sticky_moneyness_delta().unwrap(),
                        plan.evaluate_sticky_moneyness_gamma_bump(1.0).unwrap(),
                        CalibrationTreatment::Recalibrated,
                    ),
                };
                let result = task(&s, vec![delta(calibration), gamma(calibration)])
                    .compile()
                    .unwrap()
                    .evaluate()
                    .unwrap();
                let report = &result.report;
                assert_eq!(report.value().to_bits(), price.value.to_bits());
                assert_eq!(
                    report.standard_error().to_bits(),
                    price.standard_error.to_bits()
                );
                for (kind, value, error, method) in [
                    (
                        MeasureKind::SpotDelta,
                        d.delta,
                        d.delta_standard_error,
                        d.method,
                    ),
                    (
                        MeasureKind::SpotGamma,
                        g.gamma,
                        g.gamma_standard_error,
                        g.method,
                    ),
                ] {
                    let row = report.risk(kind).unwrap();
                    assert_eq!(row.values[0].to_bits(), value.to_bits(), "{kind:?}");
                    assert_eq!(
                        row.standard_errors.as_ref().unwrap()[0].to_bits(),
                        error.to_bits(),
                        "{kind:?}"
                    );
                    assert_eq!(row.method, method);
                    assert_eq!(row.calibration, treatment);
                }
                let PricingTaskOutput::RoughFamilyLsvMeasureRisk(native) = &result.native_result
                else {
                    panic!("native Spot risk expected")
                };
                assert_eq!(native.delta.as_ref(), Some(&d));
                assert_eq!(native.gamma.as_ref(), Some(&g));
            }
            // Gamma alone still reports the plan's price.
            let report = task(&s, vec![gamma(BumpCalibration::Recalibrated)])
                .compile()
                .unwrap()
                .evaluate()
                .unwrap()
                .report;
            assert_eq!(report.value().to_bits(), price.value.to_bits());
        }
    }

    /// One compiled plan serves price-only and Spot selections: the
    /// rough-family plan fixes no risk selection at compilation.
    #[test]
    fn reselection_reuses_the_plan() {
        let s = source(false, None);
        let compiled = task(&s, vec![MeasureRequest::native(Measure::Price)])
            .compile()
            .unwrap();
        let reselected = compiled
            .with_operation(PricingOperation::Measures(
                MeasureSet::new(vec![delta(BumpCalibration::Fixed)]).unwrap(),
            ))
            .unwrap();
        assert_eq!(compiled.plan_fingerprint(), reselected.plan_fingerprint());
        let direct = task(&s, vec![delta(BumpCalibration::Fixed)])
            .compile()
            .unwrap()
            .evaluate()
            .unwrap()
            .report;
        let reused = reselected.evaluate().unwrap().report;
        assert_eq!(
            reused.risk(MeasureKind::SpotDelta).unwrap().values,
            direct.risk(MeasureKind::SpotDelta).unwrap().values
        );
    }

    #[test]
    fn calibration_must_be_stated_and_the_payoff_differentiable() {
        let s = source(false, None);
        let source: PricingSource = s.clone().into();
        assert_eq!(source.family(), SourceFamily::RoughFamilyLsv);
        let refused = |request: MeasureRequest| {
            let c = source.measure_capabilities(std::slice::from_ref(&request));
            assert_eq!(c[0].status, CapabilityStatus::Unsupported, "{request:?}");
            assert!(fails(&s, request));
            c[0].reason
        };
        assert_eq!(
            refused(delta(BumpCalibration::NotApplicable)),
            Some(CapabilityReason::BumpCalibrationMismatch)
        );
        assert_eq!(
            refused(gamma(BumpCalibration::NotApplicable)),
            Some(CapabilityReason::BumpCalibrationMismatch)
        );
        // The uncalibrated native methods are not this family's.
        assert_eq!(
            refused(MeasureRequest::native(Measure::SpotDelta)),
            Some(CapabilityReason::NoNativeEstimator)
        );
        assert_eq!(
            refused(MeasureRequest::spot_gamma(
                SpotBump::relative(0.01).unwrap()
            )),
            Some(CapabilityReason::NoNativeEstimator)
        );
        let digital = source_with_digital();
        let c = PricingSource::from(digital.clone())
            .measure_capabilities(&[delta(BumpCalibration::Recalibrated)]);
        assert_eq!(c[0].reason, Some(CapabilityReason::DiscontinuousPayoff));
        assert!(fails(&digital, delta(BumpCalibration::Recalibrated)));
        // Bergomi LSV has no native Spot estimator.
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/specification-v1/bergomi_lsv.golden.json");
        let bergomi: PricingSource =
            parse_specification_json(&std::fs::read(path).unwrap(), JsonLimits::DEFAULT)
                .unwrap()
                .into();
        assert_eq!(bergomi.family(), SourceFamily::LocalStochasticVolatility);
        assert_eq!(
            bergomi.measure_capabilities(&[delta(BumpCalibration::Fixed)])[0].reason,
            Some(CapabilityReason::NoNativeEstimator)
        );
    }

    /// The task is refused when it is built or, failing that, compiled.
    fn fails(s: &PricingSpecification, request: MeasureRequest) -> bool {
        PricingTask::new(
            s.clone(),
            PricingOperation::Measures(MeasureSet::new(vec![request]).unwrap()),
        )
        .map_or(true, |t| t.compile().is_err())
    }

    fn source_with_digital() -> PricingSpecification {
        source(
            false,
            Some(json!({"type":"digital","underlying_id":1,"currency_id":2,
                "expiry":"2027-09-04","strike":100.0,"payout":1.0,
                "side":{"type":"call"},"payout_kind":{"type":"cash"},
                "payment_date":"2027-09-04"})),
        )
    }

    #[test]
    fn task_and_report_round_trip() {
        let s = source(false, None);
        let t = task(
            &s,
            vec![delta(BumpCalibration::Fixed), gamma(BumpCalibration::Fixed)],
        );
        let text = pricing_task_to_json(&t).unwrap();
        assert!(text.contains("\"native_calibrated\""), "{text}");
        assert!(text.contains("\"native_spot_bump_calibrated\""), "{text}");
        let parsed = parse_pricing_task_json(text.as_bytes(), JsonLimits::DEFAULT).unwrap();
        assert_eq!(parsed.fingerprint().unwrap(), t.fingerprint().unwrap());
        let report = t.compile().unwrap().evaluate().unwrap().report;
        let text = valuation_report_to_json(&report).unwrap();
        let back = parse_valuation_report_json(text.as_bytes(), JsonLimits::DEFAULT).unwrap();
        assert_eq!(back, report);
    }
}

#[cfg(test)]
pub(crate) mod rough_family_lsv_market_iv_tests {
    use serde_json::{Value, json};

    use crate::lsv::RoughFamilyLsvPricingPlan;
    use crate::market::MarketIvSurface;
    use crate::*;

    const TIMES: [f64; 3] = [0.2, 0.6, 1.2];
    const XS: [f64; 5] = [-0.9, -0.45, 0.05, 0.5, 0.9];

    pub(crate) fn quotes() -> Vec<f64> {
        (0..15)
            .map(|i| {
                let x = XS[i % 5];
                0.2 + 0.002 * (i / 5) as f64 - 0.003 * x + 0.002 * x * x
            })
            .collect()
    }

    /// A skewed market-IV surface, its strict Dupire target and, unless
    /// `basis` is `None`, a reporting basis holding the quotes given.
    pub(crate) fn json_source(rqmc: bool, basis: Option<Vec<f64>>, trace: bool) -> Value {
        let surface = MarketIvSurface::new(TIMES.to_vec(), XS.to_vec(), quotes()).unwrap();
        let grid = surface
            .local_variance_grid(
                vec![0.0, 0.25, 0.5, 0.75, 1.0],
                vec![-0.55, -0.13, 0.18, 0.6],
                1e-8,
                4.0,
            )
            .unwrap();
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/specification-v1/bergomi_lsv.golden.json");
        let mut v: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        let mut target = json!({"local_variance_grid":{
            "time_nodes":grid.time_nodes(),
            "log_forward_moneyness_nodes":grid.log_moneyness_nodes(),
            "shape":[grid.time_nodes().len(), grid.log_moneyness_nodes().len()],
            "values":grid.values(),"floor":grid.floor(),"cap":grid.cap()}});
        if let Some(ivs) = basis {
            target["reporting_iv_basis"] = json!({"maturity_nodes":TIMES,
                "log_forward_moneyness_nodes":XS,"shape":[3,5],"implied_volatilities":ivs});
        }
        v["model"]["equity"] = json!({
            "type":"rough_local_stochastic_volatility",
            "target":target,
            "model":{"type":"rough_heston","hurst":0.2,"initial_variance":0.04,
                "mean_reversion":0.7,"long_run_variance":0.055,"vol_of_vol":0.15,
                "correlation":-0.6},
            "particles":{"particle_count":64,"seed":429,"log_bandwidth":0.4,
                "minimum_effective_samples":2.0,"retain_reverse_trace":trace}});
        if rqmc {
            v["engine"] = json!({"type":"randomized_quasi_monte_carlo",
                "points_per_scramble":32,"scramble_count":8,"master_scramble_seed":819,
                "variance_reduction":{"antithetic":true,"brownian_bridge":true}});
        } else {
            v["engine"]["independent_sampling_units"] = json!(256);
        }
        v
    }

    pub(crate) fn source(rqmc: bool, basis: Option<Vec<f64>>, trace: bool) -> PricingSpecification {
        let v = json_source(rqmc, basis, trace);
        parse_specification_json(&serde_json::to_vec(&v).unwrap(), JsonLimits::DEFAULT).unwrap()
    }

    pub(crate) fn vega(calibration: BumpCalibration) -> MeasureRequest {
        MeasureRequest::new(
            Measure::ParallelMarketIvVega,
            MeasureMethod::NativeCalibrated { calibration },
        )
    }

    fn task(s: &PricingSpecification, requests: Vec<MeasureRequest>) -> PricingTask {
        PricingTask::new(
            s.clone(),
            PricingOperation::Measures(MeasureSet::new(requests).unwrap()),
        )
        .unwrap()
    }

    fn native(s: &PricingSpecification) -> RoughFamilyLsvPricingPlan {
        let EquityModelSpec::RoughLocalStochasticVolatility {
            model, particles, ..
        } = &s.model().equity
        else {
            unreachable!()
        };
        RoughFamilyLsvPricingPlan::compile(
            s.carrier_request(),
            model.clone(),
            particles.clone(),
            s.numerics().execution_policy,
        )
        .unwrap()
    }

    /// The Vega row is the dedicated market-IV risk's parallel Vega and error
    /// bit for bit, alone or beside a Spot Delta, under both samplers.
    #[test]
    fn vega_row_equals_the_dedicated_api() {
        for rqmc in [false, true] {
            let s = source(rqmc, Some(quotes()), true);
            let plan = native(&s);
            let surface = MarketIvSurface::new(TIMES.to_vec(), XS.to_vec(), quotes()).unwrap();
            let expected = plan
                .market_iv_risk_plan(surface)
                .unwrap()
                .evaluate()
                .unwrap();
            assert_eq!(expected.price, plan.evaluate().unwrap());
            let delta = MeasureRequest::new(
                Measure::SpotDelta,
                MeasureMethod::NativeCalibrated {
                    calibration: BumpCalibration::Recalibrated,
                },
            );
            for requests in [
                vec![vega(BumpCalibration::Recalibrated)],
                vec![delta, vega(BumpCalibration::Recalibrated)],
            ] {
                let result = task(&s, requests).compile().unwrap().evaluate().unwrap();
                let report = &result.report;
                assert_eq!(report.value().to_bits(), expected.price.value.to_bits());
                let row = report.risk(MeasureKind::ParallelMarketIvVega).unwrap();
                assert_eq!(row.values[0].to_bits(), expected.parallel_vega.to_bits());
                assert_eq!(
                    row.standard_errors.as_ref().map(|e| e[0].to_bits()),
                    expected.parallel_standard_error.map(f64::to_bits)
                );
                assert_eq!(row.standard_errors.is_some(), rqmc);
                assert_eq!(row.unit, SensitivityUnit::CurrencyPerUnitVolatility);
                assert_eq!(row.calibration, CalibrationTreatment::Recalibrated);
                assert_eq!(row.method, expected.method);
                assert_eq!(row.name, "parallel_market_iv_vega.native_calibrated");
                let PricingTaskOutput::RoughFamilyLsvMeasureRisk(native) = &result.native_result
                else {
                    panic!("native risk expected")
                };
                assert_eq!(native.market_iv.as_ref(), Some(&expected));
            }
        }
    }

    fn reason(s: &PricingSpecification, request: MeasureRequest) -> Option<CapabilityReason> {
        let c = PricingSource::from(s.clone()).measure_capabilities(std::slice::from_ref(&request));
        assert_eq!(c[0].status, CapabilityStatus::Unsupported, "{request:?}");
        let refused = PricingTask::new(
            s.clone(),
            PricingOperation::Measures(MeasureSet::new(vec![request]).unwrap()),
        )
        .map_or(true, |t| t.compile().and_then(|c| c.evaluate()).is_err());
        assert!(refused);
        c[0].reason
    }

    #[test]
    fn vega_needs_recalibration_a_reproducing_basis_and_the_reverse_trace() {
        let s = source(false, Some(quotes()), true);
        assert_eq!(
            reason(&s, vega(BumpCalibration::Fixed)),
            Some(CapabilityReason::BumpCalibrationMismatch)
        );
        assert_eq!(
            reason(&s, vega(BumpCalibration::NotApplicable)),
            Some(CapabilityReason::BumpCalibrationMismatch)
        );
        assert_eq!(
            reason(&s, MeasureRequest::native(Measure::ParallelMarketIvVega)),
            Some(CapabilityReason::NoNativeEstimator)
        );
        let bump = |size, calibration| {
            MeasureRequest::new(
                Measure::ParallelMarketIvVega,
                MeasureMethod::BumpAndRevalue { size, calibration },
            )
        };
        let absolute = BumpSize::absolute(1e-4).unwrap();
        assert_eq!(
            reason(&s, bump(absolute, BumpCalibration::Fixed)),
            Some(CapabilityReason::BumpCalibrationMismatch)
        );
        assert_eq!(
            reason(
                &s,
                bump(
                    BumpSize::relative(1e-3).unwrap(),
                    BumpCalibration::Recalibrated
                )
            ),
            Some(CapabilityReason::ModelCombination)
        );
        // No basis, or a basis that does not reproduce the target.
        let mut moved = quotes();
        moved[7] += 1e-3;
        for other in [source(false, None, true), source(false, Some(moved), true)] {
            assert_eq!(
                reason(&other, vega(BumpCalibration::Recalibrated)),
                Some(CapabilityReason::ModelCombination)
            );
            assert_eq!(
                reason(&other, bump(absolute, BumpCalibration::Recalibrated)),
                Some(CapabilityReason::ModelCombination)
            );
        }
        assert_eq!(
            reason(
                &source(false, Some(quotes()), false),
                vega(BumpCalibration::Recalibrated)
            ),
            Some(CapabilityReason::ReverseTraceRequired)
        );
    }
}

#[cfg(test)]
mod rough_family_lsv_parameter_tests {
    use crate::lsv::{RoughFamilyLsvParameterRisk, RoughFamilyLsvPricingPlan};
    use crate::rough_volatility::*;
    use crate::*;

    use super::rough_family_lsv_market_iv_tests as f;

    pub(super) fn models() -> Vec<RoughVolatilityModel> {
        let h = RoughHeston::new(0.2, 0.04, 0.7, 0.055, 0.15, -0.6).unwrap();
        vec![
            h.clone().into(),
            LiftedHeston::from_rough(&h, 8, 2.5).unwrap().into(),
            QuadraticRoughHeston::new(0.2, 0.1, 0.8, 0.3, 0.25, 0.05, 0.035)
                .unwrap()
                .into(),
            MixedRoughBergomi::new(
                0.2,
                -0.6,
                vec![0.35, 0.65],
                vec![0.25, 0.6],
                ForwardVarianceCurve::constant(0.04).unwrap(),
            )
            .unwrap()
            .into(),
        ]
    }

    /// A rough-family LSV source on `model`, with the market-IV fixture's
    /// target and engine.
    fn source(model: RoughVolatilityModel, rqmc: bool, trace: bool) -> PricingSpecification {
        let s = f::source(rqmc, None, trace);
        let EquityModelSpec::RoughLocalStochasticVolatility {
            target, particles, ..
        } = &s.model().equity
        else {
            unreachable!()
        };
        let equity = EquityModelSpec::RoughLocalStochasticVolatility {
            target: target.clone(),
            model,
            particles: particles.clone(),
        };
        let rates = s.model().rates.clone();
        let step = s.numerics().maximum_step;
        s.with_model(PricingModelSpec::new(equity, rates), step)
            .unwrap()
    }

    fn native(s: &PricingSpecification) -> RoughFamilyLsvPricingPlan {
        let EquityModelSpec::RoughLocalStochasticVolatility {
            model, particles, ..
        } = &s.model().equity
        else {
            unreachable!()
        };
        RoughFamilyLsvPricingPlan::compile(
            s.carrier_request(),
            model.clone(),
            particles.clone(),
            s.numerics().execution_policy,
        )
        .unwrap()
    }

    fn request(parameters: Vec<ModelParameterId>, calibration: BumpCalibration) -> MeasureRequest {
        MeasureRequest::new(
            Measure::ModelParameters(ModelParameterSelection::new(parameters).unwrap()),
            MeasureMethod::NativeCalibrated { calibration },
        )
    }

    fn task(s: &PricingSpecification, requests: Vec<MeasureRequest>) -> PricingTask {
        PricingTask::new(
            s.clone(),
            PricingOperation::Measures(MeasureSet::new(requests).unwrap()),
        )
        .unwrap()
    }

    /// The dedicated API each model's parameter risk comes from.
    fn dedicated(plan: &RoughFamilyLsvPricingPlan, hurst: bool) -> RoughFamilyLsvParameterRisk {
        match plan.rough_model() {
            RoughVolatilityModel::RoughHeston(_) | RoughVolatilityModel::LiftedHeston(_) => {
                RoughFamilyLsvParameterRisk::Heston(
                    plan.evaluate_heston_parameter_risk(hurst).unwrap(),
                )
            }
            RoughVolatilityModel::QuadraticRoughHeston(_) => {
                RoughFamilyLsvParameterRisk::QuadraticHeston(
                    plan.evaluate_quadratic_heston_parameter_risk(hurst)
                        .unwrap(),
                )
            }
            RoughVolatilityModel::MixedRoughBergomi(_) => {
                RoughFamilyLsvParameterRisk::MixedBergomi(if hurst {
                    plan.evaluate_mixed_bergomi_parameter_risk_with_hurst()
                        .unwrap()
                } else {
                    plan.evaluate_mixed_bergomi_parameter_risk().unwrap()
                })
            }
            _ => unreachable!(),
        }
    }

    /// Every row is the dedicated API's total derivative and error bit for
    /// bit, in the requested (here reversed) order, with and without Hurst,
    /// under both samplers; the price is the plan's price.
    #[test]
    fn parameter_rows_equal_the_dedicated_apis() {
        for model in models() {
            for rqmc in [false, true] {
                let s = source(model.clone(), rqmc, true);
                let plan = native(&s);
                let price = plan.evaluate().unwrap();
                let (available, hurst_available) = plan.native_parameters().unwrap();
                for hurst in [false, true] {
                    if hurst && !hurst_available {
                        continue;
                    }
                    let mut ids = available.clone();
                    if hurst {
                        ids.push(ModelParameterId::Hurst);
                    }
                    let expected = dedicated(&plan, hurst);
                    let mut requested = ids.clone();
                    requested.reverse();
                    let result = task(
                        &s,
                        vec![request(requested.clone(), BumpCalibration::Recalibrated)],
                    )
                    .compile()
                    .unwrap()
                    .evaluate()
                    .unwrap();
                    let report = &result.report;
                    assert_eq!(report.value().to_bits(), price.value.to_bits());
                    let rows = report
                        .risks()
                        .iter()
                        .filter(|r| r.measure == Some(MeasureKind::ModelParameters))
                        .collect::<Vec<_>>();
                    assert_eq!(
                        rows.iter()
                            .map(|r| r.model_parameter.unwrap())
                            .collect::<Vec<_>>(),
                        requested,
                        "{}",
                        model.name()
                    );
                    for (i, id) in ids.iter().enumerate() {
                        let row = report.model_parameter_risk(*id).unwrap();
                        assert_eq!(
                            row.values[0].to_bits(),
                            expected.parameter_adjoints()[i].to_bits(),
                            "{} {id:?}",
                            model.name()
                        );
                        assert_eq!(
                            row.standard_errors.as_ref().map(|e| e[0].to_bits()),
                            expected.standard_errors().map(|e| e[i].to_bits())
                        );
                        assert_eq!(row.standard_errors.is_some(), rqmc);
                        assert_eq!(row.unit, SensitivityUnit::CurrencyPerUnitParameter);
                        assert_eq!(row.calibration, CalibrationTreatment::Recalibrated);
                        assert_eq!(row.method, expected.method());
                        assert_eq!(
                            row.name,
                            format!("model_parameters.{}.native_calibrated", id.label())
                        );
                    }
                    let PricingTaskOutput::RoughFamilyLsvMeasureRisk(native) =
                        &result.native_result
                    else {
                        panic!("native risk expected")
                    };
                    let identified = native.model_parameters.as_ref().unwrap();
                    assert_eq!(identified.risk, expected);
                    assert_eq!(&*identified.parameters, &ids[..]);
                    assert_eq!(expected.price(), &price);
                }
            }
        }
    }

    fn reason(s: &PricingSpecification, request: MeasureRequest) -> Option<CapabilityReason> {
        let c = PricingSource::from(s.clone()).measure_capabilities(std::slice::from_ref(&request));
        assert_eq!(c[0].status, CapabilityStatus::Unsupported, "{request:?}");
        let refused = PricingTask::new(
            s.clone(),
            PricingOperation::Measures(MeasureSet::new(vec![request]).unwrap()),
        )
        .map_or(true, |t| t.compile().and_then(|c| c.evaluate()).is_err());
        assert!(refused);
        c[0].reason
    }

    #[test]
    fn parameters_must_belong_to_the_model_and_be_recalibrated() {
        use ModelParameterId as P;
        let models = models();
        let heston = source(models[0].clone(), false, true);
        for calibration in [BumpCalibration::Fixed, BumpCalibration::NotApplicable] {
            assert_eq!(
                reason(&heston, request(vec![P::VolOfVol], calibration)),
                Some(CapabilityReason::BumpCalibrationMismatch)
            );
        }
        assert_eq!(
            reason(
                &heston,
                MeasureRequest::native(Measure::ModelParameters(
                    ModelParameterSelection::new(vec![P::VolOfVol]).unwrap()
                ))
            ),
            Some(CapabilityReason::NoNativeEstimator)
        );
        assert_eq!(
            reason(
                &heston,
                request(vec![P::VarianceFloor], BumpCalibration::Recalibrated)
            ),
            Some(CapabilityReason::ModelCombination)
        );
        // A finite lift has no Hurst.
        let lifted = source(models[1].clone(), false, true);
        assert_eq!(
            reason(
                &lifted,
                request(vec![P::Hurst], BumpCalibration::Recalibrated)
            ),
            Some(CapabilityReason::ModelCombination)
        );
        let mixed = source(models[3].clone(), false, true);
        assert_eq!(
            reason(
                &mixed,
                request(
                    vec![P::ComponentVolOfVol { component: 2 }],
                    BumpCalibration::Recalibrated
                )
            ),
            Some(CapabilityReason::ModelCombination)
        );
        assert_eq!(
            reason(
                &source(models[0].clone(), false, false),
                request(vec![P::VolOfVol], BumpCalibration::Recalibrated)
            ),
            Some(CapabilityReason::ReverseTraceRequired)
        );
        let sabr = RoughSabr::new(
            0.2,
            0.4,
            -0.6,
            1.0,
            ForwardVarianceCurve::constant(0.04).unwrap(),
        )
        .unwrap();
        assert_eq!(
            reason(
                &source(sabr.into(), false, true),
                request(vec![P::VolOfVol], BumpCalibration::Recalibrated)
            ),
            Some(CapabilityReason::NoNativeEstimator)
        );
        assert!(ModelParameterSelection::new(vec![]).is_err());
        assert!(ModelParameterSelection::new(vec![P::Hurst, P::Hurst]).is_err());
        // Other families have no parameter measure.
        let bergomi: PricingSource = f::source(false, None, true).into();
        assert_eq!(
            bergomi
                .family()
                .measure_methods(MeasureKind::ModelParameters),
            &[MeasureMethodKind::NativeCalibrated]
        );
    }

    #[test]
    fn task_and_report_round_trip() {
        use ModelParameterId as P;
        let s = source(models()[3].clone(), true, true);
        let t = task(
            &s,
            vec![request(
                vec![
                    P::Correlation,
                    P::ComponentVolOfVol { component: 1 },
                    P::Hurst,
                ],
                BumpCalibration::Recalibrated,
            )],
        );
        let text = pricing_task_to_json(&t).unwrap();
        let compact = text.split_whitespace().collect::<String>();
        assert!(
            compact.contains(r#"{"name":"component_vol_of_vol","component":1}"#),
            "{text}"
        );
        let back = parse_pricing_task_json(text.as_bytes(), JsonLimits::DEFAULT).unwrap();
        assert_eq!(back.fingerprint().unwrap(), t.fingerprint().unwrap());
        let report = t.compile().unwrap().evaluate().unwrap().report;
        let again = parse_valuation_report_json(
            valuation_report_to_json(&report).unwrap().as_bytes(),
            JsonLimits::DEFAULT,
        )
        .unwrap();
        assert_eq!(again, report);
        assert!(
            again
                .model_parameter_risk(P::ComponentVolOfVol { component: 1 })
                .is_some()
        );
    }
}

#[cfg(test)]
mod rough_volatility_tests {
    use serde_json::{Value, json};

    use crate::risk::SpotBump;
    use crate::rough_volatility::*;
    use crate::*;

    /// The four models with a parameter estimator, then rough SABR and RFSV.
    fn models() -> Vec<RoughVolatilityModel> {
        let mut models = super::rough_family_lsv_parameter_tests::models();
        models.push(
            RoughSabr::new(
                0.2,
                0.4,
                -0.6,
                1.0,
                ForwardVarianceCurve::constant(0.04).unwrap(),
            )
            .unwrap()
            .into(),
        );
        models.push(
            Rfsv::new(0.15, 0.5, 0.6, (0.2f64).ln(), None)
                .unwrap()
                .into(),
        );
        models
    }

    /// A direct rough-volatility source on `model`.
    fn source(
        model: RoughVolatilityModel,
        rqmc: bool,
        payoff: Option<Value>,
    ) -> PricingSpecification {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/specification-v1/rough_heston.golden.json");
        let mut v: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        if rqmc {
            v["engine"] = json!({"type":"randomized_quasi_monte_carlo",
                "points_per_scramble":32,"scramble_count":8,"master_scramble_seed":819,
                "variance_reduction":{"antithetic":true,"brownian_bridge":true}});
        } else {
            v["engine"]["independent_sampling_units"] = json!(256);
        }
        if let Some(product) = payoff {
            v["product"] = product;
        }
        let s = parse_specification_json(&serde_json::to_vec(&v).unwrap(), JsonLimits::DEFAULT)
            .unwrap();
        let rates = s.model().rates.clone();
        let step = s.numerics().maximum_step;
        s.with_model(
            PricingModelSpec::new(EquityModelSpec::RoughVolatility(model), rates),
            step,
        )
        .unwrap()
    }

    fn native(s: &PricingSpecification) -> RoughVolatilityPricingPlan {
        let EquityModelSpec::RoughVolatility(model) = &s.model().equity else {
            unreachable!()
        };
        RoughVolatilityPricingPlan::compile(
            s.carrier_request(),
            model.clone(),
            s.numerics().maximum_step.unwrap(),
            s.numerics().execution_policy,
        )
        .unwrap()
    }

    fn gamma() -> MeasureRequest {
        MeasureRequest::new(
            Measure::SpotGamma,
            MeasureMethod::NativeSpotBump {
                bump: SpotBump::relative(0.01).unwrap(),
            },
        )
    }

    fn parameters(ids: Vec<ModelParameterId>) -> MeasureRequest {
        MeasureRequest::native(Measure::ModelParameters(
            ModelParameterSelection::new(ids).unwrap(),
        ))
    }

    fn task(s: &PricingSpecification, requests: Vec<MeasureRequest>) -> PricingTask {
        PricingTask::new(
            s.clone(),
            PricingOperation::Measures(MeasureSet::new(requests).unwrap()),
        )
        .unwrap()
    }

    fn evaluate(s: &PricingSpecification, requests: Vec<MeasureRequest>) -> PricingTaskResult {
        task(s, requests).compile().unwrap().evaluate().unwrap()
    }

    fn native_result(result: &PricingTaskResult) -> &RoughVolatilityMeasureRisk {
        let PricingTaskOutput::RoughVolatilityMeasureRisk(native) = &result.native_result else {
            panic!("native risk expected")
        };
        native
    }

    /// `(estimates, errors, method)` of the dedicated parameter APIs, in
    /// the native order, with Hurst last when asked.
    fn dedicated(
        plan: &RoughVolatilityPricingPlan,
        hurst: bool,
    ) -> (Vec<f64>, Vec<f64>, Vec<&'static str>) {
        match plan.rough_model() {
            RoughVolatilityModel::RoughHeston(_) | RoughVolatilityModel::LiftedHeston(_) => {
                let r = plan.evaluate_heston_parameter_risk().unwrap();
                let (mut v, mut e) = (r.parameter_adjoints.to_vec(), r.standard_errors.to_vec());
                let mut m = vec![r.method; 5];
                if hurst {
                    let h = plan.evaluate_hurst_risk().unwrap();
                    v.push(h.hurst_sensitivity);
                    e.push(h.standard_error);
                    m.push(h.method);
                }
                (v, e, m)
            }
            RoughVolatilityModel::QuadraticRoughHeston(_) => {
                let r = plan
                    .evaluate_quadratic_heston_parameter_risk(hurst)
                    .unwrap();
                let n = r.parameter_adjoints.len();
                (
                    r.parameter_adjoints.to_vec(),
                    r.standard_errors.to_vec(),
                    vec![r.method; n],
                )
            }
            RoughVolatilityModel::MixedRoughBergomi(_) => {
                let r = if hurst {
                    plan.evaluate_mixed_bergomi_parameter_risk_with_hurst()
                        .unwrap()
                } else {
                    plan.evaluate_mixed_bergomi_parameter_risk().unwrap()
                };
                let n = r.parameter_adjoints.len();
                (
                    r.parameter_adjoints.to_vec(),
                    r.standard_errors.to_vec(),
                    vec![r.method; n],
                )
            }
            _ => unreachable!(),
        }
    }

    /// Delta, Gamma and every parameter row are the dedicated APIs' values
    /// and errors bit for bit, for six models under both samplers, with the
    /// parameters requested in reverse order; the price is the plan's.
    #[test]
    fn rows_equal_the_dedicated_apis() {
        for model in models() {
            for rqmc in [false, true] {
                let s = source(model.clone(), rqmc, None);
                let plan = native(&s);
                let price = plan.evaluate().unwrap();
                let delta = plan.evaluate_delta().unwrap();
                let g = plan.evaluate_gamma_bump(0.01 * plan.spot()).unwrap();
                let name = model.name();
                let mut requests = vec![MeasureRequest::native(Measure::SpotDelta), gamma()];
                let mut ids = Vec::new();
                let available = model.native_parameters();
                if let Some((list, hurst)) = &available {
                    ids.clone_from(list);
                    if *hurst {
                        ids.push(ModelParameterId::Hurst);
                    }
                    let mut reversed = ids.clone();
                    reversed.reverse();
                    requests.push(parameters(reversed));
                }
                let result = evaluate(&s, requests);
                let report = &result.report;
                assert_eq!(report.value().to_bits(), price.value.to_bits(), "{name}");
                assert_eq!(delta.price, price, "{name}");
                let d = report.risk(MeasureKind::SpotDelta).unwrap();
                assert_eq!(d.values[0].to_bits(), delta.delta.to_bits(), "{name}");
                assert_eq!(
                    d.standard_errors.as_ref().unwrap()[0].to_bits(),
                    delta.delta_standard_error.to_bits()
                );
                assert_eq!(d.name, "spot_delta.native");
                assert_eq!(d.calibration, CalibrationTreatment::NotApplicable);
                let r = report.risk(MeasureKind::SpotGamma).unwrap();
                assert_eq!(r.values[0].to_bits(), g.gamma.to_bits(), "{name}");
                assert_eq!(
                    r.standard_errors.as_ref().unwrap()[0].to_bits(),
                    g.gamma_standard_error.to_bits()
                );
                assert_eq!(r.name, "spot_gamma.native_spot_bump");
                let native = native_result(&result);
                assert_eq!(native.delta.as_ref(), Some(&delta));
                assert_eq!(native.gamma.as_ref(), Some(&g));
                let Some((_, hurst)) = available else {
                    assert!(native.model_parameters.is_none());
                    continue;
                };
                let (values, errors, methods) = dedicated(&plan, hurst);
                let rows = report
                    .risks()
                    .iter()
                    .filter(|r| r.measure == Some(MeasureKind::ModelParameters))
                    .map(|r| r.model_parameter.unwrap())
                    .collect::<Vec<_>>();
                assert_eq!(
                    rows.iter().rev().copied().collect::<Vec<_>>(),
                    ids,
                    "{name}"
                );
                for (i, id) in ids.iter().enumerate() {
                    let row = report.model_parameter_risk(*id).unwrap();
                    assert_eq!(
                        row.values[0].to_bits(),
                        values[i].to_bits(),
                        "{name} {id:?}"
                    );
                    assert_eq!(
                        row.standard_errors.as_ref().unwrap()[0].to_bits(),
                        errors[i].to_bits(),
                        "{name} {id:?}"
                    );
                    assert_eq!(row.method, methods[i]);
                    assert_eq!(row.unit, SensitivityUnit::CurrencyPerUnitParameter);
                    assert_eq!(row.calibration, CalibrationTreatment::NotApplicable);
                    assert_eq!(row.name, format!("model_parameters.{}.native", id.label()));
                }
                assert_eq!(
                    &*native.model_parameters.as_ref().unwrap().parameters,
                    &ids[..]
                );
            }
        }
    }

    /// Rough Heston runs only the estimators the list needs: Hurst alone
    /// is `evaluate_hurst_risk`, and the price is then its price.
    #[test]
    fn rough_heston_runs_only_the_listed_estimators() {
        let s = source(models()[0].clone(), false, None);
        let plan = native(&s);
        let hurst = plan.evaluate_hurst_risk().unwrap();
        let result = evaluate(&s, vec![parameters(vec![ModelParameterId::Hurst])]);
        let p = native_result(&result).model_parameters.as_ref().unwrap();
        assert_eq!(
            p.risk,
            RoughVolatilityParameterRisk::Heston {
                parameters: None,
                hurst: Some(Box::new(hurst.clone()))
            }
        );
        assert_eq!(&*p.parameters, &[ModelParameterId::Hurst]);
        assert_eq!(native_result(&result).price, hurst.price);
        assert_eq!(
            result
                .report
                .model_parameter_risk(ModelParameterId::Hurst)
                .unwrap()
                .values[0]
                .to_bits(),
            hurst.hurst_sensitivity.to_bits()
        );
        let result = evaluate(&s, vec![parameters(vec![ModelParameterId::Correlation])]);
        let RoughVolatilityParameterRisk::Heston { parameters, hurst } = &native_result(&result)
            .model_parameters
            .as_ref()
            .unwrap()
            .risk
        else {
            panic!("Heston risk expected")
        };
        assert_eq!(
            parameters.as_deref(),
            Some(&plan.evaluate_heston_parameter_risk().unwrap())
        );
        assert!(hurst.is_none());
        // Price only stays the ordinary result.
        let result = evaluate(&s, vec![MeasureRequest::native(Measure::Price)]);
        assert!(matches!(
            result.native_result,
            PricingTaskOutput::SingleAsset(_)
        ));
    }

    fn reason(s: &PricingSpecification, request: MeasureRequest) -> Option<CapabilityReason> {
        let c = PricingSource::from(s.clone()).measure_capabilities(std::slice::from_ref(&request));
        assert_eq!(c[0].status, CapabilityStatus::Unsupported, "{request:?}");
        let refused = PricingTask::new(
            s.clone(),
            PricingOperation::Measures(MeasureSet::new(vec![request]).unwrap()),
        )
        .map_or(true, |t| t.compile().and_then(|c| c.evaluate()).is_err());
        assert!(refused);
        c[0].reason
    }

    #[test]
    fn refusals_name_their_reason() {
        use ModelParameterId as P;
        let models = models();
        let heston = source(models[0].clone(), false, None);
        assert_eq!(
            reason(&heston, parameters(vec![P::VarianceFloor])),
            Some(CapabilityReason::ModelCombination)
        );
        assert_eq!(
            reason(
                &source(models[1].clone(), false, None),
                parameters(vec![P::Hurst])
            ),
            Some(CapabilityReason::ModelCombination)
        );
        for model in &models[4..] {
            assert_eq!(
                reason(
                    &source(model.clone(), false, None),
                    parameters(vec![P::VolOfVol])
                ),
                Some(CapabilityReason::NoNativeEstimator),
                "{}",
                model.name()
            );
        }
        // Nothing is calibrated, so no calibrated method applies.
        assert_eq!(
            reason(
                &heston,
                MeasureRequest::new(
                    Measure::SpotDelta,
                    MeasureMethod::NativeCalibrated {
                        calibration: BumpCalibration::Recalibrated
                    }
                )
            ),
            Some(CapabilityReason::NoNativeEstimator)
        );
        assert_eq!(
            reason(
                &heston,
                MeasureRequest::new(
                    Measure::ModelParameters(
                        ModelParameterSelection::new(vec![P::VolOfVol]).unwrap()
                    ),
                    MeasureMethod::NativeCalibrated {
                        calibration: BumpCalibration::Recalibrated
                    }
                )
            ),
            Some(CapabilityReason::NoNativeEstimator)
        );
        let digital = source(
            models[0].clone(),
            false,
            Some(json!({"type":"digital","underlying_id":1,"currency_id":2,
                "expiry":"2027-09-04","strike":100.0,"payout":1.0,
                "side":{"type":"call"},"payout_kind":{"type":"cash"},
                "payment_date":"2027-09-04"})),
        );
        for request in [
            MeasureRequest::native(Measure::SpotDelta),
            gamma(),
            parameters(vec![P::VolOfVol]),
        ] {
            assert_eq!(
                reason(&digital, request),
                Some(CapabilityReason::DiscontinuousPayoff)
            );
        }
        // Discovery lists the native methods and the bump fallback.
        let available = PricingSource::from(heston).available_measures();
        let methods = |kind| {
            available
                .iter()
                .find(|m| m.measure == kind)
                .map(|m| m.methods.clone())
        };
        assert_eq!(
            methods(MeasureKind::SpotDelta),
            Some(vec![
                MeasureMethodKind::Native,
                MeasureMethodKind::BumpAndRevalue
            ])
        );
        assert_eq!(
            methods(MeasureKind::ModelParameters),
            Some(vec![MeasureMethodKind::Native])
        );
        assert_eq!(methods(MeasureKind::ParallelMarketIvVega), None);
    }

    /// The native Delta agrees with the bump fallback on the same draws far
    /// inside the sampling error (3e-13 against 9e-3 here).
    #[test]
    fn native_delta_agrees_with_bump_and_revalue() {
        let s = source(models()[0].clone(), true, None);
        let bump = MeasureRequest::new(
            Measure::SpotDelta,
            MeasureMethod::BumpAndRevalue {
                size: crate::BumpSize::relative(1e-4).unwrap(),
                calibration: BumpCalibration::NotApplicable,
            },
        );
        let native = evaluate(&s, vec![MeasureRequest::native(Measure::SpotDelta)]);
        let bumped = evaluate(&s, vec![bump]);
        let (n, b) = (
            native.report.risk(MeasureKind::SpotDelta).unwrap(),
            bumped.report.risk(MeasureKind::SpotDelta).unwrap(),
        );
        let se = n.standard_errors.as_ref().unwrap()[0];
        assert!(
            (n.values[0] - b.values[0]).abs() < 1e-8 && se > 1e-3,
            "{} vs {} (se {se})",
            n.values[0],
            b.values[0]
        );
    }

    #[test]
    fn task_and_report_round_trip() {
        use ModelParameterId as P;
        let s = source(models()[2].clone(), true, None);
        let t = task(
            &s,
            vec![
                MeasureRequest::native(Measure::SpotDelta),
                parameters(vec![P::Quadratic, P::Hurst]),
            ],
        );
        let text = pricing_task_to_json(&t).unwrap();
        let back = parse_pricing_task_json(text.as_bytes(), JsonLimits::DEFAULT).unwrap();
        assert_eq!(back.fingerprint().unwrap(), t.fingerprint().unwrap());
        let report = t.compile().unwrap().evaluate().unwrap().report;
        let again = parse_valuation_report_json(
            valuation_report_to_json(&report).unwrap().as_bytes(),
            JsonLimits::DEFAULT,
        )
        .unwrap();
        assert_eq!(again, report);
        assert!(again.model_parameter_risk(P::Hurst).is_some());
    }
}

#[cfg(test)]
mod stochastic_dividend_tests {
    use serde_json::{Value, json};

    use crate::risk::{GammaConfig, SpotBump};
    use crate::*;

    /// The saved task's source, optionally under RQMC or with a digital.
    fn source(name: &str, rqmc: bool, digital: bool) -> PricingSource {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/task-v1")
            .join(format!("stochastic_dividend_{name}.golden.json"));
        let mut v: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        let equity = &mut v["source"]["equity"];
        if rqmc {
            equity["engine"] = json!({"type":"randomized_quasi_monte_carlo",
                "points_per_scramble":32,"scramble_count":8,"master_scramble_seed":819,
                "variance_reduction":{"antithetic":true,"brownian_bridge":false}});
        }
        if digital {
            equity["product"] = json!({"type":"digital","underlying_id":1,"currency_id":2,
                "expiry":"2027-09-04","strike":100.0,"payout":1.0,
                "side":{"type":"call"},"payout_kind":{"type":"cash"},
                "payment_date":"2027-09-04"});
        }
        v["operation"] = json!({"type":"evaluate"});
        parse_pricing_task_json(&serde_json::to_vec(&v).unwrap(), JsonLimits::DEFAULT)
            .unwrap()
            .source()
            .clone()
    }

    /// A Bergomi-family residual equity on the BS fixture's market.
    fn sv_source(factor: Value, correlations: Vec<f64>, rqmc: bool) -> PricingSource {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/task-v1/stochastic_dividend_bs.golden.json");
        let mut v: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        v["source"]["equity"]["model"]["equity"] = json!({"type":"stochastic_volatility",
            "initial_volatility":0.2,"factor":factor});
        v["source"]["equity"]["numerics"]["maximum_step"] = json!(0.5);
        v["source"]["volatility_correlations"] = json!(correlations);
        if rqmc {
            v["source"]["equity"]["engine"] = json!({"type":"randomized_quasi_monte_carlo",
                "points_per_scramble":32,"scramble_count":8,"master_scramble_seed":819,
                "variance_reduction":{"antithetic":true,"brownian_bridge":false}});
        }
        v["operation"] = json!({"type":"evaluate"});
        parse_pricing_task_json(&serde_json::to_vec(&v).unwrap(), JsonLimits::DEFAULT)
            .unwrap()
            .source()
            .clone()
    }

    /// Sources without a calibration step, with their correlations in the
    /// native order.
    fn correlated_sources(rqmc: bool) -> Vec<(&'static str, PricingSource, Vec<CorrelationId>)> {
        use CorrelationId as C;
        let (s0, d0) = (
            C::SpotVolatility { factor: 0 },
            C::DividendVolatility { factor: 0 },
        );
        let (s1, d1) = (
            C::SpotVolatility { factor: 1 },
            C::DividendVolatility { factor: 1 },
        );
        vec![
            ("bs", source("bs", rqmc, false), vec![C::EquityDividend]),
            (
                "hw",
                source("hw", rqmc, false),
                vec![C::EquityDividend, C::EquityRate, C::DividendRate],
            ),
            (
                "bergomi",
                sv_source(
                    json!({"type":"bergomi","mean_reversion":1.2,"vol_of_vol":0.3,
                        "correlation":-0.4}),
                    vec![0.1],
                    rqmc,
                ),
                vec![C::EquityDividend, s0, d0],
            ),
            (
                "bergomi_two_factor",
                sv_source(
                    json!({"type":"bergomi_two_factor","mean_reversions":[1.0,4.0],
                        "vol_of_vol":0.5,"mixing_weight":0.3,"spot_correlations":[-0.5,-0.3],
                        "factor_correlation":0.4}),
                    vec![0.1, 0.05],
                    rqmc,
                ),
                vec![C::EquityDividend, s0, s1, d0, d1, C::VolatilityFactors],
            ),
            (
                "rough_bergomi",
                sv_source(
                    json!({"type":"rough_bergomi","hurst":0.2,"vol_of_vol":0.5,
                        "correlation":-0.5}),
                    vec![0.1],
                    rqmc,
                ),
                vec![C::EquityDividend, s0, d0],
            ),
        ]
    }

    fn correlations(ids: Vec<CorrelationId>) -> MeasureRequest {
        MeasureRequest::native(Measure::Correlations(
            CorrelationSelection::new(ids).unwrap(),
        ))
    }

    /// Every correlation row is `evaluate_correlation_aad`'s partial and
    /// error bit for bit, in the requested (reversed) order, on BS,
    /// Hull-White and the three Bergomi drivers under both samplers.
    #[test]
    fn correlation_rows_equal_the_dedicated_api() {
        for rqmc in [false, true] {
            for (name, s, ids) in correlated_sources(rqmc) {
                let mut reversed = ids.clone();
                reversed.reverse();
                let compiled = task(&s, vec![correlations(reversed.clone())])
                    .compile()
                    .unwrap();
                let result = compiled.evaluate().unwrap();
                let (expected, price) = match compiled.dividend_plans() {
                    (Some(p), None) => {
                        (p.evaluate_correlation_aad().unwrap(), p.evaluate().unwrap())
                    }
                    (None, Some(p)) => {
                        (p.evaluate_correlation_aad().unwrap(), p.evaluate().unwrap())
                    }
                    _ => unreachable!(),
                };
                let report = &result.report;
                assert_eq!(report.value().to_bits(), price.value.to_bits(), "{name}");
                let rows = report
                    .risks()
                    .iter()
                    .filter(|r| r.measure == Some(MeasureKind::Correlations))
                    .map(|r| r.correlation.unwrap())
                    .collect::<Vec<_>>();
                assert_eq!(rows, reversed, "{name}");
                for id in &ids {
                    let label = match id {
                        CorrelationId::EquityDividend => "equity_dividend_correlation".to_owned(),
                        CorrelationId::EquityRate => "equity_rate_correlation".to_owned(),
                        CorrelationId::DividendRate => "dividend_rate_correlation".to_owned(),
                        CorrelationId::SpotVolatility { factor } => {
                            format!("spot_volatility_correlation[{factor}]")
                        }
                        CorrelationId::DividendVolatility { factor } => {
                            format!("dividend_volatility_correlation[{factor}]")
                        }
                        CorrelationId::VolatilityFactors => {
                            "volatility_factor_correlation".to_owned()
                        }
                        CorrelationId::AssetPair { .. } => unreachable!("not a dividend driver"),
                    };
                    let i = expected
                        .parameter_labels
                        .iter()
                        .position(|l| *l == label)
                        .unwrap_or_else(|| panic!("{name} {label}"));
                    let row = report.correlation_risk(*id).unwrap();
                    assert_eq!(
                        row.values[0].to_bits(),
                        expected.derivatives[i].to_bits(),
                        "{name} {id:?}"
                    );
                    assert_eq!(
                        row.standard_errors.as_ref().unwrap()[0].to_bits(),
                        expected.standard_errors[i].to_bits()
                    );
                    assert_eq!(row.method, expected.method);
                    assert_eq!(row.calibration, CalibrationTreatment::NotApplicable);
                    assert_eq!(row.name, format!("correlations.{}.native", id.label()));
                }
                let PricingTaskOutput::StochasticDividendMeasureRisk(native) =
                    &result.native_result
                else {
                    panic!("native risk expected")
                };
                let identified = native.correlations.as_ref().unwrap();
                assert_eq!(identified.risk, expected);
                assert_eq!(&*identified.correlations, &ids[..]);
                assert_eq!(native.price, price, "{name}");
            }
        }
    }

    #[test]
    fn correlations_must_belong_to_the_source() {
        use CorrelationId as C;
        let bs = source("bs", false, false);
        for id in [C::EquityRate, C::SpotVolatility { factor: 0 }] {
            assert_eq!(
                reason(&bs, correlations(vec![id])),
                Some(CapabilityReason::ModelCombination)
            );
        }
        let one = &correlated_sources(false)[2].1;
        assert_eq!(
            reason(one, correlations(vec![C::SpotVolatility { factor: 1 }])),
            Some(CapabilityReason::ModelCombination)
        );
        assert_eq!(
            reason(one, correlations(vec![C::VolatilityFactors])),
            Some(CapabilityReason::ModelCombination)
        );
        // Residual LSV correlation risk is a finite difference with stated
        // bumps; it has no measure adapter yet.
        assert_eq!(
            reason(
                &source("lsv", false, false),
                correlations(vec![C::EquityDividend])
            ),
            Some(CapabilityReason::DedicatedRiskApi)
        );
        assert_eq!(
            reason(
                &source("bs", false, true),
                correlations(vec![C::EquityDividend])
            ),
            Some(CapabilityReason::DiscontinuousPayoff)
        );
        assert!(CorrelationSelection::new(vec![]).is_err());
        assert!(CorrelationSelection::new(vec![C::EquityRate, C::EquityRate]).is_err());
        for id in [
            C::EquityDividend,
            C::SpotVolatility { factor: 3 },
            C::DividendVolatility { factor: 0 },
            C::VolatilityFactors,
        ] {
            assert_eq!(C::from_label(&id.label()), Some(id));
        }
        assert_eq!(C::from_label("spot_volatility"), None);
    }

    #[test]
    fn correlation_task_and_report_round_trip() {
        use CorrelationId as C;
        let s = &correlated_sources(true)[3].1;
        let t = task(
            s,
            vec![
                MeasureRequest::native(Measure::SpotDelta),
                correlations(vec![C::VolatilityFactors, C::SpotVolatility { factor: 1 }]),
            ],
        );
        let text = pricing_task_to_json(&t).unwrap();
        let compact = text.split_whitespace().collect::<String>();
        assert!(
            compact.contains(r#"{"name":"spot_volatility","factor":1}"#),
            "{text}"
        );
        let back = parse_pricing_task_json(text.as_bytes(), JsonLimits::DEFAULT).unwrap();
        assert_eq!(back.fingerprint().unwrap(), t.fingerprint().unwrap());
        let report = t.compile().unwrap().evaluate().unwrap().report;
        let again = parse_valuation_report_json(
            valuation_report_to_json(&report).unwrap().as_bytes(),
            JsonLimits::DEFAULT,
        )
        .unwrap();
        assert_eq!(again, report);
        assert!(again.correlation_risk(C::VolatilityFactors).is_some());
    }

    fn curve(name: &str) -> MeasureRequest {
        if calibrated(name) {
            MeasureRequest::new(
                Measure::DiscountCurve,
                MeasureMethod::NativeCalibrated {
                    calibration: BumpCalibration::Recalibrated,
                },
            )
        } else {
            MeasureRequest::native(Measure::DiscountCurve)
        }
    }

    /// The log-DF row is the dedicated API's discount log-DF adjoints and
    /// errors bit for bit, and the DV01 row their `-1e-4 t` projection, on a
    /// discount-time axis: `evaluate_aad` on BS, Hull-White and Bergomi,
    /// `evaluate_lsv_market_risk` (recalibrated) on residual LSV.
    #[test]
    fn curve_rows_equal_the_dedicated_apis() {
        type Expected = (
            Vec<f64>,
            Vec<f64>,
            Vec<f64>,
            &'static str,
            CalibrationTreatment,
        );
        let aad = |r: crate::stochastic_dividends::StochasticDividendAadRisk| -> Expected {
            let pick = |x: &[f64]| {
                r.parameter_labels
                    .iter()
                    .zip(x)
                    .filter(|(l, _)| l.starts_with("discount_log_df["))
                    .map(|(_, v)| *v)
                    .collect::<Vec<_>>()
            };
            (
                r.discount_times.to_vec(),
                pick(&r.derivatives),
                pick(&r.standard_errors),
                r.method,
                CalibrationTreatment::NotApplicable,
            )
        };
        let bits = |x: &[f64]| x.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        for rqmc in [false, true] {
            let sources = vec![
                ("bs", source("bs", rqmc, false)),
                ("hw", source("hw", rqmc, false)),
                ("lsv", source("lsv", rqmc, false)),
                ("bergomi", correlated_sources(rqmc).swap_remove(2).1),
            ];
            for (name, s) in sources {
                let compiled = task(&s, vec![curve(name)]).compile().unwrap();
                let result = compiled.evaluate().unwrap();
                let (times, values, errors, method, treatment) = match compiled.dividend_plans() {
                    (Some(p), None) if calibrated(name) => {
                        let r = p.evaluate_lsv_market_risk().unwrap();
                        assert_eq!(
                            result.report.value().to_bits(),
                            r.price.value.to_bits(),
                            "{name}"
                        );
                        (
                            r.discount_times.to_vec(),
                            r.discount_log_df_adjoints.to_vec(),
                            r.discount_log_df_standard_errors.to_vec(),
                            r.method,
                            CalibrationTreatment::Recalibrated,
                        )
                    }
                    (Some(p), None) => aad(p.evaluate_aad().unwrap()),
                    (None, Some(p)) => aad(p.evaluate_aad().unwrap()),
                    _ => unreachable!(),
                };
                assert!(!values.is_empty(), "{name}");
                assert_eq!(values.len(), times.len(), "{name}");
                let rows = result
                    .report
                    .risks()
                    .iter()
                    .filter(|r| r.measure == Some(MeasureKind::DiscountCurve))
                    .collect::<Vec<_>>();
                assert_eq!(rows.len(), 2, "{name}");
                let method_name = if calibrated(name) {
                    "native_calibrated"
                } else {
                    "native"
                };
                assert_eq!(rows[0].name, format!("discount_curve.log_df.{method_name}"));
                assert_eq!(rows[1].name, format!("discount_curve.dv01.{method_name}"));
                for row in &rows {
                    assert_eq!(row.axes.len(), 1);
                    assert_eq!(row.axes[0].name, "discount_time");
                    assert_eq!(row.axes[0].values, times, "{name}");
                    assert_eq!(row.method, method);
                    assert_eq!(row.calibration, treatment, "{name}");
                }
                assert_eq!(bits(&rows[0].values), bits(&values), "{name}");
                assert_eq!(
                    bits(rows[0].standard_errors.as_ref().unwrap()),
                    bits(&errors),
                    "{name}"
                );
                let dv01 = times
                    .iter()
                    .zip(&values)
                    .map(|(t, v)| -1e-4 * t * v)
                    .collect::<Vec<_>>();
                assert_eq!(bits(&rows[1].values), bits(&dv01), "{name}");
                assert_eq!(rows[1].unit, SensitivityUnit::CurrencyPerBasisPoint);
            }
        }
    }

    /// Delta, cash and curve share one `evaluate_aad` run; on residual LSV
    /// cash and curve share one `evaluate_lsv_market_risk` run.
    #[test]
    fn curve_shares_the_cash_and_delta_runs() {
        use crate::stochastic_dividends::{StochasticDividendDelta, StochasticDividendMarketRisk};
        let hw = source("hw", false, false);
        let listed = vec![
            MeasureRequest::native(Measure::SpotDelta),
            curve("hw"),
            cash("hw"),
        ];
        let result = task(&hw, listed).compile().unwrap().evaluate().unwrap();
        let PricingTaskOutput::StochasticDividendMeasureRisk(native) = &result.native_result else {
            panic!("native risk expected")
        };
        let Some(StochasticDividendDelta::Aad(delta)) = &native.delta else {
            panic!("AAD Delta expected")
        };
        let Some(StochasticDividendMarketRisk::Aad(c)) = &native.discount_curve else {
            panic!("AAD curve expected")
        };
        assert_eq!(delta, c);
        assert_eq!(native.dividend_cash, native.discount_curve);

        let lsv = source("lsv", false, false);
        let result = task(&lsv, vec![curve("lsv"), cash("lsv")])
            .compile()
            .unwrap()
            .evaluate()
            .unwrap();
        let PricingTaskOutput::StochasticDividendMeasureRisk(native) = &result.native_result else {
            panic!("native risk expected")
        };
        assert!(matches!(
            native.discount_curve,
            Some(StochasticDividendMarketRisk::Lsv(_))
        ));
        assert_eq!(native.dividend_cash, native.discount_curve);
        assert!(native.delta.is_none());
    }

    #[test]
    fn curve_refusals_name_their_reason() {
        let lsv = source("lsv", false, false);
        let stated = |calibration| {
            MeasureRequest::new(
                Measure::DiscountCurve,
                MeasureMethod::NativeCalibrated { calibration },
            )
        };
        assert_eq!(
            reason(&lsv, stated(BumpCalibration::Fixed)),
            Some(CapabilityReason::NoNativeEstimator)
        );
        assert_eq!(
            reason(&lsv, stated(BumpCalibration::NotApplicable)),
            Some(CapabilityReason::BumpCalibrationMismatch)
        );
        assert_eq!(
            reason(&lsv, MeasureRequest::native(Measure::DiscountCurve)),
            Some(CapabilityReason::NoNativeEstimator)
        );
        let bs = source("bs", false, false);
        assert_eq!(
            reason(&bs, stated(BumpCalibration::Recalibrated)),
            Some(CapabilityReason::NoNativeEstimator)
        );
        for name in ["bs", "hw"] {
            assert_eq!(
                reason(&source(name, false, true), curve(name)),
                Some(CapabilityReason::DiscontinuousPayoff),
                "{name}"
            );
        }
        // A market without cash dividends still has a discount curve.
        for name in ["bs", "hw", "lsv"] {
            let s = without_cash(name);
            let c = s.measure_capabilities(&[curve(name)]);
            assert_ne!(c[0].status, CapabilityStatus::Unsupported, "{name}");
            let result = task(&s, vec![curve(name)])
                .compile()
                .unwrap()
                .evaluate()
                .unwrap();
            assert!(
                result
                    .report
                    .risks()
                    .iter()
                    .any(|r| r.measure == Some(MeasureKind::DiscountCurve)),
                "{name}"
            );
        }

        let t = task(&bs, vec![curve("bs")]);
        let text = pricing_task_to_json(&t).unwrap();
        let compact = text.split_whitespace().collect::<String>();
        assert!(compact.contains(r#"{"type":"discount_curve"}"#), "{text}");
        let back = parse_pricing_task_json(text.as_bytes(), JsonLimits::DEFAULT).unwrap();
        assert_eq!(back.fingerprint().unwrap(), t.fingerprint().unwrap());
        let report = t.compile().unwrap().evaluate().unwrap().report;
        let again = parse_valuation_report_json(
            valuation_report_to_json(&report).unwrap().as_bytes(),
            JsonLimits::DEFAULT,
        )
        .unwrap();
        assert_eq!(again, report);
    }

    fn cash(name: &str) -> MeasureRequest {
        if calibrated(name) {
            MeasureRequest::new(
                Measure::DividendCash,
                MeasureMethod::NativeCalibrated {
                    calibration: BumpCalibration::Recalibrated,
                },
            )
        } else {
            MeasureRequest::native(Measure::DividendCash)
        }
    }

    /// The saved source with its cash dividends removed.
    fn without_cash(name: &str) -> PricingSource {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/task-v1")
            .join(format!("stochastic_dividend_{name}.golden.json"));
        let mut v: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        v["source"]["equity"]["market"]
            .as_object_mut()
            .unwrap()
            .remove("discrete_dividends");
        v["operation"] = json!({"type":"evaluate"});
        parse_pricing_task_json(&serde_json::to_vec(&v).unwrap(), JsonLimits::DEFAULT)
            .unwrap()
            .source()
            .clone()
    }

    /// The cash row is the dedicated API's cash-mean adjoints and errors bit
    /// for bit on an ex-time axis: `evaluate_aad` on BS, Hull-White and
    /// Bergomi, `evaluate_lsv_market_risk` (recalibrated) on residual LSV,
    /// under both samplers.
    #[test]
    fn cash_rows_equal_the_dedicated_apis() {
        use crate::RiskParameterKind as K;
        type Expected = (
            crate::stochastic_dividends::StochasticDividendPrice,
            Vec<f64>,
            Vec<f64>,
            Vec<f64>,
            &'static str,
            CalibrationTreatment,
        );
        let aad = |r: crate::stochastic_dividends::StochasticDividendAadRisk| -> Expected {
            let pick = |x: &[f64]| {
                r.parameter_kinds
                    .iter()
                    .zip(x)
                    .filter(|(k, _)| **k == K::CashMean)
                    .map(|(_, v)| *v)
                    .collect::<Vec<_>>()
            };
            (
                r.price.clone(),
                r.cash_times.to_vec(),
                pick(&r.derivatives),
                pick(&r.standard_errors),
                r.method,
                CalibrationTreatment::NotApplicable,
            )
        };
        for rqmc in [false, true] {
            let sources = vec![
                ("bs", source("bs", rqmc, false)),
                ("hw", source("hw", rqmc, false)),
                ("lsv", source("lsv", rqmc, false)),
                ("bergomi", correlated_sources(rqmc).swap_remove(2).1),
            ];
            for (name, s) in sources {
                let compiled = task(&s, vec![cash(name)]).compile().unwrap();
                let result = compiled.evaluate().unwrap();
                let (price, times, values, errors, method, treatment) =
                    match compiled.dividend_plans() {
                        (Some(p), None) if calibrated(name) => {
                            let r = p.evaluate_lsv_market_risk().unwrap();
                            (
                                r.price.clone(),
                                r.cash_times.to_vec(),
                                r.cash_mean_adjoints.to_vec(),
                                r.cash_mean_standard_errors.to_vec(),
                                r.method,
                                CalibrationTreatment::Recalibrated,
                            )
                        }
                        (Some(p), None) => aad(p.evaluate_aad().unwrap()),
                        (None, Some(p)) => aad(p.evaluate_aad().unwrap()),
                        _ => unreachable!(),
                    };
                assert_eq!(values.len(), 2, "{name}");
                let report = &result.report;
                assert_eq!(report.value().to_bits(), price.value.to_bits(), "{name}");
                let row = report
                    .risks()
                    .iter()
                    .find(|r| r.measure == Some(MeasureKind::DividendCash))
                    .unwrap();
                let method_name = if calibrated(name) {
                    "native_calibrated"
                } else {
                    "native"
                };
                assert_eq!(row.name, format!("dividend_cash.{method_name}"));
                assert_eq!(row.axes.len(), 1);
                assert_eq!(row.axes[0].name, "ex_time");
                assert_eq!(row.axes[0].values, times, "{name}");
                let bits = |x: &[f64]| x.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
                assert_eq!(bits(&row.values), bits(&values), "{name}");
                assert_eq!(
                    bits(row.standard_errors.as_ref().unwrap()),
                    bits(&errors),
                    "{name}"
                );
                assert_eq!(row.method, method);
                assert_eq!(row.calibration, treatment, "{name}");
            }
        }
    }

    /// Delta and cash share one `evaluate_aad` run.
    #[test]
    fn cash_shares_the_delta_run() {
        let s = source("hw", false, false);
        let result = task(
            &s,
            vec![MeasureRequest::native(Measure::SpotDelta), cash("hw")],
        )
        .compile()
        .unwrap()
        .evaluate()
        .unwrap();
        let PricingTaskOutput::StochasticDividendMeasureRisk(native) = &result.native_result else {
            panic!("native risk expected")
        };
        let Some(crate::stochastic_dividends::StochasticDividendDelta::Aad(delta)) = &native.delta
        else {
            panic!("AAD Delta expected")
        };
        let Some(crate::stochastic_dividends::StochasticDividendMarketRisk::Aad(cash)) =
            &native.dividend_cash
        else {
            panic!("AAD cash expected")
        };
        assert_eq!(delta, cash);
        let row = result
            .report
            .risks()
            .iter()
            .find(|r| r.measure == Some(MeasureKind::DividendCash))
            .unwrap();
        assert_eq!(cash.cash_mean_adjoints(), &row.values[..]);
    }

    #[test]
    fn cash_refusals_name_their_reason() {
        let lsv = source("lsv", false, false);
        let stated = |calibration| {
            MeasureRequest::new(
                Measure::DividendCash,
                MeasureMethod::NativeCalibrated { calibration },
            )
        };
        assert_eq!(
            reason(&lsv, stated(BumpCalibration::Fixed)),
            Some(CapabilityReason::NoNativeEstimator)
        );
        assert_eq!(
            reason(&lsv, stated(BumpCalibration::NotApplicable)),
            Some(CapabilityReason::BumpCalibrationMismatch)
        );
        assert_eq!(
            reason(&lsv, MeasureRequest::native(Measure::DividendCash)),
            Some(CapabilityReason::NoNativeEstimator)
        );
        let bs = source("bs", false, false);
        assert_eq!(
            reason(&bs, stated(BumpCalibration::Recalibrated)),
            Some(CapabilityReason::NoNativeEstimator)
        );
        for name in ["bs", "hw"] {
            assert_eq!(
                reason(&source(name, false, true), cash(name)),
                Some(CapabilityReason::DiscontinuousPayoff),
                "{name}"
            );
        }
        for name in ["bs", "hw", "lsv"] {
            assert_eq!(
                reason(&without_cash(name), cash(name)),
                Some(CapabilityReason::ModelCombination),
                "{name}"
            );
        }

        let t = task(&bs, vec![cash("bs")]);
        let text = pricing_task_to_json(&t).unwrap();
        let compact = text.split_whitespace().collect::<String>();
        assert!(compact.contains(r#"{"type":"dividend_cash"}"#), "{text}");
        let back = parse_pricing_task_json(text.as_bytes(), JsonLimits::DEFAULT).unwrap();
        assert_eq!(back.fingerprint().unwrap(), t.fingerprint().unwrap());
        let report = t.compile().unwrap().evaluate().unwrap().report;
        let again = parse_valuation_report_json(
            valuation_report_to_json(&report).unwrap().as_bytes(),
            JsonLimits::DEFAULT,
        )
        .unwrap();
        assert_eq!(again, report);
    }

    fn parameters(ids: Vec<ModelParameterId>) -> MeasureRequest {
        MeasureRequest::native(Measure::ModelParameters(
            ModelParameterSelection::new(ids).unwrap(),
        ))
    }

    /// The native label the dividend AAD gives a parameter on a source.
    fn native_label(name: &str, id: ModelParameterId) -> String {
        use ModelParameterId as P;
        let prefix = if name == "rough_bergomi" {
            "rough"
        } else {
            "bergomi"
        };
        match id {
            P::FactorMeanReversion { factor } => format!("bergomi_mean_reversion[{factor}]"),
            P::RateVolatility { knot } => format!("rate_volatility[{knot}]"),
            P::VolOfVol => format!("{prefix}_vol_of_vol"),
            P::Hurst => "rough_hurst".into(),
            P::MixingWeight => "bergomi_mixing_weight".into(),
            other => other.as_str().into(),
        }
    }

    /// Sources without a calibration step with every parameter they have,
    /// in the native order.
    fn parameterized_sources(
        rqmc: bool,
    ) -> Vec<(&'static str, PricingSource, Vec<ModelParameterId>)> {
        use ModelParameterId as P;
        let basic = vec![
            P::InitialVolatility,
            P::DividendMeanReversion,
            P::EquityLinkage,
            P::DividendVolatility,
        ];
        let with =
            |extra: Vec<ModelParameterId>| basic.iter().copied().chain(extra).collect::<Vec<_>>();
        let hw = source("hw", rqmc, false);
        let PricingSource::StochasticDividend(d) = &hw else {
            unreachable!()
        };
        let RateModelSpec::HullWhite { model, .. } = &d.equity().model().rates else {
            unreachable!()
        };
        let knots = u16::try_from(model.volatilities().len()).unwrap();
        let mut rate = vec![P::RateMeanReversion];
        rate.extend((0..knots).map(|knot| P::RateVolatility { knot }));
        correlated_sources(rqmc)
            .into_iter()
            .map(|(name, s, _)| {
                let ids = match name {
                    "bs" => basic.clone(),
                    "hw" => with(rate.clone()),
                    "bergomi" => with(vec![P::FactorMeanReversion { factor: 0 }, P::VolOfVol]),
                    "bergomi_two_factor" => with(vec![
                        P::FactorMeanReversion { factor: 0 },
                        P::FactorMeanReversion { factor: 1 },
                        P::VolOfVol,
                        P::MixingWeight,
                    ]),
                    _ => with(vec![P::Hurst, P::VolOfVol]),
                };
                (name, s, ids)
            })
            .collect()
    }

    /// The dedicated API a full parameter list selects.
    fn parameter_api(
        compiled: &CompiledPricingTask,
        name: &str,
    ) -> crate::stochastic_dividends::StochasticDividendAadRisk {
        match (compiled.dividend_plans(), name) {
            ((Some(p), None), "bs") => p.evaluate_aad().unwrap(),
            ((Some(p), None), "rough_bergomi") => p.evaluate_rough_aad().unwrap(),
            ((Some(p), None), _) => p.evaluate_bergomi_aad().unwrap(),
            ((None, Some(p)), _) => p.evaluate_hull_white_aad().unwrap(),
            _ => unreachable!(),
        }
    }

    /// Every parameter row is the dedicated reverse's derivative and error
    /// bit for bit, in the requested (reversed) order, on BS, Hull-White and
    /// the three Bergomi drivers under both samplers; the specification's
    /// capability agrees with the plan's.
    #[test]
    fn parameter_rows_equal_the_dedicated_apis() {
        for rqmc in [false, true] {
            for (name, s, ids) in parameterized_sources(rqmc) {
                let mut reversed = ids.clone();
                reversed.reverse();
                let request = parameters(reversed.clone());
                let capability = s.measure_capabilities(std::slice::from_ref(&request));
                assert_ne!(
                    capability[0].status,
                    CapabilityStatus::Unsupported,
                    "{name}: {}",
                    capability[0].detail
                );
                let compiled = task(&s, vec![request]).compile().unwrap();
                let result = compiled.evaluate().unwrap();
                let expected = parameter_api(&compiled, name);
                let report = &result.report;
                assert_eq!(
                    report.value().to_bits(),
                    expected.price.value.to_bits(),
                    "{name}"
                );
                let rows = report
                    .risks()
                    .iter()
                    .filter(|r| r.measure == Some(MeasureKind::ModelParameters))
                    .map(|r| r.model_parameter.unwrap())
                    .collect::<Vec<_>>();
                assert_eq!(rows, reversed, "{name}");
                for &id in &ids {
                    let label = native_label(name, id);
                    let i = expected
                        .parameter_labels
                        .iter()
                        .position(|l| *l == label)
                        .unwrap_or_else(|| panic!("{name} {label}"));
                    let row = report.model_parameter_risk(id).unwrap();
                    assert_eq!(
                        row.values[0].to_bits(),
                        expected.derivatives[i].to_bits(),
                        "{name} {id:?}"
                    );
                    assert_eq!(
                        row.standard_errors.as_ref().unwrap()[0].to_bits(),
                        expected.standard_errors[i].to_bits()
                    );
                    assert_eq!(row.method, expected.method);
                    assert_eq!(row.calibration, CalibrationTreatment::NotApplicable);
                    assert_eq!(row.name, format!("model_parameters.{}.native", id.label()));
                }
                let PricingTaskOutput::StochasticDividendMeasureRisk(native) =
                    &result.native_result
                else {
                    panic!("native risk expected")
                };
                let identified = native.model_parameters.as_ref().unwrap();
                assert_eq!(identified.risk, expected);
                assert_eq!(&*identified.parameters, &ids[..]);
            }
        }
    }

    /// Basic parameters alone come from `evaluate_aad`, shared with the
    /// Delta's run; one extended parameter selects the model's reverse.
    #[test]
    fn parameter_runs_follow_the_listed_parameters() {
        use ModelParameterId as P;
        let s = &parameterized_sources(false)[2].1;
        let compiled = task(
            s,
            vec![
                MeasureRequest::native(Measure::SpotDelta),
                parameters(vec![P::EquityLinkage, P::InitialVolatility]),
            ],
        )
        .compile()
        .unwrap();
        let result = compiled.evaluate().unwrap();
        let (Some(plan), None) = compiled.dividend_plans() else {
            unreachable!()
        };
        let basic = plan.evaluate_aad().unwrap();
        let PricingTaskOutput::StochasticDividendMeasureRisk(native) = &result.native_result else {
            panic!("native risk expected")
        };
        let identified = native.model_parameters.as_ref().unwrap();
        assert_eq!(identified.risk, basic);
        assert_eq!(identified.parameters.len(), 4);
        let row = result
            .report
            .model_parameter_risk(P::EquityLinkage)
            .unwrap();
        assert_eq!(row.values[0].to_bits(), basic.derivatives[3].to_bits());
        assert!(identified.coordinate(P::VolOfVol).is_none());

        let compiled = task(s, vec![parameters(vec![P::InitialVolatility, P::VolOfVol])])
            .compile()
            .unwrap();
        let result = compiled.evaluate().unwrap();
        let (Some(plan), None) = compiled.dividend_plans() else {
            unreachable!()
        };
        let PricingTaskOutput::StochasticDividendMeasureRisk(native) = &result.native_result else {
            panic!("native risk expected")
        };
        assert_eq!(
            native.model_parameters.as_ref().unwrap().risk,
            plan.evaluate_bergomi_aad().unwrap()
        );
    }

    #[test]
    fn parameters_must_belong_to_the_source() {
        use ModelParameterId as P;
        let sources = parameterized_sources(false);
        let missing = [
            ("bs", P::VolOfVol),
            ("bs", P::RateMeanReversion),
            ("hw", P::RateVolatility { knot: 99 }),
            ("bergomi", P::MixingWeight),
            ("bergomi", P::FactorMeanReversion { factor: 1 }),
            ("bergomi", P::Hurst),
            ("rough_bergomi", P::FactorMeanReversion { factor: 0 }),
            ("bergomi_two_factor", P::InitialVariance),
        ];
        for (name, id) in missing {
            let s = &sources.iter().find(|(n, ..)| *n == name).unwrap().1;
            assert_eq!(
                reason(s, parameters(vec![id])),
                Some(CapabilityReason::ModelCombination),
                "{name} {id:?}"
            );
        }
        // Residual LSV parameter risks are finite differences with stated
        // bumps; they keep their dedicated APIs.
        assert_eq!(
            reason(
                &source("lsv", false, false),
                parameters(vec![P::DividendVolatility])
            ),
            Some(CapabilityReason::DedicatedRiskApi)
        );
        assert_eq!(
            reason(
                &source("bs", false, true),
                parameters(vec![P::DividendVolatility])
            ),
            Some(CapabilityReason::DiscontinuousPayoff)
        );
        for id in [
            P::InitialVolatility,
            P::FactorMeanReversion { factor: 1 },
            P::RateVolatility { knot: 3 },
            P::MixingWeight,
            P::RateMeanReversion,
        ] {
            assert_eq!(P::from_label(&id.label()), Some(id));
        }
        assert_eq!(P::from_label("rate_volatility"), None);
        assert_eq!(P::from_label("factor_mean_reversion.x"), None);

        let s = &sources[1].1;
        let t = task(
            s,
            vec![parameters(vec![
                P::RateVolatility { knot: 0 },
                P::DividendMeanReversion,
            ])],
        );
        let text = pricing_task_to_json(&t).unwrap();
        let compact = text.split_whitespace().collect::<String>();
        assert!(
            compact.contains(r#"{"name":"rate_volatility","knot":0}"#),
            "{text}"
        );
        let back = parse_pricing_task_json(text.as_bytes(), JsonLimits::DEFAULT).unwrap();
        assert_eq!(back.fingerprint().unwrap(), t.fingerprint().unwrap());
        let report = t.compile().unwrap().evaluate().unwrap().report;
        let again = parse_valuation_report_json(
            valuation_report_to_json(&report).unwrap().as_bytes(),
            JsonLimits::DEFAULT,
        )
        .unwrap();
        assert_eq!(again, report);
    }

    fn calibrated(name: &str) -> bool {
        name == "lsv"
    }

    fn delta(name: &str) -> MeasureRequest {
        if calibrated(name) {
            MeasureRequest::new(
                Measure::SpotDelta,
                MeasureMethod::NativeCalibrated {
                    calibration: BumpCalibration::Recalibrated,
                },
            )
        } else {
            MeasureRequest::native(Measure::SpotDelta)
        }
    }

    fn bump() -> SpotBump {
        SpotBump::relative(0.01).unwrap()
    }

    fn gamma(name: &str) -> MeasureRequest {
        let method = if calibrated(name) {
            MeasureMethod::NativeSpotBumpCalibrated {
                bump: bump(),
                calibration: BumpCalibration::Recalibrated,
            }
        } else {
            MeasureMethod::NativeSpotBump { bump: bump() }
        };
        MeasureRequest::new(Measure::SpotGamma, method)
    }

    fn task(s: &PricingSource, requests: Vec<MeasureRequest>) -> PricingTask {
        PricingTask::new(
            s.clone(),
            PricingOperation::Measures(MeasureSet::new(requests).unwrap()),
        )
        .unwrap()
    }

    /// Delta and Gamma rows are the dedicated APIs' estimates and errors bit
    /// for bit, on BS, Hull-White and residual-LSV dividends under both
    /// samplers; the price is the plan's price.
    #[test]
    fn spot_rows_equal_the_dedicated_apis() {
        for name in ["bs", "hw", "lsv"] {
            for rqmc in [false, true] {
                let s = source(name, rqmc, false);
                let compiled = task(&s, vec![delta(name), gamma(name)]).compile().unwrap();
                let result = compiled.evaluate().unwrap();
                let config = GammaConfig::new(bump());
                let (price, d, d_se, d_method, g) = match compiled.dividend_plans() {
                    (Some(p), None) if calibrated(name) => {
                        let d = p.evaluate_lsv_spot_risk().unwrap();
                        (
                            p.evaluate().unwrap(),
                            d.delta,
                            d.delta_standard_error,
                            d.method,
                            p.evaluate_lsv_gamma(config).unwrap(),
                        )
                    }
                    (Some(p), None) => {
                        let d = p.evaluate_aad().unwrap();
                        (
                            p.evaluate().unwrap(),
                            d.delta(),
                            d.standard_errors[0],
                            d.method,
                            p.evaluate_gamma(config).unwrap(),
                        )
                    }
                    (None, Some(p)) => {
                        let d = p.evaluate_aad().unwrap();
                        (
                            p.evaluate().unwrap(),
                            d.delta(),
                            d.standard_errors[0],
                            d.method,
                            p.evaluate_gamma(config).unwrap(),
                        )
                    }
                    _ => unreachable!(),
                };
                let report = &result.report;
                assert_eq!(report.value().to_bits(), price.value.to_bits(), "{name}");
                let row = report.risk(MeasureKind::SpotDelta).unwrap();
                assert_eq!(row.values[0].to_bits(), d.to_bits(), "{name}");
                assert_eq!(
                    row.standard_errors.as_ref().unwrap()[0].to_bits(),
                    d_se.to_bits()
                );
                assert_eq!(row.method, d_method);
                let treatment = if calibrated(name) {
                    CalibrationTreatment::Recalibrated
                } else {
                    CalibrationTreatment::NotApplicable
                };
                assert_eq!(row.calibration, treatment);
                let row = report.risk(MeasureKind::SpotGamma).unwrap();
                assert_eq!(row.values[0].to_bits(), g.gamma().to_bits(), "{name}");
                assert_eq!(
                    row.standard_errors.as_ref().unwrap()[0].to_bits(),
                    g.standard_error().to_bits()
                );
                assert_eq!(row.calibration, treatment);
                let PricingTaskOutput::StochasticDividendMeasureRisk(native) =
                    &result.native_result
                else {
                    panic!("native risk expected")
                };
                assert_eq!(native.gamma.as_ref(), Some(&g));
                assert_eq!(native.price, price, "{name}");

                // Gamma alone: the plan's price.
                let only = task(&s, vec![gamma(name)])
                    .compile()
                    .unwrap()
                    .evaluate()
                    .unwrap();
                assert_eq!(only.report.value().to_bits(), price.value.to_bits());
                assert!(only.report.risk(MeasureKind::SpotDelta).is_none());
            }
        }
    }

    fn reason(s: &PricingSource, request: MeasureRequest) -> Option<CapabilityReason> {
        let c = s.measure_capabilities(std::slice::from_ref(&request));
        assert_eq!(c[0].status, CapabilityStatus::Unsupported, "{request:?}");
        let refused = PricingTask::new(
            s.clone(),
            PricingOperation::Measures(MeasureSet::new(vec![request]).unwrap()),
        )
        .map_or(true, |t| t.compile().and_then(|c| c.evaluate()).is_err());
        assert!(refused);
        c[0].reason
    }

    #[test]
    fn refusals_name_their_reason() {
        let lsv = source("lsv", false, false);
        let stated = |calibration| {
            MeasureRequest::new(
                Measure::SpotDelta,
                MeasureMethod::NativeCalibrated { calibration },
            )
        };
        assert_eq!(
            reason(&lsv, stated(BumpCalibration::Fixed)),
            Some(CapabilityReason::NoNativeEstimator)
        );
        assert_eq!(
            reason(&lsv, stated(BumpCalibration::NotApplicable)),
            Some(CapabilityReason::BumpCalibrationMismatch)
        );
        assert_eq!(
            reason(&lsv, MeasureRequest::native(Measure::SpotDelta)),
            Some(CapabilityReason::NoNativeEstimator)
        );
        let bs = source("bs", false, false);
        assert_eq!(
            reason(&bs, stated(BumpCalibration::Recalibrated)),
            Some(CapabilityReason::NoNativeEstimator)
        );
        for name in ["bs", "hw"] {
            let digital = source(name, false, true);
            for request in [delta(name), gamma(name)] {
                assert_eq!(
                    reason(&digital, request),
                    Some(CapabilityReason::DiscontinuousPayoff),
                    "{name}"
                );
            }
        }
        // A price-only list keeps the ordinary result.
        let result = task(&bs, vec![MeasureRequest::native(Measure::Price)])
            .compile()
            .unwrap()
            .evaluate()
            .unwrap();
        assert!(matches!(
            result.native_result,
            PricingTaskOutput::StochasticDividend(_)
        ));
    }

    /// The native Delta agrees with the bump fallback on the same draws (to
    /// about 2e-11 against errors near 1e-2). The relative bump is 1e-6: a
    /// wider one lets a path cross the call's kink, where a difference and
    /// a pathwise derivative differ by up to half a path's slope.
    #[test]
    fn native_delta_agrees_with_bump_and_revalue() {
        for name in ["bs", "lsv"] {
            let s = source(name, true, false);
            let calibration = if calibrated(name) {
                BumpCalibration::Recalibrated
            } else {
                BumpCalibration::NotApplicable
            };
            let bumped = MeasureRequest::new(
                Measure::SpotDelta,
                MeasureMethod::BumpAndRevalue {
                    size: crate::BumpSize::relative(1e-6).unwrap(),
                    calibration,
                },
            );
            let evaluate = |r| {
                task(&s, vec![r])
                    .compile()
                    .unwrap()
                    .evaluate()
                    .unwrap()
                    .report
            };
            let (n, b) = (evaluate(delta(name)), evaluate(bumped));
            let (n, b) = (
                n.risk(MeasureKind::SpotDelta).unwrap(),
                b.risk(MeasureKind::SpotDelta).unwrap(),
            );
            let se = n.standard_errors.as_ref().unwrap()[0];
            assert!(
                (n.values[0] - b.values[0]).abs() < 1e-8 && se > 1e-3,
                "{name}: {} vs {} (se {se})",
                n.values[0],
                b.values[0]
            );
        }
    }
}

#[cfg(test)]
mod basket_tests {
    use serde_json::{Value, json};

    use crate::multi_asset::MultiAssetRiskConfig;
    use crate::risk::SpotBump;
    use crate::*;

    /// The saved basket, optionally under RQMC and without reverse traces.
    fn source(name: &str, rqmc: bool, trace: bool) -> PricingSource {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/task-v1")
            .join(format!("multi_asset_{name}.golden.json"));
        let mut v: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        if rqmc {
            v["source"]["engine"] = json!({"type":"randomized_quasi_monte_carlo",
                "points_per_scramble":16,"scramble_count":4,"master_scramble_seed":82,
                "variance_reduction":{"antithetic":true,"brownian_bridge":false}});
        }
        if !trace {
            let text = serde_json::to_string(&v).unwrap().replace(
                r#""retain_reverse_trace":true"#,
                r#""retain_reverse_trace":false"#,
            );
            v = serde_json::from_str(&text).unwrap();
        }
        v["operation"] = json!({"type":"evaluate"});
        parse_pricing_task_json(&serde_json::to_vec(&v).unwrap(), JsonLimits::DEFAULT)
            .unwrap()
            .source()
            .clone()
    }

    fn calibrated(name: &str) -> bool {
        name == "joint"
    }

    fn recalibrated() -> BumpCalibration {
        BumpCalibration::Recalibrated
    }

    fn delta(name: &str) -> MeasureRequest {
        if calibrated(name) {
            MeasureRequest::new(
                Measure::SpotDelta,
                MeasureMethod::NativeCalibrated {
                    calibration: recalibrated(),
                },
            )
        } else {
            MeasureRequest::native(Measure::SpotDelta)
        }
    }

    fn bump() -> SpotBump {
        SpotBump::relative(0.01).unwrap()
    }

    fn cross_gamma(name: &str, bump: SpotBump) -> MeasureRequest {
        let method = if calibrated(name) {
            MeasureMethod::NativeSpotBumpCalibrated {
                bump,
                calibration: recalibrated(),
            }
        } else {
            MeasureMethod::NativeSpotBump { bump }
        };
        MeasureRequest::new(Measure::CrossGamma, method)
    }

    fn task(s: &PricingSource, requests: Vec<MeasureRequest>) -> PricingTask {
        PricingTask::new(
            s.clone(),
            PricingOperation::Measures(MeasureSet::new(requests).unwrap()),
        )
        .unwrap()
    }

    /// Delta and cross-Gamma rows are `evaluate_aad`'s estimates and errors
    /// bit for bit, and the native result is that result unchanged, on a
    /// Black-Scholes basket and on the joint LSV, Hull-White and Local
    /// Correlation basket under both samplers.
    #[test]
    fn basket_rows_equal_evaluate_aad() {
        for name in ["bs", "joint"] {
            for rqmc in [false, true] {
                let s = source(name, rqmc, true);
                let compiled = task(&s, vec![cross_gamma(name, bump()), delta(name)])
                    .compile()
                    .unwrap();
                let result = compiled.evaluate().unwrap();
                let expected = compiled
                    .basket_plan()
                    .unwrap()
                    .evaluate_aad(MultiAssetRiskConfig {
                        gamma_relative_bump: Some(0.01),
                    })
                    .unwrap();
                let PricingTaskOutput::MultiAsset(native) = &result.native_result else {
                    panic!("basket result expected")
                };
                assert_eq!(**native, expected, "{name}");
                let report = &result.report;
                assert_eq!(
                    report.value().to_bits(),
                    expected.price.value().get().to_bits()
                );
                let treatment = if calibrated(name) {
                    CalibrationTreatment::Recalibrated
                } else {
                    CalibrationTreatment::NotApplicable
                };
                let rows = |kind| {
                    report
                        .risks()
                        .iter()
                        .filter(move |r| r.measure == Some(kind))
                        .collect::<Vec<_>>()
                };
                let deltas = rows(MeasureKind::SpotDelta);
                assert_eq!(deltas.len(), expected.risks.len());
                for (row, risk) in deltas.iter().zip(&expected.risks) {
                    assert_eq!(row.underlying_ids, vec![risk.underlying.get()]);
                    assert_eq!(row.values[0].to_bits(), risk.delta.value().get().to_bits());
                    assert_eq!(
                        row.standard_errors.as_ref().unwrap()[0].to_bits(),
                        risk.delta.standard_error().get().to_bits()
                    );
                    assert_eq!(row.calibration, treatment, "{name}");
                }
                let gammas = rows(MeasureKind::CrossGamma);
                let n = expected.underlyings.len();
                assert_eq!(gammas.len(), n * n);
                for (k, row) in gammas.iter().enumerate() {
                    let (i, j) = (k / n, k % n);
                    assert_eq!(
                        row.underlying_ids,
                        vec![expected.underlyings[i].get(), expected.underlyings[j].get()]
                    );
                    assert_eq!(
                        row.values[0].to_bits(),
                        expected.gamma[i][j].value().get().to_bits()
                    );
                    assert_eq!(row.calibration, treatment);
                }
                let method = if calibrated(name) { "_calibrated" } else { "" };
                assert_eq!(deltas[0].name, format!("spot_delta.native{method}"));
                assert_eq!(
                    gammas[0].name,
                    format!("cross_gamma.native_spot_bump{method}")
                );
            }
        }
    }

    /// Delta alone is first-order AAD; a price-only list keeps `evaluate`.
    #[test]
    fn the_list_selects_the_run() {
        let s = source("bs", false, true);
        let compiled = task(&s, vec![delta("bs")]).compile().unwrap();
        let result = compiled.evaluate().unwrap();
        let plan = compiled.basket_plan().unwrap();
        let PricingTaskOutput::MultiAsset(native) = &result.native_result else {
            panic!("basket result expected")
        };
        assert_eq!(
            **native,
            plan.evaluate_aad(MultiAssetRiskConfig::default()).unwrap()
        );
        assert!(native.gamma.is_empty());
        let compiled = task(&s, vec![MeasureRequest::native(Measure::Price)])
            .compile()
            .unwrap();
        let PricingTaskOutput::MultiAsset(native) = compiled.evaluate().unwrap().native_result
        else {
            panic!("basket result expected")
        };
        assert_eq!(*native, compiled.basket_plan().unwrap().evaluate().unwrap());
    }

    fn reason(s: &PricingSource, request: MeasureRequest) -> Option<CapabilityReason> {
        let c = s.measure_capabilities(std::slice::from_ref(&request));
        assert_eq!(c[0].status, CapabilityStatus::Unsupported, "{request:?}");
        let refused = PricingTask::new(
            s.clone(),
            PricingOperation::Measures(MeasureSet::new(vec![request]).unwrap()),
        )
        .map_or(true, |t| t.compile().and_then(|c| c.evaluate()).is_err());
        assert!(refused);
        c[0].reason
    }

    #[test]
    fn basket_refusals_name_their_reason() {
        let bs = source("bs", false, true);
        let joint = source("joint", false, true);
        let stated = |calibration| {
            MeasureRequest::new(
                Measure::SpotDelta,
                MeasureMethod::NativeCalibrated { calibration },
            )
        };
        assert_eq!(
            reason(&bs, stated(BumpCalibration::Recalibrated)),
            Some(CapabilityReason::BumpCalibrationMismatch)
        );
        assert_eq!(
            reason(&joint, MeasureRequest::native(Measure::SpotDelta)),
            Some(CapabilityReason::BumpCalibrationMismatch)
        );
        assert_eq!(
            reason(&joint, stated(BumpCalibration::NotApplicable)),
            Some(CapabilityReason::BumpCalibrationMismatch)
        );
        assert_eq!(
            reason(&joint, stated(BumpCalibration::Fixed)),
            Some(CapabilityReason::FixedCalibrationUnavailable)
        );
        assert_eq!(
            reason(&bs, cross_gamma("bs", SpotBump::absolute(1.0).unwrap())),
            Some(CapabilityReason::ModelCombination)
        );
        assert_eq!(
            reason(&bs, cross_gamma("bs", SpotBump::relative(1.5).unwrap())),
            Some(CapabilityReason::InvalidGammaBump)
        );
        assert_eq!(
            reason(&source("joint", false, false), delta("joint")),
            Some(CapabilityReason::ReverseTraceRequired)
        );
        // A plain SpotGamma has no native basket estimator.
        assert_eq!(
            reason(
                &bs,
                MeasureRequest::new(
                    Measure::SpotGamma,
                    MeasureMethod::NativeSpotBump { bump: bump() }
                )
            ),
            Some(CapabilityReason::NoNativeEstimator)
        );
    }

    /// The native Delta of each asset agrees with `bump_and_revalue` on the
    /// same draws.
    #[test]
    fn native_delta_agrees_with_bump_and_revalue() {
        let s = source("bs", true, true);
        let native = task(&s, vec![delta("bs")])
            .compile()
            .unwrap()
            .evaluate()
            .unwrap()
            .report;
        let bumped = task(
            &s,
            vec![MeasureRequest::new(
                Measure::SpotDelta,
                MeasureMethod::BumpAndRevalue {
                    size: crate::BumpSize::relative(1e-6).unwrap(),
                    calibration: BumpCalibration::NotApplicable,
                },
            )],
        )
        .compile()
        .unwrap()
        .evaluate()
        .unwrap()
        .report;
        let rows = |r: &ValuationReport| {
            r.risks()
                .iter()
                .filter(|r| r.measure == Some(MeasureKind::SpotDelta))
                .map(|r| {
                    (
                        r.underlying_ids.clone(),
                        r.values[0],
                        r.standard_errors.clone(),
                    )
                })
                .collect::<Vec<_>>()
        };
        let (a, b) = (rows(&native), rows(&bumped));
        assert_eq!(a.len(), 2);
        for ((ids, x, se), (ids_b, y, _)) in a.iter().zip(&b) {
            assert_eq!(ids, ids_b);
            let se = se.as_ref().unwrap()[0];
            assert!(
                (x - y).abs() < 1e-3 * se.max(1e-12) + 1e-8,
                "{ids:?} {x} {y} {se}"
            );
        }
    }

    #[test]
    fn cross_gamma_round_trips() {
        let s = source("bs", false, true);
        let t = task(&s, vec![cross_gamma("bs", bump())]);
        let text = pricing_task_to_json(&t).unwrap();
        let compact = text.split_whitespace().collect::<String>();
        assert!(compact.contains(r#"{"type":"cross_gamma"}"#), "{text}");
        let back = parse_pricing_task_json(text.as_bytes(), JsonLimits::DEFAULT).unwrap();
        assert_eq!(back.fingerprint().unwrap(), t.fingerprint().unwrap());
        let report = t.compile().unwrap().evaluate().unwrap().report;
        let again = parse_valuation_report_json(
            valuation_report_to_json(&report).unwrap().as_bytes(),
            JsonLimits::DEFAULT,
        )
        .unwrap();
        assert_eq!(again, report);
        assert_eq!(report.risks().len(), 4);
    }

    fn curve(name: &str) -> MeasureRequest {
        if calibrated(name) {
            MeasureRequest::new(
                Measure::DiscountCurve,
                MeasureMethod::NativeCalibrated {
                    calibration: recalibrated(),
                },
            )
        } else {
            MeasureRequest::native(Measure::DiscountCurve)
        }
    }

    /// On the shared Hull-White basket, the two curve rows are
    /// `evaluate_aad`'s `hull_white_curve_risk` bit for bit, shared with the
    /// Delta run, under both samplers.
    #[test]
    fn curve_rows_equal_evaluate_aad() {
        let bits = |x: &[f64]| x.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        for rqmc in [false, true] {
            let s = source("joint", rqmc, true);
            let compiled = task(&s, vec![curve("joint"), delta("joint")])
                .compile()
                .unwrap();
            let result = compiled.evaluate().unwrap();
            let expected = compiled
                .basket_plan()
                .unwrap()
                .evaluate_aad(MultiAssetRiskConfig::default())
                .unwrap();
            let PricingTaskOutput::MultiAsset(native) = &result.native_result else {
                panic!("basket result expected")
            };
            assert_eq!(**native, expected);
            let v = expected.hull_white_curve_risk.as_ref().unwrap();
            let rows = result
                .report
                .risks()
                .iter()
                .filter(|r| r.measure == Some(MeasureKind::DiscountCurve))
                .collect::<Vec<_>>();
            assert_eq!(rows.len(), 2);
            for (row, part, estimates) in [
                (rows[0], "log_df", &v.discount_log_df_adjoints),
                (rows[1], "dv01", &v.discount_node_dv01),
            ] {
                assert_eq!(row.name, format!("discount_curve.{part}.native_calibrated"));
                assert!(row.underlying_ids.is_empty());
                assert_eq!(row.axes[0].name, "discount_time");
                assert_eq!(row.axes[0].values, v.discount_time_nodes);
                let values = estimates
                    .iter()
                    .map(|e| e.value().get())
                    .collect::<Vec<_>>();
                let errors = estimates
                    .iter()
                    .map(|e| e.standard_error().get())
                    .collect::<Vec<_>>();
                assert_eq!(bits(&row.values), bits(&values), "{part}");
                assert_eq!(
                    bits(row.standard_errors.as_ref().unwrap()),
                    bits(&errors),
                    "{part}"
                );
                assert_eq!(row.calibration, CalibrationTreatment::Recalibrated);
            }
        }
    }

    #[test]
    fn curve_refusals_name_their_reason() {
        assert_eq!(
            reason(&source("bs", false, true), curve("bs")),
            Some(CapabilityReason::NoNativeEstimator)
        );
        let joint = source("joint", false, true);
        assert_eq!(
            reason(&joint, MeasureRequest::native(Measure::DiscountCurve)),
            Some(CapabilityReason::BumpCalibrationMismatch)
        );
        assert_eq!(
            reason(
                &joint,
                MeasureRequest::new(
                    Measure::DiscountCurve,
                    MeasureMethod::NativeCalibrated {
                        calibration: BumpCalibration::Fixed,
                    },
                )
            ),
            Some(CapabilityReason::FixedCalibrationUnavailable)
        );
        assert_eq!(
            reason(&source("joint", false, false), curve("joint")),
            Some(CapabilityReason::ReverseTraceRequired)
        );
        let t = task(&joint, vec![curve("joint")]);
        let text = pricing_task_to_json(&t).unwrap();
        let back = parse_pricing_task_json(text.as_bytes(), JsonLimits::DEFAULT).unwrap();
        assert_eq!(back.fingerprint().unwrap(), t.fingerprint().unwrap());
    }

    fn pair() -> CorrelationId {
        CorrelationId::AssetPair {
            first: 1,
            second: 2,
        }
    }

    fn bumped_correlation(size: crate::BumpSize, calibration: BumpCalibration) -> MeasureRequest {
        MeasureRequest::new(
            Measure::Correlations(CorrelationSelection::new(vec![pair()]).unwrap()),
            MeasureMethod::BumpAndRevalue { size, calibration },
        )
    }

    /// A saved basket edited as JSON: `edit` changes its source.
    fn edited(name: &str, edit: impl Fn(&mut Value)) -> PricingSource {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/task-v1")
            .join(format!("multi_asset_{name}.golden.json"));
        let mut v: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        v["operation"] = json!({"type":"evaluate"});
        edit(&mut v["source"]);
        parse_pricing_task_json(&serde_json::to_vec(&v).unwrap(), JsonLimits::DEFAULT)
            .unwrap()
            .source()
            .clone()
    }

    /// Moves the (1, 2) Spot correlation by `h` in both entries, at every
    /// date and in every explicit driver matrix.
    fn shifted(source: &mut Value, h: f64) {
        let object = source.as_object_mut().unwrap();
        let mut matrices = Vec::new();
        for (key, value) in object.iter_mut() {
            match key.as_str() {
                "correlation" => matrices.extend(
                    value["entries"]
                        .as_array_mut()
                        .unwrap()
                        .iter_mut()
                        .map(|e| &mut e["matrix"]),
                ),
                "driver_correlations" => {
                    matrices.extend(value.as_array_mut().unwrap().iter_mut());
                }
                _ => {}
            }
        }
        for m in matrices {
            for (i, j) in [(0, 1), (1, 0)] {
                let v = m[i][j].as_f64().unwrap();
                m[i][j] = json!(v + h);
            }
        }
    }

    fn price(s: &PricingSource) -> f64 {
        PricingTask::new(s.clone(), PricingOperation::Evaluate)
            .unwrap()
            .compile()
            .unwrap()
            .evaluate()
            .unwrap()
            .report
            .value()
    }

    /// The correlation row is the central difference of complete
    /// revaluations at the shifted correlation, on the task's draws: on the
    /// Black-Scholes basket, and recalibrated on the LSV and Hull-White basket
    /// with explicit driver matrices (the joint fixture without its Local
    /// Correlation).
    #[test]
    fn correlation_rows_are_paired_revaluations() {
        let h = 0.01;
        let no_lc = |v: &mut Value| {
            v.as_object_mut().unwrap().remove("local_correlation");
        };
        for (name, edit, calibration) in [
            (
                "bs",
                (&|_: &mut Value| {}) as &dyn Fn(&mut Value),
                BumpCalibration::NotApplicable,
            ),
            ("joint", &no_lc, BumpCalibration::Recalibrated),
        ] {
            let s = edited(name, edit);
            let request = bumped_correlation(crate::BumpSize::absolute(h).unwrap(), calibration);
            let result = task(&s, vec![request.clone()])
                .compile()
                .unwrap()
                .evaluate()
                .unwrap();
            let up = price(&edited(name, |v| {
                edit(v);
                shifted(v, h);
            }));
            let down = price(&edited(name, |v| {
                edit(v);
                shifted(v, -h);
            }));
            let expected = (up - down) / (2.0 * h);
            let rows = result
                .report
                .risks()
                .iter()
                .filter(|r| r.measure == Some(MeasureKind::Correlations))
                .collect::<Vec<_>>();
            assert_eq!(rows.len(), 1, "{name}");
            let row = rows[0];
            assert_eq!(row.name, "correlations.asset_pair.1.2.bump_and_revalue");
            assert_eq!(row.underlying_ids, vec![1, 2]);
            assert_eq!(row.correlation, Some(pair()));
            assert_eq!(row.unit, SensitivityUnit::CurrencyPerUnitParameter);
            assert_eq!(row.method_detail, Some(request.method));
            assert!(
                (row.values[0] - expected).abs() <= 1e-10 * (1.0 + expected.abs()),
                "{name}: {} vs {expected}",
                row.values[0]
            );
            let se = row.standard_errors.as_ref().unwrap()[0];
            assert!(se.is_finite() && se >= 0.0, "{name}");
            assert_eq!(
                row.calibration,
                if calibrated(name) {
                    CalibrationTreatment::Recalibrated
                } else {
                    CalibrationTreatment::NotApplicable
                }
            );
            assert_eq!(
                result.report.correlation_risk(pair()).map(|r| &r.values),
                Some(&row.values)
            );
        }
    }

    #[test]
    fn correlation_refusals_name_their_reason() {
        let absolute = crate::BumpSize::absolute(0.01).unwrap();
        let bs = source("bs", false, true);
        let joint = source("joint", false, true);
        let not_applicable = BumpCalibration::NotApplicable;
        // A Local Correlation basket has no single pair correlation.
        assert_eq!(
            reason(&joint, bumped_correlation(absolute, recalibrated())),
            Some(CapabilityReason::ModelCombination)
        );
        assert_eq!(
            reason(&joint, bumped_correlation(absolute, BumpCalibration::Fixed)),
            Some(CapabilityReason::FixedCalibrationUnavailable)
        );
        assert_eq!(
            reason(&joint, bumped_correlation(absolute, not_applicable)),
            Some(CapabilityReason::BumpCalibrationMismatch)
        );
        assert_eq!(
            reason(
                &bs,
                bumped_correlation(crate::BumpSize::relative(0.01).unwrap(), not_applicable)
            ),
            Some(CapabilityReason::ModelCombination)
        );
        // The bumped side leaves [-1, 1].
        assert_eq!(
            reason(
                &bs,
                bumped_correlation(crate::BumpSize::absolute(0.9).unwrap(), not_applicable)
            ),
            Some(CapabilityReason::ModelCombination)
        );
        let other = |id| {
            MeasureRequest::new(
                Measure::Correlations(CorrelationSelection::new(vec![id]).unwrap()),
                MeasureMethod::BumpAndRevalue {
                    size: absolute,
                    calibration: not_applicable,
                },
            )
        };
        for id in [
            CorrelationId::AssetPair {
                first: 1,
                second: 3,
            },
            CorrelationId::EquityRate,
        ] {
            assert_eq!(
                reason(&bs, other(id)),
                Some(CapabilityReason::ModelCombination),
                "{id:?}"
            );
        }
        // No native estimator exists for a basket correlation.
        assert_eq!(
            reason(
                &bs,
                MeasureRequest::native(Measure::Correlations(
                    CorrelationSelection::new(vec![pair()]).unwrap()
                ))
            ),
            Some(CapabilityReason::NoNativeEstimator)
        );
        for (first, second) in [(2, 1), (1, 1)] {
            assert!(
                CorrelationSelection::new(vec![CorrelationId::AssetPair { first, second }])
                    .is_err()
            );
        }
        assert_eq!(CorrelationId::from_label("asset_pair.1.2"), Some(pair()));
        let t = task(&bs, vec![bumped_correlation(absolute, not_applicable)]);
        let text = pricing_task_to_json(&t).unwrap();
        let compact = text.split_whitespace().collect::<String>();
        assert!(
            compact.contains(r#"{"name":"asset_pair","first":1,"second":2}"#),
            "{text}"
        );
        let back = parse_pricing_task_json(text.as_bytes(), JsonLimits::DEFAULT).unwrap();
        assert_eq!(back.fingerprint().unwrap(), t.fingerprint().unwrap());
        let report = t.compile().unwrap().evaluate().unwrap().report;
        let again = parse_valuation_report_json(
            valuation_report_to_json(&report).unwrap().as_bytes(),
            JsonLimits::DEFAULT,
        )
        .unwrap();
        assert_eq!(again, report);
    }
}
