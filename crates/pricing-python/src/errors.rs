//! Preserve typed Rust failures at the Python boundary. Never classify messages.
use std::error::Error;

use pricing::{
    CapabilityReason, MonteCarloError, PricingOperation, SpecificationValidationError, WireError,
};
use pyo3::prelude::*;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Stage {
    Parse,
    Validation,
    Compile,
    Calibrate,
    Evaluate,
    Serialize,
}
impl Stage {
    fn as_str(self) -> &'static str {
        match self {
            Self::Parse => "parse",
            Self::Validation => "validation",
            Self::Compile => "compile",
            Self::Evaluate => "evaluate",
            Self::Serialize => "serialize",
            Self::Calibrate => "calibrate",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ErrorMetadata {
    pub code: String,
    pub stage: Stage,
    pub field: Option<String>,
    pub reason: Option<&'static str>,
    pub causes: Vec<String>,
}
impl ErrorMetadata {
    pub fn domain(code: impl Into<String>, field: Option<String>) -> Self {
        Self {
            code: code.into(),
            stage: Stage::Validation,
            field,
            reason: None,
            causes: vec![],
        }
    }
    pub fn from_error(error: &(dyn Error + 'static), stage: Stage) -> Self {
        let (code, field, reason) = classify(error);
        let mut causes = Vec::new();
        let mut source = error.source();
        while let Some(e) = source {
            causes.push(e.to_string());
            source = e.source();
        }
        Self {
            code: code.into(),
            stage,
            field: field.map(str::to_owned),
            reason,
            causes,
        }
    }
    pub fn attach(&self, py: Python<'_>, error: PyErr, operation: Option<&str>) -> PyErr {
        let value = error.value(py);
        let attributes = (|| -> PyResult<()> {
            value.setattr("code", &self.code)?;
            value.setattr("stage", self.stage.as_str())?;
            value.setattr("field", self.field.as_deref())?;
            value.setattr("operation", operation)?;
            value.setattr("reason", self.reason)?;
            value.setattr("causes", self.causes.clone())?;
            Ok(())
        })();
        // Do not silently return a partially structured exception.
        match attributes {
            Ok(()) => error,
            Err(e) => e,
        }
    }
}

pub(crate) fn exception(
    error: impl Error + 'static,
    stage: Stage,
    operation: Option<&str>,
) -> PyErr {
    let metadata = ErrorMetadata::from_error(&error, stage);
    Python::attach(|py| {
        metadata.attach(
            py,
            crate::PricingError::new_err(error.to_string()),
            operation,
        )
    })
}
pub(crate) fn compile_exception(error: impl Error + 'static) -> PyErr {
    exception(error, Stage::Compile, None)
}
pub(crate) fn serialize_exception(error: impl Error + 'static) -> PyErr {
    exception(error, Stage::Serialize, None)
}
pub(crate) fn internal_exception(message: &'static str) -> PyErr {
    let mut metadata = ErrorMetadata::domain("internal_error", None);
    metadata.stage = Stage::Evaluate;
    Python::attach(|py| metadata.attach(py, crate::PricingError::new_err(message), None))
}
pub(crate) fn operation_name(operation: PricingOperation) -> &'static str {
    match operation {
        PricingOperation::Evaluate => "evaluate",
        PricingOperation::Aad { .. } => "aad",
        PricingOperation::LocalVarianceRisk => "local_variance_risk",
        _ => "unknown",
    }
}
pub(crate) fn capability_reason(reason: CapabilityReason) -> &'static str {
    match reason {
        CapabilityReason::CompilationRequired => "compilation_required",
        CapabilityReason::NativePreflightRequired => "native_preflight_required",
        CapabilityReason::DedicatedRiskApi => "dedicated_risk_api",
        CapabilityReason::ModelCombination => "model_combination",
        CapabilityReason::DiscontinuousPayoff => "discontinuous_payoff",
        CapabilityReason::ReverseTraceRequired => "reverse_trace_required",
        CapabilityReason::InvalidGammaBump => "invalid_gamma_bump",
        CapabilityReason::InvalidSource => "invalid_source",
        _ => "unknown",
    }
}

type Classification<'a> = (&'static str, Option<&'a str>, Option<&'static str>);
fn classify<'a>(error: &'a (dyn Error + 'static)) -> Classification<'a> {
    if let Some(e) = error.downcast_ref::<pricing::core::CoreError>() {
        use pricing::core::CoreError as E;
        return match e {
            E::NonFiniteNumber { field, .. }
            | E::NumberNotPositive { field, .. }
            | E::NumberNegative { field, .. } => ("invalid_number", Some(field), None),
            E::EmptyInput { field }
            | E::InvalidOrdering { field }
            | E::InvalidWeights { field } => ("invalid_configuration", Some(field), None),
            _ => ("invalid_core_input", None, None),
        };
    }
    if let Some(e) = error.downcast_ref::<pricing::bass_lv::BassError>() {
        use pricing::bass_lv::BassError as E;
        return match e {
            E::InvalidInput(_) | E::ConvexOrder { .. } => {
                ("invalid_bass_configuration", None, None)
            }
            E::Calibration { .. } => ("calibration_error", None, None),
            E::GridTooNarrow { .. } => ("calibration_grid_too_narrow", None, None),
            E::Numerical(_) => ("bass_numerical_error", None, None),
        };
    }
    if let Some(e) = error.downcast_ref::<pricing::rough_volatility::FourierError>() {
        use pricing::rough_volatility::FourierError as E;
        return match e {
            E::UnsupportedModel => ("unsupported_model", Some("model"), None),
            E::InvalidInput(_) => ("invalid_fourier_input", None, None),
            E::NumericalFailure(_) => ("fourier_numerical_error", None, None),
            _ => ("fourier_error", None, None),
        };
    }
    if let Some(e) = error.downcast_ref::<pricing::rough_volatility::HestonCalibrationError>() {
        use pricing::rough_volatility::HestonCalibrationError as E;
        return match e {
            E::InvalidInput(_) => ("invalid_calibration_input", None, None),
            E::Fourier(e) => classify(e),
            E::Optimizer(_) => ("calibration_error", None, None),
        };
    }
    if let Some(e) = error.downcast_ref::<SpecificationValidationError>() {
        return match e {
            SpecificationValidationError::Invalid { field, .. } => {
                ("invalid_configuration", Some(field), None)
            }
            SpecificationValidationError::UnsupportedOperation { reason, .. } => (
                "unsupported_operation",
                Some("operation"),
                reason.map(capability_reason),
            ),
            SpecificationValidationError::Request(_) => ("invalid_request", None, None),
            _ => ("invalid_configuration", None, None),
        };
    }
    if let Some(e) = error.downcast_ref::<WireError>() {
        return match e {
            WireError::Json(_) | WireError::Utf8Bom => ("invalid_json", None, None),
            WireError::ResourceLimit { name, .. } => ("resource_limit", Some(name), None),
            WireError::LimitOverrideExceedsHardCap => ("resource_limit", None, None),
            WireError::UnsupportedSchemaVersion(_) => {
                ("unsupported_schema_version", Some("/schema_version"), None)
            }
            WireError::WrongDocumentKind { .. } => {
                ("wrong_document_kind", Some("/document_kind"), None)
            }
            WireError::DomainAt { pointer, .. } => ("invalid_domain_value", Some(pointer), None),
            WireError::Domain(_) => ("invalid_domain_value", None, None),
            WireError::InvalidFingerprint(_) => ("invalid_fingerprint", None, None),
            WireError::UnsupportedSchemaFeature { .. } => {
                ("unsupported_schema_feature", None, None)
            }
        };
    }
    if let Some(e) = error.downcast_ref::<MonteCarloError>() {
        return match e {
            MonteCarloError::UnsupportedEngine => ("unsupported_engine", Some("engine"), None),
            MonteCarloError::UnsupportedModel { .. } => ("unsupported_model", Some("model"), None),
            MonteCarloError::UnsupportedRiskForModel { .. } => {
                ("unsupported_risk_for_model", Some("risk"), None)
            }
            MonteCarloError::InvalidGammaBump { .. } => {
                ("invalid_gamma_bump", Some("gamma_relative_bump"), None)
            }
            MonteCarloError::InsufficientSamplingUnits { .. } => {
                ("insufficient_sampling_units", Some("engine"), None)
            }
            MonteCarloError::NonFiniteTotalVariance { .. } => {
                ("non_finite_total_variance", None, None)
            }
            MonteCarloError::MissingLocalVolatilityReportingBasis => {
                ("missing_reporting_basis", Some("model"), None)
            }
            MonteCarloError::MismatchedLocalVolatilityReportingBasis => {
                ("mismatched_reporting_basis", Some("model"), None)
            }
            MonteCarloError::InvalidLocalVolatilityTimeGrid { .. } => {
                ("invalid_time_grid", Some("time_nodes"), None)
            }
            MonteCarloError::UnsupportedObservationUnderlying { .. } => {
                ("unsupported_observation_underlying", Some("product"), None)
            }
            MonteCarloError::Market(_) => ("invalid_market", Some("market"), None),
            MonteCarloError::Graph(_) => ("graph_error", Some("product"), None),
            MonteCarloError::ResultBuild(_) => ("invalid_result", None, None),
            MonteCarloError::AadConfig(_) | MonteCarloError::RiskConfig(_) => {
                ("invalid_risk_configuration", Some("risk"), None)
            }
            _ => e
                .source()
                .map(classify)
                .unwrap_or(("monte_carlo_error", None, None)),
        };
    }
    if let Some(e) = error.downcast_ref::<pricing::mc::lsv::LsvError>() {
        use pricing::mc::lsv::LsvError as E;
        return match e {
            E::InvalidInput { field, .. } => ("invalid_lsv_input", Some(field), None),
            E::LengthMismatch { field, .. } => ("lsv_length_mismatch", Some(field), None),
            E::NoSupportedCalibrationNode { .. } => ("calibration_error", None, None),
            E::NonFiniteState { .. } => ("non_finite_state", None, None),
            E::ReverseTraceNotRetained => {
                ("reverse_trace_required", Some("retain_reverse_trace"), None)
            }
            _ => ("lsv_error", None, None),
        };
    }
    if let Some(e) = error.downcast_ref::<pricing::multi_asset::MultiAssetError>() {
        use pricing::multi_asset::MultiAssetError as E;
        return match e {
            E::Invalid(_) => ("invalid_multi_asset_input", None, None),
            E::Numerical(_) => ("multi_asset_numerical_error", None, None),
            E::NumericalSource(source) => match classify(source.as_ref()) {
                ("pricing_error", _, _) => ("multi_asset_numerical_error", None, None),
                metadata => metadata,
            },
            E::Correlation(_) => ("invalid_correlation", Some("correlation"), None),
            E::Market(_) => ("invalid_market", Some("market"), None),
            E::Graph(_) => ("graph_error", Some("product"), None),
            E::Core(_) => ("invalid_domain_value", None, None),
            _ => ("multi_asset_error", None, None),
        };
    }
    if let Some(e) = error.downcast_ref::<pricing::models::StochasticDividendError>() {
        use pricing::models::StochasticDividendError as E;
        return match e {
            E::InvalidInput { field } => ("invalid_dividend_input", Some(field), None),
            E::Unsupported { .. } => ("unsupported_dividend_feature", None, None),
            _ => ("stochastic_dividend_error", None, None),
        };
    }
    if let Some(e) = error.downcast_ref::<pricing::models::HullWhiteError>() {
        use pricing::models::HullWhiteError as E;
        return match e {
            E::InvalidInput { field, .. } => ("invalid_hull_white_input", Some(field), None),
            E::InvalidCorrelation => ("invalid_correlation", Some("correlation"), None),
            E::NonPositiveState { .. } => ("non_positive_state", None, None),
            E::Unsupported { .. } => ("unsupported_hull_white_feature", None, None),
            _ => ("hull_white_error", None, None),
        };
    }
    if let Some(e) = error.downcast_ref::<pricing::mc::LocalVolError>() {
        use pricing::mc::LocalVolError as E;
        return match e {
            E::NonFiniteState { .. } | E::NonFiniteAdjoint { .. } => {
                ("non_finite_state", None, None)
            }
            E::StepCountOverflow => ("resource_limit", Some("time_grid"), None),
            _ => ("local_vol_error", None, None),
        };
    }
    if let Some(e) = error.downcast_ref::<pricing::mc::RqmcPlanError>() {
        use pricing::mc::RqmcPlanError as E;
        return match e {
            E::DimensionLimitExceeded { .. } => {
                ("resource_limit", Some("effective_dimension"), None)
            }
            E::TableSizeOverflow | E::AllocationFailed => {
                ("resource_limit", Some("rqmc_table"), None)
            }
            _ => ("invalid_rqmc_configuration", None, None),
        };
    }
    if let Some(e) = error.downcast_ref::<pricing::mc::ExecutorBuildError>() {
        use pricing::mc::ExecutorBuildError as E;
        return match e {
            E::ZeroWorkerThreads | E::ThreadCountUnsupported { .. } => {
                ("invalid_execution_policy", Some("worker_threads"), None)
            }
            E::ZeroReductionBlockSize => (
                "invalid_execution_policy",
                Some("reduction_block_size"),
                None,
            ),
            E::PoolBuild => ("executor_build_error", None, None),
            _ => ("executor_build_error", None, None),
        };
    }
    if error.is::<pricing::mc::ExecutionError>() {
        return ("resource_limit", Some("reduction_blocks"), None);
    }
    error
        .source()
        .map(classify)
        .unwrap_or(("pricing_error", None, None))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nested_calibration_errors_preserve_code_stage_and_causes() {
        let error = pricing::PricingTaskError::from(MonteCarloError::from(
            pricing::mc::hull_white::HullWhiteMcError::from(
                pricing::mc::lsv::LsvError::NoSupportedCalibrationNode { time_index: 3 },
            ),
        ));
        let metadata = ErrorMetadata::from_error(&error, Stage::Compile);
        assert_eq!(metadata.code, "calibration_error");
        assert_eq!(metadata.stage, Stage::Compile);
        assert_eq!(metadata.causes.len(), 3);
        assert!(metadata.field.is_none());
        let evaluated = ErrorMetadata::from_error(&error, Stage::Evaluate);
        assert_eq!(evaluated.code, metadata.code);
        assert_eq!(evaluated.stage, Stage::Evaluate);
    }

    #[test]
    fn errors_are_classified_by_variants_not_display_text() {
        for text in ["first numerical failure", "a completely different message"] {
            let error = pricing::multi_asset::MultiAssetError::Numerical(text.into());
            assert_eq!(
                classify(&error),
                ("multi_asset_numerical_error", None, None)
            );
        }
        let resource = MonteCarloError::from(pricing::mc::RqmcPlanError::AllocationFailed);
        assert_eq!(
            classify(&resource),
            ("resource_limit", Some("rqmc_table"), None)
        );
        let numerical = pricing::mc::lsv::LsvError::NonFiniteState {
            time_index: 0,
            path: 2,
        };
        assert_eq!(classify(&numerical), ("non_finite_state", None, None));
        let unsupported = SpecificationValidationError::UnsupportedOperation {
            reason: Some(CapabilityReason::ReverseTraceRequired),
            detail: "display only".into(),
        };
        assert_eq!(
            classify(&unsupported),
            (
                "unsupported_operation",
                Some("operation"),
                Some("reverse_trace_required")
            )
        );
    }

    #[test]
    fn multi_asset_sources_keep_nested_model_classification_and_unknown_fallback() {
        use pricing::multi_asset::MultiAssetError;
        let error = pricing::PricingTaskError::from(MultiAssetError::NumericalSource(Box::new(
            pricing::mc::lsv::LsvError::NoSupportedCalibrationNode { time_index: 3 },
        )));
        let metadata = ErrorMetadata::from_error(&error, Stage::Compile);
        assert_eq!(metadata.code, "calibration_error");
        assert_eq!(metadata.stage, Stage::Compile);
        assert_eq!(metadata.causes.len(), 2);
        assert!(metadata.field.is_none());

        for text in ["resource limit", "calibration error", "unknown failure"] {
            let error = MultiAssetError::NumericalSource(Box::new(std::io::Error::other(text)));
            assert_eq!(
                classify(&error),
                ("multi_asset_numerical_error", None, None)
            );
            let legacy = MultiAssetError::Numerical(text.into());
            assert_eq!(error.to_string(), legacy.to_string());
            assert_eq!(
                ErrorMetadata::from_error(&error, Stage::Evaluate).causes,
                [text]
            );
        }
    }
}
