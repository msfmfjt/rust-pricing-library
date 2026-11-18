//! Compile/evaluate once at the facade boundary; native numerical loops are reused.
use std::sync::Arc;

use crate::hull_white::HullWhiteAadRisk;
use crate::lsv::{LsvLocalVarianceRisk, RoughFamilyLsvMeasureRisk};
use crate::multi_asset::{
    MultiAssetError, MultiAssetPrice, MultiAssetPricingPlan, MultiAssetProduct,
    MultiAssetRiskConfig,
};
use crate::rough_volatility::RoughVolatilityMeasureRisk;
use crate::stochastic_dividends::{
    StochasticDividendAadRisk, StochasticDividendHullWhitePricingPlan,
    StochasticDividendLocalVarianceRisk, StochasticDividendPrice, StochasticDividendPricingPlan,
};
use crate::*;

#[derive(Debug)]
#[non_exhaustive]
pub enum PricingTaskError {
    Validation(SpecificationValidationError),
    MonteCarlo(Box<MonteCarloError>),
    MultiAsset(MultiAssetError),
    Wire(WireError),
}
impl std::fmt::Display for PricingTaskError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Validation(e) => e.fmt(f),
            Self::MonteCarlo(e) => e.fmt(f),
            Self::MultiAsset(e) => e.fmt(f),
            Self::Wire(e) => e.fmt(f),
        }
    }
}
impl std::error::Error for PricingTaskError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(match self {
            Self::Validation(e) => e,
            Self::MonteCarlo(e) => e.as_ref(),
            Self::MultiAsset(e) => e,
            Self::Wire(e) => e,
        })
    }
}
impl From<MonteCarloError> for PricingTaskError {
    fn from(e: MonteCarloError) -> Self {
        Self::MonteCarlo(Box::new(e))
    }
}
impl From<MultiAssetError> for PricingTaskError {
    fn from(e: MultiAssetError) -> Self {
        Self::MultiAsset(e)
    }
}
impl From<WireError> for PricingTaskError {
    fn from(e: WireError) -> Self {
        Self::Wire(e)
    }
}

#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum PricingTaskOutput {
    SingleAsset(SpecificationResult),
    MultiAsset(Box<MultiAssetPrice>),
    StochasticDividend(StochasticDividendPrice),
    HullWhiteRisk(Box<HullWhiteAadRisk>),
    LocalVarianceRisk(Box<LsvLocalVarianceRisk>),
    /// Native Spot measures of rough-family LSV, from a `Measures` task.
    RoughFamilyLsvMeasureRisk(Box<RoughFamilyLsvMeasureRisk>),
    /// Native measures of a direct rough-volatility source, from a
    /// `Measures` task.
    RoughVolatilityMeasureRisk(Box<RoughVolatilityMeasureRisk>),
    StochasticDividendRisk(Box<StochasticDividendAadRisk>),
    StochasticDividendLocalVarianceRisk(Box<StochasticDividendLocalVarianceRisk>),
}

#[derive(Debug)]
enum TaskBackend {
    SingleAsset(CompiledPricingSpecification),
    MultiAsset(Box<MultiAssetPricingPlan>),
    Dividend(Box<StochasticDividendPricingPlan>),
    DividendHullWhite(Box<StochasticDividendHullWhitePricingPlan>),
}

#[derive(Clone, Debug)]
pub struct CompiledPricingTask {
    task: PricingTask,
    fingerprint: Fingerprint,
    backend: Arc<TaskBackend>,
    /// Compiled bumped sides of the task's bump-and-revalue requests.
    scenarios: Option<super::bump::SharedScenarios>,
}
#[derive(Clone, Debug, PartialEq)]
pub struct PricingTaskResult {
    pub report: ValuationReport,
    pub native_result: PricingTaskOutput,
}

impl PricingTask {
    pub fn fingerprint(&self) -> Result<Fingerprint, WireError> {
        crate::fingerprint_pricing_task(self)
    }
    pub fn compile(&self) -> Result<CompiledPricingTask, PricingTaskError> {
        let capability = self.source().capability(self.operation());
        if capability.status == CapabilityStatus::Unsupported {
            return Err(PricingTaskError::Validation(
                SpecificationValidationError::UnsupportedOperation {
                    reason: capability.reason,
                    detail: capability.detail,
                },
            ));
        }
        let backend = match self.source() {
            PricingSource::SingleAsset(s) => TaskBackend::SingleAsset(match self.measures() {
                Some(measures) => s.compile_measures(measures)?,
                None => s.compile()?,
            }),
            PricingSource::MultiAsset(s) => {
                let m = s.model();
                let product = MultiAssetProduct::from_product_graph(s.product().clone());
                let plan = if let Some((local, extensions)) = &m.local_correlation {
                    MultiAssetPricingPlan::compile_with_joint_local_correlation(
                        s.valuation_date(),
                        product,
                        s.markets().to_vec(),
                        m.models.clone(),
                        m.correlation.clone(),
                        s.engine(),
                        s.execution(),
                        s.maximum_step(),
                        m.lsv.clone(),
                        m.driver_correlations.clone(),
                        m.hull_white.clone(),
                        local.clone(),
                        extensions.clone(),
                    )?
                } else if let Some(hw) = &m.hull_white {
                    MultiAssetPricingPlan::compile_with_hull_white(
                        s.valuation_date(),
                        product,
                        s.markets().to_vec(),
                        m.models.clone(),
                        m.correlation.clone(),
                        s.engine(),
                        s.execution(),
                        s.maximum_step(),
                        m.lsv.clone(),
                        m.driver_correlations.clone(),
                        hw.clone(),
                    )?
                } else {
                    MultiAssetPricingPlan::compile_with_bergomi_lsv(
                        s.valuation_date(),
                        product,
                        s.markets().to_vec(),
                        m.models.clone(),
                        m.correlation.clone(),
                        s.engine(),
                        s.execution(),
                        s.maximum_step(),
                        m.lsv.clone(),
                        m.driver_correlations.clone(),
                    )?
                };
                TaskBackend::MultiAsset(Box::new(plan))
            }
            PricingSource::StochasticDividend(s) => compile_dividend(s)?,
        };
        let mut compiled =
            CompiledPricingTask::from_backend(self.clone(), Arc::new(backend), None)?;
        let revaluation = self
            .measures()
            .map(MeasureSet::revaluation_requests)
            .unwrap_or_default();
        if !revaluation.is_empty() {
            // Fixed-calibration sides reuse the calibration just realized.
            compiled.scenarios = Some(Arc::new(super::bump::CompiledScenarios::compile(
                &compiled,
                revaluation,
            )?));
        }
        Ok(compiled)
    }
}

fn compile_dividend(s: &StochasticDividendSpecification) -> Result<TaskBackend, MonteCarloError> {
    let source = s.equity();
    let r = source.carrier_request();
    let d = s.dividends();
    let h = s.maximum_step();
    let p = source.numerics().execution_policy;
    if let RateModelSpec::HullWhite {
        model,
        equity_rate_correlation,
        ..
    } = &source.model().rates
    {
        return StochasticDividendHullWhitePricingPlan::compile_bs(
            r,
            d,
            model.clone(),
            *equity_rate_correlation,
            s.rate_correlation().expect("validated rate correlation"),
            h,
            p,
        )
        .map(|plan| TaskBackend::DividendHullWhite(Box::new(plan)));
    }
    let correlations = s.volatility_correlations();
    let plan = match &source.model().equity {
        EquityModelSpec::Standard(_) => StochasticDividendPricingPlan::compile_bs(r, d, h, p)?,
        EquityModelSpec::StochasticVolatility { factor, .. } => match factor {
            VolatilityFactorSpec::Bergomi(f) => {
                StochasticDividendPricingPlan::compile_bergomi(r, d, *f, correlations[0], h, p)?
            }
            VolatilityFactorSpec::BergomiTwoFactor(f) => {
                StochasticDividendPricingPlan::compile_bergomi_two_factor(
                    r,
                    d,
                    *f,
                    [correlations[0], correlations[1]],
                    h,
                    p,
                )?
            }
            VolatilityFactorSpec::RoughBergomi(f) => {
                StochasticDividendPricingPlan::compile_rough_bergomi(
                    r,
                    d,
                    *f,
                    correlations[0],
                    h,
                    p,
                )?
            }
        },
        EquityModelSpec::LocalStochasticVolatility {
            factor, particles, ..
        } => match factor {
            VolatilityFactorSpec::Bergomi(f) => StochasticDividendPricingPlan::compile_bergomi_lsv(
                r,
                d,
                *f,
                correlations[0],
                particles.clone(),
                h,
                p,
            )?,
            VolatilityFactorSpec::BergomiTwoFactor(f) => {
                StochasticDividendPricingPlan::compile_bergomi_two_factor_lsv(
                    r,
                    d,
                    *f,
                    [correlations[0], correlations[1]],
                    particles.clone(),
                    h,
                    p,
                )?
            }
            VolatilityFactorSpec::RoughBergomi(f) => {
                StochasticDividendPricingPlan::compile_rough_bergomi_lsv(
                    r,
                    d,
                    *f,
                    correlations[0],
                    particles.clone(),
                    h,
                    p,
                )?
            }
        },
        _ => unreachable!("dividend source constructor validates model composition"),
    };
    Ok(TaskBackend::Dividend(Box::new(plan)))
}

impl CompiledPricingTask {
    fn from_backend(
        task: PricingTask,
        backend: Arc<TaskBackend>,
        scenarios: Option<super::bump::SharedScenarios>,
    ) -> Result<Self, PricingTaskError> {
        let compiled = Self {
            fingerprint: task.fingerprint()?,
            task,
            backend,
            scenarios,
        };
        compiled.validate_operation(&compiled.operation())?;
        Ok(compiled)
    }
    /// Create an immutable task for another supported operation on the same
    /// native plan. No compilation or calibration is performed. Source and
    /// realized native preflight both apply; missing reverse traces are errors.
    pub fn with_operation(&self, operation: PricingOperation) -> Result<Self, PricingTaskError> {
        let task = self
            .task
            .with_operation(operation)
            .map_err(PricingTaskError::Validation)?;
        // Bumped sides are compiled sources of their own; reselection reuses
        // them only for the same bump requests and never compiles new ones.
        let revaluation = task
            .measures()
            .map(MeasureSet::revaluation_requests)
            .unwrap_or_default();
        let scenarios = match &self.scenarios {
            _ if revaluation.is_empty() => None,
            Some(s) if s.requests == revaluation => Some(Arc::clone(s)),
            _ => return Err(super::specification::reselected()),
        };
        Self::from_backend(task, Arc::clone(&self.backend), scenarios)
    }
    /// Calibrations and side evaluations spent on bump-and-revalue requests
    /// so far: the calibrations realized when the sides were compiled and
    /// two evaluations per scenario for every `evaluate()` call.
    #[must_use]
    pub fn bump_cost(&self) -> BumpCost {
        self.scenarios
            .as_ref()
            .map(|s| s.realized_cost())
            .unwrap_or_default()
    }
    /// The plan's price and, where the engine's reductions show them, its
    /// per-sampling-unit prices on the task's random coordinates. One native
    /// price evaluation, recorded; no value or reduction order changes.
    pub(crate) fn price_and_units(&self) -> Result<(f64, Option<Vec<f64>>), PricingTaskError> {
        let price = [MeasureRequest::native(Measure::Price)];
        let (output, recorded) = crate::engine::mc::execution::capture_reductions(|| {
            self.backend.measures().evaluate_measures(&price)
        });
        let (value, ..) = super::source_report::headline(&output?);
        Ok((
            value,
            super::bump::units_from_recorded(&recorded, self.task.source().engine()),
        ))
    }
    /// A price task on `side` that reuses this task's realized calibration.
    pub(crate) fn with_fixed_calibration(
        &self,
        side: PricingSource,
    ) -> Result<Self, PricingTaskError> {
        let backend = match (self.backend.as_ref(), &side) {
            (TaskBackend::SingleAsset(plan), PricingSource::SingleAsset(spec)) => {
                TaskBackend::SingleAsset(plan.with_fixed_calibration(spec)?)
            }
            (TaskBackend::Dividend(plan), PricingSource::StochasticDividend(spec)) => {
                TaskBackend::Dividend(Box::new(plan.with_fixed_calibration(
                    spec.equity().carrier_request(),
                    spec.maximum_step(),
                )?))
            }
            _ => {
                return Err(PricingTaskError::Validation(crate::api::source::invalid(
                    "operation.measures",
                    "a fixed calibration is available only for single-asset LSV",
                )));
            }
        };
        let task = PricingTask::new(side, PricingOperation::Evaluate)
            .map_err(PricingTaskError::Validation)?;
        Self::from_backend(task, Arc::new(backend), None)
    }
    /// Calibration diagnostics of the realized plan, without evaluating it.
    pub(crate) fn calibration_blocks(&self) -> Vec<ReportedCalibration> {
        match (self.backend.as_ref(), self.task.source()) {
            (TaskBackend::SingleAsset(p), PricingSource::SingleAsset(s)) => {
                p.calibration_diagnostics(s)
            }
            (TaskBackend::MultiAsset(p), PricingSource::MultiAsset(s)) => {
                super::calibration_report::multi_asset(p, s)
            }
            (TaskBackend::Dividend(p), PricingSource::StochasticDividend(s)) => {
                p.calibration_diagnostics(s.equity().product().underlying().get())
            }
            (TaskBackend::DividendHullWhite(_), _) => Vec::new(),
            _ => unreachable!("compiled source/backend pairing"),
        }
    }
    /// Replayable source and selected operation, without compiled state.
    #[must_use]
    pub fn task(&self) -> &PricingTask {
        &self.task
    }
    #[must_use]
    pub fn operation(&self) -> PricingOperation {
        self.task.operation()
    }
    #[must_use]
    pub fn task_fingerprint(&self) -> Fingerprint {
        self.fingerprint
    }
    #[must_use]
    pub fn plan_fingerprint(&self) -> String {
        match self.backend.as_ref() {
            TaskBackend::SingleAsset(p) => p.plan_fingerprint().to_string(),
            TaskBackend::MultiAsset(p) => format!("blake3-256:{}", p.fingerprint()),
            TaskBackend::Dividend(p) => p.plan_fingerprint().to_string(),
            TaskBackend::DividendHullWhite(p) => p.plan_fingerprint().to_string(),
        }
    }
    #[must_use]
    pub fn capability(&self) -> PricingCapability {
        self.capability_for(self.operation())
    }
    /// Query another operation against the source and realized native plan.
    /// Does not compile, calibrate or execute pricing/risk paths.
    #[must_use]
    pub fn capability_for(&self, operation: PricingOperation) -> PricingCapability {
        let source = self.task.source().capability(operation.clone());
        if source.status == CapabilityStatus::Unsupported {
            return source;
        }
        match self.validate_operation(&operation) {
            Ok(()) => PricingCapability::new(
                operation,
                CapabilityStatus::Supported,
                None,
                "compiled source and native operation preflight passed; numerical evaluation can still fail",
            ),
            Err(e) => PricingCapability::new(
                operation,
                CapabilityStatus::Unsupported,
                Some(CapabilityReason::NativePreflightRequired),
                e.to_string(),
            ),
        }
    }
    /// One answer per request against the source and the realized plan. A
    /// standard plan fixes its Greeks at compilation, so a measure it was not
    /// compiled with is unsupported here even when the source allows it.
    #[must_use]
    pub fn measure_capabilities(&self, requests: &[MeasureRequest]) -> Vec<MeasureCapability> {
        self.task
            .source()
            .measure_capabilities(requests)
            .into_iter()
            .map(|source| {
                if source.status == CapabilityStatus::Unsupported {
                    source
                } else if source.request.method.is_revaluation() {
                    let compiled = self
                        .scenarios
                        .as_ref()
                        .is_some_and(|s| s.requests.contains(&source.request));
                    if compiled {
                        MeasureCapability::new(
                            &source.request,
                            CapabilityStatus::Supported,
                            None,
                            "bumped sides are compiled; numerical evaluation can still fail",
                        )
                    } else {
                        MeasureCapability::new(
                            &source.request,
                            CapabilityStatus::Unsupported,
                            Some(CapabilityReason::NativePreflightRequired),
                            "the bumped sides of this request were not compiled; compile a task that requests it",
                        )
                    }
                } else {
                    self.backend.measures().preflight(&source.request)
                }
            })
            .collect()
    }
    fn validate_operation(&self, operation: &PricingOperation) -> Result<(), PricingTaskError> {
        use PricingOperation as O;
        match (self.backend.as_ref(), operation) {
            (TaskBackend::SingleAsset(p), operation) => {
                let PricingSource::SingleAsset(source) = self.task.source() else {
                    unreachable!("compiled source/backend pairing")
                };
                p.validate_operation(operation, source)
            }
            (backend, O::Measures(set)) => {
                backend.measures().validate_measures(&set.native_requests())
            }
            (_, O::Evaluate) => Ok(()),
            (
                TaskBackend::MultiAsset(p),
                O::Aad {
                    gamma_relative_bump,
                },
            ) => p
                .validate_aad(MultiAssetRiskConfig {
                    gamma_relative_bump: *gamma_relative_bump,
                })
                .map_err(Into::into),
            (TaskBackend::Dividend(p), O::Aad { .. }) => {
                p.validate_source_aad().map_err(Into::into)
            }
            (TaskBackend::DividendHullWhite(p), O::Aad { .. }) => {
                p.validate_source_aad().map_err(Into::into)
            }
            (TaskBackend::Dividend(p), O::LocalVarianceRisk) => {
                p.validate_source_local_variance().map_err(Into::into)
            }
            _ => Err(PricingTaskError::Validation(crate::api::source::invalid(
                "operation",
                "unsupported operation",
            ))),
        }
    }
    pub fn evaluate(&self) -> Result<PricingTaskResult, PricingTaskError> {
        use PricingOperation as O;
        let operation = self.operation();
        self.validate_operation(&operation)?;
        let output = match (self.backend.as_ref(), &operation) {
            (TaskBackend::SingleAsset(p), operation) => {
                let PricingSource::SingleAsset(source) = self.task.source() else {
                    unreachable!("compiled source/backend pairing")
                };
                p.evaluate_operation(operation, source)?
            }
            (backend, O::Measures(set)) => backend
                .measures()
                .evaluate_measures(&set.native_requests())?,
            (TaskBackend::MultiAsset(p), O::Evaluate) => {
                PricingTaskOutput::MultiAsset(Box::new(p.evaluate()?))
            }
            (
                TaskBackend::MultiAsset(p),
                O::Aad {
                    gamma_relative_bump,
                },
            ) => PricingTaskOutput::MultiAsset(Box::new(p.evaluate_aad(MultiAssetRiskConfig {
                gamma_relative_bump: *gamma_relative_bump,
            })?)),
            (TaskBackend::Dividend(p), O::Evaluate) => {
                PricingTaskOutput::StochasticDividend(p.evaluate()?)
            }
            (TaskBackend::DividendHullWhite(p), O::Evaluate) => {
                PricingTaskOutput::StochasticDividend(p.evaluate()?)
            }
            (TaskBackend::Dividend(p), O::Aad { .. }) => {
                PricingTaskOutput::StochasticDividendRisk(Box::new(p.evaluate_aad()?))
            }
            (TaskBackend::DividendHullWhite(p), O::Aad { .. }) => {
                PricingTaskOutput::StochasticDividendRisk(Box::new(p.evaluate_aad()?))
            }
            (TaskBackend::Dividend(p), O::LocalVarianceRisk) => {
                PricingTaskOutput::StochasticDividendLocalVarianceRisk(Box::new(
                    p.evaluate_local_variance_risk()?,
                ))
            }
            _ => unreachable!("validated operation"),
        };
        let mut report = super::source_report::build(
            &self.task,
            self.fingerprint,
            self.plan_fingerprint(),
            &output,
            self.calibration_blocks(),
        )?;
        if let Some(scenarios) = &self.scenarios {
            let estimates = scenarios.evaluate(self.task.source(), report.value(), || {
                self.price_and_units().map(|(_, units)| units)
            })?;
            super::source_report::bump_rows(&mut report, &estimates)?;
        }
        Ok(PricingTaskResult {
            report,
            native_result: output,
        })
    }
}

impl TaskBackend {
    fn measures(&self) -> &dyn super::measure_backend::MeasureBackend {
        match self {
            Self::SingleAsset(p) => p,
            Self::MultiAsset(p) => p.as_ref(),
            Self::Dividend(p) => p.as_ref(),
            Self::DividendHullWhite(p) => p.as_ref(),
        }
    }
}

#[cfg(test)]
mod reuse_tests {
    use super::*;

    #[test]
    fn cloned_and_reselected_tasks_share_the_realized_plan() {
        let task = parse_pricing_task_json(
            include_bytes!("../../../../../fixtures/task-v1/multi_asset_joint.golden.json"),
            JsonLimits::DEFAULT,
        )
        .unwrap();
        let risk = task.compile().unwrap();
        let price = risk.with_operation(PricingOperation::Evaluate).unwrap();
        let cloned = price.clone();
        // Output equality alone cannot detect an accidental recalibration or
        // deep clone. These must own the same realized calibration allocation.
        assert!(Arc::ptr_eq(&risk.backend, &price.backend));
        assert!(Arc::ptr_eq(&price.backend, &cloned.backend));
        assert!(std::ptr::eq(task.source(), price.task().source()));
        assert!(std::ptr::eq(price.task().source(), cloned.task().source()));
        assert_ne!(risk.task_fingerprint(), price.task_fingerprint());
        assert_eq!(risk.plan_fingerprint(), price.plan_fingerprint());
        drop(risk);
        drop(price);
        drop(task);
        assert_eq!(cloned.capability().status, CapabilityStatus::Supported);
        assert!(cloned.evaluate().is_ok());
    }
}
