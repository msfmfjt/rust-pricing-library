//! Compile/evaluate once at the facade boundary; native numerical loops are reused.
use crate::hull_white::HullWhiteAadRisk;
use crate::lsv::LsvLocalVarianceRisk;
use crate::multi_asset::{
    MultiAssetError, MultiAssetPrice, MultiAssetPricingPlan, MultiAssetProduct,
    MultiAssetRiskConfig,
};
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
    StochasticDividendRisk(Box<StochasticDividendAadRisk>),
    StochasticDividendLocalVarianceRisk(Box<StochasticDividendLocalVarianceRisk>),
}

#[derive(Clone, Debug)]
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
    backend: TaskBackend,
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
            return Err(PricingTaskError::Validation(crate::api::source::invalid(
                "operation",
                capability.detail,
            )));
        }
        let backend = match self.source() {
            PricingSource::SingleAsset(s) => TaskBackend::SingleAsset(s.compile()?),
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
        let compiled = CompiledPricingTask {
            task: self.clone(),
            fingerprint: self.fingerprint()?,
            backend,
        };
        compiled.validate_operation()?;
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
    #[must_use]
    pub fn task_fingerprint(&self) -> Fingerprint {
        self.fingerprint
    }
    #[must_use]
    pub fn plan_fingerprint(&self) -> String {
        match &self.backend {
            TaskBackend::SingleAsset(p) => p.plan_fingerprint().to_string(),
            TaskBackend::MultiAsset(p) => format!("blake3-256:{}", p.fingerprint()),
            TaskBackend::Dividend(p) => p.plan_fingerprint().to_string(),
            TaskBackend::DividendHullWhite(p) => p.plan_fingerprint().to_string(),
        }
    }
    #[must_use]
    pub fn capability(&self) -> PricingCapability {
        match self.validate_operation() {
            Ok(()) => PricingCapability::new(
                self.task.operation(),
                CapabilityStatus::Supported,
                None,
                "compiled source and native operation preflight passed; numerical evaluation can still fail",
            ),
            Err(e) => PricingCapability::new(
                self.task.operation(),
                CapabilityStatus::Unsupported,
                Some(CapabilityReason::NativePreflightRequired),
                e.to_string(),
            ),
        }
    }
    fn validate_operation(&self) -> Result<(), PricingTaskError> {
        use PricingOperation as O;
        match (&self.backend, self.task.operation()) {
            (_, O::Evaluate) => Ok(()),
            (TaskBackend::SingleAsset(p), operation) => {
                p.validate_operation(operation).map_err(Into::into)
            }
            (
                TaskBackend::MultiAsset(p),
                O::Aad {
                    gamma_relative_bump,
                },
            ) => p
                .validate_aad(MultiAssetRiskConfig {
                    gamma_relative_bump,
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
        self.validate_operation()?;
        let output = match (&self.backend, self.task.operation()) {
            (TaskBackend::SingleAsset(p), operation) => p.evaluate_operation(operation)?,
            (TaskBackend::MultiAsset(p), O::Evaluate) => {
                PricingTaskOutput::MultiAsset(Box::new(p.evaluate()?))
            }
            (
                TaskBackend::MultiAsset(p),
                O::Aad {
                    gamma_relative_bump,
                },
            ) => PricingTaskOutput::MultiAsset(Box::new(p.evaluate_aad(MultiAssetRiskConfig {
                gamma_relative_bump,
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
        let report = super::source_report::build(
            &self.task,
            self.fingerprint,
            self.plan_fingerprint(),
            &output,
        )?;
        Ok(PricingTaskResult {
            report,
            native_result: output,
        })
    }
}
