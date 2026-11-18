//! Compile-time dispatch to existing numerical implementations.
use crate::hull_white::{HullWhiteEquityPricingPlan, HullWhitePrice};
use crate::lsv::{
    BergomiLsvPricingPlan, LsvPrice, RoughBergomiLsvPricingPlan, RoughFamilyLsvPricingPlan,
};
use crate::models::{Bergomi2Factor, HybridCorrelation};
use crate::rough_volatility::RoughVolatilityPricingPlan;
use crate::stochastic_volatility::StochasticVolatilityPricingPlan;
use crate::{
    EquityModelSpec, Fingerprint, MonteCarloError, MonteCarloPrice, PricingPlan,
    PricingSpecification, RateModelSpec, VolatilityFactorSpec, WireError,
};

#[derive(Clone, Debug)]
enum Backend {
    Standard(Box<PricingPlan>),
    Stochastic(Box<StochasticVolatilityPricingPlan>),
    Hybrid(Box<HullWhiteEquityPricingPlan>),
    BergomiLsv(Box<BergomiLsvPricingPlan>),
    BergomiTwoFactorLsv(Box<BergomiLsvPricingPlan<Bergomi2Factor>>),
    RoughBergomiLsv(Box<RoughBergomiLsvPricingPlan>),
    RoughFamilyLsv(Box<RoughFamilyLsvPricingPlan>),
    Rough(Box<RoughVolatilityPricingPlan>),
}

#[derive(Clone, Debug)]
pub struct CompiledPricingSpecification {
    fingerprint: Fingerprint,
    backend: Backend,
    /// The risk selection compiled into a standard plan. An operation is valid
    /// on that plan only when it asks for exactly this selection.
    standard_risk: Option<crate::risk::RiskRequest>,
    /// A rough-family LSV target's reproducing market-IV basis, which the
    /// native parallel market-IV Vega differentiates through.
    market_iv: Option<Box<crate::market::MarketIvSurface>>,
}

/// Preserve the backend's native diagnostics and risk conventions.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum SpecificationOutput {
    Standard(Box<MonteCarloPrice>),
    Hybrid(HullWhitePrice),
    LocalStochasticVolatility(LsvPrice),
    RoughVolatility(HullWhitePrice),
}
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct SpecificationResult {
    pub specification_fingerprint: Fingerprint,
    pub output: SpecificationOutput,
}
impl SpecificationResult {
    #[must_use]
    pub fn value(&self) -> f64 {
        match &self.output {
            SpecificationOutput::Standard(p) => p.pricing_result.value.value().get(),
            SpecificationOutput::Hybrid(p) | SpecificationOutput::RoughVolatility(p) => p.value,
            SpecificationOutput::LocalStochasticVolatility(p) => p.value,
        }
    }
    #[must_use]
    pub fn standard_error(&self) -> f64 {
        match &self.output {
            SpecificationOutput::Standard(p) => p.pricing_result.value.standard_error().get(),
            SpecificationOutput::Hybrid(p) | SpecificationOutput::RoughVolatility(p) => {
                p.standard_error
            }
            SpecificationOutput::LocalStochasticVolatility(p) => p.standard_error,
        }
    }
}

impl PricingSpecification {
    pub fn fingerprint(&self) -> Result<Fingerprint, WireError> {
        crate::fingerprint_specification(self)
    }

    pub fn compile(&self) -> Result<CompiledPricingSpecification, MonteCarloError> {
        self.compile_request(self.carrier_request())
    }

    /// Compile for a `Measures` operation. A standard source compiles the
    /// plan a request with the same Greeks selected would compile; every
    /// other source prices only, so its ordinary plan is used.
    pub(crate) fn compile_measures(
        &self,
        measures: &crate::MeasureSet,
    ) -> Result<CompiledPricingSpecification, crate::PricingTaskError> {
        let standard = matches!(self.model().rates, RateModelSpec::Deterministic)
            && matches!(self.model().equity, EquityModelSpec::Standard(_));
        if !standard {
            return Ok(self.compile()?);
        }
        let request = self
            .measured_request(measures.requests())
            .map_err(crate::PricingTaskError::Validation)?;
        Ok(self.compile_request(&request)?)
    }

    fn compile_request(
        &self,
        request: &crate::PricingRequest,
    ) -> Result<CompiledPricingSpecification, MonteCarloError> {
        let policy = self.numerics().execution_policy;
        let step = || {
            self.numerics()
                .maximum_step
                .expect("validated maximum_step")
        };
        let backend = match &self.model().rates {
            RateModelSpec::HullWhite { .. } => Backend::Hybrid(Box::new(compile_hybrid(self)?)),
            RateModelSpec::Deterministic => match &self.model().equity {
                EquityModelSpec::Standard(_) => {
                    Backend::Standard(Box::new(PricingPlan::compile(request, policy)?))
                }
                EquityModelSpec::StochasticVolatility { factor, .. } => {
                    let plan = match factor {
                        VolatilityFactorSpec::Bergomi(f) => {
                            StochasticVolatilityPricingPlan::compile_bergomi(
                                request,
                                *f,
                                step(),
                                policy,
                            )?
                        }
                        VolatilityFactorSpec::BergomiTwoFactor(f) => {
                            StochasticVolatilityPricingPlan::compile_bergomi_two_factor(
                                request,
                                *f,
                                step(),
                                policy,
                            )?
                        }
                        VolatilityFactorSpec::RoughBergomi(f) => {
                            StochasticVolatilityPricingPlan::compile_rough_bergomi(
                                request,
                                *f,
                                step(),
                                policy,
                            )?
                        }
                    };
                    Backend::Stochastic(Box::new(plan))
                }
                EquityModelSpec::LocalStochasticVolatility {
                    factor, particles, ..
                } => match factor {
                    VolatilityFactorSpec::Bergomi(f) => Backend::BergomiLsv(Box::new(
                        BergomiLsvPricingPlan::compile(request, *f, particles.clone(), policy)?,
                    )),
                    VolatilityFactorSpec::BergomiTwoFactor(f) => {
                        Backend::BergomiTwoFactorLsv(Box::new(BergomiLsvPricingPlan::compile(
                            request,
                            *f,
                            particles.clone(),
                            policy,
                        )?))
                    }
                    VolatilityFactorSpec::RoughBergomi(f) => {
                        Backend::RoughBergomiLsv(Box::new(RoughBergomiLsvPricingPlan::compile(
                            request,
                            *f,
                            particles.clone(),
                            policy,
                        )?))
                    }
                },
                EquityModelSpec::RoughVolatility(model) => Backend::Rough(Box::new(
                    RoughVolatilityPricingPlan::compile(request, model.clone(), step(), policy)?,
                )),
                EquityModelSpec::RoughLocalStochasticVolatility {
                    model, particles, ..
                } => Backend::RoughFamilyLsv(Box::new(RoughFamilyLsvPricingPlan::compile(
                    request,
                    model.clone(),
                    particles.clone(),
                    policy,
                )?)),
            },
        };
        let standard_risk = matches!(backend, Backend::Standard(_)).then(|| request.risk().clone());
        let market_iv = match (&backend, &self.model().equity) {
            (
                Backend::RoughFamilyLsv(_),
                EquityModelSpec::RoughLocalStochasticVolatility { target, .. },
            ) => target.market_iv_source().ok().map(Box::new),
            _ => None,
        };
        Ok(CompiledPricingSpecification {
            fingerprint: self.fingerprint()?,
            backend,
            standard_risk,
            market_iv,
        })
    }
}

fn compile_hybrid(
    source: &PricingSpecification,
) -> Result<HullWhiteEquityPricingPlan, MonteCarloError> {
    let RateModelSpec::HullWhite {
        model: rates,
        equity_rate_correlation,
        vol_rate_correlations,
    } = &source.model().rates
    else {
        unreachable!("hybrid adapter requires Hull-White rates")
    };
    let request = source.carrier_request();
    let policy = source.numerics().execution_policy;
    let step = || {
        source
            .numerics()
            .maximum_step
            .expect("validated maximum_step")
    };
    let correlation =
        |spot| HybridCorrelation::new(spot, *equity_rate_correlation, vol_rate_correlations[0]);
    match &source.model().equity {
        EquityModelSpec::Standard(_) => HullWhiteEquityPricingPlan::compile_bs(
            request,
            rates.clone(),
            *equity_rate_correlation,
            step(),
            policy,
        ),
        EquityModelSpec::StochasticVolatility { factor, .. } => match factor {
            VolatilityFactorSpec::Bergomi(f) => HullWhiteEquityPricingPlan::compile_bergomi(
                request,
                *f,
                rates.clone(),
                correlation(f.correlation())?,
                step(),
                policy,
            ),
            VolatilityFactorSpec::BergomiTwoFactor(f) => {
                HullWhiteEquityPricingPlan::compile_bergomi_two_factor(
                    request,
                    *f,
                    rates.clone(),
                    *equity_rate_correlation,
                    [vol_rate_correlations[0], vol_rate_correlations[1]],
                    step(),
                    policy,
                )
            }
            VolatilityFactorSpec::RoughBergomi(f) => {
                HullWhiteEquityPricingPlan::compile_rough_bergomi(
                    request,
                    *f,
                    rates.clone(),
                    correlation(f.correlation())?,
                    step(),
                    policy,
                )
            }
        },
        EquityModelSpec::LocalStochasticVolatility {
            factor,
            particles,
            forward_density,
            ..
        } => {
            let target = forward_density
                .as_ref()
                .expect("validated Hull-White LSV target");
            match factor {
                VolatilityFactorSpec::Bergomi(f) => HullWhiteEquityPricingPlan::compile_lsv(
                    request,
                    target,
                    *f,
                    rates.clone(),
                    correlation(f.correlation())?,
                    particles.clone(),
                    policy,
                ),
                VolatilityFactorSpec::BergomiTwoFactor(f) => {
                    HullWhiteEquityPricingPlan::compile_lsv_two_factor(
                        request,
                        target,
                        *f,
                        rates.clone(),
                        *equity_rate_correlation,
                        [vol_rate_correlations[0], vol_rate_correlations[1]],
                        particles.clone(),
                        policy,
                    )
                }
                VolatilityFactorSpec::RoughBergomi(f) => {
                    HullWhiteEquityPricingPlan::compile_rough_lsv(
                        request,
                        target,
                        *f,
                        rates.clone(),
                        correlation(f.correlation())?,
                        particles.clone(),
                        policy,
                    )
                }
            }
        }
        _ => unreachable!("unsupported Hull-White composition rejected at source construction"),
    }
}

impl CompiledPricingSpecification {
    pub(crate) fn calibration_diagnostics(
        &self,
        source: &PricingSpecification,
    ) -> Vec<crate::ReportedCalibration> {
        use super::calibration_report as report;
        let id = source.product().underlying().get();
        let seed = match &source.model().equity {
            EquityModelSpec::LocalStochasticVolatility { particles, .. }
            | EquityModelSpec::RoughLocalStochasticVolatility { particles, .. } => {
                Some(particles.seed())
            }
            _ => None,
        };
        match &self.backend {
            Backend::Standard(p) => {
                let mut blocks = Vec::new();
                if let Some(rows) = p.bass_calibration_diagnostics().filter(|r| !r.is_empty()) {
                    blocks.push(report::bass(id, rows));
                }
                if let Some(rows) = p.bass_projection_diagnostics().filter(|r| !r.is_empty()) {
                    blocks.push(report::bass_projection(id, rows));
                }
                blocks
            }
            Backend::Hybrid(p) => p
                .calibration()
                .map(|c| report::hull_white(id, seed.expect("LSV seed"), &c.diagnostics))
                .into_iter()
                .collect(),
            Backend::BergomiLsv(p) => vec![report::lsv(
                id,
                p.calibration().config().seed(),
                p.calibration().diagnostics(),
            )],
            Backend::BergomiTwoFactorLsv(p) => vec![report::lsv(
                id,
                p.calibration().config().seed(),
                p.calibration().diagnostics(),
            )],
            Backend::RoughBergomiLsv(p) => vec![report::lsv(
                id,
                p.calibration().config().seed(),
                p.calibration().diagnostics(),
            )],
            Backend::RoughFamilyLsv(p) => vec![report::lsv(
                id,
                p.calibration().config().seed(),
                p.calibration().diagnostics(),
            )],
            Backend::Stochastic(_) | Backend::Rough(_) => Vec::new(),
        }
    }
    /// `source` is the specification this plan was compiled from.
    pub(crate) fn validate_operation(
        &self,
        operation: &crate::PricingOperation,
        source: &PricingSpecification,
    ) -> Result<(), crate::PricingTaskError> {
        use crate::PricingOperation as O;
        use crate::engine::risk::measure_backend::MeasureBackend;
        match (&self.backend, operation) {
            (_, O::Measures(set)) => self.validate_measures(&set.native_requests()),
            (Backend::Standard(_), O::Evaluate) => {
                if self.standard_risk.as_ref() == Some(source.risk()) {
                    Ok(())
                } else {
                    Err(reselected())
                }
            }
            (_, O::Evaluate) => Ok(()),
            (
                Backend::Stochastic(p),
                O::Aad {
                    gamma_relative_bump: None,
                },
            ) => p.validate_source_aad().map_err(Into::into),
            (
                Backend::Hybrid(p),
                O::Aad {
                    gamma_relative_bump: None,
                },
            ) => p.validate_source_aad().map_err(Into::into),
            (Backend::BergomiLsv(p), O::LocalVarianceRisk) => {
                p.validate_source_local_variance().map_err(Into::into)
            }
            (Backend::BergomiTwoFactorLsv(p), O::LocalVarianceRisk) => {
                p.validate_source_local_variance().map_err(Into::into)
            }
            (Backend::RoughBergomiLsv(p), O::LocalVarianceRisk) => {
                p.validate_source_local_variance().map_err(Into::into)
            }
            (Backend::RoughFamilyLsv(p), O::LocalVarianceRisk) => {
                p.validate_source_local_variance().map_err(Into::into)
            }
            _ => Err(MonteCarloError::UnsupportedRiskForModel {
                model: "operation is not exposed by this source adapter",
            }
            .into()),
        }
    }
    pub(crate) fn evaluate_operation(
        &self,
        operation: &crate::PricingOperation,
        source: &PricingSpecification,
    ) -> Result<crate::PricingTaskOutput, crate::PricingTaskError> {
        use crate::engine::risk::measure_backend::MeasureBackend;
        use crate::{PricingOperation as O, PricingTaskOutput as Out};
        self.validate_operation(operation, source)?;
        Ok(match (&self.backend, operation) {
            (_, O::Measures(set)) => self.evaluate_measures(&set.native_requests())?,
            (_, O::Evaluate) => Out::SingleAsset(self.evaluate()?),
            (Backend::Stochastic(p), O::Aad { .. }) => {
                Out::HullWhiteRisk(Box::new(p.evaluate_aad()?))
            }
            (Backend::Hybrid(p), O::Aad { .. }) => Out::HullWhiteRisk(Box::new(p.evaluate_aad()?)),
            (Backend::BergomiLsv(p), O::LocalVarianceRisk) => {
                Out::LocalVarianceRisk(Box::new(p.evaluate_local_variance_risk()?))
            }
            (Backend::BergomiTwoFactorLsv(p), O::LocalVarianceRisk) => {
                Out::LocalVarianceRisk(Box::new(p.evaluate_local_variance_risk()?))
            }
            (Backend::RoughBergomiLsv(p), O::LocalVarianceRisk) => {
                Out::LocalVarianceRisk(Box::new(p.evaluate_local_variance_risk()?))
            }
            (Backend::RoughFamilyLsv(p), O::LocalVarianceRisk) => {
                Out::LocalVarianceRisk(Box::new(p.evaluate_local_variance_risk()?))
            }
            _ => unreachable!("validated operation"),
        })
    }
    /// The risk selection of a standard plan, `None` for every other backend.
    pub(crate) fn standard_risk(&self) -> Option<&crate::risk::RiskRequest> {
        self.standard_risk.as_ref()
    }
    /// The reproducing market-IV basis of a rough-family LSV source.
    pub(crate) fn market_iv(&self) -> Option<&crate::market::MarketIvSurface> {
        self.market_iv.as_deref()
    }
    /// The plan of a rough-family LSV source.
    pub(crate) fn rough_family_lsv(&self) -> Option<&RoughFamilyLsvPricingPlan> {
        match &self.backend {
            Backend::RoughFamilyLsv(p) => Some(p),
            _ => None,
        }
    }
    /// The plan of a direct rough-volatility source.
    pub(crate) fn rough_volatility(&self) -> Option<&RoughVolatilityPricingPlan> {
        match &self.backend {
            Backend::Rough(p) => Some(p),
            _ => None,
        }
    }
    /// A plan for `side` that reuses this plan's calibration. Implemented for
    /// single-asset LSV; see `PricingSource::supports_fixed_calibration`.
    pub(crate) fn with_fixed_calibration(
        &self,
        side: &PricingSpecification,
    ) -> Result<Self, MonteCarloError> {
        let request = side.carrier_request();
        let backend = match &self.backend {
            Backend::BergomiLsv(p) => {
                Backend::BergomiLsv(Box::new(p.with_fixed_calibration(request)?))
            }
            Backend::BergomiTwoFactorLsv(p) => {
                Backend::BergomiTwoFactorLsv(Box::new(p.with_fixed_calibration(request)?))
            }
            Backend::RoughBergomiLsv(p) => {
                Backend::RoughBergomiLsv(Box::new(p.with_fixed_calibration(request)?))
            }
            Backend::RoughFamilyLsv(p) => {
                Backend::RoughFamilyLsv(Box::new(p.with_fixed_calibration(request)?))
            }
            Backend::Hybrid(p) => {
                let EquityModelSpec::LocalStochasticVolatility {
                    particles,
                    forward_density: Some(target),
                    ..
                } = &side.model().equity
                else {
                    return Err(MonteCarloError::UnsupportedRiskForModel {
                        model: "a fixed calibration requires a Hull-White LSV source",
                    });
                };
                Backend::Hybrid(Box::new(
                    p.with_fixed_calibration(request, target, particles)?,
                ))
            }
            _ => {
                return Err(MonteCarloError::UnsupportedRiskForModel {
                    model: "a fixed calibration is available only for single-asset LSV",
                });
            }
        };
        Ok(Self {
            fingerprint: side.fingerprint()?,
            backend,
            standard_risk: None,
            market_iv: None,
        })
    }
    #[must_use]
    pub fn specification_fingerprint(&self) -> Fingerprint {
        self.fingerprint
    }
    #[must_use]
    pub fn plan_fingerprint(&self) -> Fingerprint {
        match &self.backend {
            Backend::Standard(p) => p.plan_fingerprint(),
            Backend::Stochastic(p) => p.plan_fingerprint(),
            Backend::Hybrid(p) => p.plan_fingerprint(),
            Backend::BergomiLsv(p) => p.plan_fingerprint(),
            Backend::BergomiTwoFactorLsv(p) => p.plan_fingerprint(),
            Backend::RoughBergomiLsv(p) => p.plan_fingerprint(),
            Backend::RoughFamilyLsv(p) => p.plan_fingerprint(),
            Backend::Rough(p) => p.plan_fingerprint(),
        }
    }
    pub fn evaluate(&self) -> Result<SpecificationResult, MonteCarloError> {
        let output = match &self.backend {
            Backend::Standard(p) => SpecificationOutput::Standard(Box::new(p.evaluate()?)),
            Backend::Stochastic(p) => SpecificationOutput::Hybrid(p.evaluate()?),
            Backend::Hybrid(p) => SpecificationOutput::Hybrid(p.evaluate()?),
            Backend::BergomiLsv(p) => SpecificationOutput::LocalStochasticVolatility(p.evaluate()?),
            Backend::BergomiTwoFactorLsv(p) => {
                SpecificationOutput::LocalStochasticVolatility(p.evaluate()?)
            }
            Backend::RoughBergomiLsv(p) => {
                SpecificationOutput::LocalStochasticVolatility(p.evaluate()?)
            }
            Backend::RoughFamilyLsv(p) => {
                SpecificationOutput::LocalStochasticVolatility(p.evaluate()?)
            }
            Backend::Rough(p) => SpecificationOutput::RoughVolatility(p.evaluate()?),
        };
        Ok(SpecificationResult {
            specification_fingerprint: self.fingerprint,
            output,
        })
    }
}

/// A standard plan fixes its Greeks at compilation, so another selection
/// cannot be evaluated on it.
pub(crate) fn reselected() -> crate::PricingTaskError {
    crate::PricingTaskError::Validation(crate::SpecificationValidationError::Invalid {
        field: "operation",
        message: "this standard plan was compiled for a different risk selection; compile a task for the requested operation".into(),
    })
}
