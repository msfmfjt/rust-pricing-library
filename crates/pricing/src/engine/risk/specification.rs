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
        let request = self.carrier_request();
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
        Ok(CompiledPricingSpecification {
            fingerprint: self.fingerprint()?,
            backend,
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
    pub(crate) fn validate_operation(
        &self,
        operation: crate::PricingOperation,
    ) -> Result<(), MonteCarloError> {
        use crate::PricingOperation as O;
        match (&self.backend, operation) {
            (_, O::Evaluate) => Ok(()),
            (
                Backend::Stochastic(p),
                O::Aad {
                    gamma_relative_bump: None,
                },
            ) => p.validate_source_aad(),
            (
                Backend::Hybrid(p),
                O::Aad {
                    gamma_relative_bump: None,
                },
            ) => p.validate_source_aad(),
            (Backend::BergomiLsv(p), O::LocalVarianceRisk) => p.validate_source_local_variance(),
            (Backend::BergomiTwoFactorLsv(p), O::LocalVarianceRisk) => {
                p.validate_source_local_variance()
            }
            (Backend::RoughBergomiLsv(p), O::LocalVarianceRisk) => {
                p.validate_source_local_variance()
            }
            (Backend::RoughFamilyLsv(p), O::LocalVarianceRisk) => {
                p.validate_source_local_variance()
            }
            _ => Err(MonteCarloError::UnsupportedRiskForModel {
                model: "operation is not exposed by this source adapter",
            }),
        }
    }
    pub(crate) fn evaluate_operation(
        &self,
        operation: crate::PricingOperation,
    ) -> Result<crate::PricingTaskOutput, MonteCarloError> {
        use crate::{PricingOperation as O, PricingTaskOutput as Out};
        self.validate_operation(operation)?;
        Ok(match (&self.backend, operation) {
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
