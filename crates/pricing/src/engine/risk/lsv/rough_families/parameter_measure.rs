//! The rough-family LSV parameter risks behind the `model_parameters` measure.
//! Each variant is a dedicated API's result unchanged.
use super::*;
use crate::ModelParameterId;
use crate::engine::risk::rough_volatility::native_name;

/// The native parameter risk of one rough-family LSV source: total
/// derivatives at the fixed Local Variance target, with the particle
/// calibration differentiated (recalibrated).
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum RoughFamilyLsvParameterRisk {
    /// Rough or lifted Heston (`evaluate_heston_parameter_risk`).
    Heston(HestonLsvParameterRisk),
    /// Quadratic rough Heston (`evaluate_quadratic_heston_parameter_risk`).
    QuadraticHeston(QuadraticHestonLsvParameterRisk),
    /// Mixed rough Bergomi (`evaluate_mixed_bergomi_parameter_risk`, or its
    /// `_with_hurst` form).
    MixedBergomi(MixedBergomiLsvParameterRisk),
}

impl RoughFamilyLsvParameterRisk {
    #[must_use]
    pub fn price(&self) -> &LsvPrice {
        match self {
            Self::Heston(r) => &r.price,
            Self::QuadraticHeston(r) => &r.price,
            Self::MixedBergomi(r) => &r.price,
        }
    }
    /// Total derivatives in the native order.
    #[must_use]
    pub fn parameter_adjoints(&self) -> &[f64] {
        match self {
            Self::Heston(r) => &r.parameter_adjoints,
            Self::QuadraticHeston(r) => &r.parameter_adjoints,
            Self::MixedBergomi(r) => &r.parameter_adjoints,
        }
    }
    /// RQMC errors of the total derivatives; `None` for pseudo-random runs.
    #[must_use]
    pub fn standard_errors(&self) -> Option<&[f64]> {
        match self {
            Self::Heston(r) => r.standard_errors.as_deref(),
            Self::QuadraticHeston(r) => r.standard_errors.as_deref(),
            Self::MixedBergomi(r) => r.standard_errors.as_deref(),
        }
    }
    #[must_use]
    pub fn method(&self) -> &'static str {
        match self {
            Self::Heston(r) => r.method,
            Self::QuadraticHeston(r) => r.method,
            Self::MixedBergomi(r) => r.method,
        }
    }
    fn names(&self) -> Vec<&str> {
        match self {
            Self::Heston(r) => r.parameter_names.to_vec(),
            Self::QuadraticHeston(r) => r.parameter_names.iter().map(String::as_str).collect(),
            Self::MixedBergomi(r) => r.parameter_names.iter().map(String::as_str).collect(),
        }
    }
}

/// A native parameter risk with the identifier of each of its coordinates,
/// in the native order (the task's rows follow the request's order).
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct RoughFamilyLsvModelParameterRisk {
    pub parameters: Box<[ModelParameterId]>,
    pub risk: RoughFamilyLsvParameterRisk,
}

impl RoughFamilyLsvPricingPlan {
    /// The parameters `evaluate_parameter_measure` can produce, and whether
    /// Hurst can be added.
    pub(crate) fn native_parameters(&self) -> Option<(Vec<ModelParameterId>, bool)> {
        self.rough_model().native_parameters()
    }

    /// The rough model the plan simulates.
    pub(crate) fn rough_model(&self) -> &RoughVolatilityModel {
        self.core.path_plan.model()
    }

    /// The checks the dedicated parameter risks make before running: a
    /// retained reverse trace and a differentiable payoff.
    pub(crate) fn parameter_preflight(&self) -> Result<(), MonteCarloError> {
        self.core.calibration.validate_calibration_reverse()?;
        if !self.core.risk_supported {
            return Err(MonteCarloError::UnsupportedRiskForModel {
                model: "rough-family LSV parameter risk requires pathwise payoff or explicit smoothing",
            });
        }
        Ok(())
    }

    /// Runs the model's dedicated parameter risk, with Hurst exactly when
    /// `hurst` is set, and identifies each coordinate.
    pub(crate) fn evaluate_parameter_measure(
        &self,
        hurst: bool,
    ) -> Result<RoughFamilyLsvModelParameterRisk, MonteCarloError> {
        let unsupported = || MonteCarloError::UnsupportedRiskForModel {
            model: "this rough-family model has no native parameter risk",
        };
        let (mut parameters, hurst_available) = self.native_parameters().ok_or_else(unsupported)?;
        if hurst && !hurst_available {
            return Err(MonteCarloError::UnsupportedRiskForModel {
                model: "Hurst risk is defined for power kernels only, not for a finite lift",
            });
        }
        let risk = match self.core.path_plan.model() {
            RoughVolatilityModel::RoughHeston(_) | RoughVolatilityModel::LiftedHeston(_) => {
                RoughFamilyLsvParameterRisk::Heston(self.evaluate_heston_parameter_risk(hurst)?)
            }
            RoughVolatilityModel::QuadraticRoughHeston(_) => {
                RoughFamilyLsvParameterRisk::QuadraticHeston(
                    self.evaluate_quadratic_heston_parameter_risk(hurst)?,
                )
            }
            RoughVolatilityModel::MixedRoughBergomi(_) => {
                RoughFamilyLsvParameterRisk::MixedBergomi(if hurst {
                    self.evaluate_mixed_bergomi_parameter_risk_with_hurst()?
                } else {
                    self.evaluate_mixed_bergomi_parameter_risk()?
                })
            }
            _ => return Err(unsupported()),
        };
        if hurst {
            parameters.push(ModelParameterId::Hurst);
        }
        let expected = parameters
            .iter()
            .map(|&p| native_name(p))
            .collect::<Vec<_>>();
        if risk.names() != expected {
            return Err(MonteCarloError::UnsupportedRiskForModel {
                model: "native parameter names do not match the model's parameter list",
            });
        }
        Ok(RoughFamilyLsvModelParameterRisk {
            parameters: parameters.into(),
            risk,
        })
    }
}
