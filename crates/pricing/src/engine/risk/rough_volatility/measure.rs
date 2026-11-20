//! The direct rough-volatility risks behind the measures facade (ADR 0043,
//! phase 3 step C). Every field is a dedicated API's result unchanged.
use super::*;
use crate::ModelParameterId;

/// The fixed-model parameter risk of one direct rough-volatility source.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum RoughVolatilityParameterRisk {
    /// Rough or lifted Heston. `parameters` is
    /// `evaluate_heston_parameter_risk`, run when one of its five
    /// coordinates is listed; `hurst` is `evaluate_hurst_risk` (rough Heston
    /// only), run when Hurst is listed.
    Heston {
        parameters: Option<Box<HestonMcParameterRisk>>,
        hurst: Option<Box<RoughHestonMcHurstRisk>>,
    },
    /// Quadratic rough Heston (`evaluate_quadratic_heston_parameter_risk`).
    QuadraticHeston(QuadraticHestonMcParameterRisk),
    /// Mixed rough Bergomi (`evaluate_mixed_bergomi_parameter_risk`, or its
    /// `_with_hurst` form).
    MixedBergomi(MixedBergomiMcParameterRisk),
}

impl RoughVolatilityParameterRisk {
    /// The price of the first run; `None` only for a Heston risk with no run.
    #[must_use]
    pub fn price(&self) -> Option<&RoughVolatilityPrice> {
        match self {
            Self::Heston { parameters, hurst } => parameters
                .as_deref()
                .map(|r| &r.price)
                .or_else(|| hurst.as_deref().map(|r| &r.price)),
            Self::QuadraticHeston(r) => Some(&r.price),
            Self::MixedBergomi(r) => Some(&r.price),
        }
    }

    /// Every coordinate the runs produced, in the native order, as
    /// `(native name, estimate, standard error, method)`.
    fn coordinates(&self) -> Vec<(&str, f64, f64, &'static str)> {
        match self {
            Self::Heston { parameters, hurst } => parameters
                .iter()
                .flat_map(|r| {
                    HESTON_MC_PARAMETER_NAMES
                        .iter()
                        .zip(r.parameter_adjoints.iter().zip(&r.standard_errors))
                        .map(|(&n, (&v, &e))| (n, v, e, r.method))
                })
                .chain(
                    hurst
                        .iter()
                        .map(|r| ("hurst", r.hurst_sensitivity, r.standard_error, r.method)),
                )
                .collect(),
            Self::QuadraticHeston(r) => named(
                &r.parameter_names,
                &r.parameter_adjoints,
                &r.standard_errors,
                r.method,
            ),
            Self::MixedBergomi(r) => named(
                &r.parameter_names,
                &r.parameter_adjoints,
                &r.standard_errors,
                r.method,
            ),
        }
    }
}

fn named<'a>(
    names: &'a [String],
    values: &[f64],
    errors: &[f64],
    method: &'static str,
) -> Vec<(&'a str, f64, f64, &'static str)> {
    names
        .iter()
        .zip(values.iter().zip(errors))
        .map(|(n, (&v, &e))| (n.as_str(), v, e, method))
        .collect()
}

/// The name the native estimators give a parameter.
pub(crate) fn native_name(parameter: ModelParameterId) -> String {
    match parameter {
        ModelParameterId::ComponentVolOfVol { component } => format!("vol_of_vol[{component}]"),
        other => other.as_str().to_owned(),
    }
}

/// A native parameter risk with the identifier of each coordinate it
/// produced, in the native order (the task's rows follow the request).
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct RoughVolatilityModelParameterRisk {
    pub parameters: Box<[ModelParameterId]>,
    pub risk: RoughVolatilityParameterRisk,
}

impl RoughVolatilityModelParameterRisk {
    /// Estimate, standard error and method label of one coordinate.
    #[must_use]
    pub fn coordinate(&self, parameter: ModelParameterId) -> Option<(f64, f64, &'static str)> {
        let index = self.parameters.iter().position(|&p| p == parameter)?;
        self.risk
            .coordinates()
            .get(index)
            .map(|&(_, v, e, m)| (v, e, m))
    }
}

/// The native measures of one direct rough-volatility source, from a
/// `Measures` task. Each risk is its dedicated API's result unchanged.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct RoughVolatilityMeasureRisk {
    /// The Delta's price, else the parameter risk's, else the plan's.
    pub price: RoughVolatilityPrice,
    /// `evaluate_delta`.
    pub delta: Option<RoughVolatilityDelta>,
    /// `evaluate_gamma_bump` with the absolute bump the request resolved to.
    pub gamma: Option<RoughVolatilityGamma>,
    pub model_parameters: Option<RoughVolatilityModelParameterRisk>,
}

impl RoughVolatilityPricingPlan {
    /// The rough model the plan simulates.
    pub(crate) fn rough_model(&self) -> &RoughVolatilityModel {
        self.path.model()
    }

    /// Whether the payoff admits the native pathwise estimators.
    pub(crate) const fn supports_spot_risk(&self) -> bool {
        self.risk_supported
    }

    /// The physical Spot the plan prices at.
    pub(crate) const fn spot(&self) -> f64 {
        self.base.spot
    }

    /// Runs the requested dedicated estimators. `gamma` carries the absolute
    /// physical Spot bump; `parameters` the listed model parameters, which
    /// select the parameter runs.
    pub(crate) fn evaluate_measure_risk(
        &self,
        delta: bool,
        gamma: Option<f64>,
        parameters: Option<&[ModelParameterId]>,
    ) -> Result<RoughVolatilityMeasureRisk, MonteCarloError> {
        let delta = delta.then(|| self.evaluate_delta()).transpose()?;
        let gamma = gamma.map(|h| self.evaluate_gamma_bump(h)).transpose()?;
        let model_parameters = parameters
            .map(|p| self.evaluate_parameter_measure(p))
            .transpose()?;
        let price = match delta
            .as_ref()
            .map(|d| &d.price)
            .or_else(|| model_parameters.as_ref().and_then(|p| p.risk.price()))
        {
            Some(price) => price.clone(),
            None => self.evaluate()?,
        };
        Ok(RoughVolatilityMeasureRisk {
            price,
            delta,
            gamma,
            model_parameters,
        })
    }

    /// Runs the model's dedicated parameter risks the listed parameters need
    /// and identifies each coordinate they produce.
    fn evaluate_parameter_measure(
        &self,
        listed: &[ModelParameterId],
    ) -> Result<RoughVolatilityModelParameterRisk, MonteCarloError> {
        let unsupported = |model| MonteCarloError::UnsupportedRiskForModel { model };
        let (available, hurst_available) =
            self.rough_model().native_parameters().ok_or_else(|| {
                unsupported("this rough-volatility model has no native parameter risk")
            })?;
        let hurst = listed.contains(&ModelParameterId::Hurst);
        if hurst && !hurst_available {
            return Err(unsupported(
                "Hurst risk is defined for power kernels only, not for a finite lift",
            ));
        }
        let mut parameters = available;
        let risk = match self.rough_model() {
            RoughVolatilityModel::RoughHeston(_) | RoughVolatilityModel::LiftedHeston(_) => {
                let scalar = listed.iter().any(|&p| p != ModelParameterId::Hurst);
                if !scalar {
                    parameters.clear();
                }
                RoughVolatilityParameterRisk::Heston {
                    parameters: scalar
                        .then(|| self.evaluate_heston_parameter_risk().map(Box::new))
                        .transpose()?,
                    hurst: hurst
                        .then(|| self.evaluate_hurst_risk().map(Box::new))
                        .transpose()?,
                }
            }
            RoughVolatilityModel::QuadraticRoughHeston(_) => {
                RoughVolatilityParameterRisk::QuadraticHeston(
                    self.evaluate_quadratic_heston_parameter_risk(hurst)?,
                )
            }
            RoughVolatilityModel::MixedRoughBergomi(_) => {
                RoughVolatilityParameterRisk::MixedBergomi(if hurst {
                    self.evaluate_mixed_bergomi_parameter_risk_with_hurst()?
                } else {
                    self.evaluate_mixed_bergomi_parameter_risk()?
                })
            }
            _ => {
                return Err(unsupported(
                    "this rough-volatility model has no native parameter risk",
                ));
            }
        };
        if hurst {
            parameters.push(ModelParameterId::Hurst);
        }
        let expected = parameters
            .iter()
            .map(|&p| native_name(p))
            .collect::<Vec<_>>();
        let names = risk
            .coordinates()
            .into_iter()
            .map(|(n, ..)| n)
            .collect::<Vec<_>>();
        if names != expected {
            return Err(unsupported(
                "native parameter names do not match the model's parameter list",
            ));
        }
        Ok(RoughVolatilityModelParameterRisk {
            parameters: parameters.into(),
            risk,
        })
    }
}
