//! Validated source inputs, independent of executable pricing workspaces.
use std::{error::Error, fmt};

use crate::core::Date;
use crate::market::MarketContext;
use crate::mc::hull_white::HullWhiteLsvTarget;
use crate::mc::lsv::LsvParticleConfig;
use crate::mc::{EngineConfig, ExecutionPolicy, LsmConfig};
use crate::models::{
    Bergomi1Factor, Bergomi2Factor, BlackScholesSpec, HullWhite1Factor, LocalVolatilitySpec,
    ModelSpec, RoughBergomi, RoughVolatilityModel,
};
use crate::product::ProductSpec;
use crate::risk::RiskRequest;
use crate::{PricingRequest, RequestValidationError};

#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum VolatilityFactorSpec {
    Bergomi(Bergomi1Factor),
    BergomiTwoFactor(Bergomi2Factor),
    RoughBergomi(RoughBergomi),
}

#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum EquityModelSpec {
    Standard(ModelSpec),
    StochasticVolatility {
        initial_volatility: f64,
        factor: VolatilityFactorSpec,
    },
    LocalStochasticVolatility {
        target: LocalVolatilitySpec,
        factor: VolatilityFactorSpec,
        particles: LsvParticleConfig,
        forward_density: Option<HullWhiteLsvTarget>,
    },
    RoughVolatility(RoughVolatilityModel),
    RoughLocalStochasticVolatility {
        target: LocalVolatilitySpec,
        model: RoughVolatilityModel,
        particles: LsvParticleConfig,
    },
}

#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum RateModelSpec {
    Deterministic,
    HullWhite {
        model: HullWhite1Factor,
        equity_rate_correlation: f64,
        /// In the volatility driver's declared order; empty for BS.
        vol_rate_correlations: Vec<f64>,
    },
}

#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct PricingModelSpec {
    pub equity: EquityModelSpec,
    pub rates: RateModelSpec,
}
impl PricingModelSpec {
    #[must_use]
    pub fn new(equity: EquityModelSpec, rates: RateModelSpec) -> Self {
        Self { equity, rates }
    }
}
impl From<ModelSpec> for PricingModelSpec {
    fn from(model: ModelSpec) -> Self {
        Self::new(
            EquityModelSpec::Standard(model),
            RateModelSpec::Deterministic,
        )
    }
}

#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct PricingNumerics {
    pub execution_policy: ExecutionPolicy,
    pub maximum_step: Option<f64>,
    pub lsm: Option<LsmConfig>,
}
impl PricingNumerics {
    #[must_use]
    pub fn new(execution_policy: ExecutionPolicy) -> Self {
        Self {
            execution_policy,
            maximum_step: None,
            lsm: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum SpecificationValidationError {
    Request(RequestValidationError),
    UnsupportedOperation {
        reason: Option<crate::CapabilityReason>,
        detail: String,
    },
    Invalid {
        field: &'static str,
        message: String,
    },
}
impl fmt::Display for SpecificationValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Request(error) => error.fmt(f),
            Self::UnsupportedOperation { detail, .. } => write!(f, "operation: {detail}"),
            Self::Invalid { field, message } => write!(f, "{field}: {message}"),
        }
    }
}
impl Error for SpecificationValidationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Request(e) => Some(e),
            Self::Invalid { .. } | Self::UnsupportedOperation { .. } => None,
        }
    }
}
fn invalid(field: &'static str, message: impl ToString) -> SpecificationValidationError {
    SpecificationValidationError::Invalid {
        field,
        message: message.to_string(),
    }
}

/// Complete, immutable source specification. A private legacy carrier lets the
/// adapters reuse contractual validation without exposing a placeholder model.
#[derive(Clone, Debug, PartialEq)]
pub struct PricingSpecification {
    request: PricingRequest,
    model: PricingModelSpec,
    numerics: PricingNumerics,
}
impl PricingSpecification {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        valuation_date: Date,
        product: ProductSpec,
        market: MarketContext,
        model: PricingModelSpec,
        engine: EngineConfig,
        risk: RiskRequest,
        numerics: PricingNumerics,
    ) -> Result<Self, SpecificationValidationError> {
        validate(&model, &numerics, &risk)?;
        let carrier = match &model.equity {
            EquityModelSpec::Standard(m) => m.clone(),
            EquityModelSpec::StochasticVolatility {
                initial_volatility, ..
            } => ModelSpec::BlackScholes(
                BlackScholesSpec::new(*initial_volatility)
                    .map_err(|e| invalid("model.equity.initial_volatility", e))?,
            ),
            EquityModelSpec::RoughVolatility(_) => {
                ModelSpec::BlackScholes(BlackScholesSpec::new(0.0).expect("finite zero"))
            }
            EquityModelSpec::LocalStochasticVolatility { target, .. }
            | EquityModelSpec::RoughLocalStochasticVolatility { target, .. } => {
                ModelSpec::LocalVolatility(target.clone())
            }
        };
        let request = PricingRequest::new_with_lsm(
            valuation_date,
            product,
            market,
            carrier,
            engine,
            risk,
            numerics.lsm.clone(),
        )
        .map_err(SpecificationValidationError::Request)?;
        Ok(Self {
            request,
            model,
            numerics,
        })
    }

    #[must_use]
    pub fn from_request(request: PricingRequest, execution_policy: ExecutionPolicy) -> Self {
        let mut numerics = PricingNumerics::new(execution_policy);
        numerics.lsm = request.lsm().cloned();
        let model = request.model().clone().into();
        Self {
            request,
            model,
            numerics,
        }
    }

    pub fn with_model(
        self,
        model: PricingModelSpec,
        maximum_step: Option<f64>,
    ) -> Result<Self, SpecificationValidationError> {
        let mut numerics = self.numerics;
        numerics.maximum_step = maximum_step;
        let mut updated = Self::new(
            self.request.valuation_date(),
            self.request.product().clone(),
            self.request.market().clone(),
            model,
            self.request.engine(),
            self.request.risk().clone(),
            numerics,
        )?;
        // Migration provenance belongs to the original legacy input only.
        // The new document stores normalized sources, not its historical bytes.
        if updated.model == self.model {
            updated.request = self.request;
        }
        Ok(updated)
    }

    #[must_use]
    pub fn valuation_date(&self) -> Date {
        self.request.valuation_date()
    }
    #[must_use]
    pub fn product(&self) -> &ProductSpec {
        self.request.product()
    }
    #[must_use]
    pub fn market(&self) -> &MarketContext {
        self.request.market()
    }
    #[must_use]
    pub fn model(&self) -> &PricingModelSpec {
        &self.model
    }
    #[must_use]
    pub fn engine(&self) -> EngineConfig {
        self.request.engine()
    }
    #[must_use]
    pub fn risk(&self) -> &RiskRequest {
        self.request.risk()
    }
    #[must_use]
    pub fn numerics(&self) -> &PricingNumerics {
        &self.numerics
    }
    pub(crate) fn carrier_request(&self) -> &PricingRequest {
        &self.request
    }
}

fn validate(
    model: &PricingModelSpec,
    numerics: &PricingNumerics,
    risk: &RiskRequest,
) -> Result<(), SpecificationValidationError> {
    let hybrid = matches!(model.rates, RateModelSpec::HullWhite { .. });
    let extension = hybrid || !matches!(model.equity, EquityModelSpec::Standard(_));
    if extension {
        if risk.delta()
            || risk.gamma().is_some()
            || risk.vega()
            || risk.vega_kt().is_some()
            || risk.payoff_smoothing_width_ladder().is_some()
            || risk.checkpoint_interval().is_some()
            || risk.aad_tile_capacity().is_some()
        {
            return Err(invalid(
                "risk",
                "extension specifications support price-only requests; use the dedicated risk APIs",
            ));
        }
        if numerics.lsm.is_some() {
            return Err(invalid(
                "lsm",
                "extension specifications do not support early exercise",
            ));
        }
    }
    let requires_step = matches!(
        model.equity,
        EquityModelSpec::StochasticVolatility { .. } | EquityModelSpec::RoughVolatility(_)
    ) || (hybrid && matches!(model.equity, EquityModelSpec::Standard(_)));
    match (requires_step, numerics.maximum_step) {
        (true, Some(step)) if step.is_finite() && step > 0.0 => {}
        (true, _) => {
            return Err(invalid(
                "numerics.maximum_step",
                "a finite positive maximum step is required",
            ));
        }
        (false, Some(_)) => {
            return Err(invalid(
                "numerics.maximum_step",
                "this model uses its native or calibration grid",
            ));
        }
        (false, None) => {}
    }
    if let EquityModelSpec::LocalStochasticVolatility {
        target,
        forward_density,
        ..
    } = &model.equity
    {
        match (hybrid, forward_density) {
            (true, Some(density)) if density.grid() == target.local_variance_grid() => {}
            (true, Some(_)) => {
                return Err(invalid(
                    "model.equity.forward_density",
                    "density and local variance must use the same target grid",
                ));
            }
            (true, None) => {
                return Err(invalid(
                    "model.equity.forward_density",
                    "Hull-White LSV requires a matching forward-density target",
                ));
            }
            (false, Some(_)) => {
                return Err(invalid(
                    "model.equity.forward_density",
                    "forward density is only used by Hull-White LSV",
                ));
            }
            (false, None) => {}
        }
    }
    if let RateModelSpec::HullWhite {
        equity_rate_correlation,
        vol_rate_correlations,
        ..
    } = &model.rates
    {
        let count = match &model.equity {
            EquityModelSpec::Standard(ModelSpec::BlackScholes(_)) => 0,
            EquityModelSpec::StochasticVolatility { factor, .. }
            | EquityModelSpec::LocalStochasticVolatility { factor, .. } => {
                if matches!(factor, VolatilityFactorSpec::BergomiTwoFactor(_)) {
                    2
                } else {
                    1
                }
            }
            _ => {
                return Err(invalid(
                    "model.rates",
                    "Hull-White composition supports BS and Bergomi SV/LSV only",
                ));
            }
        };
        if vol_rate_correlations.len() != count {
            return Err(invalid(
                "model.rates.vol_rate_correlations",
                format!("expected {count} ordered driver correlations"),
            ));
        }
        if !equity_rate_correlation.is_finite() || equity_rate_correlation.abs() > 1.0 {
            return Err(invalid(
                "model.rates.equity_rate_correlation",
                "correlation must be finite and in [-1, 1]",
            ));
        }
        if vol_rate_correlations
            .iter()
            .any(|v| !v.is_finite() || v.abs() > 1.0)
        {
            return Err(invalid(
                "model.rates.vol_rate_correlations",
                "correlations must be finite and in [-1, 1]",
            ));
        }
    }
    Ok(())
}
