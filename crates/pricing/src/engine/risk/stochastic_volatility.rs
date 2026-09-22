//! Pure SV at deterministic rates. Shared pricing, dividends and reverse
//! delegate to the hybrid engine with exactly zero rate volatility.

use crate::hull_white::{HullWhiteAadRisk, HullWhiteEquityPricingPlan, HullWhitePrice};
use crate::mc::ExecutionPolicy;
use crate::models::{
    Bergomi1Factor, Bergomi2Factor, HullWhite1Factor, HybridCorrelation, RoughBergomi,
};
use crate::{Fingerprint, MonteCarloError, PricingRequest};

/// Calibration-free 1F/2F/rough Bergomi with flat initial forward variance.
/// Requests supply sigma0 through ModelSpec::BlackScholes. All factor
/// parameters are fixed during AAD; vega differentiates sigma0.
#[derive(Clone, Debug)]
pub struct StochasticVolatilityPricingPlan {
    inner: HullWhiteEquityPricingPlan,
}

impl StochasticVolatilityPricingPlan {
    pub fn compile_bergomi(
        request: &PricingRequest,
        factor: Bergomi1Factor,
        maximum_step: f64,
        policy: ExecutionPolicy,
    ) -> Result<Self, MonteCarloError> {
        HullWhiteEquityPricingPlan::compile_bergomi(
            request,
            factor,
            deterministic_rates()?,
            HybridCorrelation::new(factor.correlation(), 0.0, 0.0)?,
            maximum_step,
            policy,
        )
        .map(|inner| Self { inner })
    }
    pub fn compile_bergomi_two_factor(
        request: &PricingRequest,
        factor: Bergomi2Factor,
        maximum_step: f64,
        policy: ExecutionPolicy,
    ) -> Result<Self, MonteCarloError> {
        HullWhiteEquityPricingPlan::compile_bergomi_two_factor(
            request,
            factor,
            deterministic_rates()?,
            0.0,
            [0.0; 2],
            maximum_step,
            policy,
        )
        .map(|inner| Self { inner })
    }
    pub fn compile_rough_bergomi(
        request: &PricingRequest,
        factor: RoughBergomi,
        maximum_step: f64,
        policy: ExecutionPolicy,
    ) -> Result<Self, MonteCarloError> {
        HullWhiteEquityPricingPlan::compile_rough_bergomi(
            request,
            factor,
            deterministic_rates()?,
            HybridCorrelation::new(factor.correlation(), 0.0, 0.0)?,
            maximum_step,
            policy,
        )
        .map(|inner| Self { inner })
    }
    pub fn evaluate(&self) -> Result<HullWhitePrice, MonteCarloError> {
        self.inner.evaluate()
    }
    pub fn evaluate_aad(&self) -> Result<HullWhiteAadRisk, MonteCarloError> {
        self.inner.evaluate_aad()
    }
    #[must_use]
    pub fn plan_fingerprint(&self) -> Fingerprint {
        self.inner.plan_fingerprint()
    }
    #[must_use]
    pub fn time_nodes(&self) -> &[f64] {
        self.inner.time_nodes()
    }
    #[must_use]
    pub fn random_factor_count(&self) -> usize {
        self.inner.random_factor_count()
    }
    #[must_use]
    pub fn risky_spot(&self) -> f64 {
        self.inner.risky_spot()
    }
    #[must_use]
    pub fn cash_dividend_model(&self) -> Option<&'static str> {
        self.inner.cash_dividend_model()
    }
}

fn deterministic_rates() -> Result<HullWhite1Factor, MonteCarloError> {
    Ok(HullWhite1Factor::new(0.0, vec![0.0], vec![0.0])?)
}
