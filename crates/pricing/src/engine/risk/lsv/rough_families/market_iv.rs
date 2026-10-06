//! Fixed-model market-IV quote risk through the entire discrete LSV calibration.
use super::*;
use crate::market::{MARKET_IV_INTERPOLATION, MarketIvSurface};
use pricing_numerics::NeumaierSum;

/// Immutable binding of a pricing plan to its exact market-IV Dupire source.
/// No calibration or valuation is repeated by binding the source.
#[derive(Clone, Debug)]
pub struct RoughFamilyLsvMarketIvRiskPlan {
    pricing: RoughFamilyLsvPricingPlan,
    surface: MarketIvSurface,
    fingerprint: Fingerprint,
}

/// Natural derivatives per one absolute annualized Black IV unit. Quote order
/// is maturity-major. RQMC errors are computed AFTER the full transpose on each
/// independent scramble, including the covariance for the parallel direction.
/// All errors condition on one calibration. Pseudo-MC errors are unavailable.
#[derive(Clone, Debug, PartialEq)]
pub struct RoughFamilyLsvMarketIvRisk {
    pub price: LsvPrice,
    pub maturity_nodes: Box<[f64]>,
    pub log_moneyness_nodes: Box<[f64]>,
    pub implied_volatilities: Box<[f64]>,
    pub quote_adjoints: Box<[f64]>,
    pub standard_errors: Option<Box<[f64]>>,
    pub parallel_vega: f64,
    pub parallel_standard_error: Option<f64>,
    pub risk_fingerprint: Fingerprint,
    pub method: &'static str,
}
impl RoughFamilyLsvPricingPlan {
    /// Bind only a source that reproduces the original target exactly and
    /// requires no Dupire floor/cap repair. Model parameters, quotes' relative
    /// coordinates, Spot, curves, cash dividends and calibration choices are
    /// fixed. NOT a simultaneous stochastic-model parameter recalibration.
    pub fn market_iv_risk_plan(
        &self,
        surface: MarketIvSurface,
    ) -> Result<RoughFamilyLsvMarketIvRiskPlan, MonteCarloError> {
        if !self.core.risk_supported {
            return Err(MonteCarloError::UnsupportedRiskForModel {
                model: "rough-family market IV requires pathwise payoff or explicit smoothing",
            });
        }
        self.core.calibration.validate_calibration_reverse()?;
        let quotes = surface.implied_volatilities().len();
        let work = self
            .core
            .original_target
            .values()
            .len()
            .checked_mul(surface.log_moneyness_nodes().len());
        if quotes > 4096 || work.is_none_or(|w| w > 16_777_216) {
            return Err(LsvError::InvalidInput {
                field: "rough_lsv_market_iv_work_limit",
                index: quotes,
            }
            .into());
        }
        surface.validate_local_variance_source(&self.core.original_target)?;
        let mut hash = blake3::Hasher::new();
        hash.update(b"rough-family-market-iv-particle-adjoint/v1\0");
        hash.update(self.core.fingerprint.as_bytes());
        hash.update(MARKET_IV_INTERPOLATION.as_bytes());
        for axis in [
            surface.maturity_nodes(),
            surface.log_moneyness_nodes(),
            surface.implied_volatilities(),
        ] {
            hash.update(&(axis.len() as u64).to_le_bytes());
            for &v in axis {
                hash.update(&v.to_le_bytes());
            }
        }
        Ok(RoughFamilyLsvMarketIvRiskPlan {
            pricing: self.clone(),
            surface,
            fingerprint: Fingerprint::from_bytes(*hash.finalize().as_bytes()),
        })
    }
}
impl RoughFamilyLsvMarketIvRiskPlan {
    pub const METHOD: &'static str = "rough-family-market-iv-particle-adjoint-v1";
    pub const COORDINATE: &'static str = "market_black_iv_nodes_in_relative_log_moneyness";

    #[must_use]
    pub const fn risk_fingerprint(&self) -> Fingerprint {
        self.fingerprint
    }
    #[must_use]
    pub fn surface(&self) -> &MarketIvSurface {
        &self.surface
    }

    pub fn evaluate(&self) -> Result<RoughFamilyLsvMarketIvRisk, MonteCarloError> {
        let core = &self.pricing.core;
        let count = self.surface.implied_volatilities().len();
        let (price, risks) = core.run_projected::<Recalibrated>(count + 1, |a| {
            let mut quotes = self
                .surface
                .local_variance_pullback(&core.original_target, &a)?;
            let parallel = quotes.iter().copied().collect::<NeumaierSum>().total();
            if !parallel.is_finite() {
                return Err(LsvError::InvalidInput {
                    field: "rough_lsv_market_iv_parallel",
                    index: 0,
                }
                .into());
            }
            quotes.push(parallel);
            Ok(quotes)
        })?;
        let (mut values, mut errors) = risks.expect("recalibrated risk requested");
        let parallel_vega = values.pop().expect("parallel direction retained");
        let parallel_standard_error = errors.as_mut().and_then(Vec::pop);
        Ok(RoughFamilyLsvMarketIvRisk {
            price,
            maturity_nodes: self.surface.maturity_nodes().into(),
            log_moneyness_nodes: self.surface.log_moneyness_nodes().into(),
            implied_volatilities: self.surface.implied_volatilities().into(),
            quote_adjoints: values.into(),
            standard_errors: errors.map(Vec::into_boxed_slice),
            parallel_vega,
            parallel_standard_error,
            risk_fingerprint: self.fingerprint,
            method: Self::METHOD,
        })
    }
}
