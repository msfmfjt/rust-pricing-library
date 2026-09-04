use pricing_core::Date;
use pricing_market::MarketContext;
use pricing_mc::EngineConfig;
use pricing_models::ModelSpec;
use pricing_product::ProductSpec;
use pricing_risk::RiskRequest;

use crate::RequestValidationError;

#[derive(Clone, Debug, PartialEq)]
pub struct PricingRequest {
    valuation_date: Date,
    product: ProductSpec,
    market: MarketContext,
    model: ModelSpec,
    engine: EngineConfig,
    risk: RiskRequest,
}

impl PricingRequest {
    pub fn new(
        valuation_date: Date,
        product: ProductSpec,
        market: MarketContext,
        model: ModelSpec,
        engine: EngineConfig,
        risk: RiskRequest,
    ) -> Result<Self, RequestValidationError> {
        if product.currency() != market.currency() {
            return Err(RequestValidationError::CurrencyMismatch {
                product: product.currency(),
                market: market.currency(),
            });
        }
        if product.underlying() != market.equity().forward().underlying() {
            return Err(RequestValidationError::UnderlyingMismatch {
                product: product.underlying(),
                market: market.equity().forward().underlying(),
            });
        }
        if product.expiry() < valuation_date {
            return Err(RequestValidationError::ExpiryBeforeValuation {
                valuation_date,
                expiry: product.expiry(),
            });
        }
        if risk.vega_kt().is_some() && matches!(&model, ModelSpec::BlackScholes(_)) {
            return Err(RequestValidationError::VegaKtUnsupportedForBlackScholes);
        }
        Ok(Self {
            valuation_date,
            product,
            market,
            model,
            engine,
            risk,
        })
    }

    #[must_use]
    pub const fn valuation_date(&self) -> Date {
        self.valuation_date
    }

    #[must_use]
    pub const fn product(&self) -> &ProductSpec {
        &self.product
    }

    #[must_use]
    pub const fn market(&self) -> &MarketContext {
        &self.market
    }

    #[must_use]
    pub const fn model(&self) -> &ModelSpec {
        &self.model
    }

    #[must_use]
    pub const fn engine(&self) -> EngineConfig {
        self.engine
    }

    #[must_use]
    pub const fn risk(&self) -> &RiskRequest {
        &self.risk
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use pricing_core::{CurrencyId, CurveId, PositiveF64, UnderlyingId};
    use pricing_market::{EquityForward, EquityMarket, LogLinearDiscountCurve};
    use pricing_mc::{PseudoMcConfig, VarianceReduction};
    use pricing_models::BlackScholesSpec;
    use pricing_product::{EuropeanVanillaSpec, OptionSide};
    use pricing_risk::SmileDynamics;

    use super::*;

    fn curve(id: u32, discount: f64) -> Arc<LogLinearDiscountCurve> {
        Arc::new(
            LogLinearDiscountCurve::new(
                CurveId::new(id),
                vec![0.0, 1.0],
                vec![1.0, discount],
            )
            .expect("valid curve"),
        )
    }

    fn components(
        product_currency: CurrencyId,
        market_currency: CurrencyId,
    ) -> (ProductSpec, MarketContext, ModelSpec, EngineConfig, RiskRequest) {
        let underlying = UnderlyingId::new(1);
        let product = ProductSpec::EuropeanVanilla(
            EuropeanVanillaSpec::new(
                underlying,
                product_currency,
                "2027-09-04".parse().expect("expiry"),
                100.0,
                1.0,
                OptionSide::Call,
            )
            .expect("valid product"),
        );
        let forward = EquityForward::new(
            underlying,
            PositiveF64::new(100.0, "spot").expect("spot"),
            curve(1, 0.95),
            curve(2, 0.98),
        );
        let market = MarketContext::Equity(EquityMarket::new(market_currency, forward));
        let model = ModelSpec::BlackScholes(BlackScholesSpec::new(0.2).expect("model"));
        let engine = EngineConfig::PseudoMonteCarlo(
            PseudoMcConfig::new(7, 1024, VarianceReduction::new(true, false))
                .expect("engine"),
        );
        let risk = RiskRequest::price_only(SmileDynamics::StickyLogMoneyness);
        (product, market, model, engine, risk)
    }

    #[test]
    fn request_accepts_consistent_single_currency_equity_inputs() {
        let currency = CurrencyId::new(1);
        let (product, market, model, engine, risk) = components(currency, currency);
        let request = PricingRequest::new(
            "2026-09-04".parse().expect("valuation date"),
            product,
            market,
            model,
            engine,
            risk,
        )
        .expect("consistent request");
        assert_eq!(request.valuation_date().to_string(), "2026-09-04");
    }

    #[test]
    fn request_rejects_currency_mismatch() {
        let (product, market, model, engine, risk) =
            components(CurrencyId::new(1), CurrencyId::new(2));
        assert!(matches!(
            PricingRequest::new(
                "2026-09-04".parse().expect("valuation date"),
                product,
                market,
                model,
                engine,
                risk,
            ),
            Err(RequestValidationError::CurrencyMismatch { .. })
        ));
    }
}
