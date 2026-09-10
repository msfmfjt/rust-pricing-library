use pricing_core::Date;
use pricing_market::MarketContext;
use pricing_mc::EngineConfig;
use pricing_models::ModelSpec;
use pricing_product::{AsianObservationValue, ProductSpec};
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
        if product.payment_date() < valuation_date {
            return Err(RequestValidationError::PaymentBeforeValuation {
                valuation_date,
                payment_date: product.payment_date(),
            });
        }
        if product.expiry() < valuation_date && !product.payoff_determined_by(valuation_date) {
            return Err(RequestValidationError::ExpiryBeforeValuation {
                valuation_date,
                expiry: product.expiry(),
            });
        }
        if let ProductSpec::ArithmeticAsian(asian) = &product {
            for observation in asian.observations() {
                match observation.value() {
                    AsianObservationValue::Known(_) if observation.date() > valuation_date => {
                        return Err(
                            RequestValidationError::AsianFutureObservationCannotCarryFixing {
                                observation_date: observation.date(),
                                valuation_date,
                            },
                        );
                    }
                    AsianObservationValue::Unknown if observation.date() < valuation_date => {
                        return Err(
                            RequestValidationError::AsianPastObservationRequiresKnownFixing {
                                observation_date: observation.date(),
                                valuation_date,
                            },
                        );
                    }
                    AsianObservationValue::Known(_) | AsianObservationValue::Unknown => {}
                }
            }
        }
        if let ProductSpec::Barrier(barrier) = &product
            && let Some(monitoring_date) = barrier
                .monitoring_dates()
                .iter()
                .copied()
                .find(|date| *date < valuation_date)
        {
            return Err(RequestValidationError::BarrierPastMonitoringUnsupported {
                monitoring_date,
                valuation_date,
            });
        }
        if let ProductSpec::FixedLookback(lookback) = &product {
            let has_past_monitoring = lookback
                .monitoring_dates()
                .iter()
                .any(|date| *date < valuation_date);
            match (
                has_past_monitoring,
                lookback.historical_extremum().is_some(),
            ) {
                (true, false) => {
                    return Err(
                        RequestValidationError::LookbackPastMonitoringRequiresHistoricalExtremum {
                            valuation_date,
                        },
                    );
                }
                (false, true) => {
                    return Err(
                        RequestValidationError::LookbackHistoricalExtremumWithoutPastMonitoring {
                            valuation_date,
                        },
                    );
                }
                (true, true) | (false, false) => {}
            }
        }
        if risk.vega_kt().is_some()
            && matches!(&model, ModelSpec::BlackScholes(_) | ModelSpec::Black76(_))
        {
            return Err(RequestValidationError::VegaKtUnsupportedForConstantVolatility);
        }
        if !product.supports_pathwise_risk()
            && (risk.delta() || risk.gamma().is_some() || risk.vega() || risk.vega_kt().is_some())
        {
            return Err(RequestValidationError::RiskUnsupportedForDiscontinuousProduct);
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
    use pricing_models::{Black76Spec, BlackScholesSpec};
    use pricing_product::{
        ArithmeticAsianSpec, AsianObservation, BarrierDirection, BarrierSpec, BarrierStyle,
        DigitalPayout, DigitalSpec, EuropeanVanillaSpec, FixedLookbackSpec, OptionSide,
    };
    use pricing_risk::{GammaConfig, SmileDynamics, SpotBump, VegaKtConfig};

    use super::*;

    fn curve(id: u32, discount: f64) -> Arc<LogLinearDiscountCurve> {
        Arc::new(
            LogLinearDiscountCurve::new(CurveId::new(id), vec![0.0, 1.0], vec![1.0, discount])
                .expect("valid curve"),
        )
    }

    fn components(
        product_currency: CurrencyId,
        market_currency: CurrencyId,
    ) -> (
        ProductSpec,
        MarketContext,
        ModelSpec,
        EngineConfig,
        RiskRequest,
    ) {
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
            PseudoMcConfig::new(7, 1024, VarianceReduction::new(true, false)).expect("engine"),
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

    #[test]
    fn request_rejects_pathwise_risk_for_discontinuous_products() {
        let currency = CurrencyId::new(1);
        let (base, market, model, engine, _) = components(currency, currency);
        let digital = ProductSpec::Digital(
            DigitalSpec::new(
                base.underlying(),
                currency,
                base.expiry(),
                100.0,
                10.0,
                OptionSide::Call,
                DigitalPayout::Cash,
            )
            .expect("digital"),
        );
        let barrier = ProductSpec::Barrier(
            BarrierSpec::new(
                base.underlying(),
                currency,
                base.expiry(),
                100.0,
                120.0,
                1.0,
                OptionSide::Call,
                BarrierDirection::Up,
                BarrierStyle::KnockOut,
                vec!["2027-03-04".parse().expect("monitoring"), base.expiry()],
                None,
                base.expiry(),
            )
            .expect("barrier"),
        );
        let risk = RiskRequest::new(
            true,
            None,
            false,
            None,
            SmileDynamics::StickyLogMoneyness,
            None,
            None,
        )
        .expect("risk");
        for product in [digital, barrier] {
            assert!(matches!(
                PricingRequest::new(
                    "2026-09-04".parse().expect("valuation date"),
                    product,
                    market.clone(),
                    model.clone(),
                    engine,
                    risk.clone(),
                ),
                Err(RequestValidationError::RiskUnsupportedForDiscontinuousProduct)
            ));
        }
    }

    #[test]
    fn request_accepts_pathwise_risk_for_supported_path_products() {
        let currency = CurrencyId::new(1);
        let (base, market, model, engine, _) = components(currency, currency);
        let risk = RiskRequest::new(
            true,
            Some(GammaConfig::new(
                SpotBump::relative(0.01).expect("gamma bump"),
            )),
            true,
            None,
            SmileDynamics::StickyLogMoneyness,
            None,
            None,
        )
        .expect("risk");
        let asian = ProductSpec::ArithmeticAsian(
            ArithmeticAsianSpec::new(
                base.underlying(),
                currency,
                100.0,
                1.0,
                OptionSide::Call,
                vec![
                    AsianObservation::unknown("2026-09-04".parse().expect("first"), 0.25)
                        .expect("first"),
                    AsianObservation::unknown("2027-09-04".parse().expect("second"), 0.75)
                        .expect("second"),
                ],
                "2027-09-04".parse().expect("payment"),
            )
            .expect("asian"),
        );
        PricingRequest::new(
            "2026-09-04".parse().expect("valuation date"),
            asian,
            market.clone(),
            model.clone(),
            engine,
            risk.clone(),
        )
        .expect("asian risk");

        let lookback = ProductSpec::FixedLookback(
            FixedLookbackSpec::new(
                base.underlying(),
                currency,
                100.0,
                1.0,
                OptionSide::Call,
                vec![
                    "2026-09-04".parse().expect("first"),
                    "2027-09-04".parse().expect("second"),
                ],
                None,
                "2027-09-04".parse().expect("payment"),
            )
            .expect("lookback"),
        );
        PricingRequest::new(
            "2026-09-04".parse().expect("valuation date"),
            lookback,
            market,
            model,
            engine,
            risk,
        )
        .expect("lookback risk");
    }

    #[test]
    fn request_rejects_vega_kt_for_constant_volatility_models() {
        let currency = CurrencyId::new(1);
        let (product, market, _, engine, _) = components(currency, currency);
        let vega_kt = VegaKtConfig::new(
            vec![
                "2027-03-04".parse().expect("first maturity"),
                "2027-09-04".parse().expect("second maturity"),
            ],
            vec![-0.2, 0.0, 0.2],
            1.0e-8,
            false,
        )
        .expect("vega kt");
        let risk = RiskRequest::new(
            false,
            None,
            true,
            Some(vega_kt),
            SmileDynamics::StickyLogMoneyness,
            None,
            None,
        )
        .expect("risk");

        for model in [
            ModelSpec::BlackScholes(BlackScholesSpec::new(0.2).expect("black scholes")),
            ModelSpec::Black76(Black76Spec::new(0.2).expect("black 76")),
        ] {
            assert!(matches!(
                PricingRequest::new(
                    "2026-09-04".parse().expect("valuation date"),
                    product.clone(),
                    market.clone(),
                    model,
                    engine,
                    risk.clone(),
                ),
                Err(RequestValidationError::VegaKtUnsupportedForConstantVolatility)
            ));
        }
    }

    #[test]
    fn request_validates_asian_observation_fixing_status_against_valuation_date() {
        let currency = CurrencyId::new(1);
        let (base, market, model, engine, risk) = components(currency, currency);
        let asian = |observations| {
            ProductSpec::ArithmeticAsian(
                ArithmeticAsianSpec::new(
                    base.underlying(),
                    currency,
                    100.0,
                    1.0,
                    OptionSide::Call,
                    observations,
                    "2027-09-04".parse().expect("payment"),
                )
                .expect("asian"),
            )
        };
        let valid = asian(vec![
            AsianObservation::known("2026-03-04".parse().expect("date"), 0.25, 95.0)
                .expect("known"),
            AsianObservation::unknown("2027-09-04".parse().expect("date"), 0.75).expect("unknown"),
        ]);
        PricingRequest::new(
            "2026-09-04".parse().expect("valuation date"),
            valid,
            market.clone(),
            model.clone(),
            engine,
            risk.clone(),
        )
        .expect("valid asian");

        let past_unknown = asian(vec![
            AsianObservation::unknown("2026-03-04".parse().expect("date"), 0.25).expect("unknown"),
            AsianObservation::unknown("2027-09-04".parse().expect("date"), 0.75).expect("unknown"),
        ]);
        assert!(matches!(
            PricingRequest::new(
                "2026-09-04".parse().expect("valuation date"),
                past_unknown,
                market.clone(),
                model.clone(),
                engine,
                risk.clone(),
            ),
            Err(RequestValidationError::AsianPastObservationRequiresKnownFixing { .. })
        ));

        let future_known = asian(vec![
            AsianObservation::known("2026-03-04".parse().expect("date"), 0.25, 95.0)
                .expect("known"),
            AsianObservation::known("2027-09-04".parse().expect("date"), 0.75, 105.0)
                .expect("known"),
        ]);
        assert!(matches!(
            PricingRequest::new(
                "2026-09-04".parse().expect("valuation date"),
                future_known,
                market,
                model,
                engine,
                risk,
            ),
            Err(RequestValidationError::AsianFutureObservationCannotCarryFixing { .. })
        ));
    }

    #[test]
    fn request_accepts_fully_fixed_asian_with_future_payment() {
        let currency = CurrencyId::new(1);
        let (base, market, model, engine, risk) = components(currency, currency);
        let product = ProductSpec::ArithmeticAsian(
            ArithmeticAsianSpec::new(
                base.underlying(),
                currency,
                100.0,
                1.0,
                OptionSide::Call,
                vec![
                    AsianObservation::known("2026-03-04".parse().expect("first"), 0.25, 95.0)
                        .expect("first"),
                    AsianObservation::known("2026-06-04".parse().expect("second"), 0.75, 115.0)
                        .expect("second"),
                ],
                "2027-09-04".parse().expect("payment"),
            )
            .expect("asian"),
        );
        PricingRequest::new(
            "2026-09-04".parse().expect("valuation date"),
            product,
            market,
            model,
            engine,
            risk,
        )
        .expect("fully fixed asian");
    }

    #[test]
    fn request_rejects_payment_before_valuation() {
        let currency = CurrencyId::new(1);
        let (base, market, model, engine, risk) = components(currency, currency);
        let product = ProductSpec::ArithmeticAsian(
            ArithmeticAsianSpec::new(
                base.underlying(),
                currency,
                100.0,
                1.0,
                OptionSide::Call,
                vec![
                    AsianObservation::known("2026-03-04".parse().expect("first"), 0.25, 95.0)
                        .expect("first"),
                    AsianObservation::known("2026-06-04".parse().expect("second"), 0.75, 115.0)
                        .expect("second"),
                ],
                "2026-06-04".parse().expect("payment"),
            )
            .expect("asian"),
        );
        assert!(matches!(
            PricingRequest::new(
                "2026-09-04".parse().expect("valuation date"),
                product,
                market,
                model,
                engine,
                risk,
            ),
            Err(RequestValidationError::PaymentBeforeValuation { .. })
        ));
    }

    #[test]
    fn request_rejects_barrier_past_monitoring_without_state() {
        let currency = CurrencyId::new(1);
        let (base, market, model, engine, risk) = components(currency, currency);
        let product = ProductSpec::Barrier(
            BarrierSpec::new(
                base.underlying(),
                currency,
                "2027-09-04".parse().expect("expiry"),
                100.0,
                120.0,
                1.0,
                OptionSide::Call,
                BarrierDirection::Up,
                BarrierStyle::KnockOut,
                vec![
                    "2026-03-04".parse().expect("past"),
                    "2027-09-04".parse().expect("future"),
                ],
                None,
                "2027-09-04".parse().expect("payment"),
            )
            .expect("barrier"),
        );
        assert!(matches!(
            PricingRequest::new(
                "2026-09-04".parse().expect("valuation date"),
                product,
                market,
                model,
                engine,
                risk,
            ),
            Err(RequestValidationError::BarrierPastMonitoringUnsupported { .. })
        ));
    }

    #[test]
    fn request_validates_lookback_historical_extremum_against_valuation_date() {
        let currency = CurrencyId::new(1);
        let (base, market, model, engine, risk) = components(currency, currency);
        let lookback = |historical_extremum| {
            ProductSpec::FixedLookback(
                FixedLookbackSpec::new(
                    base.underlying(),
                    currency,
                    100.0,
                    1.0,
                    OptionSide::Call,
                    vec![
                        "2026-03-04".parse().expect("past"),
                        "2027-09-04".parse().expect("future"),
                    ],
                    historical_extremum,
                    "2027-09-04".parse().expect("payment"),
                )
                .expect("lookback"),
            )
        };
        PricingRequest::new(
            "2026-09-04".parse().expect("valuation date"),
            lookback(Some(110.0)),
            market.clone(),
            model.clone(),
            engine,
            risk.clone(),
        )
        .expect("valid lookback");

        let fully_fixed = ProductSpec::FixedLookback(
            FixedLookbackSpec::new(
                base.underlying(),
                currency,
                100.0,
                1.0,
                OptionSide::Call,
                vec![
                    "2026-03-04".parse().expect("first"),
                    "2026-06-04".parse().expect("second"),
                ],
                Some(115.0),
                "2027-09-04".parse().expect("payment"),
            )
            .expect("lookback"),
        );
        PricingRequest::new(
            "2026-09-04".parse().expect("valuation date"),
            fully_fixed,
            market.clone(),
            model.clone(),
            engine,
            risk.clone(),
        )
        .expect("fully fixed lookback");
        assert!(matches!(
            PricingRequest::new(
                "2026-09-04".parse().expect("valuation date"),
                lookback(None),
                market.clone(),
                model.clone(),
                engine,
                risk.clone(),
            ),
            Err(RequestValidationError::LookbackPastMonitoringRequiresHistoricalExtremum { .. })
        ));

        let future_only = ProductSpec::FixedLookback(
            FixedLookbackSpec::new(
                base.underlying(),
                currency,
                100.0,
                1.0,
                OptionSide::Call,
                vec!["2027-09-04".parse().expect("future")],
                Some(110.0),
                "2027-09-04".parse().expect("payment"),
            )
            .expect("lookback"),
        );
        assert!(matches!(
            PricingRequest::new(
                "2026-09-04".parse().expect("valuation date"),
                future_only,
                market,
                model,
                engine,
                risk,
            ),
            Err(RequestValidationError::LookbackHistoricalExtremumWithoutPastMonitoring { .. })
        ));
    }
}
