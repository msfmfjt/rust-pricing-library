use std::sync::Arc;

use pricing::analytical::{
    BlackScholesOracleInputs, BlackScholesOracleResult, black_76_oracle, black_scholes_oracle,
    evaluate,
};
use pricing::core::{CurrencyId, CurveId, EventId, PositiveF64, UnderlyingId};
use pricing::market::{
    DividendEvent, DividendQuote, EquityForward, EquityMarket, LogLinearDiscountCurve,
    MarketContext,
};
use pricing::mc::{EngineConfig, ExecutionPolicy, PseudoMcConfig, VarianceReduction};
use pricing::models::{Black76Spec, BlackScholesSpec, ModelSpec};
use pricing::product::{EuropeanVanillaSpec, OptionSide, ProductSpec};
use pricing::risk::{RiskRequest, SmileDynamics};
use pricing::{PricingRequest, price_monte_carlo};

#[derive(Clone)]
struct Case {
    spot: f64,
    strike: f64,
    volatility: f64,
    rate: f64,
    repo: f64,
    side: OptionSide,
    // (ex-time, fixed cash, proportional fraction)
    dividends: Vec<(f64, f64, f64)>,
}

impl Default for Case {
    fn default() -> Self {
        Self {
            spot: 100.0,
            strike: 90.0,
            volatility: 0.2,
            rate: 0.05,
            repo: 0.02,
            side: OptionSide::Call,
            dividends: vec![],
        }
    }
}

impl Case {
    fn request(&self, black76: bool) -> PricingRequest {
        let underlying = UnderlyingId::new(1);
        let currency = CurrencyId::new(1);
        let curve = |id, rate: f64| {
            Arc::new(
                LogLinearDiscountCurve::new(
                    CurveId::new(id),
                    vec![0.0, 2.0],
                    vec![1.0, (-2.0 * rate).exp()],
                )
                .unwrap(),
            )
        };
        let events = self
            .dividends
            .iter()
            .enumerate()
            .map(|(i, &(time, cash, beta))| {
                let event = EventId::new(i as u32 + 1);
                DividendEvent::new(
                    event,
                    time,
                    DividendQuote::fixed_cash_and_proportional(cash, beta, event).unwrap(),
                )
                .unwrap()
            })
            .collect();
        let spot = PositiveF64::new(self.spot, "spot").unwrap();
        let forward = if self.dividends.is_empty() {
            EquityForward::new(underlying, spot, curve(1, self.rate), curve(2, self.repo))
        } else {
            EquityForward::with_discrete_dividends(
                underlying,
                spot,
                curve(1, self.rate),
                curve(2, self.repo),
                events,
            )
            .unwrap()
        };
        PricingRequest::new(
            "2026-09-04".parse().unwrap(),
            ProductSpec::EuropeanVanilla(
                EuropeanVanillaSpec::new(
                    underlying,
                    currency,
                    "2027-09-04".parse().unwrap(),
                    self.strike,
                    1.0,
                    self.side,
                )
                .unwrap(),
            ),
            MarketContext::Equity(EquityMarket::new(currency, forward)),
            if black76 {
                ModelSpec::Black76(Black76Spec::new(self.volatility).unwrap())
            } else {
                ModelSpec::BlackScholes(BlackScholesSpec::new(self.volatility).unwrap())
            },
            EngineConfig::PseudoMonteCarlo(
                PseudoMcConfig::new(42, 65_536, VarianceReduction::new(true, false)).unwrap(),
            ),
            RiskRequest::price_only(SmileDynamics::StickyLogMoneyness),
        )
        .unwrap()
    }

    fn oracle(&self, black76: bool) -> BlackScholesOracleResult {
        let request = self.request(black76);
        if black76 {
            black_76_oracle(&request)
        } else {
            black_scholes_oracle(&request)
        }
        .unwrap()
    }
}

fn close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{actual} != {expected}"
    );
}

#[test]
fn cash_and_proportional_dividends_match_deterministic_simulation() {
    for (time, cash, beta, terminal_spot) in [
        (0.5, 20.0, 0.0, 80.0),
        (1.0, 20.0, 0.0, 80.0), // Dividend before the expiry payoff.
        (1.5, 20.0, 0.0, 100.0),
        (0.5, 0.0, 0.2, 80.0),
        (0.5, 20.0, 0.1, 70.0),
    ] {
        for side in [OptionSide::Call, OptionSide::Put] {
            let case = Case {
                volatility: 0.0,
                rate: 0.0,
                repo: 0.0,
                side,
                dividends: vec![(time, cash, beta)],
                ..Case::default()
            };
            let sign = if side == OptionSide::Call { 1.0 } else { -1.0 };
            let expected = (sign * (terminal_spot - case.strike)).max(0.0);
            for black76 in [false, true] {
                let oracle = case.oracle(black76);
                close(oracle.price, expected, 1e-12);
                let result = price_monte_carlo(
                    &case.request(black76),
                    ExecutionPolicy::new(1, None).unwrap(),
                )
                .unwrap();
                close(result.pricing_result.value.value().get(), expected, 1e-12);
                close(oracle.gamma, 0.0, 0.0);
                close(oracle.vega, 0.0, 0.0);
            }
        }
    }
}

#[test]
fn mixed_dividend_uses_discounted_cash_reserve_and_fixed_cash_greeks() {
    for side in [OptionSide::Call, OptionSide::Put] {
        let case = Case {
            side,
            dividends: vec![(0.5, 20.0, 0.1)],
            ..Case::default()
        };
        let reserve = 20.0 / 0.9 * (-(case.rate - case.repo) * 0.5).exp();
        let reference = evaluate(BlackScholesOracleInputs {
            side,
            spot: 0.9 * (case.spot - reserve),
            strike: case.strike,
            notional: 1.0,
            discount: (-case.rate).exp(),
            dividend_discount: (-case.repo).exp(),
            volatility: case.volatility,
            time: 1.0,
        })
        .unwrap();
        for black76 in [false, true] {
            let result = case.oracle(black76);
            let scale = if black76 {
                0.9 * (case.repo - case.rate).exp()
            } else {
                0.9
            };
            close(result.price, reference.price, 1e-12);
            close(result.delta, reference.delta * scale, 1e-13);
            close(result.gamma, reference.gamma * scale * scale, 1e-14);
            close(result.vega, reference.vega, 1e-12);
        }
    }
}

#[test]
fn future_reserve_supports_nonpositive_adjusted_strikes() {
    for strike in [10.0, 20.0] {
        for side in [OptionSide::Call, OptionSide::Put] {
            let case = Case {
                strike,
                side,
                rate: 0.0,
                repo: 0.0,
                dividends: vec![(1.5, 20.0, 0.0)],
                ..Case::default()
            };
            for black76 in [false, true] {
                let result = case.oracle(black76);
                let call = side == OptionSide::Call;
                close(result.price, if call { 100.0 - strike } else { 0.0 }, 1e-12);
                close(result.delta, if call { 1.0 } else { 0.0 }, 1e-14);
                close(result.gamma, 0.0, 0.0);
                close(result.vega, 0.0, 0.0);
            }
        }
    }
}

#[test]
fn affine_prices_match_simulation_and_greeks_match_rebuilt_market_bumps() {
    for dividends in [
        vec![],
        vec![(0.5, 20.0, 0.0)],
        vec![(0.5, 0.0, 0.2)],
        vec![(0.25, 10.0, 0.1), (1.0, 5.0, 0.05), (1.5, 15.0, 0.1)],
    ] {
        for side in [OptionSide::Call, OptionSide::Put] {
            let case = Case {
                side,
                dividends: dividends.clone(),
                ..Case::default()
            };
            for black76 in [false, true] {
                let result = case.oracle(black76);
                let up = Case {
                    spot: case.spot + 0.01,
                    ..case.clone()
                }
                .oracle(black76);
                let down = Case {
                    spot: case.spot - 0.01,
                    ..case.clone()
                }
                .oracle(black76);
                let bump = 0.01
                    * if black76 {
                        (case.rate - case.repo).exp()
                    } else {
                        1.0
                    };
                close(result.delta, (up.price - down.price) / (2.0 * bump), 2e-8);
                close(
                    result.gamma,
                    (up.price - 2.0 * result.price + down.price) / bump.powi(2),
                    1e-8,
                );
                let up = Case {
                    volatility: case.volatility + 1e-5,
                    ..case.clone()
                }
                .oracle(black76);
                let down = Case {
                    volatility: case.volatility - 1e-5,
                    ..case.clone()
                }
                .oracle(black76);
                close(result.vega, (up.price - down.price) / 2e-5, 1e-7);
                let mc = price_monte_carlo(
                    &case.request(black76),
                    ExecutionPolicy::new(1, None).unwrap(),
                )
                .unwrap();
                let estimate = mc.pricing_result.value;
                close(
                    result.price,
                    estimate.value().get(),
                    6.0 * estimate.standard_error().get() + 1e-12,
                );
            }
        }
    }
}
