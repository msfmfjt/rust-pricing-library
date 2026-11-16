use std::sync::Arc;

use pricing::core::{CurrencyId, CurveId, PositiveF64, UnderlyingId};
use pricing::cross_currency::{CrossCurrencyEquityOption, EquityFxDynamics, EquityFxSettlement};
use pricing::market::{EquityForward, EquityMarket, LogLinearDiscountCurve};
use pricing::product::{EuropeanVanillaSpec, OptionSide};

#[test]
fn fx_linked_calls_on_nonflat_curves_with_delayed_settlement() {
    let curve = |id, discounts| {
        Arc::new(
            LogLinearDiscountCurve::new(CurveId::new(id), vec![0., 0.5, 1.5, 2.], discounts)
                .unwrap(),
        )
    };
    let stock = UnderlyingId::new(10);
    let jpy = CurrencyId::new(2);
    let domestic = curve(1, vec![1., 0.99, 0.94, 0.90]);
    let foreign = EquityMarket::new(
        CurrencyId::new(1),
        EquityForward::new(
            stock,
            PositiveF64::new(100., "spot").unwrap(),
            curve(2, vec![1., 0.97, 0.93, 0.89]),
            curve(3, vec![1., 0.995, 0.978, 0.97]),
        ),
    );
    // Independent Black values with expiry DFs equal to the geometric means
    // of the middle two pillars, the domestic payment DF 0.90 and 100 shares.
    for (settlement, strike, expected) in [
        (EquityFxSettlement::FloatingFx, 99., 147_961.003_168_257_16),
        (
            EquityFxSettlement::Composite,
            14_850.,
            130_493.974_160_330_3,
        ),
    ] {
        let product = EuropeanVanillaSpec::new(
            stock,
            jpy,
            "2027-01-01".parse().unwrap(),
            strike,
            100.,
            OptionSide::Call,
        )
        .unwrap();
        let option = CrossCurrencyEquityOption::new(
            "2026-01-01".parse().unwrap(),
            product,
            &foreign,
            domestic.clone(),
            150.,
            EquityFxDynamics::new(0.21, 0.13, -0.4).unwrap(),
            settlement,
            "2028-01-01".parse().unwrap(),
        )
        .unwrap();
        assert!((option.analytical_price().unwrap() - expected).abs() < 1e-8);
    }
}
