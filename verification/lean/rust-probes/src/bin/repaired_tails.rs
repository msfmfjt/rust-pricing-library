//! Native differential replay of the two previously reported barrier/dividend defects.
use std::sync::Arc;

use pricing::core::{CurrencyId, CurveId, EventId, PathIndex, PositiveF64, UnderlyingId};
use pricing::market::{
    AffineDividendTransform, DividendEvent, DividendQuote, EquityForward, EquityMarket,
    LogLinearDiscountCurve, MarketContext,
};
use pricing::mc::{
    BarrierBridgeDirection, EngineConfig, ExecutionPolicy, PseudoMcConfig,
    SmoothedBarrierBridgeEndpoint, SmoothedBarrierBridgeEndpointInput, VarianceReduction,
};
use pricing::models::{BlackScholesSpec, ModelSpec};
use pricing::product::{
    BarrierDirection, BarrierMonitoring, BarrierSpec, BarrierStyle, CompactC2Smoothing, OptionSide,
    ProductSpec,
};
use pricing::risk::{PayoffSmoothing, RiskRequest, SmileDynamics};
use pricing::{MonteCarloError, PricingRequest, price_monte_carlo};

const SPOT: f64 = 100.0;
const HALF_WIDTH: f64 = 1.0;

fn discount_curve(id: u32) -> Arc<LogLinearDiscountCurve> {
    Arc::new(LogLinearDiscountCurve::new(CurveId::new(id), vec![0.0, 1.0], vec![1.0, 1.0]).unwrap())
}

fn price(barrier: f64, volatility: f64) -> Result<f64, MonteCarloError> {
    let asset = UnderlyingId::new(1);
    let ccy = CurrencyId::new(1);
    let expiry = "2027-09-04".parse().unwrap();
    let product = BarrierSpec::new(
        asset,
        ccy,
        expiry,
        90.0,
        barrier,
        1.0,
        OptionSide::Call,
        BarrierDirection::Up,
        BarrierStyle::KnockOut,
        BarrierMonitoring::Continuous,
        vec![expiry],
        None,
        expiry,
    )
    .unwrap();
    let forward = EquityForward::new(
        asset,
        PositiveF64::new(SPOT, "spot").unwrap(),
        discount_curve(1),
        discount_curve(2),
    );
    let risk = RiskRequest::price_only(SmileDynamics::StickyLogMoneyness)
        .with_payoff_smoothing(PayoffSmoothing::compact_c2(HALF_WIDTH).unwrap());
    let engine = PseudoMcConfig::new(7, 16, VarianceReduction::new(false, false)).unwrap();
    let request = PricingRequest::new(
        "2026-09-04".parse().unwrap(),
        ProductSpec::Barrier(product),
        MarketContext::Equity(EquityMarket::new(ccy, forward)),
        ModelSpec::BlackScholes(BlackScholesSpec::new(volatility).unwrap()),
        EngineConfig::PseudoMonteCarlo(engine),
        risk,
    )
    .unwrap();
    let result = price_monte_carlo(&request, ExecutionPolicy::new(1, Some(16)).unwrap())?;
    Ok(result.pricing_result.value.value().get())
}

fn barrier_is_valid() -> bool {
    let x = 1.0 - 2.0_f64.powi(-25);
    let barrier = SPOT - x;
    let smoothing = CompactC2Smoothing::new(HALF_WIDTH).unwrap();
    let hit = smoothing.indicator(x).value;
    let endpoint = SmoothedBarrierBridgeEndpoint::evaluate(SmoothedBarrierBridgeEndpointInput {
        direction: BarrierBridgeDirection::Up,
        state: SPOT,
        transformed_barrier: barrier,
        affine_scale: 1.0,
        smoothing,
    })
    .unwrap();
    let safety = endpoint.safety_weight();
    println!(
        "x={x:.17}, H={hit:.17} (0x{:016x}), safety={safety:e}, log(safety)={}",
        hit.to_bits(),
        safety.ln()
    );
    let expected_safety = f64::from_bits(0x3b43_ffff_f880_0001);
    let mut valid = (0.0..=1.0).contains(&hit)
        && endpoint.hit_weight().to_bits() == hit.to_bits()
        && safety.to_bits() == expected_safety.to_bits();
    for volatility in [0.0, 0.2] {
        assert!(
            price(99.5, volatility)
                .expect("interior control price")
                .is_finite()
        );
        let outcome = price(barrier, volatility);
        println!("volatility={volatility}, endpoint price={outcome:?}");
        valid &= outcome.is_ok_and(|amount| amount.is_finite() && amount > 0.0);
    }
    valid
}

fn dividend_is_valid() -> bool {
    let spot = PositiveF64::new(128.0, "spot").unwrap();
    let id = EventId::new(1);
    let quote = DividendQuote::fixed_cash(64.0, id).unwrap();
    let event = DividendEvent::new(id, 0.5, quote).unwrap();
    let dividends = AffineDividendTransform::new(UnderlyingId::new(1), spot, vec![event]).unwrap();
    let f = 2.0_f64.powi(-47);
    let before = dividends
        .coordinate_before_event(0)
        .unwrap()
        .reconstruct_spot(spot, f);
    let after = dividends
        .coordinate_after_time(0.5)
        .unwrap()
        .reconstruct_spot(spot, f);
    let checked = dividends.validate_post_event_f_state(0, PathIndex::new(0), f);
    println!(
        "cash=64, F={f:e}, pre={before}, post from coordinate={after:e}, validator={checked:?}"
    );
    assert_eq!(before, 64.0);
    assert_eq!(after, 2.0_f64.powi(-48));
    let control = dividends
        .validate_post_event_f_state(0, PathIndex::new(0), 64.0)
        .unwrap();
    assert_eq!(control, 32.0);
    checked.is_ok_and(|value| value.to_bits() == after.to_bits())
}

fn main() {
    let discrepancies = usize::from(!barrier_is_valid()) + usize::from(!dividend_is_valid());
    println!("Tail contract discrepancies: {discrepancies}");
    if discrepancies != 0 {
        std::process::exit(1);
    }
}
