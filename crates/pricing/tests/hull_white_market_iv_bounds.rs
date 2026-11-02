use pricing::hull_white::HullWhiteEquityPricingPlan;
use pricing::market::{LocalVarianceGrid, MarketIvSurface};
use pricing::mc::{ExecutionPolicy, hull_white::HullWhiteLsvTarget, lsv::LsvParticleConfig};
use pricing::models::{
    Bergomi1Factor, HullWhite1Factor, HybridCorrelation, LocalVolatilitySpec, ModelSpec,
};
use pricing::{JsonLimits, PricingRequest, parse_request_json};

const BOUND_ERROR: &str = "market_iv_local_variance_bound_kink";

fn surface(volatility: f64) -> MarketIvSurface {
    MarketIvSurface::new(vec![0.5, 1.0], vec![-0.5, 0.5], vec![volatility; 4]).unwrap()
}

fn target(volatility: f64, floor: f64, cap: f64) -> HullWhiteLsvTarget {
    HullWhiteLsvTarget::from_market_iv(
        surface(volatility),
        vec![0.0, 0.5, 1.0],
        vec![-0.5, 0.5],
        floor,
        cap,
    )
    .unwrap()
}

fn local_model(t: &HullWhiteLsvTarget) -> LocalVolatilitySpec {
    let g = t.grid();
    LocalVolatilitySpec::from_explicit_grid(
        g.time_nodes().to_vec(),
        g.log_moneyness_nodes().to_vec(),
        g.values().to_vec(),
        g.floor(),
        g.cap(),
    )
    .unwrap()
}

fn policy() -> ExecutionPolicy {
    ExecutionPolicy::new(1, Some(16)).unwrap()
}

fn particles() -> LsvParticleConfig {
    LsvParticleConfig::new(32, 72, 0.5, 2.0, true).unwrap()
}

fn rates() -> HullWhite1Factor {
    HullWhite1Factor::new(0.2, vec![0.0], vec![0.0]).unwrap()
}

fn plan(t: &HullWhiteLsvTarget) -> HullWhiteEquityPricingPlan {
    let golden = parse_request_json(
        include_bytes!("../../../fixtures/v1/pricing_request.golden.json"),
        JsonLimits::DEFAULT,
    )
    .unwrap();
    // Same effective variance target as the forward-density target.
    let request = PricingRequest::new(
        golden.valuation_date(),
        golden.product().clone(),
        golden.market().clone(),
        ModelSpec::LocalVolatility(local_model(t)),
        golden.engine(),
        golden.risk().clone(),
    )
    .unwrap();
    HullWhiteEquityPricingPlan::compile_lsv(
        &request,
        t,
        Bergomi1Factor::new(1.0, 0.0, -0.5).unwrap(),
        rates(),
        HybridCorrelation::new(-0.5, 0.0, 0.0).unwrap(),
        particles(),
        policy(),
    )
    .unwrap()
}

#[test]
fn paired_quote_reverse_rejects_exact_bounds_with_unequal_one_sided_derivatives() {
    // Binary-exact sigma/variance make this an equality test, not a tolerance test.
    let sigma = 0.25;
    let variance = sigma * sigma;
    for (floor, cap) in [(1e-8, variance), (variance, 4.0)] {
        let t = target(sigma, floor, cap);
        assert!(t.grid().repairs().is_empty());
        assert!(t.grid().values().iter().all(|v| *v == variance));
        // Independently expose the kink in the primal floor/cap operation.
        let bumped = |bump| {
            LocalVarianceGrid::from_surface(
                &surface(sigma + bump),
                vec![0.0, 0.5, 1.0],
                vec![-0.5, 0.5],
                floor,
                cap,
            )
            .unwrap()
            .values()[0]
        };
        let h = 1e-6;
        let left = (variance - bumped(-h)) / h;
        let right = (bumped(h) - variance) / h;
        assert!((left - right).abs() > 0.49, "left={left}, right={right}");
        // Neither a nonzero nor a zero cotangent establishes differentiability.
        for seed in [0.0, 1.0] {
            let error = t.reverse_market_iv(&[seed; 6], &[0.0; 6]).unwrap_err();
            assert!(error.to_string().contains(BOUND_ERROR), "{error}");
        }
    }
}

#[test]
fn single_asset_native_plan_rejects_bound_quote_risk_but_keeps_price() {
    for (floor, cap) in [(1e-8, 0.0625), (0.0625, 4.0)] {
        let plan = plan(&target(0.25, floor, cap));
        let price = plan.evaluate().unwrap();
        let error = plan.evaluate_aad().unwrap_err();
        assert!(error.to_string().contains(BOUND_ERROR), "{error}");
        // A failed risk request must leave the compiled plan reusable.
        assert_eq!(price, plan.evaluate().unwrap());
    }
}

#[test]
fn adjacent_interior_bounds_and_unquoted_targets_keep_valid_risk() {
    let variance = 0.0625_f64;
    let t = target(
        0.25,
        f64::from_bits(variance.to_bits() - 1),
        f64::from_bits(variance.to_bits() + 1),
    );
    // No epsilon band: even the immediately adjacent representable bounds are interior.
    let quotes = t.reverse_market_iv(&[1.0; 6], &[0.0; 6]).unwrap();
    assert!((quotes.iter().sum::<f64>() - 3.0).abs() < 1e-12);
    assert!(plan(&t).evaluate_aad().is_ok());
    // Explicit effective-node risk has no IV floor/cap derivative to take.
    let bounded = target(0.25, 1e-8, variance);
    let unquoted =
        HullWhiteLsvTarget::new(bounded.grid().clone(), bounded.log_densities().to_vec()).unwrap();
    assert!(plan(&unquoted).evaluate_aad().is_ok());
}
