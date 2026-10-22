use pricing::hull_white::HullWhiteEquityPricingPlan;
use pricing::market::{LocalVarianceGrid, MarketIvSurface};
use pricing::mc::{ExecutionPolicy, hull_white::HullWhiteLsvTarget, lsv::LsvParticleConfig};
use pricing::models::{
    Bergomi1Factor, HullWhite1Factor, HybridCorrelation, LocalVolatilitySpec, ModelSpec,
};
use pricing::{
    CapabilityStatus, EquityModelSpec, JsonLimits, MultiAssetSpecification, PricingModelSpec,
    PricingOperation, PricingSource, PricingSpecification, PricingTask, RateModelSpec,
    VolatilityFactorSpec, parse_pricing_task_json, parse_request_json,
};

const AAD: PricingOperation = PricingOperation::Aad {
    gamma_relative_bump: None,
};
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

fn single(t: &HullWhiteLsvTarget) -> PricingSpecification {
    let request = parse_request_json(
        include_bytes!("../../../fixtures/v1/pricing_request.golden.json"),
        JsonLimits::DEFAULT,
    )
    .unwrap();
    PricingSpecification::from_request(request, policy())
        .with_model(
            PricingModelSpec::new(
                EquityModelSpec::LocalStochasticVolatility {
                    target: local_model(t),
                    factor: VolatilityFactorSpec::Bergomi(
                        Bergomi1Factor::new(1.0, 0.0, -0.5).unwrap(),
                    ),
                    particles: particles(),
                    forward_density: Some(t.clone()),
                },
                RateModelSpec::HullWhite {
                    model: rates(),
                    equity_rate_correlation: 0.0,
                    vol_rate_correlations: vec![0.0],
                },
            ),
            None,
        )
        .unwrap()
}

fn basket(t: &HullWhiteLsvTarget, basket_target: bool) -> MultiAssetSpecification {
    let fixture = parse_pricing_task_json(
        include_bytes!("../../../fixtures/task-v1/multi_asset_joint.golden.json"),
        JsonLimits::DEFAULT,
    )
    .unwrap();
    let PricingSource::MultiAsset(s) = fixture.source() else {
        panic!("basket fixture")
    };
    let mut model = s.model().clone();
    model.hull_white.as_mut().unwrap().rate_model = rates();
    if basket_target {
        let (config, extensions) = model.local_correlation.as_mut().unwrap();
        config.target = t.grid().clone();
        extensions.hull_white_target = Some(t.clone());
    } else {
        model.local_correlation = None;
        model.models[0] = ModelSpec::LocalVolatility(local_model(t));
        model.hull_white.as_mut().unwrap().lsv_targets[0] = Some(t.clone());
    }
    MultiAssetSpecification::new(
        s.valuation_date(),
        s.product().clone(),
        s.markets().to_vec(),
        model,
        s.engine(),
        policy(),
        0.5,
    )
    .unwrap()
}

fn assert_price_only_at_bound(source: impl Into<PricingSource>) {
    let task = PricingTask::new(source, PricingOperation::Evaluate).unwrap();
    let compiled = task.compile().unwrap();
    let price = compiled.evaluate().unwrap();
    let capability = compiled.capability_for(AAD);
    assert_eq!(
        capability.status,
        CapabilityStatus::Unsupported,
        "{capability:?}"
    );
    assert!(
        capability.detail.contains(BOUND_ERROR),
        "{}",
        capability.detail
    );
    assert!(
        compiled
            .with_operation(AAD)
            .unwrap_err()
            .to_string()
            .contains(BOUND_ERROR)
    );
    assert!(
        task.with_operation(AAD)
            .unwrap()
            .compile()
            .unwrap_err()
            .to_string()
            .contains(BOUND_ERROR)
    );
    // Failed operation selection must leave the existing price task reusable.
    assert_eq!(price, compiled.evaluate().unwrap());
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
fn single_asset_native_and_task_preflight_reject_bound_quote_risk() {
    for (floor, cap) in [(1e-8, 0.0625), (0.0625, 4.0)] {
        let t = target(0.25, floor, cap);
        let s = single(&t);
        // Construct a legacy request with the exact same effective variance target.
        let request = pricing::PricingRequest::new(
            s.valuation_date(),
            s.product().clone(),
            s.market().clone(),
            ModelSpec::LocalVolatility(local_model(&t)),
            s.engine(),
            s.risk().clone(),
        )
        .unwrap();
        let plan = HullWhiteEquityPricingPlan::compile_lsv(
            &request,
            &t,
            Bergomi1Factor::new(1.0, 0.0, -0.5).unwrap(),
            rates(),
            HybridCorrelation::new(-0.5, 0.0, 0.0).unwrap(),
            particles(),
            policy(),
        )
        .unwrap();
        let price = plan.evaluate().unwrap();
        assert!(
            plan.evaluate_aad()
                .unwrap_err()
                .to_string()
                .contains(BOUND_ERROR)
        );
        assert_eq!(price, plan.evaluate().unwrap());
        assert_price_only_at_bound(s);
    }
}

#[test]
fn basket_marginal_and_local_correlation_quote_bounds_reach_task_preflight() {
    for basket_target in [false, true] {
        let sigma = if basket_target { 0.1875 } else { 0.25 };
        let variance = sigma * sigma;
        for (floor, cap) in [(1e-8, variance), (variance, 4.0)] {
            assert_price_only_at_bound(basket(&target(sigma, floor, cap), basket_target));
        }
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
    let strict = PricingTask::new(single(&t), AAD)
        .unwrap()
        .compile()
        .unwrap();
    assert_eq!(strict.capability().status, CapabilityStatus::Supported);
    assert!(strict.evaluate().is_ok());
    // Explicit effective-node risk has no IV floor/cap derivative to take.
    let bounded = target(0.25, 1e-8, variance);
    let unquoted =
        HullWhiteLsvTarget::new(bounded.grid().clone(), bounded.log_densities().to_vec()).unwrap();
    let task = PricingTask::new(single(&unquoted), AAD)
        .unwrap()
        .compile()
        .unwrap();
    assert!(task.evaluate().is_ok());
}
