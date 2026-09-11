use std::sync::Arc;

use pricing::core::{CurrencyId, CurveId, PositiveF64, UnderlyingId};
use pricing::market::{EquityForward, EquityMarket, LogLinearDiscountCurve, MarketContext};
use pricing::mc::{EngineConfig, ExecutionPolicy, PseudoMcConfig, RqmcConfig, VarianceReduction};
use pricing::models::{BlackScholesSpec, ModelSpec};
use pricing::product::{
    ArithmeticAsianSpec, AsianObservation, BarrierDirection, BarrierMonitoring, BarrierSpec,
    BarrierStyle, DigitalPayout, DigitalSpec, FixedLookbackSpec, OptionSide, ProductSpec,
};
use pricing::risk::{GammaConfig, PayoffSmoothing, RiskRequest, SmileDynamics, SpotBump};
use pricing::{Estimate, PricingRequest, RiskValidation, price_monte_carlo};

#[test]
fn zero_variance_path_product_grid_has_deterministic_prices() {
    let cases: [(&str, ProductSpec, RiskRequest, f64); 4] = [
        ("digital", digital(), price_only(), 10.0),
        (
            "continuous_barrier",
            continuous_barrier(),
            price_only(),
            10.0,
        ),
        ("arithmetic_asian", arithmetic_asian(), price_only(), 5.0),
        ("fixed_lookback", fixed_lookback(), price_only(), 5.0),
    ];
    for (name, product, risk, expected) in cases {
        let result = price_monte_carlo(
            &request(product, risk, pseudo_engine(7, 1, false), 0.0),
            policy(),
        )
        .unwrap_or_else(|error| panic!("{name}: {error}"));
        let estimate = result.pricing_result.value;
        assert_eq!(
            estimate.value().get().to_bits(),
            expected.to_bits(),
            "{name}"
        );
        assert_eq!(
            estimate.standard_error().get().to_bits(),
            0.0_f64.to_bits(),
            "{name}"
        );
    }
}

#[test]
#[ignore = "path-product statistical acceptance is run explicitly by CI"]
fn pseudo_and_rqmc_path_product_grid_agree_with_reported_uncertainty() {
    let cases = [
        ("digital", digital(), smoothed_price_only(2.0)),
        ("continuous_barrier", continuous_barrier(), price_only()),
        ("arithmetic_asian", arithmetic_asian(), price_only()),
        ("fixed_lookback", fixed_lookback(), price_only()),
    ];
    for (index, (name, product, risk)) in cases.into_iter().enumerate() {
        let pseudo = price_monte_carlo(
            &request(
                product.clone(),
                risk.clone(),
                pseudo_engine(0x1234_5678_9abc_def0 + index as u64, 16_384, true),
                0.2,
            ),
            policy(),
        )
        .unwrap_or_else(|error| panic!("{name} Pseudo-MC: {error}"));
        let rqmc = price_monte_carlo(
            &request(
                product,
                risk,
                rqmc_engine(0xfedc_ba98_7654_3210 + index as u64),
                0.2,
            ),
            policy(),
        )
        .unwrap_or_else(|error| panic!("{name} RQMC: {error}"));
        assert_estimates_agree(
            name,
            pseudo.pricing_result.value,
            rqmc.pricing_result.value,
            6.0,
        );
    }
}

#[test]
#[ignore = "path-product statistical acceptance is run explicitly by CI"]
fn smoothed_digital_and_barrier_aad_agree_with_crn_validation() {
    for (name, product, width) in [
        ("digital", digital(), 2.0),
        ("discrete_barrier", discrete_barrier(), 3.0),
    ] {
        let result = price_monte_carlo(
            &request(
                product,
                smoothed_full_risk(width),
                pseudo_engine(0x3141_5926_5358_9793, 32_768, true),
                0.2,
            ),
            policy(),
        )
        .unwrap_or_else(|error| panic!("{name}: {error}"));
        for (risk, validation) in [
            ("Delta", result.risk_diagnostics.delta_validation),
            ("Gamma", result.risk_diagnostics.gamma_validation),
            ("Vega", result.risk_diagnostics.vega_validation),
        ] {
            assert_validation_contains_zero(name, risk, validation.expect("validation"));
        }
    }
}

fn assert_estimates_agree(name: &str, left: Estimate, right: Estimate, z_limit: f64) {
    let difference = (left.value().get() - right.value().get()).abs();
    let combined_error = left
        .standard_error()
        .get()
        .hypot(right.standard_error().get());
    let bound = z_limit * combined_error.max(1.0e-14);
    println!(
        "{name}: pseudo={:.12} rqmc={:.12} difference={difference:.6e} bound={bound:.6e}",
        left.value().get(),
        right.value().get(),
    );
    assert!(
        difference <= bound,
        "{name}: difference={difference}, bound={bound}"
    );
}

fn assert_validation_contains_zero(name: &str, risk: &str, validation: RiskValidation) {
    let difference = validation.bump_minus_primary;
    let value = difference.value().get();
    let standard_error = difference.standard_error().get();
    let bound = (8.0 * standard_error).max(1.0e-10);
    println!("{name} {risk}: difference={value:.6e} bound={bound:.6e}");
    assert!(
        value.abs() <= bound,
        "{name} {risk}: difference={value}, bound={bound}"
    );
}

fn request(
    product: ProductSpec,
    risk: RiskRequest,
    engine: EngineConfig,
    volatility: f64,
) -> PricingRequest {
    let underlying = UnderlyingId::new(1);
    let currency = CurrencyId::new(1);
    PricingRequest::new(
        "2026-09-04".parse().expect("valuation"),
        product,
        MarketContext::Equity(EquityMarket::new(
            currency,
            EquityForward::new(
                underlying,
                PositiveF64::new(100.0, "spot").expect("spot"),
                curve(1),
                curve(2),
            ),
        )),
        ModelSpec::BlackScholes(BlackScholesSpec::new(volatility).expect("model")),
        engine,
        risk,
    )
    .expect("request")
}

fn digital() -> ProductSpec {
    ProductSpec::Digital(
        DigitalSpec::new(
            UnderlyingId::new(1),
            CurrencyId::new(1),
            "2027-09-04".parse().expect("expiry"),
            100.0,
            10.0,
            OptionSide::Call,
            DigitalPayout::Cash,
        )
        .expect("Digital"),
    )
}

fn discrete_barrier() -> ProductSpec {
    barrier(BarrierMonitoring::Discrete)
}

fn continuous_barrier() -> ProductSpec {
    barrier(BarrierMonitoring::Continuous)
}

fn barrier(monitoring: BarrierMonitoring) -> ProductSpec {
    let expiry = "2027-09-04".parse().expect("expiry");
    ProductSpec::Barrier(
        BarrierSpec::new(
            UnderlyingId::new(1),
            CurrencyId::new(1),
            expiry,
            90.0,
            125.0,
            1.0,
            OptionSide::Call,
            BarrierDirection::Up,
            BarrierStyle::KnockOut,
            monitoring,
            vec!["2027-03-05".parse().expect("monitoring"), expiry],
            None,
            expiry,
        )
        .expect("Barrier"),
    )
}

fn arithmetic_asian() -> ProductSpec {
    ProductSpec::ArithmeticAsian(
        ArithmeticAsianSpec::new(
            UnderlyingId::new(1),
            CurrencyId::new(1),
            95.0,
            1.0,
            OptionSide::Call,
            vec![
                AsianObservation::known(
                    "2026-06-04".parse().expect("past observation"),
                    0.2,
                    100.0,
                )
                .expect("known observation"),
                AsianObservation::unknown("2027-03-05".parse().expect("future observation"), 0.3)
                    .expect("unknown observation"),
                AsianObservation::unknown("2027-09-04".parse().expect("expiry observation"), 0.5)
                    .expect("unknown observation"),
            ],
            "2027-09-04".parse().expect("payment"),
        )
        .expect("Arithmetic Asian"),
    )
}

fn fixed_lookback() -> ProductSpec {
    ProductSpec::FixedLookback(
        FixedLookbackSpec::new(
            UnderlyingId::new(1),
            CurrencyId::new(1),
            100.0,
            1.0,
            OptionSide::Call,
            vec![
                "2026-06-04".parse().expect("past monitoring"),
                "2027-03-05".parse().expect("future monitoring"),
                "2027-09-04".parse().expect("expiry monitoring"),
            ],
            Some(105.0),
            "2027-09-04".parse().expect("payment"),
        )
        .expect("Fixed Lookback"),
    )
}

fn price_only() -> RiskRequest {
    RiskRequest::price_only(SmileDynamics::StickyLogMoneyness)
}

fn smoothed_price_only(half_width: f64) -> RiskRequest {
    price_only()
        .with_payoff_smoothing(PayoffSmoothing::compact_c2(half_width).expect("payoff smoothing"))
}

fn smoothed_full_risk(half_width: f64) -> RiskRequest {
    RiskRequest::new(
        true,
        Some(GammaConfig::new(
            SpotBump::relative(0.01).expect("Gamma bump"),
        )),
        true,
        None,
        SmileDynamics::StickyLogMoneyness,
        Some(16),
        Some(128),
    )
    .expect("risk")
    .with_payoff_smoothing(PayoffSmoothing::compact_c2(half_width).expect("payoff smoothing"))
}

fn pseudo_engine(seed: u64, units: u64, antithetic: bool) -> EngineConfig {
    EngineConfig::PseudoMonteCarlo(
        PseudoMcConfig::new(seed, units, VarianceReduction::new(antithetic, false))
            .expect("Pseudo-MC"),
    )
}

fn rqmc_engine(seed: u64) -> EngineConfig {
    EngineConfig::RandomizedQuasiMonteCarlo(
        RqmcConfig::new(1_024, 16, seed, VarianceReduction::new(true, true)).expect("RQMC"),
    )
}

fn policy() -> ExecutionPolicy {
    ExecutionPolicy::new(2, Some(256)).expect("execution policy")
}

fn curve(id: u32) -> Arc<LogLinearDiscountCurve> {
    Arc::new(
        LogLinearDiscountCurve::new(CurveId::new(id), vec![0.0, 1.0], vec![1.0, 1.0])
            .expect("curve"),
    )
}
