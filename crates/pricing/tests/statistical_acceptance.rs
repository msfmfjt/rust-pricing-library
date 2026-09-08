use std::sync::Arc;

use pricing::analytical::black_scholes_oracle;
use pricing::core::{CurrencyId, CurveId, PositiveF64, UnderlyingId};
use pricing::market::{EquityForward, EquityMarket, LogLinearDiscountCurve, MarketContext};
use pricing::mc::{
    EngineConfig, ExecutionPolicy, PseudoMcConfig, RqmcConfig, VarianceReduction,
};
use pricing::models::{BlackScholesSpec, ModelSpec};
use pricing::product::{EuropeanVanillaSpec, OptionSide, ProductSpec};
use pricing::risk::{RiskRequest, SmileDynamics};
use pricing::{PricingRequest, price_monte_carlo};

const PSEUDO_SEEDS: [u64; 16] = [
    0x0000_0000_0000_0000,
    0x0000_0000_0000_0001,
    0x0123_4567_89ab_cdef,
    0xfedc_ba98_7654_3210,
    0xffff_ffff_ffff_ffff,
    0x9e37_79b9_7f4a_7c15,
    0xd1b5_4a32_d192_ed03,
    0x94d0_49bb_1331_11eb,
    0x243f_6a88_85a3_08d3,
    0x1319_8a2e_0370_7344,
    0xa409_3822_299f_31d0,
    0x082e_fa98_ec4e_6c89,
    0x4528_21e6_38d0_1377,
    0xbe54_66cf_34e9_0c6c,
    0xc0ac_29b7_c97c_50dd,
    0x3f84_d5b5_b547_0917,
];

const RQMC_SEEDS: [u64; 8] = [
    0x0000_0000_0000_0000,
    0x0000_0000_0000_0001,
    0x0123_4567_89ab_cdef,
    0xfedc_ba98_7654_3210,
    0x9e37_79b9_7f4a_7c15,
    0xd1b5_4a32_d192_ed03,
    0x243f_6a88_85a3_08d3,
    0xa409_3822_299f_31d0,
];

#[derive(Clone, Copy, Debug)]
struct Coverage {
    runs: usize,
    covered: usize,
    squared_z_sum: f64,
    maximum_absolute_z: f64,
}

impl Coverage {
    const fn new() -> Self {
        Self {
            runs: 0,
            covered: 0,
            squared_z_sum: 0.0,
            maximum_absolute_z: 0.0,
        }
    }

    fn record(&mut self, oracle: f64, value: f64, standard_error: f64, lower: f64, upper: f64) {
        assert!(standard_error.is_finite() && standard_error > 0.0);
        self.runs += 1;
        self.covered += usize::from(lower <= oracle && oracle <= upper);
        let absolute_z = ((value - oracle) / standard_error).abs();
        self.squared_z_sum += absolute_z * absolute_z;
        self.maximum_absolute_z = self.maximum_absolute_z.max(absolute_z);
    }

    fn root_mean_square_z(self) -> f64 {
        (self.squared_z_sum / self.runs as f64).sqrt()
    }
}

#[test]
#[ignore = "statistical acceptance is run explicitly by CI"]
fn reported_uncertainty_is_calibrated_across_fixed_seeds() {
    let policy = ExecutionPolicy::new(2, Some(256)).expect("execution policy");
    let oracle_request = request(EngineConfig::PseudoMonteCarlo(
        PseudoMcConfig::new(
            PSEUDO_SEEDS[0],
            4_096,
            VarianceReduction::new(true, false),
        )
        .expect("pseudo-MC config"),
    ));
    let oracle = black_scholes_oracle(&oracle_request).expect("oracle").price;

    let mut pseudo = Coverage::new();
    for seed in PSEUDO_SEEDS {
        let engine = EngineConfig::PseudoMonteCarlo(
            PseudoMcConfig::new(seed, 4_096, VarianceReduction::new(true, false))
                .expect("pseudo-MC config"),
        );
        record_run(&mut pseudo, oracle, &request(engine), policy);
    }

    let mut rqmc = Coverage::new();
    for seed in RQMC_SEEDS {
        let engine = EngineConfig::RandomizedQuasiMonteCarlo(
            RqmcConfig::new(256, 16, seed, VarianceReduction::new(false, true))
                .expect("RQMC config"),
        );
        record_run(&mut rqmc, oracle, &request(engine), policy);
    }

    println!(
        "pseudo_mc runs={} covered_95={} rms_z={:.6} max_abs_z={:.6}",
        pseudo.runs,
        pseudo.covered,
        pseudo.root_mean_square_z(),
        pseudo.maximum_absolute_z
    );
    println!(
        "rqmc runs={} covered_95={} rms_z={:.6} max_abs_z={:.6}",
        rqmc.runs,
        rqmc.covered,
        rqmc.root_mean_square_z(),
        rqmc.maximum_absolute_z
    );

    assert!(pseudo.covered >= 12, "pseudo-MC 95% interval under-coverage");
    assert!(rqmc.covered >= 6, "RQMC 95% interval under-coverage");
    assert!(pseudo.maximum_absolute_z < 5.0);
    assert!(rqmc.maximum_absolute_z < 5.0);
}

fn record_run(
    coverage: &mut Coverage,
    oracle: f64,
    request: &PricingRequest,
    policy: ExecutionPolicy,
) {
    let result = price_monte_carlo(request, policy).expect("valuation");
    let estimate = result.pricing_result.value;
    let interval = estimate.confidence_interval();
    coverage.record(
        oracle,
        estimate.value().get(),
        estimate.standard_error().get(),
        interval.lower().get(),
        interval.upper().get(),
    );
}

fn request(engine: EngineConfig) -> PricingRequest {
    let underlying = UnderlyingId::new(1);
    let currency = CurrencyId::new(1);
    let product = ProductSpec::EuropeanVanilla(
        EuropeanVanillaSpec::new(
            underlying,
            currency,
            "2027-09-04".parse().expect("expiry"),
            100.0,
            1.0,
            OptionSide::Call,
        )
        .expect("product"),
    );
    let market = MarketContext::Equity(EquityMarket::new(
        currency,
        EquityForward::new(
            underlying,
            PositiveF64::new(100.0, "spot").expect("spot"),
            curve(1, 0.05),
            curve(2, 0.02),
        ),
    ));
    PricingRequest::new(
        "2026-09-04".parse().expect("valuation date"),
        product,
        market,
        ModelSpec::BlackScholes(BlackScholesSpec::new(0.2).expect("model")),
        engine,
        RiskRequest::price_only(SmileDynamics::StickyLogMoneyness),
    )
    .expect("request")
}

fn curve(id: u32, rate: f64) -> Arc<LogLinearDiscountCurve> {
    Arc::new(
        LogLinearDiscountCurve::new(
            CurveId::new(id),
            vec![0.0, 1.0],
            vec![1.0, (-rate).exp()],
        )
        .expect("curve"),
    )
}
