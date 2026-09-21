//! Same source is compiled against the pre-S1 baseline and the candidate.
//! JSON construction and result hashing are outside the operation timer.
use pricing::core::{CurrencyId, CurveId, Date, EventId, PositiveF64, UnderlyingId};
use pricing::hull_white::HullWhiteEquityPricingPlan;
use pricing::lsv::{BergomiLsvPricingPlan, RoughBergomiLsvPricingPlan};
use pricing::market::{
    DividendEvent, DividendQuote, EquityForward, EquityMarket, LocalVarianceGrid,
    LogLinearDiscountCurve, MarketIvSurface,
};
use pricing::mc::hull_white::HullWhiteLsvTarget;
use pricing::mc::lsv::LsvParticleConfig;
use pricing::mc::{EngineConfig, ExecutionPolicy, PseudoMcConfig, VarianceReduction};
use pricing::models::{
    Bergomi1Factor, Bergomi2Factor, BergomiDynamics, BlackScholesSpec, HullWhite1Factor,
    HybridCorrelation, LocalVolatilitySpec, ModelSpec, RoughBergomi,
};
use pricing::multi_asset::*;
use pricing::product::{CompactC2Smoothing, OptionSide};
use pricing::{JsonLimits, PricingRequest, parse_request_json};
use serde_json::{Value, json};
use std::{fmt::Debug, hint::black_box, sync::Arc, time::Instant};

struct Settings {
    case: String,
    phase: String,
    workers: u32,
    steps: usize,
    particles: usize,
    paths: u64,
    warmups: usize,
    repeats: usize,
}

impl Settings {
    fn policy(&self) -> ExecutionPolicy {
        ExecutionPolicy::new(self.workers, Some(128)).unwrap()
    }
    fn particles(&self) -> LsvParticleConfig {
        LsvParticleConfig::new(self.particles, 8401, 0.7, 2.0, true).unwrap()
    }
    fn times(&self) -> Vec<f64> {
        (0..=self.steps)
            .map(|i| i as f64 / self.steps as f64)
            .collect()
    }
    fn engine(&self) -> EngineConfig {
        EngineConfig::PseudoMonteCarlo(
            PseudoMcConfig::new(971, self.paths, VarianceReduction::new(true, true)).unwrap(),
        )
    }
    fn target(&self, sigma: f64, quotes: bool) -> HullWhiteLsvTarget {
        if quotes {
            HullWhiteLsvTarget::from_market_iv(
                MarketIvSurface::new(vec![0.5, 1.0], vec![-0.5, 0.0, 0.5], vec![sigma; 6]).unwrap(),
                self.times(),
                vec![-0.35, 0.0, 0.35],
                1e-8,
                4.0,
            )
            .unwrap()
        } else {
            HullWhiteLsvTarget::flat(sigma, self.times(), vec![-0.35, 0.0, 0.35], 1e-8, 4.0)
                .unwrap()
        }
    }
    fn request(&self, grid: &LocalVarianceGrid) -> PricingRequest {
        let mut v: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/v1/pricing_request.golden.json"
        )))
        .unwrap();
        v["model"] = json!({"type":"local_volatility","local_variance_grid":{
            "time_nodes":grid.time_nodes(),"log_forward_moneyness_nodes":grid.log_moneyness_nodes(),
            "shape":[grid.time_nodes().len(),grid.log_moneyness_nodes().len()],"values":grid.values(),
            "floor":grid.floor(),"cap":grid.cap()}});
        v["market"]["discrete_dividends"] = json!([
            {"event_id":1,"ex_time":0.5,"quote":{"type":"fixed_cash_and_proportional","fixed_cash":2.0,"beta":0.02}},
            {"event_id":2,"ex_time":1.5,"quote":{"type":"fixed_cash","amount":1.0}}]);
        v["engine"] = json!({"type":"pseudo_monte_carlo","master_seed":971,
            "independent_sampling_units":self.paths,"variance_reduction":{"antithetic":true,"brownian_bridge":true}});
        parse_request_json(&serde_json::to_vec(&v).unwrap(), JsonLimits::DEFAULT).unwrap()
    }
}

fn measure<P, R: Debug>(
    s: &Settings,
    build: impl Fn() -> P,
    identity: impl Fn(&P) -> String,
    evaluate: impl Fn(&P) -> R,
) -> Value {
    let mut samples = Vec::new();
    let mut checksum = None;
    let mut record = |elapsed: u128, signature: String, iteration: usize| {
        let hash = blake3::hash(signature.as_bytes()).to_hex().to_string();
        if let Some(ref expected) = checksum {
            assert_eq!(expected, &hash);
        }
        checksum = Some(hash);
        if iteration >= s.warmups {
            samples.push(elapsed);
        }
    };
    if s.phase == "compile" {
        for i in 0..s.warmups + s.repeats {
            let start = Instant::now();
            let plan = black_box(build());
            let elapsed = start.elapsed().as_nanos();
            record(elapsed, identity(&plan), i);
        }
    } else {
        let plan = build();
        for i in 0..s.warmups + s.repeats {
            let start = Instant::now();
            let result = black_box(evaluate(&plan));
            let elapsed = start.elapsed().as_nanos();
            record(elapsed, format!("{result:?}"), i);
        }
    }
    json!({"case":s.case,"phase":s.phase,"workers":s.workers,"steps":s.steps,
        "particles":s.particles,"independent_units":s.paths,"warmups":s.warmups,
        "repeats":s.repeats,"operation_ns":samples,"checksum":checksum})
}

fn markov<F: BergomiDynamics>(s: &Settings, request: &PricingRequest, factor: F) -> Value {
    let build =
        || BergomiLsvPricingPlan::compile(request, factor, s.particles(), s.policy()).unwrap();
    let identity = |p: &BergomiLsvPricingPlan<F>| format!("{:?}", p.plan_fingerprint());
    if s.phase == "aad" {
        measure(s, build, identity, |p| {
            p.evaluate_local_variance_risk().unwrap()
        })
    } else {
        measure(s, build, identity, |p| p.evaluate().unwrap())
    }
}

fn rates() -> HullWhite1Factor {
    HullWhite1Factor::new(0.13, vec![0.0, 0.37], vec![0.006, 0.008]).unwrap()
}
fn factor1() -> Bergomi1Factor {
    Bergomi1Factor::new(0.7, 0.3, -0.35).unwrap()
}
fn factor2() -> Bergomi2Factor {
    Bergomi2Factor::new([0.4, 2.0], 0.3, 0.35, [-0.35, -0.2], 0.25).unwrap()
}
fn rough() -> RoughBergomi {
    RoughBergomi::new(0.16, 0.5, -0.35).unwrap()
}
fn model(g: &LocalVarianceGrid) -> ModelSpec {
    ModelSpec::LocalVolatility(
        LocalVolatilitySpec::from_explicit_grid(
            g.time_nodes().to_vec(),
            g.log_moneyness_nodes().to_vec(),
            g.values().to_vec(),
            g.floor(),
            g.cap(),
        )
        .unwrap(),
    )
}
fn date(s: &str) -> Date {
    s.parse().unwrap()
}
fn curve(id: u32, rate: f64) -> Arc<LogLinearDiscountCurve> {
    Arc::new(
        LogLinearDiscountCurve::new(
            CurveId::new(id),
            vec![0.0, 2.0],
            vec![1.0, (-2.0 * rate).exp()],
        )
        .unwrap(),
    )
}
fn market(i: u32) -> EquityMarket {
    EquityMarket::new(
        CurrencyId::new(1),
        EquityForward::with_discrete_dividends(
            UnderlyingId::new(i),
            PositiveF64::new(100.0, "spot").unwrap(),
            curve(1, 0.025),
            curve(10 + i, 0.01),
            vec![
                DividendEvent::new(
                    EventId::new(i),
                    0.5,
                    DividendQuote::fixed_cash_and_proportional(2.0, 0.02, EventId::new(i)).unwrap(),
                )
                .unwrap(),
            ],
        )
        .unwrap(),
    )
}
fn correlation(n: usize, rho: f64) -> CorrelationTermStructure {
    CorrelationTermStructure::new(
        (1..=n as u32).map(UnderlyingId::new).collect(),
        vec![(
            date("2026-01-01"),
            (0..n)
                .map(|i| (0..n).map(|j| if i == j { 1.0 } else { rho }).collect())
                .collect(),
        )],
        CorrelationToleranceConfig {
            symmetry_abs_tol: 1e-12,
            diagonal_abs_tol: 1e-12,
            psd_abs_tol: 1e-12,
            psd_rel_tol: 1e-12,
            zero_pivot_abs_tol: 1e-12,
            zero_pivot_rel_tol: 1e-12,
        },
    )
    .unwrap()
}

fn multi(s: &Settings) -> MultiAssetPricingPlan {
    let local = s.case == "local_correlation_hw";
    let n = if local { 2 } else { 3 };
    let target = s.target(0.28, false);
    let configs = if local {
        vec![
            Some(
                MultiAssetLsvConfig {
                    factor: factor1(),
                    particles: s.particles(),
                }
                .into(),
            ),
            None,
        ]
    } else {
        vec![
            Some(
                MultiAssetLsvConfig {
                    factor: factor1(),
                    particles: s.particles(),
                }
                .into(),
            ),
            Some(
                MultiAssetLsv2FactorConfig {
                    factor: factor2(),
                    particles: s.particles(),
                }
                .into(),
            ),
            Some(
                MultiAssetRoughLsvConfig {
                    factor: rough(),
                    particles: s.particles(),
                }
                .into(),
            ),
        ]
    };
    let models = configs
        .iter()
        .map(|c| {
            if c.is_some() {
                model(target.grid())
            } else {
                ModelSpec::BlackScholes(BlackScholesSpec::new(0.3).unwrap())
            }
        })
        .collect();
    let targets = configs
        .iter()
        .map(|c| c.as_ref().map(|_| target.clone()))
        .collect();
    let count = n + configs
        .iter()
        .flatten()
        .map(MultiAssetBergomiLsvConfig::factor_count)
        .sum::<usize>();
    let hw = MultiAssetHullWhiteConfig {
        rate_model: rates(),
        rate_correlations: vec![0.02; count],
        lsv_targets: targets,
    };
    let product = MultiAssetProduct::basket(
        CurrencyId::new(1),
        (1..=n as u32)
            .map(|i| BasketComponent {
                underlying: UnderlyingId::new(i),
                weight: 1.0 / n as f64,
                scale: 1.0,
            })
            .collect(),
        OptionSide::Call,
        96.0,
        1.0,
        date("2027-01-01"),
        date("2027-01-01"),
        Some(CompactC2Smoothing::new(3.0).unwrap()),
    )
    .unwrap();
    let markets = (1..=n as u32).map(market).collect();
    if local {
        let basket = s.target(0.235, false);
        MultiAssetPricingPlan::compile_with_joint_local_correlation(
            date("2026-01-01"),
            product,
            markets,
            models,
            correlation(n, -0.3),
            s.engine(),
            s.policy(),
            1.0 / s.steps as f64,
            configs,
            None,
            Some(hw),
            LocalCorrelationConfig {
                basket_weights: vec![0.5; 2],
                target: basket.grid().clone(),
                second_correlation: correlation(n, 0.95),
                particles: s.particles(),
                feasibility: LocalCorrelationFeasibility::ProjectAndReport,
                minimum_variance_span: 1e-12,
            },
            LocalCorrelationExtensions {
                second_driver_correlations: None,
                hull_white_target: Some(basket),
            },
        )
        .unwrap()
    } else {
        MultiAssetPricingPlan::compile_with_hull_white(
            date("2026-01-01"),
            product,
            markets,
            models,
            correlation(n, 0.2),
            s.engine(),
            s.policy(),
            1.0 / s.steps as f64,
            configs,
            None,
            hw,
        )
        .unwrap()
    }
}

fn main() {
    let a: Vec<_> = std::env::args().collect();
    assert_eq!(
        a.len(),
        9,
        "case phase workers steps particles units warmups repeats"
    );
    let s = Settings {
        case: a[1].clone(),
        phase: a[2].clone(),
        workers: a[3].parse().unwrap(),
        steps: a[4].parse().unwrap(),
        particles: a[5].parse().unwrap(),
        paths: a[6].parse().unwrap(),
        warmups: a[7].parse().unwrap(),
        repeats: a[8].parse().unwrap(),
    };
    assert!(s.steps >= 2 && s.steps.is_multiple_of(2) && s.repeats > 0);
    assert!(["compile", "price", "aad", "aad_vegakt"].contains(&s.phase.as_str()));
    assert!(s.phase != "aad_vegakt" || s.case.starts_with("hw_"));
    let target = s.target(0.28, s.phase == "aad_vegakt");
    let request = s.request(target.grid());
    let report = match s.case.as_str() {
        "bergomi_1f" => markov(&s, &request, factor1()),
        "bergomi_2f" => markov(&s, &request, factor2()),
        "rough" => {
            let build = || {
                RoughBergomiLsvPricingPlan::compile(&request, rough(), s.particles(), s.policy())
                    .unwrap()
            };
            let identity = |p: &RoughBergomiLsvPricingPlan| format!("{:?}", p.plan_fingerprint());
            if s.phase == "aad" {
                measure(&s, build, identity, |p| {
                    p.evaluate_local_variance_risk().unwrap()
                })
            } else {
                measure(&s, build, identity, |p| p.evaluate().unwrap())
            }
        }
        "hw_1f" | "hw_2f" | "hw_rough" => {
            let build = || {
                let corr = HybridCorrelation::new(-0.35, 0.1, 0.02).unwrap();
                match s.case.as_str() {
                    "hw_1f" => HullWhiteEquityPricingPlan::compile_lsv(
                        &request,
                        &target,
                        factor1(),
                        rates(),
                        corr,
                        s.particles(),
                        s.policy(),
                    ),
                    "hw_2f" => HullWhiteEquityPricingPlan::compile_lsv_two_factor(
                        &request,
                        &target,
                        factor2(),
                        rates(),
                        0.1,
                        [0.02, 0.03],
                        s.particles(),
                        s.policy(),
                    ),
                    _ => HullWhiteEquityPricingPlan::compile_rough_lsv(
                        &request,
                        &target,
                        rough(),
                        rates(),
                        corr,
                        s.particles(),
                        s.policy(),
                    ),
                }
                .unwrap()
            };
            let identity = |p: &HullWhiteEquityPricingPlan| format!("{:?}", p.plan_fingerprint());
            if s.phase.starts_with("aad") {
                measure(&s, build, identity, |p| p.evaluate_aad().unwrap())
            } else {
                measure(&s, build, identity, |p| p.evaluate().unwrap())
            }
        }
        "multi_hw_mixed" | "local_correlation_hw" => {
            let build = || multi(&s);
            let identity = |p: &MultiAssetPricingPlan| p.fingerprint().to_owned();
            if s.phase == "aad" {
                measure(&s, build, identity, |p| {
                    p.evaluate_aad(MultiAssetRiskConfig::default()).unwrap()
                })
            } else {
                measure(&s, build, identity, |p| p.evaluate().unwrap())
            }
        }
        _ => panic!("unknown case"),
    };
    println!("{report}");
}
