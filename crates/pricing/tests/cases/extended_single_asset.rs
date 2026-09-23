use super::*;
use pricing::hull_white::HullWhiteEquityPricingPlan;
use pricing::lsv::{BergomiLsvPricingPlan, RoughBergomiLsvPricingPlan};
use pricing::mc::hull_white::CalibratedHullWhiteLsv;
use pricing::mc::lsv::{LsvConditionalMoments, LsvLeverageSurface};
use pricing::models::{
    Bergomi1Factor, Bergomi2Factor, HullWhite1Factor, HybridCorrelation, RoughBergomi,
};

#[derive(Clone, Copy, Debug, Serialize)]
enum Factor {
    One,
    Two,
    Rough,
}

fn one() -> Bergomi1Factor {
    Bergomi1Factor::new(2.0, 0.7, -0.5).unwrap()
}
fn two() -> Bergomi2Factor {
    Bergomi2Factor::new([3.0, 0.3], 0.5, 0.35, [-0.55, -0.2], 0.25).unwrap()
}
fn rough() -> RoughBergomi {
    RoughBergomi::new(0.12, 0.6, -0.5).unwrap()
}

fn zero_factor_price(
    factor: Factor,
    req: &PricingRequest,
    res: Resolution,
) -> pricing::lsv::LsvPrice {
    match factor {
        Factor::One => BergomiLsvPricingPlan::compile(
            req,
            Bergomi1Factor::new(2.0, 0.0, -0.5).unwrap(),
            res.particles(1709),
            res.policy(),
        )
        .unwrap()
        .evaluate(),
        Factor::Two => BergomiLsvPricingPlan::compile(
            req,
            Bergomi2Factor::new([3.0, 0.3], 0.0, 0.35, [-0.55, -0.2], 0.25).unwrap(),
            res.particles(1709),
            res.policy(),
        )
        .unwrap()
        .evaluate(),
        Factor::Rough => RoughBergomiLsvPricingPlan::compile(
            req,
            RoughBergomi::new(0.12, 0.0, -0.5).unwrap(),
            res.particles(1709),
            res.policy(),
        )
        .unwrap()
        .evaluate(),
    }
    .unwrap()
}

#[test]
fn zero_vol_of_vol_preserves_continuous_carry_and_escrowed_cash() {
    let res = Resolution {
        particles: 128,
        steps: 4,
        points: 4096,
        scrambles: 4,
        ..FINE
    };
    let expiry = "2028-01-01";
    let t = time(expiry);
    let target = Smile::Flat.target(expiry, res);
    for factor in [Factor::One, Factor::Two, Factor::Rough] {
        for cash in [Cash::None, Cash::Midpoint, Cash::Expiry] {
            let req = request(expiry, 0.0, cash, model_json(target.grid()), res, 1709);
            let p = zero_factor_price(factor, &req, res);
            let forward = cash.affine(t).0 * 100.0 * ((RATE - DIVIDEND_RATE) * t).exp();
            let expected = black(forward, forward, (-RATE * t).exp(), t, 0.2).0;
            assert!(p.standard_error < 0.01);
            assert!(
                (p.value - expected).abs() < 5.0 * p.standard_error + 0.001,
                "{factor:?}/{cash:?}: {} vs {expected}",
                p.value
            );
        }
    }
}

#[test]
fn carried_payoff_variance_adjoint_matches_repriced_bump() {
    let res = Resolution {
        particles: 128,
        steps: 4,
        points: 2048,
        scrambles: 4,
        ..FINE
    };
    let expiry = "2028-01-01";
    let target = Smile::Flat.target(expiry, res);
    let particles = LsvParticleConfig::new(128, 1709, res.bandwidth, 20.0, true).unwrap();
    let req = request(
        expiry,
        0.0,
        Cash::Midpoint,
        model_json(target.grid()),
        res,
        1709,
    );
    for factor in [Factor::One, Factor::Two, Factor::Rough] {
        let risk = match factor {
            Factor::One => BergomiLsvPricingPlan::compile(
                &req,
                Bergomi1Factor::new(2.0, 0.0, -0.5).unwrap(),
                particles.clone(),
                res.policy(),
            )
            .unwrap()
            .evaluate_local_variance_risk(),
            Factor::Two => BergomiLsvPricingPlan::compile(
                &req,
                Bergomi2Factor::new([3.0, 0.3], 0.0, 0.35, [-0.55, -0.2], 0.25).unwrap(),
                particles.clone(),
                res.policy(),
            )
            .unwrap()
            .evaluate_local_variance_risk(),
            Factor::Rough => RoughBergomiLsvPricingPlan::compile(
                &req,
                RoughBergomi::new(0.12, 0.0, -0.5).unwrap(),
                particles.clone(),
                res.policy(),
            )
            .unwrap()
            .evaluate_local_variance_risk(),
        }
        .unwrap();
        let h = 1e-6;
        let bump = |shift| {
            let g = target.grid();
            let grid = LocalVarianceGrid::new(
                g.time_nodes().to_vec(),
                g.log_moneyness_nodes().to_vec(),
                g.values().iter().map(|v| v + shift).collect(),
                g.floor(),
                g.cap(),
            )
            .unwrap();
            zero_factor_price(
                factor,
                &request(expiry, 0.0, Cash::Midpoint, model_json(&grid), res, 1709),
                res,
            )
            .value
        };
        let fd = (bump(h) - bump(-h)) / (2.0 * h);
        let aad = risk.node_adjoints.iter().sum::<f64>();
        assert!(
            (aad - fd).abs() < 1e-5 * (1.0 + fd.abs()),
            "{factor:?}: {aad} vs {fd}"
        );
        let p = zero_factor_price(factor, &req, res);
        assert!((risk.price.value - p.value).abs() < 1e-12);
    }
}
fn rates(t: f64) -> HullWhite1Factor {
    HullWhite1Factor::new(0.2, vec![0.0, 0.4 * t], vec![0.012, 0.015]).unwrap()
}

#[derive(Clone, Copy, Debug, Serialize)]
struct Case {
    factor: Factor,
    smile: Smile,
    expiry: &'static str,
    cash: Cash,
    hw: bool,
}

fn panel(case: Case, res: Resolution) -> Vec<QuoteReport> {
    panel_at(case, res, &STRIKES)
}

fn panel_at(case: Case, res: Resolution, strikes: &[f64]) -> Vec<QuoteReport> {
    let t = time(case.expiry);
    let target = case.smile.target(case.expiry, res);
    strikes
        .iter()
        .copied()
        .map(|x| {
            let runs = SEEDS
                .into_iter()
                .map(|s| {
                    let seed = s + res.seed_offset;
                    let req = request(
                        case.expiry,
                        x,
                        case.cash,
                        model_json(target.grid()),
                        res,
                        seed,
                    );
                    let (value, conditional_se) = if case.hw {
                        let corr = HybridCorrelation::new(-0.5, 0.25, -0.1).unwrap();
                        let plan = match case.factor {
                            Factor::One => HullWhiteEquityPricingPlan::compile_lsv(
                                &req,
                                &target,
                                case.one(),
                                rates(t),
                                corr,
                                res.particles(seed),
                                res.policy(),
                            ),
                            Factor::Two => HullWhiteEquityPricingPlan::compile_lsv_two_factor(
                                &req,
                                &target,
                                case.two(),
                                rates(t),
                                0.25,
                                [-0.1, 0.02],
                                res.particles(seed),
                                res.policy(),
                            ),
                            Factor::Rough => HullWhiteEquityPricingPlan::compile_rough_lsv(
                                &req,
                                &target,
                                case.rough(),
                                rates(t),
                                corr,
                                res.particles(seed),
                                res.policy(),
                            ),
                        }
                        .unwrap();
                        record_hw(case, res, seed, x, plan.calibration().unwrap());
                        let p = plan.evaluate().unwrap();
                        (p.value, p.standard_error)
                    } else {
                        let p = match case.factor {
                            Factor::One => {
                                let plan = BergomiLsvPricingPlan::compile(
                                    &req,
                                    case.one(),
                                    res.particles(seed),
                                    res.policy(),
                                )
                                .unwrap();
                                record_lsv(
                                    case,
                                    res,
                                    seed,
                                    x,
                                    plan.calibration().surface(),
                                    plan.calibration().conditional_moments(),
                                );
                                plan.evaluate()
                            }
                            Factor::Two => {
                                let plan = BergomiLsvPricingPlan::compile(
                                    &req,
                                    case.two(),
                                    res.particles(seed),
                                    res.policy(),
                                )
                                .unwrap();
                                record_lsv(
                                    case,
                                    res,
                                    seed,
                                    x,
                                    plan.calibration().surface(),
                                    plan.calibration().conditional_moments(),
                                );
                                plan.evaluate()
                            }
                            Factor::Rough => {
                                let plan = RoughBergomiLsvPricingPlan::compile(
                                    &req,
                                    case.rough(),
                                    res.particles(seed),
                                    res.policy(),
                                )
                                .unwrap();
                                record_lsv(
                                    case,
                                    res,
                                    seed,
                                    x,
                                    plan.calibration().surface(),
                                    plan.calibration().conditional_moments(),
                                );
                                plan.evaluate()
                            }
                        }
                        .unwrap();
                        (p.value, p.standard_error)
                    };
                    PriceRun {
                        calibration_seed: seed,
                        pricing_seed: res.pricing_seed(seed),
                        value,
                        conditional_se,
                    }
                })
                .collect();
            let forward = case.cash.affine(t).0 * 100.0 * ((RATE - DIVIDEND_RATE) * t).exp();
            summarize(runs, forward, t, x, case.smile.iv(t, x))
        })
        .collect()
}

#[test]
#[ignore = "release-mode multi-seed extended-model acceptance"]
fn deterministic_lsv_smile_repricing() {
    for case in [
        Case {
            factor: Factor::One,
            smile: Smile::Skew,
            expiry: "2028-01-01",
            cash: Cash::None,
            hw: false,
        },
        Case {
            factor: Factor::Two,
            smile: Smile::Flat,
            expiry: "2026-07-02",
            cash: Cash::None,
            hw: false,
        },
        Case {
            factor: Factor::Two,
            smile: Smile::Skew,
            expiry: "2028-01-01",
            cash: Cash::Midpoint,
            hw: false,
        },
        Case {
            factor: Factor::Rough,
            smile: Smile::Flat,
            expiry: "2026-07-02",
            cash: Cash::None,
            hw: false,
        },
        Case {
            factor: Factor::Rough,
            smile: Smile::Skew,
            expiry: "2028-01-01",
            cash: Cash::Midpoint,
            hw: false,
        },
    ] {
        report(json!(case), FINE, &panel(case, FINE), LSV_BUDGET);
    }
}

#[test]
#[ignore = "release-mode multi-seed extended-model acceptance"]
fn stochastic_rate_lsv_smile_repricing() {
    for factor in [Factor::One, Factor::Two, Factor::Rough] {
        let case = Case {
            factor,
            smile: Smile::Skew,
            expiry: "2028-01-01",
            cash: Cash::Expiry,
            hw: true,
        };
        report(json!(case), FINE, &panel(case, FINE), LSV_BUDGET);
    }
}

// Independent covariance-kernel quadrature. This does not call the HW
// transition's covariance builder, path engine or calibration routines.
fn integrated_variance(t: f64, a: f64, rho: f64, sigma: f64) -> f64 {
    let integrate = |lo: f64, hi: f64, rate_vol: f64| {
        let n = 1024;
        let h = (hi - lo) / n as f64;
        (0..=n)
            .map(|i| {
                let u = lo + i as f64 * h;
                let b = if a == 0.0 {
                    t - u
                } else {
                    -(-a * (t - u)).exp_m1() / a
                };
                let weight = if i == 0 || i == n {
                    1.0
                } else if i % 2 == 0 {
                    2.0
                } else {
                    4.0
                };
                weight * (rate_vol.powi(2) * b * b + 2.0 * rho * sigma * rate_vol * b)
            })
            .sum::<f64>()
            * h
            / 3.0
    };
    sigma * sigma * t + integrate(0.0, 0.4 * t, 0.012) + integrate(0.4 * t, t, 0.015)
}

#[test]
fn rate_kernel_quadrature_matches_zero_reversion_polynomial() {
    let t: f64 = 2.0;
    let rho = -0.4;
    let sigma = 0.2;
    let mut expected = sigma * sigma * t;
    for (lo, hi, v) in [(0.0, 0.4 * t, 0.012), (0.4 * t, t, 0.015)] {
        expected += v * v * ((t - lo).powi(3) - (t - hi).powi(3)) / 3.0
            + rho * sigma * v * ((t - lo).powi(2) - (t - hi).powi(2));
    }
    assert!((integrated_variance(t, 0.0, rho, sigma) - expected).abs() < 1e-14);
}

#[test]
#[ignore = "release-mode multi-seed extended-model acceptance"]
fn gaussian_hull_white_independent_price_reference() {
    let expiry = "2028-01-01";
    let t = time(expiry);
    // Only eight steps are needed: the constant equity-volatility HW path
    // uses exact joint Gaussian transitions, including rate-volatility knots.
    let res = Resolution {
        steps: 8,
        points: 16_384,
        ..FINE
    };
    for a in [0.0, 0.2] {
        for rho in [-0.4, 0.0, 0.4] {
            let target_iv = (integrated_variance(t, a, rho, 0.2) / t).sqrt();
            let quotes: Vec<_> = STRIKES
                .into_iter()
                .map(|x| {
                    let runs = SEEDS
                        .into_iter()
                        .map(|seed| {
                            let req = request(
                                expiry,
                                x,
                                Cash::Expiry,
                                json!({"type":"black_scholes","volatility":0.2}),
                                res,
                                seed,
                            );
                            let model =
                                HullWhite1Factor::new(a, vec![0.0, 0.4 * t], vec![0.012, 0.015])
                                    .unwrap();
                            let p = HullWhiteEquityPricingPlan::compile_bs(
                                &req,
                                model,
                                rho,
                                t / res.steps as f64,
                                res.policy(),
                            )
                            .unwrap()
                            .evaluate()
                            .unwrap();
                            PriceRun {
                                calibration_seed: seed,
                                pricing_seed: res.pricing_seed(seed),
                                value: p.value,
                                conditional_se: p.standard_error,
                            }
                        })
                        .collect();
                    summarize(
                        runs,
                        0.97 * 100.0 * ((RATE - DIVIDEND_RATE) * t).exp(),
                        t,
                        x,
                        target_iv,
                    )
                })
                .collect();
            report(
                json!({"model":"BS-HW","expiry":expiry,"mean_reversion":a,"equity_rate_correlation":rho,"cash":Cash::Expiry}),
                res,
                &quotes,
                GAUSSIAN_BUDGET,
            );
        }
    }
}

fn refinement_case(case: Case) {
    let base = panel(case, FINE);
    report(
        json!({"refinement":"base","case":case}),
        FINE,
        &base,
        LSV_BUDGET,
    );
    for (axis, res) in refinements() {
        let refined = panel(case, res);
        report(
            json!({"refinement":axis,"case":case}),
            res,
            &refined,
            LSV_BUDGET,
        );
        compare_refinement(json!(case), axis, &base, &refined);
    }
}

fn stress_case(factor: Factor, hw: bool) {
    // Strong tails can breach a fixed-cash dividend's positive-spot domain.
    // Keep that domain check intact; isolate the smile/vol-of-vol price study
    // from escrowed-reserve feasibility (covered by the original/refinement cases).
    let case = Case {
        factor,
        smile: Smile::Stress,
        expiry: "2029-01-01",
        cash: Cash::None,
        hw,
    };
    let res = Resolution {
        particles: 131_072,
        points: 16_384,
        ..STRESS
    };
    report_at(
        json!({"stress":true,"case":case}),
        res,
        &panel_at(case, res, &WINGS),
        &WINGS,
        LSV_BUDGET,
    );
}

// Separate test entries prevent a failure in one model from hiding the rest
// and allow an expensive failing case to be rerun by name.
macro_rules! model_cases {
    ($refinement:ident, $stress:ident, $factor:expr, $hw:expr) => {
        #[test]
        #[ignore = "release-mode multi-seed extended-model acceptance"]
        fn $refinement() {
            refinement_case(Case {
                factor: $factor,
                smile: Smile::Skew,
                expiry: "2027-01-01",
                cash: if $hw { Cash::Expiry } else { Cash::None },
                hw: $hw,
            });
        }
        #[test]
        #[ignore = "release-mode multi-seed extended-model acceptance"]
        fn $stress() {
            stress_case($factor, $hw);
        }
    };
}
model_cases!(
    one_factor_refinement_consistency,
    one_factor_stress_repricing,
    Factor::One,
    false
);
model_cases!(
    two_factor_refinement_consistency,
    two_factor_stress_repricing,
    Factor::Two,
    false
);
model_cases!(
    rough_refinement_consistency,
    rough_stress_repricing,
    Factor::Rough,
    false
);
model_cases!(
    one_factor_hw_refinement_consistency,
    one_factor_hw_stress_repricing,
    Factor::One,
    true
);
model_cases!(
    two_factor_hw_refinement_consistency,
    two_factor_hw_stress_repricing,
    Factor::Two,
    true
);
model_cases!(
    rough_hw_refinement_consistency,
    rough_hw_stress_repricing,
    Factor::Rough,
    true
);

impl Case {
    fn one(self) -> Bergomi1Factor {
        if matches!(self.smile, Smile::Stress) {
            Bergomi1Factor::new(2.0, 1.2, -0.5).unwrap()
        } else {
            one()
        }
    }
    fn two(self) -> Bergomi2Factor {
        if matches!(self.smile, Smile::Stress) {
            Bergomi2Factor::new([3.0, 0.3], 0.9, 0.35, [-0.55, -0.2], 0.25).unwrap()
        } else {
            two()
        }
    }
    fn rough(self) -> RoughBergomi {
        if matches!(self.smile, Smile::Stress) {
            RoughBergomi::new(0.12, 1.0, -0.5).unwrap()
        } else {
            rough()
        }
    }
}

fn record_lsv(
    case: Case,
    res: Resolution,
    seed: u64,
    x: f64,
    surface: &LsvLeverageSurface,
    moments: &[LsvConditionalMoments],
) {
    let m = surface.log_nodes().len();
    let start = moments.len() - m;
    let nodes: Vec<_> = bracket(surface.log_nodes(), x).map(|j| {
        let d = &moments[start+j];
        json!({"x":surface.log_nodes()[j],"ess":d.effective_samples,"fallback":d.extrapolated,"source_node":d.source_node})
    }).collect();
    println!(
        "EXTENDED_CALIBRATION {}",
        json!({"case":case,"resolution":res,"seed":seed,"x":x,
        "diagnostic_scope":"node","global_fallback_nodes":moments.iter().filter(|d| d.extrapolated).count(),
        "total_nodes":moments.len(),"terminal_quote_nodes":nodes})
    );
    for j in bracket(surface.log_nodes(), x) {
        let d = &moments[start + j];
        assert!(
            !d.extrapolated && d.effective_samples.is_finite() && d.effective_samples >= 100.0,
            "{case:?}, seed={seed}, x={x}: unsupported terminal node {d:?}"
        );
    }
}

fn record_hw(case: Case, res: Resolution, seed: u64, x: f64, cal: &CalibratedHullWhiteLsv) {
    // The public HW snapshot exposes row minima, not per-node ESS/fallback.
    // Report that scope explicitly; it cannot certify support at a quote node.
    let rows = &cal.diagnostics;
    let last = rows.last().unwrap();
    println!(
        "EXTENDED_CALIBRATION {}",
        json!({"case":case,"resolution":res,"seed":seed,"x":x,
        "diagnostic_scope":"row","global_fallback_nodes":rows.iter().map(|d|d.fallback_nodes).sum::<usize>(),
        "total_nodes":cal.conditional_second_moments.len(),
        "minimum_row_ess":rows.iter().map(|d|d.minimum_effective_samples).fold(f64::INFINITY,f64::min),
        "terminal_minimum_ess":last.minimum_effective_samples,"terminal_fallback_nodes":last.fallback_nodes,
        "maximum_rate_correction":rows.iter().map(|d|d.maximum_rate_correction).fold(0.0,f64::max)})
    );
    assert!(rows.iter().all(|d| d.minimum_effective_samples.is_finite()
        && d.minimum_effective_samples >= 0.0
        && d.maximum_rate_correction.is_finite()));
    assert!(
        cal.conditional_second_moments
            .iter()
            .all(|v| v.is_finite() && *v > 0.0)
    );
}
