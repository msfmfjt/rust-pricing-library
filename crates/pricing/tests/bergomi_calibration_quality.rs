//! Smile-wide acceptance for the finite one-factor Bergomi particle algorithm.
//! Heavy tests are ignored by default but run explicitly in CI in release mode.
//! No model algorithm, bandwidth selection or quote repair is performed here.

use pricing::analytical::{BlackScholesOracleInputs, evaluate};
use pricing::market::{ImpliedVarianceSurface, LocalVarianceGrid, MarketIvSurface};
use pricing::mc::lsv::{BergomiLsvPlan, LsvParticleConfig, calibrate_bergomi_lsv};
use pricing::mc::{
    BrownianBridgePlan, DeterministicExecutor, DeterministicStatistics, ExecutionPolicy,
    LocalVolLogEulerPlan, LocalVolTimeGrid, RqmcConfig, RqmcPlan, VarianceReduction,
    inverse_standard_normal,
};
use pricing::models::Bergomi1Factor;
use pricing::product::OptionSide;
use serde::Serialize;

const SPOT: f64 = 100.0;
const BP: f64 = 1e-4;
const MATURITIES: [f64; 3] = [0.25, 0.5, 1.0];
const LOG_STRIKES: [f64; 5] = [-0.2, -0.1, 0.0, 0.1, 0.2];
const CALIBRATION_SEEDS: [u64; 4] = [42, 137, 711, 2027];
const QUOTES: usize = MATURITIES.len() * LOG_STRIKES.len();

#[derive(Clone, Copy, Debug, Serialize)]
struct Settings {
    particles: usize,
    bandwidth: f64,
    steps: usize,
    spatial_intervals: usize,
    points_per_scramble: u64,
    scrambles: u32,
}

const ACCEPTANCE: Settings = Settings {
    particles: 32_768,
    bandwidth: 0.02,
    steps: 128,
    spatial_intervals: 160,
    points_per_scramble: 8192,
    scrambles: 8,
};

#[derive(Clone, Copy, Debug)]
enum Smile {
    Flat,
    Skew,
}

impl Smile {
    fn name(self) -> &'static str {
        match self {
            Self::Flat => "flat_20_percent",
            Self::Skew => "skew_and_term_structure",
        }
    }

    fn surface(self) -> MarketIvSurface {
        // Evaluation includes off-quote strikes; the grid is not a list of ATM tests.
        let xs = vec![-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0];
        let quotes = MATURITIES
            .iter()
            .flat_map(|&t| {
                xs.iter().map(move |&x| match self {
                    Self::Flat => 0.2,
                    Self::Skew => ((0.04 + 0.004 * t) * (1.0 - 0.25 * x + 0.1 * x * x)).sqrt(),
                })
            })
            .collect();
        MarketIvSurface::new(MATURITIES.to_vec(), xs, quotes).unwrap()
    }
}

fn target(surface: &MarketIvSurface, settings: Settings) -> LocalVarianceGrid {
    assert_eq!(settings.steps % 4, 0);
    let times = (0..=settings.steps)
        .map(|i| i as f64 / settings.steps as f64)
        .collect();
    let xs = (0..=settings.spatial_intervals)
        .map(|i| -1.0 + 2.0 * i as f64 / settings.spatial_intervals as f64)
        .collect();
    let grid = LocalVarianceGrid::from_surface(surface, times, xs, 1e-8, 4.0).unwrap();
    assert!(
        grid.repairs().is_empty(),
        "acceptance targets must not be repaired"
    );
    grid
}

fn side(x: f64) -> OptionSide {
    if x < 0.0 {
        OptionSide::Put
    } else {
        OptionSide::Call
    }
}

fn black(t: f64, x: f64, sigma: f64) -> (f64, f64) {
    let result = evaluate(BlackScholesOracleInputs {
        side: side(x),
        spot: SPOT,
        strike: SPOT * x.exp(),
        notional: 1.0,
        discount: 1.0,
        dividend_discount: 1.0,
        volatility: sigma,
        time: t,
    })
    .unwrap();
    (result.price, result.vega)
}

// Bracketed test-only inversion. Invalid/no-time-value prices FAIL; no clipping
// to intrinsic, replacing NaNs, dropping quotes or extrapolating an IV result.
fn implied_volatility(price: f64, t: f64, x: f64) -> Result<f64, &'static str> {
    if !price.is_finite() || !t.is_finite() || t <= 0.0 || !x.is_finite() {
        return Err("nonfinite price/coordinate or nonpositive maturity");
    }
    let upper_bound = if x < 0.0 { SPOT * x.exp() } else { SPOT };
    if price <= 0.0 || price >= upper_bound {
        return Err("OTM price outside strict no-arbitrage bounds");
    }
    let (mut lo, mut hi) = (0.0, 4.0);
    if price >= black(t, x, hi).0 {
        return Err("price outside volatility bracket");
    }
    for _ in 0..80 {
        let mid = 0.5 * (lo + hi);
        if black(t, x, mid).0 < price {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Ok(0.5 * (lo + hi))
}

#[derive(Clone, Copy, Debug, Serialize)]
struct Estimate {
    value: f64,
    se: f64,
}

fn estimate(values: &[f64]) -> Estimate {
    assert!(values.len() >= 2 && values.iter().all(|v| v.is_finite()));
    let stats = DeterministicStatistics::from_ordered_values_two_pass(values);
    let n = values.len() as f64;
    Estimate {
        value: stats.sum().total() / n,
        se: (stats.moments().sample_variance().unwrap() / n).sqrt(),
    }
}

// Independent complete calibration+pricing runs are the outer sampling units.
// Between-run variance already contains pricing variance: do NOT add it again.
// Use a pricing-only floor to avoid an accidentally tiny four-seed estimate.
fn ensemble(runs: &[Estimate]) -> (Estimate, f64) {
    let mut combined = estimate(&runs.iter().map(|e| e.value).collect::<Vec<_>>());
    let pricing_se = runs.iter().map(|e| e.se * e.se).sum::<f64>().sqrt() / runs.len() as f64;
    combined.se = combined.se.max(pricing_se);
    (combined, pricing_se)
}

#[derive(Debug, Serialize)]
struct Support {
    time: f64,
    minimum_quote_ess: f64,
    quote_fallbacks: usize,
    all_row_fallbacks: usize,
    particle_mean_f: f64,
}

#[derive(Debug, Serialize)]
struct Run {
    calibration_seed: u64,
    pricing_seed: u64,
    // In order: LSV OTM, LV OTM, paired LSV-LV, then one martingale per maturity.
    estimates: Vec<Estimate>,
    support: Vec<Support>,
}

fn price_surface(
    lsv: &BergomiLsvPlan,
    grid: &LocalVarianceGrid,
    settings: Settings,
    pricing_seed: u64,
) -> Vec<Estimate> {
    let time_grid = LocalVolTimeGrid::compile(grid.time_nodes().to_vec(), 1.0).unwrap();
    let lv = LocalVolLogEulerPlan::with_constant_forward(time_grid, SPOT).unwrap();
    let bridge = BrownianBridgePlan::compile(grid.time_nodes().to_vec(), 1).unwrap();
    let dimension = 2 * settings.steps;
    let qmc = RqmcPlan::compile(
        RqmcConfig::new(
            settings.points_per_scramble,
            settings.scrambles,
            pricing_seed,
            VarianceReduction::new(true, true),
        )
        .unwrap(),
        dimension as u32,
    )
    .unwrap();
    let executor = DeterministicExecutor::new(ExecutionPolicy::new(2, Some(256)).unwrap()).unwrap();
    let width = 3 * QUOTES + MATURITIES.len();
    let mut means = Vec::new();
    for scramble in 0..settings.scrambles {
        let stats = executor
            .try_map_reduce_statistics_vector(settings.points_per_scramble, width, |p, out| {
                let normals = (0..dimension)
                    .map(|d| {
                        inverse_standard_normal(qmc.uniform(scramble, p, d as u32).unwrap())
                            .unwrap()
                    })
                    .collect::<Vec<_>>();
                let mut z = Vec::with_capacity(dimension);
                for block in normals.chunks_exact(settings.steps) {
                    z.extend(bridge.apply_one_factor(block).unwrap());
                }
                for sign in [1.0, -1.0] {
                    if sign < 0.0 {
                        z.iter_mut().for_each(|v| *v = -*v);
                    }
                    let path = lsv.evolve_path(SPOT, &z).unwrap();
                    let control = lv.evolve_path(grid, SPOT, &z[..settings.steps]).unwrap();
                    for (r, &t) in MATURITIES.iter().enumerate() {
                        let i = (t * settings.steps as f64) as usize;
                        assert_eq!(grid.time_nodes()[i], t);
                        let f = path.states()[i];
                        let f_lv = control.states()[i];
                        out[3 * QUOTES + r] += 0.5 * f;
                        for (j, &x) in LOG_STRIKES.iter().enumerate() {
                            let strike = SPOT * x.exp();
                            let payoff = |s: f64| {
                                if x < 0.0 {
                                    (strike - s).max(0.0)
                                } else {
                                    (s - strike).max(0.0)
                                }
                            };
                            let q = r * LOG_STRIKES.len() + j;
                            let a = payoff(f);
                            let b = payoff(f_lv);
                            out[q] += 0.5 * a;
                            out[QUOTES + q] += 0.5 * b;
                            out[2 * QUOTES + q] += 0.5 * (a - b);
                        }
                    }
                }
                Ok::<_, std::convert::Infallible>(())
            })
            .unwrap();
        means.push(
            stats
                .iter()
                .map(|s| s.sum().total() / settings.points_per_scramble as f64)
                .collect::<Vec<_>>(),
        );
    }
    // QMC points, antithetic legs and strikes are NOT independent observations.
    (0..width)
        .map(|i| estimate(&means.iter().map(|r| r[i]).collect::<Vec<_>>()))
        .collect()
}

fn run(surface: &MarketIvSurface, settings: Settings, seed: u64) -> Run {
    let grid = target(surface, settings);
    let calibration = calibrate_bergomi_lsv(
        &grid,
        Bergomi1Factor::new(2.0, 0.7, -0.5).unwrap(),
        SPOT,
        LsvParticleConfig::new(settings.particles, seed, settings.bandwidth, 20.0, false).unwrap(),
    )
    .unwrap();
    let xs = grid.log_moneyness_nodes();
    let support = MATURITIES
        .iter()
        .map(|&t| {
            let row = (t * settings.steps as f64) as usize;
            let mut ess = f64::INFINITY;
            let mut fallbacks = 0;
            for x in LOG_STRIKES {
                let j = xs
                    .partition_point(|v| *v <= x)
                    .saturating_sub(1)
                    .min(xs.len() - 2);
                for col in [j, j + 1] {
                    let moment = calibration.conditional_moments()[row * xs.len() + col];
                    ess = ess.min(moment.effective_samples);
                    fallbacks += usize::from(moment.extrapolated);
                }
            }
            Support {
                time: t,
                minimum_quote_ess: ess,
                quote_fallbacks: fallbacks,
                all_row_fallbacks: calibration.diagnostics()[row].extrapolated_nodes,
                particle_mean_f: calibration.diagnostics()[row].particle_mean_f,
            }
        })
        .collect();
    let time_grid = LocalVolTimeGrid::compile(grid.time_nodes().to_vec(), 1.0).unwrap();
    // Different algorithm/domain AND seed from particle calibration. Each full
    // run also has its own pricing scrambles, so the outer means are independent.
    let pricing_seed = seed ^ 0xd1b5_4a32_d192_ed03;
    let estimates = price_surface(
        &calibration.pricing_plan(&time_grid).unwrap(),
        &grid,
        settings,
        pricing_seed,
    );
    Run {
        calibration_seed: seed,
        pricing_seed,
        estimates,
        support,
    }
}

#[derive(Clone, Debug, Serialize)]
struct Node {
    maturity: f64,
    log_strike: f64,
    target_iv: f64,
    model_iv: f64,
    lv_iv: f64,
    iv_error_bp: f64,
    lv_error_bp: f64,
    pricing_se_bp: f64,
    total_se_bp: f64,
    seed_to_seed_sd_bp: f64,
    paired_price_residual_bp: f64,
    paired_price_se_bp: f64,
    worst_seed_iv_error_bp: f64,
}

#[derive(Debug, Serialize)]
struct Report {
    smile: &'static str,
    settings: Settings,
    nodes: Vec<Node>,
    martingales: Vec<Estimate>,
    quote_fallbacks: usize,
}

fn experiment(smile: Smile, settings: Settings) -> Report {
    let surface = smile.surface();
    let runs = CALIBRATION_SEEDS.map(|seed| {
        let result = run(&surface, settings, seed);
        println!(
            "BERGOMI_RUN {}",
            serde_json::to_string(&serde_json::json!({
                "smile": smile.name(), "settings": settings, "run": result,
            }))
            .unwrap()
        );
        result
    });
    let nodes = (0..QUOTES)
        .map(|i| {
            let t = MATURITIES[i / LOG_STRIKES.len()];
            let x = LOG_STRIKES[i % LOG_STRIKES.len()];
            let target_iv = (surface
                .total_variance_derivatives(t, x)
                .unwrap()
                .total_variance
                / t)
                .sqrt();
            let (lsv, pricing_se) =
                ensemble(&runs.iter().map(|r| r.estimates[i]).collect::<Vec<_>>());
            let (lv, _) = ensemble(
                &runs
                    .iter()
                    .map(|r| r.estimates[QUOTES + i])
                    .collect::<Vec<_>>(),
            );
            let (paired, _) = ensemble(
                &runs
                    .iter()
                    .map(|r| r.estimates[2 * QUOTES + i])
                    .collect::<Vec<_>>(),
            );
            let model_iv = implied_volatility(lsv.value, t, x).expect("valid LSV IV");
            let lv_iv = implied_volatility(lv.value, t, x).expect("valid LV IV");
            let vega = black(t, x, target_iv).1;
            assert!(vega > 1.0, "do not silently skip low-vega quotes");
            let seed_prices = runs
                .iter()
                .map(|r| r.estimates[i].value)
                .collect::<Vec<_>>();
            let worst_seed_iv_error_bp = seed_prices
                .iter()
                .map(|&p| {
                    (implied_volatility(p, t, x).expect("valid per-seed IV") - target_iv).abs() / BP
                })
                .fold(0.0, f64::max);
            Node {
                maturity: t,
                log_strike: x,
                target_iv,
                model_iv,
                lv_iv,
                iv_error_bp: (model_iv - target_iv) / BP,
                lv_error_bp: (lv_iv - target_iv) / BP,
                pricing_se_bp: pricing_se / vega / BP,
                total_se_bp: lsv.se / vega / BP,
                seed_to_seed_sd_bp: estimate(&seed_prices).se * (runs.len() as f64).sqrt()
                    / vega
                    / BP,
                paired_price_residual_bp: paired.value / vega / BP,
                paired_price_se_bp: paired.se / vega / BP,
                worst_seed_iv_error_bp,
            }
        })
        .collect();
    let martingales = (0..MATURITIES.len())
        .map(|i| {
            ensemble(
                &runs
                    .iter()
                    .map(|r| r.estimates[3 * QUOTES + i])
                    .collect::<Vec<_>>(),
            )
            .0
        })
        .collect();
    let report = Report {
        smile: smile.name(),
        settings,
        nodes,
        martingales,
        quote_fallbacks: runs
            .iter()
            .flat_map(|r| &r.support)
            .map(|s| s.quote_fallbacks)
            .sum(),
    };
    println!(
        "BERGOMI_QUALITY {}",
        serde_json::to_string(&report).unwrap()
    );
    report
}

fn rms(values: impl Iterator<Item = f64>) -> f64 {
    let v = values.collect::<Vec<_>>();
    assert!(!v.is_empty());
    (v.iter().map(|x| x * x).sum::<f64>() / v.len() as f64).sqrt()
}

fn failures(report: &Report) -> Vec<String> {
    let mut failures = Vec::new();
    let mut check = |label: String, actual: f64, limit: f64| {
        if !actual.is_finite() || actual < 0.0 || !limit.is_finite() || actual > limit {
            failures.push(format!("{label}: {actual:.6} exceeds {limit:.6}"));
        }
    };
    check(
        "node count mismatch".into(),
        report.nodes.len().abs_diff(QUOTES) as f64,
        0.0,
    );
    check(
        "martingale count mismatch".into(),
        report.martingales.len().abs_diff(MATURITIES.len()) as f64,
        0.0,
    );
    check(
        "quote-neighbor fallback count".into(),
        report.quote_fallbacks as f64,
        0.0,
    );
    for n in &report.nodes {
        let label = format!("T={} k={}", n.maturity, n.log_strike);
        check(format!("{label} |IV error| bp"), n.iv_error_bp.abs(), 20.0);
        check(format!("{label} |LV error| bp"), n.lv_error_bp.abs(), 10.0);
        check(format!("{label} total SE bp"), n.total_se_bp, 5.0);
        check(format!("{label} pricing SE bp"), n.pricing_se_bp, 3.0);
        check(
            format!("{label} worst calibration seed |IV error| bp"),
            n.worst_seed_iv_error_bp,
            35.0,
        );
        check(
            format!("{label} paired LSV-LV residual bp"),
            n.paired_price_residual_bp.abs(),
            15.0,
        );
    }
    // Fixed hard budgets, not error < an arbitrarily large reported uncertainty.
    check(
        "LSV IV RMSE bp".into(),
        rms(report.nodes.iter().map(|n| n.iv_error_bp)),
        10.0,
    );
    check(
        "LV IV RMSE bp".into(),
        rms(report.nodes.iter().map(|n| n.lv_error_bp)),
        5.0,
    );
    for (&t, m) in MATURITIES.iter().zip(&report.martingales) {
        check(
            format!("martingale T={t}"),
            (m.value - SPOT).abs(),
            4.0 * m.se + 0.02,
        );
        check(format!("martingale SE T={t}"), m.se, 0.02);
    }
    failures
}

fn accept(report: &Report) {
    let errors = failures(report);
    assert!(
        errors.is_empty(),
        "{} calibration quality:\n{}",
        report.smile,
        errors.join("\n")
    );
}

#[test]
fn iv_inversion_round_trips_prices_and_rejects_invalid_quotes() {
    for t in MATURITIES {
        for x in LOG_STRIKES {
            for sigma in [0.1, 0.2, 0.4] {
                let iv = implied_volatility(black(t, x, sigma).0, t, x).unwrap();
                assert!((iv - sigma).abs() < 1e-10, "{t} {x} {sigma} {iv}");
            }
        }
    }
    for price in [-1.0, 0.0, SPOT, f64::INFINITY, f64::NAN] {
        assert!(implied_volatility(price, 1.0, 0.0).is_err());
    }
    assert!(implied_volatility(1.0, 0.0, 0.0).is_err());
    assert!(implied_volatility(SPOT * (-0.2_f64).exp(), 1.0, -0.2).is_err());
    assert!(implied_volatility(99.0, 0.25, 0.0).is_err());
}

#[test]
fn smile_fixtures_have_no_dupire_repairs() {
    for smile in [Smile::Flat, Smile::Skew] {
        let grid = target(&smile.surface(), ACCEPTANCE);
        assert!(grid.values().iter().all(|v| v.is_finite() && *v > 0.0));
    }
}

#[test]
fn uncertainty_uses_complete_runs_without_double_counting_pricing_noise() {
    let runs = [Estimate {
        value: 1.0,
        se: 2.0,
    }; 4];
    let (e, pricing_se) = ensemble(&runs);
    assert_eq!(e.value, 1.0);
    assert_eq!(pricing_se, 1.0);
    assert_eq!(e.se, 1.0);
    let runs = [0.0, 2.0, 4.0, 6.0].map(|value| Estimate { value, se: 0.1 });
    let (e, _) = ensemble(&runs);
    assert!((e.se - (5.0_f64 / 3.0).sqrt()).abs() < 1e-14);
}

#[test]
fn quality_gate_rejects_bias_noise_fallbacks_and_nonfinite_values() {
    let node = Node {
        maturity: 1.0,
        log_strike: 0.0,
        target_iv: 0.2,
        model_iv: 0.2,
        lv_iv: 0.2,
        iv_error_bp: 0.0,
        lv_error_bp: 0.0,
        pricing_se_bp: 0.0,
        total_se_bp: 0.0,
        seed_to_seed_sd_bp: 0.0,
        paired_price_residual_bp: 0.0,
        paired_price_se_bp: 0.0,
        worst_seed_iv_error_bp: 0.0,
    };
    let mut report = Report {
        smile: "gate unit test",
        settings: ACCEPTANCE,
        nodes: vec![node; QUOTES],
        martingales: vec![
            Estimate {
                value: SPOT,
                se: 0.0
            };
            3
        ],
        quote_fallbacks: 0,
    };
    assert!(failures(&report).is_empty());
    for error in [25.0, f64::NAN, f64::INFINITY] {
        report.nodes[0].iv_error_bp = error;
        assert!(!failures(&report).is_empty());
    }
    report.nodes[0].iv_error_bp = 0.0;
    report.nodes[0].total_se_bp = 6.0;
    assert!(!failures(&report).is_empty());
    report.nodes[0].total_se_bp = 0.0;
    report.quote_fallbacks = 1;
    assert!(!failures(&report).is_empty());
    report.quote_fallbacks = 0;
    report.nodes.iter_mut().for_each(|n| n.iv_error_bp = 11.0);
    assert!(failures(&report).iter().any(|e| e.contains("RMSE")));
}

#[test]
#[ignore = "smile-wide numerical acceptance: explicitly run in release-mode CI"]
fn iv_round_trip_flat_one_factor_bergomi() {
    accept(&experiment(Smile::Flat, ACCEPTANCE));
}

#[test]
#[ignore = "smile-wide numerical acceptance: explicitly run in release-mode CI"]
fn iv_round_trip_skew_one_factor_bergomi() {
    accept(&experiment(Smile::Skew, ACCEPTANCE));
}

#[test]
#[ignore = "manual one-at-a-time particle/bandwidth/time/space refinement study"]
fn one_factor_bergomi_refinement_report() {
    // Compare ensemble errors, not pathwise monotonicity: changing the time
    // grid changes random coordinates. Coarse runs are diagnostic, not gates.
    let reference = Settings {
        bandwidth: 0.035,
        ..ACCEPTANCE
    };
    for settings in [
        reference,
        Settings {
            particles: 4096,
            ..reference
        },
        Settings {
            bandwidth: 0.12,
            ..reference
        },
        Settings {
            steps: 32,
            ..reference
        },
        Settings {
            spatial_intervals: 40,
            ..reference
        },
    ] {
        let report = experiment(Smile::Flat, settings);
        println!("BERGOMI_REFINEMENT_FAILURES {:?}", failures(&report));
    }
    accept(&experiment(Smile::Flat, ACCEPTANCE));
}
