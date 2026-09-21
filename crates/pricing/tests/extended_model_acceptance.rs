//! Public-plan accuracy gates. Run the ignored panels explicitly in release mode.
//! Calibration populations and pricing streams are independent; no test-only
//! simulation or Sobol dimension mapping replaces the public pricing engine.

use pricing::core::{Date, DayCountConvention};
use pricing::market::{LocalVarianceGrid, MarketIvSurface};
use pricing::mc::hull_white::HullWhiteLsvTarget;
use pricing::mc::lsv::LsvParticleConfig;
use pricing::mc::{EngineConfig, ExecutionPolicy, RqmcConfig, VarianceReduction};
use pricing::{JsonLimits, PricingRequest, parse_request_json};
use pricing_numerics::{standard_normal_cdf, standard_normal_pdf};
use serde::Serialize;
use serde_json::{Value, json};

#[path = "cases/extended_multi_asset.rs"]
mod multi_asset;
#[path = "cases/extended_single_asset.rs"]
mod single_asset;

const BP: f64 = 1e-4;
const SEEDS: [u64; 3] = [1709, 2903, 4001];
const STRIKES: [f64; 3] = [-0.12, 0.0, 0.12];
const WINGS: [f64; 3] = [-0.36, 0.0, 0.36];
const RATE: f64 = 0.03;
const DIVIDEND_RATE: f64 = 0.01;

#[derive(Clone, Copy, Debug, Serialize)]
struct Resolution {
    particles: usize,
    steps: usize,
    bandwidth: f64,
    points: u64,
    scrambles: u32,
    seed_offset: u64,
}
const FINE: Resolution = Resolution {
    particles: 16_384,
    steps: 128,
    bandwidth: 0.05,
    points: 4096,
    scrambles: 8,
    seed_offset: 0,
};
const STRESS: Resolution = Resolution {
    particles: 65_536,
    steps: 192,
    bandwidth: 0.035,
    points: 8192,
    seed_offset: 40_000,
    ..FINE
};
impl Resolution {
    fn particles(self, seed: u64) -> LsvParticleConfig {
        LsvParticleConfig::new(self.particles, seed, self.bandwidth, 20.0, false).unwrap()
    }
    fn pricing_seed(self, seed: u64) -> u64 {
        seed ^ 0xd1b5_4a32_d192_ed03
    }
    fn engine(self, seed: u64) -> EngineConfig {
        EngineConfig::RandomizedQuasiMonteCarlo(
            RqmcConfig::new(
                self.points,
                self.scrambles,
                self.pricing_seed(seed),
                VarianceReduction::new(true, true),
            )
            .unwrap(),
        )
    }
    fn policy(self) -> ExecutionPolicy {
        ExecutionPolicy::new(2, Some(256)).unwrap()
    }
}

fn date(s: &str) -> Date {
    s.parse().unwrap()
}
fn today() -> Date {
    date("2026-01-01")
}
fn time(expiry: &str) -> f64 {
    DayCountConvention::Act365F.year_fraction(today(), date(expiry))
}

fn grid_times(t: f64, steps: usize) -> Vec<f64> {
    (0..=steps)
        .map(|i| {
            if i == steps {
                t
            } else {
                t * i as f64 / steps as f64
            }
        })
        .collect()
}

#[derive(Clone, Copy, Debug, Serialize)]
enum Smile {
    Flat,
    Skew,
    Stress,
    BasketStress,
    InfeasibleBasketStress,
}
impl Smile {
    fn iv(self, t: f64, x: f64) -> f64 {
        match self {
            Self::Flat => 0.2,
            Self::Skew => ((0.04 + 0.002 * t) * (1.0 - 0.2 * x + 0.05 * x * x)).sqrt(),
            Self::Stress => ((0.0625 + 0.002 * t) * (1.0 - 0.6 * x + 0.15 * x * x)).sqrt(),
            Self::BasketStress => (0.235_f64.powi(2) * (1.0 - 0.15 * x + 0.025 * x * x)).sqrt(),
            Self::InfeasibleBasketStress => {
                (0.235_f64.powi(2) * (1.0 - 0.3 * x + 0.05 * x * x)).sqrt()
            }
        }
    }
    fn target(self, expiry: &str, resolution: Resolution) -> HullWhiteLsvTarget {
        let mut ts = vec![time("2026-07-02"), 1.0, 2.0];
        if matches!(
            self,
            Self::Stress | Self::BasketStress | Self::InfeasibleBasketStress
        ) {
            ts.push(time("2029-01-01"));
        }
        let xs = vec![-1.2, -0.8, -0.4, -0.2, 0.0, 0.2, 0.4, 0.8, 1.2];
        let ivs = ts
            .iter()
            .flat_map(|&t| xs.iter().map(move |&x| self.iv(t, x)))
            .collect();
        HullWhiteLsvTarget::from_market_iv(
            MarketIvSurface::new(ts, xs, ivs).unwrap(),
            grid_times(time(expiry), resolution.steps),
            (0..=80).map(|i| -0.8 + i as f64 * 0.02).collect(),
            1e-8,
            4.0,
        )
        .unwrap()
    }
}

fn model_json(grid: &LocalVarianceGrid) -> Value {
    json!({"type":"local_volatility","local_variance_grid": {
        "time_nodes":grid.time_nodes(),
        "log_forward_moneyness_nodes":grid.log_moneyness_nodes(),
        "shape":[grid.time_nodes().len(),grid.log_moneyness_nodes().len()],
        "values":grid.values(),"floor":grid.floor(),"cap":grid.cap()
    }})
}

#[derive(Clone, Copy, Debug, Serialize)]
enum Cash {
    None,
    Midpoint,
    Expiry,
}
impl Cash {
    fn affine(self, t: f64) -> (f64, f64) {
        match self {
            Self::None => (1.0, 0.0),
            // Independent terminal escrow map: all cash has been paid at expiry.
            // B*alpha = B - D*exp(-(r-q)*ex_date)/S0.
            Self::Midpoint => (0.97 - 0.02 * (-(RATE - DIVIDEND_RATE) * t / 2.0).exp(), 0.0),
            Self::Expiry => (0.97 - 0.02 * (-(RATE - DIVIDEND_RATE) * t).exp(), 0.0),
        }
    }
}

fn request(
    expiry: &str,
    x: f64,
    cash: Cash,
    model: Value,
    resolution: Resolution,
    seed: u64,
) -> PricingRequest {
    let t = time(expiry);
    let (a, c) = cash.affine(t);
    let mut v: Value = serde_json::from_str(include_str!(
        "../../../fixtures/v1/pricing_request.golden.json"
    ))
    .unwrap();
    v["valuation_date"] = json!("2026-01-01");
    v["product"]["expiry"] = json!(expiry);
    v["product"]["strike"] = json!(a * 100.0 * ((RATE - DIVIDEND_RATE) * t + x).exp() + c);
    v["product"]["side"]["type"] = json!(if x < 0.0 { "put" } else { "call" });
    v["market"]["discount_curve"] =
        json!({"curve_id":10,"times":[0.0,3.0],"discount_factors":[1.0,(-3.0*RATE).exp()]});
    v["market"]["dividend_curve"] = json!({"curve_id":11,"times":[0.0,3.0],"discount_factors":[1.0,(-3.0*DIVIDEND_RATE).exp()]});
    if !matches!(cash, Cash::None) {
        v["market"]["discrete_dividends"] = json!([{
            "event_id":1,"ex_time":if matches!(cash,Cash::Midpoint) {t/2.0} else {t},
            "quote":{"type":"fixed_cash_and_proportional","fixed_cash":2.0,"beta":0.03}
        }]);
    }
    v["model"] = model;
    v["engine"] = json!({"type":"randomized_quasi_monte_carlo", "points_per_scramble":resolution.points,
        "scramble_count":resolution.scrambles,"master_scramble_seed":resolution.pricing_seed(seed),
        "variance_reduction":{"antithetic":true,"brownian_bridge":true}});
    parse_request_json(&serde_json::to_vec(&v).unwrap(), JsonLimits::DEFAULT).unwrap()
}

// Black's formula is an independent reference for these synthetic marginal
// targets. No simulated LV price is used as the expected LSV/basket price.
fn black(forward: f64, strike: f64, discount: f64, t: f64, iv: f64) -> (f64, f64) {
    let s = iv * t.sqrt();
    let d1 = (forward / strike).ln() / s + 0.5 * s;
    let d2 = d1 - s;
    let price = if strike < forward {
        discount * (strike * standard_normal_cdf(-d2) - forward * standard_normal_cdf(-d1))
    } else {
        discount * (forward * standard_normal_cdf(d1) - strike * standard_normal_cdf(d2))
    };
    (
        price,
        discount * forward * standard_normal_pdf(d1) * t.sqrt(),
    )
}

fn invert(price: f64, forward: f64, strike: f64, discount: f64, t: f64) -> f64 {
    assert!(
        [price, forward, strike, discount, t]
            .iter()
            .all(|v| v.is_finite() && *v > 0.0)
            && price < discount * forward.min(strike),
        "invalid OTM option price or coordinate"
    );
    let (mut lo, mut hi) = (1e-8, 3.0);
    assert!(price < black(forward, strike, discount, t, hi).0);
    for _ in 0..70 {
        let mid = (lo + hi) * 0.5;
        if black(forward, strike, discount, t, mid).0 < price {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    (lo + hi) * 0.5
}

#[derive(Clone, Copy, Debug, Serialize)]
struct PriceRun {
    calibration_seed: u64,
    pricing_seed: u64,
    value: f64,
    conditional_se: f64,
}
#[derive(Clone, Debug, Serialize)]
struct QuoteReport {
    log_strike: f64,
    target_iv: f64,
    iv_error_bp: f64,
    ensemble_se_bp: f64,
    pricing_se_bp: f64,
    worst_seed_error_bp: f64,
    runs: Vec<PriceRun>,
}

fn summarize(runs: Vec<PriceRun>, forward: f64, t: f64, x: f64, target_iv: f64) -> QuoteReport {
    assert!(runs.len() >= 3);
    for r in &runs {
        assert!(r.value.is_finite() && r.conditional_se.is_finite() && r.conditional_se >= 0.0);
        assert_ne!(r.calibration_seed, r.pricing_seed);
    }
    let n = runs.len() as f64;
    let mean = runs.iter().map(|r| r.value).sum::<f64>() / n;
    let between =
        (runs.iter().map(|r| (r.value - mean).powi(2)).sum::<f64>() / (n * (n - 1.0))).sqrt();
    let within = runs
        .iter()
        .map(|r| r.conditional_se.powi(2))
        .sum::<f64>()
        .sqrt()
        / n;
    let strike = forward * x.exp();
    let discount = (-RATE * t).exp();
    let vega = black(forward, strike, discount, t, target_iv).1;
    assert!(vega > 5.0, "fixed panel must have adequate vega");
    // The between-run variance already includes pricing noise: do not add it
    // to the conditional variance again. Three seeds are a regression panel,
    // not a calibrated confidence-interval coverage statement.
    QuoteReport {
        log_strike: x,
        target_iv,
        iv_error_bp: (invert(mean, forward, strike, discount, t) - target_iv) / BP,
        ensemble_se_bp: between.max(within) / vega / BP,
        pricing_se_bp: within / vega / BP,
        worst_seed_error_bp: runs
            .iter()
            .map(|r| (invert(r.value, forward, strike, discount, t) - target_iv).abs() / BP)
            .fold(0.0, f64::max),
        runs,
    }
}

#[derive(Clone, Copy, Debug, Serialize)]
struct Budget {
    max_error_bp: f64,
    rmse_bp: f64,
    worst_seed_bp: f64,
    ensemble_se_bp: f64,
    pricing_se_bp: f64,
}
const LSV_BUDGET: Budget = Budget {
    max_error_bp: 15.0,
    rmse_bp: 8.0,
    worst_seed_bp: 30.0,
    ensemble_se_bp: 4.0,
    pricing_se_bp: 3.0,
};
const GAUSSIAN_BUDGET: Budget = Budget {
    max_error_bp: 2.0,
    rmse_bp: 1.0,
    worst_seed_bp: 4.0,
    ensemble_se_bp: 1.0,
    pricing_se_bp: 1.0,
};

fn violations_at(quotes: &[QuoteReport], strikes: &[f64], budget: Budget) -> Vec<String> {
    let mut failures = Vec::new();
    if quotes.len() != strikes.len() {
        failures.push("missing quote".into());
    }
    for (q, &expected_x) in quotes.iter().zip(strikes) {
        if q.log_strike != expected_x {
            failures.push("quote order/strike mismatch".into());
        }
        for (name, value, limit) in [
            ("IV error", q.iv_error_bp.abs(), budget.max_error_bp),
            ("ensemble SE", q.ensemble_se_bp, budget.ensemble_se_bp),
            (
                "conditional pricing SE",
                q.pricing_se_bp,
                budget.pricing_se_bp,
            ),
            ("worst seed", q.worst_seed_error_bp, budget.worst_seed_bp),
        ] {
            if !value.is_finite() || value < 0.0 || value > limit {
                failures.push(format!(
                    "k={}: {name} {value:.4} bp exceeds {limit}",
                    q.log_strike
                ));
            }
        }
    }
    let rmse =
        (quotes.iter().map(|q| q.iv_error_bp.powi(2)).sum::<f64>() / quotes.len() as f64).sqrt();
    if !rmse.is_finite() || rmse > budget.rmse_bp {
        failures.push(format!("RMSE {rmse:.4} bp exceeds {}", budget.rmse_bp));
    }
    failures
}

fn violations(quotes: &[QuoteReport], budget: Budget) -> Vec<String> {
    violations_at(quotes, &STRIKES, budget)
}

fn report(case: Value, resolution: Resolution, quotes: &[QuoteReport], budget: Budget) {
    report_at(case, resolution, quotes, &STRIKES, budget);
}

fn report_at(
    case: Value,
    resolution: Resolution,
    quotes: &[QuoteReport],
    strikes: &[f64],
    budget: Budget,
) {
    let failures = violations_at(quotes, strikes, budget);
    println!(
        "EXTENDED_ACCURACY {}",
        json!({"case":case,"resolution":resolution,"strikes":strikes,"budget":budget,"quotes":quotes,"failures":failures})
    );
    assert!(failures.is_empty(), "{case}: {failures:?}");
}

fn refinements() -> [(&'static str, Resolution); 3] {
    [
        (
            "particles",
            Resolution {
                particles: 32_768,
                seed_offset: 10_000,
                ..FINE
            },
        ),
        (
            "time_steps",
            Resolution {
                steps: 256,
                seed_offset: 20_000,
                ..FINE
            },
        ),
        (
            "bandwidth",
            Resolution {
                bandwidth: 0.035,
                seed_offset: 30_000,
                ..FINE
            },
        ),
    ]
}

fn compare_refinement(case: Value, axis: &str, base: &[QuoteReport], refined: &[QuoteReport]) {
    assert_eq!(base.len(), refined.len());
    for (b, r) in base.iter().zip(refined) {
        assert_eq!(b.log_strike, r.log_strike);
        // Disjoint calibration AND pricing streams across resolutions. This
        // checks stability; it does not assume monotonic sampled MC error.
        let change = (b.iv_error_bp - r.iv_error_bp).abs();
        let bound = 5.0 + 3.0 * b.ensemble_se_bp.hypot(r.ensemble_se_bp);
        println!(
            "EXTENDED_REFINEMENT {}",
            json!({"case":case,"axis":axis,
            "log_strike":b.log_strike,"change_bp":change,"bound_bp":bound})
        );
        assert!(
            change <= bound,
            "{case}/{axis}: change={change}, bound={bound}"
        );
    }
}

// Includes both interpolation neighbours, even when x is exactly a node.
fn bracket(nodes: &[f64], x: f64) -> std::ops::RangeInclusive<usize> {
    assert!(x >= nodes[0] && x <= nodes[nodes.len() - 1]);
    let right = nodes.partition_point(|&node| node < x).min(nodes.len() - 1);
    right.saturating_sub(1)..=right
}

#[test]
fn acceptance_rejects_bias_noise_missing_quotes_and_nonfinite_metrics() {
    let make = |x| QuoteReport {
        log_strike: x,
        target_iv: 0.2,
        iv_error_bp: 0.0,
        ensemble_se_bp: 0.2,
        pricing_se_bp: 0.1,
        worst_seed_error_bp: 0.0,
        runs: vec![],
    };
    let good: Vec<_> = STRIKES.into_iter().map(make).collect();
    assert!(violations(&good, LSV_BUDGET).is_empty());
    for bad in [f64::NAN, 16.0] {
        let mut changed = good.clone();
        changed[0].iv_error_bp = bad;
        assert!(!violations(&changed, LSV_BUDGET).is_empty());
    }
    let mut noisy = good.clone();
    noisy[1].ensemble_se_bp = 100.0;
    assert!(!violations(&noisy, LSV_BUDGET).is_empty());
    assert!(!violations(&good[..2], LSV_BUDGET).is_empty());
    let wings: Vec<_> = WINGS.into_iter().map(make).collect();
    assert!(violations_at(&wings, &WINGS, LSV_BUDGET).is_empty());
    assert!(!violations_at(&good, &WINGS, LSV_BUDGET).is_empty());
    let mut duplicate = wings.clone();
    duplicate[2] = duplicate[0].clone();
    assert!(!violations_at(&duplicate, &WINGS, LSV_BUDGET).is_empty());
}

#[test]
fn independent_black_inversion_and_ensemble_noise_accounting() {
    for x in STRIKES {
        let (p, _) = black(100.0, 100.0 * x.exp(), 1.0, 0.5, 0.24);
        assert!((invert(p, 100.0, 100.0 * x.exp(), 1.0, 0.5) - 0.24).abs() < 1e-13);
    }
    let runs = SEEDS
        .into_iter()
        .map(|seed| PriceRun {
            calibration_seed: seed,
            pricing_seed: FINE.pricing_seed(seed),
            value: black(100.0, 100.0, (-RATE).exp(), 1.0, 0.2).0,
            conditional_se: 0.001,
        })
        .collect();
    let q = summarize(runs, 100.0, 1.0, 0.0, 0.2);
    assert!(q.iv_error_bp.abs() < 1e-9);
    assert_eq!(q.ensemble_se_bp, q.pricing_se_bp);
}

#[test]
#[should_panic(expected = "invalid OTM")]
fn invalid_prices_are_not_clipped_into_an_iv() {
    invert(0.0, 100.0, 100.0, 1.0, 1.0);
}

#[test]
fn independent_smile_oracle_matches_interpolated_panel_coordinates() {
    use pricing::market::ImpliedVarianceSurface;

    // The independent polynomial oracle is sampled off the market strike
    // nodes. Bound the natural-spline interpolation difference explicitly so
    // it cannot masquerade as a calibration or simulation error.
    for smile in [Smile::Flat, Smile::Skew] {
        for expiry in ["2026-07-02", "2027-01-01", "2028-01-01"] {
            let target = smile.target(expiry, FINE);
            for x in STRIKES {
                let t = time(expiry);
                let w = target
                    .market_iv_surface()
                    .unwrap()
                    .total_variance_derivatives(t, x)
                    .unwrap()
                    .total_variance;
                assert!(((w / t).sqrt() - smile.iv(t, x)).abs() < 0.1 * BP);
            }
        }
    }
    let t = time("2029-01-01");
    assert_eq!(*grid_times(t, STRESS.steps).last().unwrap(), t);
    for smile in [
        Smile::Stress,
        Smile::BasketStress,
        Smile::InfeasibleBasketStress,
    ] {
        let target = smile.target("2029-01-01", STRESS);
        for x in WINGS {
            let w = target
                .market_iv_surface()
                .unwrap()
                .total_variance_derivatives(t, x)
                .unwrap()
                .total_variance;
            assert!(((w / t).sqrt() - smile.iv(t, x)).abs() < 0.1 * BP);
            assert!(black(100.0, 100.0 * x.exp(), (-RATE * t).exp(), t, smile.iv(t, x)).1 > 5.0);
        }
    }
}
