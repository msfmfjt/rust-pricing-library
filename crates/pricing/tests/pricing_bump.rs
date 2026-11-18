//! Bump-and-revalue measures (ADR 0043, phase 2): central differences of
//! complete revaluations on the task's own random coordinates.
use pricing::core::PositiveF64;
use pricing::market::{EquityMarket, MarketContext};
use pricing::mc::ExecutionPolicy;
use pricing::risk::SpotBump;
use pricing::*;
use serde_json::{Value, json};

fn policy() -> ExecutionPolicy {
    ExecutionPolicy::new(1, Some(128)).unwrap()
}
fn request_json(units: u64) -> Value {
    let mut v: Value = serde_json::from_str(include_str!(
        "../../../fixtures/v3/pricing_request.golden.json"
    ))
    .unwrap();
    v["engine"]["independent_sampling_units"] = json!(units);
    v["risk"] =
        json!({"delta":false,"vega":false,"smile_dynamics":{"type":"sticky_log_moneyness"}});
    v
}
fn source(v: &Value) -> PricingSpecification {
    PricingSpecification::from_request(
        parse_request_json(&serde_json::to_vec(v).unwrap(), JsonLimits::DEFAULT).unwrap(),
        policy(),
    )
}
fn fixture(name: &str) -> PricingSource {
    let bytes = std::fs::read(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures")
            .join(name),
    )
    .unwrap();
    if name.starts_with("task-") {
        parse_pricing_task_json(&bytes, JsonLimits::DEFAULT)
            .unwrap()
            .source()
            .clone()
    } else {
        parse_specification_json(&bytes, JsonLimits::DEFAULT)
            .unwrap()
            .into()
    }
}
fn bump(measure: Measure, size: BumpSize, calibration: BumpCalibration) -> MeasureRequest {
    MeasureRequest::new(measure, MeasureMethod::BumpAndRevalue { size, calibration })
}
fn relative(h: f64) -> BumpSize {
    BumpSize::relative(h).unwrap()
}
fn measures(requests: Vec<MeasureRequest>) -> PricingOperation {
    PricingOperation::Measures(MeasureSet::new(requests).unwrap())
}
fn price(s: PricingSpecification) -> f64 {
    PricingTask::new(s, PricingOperation::Evaluate)
        .unwrap()
        .compile()
        .unwrap()
        .evaluate()
        .unwrap()
        .report
        .value()
}
fn with_spot(s: &PricingSpecification, spot: f64) -> PricingSpecification {
    let market = s.market().equity();
    let forward = market
        .forward()
        .with_spot(PositiveF64::new(spot, "spot").unwrap())
        .unwrap();
    PricingSpecification::new(
        s.valuation_date(),
        s.product().clone(),
        MarketContext::Equity(EquityMarket::new(market.currency(), forward)),
        s.model().clone(),
        s.engine(),
        s.risk().clone(),
        s.numerics().clone(),
    )
    .unwrap()
}
fn with_volatility(s: &PricingSpecification, volatility: f64) -> PricingSpecification {
    s.clone()
        .with_model(
            PricingModelSpec::new(
                EquityModelSpec::Standard(pricing::models::ModelSpec::BlackScholes(
                    pricing::models::BlackScholesSpec::new(volatility).unwrap(),
                )),
                RateModelSpec::Deterministic,
            ),
            None,
        )
        .unwrap()
}

fn bass_json() -> Value {
    let mut v = request_json(64);
    v["model"] = json!({"type":"bass_local_volatility","parameters":{
        "maturity_nodes":[0.5,1.0],"log_forward_moneyness_nodes":[-2.0,0.0,2.0],
        "implied_volatilities":vec![0.2;6],
        "projection_nodes":(0..=200).map(|i|-2.0+4.0*f64::from(i)/200.0).collect::<Vec<_>>(),
        "grid_points":201,"grid_width":8.0,"max_iterations":2000,"cdf_tolerance":1e-7,
        "tail_tolerance":1e-7,"projection_tail_tolerance":1e-7,
        "projection_mean_tolerance":1e-3,"iv_bump":1e-4}});
    v
}

#[test]
fn paired_differences_reproduce_hand_built_revaluations() {
    let s = source(&request_json(256));
    let spot = s.market().equity().forward().spot().get();
    let n = MeasureCalibration::NotApplicable;
    let requests = vec![
        bump(Measure::SpotDelta, relative(0.01), n),
        bump(Measure::SpotGamma, relative(0.01), n),
        bump(
            Measure::ParallelMarketIvVega,
            BumpSize::absolute(0.01).unwrap(),
            n,
        ),
    ];
    let source: PricingSource = s.clone().into();
    // Delta and Gamma of one size share their sides; Gamma adds one readout
    // of the base plan for its paired error.
    assert_eq!(
        source.bump_cost(&requests).unwrap(),
        BumpCost {
            calibrations: 0,
            evaluations: 5
        }
    );
    let compiled = PricingTask::new(s.clone(), measures(requests))
        .unwrap()
        .compile()
        .unwrap();
    assert_eq!(compiled.capability().status, CapabilityStatus::Supported);
    assert_eq!(compiled.bump_cost().evaluations, 0);
    let report = compiled.evaluate().unwrap().report;
    assert_eq!(compiled.bump_cost().evaluations, 5);

    let base = price(s.clone());
    assert_eq!(report.value().to_bits(), base.to_bits());
    let h = 0.01 * spot;
    let (up, down) = (
        price(with_spot(&s, spot + h)),
        price(with_spot(&s, spot - h)),
    );
    let EquityModelSpec::Standard(pricing::models::ModelSpec::BlackScholes(model)) =
        &s.model().equity
    else {
        panic!("Black-Scholes fixture")
    };
    let sigma = model.volatility().get();
    let (vu, vd) = (
        price(with_volatility(&s, sigma + 0.01)),
        price(with_volatility(&s, sigma - 0.01)),
    );
    for (kind, expected) in [
        (MeasureKind::SpotDelta, (up - down) / (2.0 * h)),
        (
            MeasureKind::SpotGamma,
            ((up - base) + (down - base)) / (h * h),
        ),
        (MeasureKind::ParallelMarketIvVega, (vu - vd) / (2.0 * 0.01)),
    ] {
        // The estimate is the mean of paired per-unit differences, which
        // equals the difference of the reduced prices up to summation order.
        let row = report.risk(kind).unwrap();
        let tolerance = 1e-10 * (base.abs() / (0.01 * 0.01)).max(1.0);
        assert!(
            (row.values[0] - expected).abs() <= tolerance,
            "{kind:?}: {} vs {expected}",
            row.values[0]
        );
        let se = row.standard_errors.as_ref().unwrap()[0];
        assert!(se.is_finite() && se > 0.0, "{kind:?}");
        assert_eq!(row.method, "bump_and_revalue");
        assert_eq!(row.calibration, CalibrationTreatment::NotApplicable);
    }
}

// Shorthand used above; the public name is `BumpCalibration`.
use BumpCalibration as MeasureCalibration;

#[test]
fn revaluation_agrees_with_native_estimators_on_black_scholes() {
    let s = source(&request_json(8192));
    let n = BumpCalibration::NotApplicable;
    let native = PricingTask::new(
        s.clone(),
        measures(vec![
            MeasureRequest::native(Measure::SpotDelta),
            MeasureRequest::spot_gamma(SpotBump::relative(0.05).unwrap()),
            MeasureRequest::native(Measure::ParallelMarketIvVega),
        ]),
    )
    .unwrap()
    .compile()
    .unwrap()
    .evaluate()
    .unwrap()
    .report;
    let bumped = PricingTask::new(
        s,
        measures(vec![
            bump(Measure::SpotDelta, relative(0.01), n),
            bump(Measure::SpotGamma, relative(0.05), n),
            bump(
                Measure::ParallelMarketIvVega,
                BumpSize::absolute(0.005).unwrap(),
                n,
            ),
        ]),
    )
    .unwrap()
    .compile()
    .unwrap()
    .evaluate()
    .unwrap()
    .report;
    for kind in [
        MeasureKind::SpotDelta,
        MeasureKind::SpotGamma,
        MeasureKind::ParallelMarketIvVega,
    ] {
        let a = native.risk(kind).unwrap();
        let b = bumped.risk(kind).unwrap();
        let se = a.standard_errors.as_ref().unwrap()[0];
        let paired = b.standard_errors.as_ref().unwrap()[0];
        assert!(
            (a.values[0] - b.values[0]).abs() <= 4.0 * se.hypot(paired),
            "{kind:?}: native {} +/- {se}, bumped {} +/- {paired}",
            a.values[0],
            b.values[0]
        );
    }
}

#[test]
fn calibration_treatment_is_stated_and_must_fit_the_source() {
    let reason = |source: &PricingSource, request: MeasureRequest| {
        let c = source.measure_capabilities(&[request]).remove(0);
        (c.status, c.reason)
    };
    let unsupported = |r| (CapabilityStatus::Unsupported, Some(r));
    let bs: PricingSource = source(&request_json(64)).into();
    let lsv = fixture("specification-v1/bergomi_lsv.golden.json");
    let delta = |c| bump(Measure::SpotDelta, relative(0.01), c);
    assert_eq!(
        reason(&bs, delta(BumpCalibration::Recalibrated)),
        unsupported(CapabilityReason::BumpCalibrationMismatch)
    );
    assert_eq!(
        reason(&bs, delta(BumpCalibration::Fixed)),
        unsupported(CapabilityReason::BumpCalibrationMismatch)
    );
    assert_eq!(
        reason(&lsv, delta(BumpCalibration::NotApplicable)),
        unsupported(CapabilityReason::BumpCalibrationMismatch)
    );
    assert_eq!(
        reason(&lsv, delta(BumpCalibration::Fixed)),
        (
            CapabilityStatus::Conditional,
            Some(CapabilityReason::CompilationRequired)
        )
    );
    // Single-asset LSV can hold its calibration; Bass-LV cannot.
    let dividend = fixture("task-v1/stochastic_dividend_lsv.golden.json");
    assert_eq!(
        reason(&dividend, delta(BumpCalibration::Fixed)),
        (
            CapabilityStatus::Conditional,
            Some(CapabilityReason::CompilationRequired)
        )
    );
    let bass: PricingSource = source(&bass_json()).into();
    assert_eq!(
        reason(&bass, delta(BumpCalibration::Fixed)),
        unsupported(CapabilityReason::FixedCalibrationUnavailable)
    );
    assert_eq!(
        reason(&lsv, delta(BumpCalibration::Recalibrated)),
        (
            CapabilityStatus::Conditional,
            Some(CapabilityReason::CompilationRequired)
        )
    );
    // A local-volatility surface has no single volatility to move.
    let mut lv = request_json(64);
    lv["model"] = json!({"type":"local_volatility","local_variance_grid":{
        "time_nodes":[0.0,1.0],"log_forward_moneyness_nodes":[-1.0,1.0],"shape":[2,2],
        "values":[0.04,0.04,0.04,0.04],"floor":1e-8,"cap":4.0}});
    if let Ok(request) = parse_request_json(&serde_json::to_vec(&lv).unwrap(), JsonLimits::DEFAULT)
    {
        let lv: PricingSource = PricingSpecification::from_request(request, policy()).into();
        assert_eq!(
            reason(
                &lv,
                bump(
                    Measure::ParallelMarketIvVega,
                    relative(0.01),
                    BumpCalibration::NotApplicable
                )
            ),
            unsupported(CapabilityReason::ModelCombination)
        );
    }
    // A refused request makes the whole task refuse to construct.
    assert!(PricingTask::new(bass, measures(vec![delta(BumpCalibration::Fixed)])).is_err());
}

#[test]
fn recalibrated_bumps_state_and_realize_their_cost() {
    let lsv = fixture("specification-v1/bergomi_lsv.golden.json");
    let requests = vec![
        bump(
            Measure::SpotDelta,
            relative(0.01),
            BumpCalibration::Recalibrated,
        ),
        bump(
            Measure::SpotGamma,
            relative(0.01),
            BumpCalibration::Recalibrated,
        ),
    ];
    // Delta and Gamma share their sides; Gamma records the base plan once.
    let stated = lsv.bump_cost(&requests).unwrap();
    assert_eq!(
        stated,
        BumpCost {
            calibrations: 2,
            evaluations: 3
        }
    );
    let detail = lsv.measure_capabilities(&requests[..1]).remove(0).detail;
    assert!(
        detail.contains("2 calibrations and 2 evaluations"),
        "{detail}"
    );
    let compiled = PricingTask::new(lsv, measures(requests))
        .unwrap()
        .compile()
        .unwrap();
    assert_eq!(
        compiled.bump_cost(),
        BumpCost {
            calibrations: 2,
            evaluations: 0
        }
    );
    let report = compiled.evaluate().unwrap().report;
    assert_eq!(compiled.bump_cost(), stated);
    assert_eq!(
        report.uncertainty_scope(),
        UncertaintyScope::ConditionalOnCalibration
    );
    let row = report.risk(MeasureKind::SpotDelta).unwrap();
    assert_eq!(row.calibration, CalibrationTreatment::Recalibrated);
    assert!(row.values[0].is_finite());
}

#[test]
fn baskets_move_each_asset_separately() {
    for name in ["multi_asset_bs", "multi_asset_joint"] {
        let source = fixture(&format!("task-v1/{name}.golden.json"));
        let calibration = if name == "multi_asset_bs" {
            BumpCalibration::NotApplicable
        } else {
            BumpCalibration::Recalibrated
        };
        let request = bump(Measure::SpotDelta, relative(0.01), calibration);
        let stated = source.bump_cost(std::slice::from_ref(&request)).unwrap();
        let compiled = PricingTask::new(source.clone(), measures(vec![request]))
            .unwrap()
            .compile()
            .unwrap();
        let report = compiled.evaluate().unwrap().report;
        assert_eq!(compiled.bump_cost(), stated, "{name}");
        let rows: Vec<_> = report
            .risks()
            .iter()
            .filter(|r| r.measure == Some(MeasureKind::SpotDelta))
            .collect();
        let PricingSource::MultiAsset(basket) = &source else {
            unreachable!()
        };
        assert_eq!(rows.len(), basket.markets().len(), "{name}");
        assert_eq!(stated.evaluations, 2 * rows.len() as u64);
    }
}

#[test]
fn bump_requests_round_trip_and_change_identity() {
    let s = source(&request_json(64));
    let task = |h| {
        PricingTask::new(
            s.clone(),
            measures(vec![bump(
                Measure::SpotDelta,
                relative(h),
                BumpCalibration::NotApplicable,
            )]),
        )
        .unwrap()
    };
    let a = task(0.01);
    assert_ne!(a.fingerprint().unwrap(), task(0.02).fingerprint().unwrap());
    let json = pricing_task_to_json(&a).unwrap();
    let document: Value = serde_json::from_str(&json).unwrap();
    assert_eq!(
        document["operation"]["measures"][0]["method"],
        json!({"type":"bump_and_revalue","size":{"type":"relative","value":0.01},
               "calibration":"not_applicable"})
    );
    let restored = parse_pricing_task_json(json.as_bytes(), JsonLimits::DEFAULT).unwrap();
    assert_eq!(restored.operation(), a.operation());
    assert_eq!(restored.fingerprint().unwrap(), a.fingerprint().unwrap());

    let compiled = a.compile().unwrap();
    let report = compiled.evaluate().unwrap().report;
    let saved = valuation_report_to_json(&report).unwrap();
    assert_eq!(
        parse_valuation_report_json(saved.as_bytes(), JsonLimits::DEFAULT).unwrap(),
        report
    );
    // Reselection reuses compiled sides only for the same bump requests.
    assert!(compiled.with_operation(a.operation()).is_ok());
    assert!(compiled.with_operation(task(0.02).operation()).is_err());
    let reselected = compiled
        .with_operation(measures(vec![MeasureRequest::native(Measure::Price)]))
        .unwrap();
    assert!(reselected.evaluate().unwrap().report.risks().is_empty());
}

/// Engines other than the standard one read out their units too: a direct
/// rough-volatility source gets a paired error, and its estimate still agrees
/// with the hand-built difference of side prices to rounding.
#[test]
fn direct_rough_volatility_gets_a_paired_error() {
    let PricingSource::SingleAsset(s) = fixture("specification-v1/rough_heston.golden.json") else {
        unreachable!()
    };
    let request = bump(
        Measure::SpotDelta,
        relative(0.01),
        BumpCalibration::NotApplicable,
    );
    let report = PricingTask::new(s.clone(), measures(vec![request]))
        .unwrap()
        .compile()
        .unwrap()
        .evaluate()
        .unwrap()
        .report;
    let spot = s.market().equity().forward().spot().get();
    let h = 0.01 * spot;
    let expected = (price(with_spot(&s, spot + h)) - price(with_spot(&s, spot - h))) / (2.0 * h);
    let row = report.risk(MeasureKind::SpotDelta).unwrap();
    assert!((row.values[0] - expected).abs() <= 1e-10 * expected.abs().max(1.0));
    let se = row.standard_errors.as_ref().unwrap()[0];
    assert!(se.is_finite() && se > 0.0);
}

/// Recalibrated LSV bumps carry a paired error conditional on calibration.
#[test]
fn recalibrated_lsv_bumps_carry_a_paired_error() {
    let lsv = fixture("specification-v1/bergomi_lsv.golden.json");
    let report = PricingTask::new(
        lsv,
        measures(vec![
            bump(
                Measure::SpotDelta,
                relative(0.02),
                BumpCalibration::Recalibrated,
            ),
            bump(
                Measure::SpotGamma,
                relative(0.02),
                BumpCalibration::Recalibrated,
            ),
        ]),
    )
    .unwrap()
    .compile()
    .unwrap()
    .evaluate()
    .unwrap()
    .report;
    for kind in [MeasureKind::SpotDelta, MeasureKind::SpotGamma] {
        let row = report.risk(kind).unwrap();
        let se = row.standard_errors.as_ref().unwrap()[0];
        assert!(se.is_finite() && se > 0.0, "{kind:?}");
        assert!(row.convention.contains("paired"), "{kind:?}");
    }
}

/// The paired error describes the estimator's spread: over independent
/// seeds, the observed standard deviation of the bumped Delta and Gamma
/// matches the mean reported error, under pseudo-random and RQMC sampling.
#[test]
fn paired_errors_match_the_spread_across_seeds() {
    let n = BumpCalibration::NotApplicable;
    for rqmc in [false, true] {
        let mut deltas = Vec::new();
        let mut gammas = Vec::new();
        for seed in 0..40_u64 {
            let mut v = request_json(256);
            if rqmc {
                v["engine"] = json!({"type":"randomized_quasi_monte_carlo",
                    "points_per_scramble":32,"scramble_count":16,"master_scramble_seed":seed + 1,
                    "variance_reduction":{"antithetic":false,"brownian_bridge":false}});
            } else {
                v["engine"]["master_seed"] = json!(seed + 1);
            }
            let report = PricingTask::new(
                source(&v),
                measures(vec![
                    bump(Measure::SpotDelta, relative(0.05), n),
                    bump(Measure::SpotGamma, relative(0.05), n),
                ]),
            )
            .unwrap()
            .compile()
            .unwrap()
            .evaluate()
            .unwrap()
            .report;
            for (kind, out) in [
                (MeasureKind::SpotDelta, &mut deltas),
                (MeasureKind::SpotGamma, &mut gammas),
            ] {
                let row = report.risk(kind).unwrap();
                out.push((row.values[0], row.standard_errors.as_ref().unwrap()[0]));
            }
        }
        for (label, rows) in [("delta", &deltas), ("gamma", &gammas)] {
            let m = rows.len() as f64;
            let mean = rows.iter().map(|r| r.0).sum::<f64>() / m;
            let spread =
                (rows.iter().map(|r| (r.0 - mean).powi(2)).sum::<f64>() / (m - 1.0)).sqrt();
            let reported = rows.iter().map(|r| r.1).sum::<f64>() / m;
            let ratio = spread / reported;
            // 40 seeds put the sample standard deviation within about 25% of
            // its expectation at two standard errors.
            assert!(
                (0.6..1.5).contains(&ratio),
                "rqmc={rqmc} {label}: spread {spread}, reported {reported}"
            );
        }
    }
}

fn lsv_with_vol_of_vol(vol_of_vol: f64) -> PricingSpecification {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/specification-v1/bergomi_lsv.golden.json");
    let mut v: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    v["model"]["equity"]["factor"]["vol_of_vol"] = json!(vol_of_vol);
    v["model"]["equity"]["particles"]["particle_count"] = json!(512);
    v["engine"]["independent_sampling_units"] = json!(2048);
    parse_specification_json(&serde_json::to_vec(&v).unwrap(), JsonLimits::DEFAULT).unwrap()
}

/// A fixed calibration costs no calibration: both sides reuse the realized
/// leverage of the task's own plan.
#[test]
fn fixed_bumps_reuse_the_calibration() {
    let lsv: PricingSource = lsv_with_vol_of_vol(0.3).into();
    let fixed = |m| bump(m, relative(0.01), BumpCalibration::Fixed);
    let requests = vec![fixed(Measure::SpotDelta), fixed(Measure::SpotGamma)];
    let stated = lsv.bump_cost(&requests).unwrap();
    assert_eq!(
        stated,
        BumpCost {
            calibrations: 0,
            evaluations: 3
        }
    );
    let compiled = PricingTask::new(lsv.clone(), measures(requests))
        .unwrap()
        .compile()
        .unwrap();
    assert_eq!(compiled.bump_cost(), BumpCost::default());
    let report = compiled.evaluate().unwrap().report;
    assert_eq!(compiled.bump_cost(), stated);
    for kind in [MeasureKind::SpotDelta, MeasureKind::SpotGamma] {
        let row = report.risk(kind).unwrap();
        assert_eq!(row.calibration, CalibrationTreatment::FixedCalibration);
        assert!(
            row.convention
                .contains("absolute level of its calibrated state"),
            "{kind:?}"
        );
        let se = row.standard_errors.as_ref().unwrap()[0];
        assert!(row.values[0].is_finite() && se > 0.0, "{kind:?}");
    }
    // Fixed and recalibrated sides are different calculations.
    let recalibrated = PricingTask::new(
        lsv,
        measures(vec![bump(
            Measure::SpotDelta,
            relative(0.01),
            BumpCalibration::Recalibrated,
        )]),
    )
    .unwrap()
    .compile()
    .unwrap()
    .evaluate()
    .unwrap()
    .report;
    assert_ne!(
        recalibrated.risk(MeasureKind::SpotDelta).unwrap().values[0].to_bits(),
        report.risk(MeasureKind::SpotDelta).unwrap().values[0].to_bits()
    );
}

/// With no vol-of-vol and a flat target the leverage does not depend on the
/// forward level, so holding it in absolute level (fixed) or in moneyness
/// (recalibrated) is the same model: the two Deltas agree within error.
#[test]
fn fixed_and_recalibrated_agree_when_leverage_is_flat() {
    let lsv = lsv_with_vol_of_vol(0.0);
    let delta = |c| bump(Measure::SpotDelta, relative(0.01), c);
    let estimate = |c| {
        let report = PricingTask::new(lsv.clone(), measures(vec![delta(c)]))
            .unwrap()
            .compile()
            .unwrap()
            .evaluate()
            .unwrap()
            .report;
        let row = report.risk(MeasureKind::SpotDelta).unwrap().clone();
        (row.values[0], row.standard_errors.unwrap()[0])
    };
    let (fixed, fixed_se) = estimate(BumpCalibration::Fixed);
    let (recalibrated, recalibrated_se) = estimate(BumpCalibration::Recalibrated);
    assert!(
        (fixed - recalibrated).abs() <= 4.0 * fixed_se.hypot(recalibrated_se),
        "fixed {fixed} +/- {fixed_se}, recalibrated {recalibrated} +/- {recalibrated_se}"
    );
}

/// Bergomi LSV with Hull-White rates on a flat escrowed-forward density.
fn hull_white_lsv(vol_of_vol: f64) -> PricingSpecification {
    use pricing::mc::{hull_white::HullWhiteLsvTarget, lsv::LsvParticleConfig};
    use pricing::models::{Bergomi1Factor, HullWhite1Factor, LocalVolatilitySpec};
    let density = HullWhiteLsvTarget::flat(
        0.2,
        vec![0.0, 0.25, 0.5, 0.75, 1.0],
        vec![-0.6, 0.0, 0.6],
        1e-8,
        4.0,
    )
    .unwrap();
    let g = density.grid();
    let target = LocalVolatilitySpec::from_explicit_grid(
        g.time_nodes().to_vec(),
        g.log_moneyness_nodes().to_vec(),
        g.values().to_vec(),
        g.floor(),
        g.cap(),
    )
    .unwrap();
    let equity = EquityModelSpec::LocalStochasticVolatility {
        target,
        factor: VolatilityFactorSpec::Bergomi(Bergomi1Factor::new(1.2, vol_of_vol, -0.4).unwrap()),
        particles: LsvParticleConfig::new(256, 429, 0.4, 2.0, false).unwrap(),
        forward_density: Some(density),
    };
    let rates = RateModelSpec::HullWhite {
        model: HullWhite1Factor::new(0.2, vec![0.0, 0.5], vec![0.005, 0.007]).unwrap(),
        equity_rate_correlation: 0.1,
        vol_rate_correlations: vec![0.0],
    };
    source(&request_json(1024))
        .with_model(PricingModelSpec::new(equity, rates), None)
        .unwrap()
}

fn spot_delta(source: impl Into<PricingSource>, calibration: BumpCalibration) -> (f64, f64) {
    let compiled = PricingTask::new(
        source,
        measures(vec![bump(Measure::SpotDelta, relative(0.01), calibration)]),
    )
    .unwrap()
    .compile()
    .unwrap();
    if calibration == BumpCalibration::Fixed {
        assert_eq!(compiled.bump_cost(), BumpCost::default());
    }
    let report = compiled.evaluate().unwrap().report;
    let row = report.risk(MeasureKind::SpotDelta).unwrap();
    assert_eq!(
        row.calibration == CalibrationTreatment::FixedCalibration,
        calibration == BumpCalibration::Fixed
    );
    (row.values[0], row.standard_errors.as_ref().unwrap()[0])
}

/// Hull-White LSV re-anchors the reused leverage at the bumped Spot: no
/// calibration runs, the paths move, and the answer differs from the
/// recalibrated (sticky-moneyness) one once the leverage depends on the
/// level. With no vol-of-vol on a flat density the two agree within error.
#[test]
fn hull_white_lsv_holds_its_calibration() {
    let source: PricingSource = hull_white_lsv(0.3).into();
    assert!(source.supports_fixed_calibration());
    let fixed = bump(Measure::SpotDelta, relative(0.01), BumpCalibration::Fixed);
    assert_eq!(
        source.bump_cost(&[fixed]).unwrap(),
        BumpCost {
            calibrations: 0,
            evaluations: 2
        }
    );
    let (held, held_se) = spot_delta(source.clone(), BumpCalibration::Fixed);
    let (moved, _) = spot_delta(source, BumpCalibration::Recalibrated);
    assert!(held.is_finite() && held > 0.0 && held_se > 0.0, "{held}");
    assert_ne!(held.to_bits(), moved.to_bits());

    let flat = hull_white_lsv(0.0);
    let (held, held_se) = spot_delta(flat.clone(), BumpCalibration::Fixed);
    let (moved, moved_se) = spot_delta(flat, BumpCalibration::Recalibrated);
    assert!(
        (held - moved).abs() <= 4.0 * held_se.hypot(moved_se),
        "fixed {held} +/- {held_se}, recalibrated {moved} +/- {moved_se}"
    );
}

/// Residual-equity LSV with stochastic dividends re-anchors at the bumped
/// funded residual: no calibration runs and Delta is a real Delta.
#[test]
fn stochastic_dividend_lsv_holds_its_calibration() {
    let source = fixture("task-v1/stochastic_dividend_lsv.golden.json");
    let requests = vec![
        bump(Measure::SpotDelta, relative(0.01), BumpCalibration::Fixed),
        bump(Measure::SpotGamma, relative(0.01), BumpCalibration::Fixed),
    ];
    let stated = source.bump_cost(&requests).unwrap();
    assert_eq!(
        stated,
        BumpCost {
            calibrations: 0,
            evaluations: 3
        }
    );
    let compiled = PricingTask::new(source.clone(), measures(requests))
        .unwrap()
        .compile()
        .unwrap();
    let report = compiled.evaluate().unwrap().report;
    assert_eq!(compiled.bump_cost(), stated);
    let row = report.risk(MeasureKind::SpotDelta).unwrap();
    assert_eq!(row.calibration, CalibrationTreatment::FixedCalibration);
    let (held, held_se) = (row.values[0], row.standard_errors.as_ref().unwrap()[0]);
    assert!(
        held > 0.0 && held < 1.0 && held_se > 0.0,
        "{held} +/- {held_se}"
    );
    let (moved, _) = spot_delta(source, BumpCalibration::Recalibrated);
    assert_ne!(held.to_bits(), moved.to_bits());
}
