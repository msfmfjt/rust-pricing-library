//! Public source documents must reproduce the existing numerical APIs exactly.
use pricing::hull_white::HullWhiteEquityPricingPlan;
use pricing::lsv::{BergomiLsvPricingPlan, RoughBergomiLsvPricingPlan, RoughFamilyLsvPricingPlan};
use pricing::market::MarketIvSurface;
use pricing::mc::{ExecutionPolicy, hull_white::HullWhiteLsvTarget, lsv::LsvParticleConfig};
use pricing::models::{
    Bergomi1Factor, Bergomi2Factor, HullWhite1Factor, HybridCorrelation, LocalVolatilitySpec,
    ModelSpec, RoughBergomi,
};
use pricing::rough_volatility::*;
use pricing::stochastic_volatility::StochasticVolatilityPricingPlan;
use pricing::{
    EquityModelSpec, JsonLimits, PricingModelSpec, PricingPlan, PricingRequest,
    PricingSpecification, RateModelSpec, SpecificationOutput, VolatilityFactorSpec,
    parse_request_json, parse_specification_json, specification_to_json,
    specification_to_pretty_json,
};
use serde_json::{Value, json};

fn policy() -> ExecutionPolicy {
    ExecutionPolicy::new(2, Some(32)).unwrap()
}
fn payload() -> Value {
    let mut v: Value = serde_json::from_str(include_str!(
        "../../../fixtures/v3/pricing_request.golden.json"
    ))
    .unwrap();
    v["engine"]["independent_sampling_units"] = json!(64);
    v
}
fn request(v: Value) -> PricingRequest {
    parse_request_json(&serde_json::to_vec(&v).unwrap(), JsonLimits::DEFAULT).unwrap()
}
fn source(
    equity: EquityModelSpec,
    rates: RateModelSpec,
    step: Option<f64>,
) -> PricingSpecification {
    PricingSpecification::from_request(request(payload()), policy())
        .with_model(PricingModelSpec::new(equity, rates), step)
        .unwrap()
}
fn parse(v: &Value) -> Result<PricingSpecification, pricing::WireError> {
    parse_specification_json(&serde_json::to_vec(v).unwrap(), JsonLimits::DEFAULT)
}
fn wire(s: &PricingSpecification) -> Value {
    serde_json::from_str(&specification_to_json(s).unwrap()).unwrap()
}
fn replay(s: &PricingSpecification) {
    let text = specification_to_json(s).unwrap();
    let pretty = specification_to_pretty_json(s).unwrap();
    assert!(text.ends_with('\n') && !text.ends_with("\n\n"));
    let restored = parse_specification_json(pretty.as_bytes(), JsonLimits::DEFAULT).unwrap();
    assert_eq!(s.model(), restored.model());
    assert_eq!(s.numerics(), restored.numerics());
    assert_eq!(s.fingerprint().unwrap(), restored.fingerprint().unwrap());
    assert_eq!(text, specification_to_json(&restored).unwrap());
    let original = s.compile().unwrap();
    let saved = restored.compile().unwrap();
    assert_eq!(original.plan_fingerprint(), saved.plan_fingerprint());
    assert_eq!(original.evaluate().unwrap(), saved.evaluate().unwrap());
}
fn factors() -> Vec<VolatilityFactorSpec> {
    vec![
        VolatilityFactorSpec::Bergomi(Bergomi1Factor::new(1.2, 0.3, -0.4).unwrap()),
        VolatilityFactorSpec::BergomiTwoFactor(
            Bergomi2Factor::new([0.5, 2.0], 0.3, 0.4, [-0.3, -0.2], 0.1).unwrap(),
        ),
        VolatilityFactorSpec::RoughBergomi(RoughBergomi::new(0.2, 0.3, -0.4).unwrap()),
    ]
}
fn rate_model() -> HullWhite1Factor {
    HullWhite1Factor::new(0.2, vec![0.0, 0.5], vec![0.005, 0.007]).unwrap()
}
fn rates(count: usize) -> RateModelSpec {
    RateModelSpec::HullWhite {
        model: rate_model(),
        equity_rate_correlation: 0.1,
        vol_rate_correlations: vec![0.0; count],
    }
}
fn particles() -> LsvParticleConfig {
    LsvParticleConfig::new(64, 429, 0.4, 2.0, true).unwrap()
}
fn density() -> HullWhiteLsvTarget {
    HullWhiteLsvTarget::flat(
        0.2,
        vec![0.0, 0.25, 0.5, 0.75, 1.0],
        vec![-0.6, 0.0, 0.6],
        1e-8,
        4.0,
    )
    .unwrap()
}
fn target(density: &HullWhiteLsvTarget) -> LocalVolatilitySpec {
    let g = density.grid();
    LocalVolatilitySpec::from_explicit_grid(
        g.time_nodes().to_vec(),
        g.log_moneyness_nodes().to_vec(),
        g.values().to_vec(),
        g.floor(),
        g.cap(),
    )
    .unwrap()
}
fn local_request(t: &LocalVolatilitySpec) -> PricingRequest {
    let r = request(payload());
    PricingRequest::new(
        r.valuation_date(),
        r.product().clone(),
        r.market().clone(),
        ModelSpec::LocalVolatility(t.clone()),
        r.engine(),
        r.risk().clone(),
    )
    .unwrap()
}
fn families() -> Vec<RoughVolatilityModel> {
    let h = RoughHeston::new(0.2, 0.04, 0.7, 0.055, 0.15, -0.6).unwrap();
    vec![
        h.clone().into(),
        LiftedHeston::from_rough(&h, 4, 2.5).unwrap().into(),
        QuadraticRoughHeston::new(0.2, 0.1, 0.8, 0.3, 0.25, 0.05, 0.035)
            .unwrap()
            .into(),
        MixedRoughBergomi::new(
            0.2,
            -0.6,
            vec![0.35, 0.65],
            vec![0.25, 0.6],
            ForwardVarianceCurve::piecewise_linear(vec![0.0, 1.0], vec![0.04, 0.05]).unwrap(),
        )
        .unwrap()
        .into(),
        RoughSabr::new(
            0.2,
            0.4,
            -0.6,
            1.0,
            ForwardVarianceCurve::exponential(0.04, 0.1).unwrap(),
        )
        .unwrap()
        .into(),
        Rfsv::new(0.2, 0.7, 0.12, -1.7, None).unwrap().into(),
        Rfsv::new(0.2, 0.7, 0.12, -1.7, Some(-1.6)).unwrap().into(),
    ]
}

#[test]
fn legacy_requests_keep_native_results_risks_and_migration_provenance() {
    for fixture in [
        include_str!("../../../fixtures/v1/pricing_request.golden.json"),
        include_str!("../../../fixtures/v2/pricing_request.golden.json"),
        include_str!("../../../fixtures/v3/pricing_request.golden.json"),
        include_str!("../../../fixtures/v3/pricing_request_american.golden.json"),
    ] {
        let mut v: Value = serde_json::from_str(fixture).unwrap();
        v["engine"]["independent_sampling_units"] = json!(64);
        v["risk"]["delta"] = json!(true);
        v["risk"]["vega"] = json!(true);
        let r = request(v);
        let old = PricingPlan::compile(&r, policy()).unwrap();
        let s = PricingSpecification::from_request(r, policy());
        let new = s.compile().unwrap();
        assert_eq!(old.plan_fingerprint(), new.plan_fingerprint());
        assert_eq!(
            SpecificationOutput::Standard(Box::new(old.evaluate().unwrap())),
            new.evaluate().unwrap().output
        );
        let restored = parse(&wire(&s)).unwrap();
        // The new wire document stores normalized sources, not legacy provenance.
        assert_eq!(
            new.evaluate().unwrap().value(),
            restored.compile().unwrap().evaluate().unwrap().value()
        );
        assert_eq!(s.fingerprint().unwrap(), restored.fingerprint().unwrap());
    }
}

#[test]
fn all_sv_factors_and_hull_white_compositions_match_native_plans() {
    let r = request(payload());
    let bs = source(
        EquityModelSpec::Standard(r.model().clone()),
        rates(0),
        Some(0.25),
    );
    let native =
        HullWhiteEquityPricingPlan::compile_bs(&r, rate_model(), 0.1, 0.25, policy()).unwrap();
    assert_eq!(
        bs.compile().unwrap().plan_fingerprint(),
        native.plan_fingerprint()
    );
    assert_eq!(
        bs.compile().unwrap().evaluate().unwrap().output,
        SpecificationOutput::Hybrid(native.evaluate().unwrap())
    );
    replay(&bs);
    for f in factors() {
        let (plain, hybrid, count) = match f {
            VolatilityFactorSpec::Bergomi(f) => (
                StochasticVolatilityPricingPlan::compile_bergomi(&r, f, 0.25, policy()).unwrap(),
                HullWhiteEquityPricingPlan::compile_bergomi(
                    &r,
                    f,
                    rate_model(),
                    HybridCorrelation::new(f.correlation(), 0.1, 0.0).unwrap(),
                    0.25,
                    policy(),
                )
                .unwrap(),
                1,
            ),
            VolatilityFactorSpec::BergomiTwoFactor(f) => (
                StochasticVolatilityPricingPlan::compile_bergomi_two_factor(&r, f, 0.25, policy())
                    .unwrap(),
                HullWhiteEquityPricingPlan::compile_bergomi_two_factor(
                    &r,
                    f,
                    rate_model(),
                    0.1,
                    [0.0; 2],
                    0.25,
                    policy(),
                )
                .unwrap(),
                2,
            ),
            VolatilityFactorSpec::RoughBergomi(f) => (
                StochasticVolatilityPricingPlan::compile_rough_bergomi(&r, f, 0.25, policy())
                    .unwrap(),
                HullWhiteEquityPricingPlan::compile_rough_bergomi(
                    &r,
                    f,
                    rate_model(),
                    HybridCorrelation::new(f.correlation(), 0.1, 0.0).unwrap(),
                    0.25,
                    policy(),
                )
                .unwrap(),
                1,
            ),
            _ => unreachable!(),
        };
        let equity = EquityModelSpec::StochasticVolatility {
            initial_volatility: 0.2,
            factor: f,
        };
        let s = source(equity.clone(), RateModelSpec::Deterministic, Some(0.25));
        assert_eq!(
            s.compile().unwrap().plan_fingerprint(),
            plain.plan_fingerprint()
        );
        assert_eq!(
            s.compile().unwrap().evaluate().unwrap().output,
            SpecificationOutput::Hybrid(plain.evaluate().unwrap())
        );
        replay(&s);
        let s = source(equity, rates(count), Some(0.25));
        assert_eq!(
            s.compile().unwrap().plan_fingerprint(),
            hybrid.plan_fingerprint()
        );
        assert_eq!(
            s.compile().unwrap().evaluate().unwrap().output,
            SpecificationOutput::Hybrid(hybrid.evaluate().unwrap())
        );
        replay(&s);
    }
}

#[test]
fn lsv_factors_retain_targets_particles_and_native_diagnostics() {
    let d = density();
    let t = target(&d);
    let r = local_request(&t);
    for f in factors() {
        let (plain_hash, plain_value, hybrid, count) = match f {
            VolatilityFactorSpec::Bergomi(f) => {
                let p = BergomiLsvPricingPlan::compile(&r, f, particles(), policy()).unwrap();
                (
                    p.plan_fingerprint(),
                    p.evaluate().unwrap(),
                    HullWhiteEquityPricingPlan::compile_lsv(
                        &r,
                        &d,
                        f,
                        rate_model(),
                        HybridCorrelation::new(f.correlation(), 0.1, 0.0).unwrap(),
                        particles(),
                        policy(),
                    )
                    .unwrap(),
                    1,
                )
            }
            VolatilityFactorSpec::BergomiTwoFactor(f) => {
                let p = BergomiLsvPricingPlan::compile(&r, f, particles(), policy()).unwrap();
                (
                    p.plan_fingerprint(),
                    p.evaluate().unwrap(),
                    HullWhiteEquityPricingPlan::compile_lsv_two_factor(
                        &r,
                        &d,
                        f,
                        rate_model(),
                        0.1,
                        [0.0; 2],
                        particles(),
                        policy(),
                    )
                    .unwrap(),
                    2,
                )
            }
            VolatilityFactorSpec::RoughBergomi(f) => {
                let p = RoughBergomiLsvPricingPlan::compile(&r, f, particles(), policy()).unwrap();
                (
                    p.plan_fingerprint(),
                    p.evaluate().unwrap(),
                    HullWhiteEquityPricingPlan::compile_rough_lsv(
                        &r,
                        &d,
                        f,
                        rate_model(),
                        HybridCorrelation::new(f.correlation(), 0.1, 0.0).unwrap(),
                        particles(),
                        policy(),
                    )
                    .unwrap(),
                    1,
                )
            }
            _ => unreachable!(),
        };
        let s = source(
            EquityModelSpec::LocalStochasticVolatility {
                target: t.clone(),
                factor: f.clone(),
                particles: particles(),
                forward_density: None,
            },
            RateModelSpec::Deterministic,
            None,
        );
        assert_eq!(s.compile().unwrap().plan_fingerprint(), plain_hash);
        assert_eq!(
            s.compile().unwrap().evaluate().unwrap().output,
            SpecificationOutput::LocalStochasticVolatility(plain_value)
        );
        replay(&s);
        let s = source(
            EquityModelSpec::LocalStochasticVolatility {
                target: t.clone(),
                factor: f,
                particles: particles(),
                forward_density: Some(d.clone()),
            },
            rates(count),
            None,
        );
        assert_eq!(
            s.compile().unwrap().plan_fingerprint(),
            hybrid.plan_fingerprint()
        );
        assert_eq!(
            s.compile().unwrap().evaluate().unwrap().output,
            SpecificationOutput::Hybrid(hybrid.evaluate().unwrap())
        );
        replay(&s);
    }
}

#[test]
fn all_rough_families_replay_direct_and_calibrated_prices() {
    let mut v = payload();
    v["model"]["volatility"] = json!(0.0);
    let r = request(v);
    let t = target(&density());
    for model in families() {
        let native =
            RoughVolatilityPricingPlan::compile(&r, model.clone(), 0.25, policy()).unwrap();
        let s = source(
            EquityModelSpec::RoughVolatility(model.clone()),
            RateModelSpec::Deterministic,
            Some(0.25),
        );
        assert_eq!(
            s.compile().unwrap().plan_fingerprint(),
            native.plan_fingerprint()
        );
        assert_eq!(
            s.compile().unwrap().evaluate().unwrap().output,
            SpecificationOutput::RoughVolatility(native.evaluate().unwrap())
        );
        replay(&s);
        let native = RoughFamilyLsvPricingPlan::compile(
            &local_request(&t),
            model.clone(),
            particles(),
            policy(),
        )
        .unwrap();
        let s = source(
            EquityModelSpec::RoughLocalStochasticVolatility {
                target: t.clone(),
                model,
                particles: particles(),
            },
            RateModelSpec::Deterministic,
            None,
        );
        assert_eq!(
            s.compile().unwrap().plan_fingerprint(),
            native.plan_fingerprint()
        );
        assert_eq!(
            s.compile().unwrap().evaluate().unwrap().output,
            SpecificationOutput::LocalStochasticVolatility(native.evaluate().unwrap())
        );
        replay(&s);
    }
}

#[test]
fn market_iv_density_retains_source_and_rejects_inconsistent_cached_values() {
    let xs = vec![-0.8, -0.3, 0.0, 0.4, 0.8];
    let iv = MarketIvSurface::new(
        vec![0.4, 1.0],
        xs.clone(),
        vec![
            0.23, 0.224, 0.22, 0.216, 0.22, 0.236, 0.23, 0.226, 0.222, 0.226,
        ],
    )
    .unwrap();
    let d = HullWhiteLsvTarget::from_market_iv(iv, vec![0.0, 0.25, 0.5, 0.75, 1.0], xs, 1e-8, 4.0)
        .unwrap();
    let s = source(
        EquityModelSpec::LocalStochasticVolatility {
            target: target(&d),
            factor: factors().remove(0),
            particles: particles(),
            forward_density: Some(d),
        },
        rates(1),
        None,
    );
    replay(&s);
    let mut v = wire(&s);
    assert!(v["model"]["equity"]["forward_density"]["market_iv"].is_object());
    v["model"]["equity"]["forward_density"]["log_densities"][0] = json!(0.123);
    assert!(
        matches!(parse(&v), Err(pricing::WireError::DomainAt { pointer, .. })
        if pointer == "/model/equity/forward_density")
    );
}

#[test]
fn source_fingerprint_covers_models_numerics_calibration_and_market() {
    let d = density();
    let s = source(
        EquityModelSpec::LocalStochasticVolatility {
            target: target(&d),
            factor: factors().remove(0),
            particles: particles(),
            forward_density: Some(d),
        },
        rates(1),
        None,
    );
    let original = wire(&s);
    for (pointer, value) in [
        ("/model/equity/factor/vol_of_vol", json!(0.31)),
        ("/model/equity/particles/seed", json!(430)),
        ("/model/equity/particles/log_bandwidth", json!(0.45)),
        ("/model/equity/particles/retain_reverse_trace", json!(false)),
        ("/model/rates/volatilities/0", json!(0.006)),
        ("/model/rates/equity_rate_correlation", json!(0.2)),
        ("/model/rates/vol_rate_correlations/0", json!(0.1)),
        ("/numerics/worker_threads", json!(3)),
        ("/numerics/reduction_block_size", json!(64)),
        ("/engine/master_seed", json!(8)),
        ("/market/spot", json!(101.0)),
    ] {
        let mut v = original.clone();
        *v.pointer_mut(pointer).unwrap() = value;
        assert_ne!(
            s.fingerprint().unwrap(),
            parse(&v).unwrap().fingerprint().unwrap(),
            "{pointer}"
        );
    }
    let sv = source(
        EquityModelSpec::StochasticVolatility {
            initial_volatility: 0.2,
            factor: factors().remove(0),
        },
        rates(1),
        Some(0.25),
    );
    let mut v = wire(&sv);
    v["numerics"]["maximum_step"] = json!(0.125);
    assert_ne!(
        sv.fingerprint().unwrap(),
        parse(&v).unwrap().fingerprint().unwrap()
    );
    v = wire(&sv);
    v["model"]["equity"]["initial_volatility"] = json!(0.21);
    assert_ne!(
        sv.fingerprint().unwrap(),
        parse(&v).unwrap().fingerprint().unwrap()
    );
}

#[test]
fn unsupported_ignored_or_malformed_inputs_are_rejected() {
    let s = source(
        EquityModelSpec::StochasticVolatility {
            initial_volatility: 0.2,
            factor: factors().remove(0),
        },
        rates(1),
        Some(0.25),
    );
    let good = wire(&s);
    for (pointer, value) in [
        ("/schema_version", json!(2)),
        ("/document_kind", json!("pricing_request")),
        ("/numerics/worker_threads", json!(0)),
        ("/numerics/maximum_step", json!(0)),
        ("/model/rates/vol_rate_correlations", json!([])),
        ("/model/rates/equity_rate_correlation", json!(1.1)),
        ("/model/equity/initial_volatility", json!(-0.2)),
        ("/model/equity/factor/mean_reversion", json!(-1)),
        ("/risk/delta", json!(true)),
        ("/risk/vega", json!(true)),
    ] {
        let mut v = good.clone();
        *v.pointer_mut(pointer).unwrap() = value;
        assert!(parse(&v).is_err(), "{pointer}");
    }
    for key in ["checkpoint_interval", "aad_tile_capacity"] {
        let mut v = good.clone();
        v["risk"][key] = json!(4);
        assert!(parse(&v).is_err(), "{key}");
    }
    let mut v = good.clone();
    v["numerics"]
        .as_object_mut()
        .unwrap()
        .remove("maximum_step");
    assert!(parse(&v).is_err());
    let mut v = good.clone();
    v["model"]["rates"] = json!({"type":"deterministic", "ignored":1});
    assert!(parse(&v).is_err());
    let mut v = good.clone();
    v["model"]["equity"]["ignored"] = json!(1);
    assert!(parse(&v).is_err());
    let mut v = good.clone();
    v["unexpected"] = json!(1);
    assert!(parse(&v).is_err());
    let mut v = wire(&source(
        EquityModelSpec::RoughVolatility(families().remove(0)),
        RateModelSpec::Deterministic,
        Some(0.25),
    ));
    v["model"]["rates"] = good["model"]["rates"].clone();
    assert!(parse(&v).is_err());
    let mut v = wire(&PricingSpecification::from_request(
        request(payload()),
        policy(),
    ));
    v["numerics"]["maximum_step"] = json!(0.25);
    assert!(parse(&v).is_err());
    let text = specification_to_json(&s).unwrap();
    let duplicate = text.replacen(
        "\"worker_threads\":2",
        "\"worker_threads\":2,\"worker_threads\":3",
        1,
    );
    assert_ne!(duplicate, text);
    assert!(parse_specification_json(duplicate.as_bytes(), JsonLimits::DEFAULT).is_err());
    assert!(parse_request_json(text.as_bytes(), JsonLimits::DEFAULT).is_err());
}

#[test]
fn lsv_requires_matching_hybrid_density_and_rejects_unused_deterministic_density() {
    let d = density();
    let equity = EquityModelSpec::LocalStochasticVolatility {
        target: target(&d),
        factor: factors().remove(0),
        particles: particles(),
        forward_density: Some(d),
    };
    let good = wire(&source(equity, rates(1), None));
    let mut v = good.clone();
    v["model"]["equity"]
        .as_object_mut()
        .unwrap()
        .remove("forward_density");
    assert!(parse(&v).is_err());
    let mut v = good.clone();
    v["model"]["rates"] = json!({"type":"deterministic"});
    assert!(parse(&v).is_err());
    let mut v = good;
    v["model"]["equity"]["target"]["local_variance_grid"]["values"][0] = json!(0.041);
    assert!(parse(&v).is_err());
}

#[test]
fn correlation_psd_is_checked_by_the_existing_compiler() {
    let s = source(
        EquityModelSpec::StochasticVolatility {
            initial_volatility: 0.2,
            factor: factors().remove(0),
        },
        rates(1),
        Some(0.25),
    );
    let mut v = wire(&s);
    v["model"]["rates"]["equity_rate_correlation"] = json!(0.99);
    v["model"]["rates"]["vol_rate_correlations"] = json!([0.99]);
    assert!(parse(&v).unwrap().compile().is_err());
}

#[test]
fn mixed_weights_do_not_drift_on_repeated_serialization() {
    let m = MixedRoughBergomi::new(
        0.2,
        -0.4,
        vec![0.1, 0.2, 0.7 + 1e-13],
        vec![0.1, 0.2, 0.3],
        ForwardVarianceCurve::constant(0.04).unwrap(),
    )
    .unwrap();
    let mut s = source(
        EquityModelSpec::RoughVolatility(m.into()),
        RateModelSpec::Deterministic,
        Some(0.25),
    );
    let original = specification_to_json(&s).unwrap();
    for _ in 0..8 {
        s = parse_specification_json(
            specification_to_json(&s).unwrap().as_bytes(),
            JsonLimits::DEFAULT,
        )
        .unwrap();
        assert_eq!(specification_to_json(&s).unwrap(), original);
    }
}

#[test]
fn golden_sources_replay_and_bounded_reader_rejects_excessive_input() {
    for text in [
        include_str!("../../../fixtures/specification-v1/black_scholes.golden.json"),
        include_str!("../../../fixtures/specification-v1/bergomi_hull_white.golden.json"),
        include_str!("../../../fixtures/specification-v1/bergomi_lsv.golden.json"),
        include_str!("../../../fixtures/specification-v1/rough_heston.golden.json"),
    ] {
        let s = parse_specification_json(text.as_bytes(), JsonLimits::DEFAULT).unwrap();
        replay(&s);
        let mut limits = JsonLimits::DEFAULT;
        limits.max_input_bytes = text.len() - 1;
        assert!(parse_specification_json(text.as_bytes(), limits).is_err());
        limits = JsonLimits::DEFAULT;
        limits.max_nesting_depth = 2;
        assert!(parse_specification_json(text.as_bytes(), limits).is_err());
        let expected: Value = serde_json::from_str(text).unwrap();
        assert_eq!(wire(&s), expected);
    }
}

#[test]
fn qmc_and_escrowed_dividends_keep_existing_sampling_and_coordinates() {
    let mut v = payload();
    v["engine"] = json!({"type":"randomized_quasi_monte_carlo", "points_per_scramble":32,
        "scramble_count":4, "master_scramble_seed":19,
        "variance_reduction":{"antithetic":true,"brownian_bridge":true}});
    v["market"]["discrete_dividends"] = json!([
        {"event_id":1,"ex_time":0.35,"quote":{"type":"fixed_cash_and_proportional","fixed_cash":3.0,"beta":0.02}},
        {"event_id":2,"ex_time":1.4,"quote":{"type":"fixed_cash","amount":2.0}}]);
    let r = request(v);
    let f = Bergomi1Factor::new(1.2, 0.3, -0.4).unwrap();
    let native = StochasticVolatilityPricingPlan::compile_bergomi(&r, f, 0.25, policy()).unwrap();
    let s = PricingSpecification::from_request(r, policy())
        .with_model(
            PricingModelSpec::new(
                EquityModelSpec::StochasticVolatility {
                    initial_volatility: 0.2,
                    factor: VolatilityFactorSpec::Bergomi(f),
                },
                RateModelSpec::Deterministic,
            ),
            Some(0.25),
        )
        .unwrap();
    assert_eq!(
        s.compile().unwrap().plan_fingerprint(),
        native.plan_fingerprint()
    );
    assert_eq!(
        s.compile().unwrap().evaluate().unwrap().output,
        SpecificationOutput::Hybrid(native.evaluate().unwrap())
    );
    replay(&s);
}
