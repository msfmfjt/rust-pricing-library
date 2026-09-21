use pricing::hull_white::HullWhiteEquityPricingPlan as Plan;
use pricing::market::{LocalVarianceGrid, MarketIvSurface};
use pricing::mc::{ExecutionPolicy, hull_white::HullWhiteLsvTarget, lsv::LsvParticleConfig};
use pricing::models::{Bergomi1Factor, HullWhite1Factor, HybridCorrelation};
use pricing::{JsonLimits, PricingRequest, parse_request_json};
use serde_json::{Value, json};

fn payload(cash: bool) -> Value {
    let mut v: Value = serde_json::from_str(include_str!(
        "../../../fixtures/v1/pricing_request.golden.json"
    ))
    .unwrap();
    v["market"]["discount_curve"] =
        json!({"curve_id":10,"times":[0.0,0.4,1.0,2.0],"discount_factors":[1.0,0.985,0.95,0.88]});
    v["market"]["dividend_curve"] =
        json!({"curve_id":11,"times":[0.0,0.7,2.0],"discount_factors":[1.0,0.99,0.96]});
    if cash {
        v["market"]["discrete_dividends"] = json!([
            {"event_id":1,"ex_time":0.5,"quote":{"type":"fixed_cash","amount":6.0}},
            {"event_id":2,"ex_time":1.5,"quote":{"type":"fixed_cash_and_proportional","fixed_cash":4.0,"beta":0.1}}
        ]);
    } else {
        // This date is deliberately absent from the default simulation grid.
        v["market"]["discrete_dividends"] = json!([
            {"event_id":1,"ex_time":0.37,"quote":{"type":"proportional","beta":0.03}}
        ]);
    }
    v
}
fn request(v: Value) -> PricingRequest {
    parse_request_json(&serde_json::to_vec(&v).unwrap(), JsonLimits::DEFAULT).unwrap()
}
fn rates(sigma: f64) -> HullWhite1Factor {
    HullWhite1Factor::new(0.2, vec![0.0, 0.45], vec![sigma, 1.4 * sigma]).unwrap()
}
fn policy(n: u32) -> ExecutionPolicy {
    ExecutionPolicy::new(n, Some(128)).unwrap()
}
fn bs(v: Value, cash: bool, workers: u32) -> Plan {
    let compile = if cash {
        Plan::compile_bs_with_cash_dividends
    } else {
        Plan::compile_bs
    };
    compile(&request(v), rates(0.016), 0.35, 0.25, policy(workers)).unwrap()
}
fn close(label: &str, aad: f64, fd: f64) {
    assert!(
        (aad - fd).abs() < 2e-6 + 2e-5 * fd.abs(),
        "{label}: AAD={aad}, FD={fd}"
    );
}
fn shock(v: &Value, name: &str, index: usize, amount: f64) -> Value {
    let mut v = v.clone();
    if name == "spot" || name == "volatility" {
        let root = if name == "spot" { "market" } else { "model" };
        let old = v[root][name].as_f64().unwrap();
        v[root][name] = json!(old + amount);
    } else {
        let old = v["market"][name]["discount_factors"][index]
            .as_f64()
            .unwrap();
        v["market"][name]["discount_factors"][index] = json!(old * amount.exp());
    }
    v
}

#[test]
fn bs_aad_reverses_cash_reserve_curve_fit_and_payment_lag() {
    for cash in [false, true] {
        let mut v = payload(cash);
        v["product"] = json!({"type":"arithmetic_asian","underlying_id":1,"currency_id":2,"strike":100.0,"notional":1.0,"side":{"type":"call"},"observations":[{"date":"2027-09-04","weight":1.0,"value":{"type":"unknown"}}],"payment_date":"2027-12-04"});
        let plan = bs(v.clone(), cash, 1);
        let risk = plan.evaluate_aad().unwrap();
        assert_eq!(
            risk.price.value.to_bits(),
            plan.evaluate().unwrap().value.to_bits()
        );
        assert!(risk.standard_errors.is_none());
        assert_eq!(risk.parameter_labels.len(), risk.derivatives.len());
        for (name, index, bar, h) in [
            ("spot", 0, risk.delta(), 1e-5),
            ("volatility", 0, risk.vega().unwrap(), 1e-6),
            (
                "discount_curve",
                1,
                risk.discount_log_df_adjoints()[1],
                1e-6,
            ),
            (
                "discount_curve",
                2,
                risk.discount_log_df_adjoints()[2],
                1e-6,
            ),
            (
                "discount_curve",
                3,
                risk.discount_log_df_adjoints()[3],
                1e-6,
            ),
            (
                "dividend_curve",
                1,
                risk.dividend_log_df_adjoints()[1],
                1e-6,
            ),
            (
                "dividend_curve",
                2,
                risk.dividend_log_df_adjoints()[2],
                1e-6,
            ),
        ] {
            let up = bs(shock(&v, name, index, h), cash, 1)
                .evaluate()
                .unwrap()
                .value;
            let down = bs(shock(&v, name, index, -h), cash, 1)
                .evaluate()
                .unwrap()
                .value;
            close(name, bar, (up - down) / (2.0 * h));
        }
        assert_eq!(risk.discount_log_df_adjoints()[0], 0.0);
        assert_eq!(risk.dividend_log_df_adjoints()[0], 0.0);
        let replay = bs(v, cash, 4).evaluate_aad().unwrap();
        assert_eq!(risk.derivatives, replay.derivatives);
    }
}

fn flat(sigma: f64) -> HullWhiteLsvTarget {
    HullWhiteLsvTarget::flat(
        sigma,
        vec![0.0, 0.25, 0.5, 0.75, 1.0],
        vec![-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0],
        1e-8,
        4.0,
    )
    .unwrap()
}
fn lsv(v: Value, target: &HullWhiteLsvTarget, cash: bool, trace: bool, workers: u32) -> Plan {
    let mut v = v;
    let g = target.grid();
    v["model"] = json!({"type":"local_volatility","local_variance_grid":{
        "time_nodes":g.time_nodes(),"log_forward_moneyness_nodes":g.log_moneyness_nodes(),
        "shape":[g.time_nodes().len(),g.log_moneyness_nodes().len()],"values":g.values(),"floor":g.floor(),"cap":g.cap()}});
    let compile = if cash {
        Plan::compile_lsv_with_cash_dividends
    } else {
        Plan::compile_lsv
    };
    compile(
        &request(v),
        target,
        Bergomi1Factor::new(2.0, 0.35, -0.5).unwrap(),
        rates(0.012),
        HybridCorrelation::new(-0.5, 0.3, -0.1).unwrap(),
        LsvParticleConfig::new(2048, 711, 0.32, 20.0, trace).unwrap(),
        policy(workers),
    )
    .unwrap()
}
fn target_shock(
    target: &HullWhiteLsvTarget,
    density: bool,
    index: usize,
    amount: f64,
) -> HullWhiteLsvTarget {
    let g = target.grid();
    let mut values = g.values().to_vec();
    let mut densities = target.log_densities().to_vec();
    if density {
        densities[index] += amount;
    } else {
        values[index] += amount;
    }
    HullWhiteLsvTarget::new(
        LocalVarianceGrid::new(
            g.time_nodes().to_vec(),
            g.log_moneyness_nodes().to_vec(),
            values,
            g.floor(),
            g.cap(),
        )
        .unwrap(),
        densities,
    )
    .unwrap()
}

#[test]
fn lsv_aad_matches_full_recalibration_for_spot_curves_and_paired_target() {
    let target = flat(0.23);
    for cash in [false, true] {
        let v = payload(cash);
        let plan = lsv(v.clone(), &target, cash, true, 1);
        let risk = plan.evaluate_aad().unwrap();
        assert_eq!(
            risk.price.value.to_bits(),
            plan.evaluate().unwrap().value.to_bits()
        );
        assert!(risk.vega().is_none());
        let calibration = plan.calibration().unwrap();
        assert!(calibration.reverse_leverage(&[1.0]).is_err());
        assert!(
            calibration
                .reverse_leverage(&vec![f64::NAN; target.grid().values().len()])
                .is_err()
        );
        let mut changed = calibration.clone();
        changed.conditional_second_moments[0] *= 1.01;
        assert!(
            changed
                .reverse_leverage(&vec![1.0; target.grid().values().len()])
                .is_err()
        );
        assert!(
            plan.calibration()
                .unwrap()
                .diagnostics
                .iter()
                .any(|d| d.fallback_nodes > 0)
        );
        assert!(
            risk.forward_log_density_adjoints()
                .iter()
                .any(|v| v.abs() > 1e-7)
        );
        for (name, index, bar, h) in [
            ("spot", 0, risk.delta(), 1e-5),
            (
                "discount_curve",
                2,
                risk.discount_log_df_adjoints()[2],
                1e-7,
            ),
            (
                "discount_curve",
                3,
                risk.discount_log_df_adjoints()[3],
                1e-7,
            ),
            (
                "dividend_curve",
                1,
                risk.dividend_log_df_adjoints()[1],
                1e-7,
            ),
        ] {
            let up = lsv(shock(&v, name, index, h), &target, cash, false, 1)
                .evaluate()
                .unwrap()
                .value;
            let down = lsv(shock(&v, name, index, -h), &target, cash, false, 1)
                .evaluate()
                .unwrap()
                .value;
            close(name, bar, (up - down) / (2.0 * h));
        }
        for (density, index) in [(false, 3), (false, 10), (false, 15), (true, 10), (true, 17)] {
            let h = 1e-7;
            let up = lsv(
                v.clone(),
                &target_shock(&target, density, index, h),
                cash,
                false,
                1,
            )
            .evaluate()
            .unwrap()
            .value;
            let down = lsv(
                v.clone(),
                &target_shock(&target, density, index, -h),
                cash,
                false,
                1,
            )
            .evaluate()
            .unwrap()
            .value;
            let bar = if density {
                risk.forward_log_density_adjoints()[index]
            } else {
                risk.local_variance_adjoints()[index]
            };
            close(
                if density { "density" } else { "variance" },
                bar,
                (up - down) / (2.0 * h),
            );
        }
        // An actual flat-smile bump moves both inputs. This checks their joint
        // chain rule rather than reporting raw local-variance risk as market Vega.
        let sigma = 0.23;
        let mut vega = 2.0 * sigma * risk.local_variance_adjoints().iter().sum::<f64>();
        for (r, &t) in target.grid().time_nodes().iter().enumerate().skip(1) {
            for (j, &x) in target.grid().log_moneyness_nodes().iter().enumerate() {
                let index = r * target.grid().log_moneyness_nodes().len() + j;
                vega += risk.forward_log_density_adjoints()[index]
                    * target.log_densities()[index]
                    * (x * x / (sigma.powi(3) * t) - sigma * t / 4.0 - 1.0 / sigma);
            }
        }
        let h = 1e-7;
        let up = lsv(v.clone(), &flat(sigma + h), cash, false, 1)
            .evaluate()
            .unwrap()
            .value;
        let down = lsv(v.clone(), &flat(sigma - h), cash, false, 1)
            .evaluate()
            .unwrap()
            .value;
        close("paired smile", vega, (up - down) / (2.0 * h));
        let no_trace = lsv(v.clone(), &target, cash, false, 1);
        assert!(matches!(
            no_trace.evaluate_aad(),
            Err(pricing::MonteCarloError::Lsv(
                pricing::mc::lsv::LsvError::ReverseTraceNotRetained
            ))
        ));
        assert_ne!(plan.plan_fingerprint(), no_trace.plan_fingerprint());
        assert_eq!(
            plan.evaluate().unwrap().value.to_bits(),
            no_trace.evaluate().unwrap().value.to_bits()
        );
        let replay = lsv(v, &target, cash, true, 4).evaluate_aad().unwrap();
        assert_eq!(risk.derivatives, replay.derivatives);
    }
}

#[test]
fn rqmc_hybrid_risk_matches_gaussian_greeks_and_worker_replay() {
    use pricing_numerics::{standard_normal_cdf as cdf, standard_normal_pdf as pdf};
    let mut v: Value = serde_json::from_str(include_str!(
        "../../../fixtures/v1/pricing_request.golden.json"
    ))
    .unwrap();
    v["engine"] = json!({"type":"randomized_quasi_monte_carlo","points_per_scramble":2048,
        "scramble_count":8,"master_scramble_seed":612,
        "variance_reduction":{"antithetic":true,"brownian_bridge":true}});
    let compile = |v: Value, workers| {
        Plan::compile_bs(&request(v), rates(0.016), 0.35, 0.25, policy(workers)).unwrap()
    };
    let plan = compile(v.clone(), 1);
    let risk = plan.evaluate_aad().unwrap();
    let covariance = rates(0.016)
        .transition(
            0.0,
            1.0,
            0.0,
            HybridCorrelation::new(0.0, 0.35, 0.0).unwrap(),
        )
        .unwrap()
        .covariance;
    let variance = 0.04 + covariance[3][3] + 0.4 * covariance[0][3];
    let forward = 100.0 * 0.98 / 0.95;
    let d1 = ((forward / 100.0_f64).ln() + 0.5 * variance) / variance.sqrt();
    let expected_delta = 0.98 * cdf(d1);
    let expected_vega = 0.95 * forward * pdf(d1) * (0.2 + covariance[0][3]) / variance.sqrt();
    let expected_dv01 = 0.95 * 100.0 * cdf(d1 - variance.sqrt()) * 1e-4;
    let errors = risk.standard_errors.as_ref().unwrap();
    assert!((risk.delta() - expected_delta).abs() < 6.0 * errors[0] + 1e-4);
    assert!((risk.vega().unwrap() - expected_vega).abs() < 6.0 * errors[1] + 1e-3);
    assert!((risk.parallel_discount_dv01() - expected_dv01).abs() < 6.0 * errors[3] * 1e-4 + 1e-7);
    assert_eq!(
        risk.price.value.to_bits(),
        plan.evaluate().unwrap().value.to_bits()
    );
    let replay = compile(v.clone(), 4).evaluate_aad().unwrap();
    assert_eq!(risk.derivatives, replay.derivatives);
    assert_eq!(risk.standard_errors, replay.standard_errors);
    v["model"]["volatility"] = json!(0.0);
    let zero = Plan::compile_bs(&request(v), rates(0.0), 0.0, 1.0, policy(1))
        .unwrap()
        .evaluate_aad()
        .unwrap();
    assert!((zero.delta() - 0.98).abs() < 1e-14);
    assert!(zero.vega().unwrap().abs() < 1e-14);
    assert!((zero.parallel_discount_dv01() - 0.0095).abs() < 1e-14);
}

#[test]
fn smoothed_cash_barrier_reverses_both_sides_of_jump() {
    let mut v = payload(true);
    v["market"]["discrete_dividends"] = json!([
        {"event_id":1,"ex_time":0.0,"quote":{"type":"fixed_cash","amount":2.0}},
        {"event_id":2,"ex_time":1.0,"quote":{"type":"fixed_cash_and_proportional","fixed_cash":10.0,"beta":0.1}}
    ]);
    v["product"] = json!({"type":"barrier","underlying_id":1,"currency_id":2,"expiry":"2027-09-04",
        "strike":80.0,"barrier":100.0,"notional":1.0,"side":{"type":"call"},
        "direction":{"type":"up"},"style":{"type":"knock_in"},
        "monitoring":{"type":"discrete"},
        "monitoring_dates":["2027-09-04"],"payment_date":"2027-09-04"});
    assert!(bs(v.clone(), true, 1).evaluate_aad().is_err());
    v["risk"]["payoff_smoothing"] = json!({"type":"compact_c2","half_width":8.0});
    let target = flat(0.23);
    for local in [false, true] {
        let compile = |v| {
            if local {
                lsv(v, &target, true, true, 1)
            } else {
                bs(v, true, 1)
            }
        };
        let plan = compile(v.clone());
        let risk = plan.evaluate_aad().unwrap();
        let h = 1e-5;
        let up = compile(shock(&v, "spot", 0, h)).evaluate().unwrap().value;
        let down = compile(shock(&v, "spot", 0, -h)).evaluate().unwrap().value;
        close("smoothed jump Delta", risk.delta(), (up - down) / (2.0 * h));
        assert_eq!(
            risk.price.value.to_bits(),
            plan.evaluate().unwrap().value.to_bits()
        );
    }
}

#[test]
fn zero_rate_and_vol_factor_aad_limit_and_rqmc_density_contract() {
    let target = flat(0.23);
    let mut v = payload(true);
    v["engine"] = json!({"type":"randomized_quasi_monte_carlo","points_per_scramble":128,
        "scramble_count":4,"master_scramble_seed":612,
        "variance_reduction":{"antithetic":true,"brownian_bridge":true}});
    let g = target.grid();
    v["model"] = json!({"type":"local_volatility","local_variance_grid":{
        "time_nodes":g.time_nodes(),"log_forward_moneyness_nodes":g.log_moneyness_nodes(),
        "shape":[g.time_nodes().len(),g.log_moneyness_nodes().len()],"values":g.values(),"floor":g.floor(),"cap":g.cap()}});
    let plan = Plan::compile_lsv_with_cash_dividends(
        &request(v.clone()),
        &target,
        Bergomi1Factor::new(2.0, 0.0, -0.5).unwrap(),
        rates(0.0),
        HybridCorrelation::new(-0.5, 0.3, -0.1).unwrap(),
        LsvParticleConfig::new(128, 711, 0.32, 5.0, true).unwrap(),
        policy(1),
    )
    .unwrap();
    let risk = plan.evaluate_aad().unwrap();
    assert!(
        risk.forward_log_density_adjoints()
            .iter()
            .all(|v| *v == 0.0)
    );
    assert!(
        risk.standard_errors
            .as_ref()
            .unwrap()
            .iter()
            .all(|v| v.is_finite() && *v >= 0.0)
    );
    assert_eq!(
        risk.local_variance_adjoints().len(),
        target.grid().values().len()
    );
    // A stochastic-rate RQMC run exercises one calibration VJP per scramble.
    let risk = lsv(v.clone(), &target, true, true, 1)
        .evaluate_aad()
        .unwrap();
    let replay = lsv(v, &target, true, true, 3).evaluate_aad().unwrap();
    assert_eq!(risk.derivatives, replay.derivatives);
    assert_eq!(risk.standard_errors, replay.standard_errors);
}

fn quoted_target(quotes: Vec<f64>) -> HullWhiteLsvTarget {
    HullWhiteLsvTarget::from_market_iv(
        MarketIvSurface::new(vec![0.4, 1.0], vec![-1.0, -0.3, 0.0, 0.4, 1.0], quotes).unwrap(),
        vec![0.0, 0.25, 0.5, 0.75, 1.0],
        vec![-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0],
        1e-8,
        4.0,
    )
    .unwrap()
}
fn quotes() -> Vec<f64> {
    vec![
        0.23, 0.224, 0.22, 0.216, 0.22, 0.236, 0.23, 0.226, 0.222, 0.226,
    ]
}

#[test]
fn paired_market_iv_transpose_includes_density_and_time_zero_variance() {
    let q = quotes();
    let target = quoted_target(q.clone());
    let n = target.grid().values().len();
    let vb = (0..n).map(|i| 0.7 - (i as f64) * 0.02).collect::<Vec<_>>();
    let db = (0..n).map(|i| -0.2 + (i as f64) * 0.01).collect::<Vec<_>>();
    let bar = target.reverse_market_iv(&vb, &db).unwrap();
    let contract = |s: &HullWhiteLsvTarget| -> f64 {
        s.grid()
            .values()
            .iter()
            .zip(&vb)
            .map(|(v, b)| v * b)
            .sum::<f64>()
            + s.log_densities()
                .iter()
                .zip(&db)
                .map(|(v, b)| v * b)
                .sum::<f64>()
    };
    for (i, &b) in bar.iter().enumerate() {
        let mut up = q.clone();
        let mut down = q.clone();
        up[i] += 1e-6;
        down[i] -= 1e-6;
        close(
            "paired IV transpose",
            b,
            (contract(&quoted_target(up)) - contract(&quoted_target(down))) / 2e-6,
        );
    }
    assert!(target.reverse_market_iv(&[1.0], &db).is_err());
    assert!(flat(0.2).reverse_market_iv(&vb, &db).is_err());
    let zero = target
        .reverse_market_iv(&vec![0.0; n], &vec![0.0; n])
        .unwrap();
    assert_eq!(zero, vec![0.0; q.len()]);
    let mut time_zero_density = vec![0.0; n];
    time_zero_density[..7].fill(1.0);
    assert_eq!(
        target
            .reverse_market_iv(&vec![0.0; n], &time_zero_density)
            .unwrap(),
        zero
    );
}

#[test]
fn market_iv_vegakt_matches_every_recalibrated_bucket_with_cash_and_proportional_dividends() {
    let q = quotes();
    let target = quoted_target(q.clone());
    for cash in [false, true] {
        let v = payload(cash);
        let plan = lsv(v.clone(), &target, cash, true, 1);
        let risk = plan.evaluate_aad().unwrap();
        let raw = risk.vega_kt_raw().unwrap();
        assert_eq!(risk.price.value, plan.evaluate().unwrap().value);
        assert_eq!(raw.len(), q.len());
        assert_eq!(
            risk.forward_log_density_adjoints().len(),
            target.grid().values().len()
        );
        assert_eq!(risk.derivatives.len(), risk.parameter_labels.len());
        assert_eq!(risk.parameter_labels.last().unwrap(), "parallel_market_iv");
        assert_eq!(risk.vega_kt_maturity_nodes.as_ref(), &[0.4, 1.0]);
        assert_eq!(risk.vega_kt_implied_volatilities.as_ref(), q);
        assert!(risk.vega_kt_standard_errors().is_none());
        close("bucket sum", risk.vega().unwrap(), raw.iter().sum());
        for (a, b) in risk.vega_kt_market_scaled().unwrap().iter().zip(raw) {
            assert_eq!(*a, 0.01 * b);
        }
        let h = 1e-7;
        for (i, &b) in raw.iter().enumerate() {
            let mut up = q.clone();
            let mut down = q.clone();
            up[i] += h;
            down[i] -= h;
            let price = |qs| {
                lsv(v.clone(), &quoted_target(qs), cash, false, 1)
                    .evaluate()
                    .unwrap()
                    .value
            };
            close(
                &format!("market IV[{i}] cash={cash}"),
                b,
                (price(up) - price(down)) / (2.0 * h),
            );
        }
        let price = |h: f64| {
            lsv(
                v.clone(),
                &quoted_target(q.iter().map(|q| q + h).collect()),
                cash,
                false,
                1,
            )
            .evaluate()
            .unwrap()
            .value
        };
        close(
            "parallel market IV",
            risk.vega().unwrap(),
            (price(h) - price(-h)) / (2.0 * h),
        );
        let variance_only = target
            .reverse_market_iv(
                risk.local_variance_adjoints(),
                &vec![0.0; target.grid().values().len()],
            )
            .unwrap();
        assert!(
            raw.iter()
                .zip(&variance_only)
                .any(|(a, b)| (a - b).abs() > 1e-5)
        );
        let replay = lsv(v.clone(), &target, cash, true, 4)
            .evaluate_aad()
            .unwrap();
        assert_eq!(risk.derivatives, replay.derivatives);
        let explicit =
            HullWhiteLsvTarget::new(target.grid().clone(), target.log_densities().to_vec())
                .unwrap();
        let untracked = lsv(v, &explicit, cash, true, 1);
        assert_eq!(untracked.evaluate().unwrap().value, risk.price.value);
        assert_ne!(plan.plan_fingerprint(), untracked.plan_fingerprint());
        let risk = untracked.evaluate_aad().unwrap();
        assert!(risk.vega_kt_raw().is_none() && risk.vega().is_none());
    }
}

#[test]
fn market_iv_rqmc_uncertainty_and_deterministic_factor_limit() {
    let target = quoted_target(vec![0.23; 10]);
    let mut v = payload(true);
    v["engine"] = json!({"type":"randomized_quasi_monte_carlo","points_per_scramble":128,"scramble_count":4,"master_scramble_seed":612,"variance_reduction":{"antithetic":true,"brownian_bridge":true}});
    let plan = lsv(v.clone(), &target, true, true, 1);
    let risk = plan.evaluate_aad().unwrap();
    let replay = lsv(v.clone(), &target, true, true, 3)
        .evaluate_aad()
        .unwrap();
    assert_eq!(risk.derivatives, replay.derivatives);
    assert_eq!(risk.standard_errors, replay.standard_errors);
    assert_eq!(risk.vega_kt_standard_errors().unwrap().len(), 10);
    assert!(risk.parallel_vega_standard_error().unwrap().is_finite());
    assert!(risk.parallel_vega_standard_error().unwrap() > 0.0);
    assert_eq!(risk.price.value, plan.evaluate().unwrap().value);
    let h = 1e-7;
    let price = |h: f64| {
        lsv(
            v.clone(),
            &quoted_target(vec![0.23 + h; 10]),
            true,
            false,
            1,
        )
        .evaluate()
        .unwrap()
        .value
    };
    close(
        "RQMC parallel IV",
        risk.vega().unwrap(),
        (price(h) - price(-h)) / (2.0 * h),
    );
    let g = target.grid();
    v["model"] = json!({"type":"local_volatility","local_variance_grid":{"time_nodes":g.time_nodes(),"log_forward_moneyness_nodes":g.log_moneyness_nodes(),"shape":[5,7],"values":g.values(),"floor":g.floor(),"cap":g.cap()}});
    let deterministic = Plan::compile_lsv_with_cash_dividends(
        &request(v),
        &target,
        Bergomi1Factor::new(2.0, 0.0, 0.0).unwrap(),
        rates(0.0),
        HybridCorrelation::new(0.0, 0.0, 0.0).unwrap(),
        LsvParticleConfig::new(128, 711, 0.32, 5.0, true).unwrap(),
        policy(1),
    )
    .unwrap()
    .evaluate_aad()
    .unwrap();
    assert!(
        deterministic
            .forward_log_density_adjoints()
            .iter()
            .all(|v| *v == 0.0)
    );
    assert!(deterministic.vega().unwrap() > 0.0);
}

#[test]
fn two_factor_hw_aad_recalibrates_both_targets_quotes_curves_and_dividend_modes() {
    let compile = |mut v: Value, target: &HullWhiteLsvTarget, escrow: bool, trace, workers| {
        let g = target.grid();
        v["engine"]["independent_sampling_units"] = json!(256);
        v["model"] = json!({"type":"local_volatility","local_variance_grid":{"time_nodes":g.time_nodes(),"log_forward_moneyness_nodes":g.log_moneyness_nodes(),"shape":[g.time_nodes().len(),g.log_moneyness_nodes().len()],"values":g.values(),"floor":g.floor(),"cap":g.cap()}});
        let f = pricing::models::Bergomi2Factor::new([2.0, 0.15], 0.35, 0.4, [-0.5, -0.2], 0.4)
            .unwrap();
        let build = if escrow {
            Plan::compile_lsv_two_factor_with_cash_dividends
        } else {
            Plan::compile_lsv_two_factor
        };
        build(
            &request(v),
            target,
            f,
            rates(0.012),
            0.3,
            [-0.1, 0.05],
            LsvParticleConfig::new(2048, 711, 0.32, 20.0, trace).unwrap(),
            policy(workers),
        )
        .unwrap()
    };
    let target = quoted_target(quotes());
    let v = payload(true);
    for escrow in [false, true] {
        let plan = compile(v.clone(), &target, escrow, true, 1);
        assert_eq!(plan.random_factor_count(), 5);
        let risk = plan.evaluate_aad().unwrap();
        assert_eq!(risk.price.value, plan.evaluate().unwrap().value);
        assert_eq!(
            risk.derivatives,
            compile(v.clone(), &target, escrow, true, 3)
                .evaluate_aad()
                .unwrap()
                .derivatives
        );
        assert!(
            compile(v.clone(), &target, escrow, false, 1)
                .evaluate_aad()
                .is_err()
        );
        for (name, index, bar, h) in [
            ("spot", 0, risk.delta(), 1e-5),
            (
                "discount_curve",
                2,
                risk.discount_log_df_adjoints()[2],
                1e-7,
            ),
            (
                "dividend_curve",
                1,
                risk.dividend_log_df_adjoints()[1],
                1e-7,
            ),
        ] {
            let up = compile(shock(&v, name, index, h), &target, escrow, false, 1)
                .evaluate()
                .unwrap()
                .value;
            let dn = compile(shock(&v, name, index, -h), &target, escrow, false, 1)
                .evaluate()
                .unwrap()
                .value;
            close(name, bar, (up - dn) / (2.0 * h));
        }
        for density in [false, true] {
            let index = 10;
            let h = 1e-7;
            let up = compile(
                v.clone(),
                &target_shock(&target, density, index, h),
                escrow,
                false,
                1,
            )
            .evaluate()
            .unwrap()
            .value;
            let dn = compile(
                v.clone(),
                &target_shock(&target, density, index, -h),
                escrow,
                false,
                1,
            )
            .evaluate()
            .unwrap()
            .value;
            let bar = if density {
                risk.forward_log_density_adjoints()[index]
            } else {
                risk.local_variance_adjoints()[index]
            };
            close("two-factor paired target", bar, (up - dn) / (2.0 * h));
        }
        for j in 0..quotes().len() {
            let shifted = |h| {
                let mut q = quotes();
                q[j] += h;
                quoted_target(q)
            };
            let h = 1e-7;
            let up = compile(v.clone(), &shifted(h), escrow, false, 1)
                .evaluate()
                .unwrap()
                .value;
            let dn = compile(v.clone(), &shifted(-h), escrow, false, 1)
                .evaluate()
                .unwrap()
                .value;
            close(
                "two-factor market IV",
                risk.vega_kt_raw().unwrap()[j],
                (up - dn) / (2.0 * h),
            );
        }
    }
}
