//! Raw correlation entry partials versus full-recompile prices.
use pricing::mc::ExecutionPolicy;
use pricing::models::{Bergomi1Factor, Bergomi2Factor};
use pricing::stochastic_dividends::{BuehlerDividendModel, StochasticDividendPricingPlan as Plan};
use pricing::{JsonLimits, parse_request_json};
use serde_json::{Value, json};

// [S-D, S-V1, S-V2, D-V1, D-V2, V1-V2].
const RHO: [f64; 6] = [-0.25, -0.4, -0.2, 0.15, -0.1, 0.3];
fn payload(qmc: bool, seed: u64) -> Value {
    let mut v: Value = serde_json::from_str(include_str!(
        "../../../fixtures/v1/pricing_request.golden.json"
    ))
    .unwrap();
    v["schema_version"] = json!(2);
    v["market"]["discrete_dividends"] = json!([
        {"event_id":1,"ex_time":0.5,"quote":{"type":"fixed_cash","amount":4.0}},
        {"event_id":2,"ex_time":1.0,"quote":{"type":"fixed_cash","amount":3.0}},
        {"event_id":3,"ex_time":1.4,"quote":{"type":"fixed_cash","amount":8.0}}]);
    v["engine"] = if qmc {
        json!({"type":"randomized_quasi_monte_carlo","points_per_scramble":128,
            "scramble_count":4,"master_scramble_seed":seed,
            "variance_reduction":{"antithetic":true,"brownian_bridge":true}})
    } else {
        json!({"type":"pseudo_monte_carlo","independent_sampling_units":512,"master_seed":seed,
            "variance_reduction":{"antithetic":true,"brownian_bridge":false}})
    };
    v
}
fn compile(v: &Value, family: usize, rho: [f64; 6], b: [f64; 4], workers: u32) -> Plan {
    let r = parse_request_json(&serde_json::to_vec(v).unwrap(), JsonLimits::DEFAULT).unwrap();
    let d = BuehlerDividendModel::new(0.7, 0.6, 0.35, rho[0]).unwrap();
    let policy = ExecutionPolicy::new(workers, Some(64)).unwrap();
    match family {
        0 => Plan::compile_bs(&r, d, 0.125, policy),
        1 => Plan::compile_bergomi(
            &r,
            d,
            Bergomi1Factor::new(b[0], b[2], rho[1]).unwrap(),
            rho[3],
            0.125,
            policy,
        ),
        _ => Plan::compile_bergomi_two_factor(
            &r,
            d,
            Bergomi2Factor::new([b[0], b[1]], b[2], b[3], [rho[1], rho[2]], rho[5]).unwrap(),
            [rho[3], rho[4]],
            0.125,
            policy,
        ),
    }
    .unwrap()
}
fn check(v: &Value, family: usize, rho: [f64; 6], b: [f64; 4]) {
    let plan = compile(v, family, rho, b, 1);
    let prefix = if family == 0 {
        plan.evaluate_aad()
    } else {
        plan.evaluate_bergomi_aad()
    }
    .unwrap();
    let risk = plan.evaluate_correlation_aad().unwrap();
    let start = prefix.derivatives.len();
    assert_eq!(risk.price, plan.evaluate().unwrap());
    assert_eq!(risk.price, prefix.price);
    assert_eq!(&risk.parameter_labels[..start], &*prefix.parameter_labels);
    assert_eq!(&risk.derivatives[..start], &*prefix.derivatives);
    assert_eq!(&risk.standard_errors[..start], &*prefix.standard_errors);
    assert_eq!(risk.cash_mean_adjoints(), prefix.cash_mean_adjoints());
    assert_eq!(risk.discount_node_dv01(), prefix.discount_node_dv01());
    assert_eq!(risk.repo_spread_node_dv01(), prefix.repo_spread_node_dv01());
    assert_eq!(risk.method, "buehler-joint-correlation-reverse-v1");
    let active: &[usize] = match family {
        0 => &[0],
        1 => &[0, 1, 3],
        _ => &[0, 1, 2, 3, 4, 5],
    };
    let labels = [
        "equity_dividend_correlation",
        "spot_volatility_correlation[0]",
        "spot_volatility_correlation[1]",
        "dividend_volatility_correlation[0]",
        "dividend_volatility_correlation[1]",
        "volatility_factor_correlation",
    ];
    assert_eq!(risk.derivatives.len(), start + active.len());
    assert!(
        risk.standard_errors
            .iter()
            .all(|v| v.is_finite() && *v >= 0.0)
    );
    for (j, &p) in active.iter().enumerate() {
        assert_eq!(risk.parameter_labels[start + j], labels[p]);
        for h in [1e-5, 1e-6] {
            let mut plus = rho;
            plus[p] += h;
            let mut minus = rho;
            minus[p] -= h;
            let fd = (compile(v, family, plus, b, 1).evaluate().unwrap().value
                - compile(v, family, minus, b, 1).evaluate().unwrap().value)
                / (2.0 * h);
            let aad = risk.derivatives[start + j];
            let budget = 3e-5 + 2e-5 * aad.abs().max(fd.abs());
            assert!(
                (aad - fd).abs() <= budget,
                "family={family}, p={p}, h={h}, AAD={aad}, FD={fd}, budget={budget}"
            );
        }
    }
    let replay = compile(v, family, rho, b, 3)
        .evaluate_correlation_aad()
        .unwrap();
    assert_eq!(risk.derivatives, replay.derivatives);
    assert_eq!(risk.standard_errors, replay.standard_errors);
    assert_eq!(risk.price.value, replay.price.value);
}
#[test]
fn all_correlations_match_recompiled_prices_and_preserve_aad_prefix() {
    for family in [0, 1, 2] {
        for qmc in [false, true] {
            for seed in [731, 912] {
                check(&payload(qmc, seed), family, RHO, [0.8, 2.1, 0.3, 0.35]);
            }
        }
    }
}
#[test]
fn correlations_include_asian_payment_and_smoothed_pre_post_cash_barrier() {
    let mut v = payload(true, 344);
    v["product"] = json!({"type":"arithmetic_asian","underlying_id":1,"currency_id":2,
        "strike":90.0,"notional":1.0,"side":{"type":"call"},
        "observations":[{"date":"2027-03-05","weight":0.4,"value":{"type":"unknown"}},
            {"date":"2027-09-04","weight":0.6,"value":{"type":"unknown"}}],"payment_date":"2027-12-04"});
    for family in [0, 1, 2] {
        check(&v, family, RHO, [0.8, 2.1, 0.3, 0.35]);
    }
    v["product"] = json!({"type":"barrier","underlying_id":1,"currency_id":2,
        "expiry":"2027-09-04","strike":80.0,"barrier":100.0,"notional":1.0,
        "side":{"type":"call"},"direction":{"type":"up"},"style":{"type":"knock_in"},
        "monitoring":{"type":"discrete"},"monitoring_dates":["2027-09-04"],"payment_date":"2027-12-04"});
    for family in [0, 1, 2] {
        assert!(
            compile(&v, family, RHO, [0.8, 2.1, 0.3, 0.35], 1)
                .evaluate_correlation_aad()
                .is_err()
        );
    }
    v["risk"]["payoff_smoothing"] = json!({"type":"compact_c2","half_width":8.0});
    for family in [0, 1, 2] {
        check(&v, family, RHO, [0.8, 2.1, 0.3, 0.35]);
    }
}
#[test]
fn zero_correlations_and_zero_loadings_do_not_divide_by_parameters() {
    let mut v = payload(true, 992);
    v["product"]["strike"] = json!(60.0);
    for family in [0, 1, 2] {
        check(&v, family, [0.0; 6], [0.0, 0.0, 0.3, 0.35]);
    }
    for family in [1, 2] {
        for b in [
            [0.8, 2.1, 0.0, 0.35],
            [0.8, 2.1, 0.3, 0.0],
            [0.8, 2.1, 0.3, 1.0],
        ] {
            check(&v, family, RHO, b);
        }
        let b = [0.8, 2.1, 0.0, 0.35];
        let p = compile(&v, family, RHO, b, 1);
        let prefix = p.evaluate_bergomi_aad().unwrap().derivatives.len();
        let r = p.evaluate_correlation_aad().unwrap();
        // No vol-of-vol: only the direct S-D driver rotation can contribute.
        assert!(r.derivatives[prefix + 1..].iter().all(|v| *v == 0.0));
        assert!(r.standard_errors[prefix + 1..].iter().all(|v| *v == 0.0));
    }
}
#[test]
fn instantaneous_singular_domain_is_rejected_even_when_ou_covariance_is_spd() {
    let v = payload(false, 442);
    for family in [0, 1, 2] {
        for sd in [1.0, -1.0, 1.0 - 1e-12] {
            let rho = [sd, 0.0, 0.0, 0.0, 0.0, 0.0];
            let p = compile(&v, family, rho, [0.8, 2.1, 0.3, 0.35], 1);
            assert!(p.evaluate().is_ok());
            assert!(p.evaluate_aad().is_ok());
            assert!(
                p.evaluate_correlation_aad()
                    .unwrap_err()
                    .to_string()
                    .contains("correlation AAD requires")
            );
        }
    }
    for vv in [1.0, 1.0 - 1e-12] {
        let rho = [-0.25, -0.4, -0.4, 0.15, 0.15, vv];
        let p = compile(&v, 2, rho, [0.8, 2.1, 0.3, 0.35], 1);
        assert!(p.evaluate().is_ok());
        assert!(p.evaluate_aad().is_ok());
        // Unequal kernels make the integrated OU covariance SPD. This does NOT
        // make independent instantaneous correlation entry bumps admissible.
        assert!(p.evaluate_bergomi_aad().is_ok());
        assert!(
            p.evaluate_correlation_aad()
                .unwrap_err()
                .to_string()
                .contains("instantaneous")
        );
    }
}
