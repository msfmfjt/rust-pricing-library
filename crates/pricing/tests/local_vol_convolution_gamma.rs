use pricing::mc::ExecutionPolicy;
use pricing::risk::{VegaKtConvolutionPlan, VegaKtGammaAtoms};
use pricing::{JsonLimits, PricingPlan, parse_request_json};
use serde_json::{Value, json};

fn nodes(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| -0.6 + 1.2 * i as f64 / (n - 1) as f64)
        .collect()
}
fn strikes(n: usize) -> Vec<f64> {
    nodes(n).iter().map(|x| 100. * x.exp()).collect()
}
fn operator(lambda: f64) -> VegaKtConvolutionPlan {
    VegaKtConvolutionPlan::new(100., 0.02, 0.01, 0.05, &nodes(17), strikes(17), lambda).unwrap()
}
fn norm(x: &[f64]) -> f64 {
    x.iter().map(|x| x * x).sum::<f64>().sqrt()
}

#[test]
fn analytic_conditional_operator_and_inverse_match_independent_quadrature_and_svd() {
    let f: Value = serde_json::from_str(include_str!(
        "../../../fixtures/local-vol/convolution-gamma-v0.1.json"
    ))
    .unwrap();
    let array = |key: &str| {
        f[key]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect::<Vec<_>>()
    };
    let op = VegaKtConvolutionPlan::new(
        f["spot"].as_f64().unwrap(),
        f["source_variance"].as_f64().unwrap(),
        f["step_variance"].as_f64().unwrap(),
        f["sigma_dt"].as_f64().unwrap(),
        &array("source_log_nodes"),
        array("gamma_strikes"),
        f["relative_regularization"].as_f64().unwrap(),
    )
    .unwrap();
    for (row, reference) in op
        .forward_matrix()
        .iter()
        .zip(f["forward_matrix"].as_array().unwrap())
    {
        for (&value, reference) in row.iter().zip(reference.as_array().unwrap()) {
            assert!((value - reference.as_f64().unwrap()).abs() < 1e-10);
        }
    }
    let q = op.recover(&array("local_vega")).unwrap();
    for (&value, reference) in q.iter().zip(array("recovered_masses")) {
        assert!((value - reference).abs() < 2e-8);
    }
}

#[test]
fn exact_operator_preserves_source_hats_and_signed_terminal_atoms() {
    let op = operator(1e-4);
    let mut q = vec![0.; 17];
    q[8] = 1.;
    q[10] = -0.4;
    let b = op.reconstruct(&q).unwrap();
    let exact = VegaKtGammaAtoms::new(vec![
        (op.gamma_strikes()[8], 1.),
        (op.gamma_strikes()[10], -0.4),
    ])
    .unwrap()
    .expected_source_hat_vegas(100., 0.02, 0.01, 0.05, &nodes(17))
    .unwrap();
    for (a, b) in b.iter().zip(exact) {
        assert!((a - b).abs() < 1e-12);
    }
    let recovered = op.recover(&b).unwrap();
    assert!(recovered[10] < 0.);
    let diff: Vec<_> = op
        .reconstruct(&recovered)
        .unwrap()
        .iter()
        .zip(&b)
        .map(|(a, b)| a - b)
        .collect();
    assert!(norm(&diff) / norm(&b) < 2e-4);
}

#[test]
fn ridge_tradeoff_is_audited_and_linear_for_signed_inputs() {
    let weak = operator(1e-4);
    let strong = operator(0.1);
    assert!(weak.inverse_noise_gain_bound() > strong.inverse_noise_gain_bound());
    assert!(weak.resolution_trace() > strong.resolution_trace());
    let b: Vec<_> = (0..17).map(|i| (i as f64).sin()).collect();
    let neg = weak
        .recover(&b.iter().map(|v| -2. * v).collect::<Vec<_>>())
        .unwrap();
    let q = weak.recover(&b).unwrap();
    for (a, b) in neg.iter().zip(q) {
        assert!((a + 2. * b).abs() < 1e-10);
    }
    let mut atom = vec![0.; 17];
    atom[8] = 1.;
    let b = weak.reconstruct(&atom).unwrap();
    let error = |p: &VegaKtConvolutionPlan| {
        norm(
            &p.reconstruct(&p.recover(&b).unwrap())
                .unwrap()
                .iter()
                .zip(&b)
                .map(|(a, b)| a - b)
                .collect::<Vec<_>>(),
        )
    };
    assert!(error(&weak) < error(&strong));
}

#[test]
fn rejects_initial_singularity_nonfinite_and_unbounded_inverse_requests() {
    for lambda in [0., 1e-8, f64::NAN, 2.] {
        assert!(
            VegaKtConvolutionPlan::new(100., 0.02, 0.01, 0.05, &nodes(17), strikes(17), lambda)
                .is_err()
        );
    }
    assert!(
        VegaKtConvolutionPlan::new(100., 0., 0.01, 0.05, &nodes(17), strikes(17), 0.01).is_err()
    );
    assert!(
        VegaKtConvolutionPlan::new(100., 0.02, 0.01, 0.05, &nodes(130), strikes(17), 0.01).is_err()
    );
    assert!(operator(0.01).recover(&[0.; 2]).is_err());
    assert!(operator(0.01).recover(&[f64::NAN; 17]).is_err());
}

fn payload(kind: &str) -> Value {
    let mut p: Value = serde_json::from_str(include_str!(
        "../../../fixtures/v3/pricing_request.golden.json"
    ))
    .unwrap();
    let dates = ["2026-12-04", "2027-03-04", "2027-09-04"];
    let times = vec![91. / 365., 181. / 365., 1.];
    let x = nodes(9);
    p["market"]["discount_curve"]["discount_factors"] = json!([1., 1.]);
    p["market"]["dividend_curve"]["discount_factors"] = json!([1., 1.]);
    p["model"] = json!({"type":"local_volatility","local_variance_grid":{"time_nodes":[0.,times[0],times[1],times[2]],"log_forward_moneyness_nodes":x,"shape":[4,9],"values":vec![0.04;36],"floor":1e-8,"cap":4.},"reporting_iv_basis":{"maturity_nodes":times,"log_forward_moneyness_nodes":x,"shape":[3,9],"implied_volatilities":vec![0.2;27]}});
    p["risk"] = json!({"vega":true,"delta":false,"smile_dynamics":{"type":"sticky_log_moneyness"}});
    p["engine"]["independent_sampling_units"] = json!(128);
    if kind == "asian" {
        p["product"] = json!({"type":"arithmetic_asian","underlying_id":1,"currency_id":2,"strike":100.,"notional":1.,"side":{"type":"call"},"observations":[{"date":"2026-09-03","weight":0.25,"value":{"type":"known","fixing":98.}},{"date":dates[0],"weight":0.25,"value":{"type":"unknown"}},{"date":dates[1],"weight":0.25,"value":{"type":"unknown"}},{"date":dates[2],"weight":0.25,"value":{"type":"unknown"}}],"payment_date":dates[2]});
    }
    if kind == "lookback" {
        p["product"] = json!({"type":"fixed_lookback","underlying_id":1,"currency_id":2,"strike":100.,"notional":1.,"side":{"type":"call"},"monitoring_dates":dates,"payment_date":dates[2]});
    }
    p
}
fn compile(p: &Value, w: u32) -> PricingPlan {
    PricingPlan::compile(
        &parse_request_json(&serde_json::to_vec(p).unwrap(), JsonLimits::DEFAULT).unwrap(),
        ExecutionPolicy::new(w, Some(32)).unwrap(),
    )
    .unwrap()
}
#[test]
fn compiled_asian_and_lookback_recover_rows_retain_uncertainty_and_replay() {
    for kind in ["call", "asian", "lookback"] {
        let p = payload(kind);
        let plan = compile(&p, 1);
        let before = plan.evaluate().unwrap();
        let d = plan
            .evaluate_vega_kt_gamma_recovery_diagnostic(strikes(9), 0.01)
            .unwrap();
        assert_eq!(d.rows.len(), 2);
        let total = d.unrecovered_initial_local_vega.mean()
            + d.rows
                .iter()
                .map(|r| r.local_vega_total.mean())
                .sum::<f64>();
        assert!(
            (total
                - before
                    .pricing_result
                    .risks
                    .vega
                    .as_ref()
                    .unwrap()
                    .raw()
                    .value()
                    .get())
            .abs()
                < 1e-11
        );
        for r in &d.rows {
            assert!(r.gamma_masses.iter().all(|m| m.standard_error().is_some()));
            assert!(
                (r.reconstructed_total.mean()
                    - r.local_vega_total.mean()
                    - r.total_difference.mean())
                .abs()
                    < 1e-11
            );
        }
        let other = compile(&p, 4)
            .evaluate_vega_kt_gamma_recovery_diagnostic(strikes(9), 0.01)
            .unwrap();
        assert_eq!(d.rows, other.rows);
        assert_eq!(
            d.unrecovered_initial_local_vega,
            other.unrecovered_initial_local_vega
        );
        assert_eq!(plan.evaluate().unwrap(), before);
    }
}
#[test]
fn nonflat_model_is_not_silently_flattened() {
    let mut p = payload("asian");
    p["model"]["local_variance_grid"]["values"][1] = json!(0.05);
    assert!(
        compile(&p, 1)
            .evaluate_vega_kt_gamma_recovery_diagnostic(strikes(9), 0.01)
            .is_err()
    );
}

#[test]
fn initial_measure_recovers_conditional_payoff_and_preserves_native_samples() {
    for kind in ["call", "asian", "lookback"] {
        let mut p = payload(kind);
        // Spatially flat term structure, unequal time intervals, three fixings.
        p["model"]["local_variance_grid"]["values"] = json!(
            [
                vec![0.0289; 9],
                vec![0.0676; 9],
                vec![0.0441; 9],
                vec![0.0441; 9]
            ]
            .concat()
        );
        let plan = compile(&p, 1);
        let d = plan
            .evaluate_vega_kt_initial_gamma_diagnostic(vec![85., 100., 115.])
            .unwrap();
        let old = plan
            .evaluate_vega_kt_gamma_recovery_diagnostic(strikes(9), 0.01)
            .unwrap();
        assert_eq!(
            d.native_initial_local_vega,
            old.unrecovered_initial_local_vega
        );
        assert!(d.payoff_reconstruction_difference.mean().abs() < 1e-12);
        assert!(d.payoff_reconstruction_difference.standard_error().unwrap() < 1e-12);
        assert!(
            (d.gamma_bin_masses.iter().map(|m| m.mean()).sum::<f64>() - d.total_gamma_mass.mean())
                .abs()
                < 1e-12
        );
        assert!(
            (d.gamma_bin_first_step_vegas
                .iter()
                .map(|m| m.mean())
                .sum::<f64>()
                - d.recovered_initial_local_vega.mean())
            .abs()
                < 1e-12
        );
        let finer = plan
            .evaluate_vega_kt_initial_gamma_diagnostic(strikes(33))
            .unwrap();
        assert_eq!(
            d.recovered_initial_local_vega,
            finer.recovered_initial_local_vega
        );
        assert_eq!(d.total_gamma_mass, finer.total_gamma_mass);
        assert_eq!(d.conditional_price, finer.conditional_price);
        let mut other = compile(&p, 4)
            .evaluate_vega_kt_initial_gamma_diagnostic(vec![85., 100., 115.])
            .unwrap();
        other.plan_fingerprint = d.plan_fingerprint;
        assert_eq!(d, other);
    }
}

#[test]
fn initial_measure_rejects_nonflat_and_invalid_bin_domain() {
    let p = payload("asian");
    for bins in [
        vec![],
        vec![0., 100.],
        vec![100., 100.],
        vec![100., 90.],
        vec![f64::NAN],
        vec![f64::INFINITY],
        vec![1.; 130],
    ] {
        assert!(
            compile(&p, 1)
                .evaluate_vega_kt_initial_gamma_diagnostic(bins)
                .is_err()
        );
    }
    let mut p = p;
    p["model"]["local_variance_grid"]["values"][1] = json!(0.05);
    assert!(
        compile(&p, 1)
            .evaluate_vega_kt_initial_gamma_diagnostic(vec![100.])
            .is_err()
    );
}

#[test]
fn split_sample_gamma_selection_uses_disjoint_counters_and_preserves_union_moments() {
    use pricing::risk::VegaKtGammaRecoveryPolicy;
    let mut p = payload("asian");
    let policy = VegaKtGammaRecoveryPolicy::new(vec![0.01, 0.001], 128, 0.9, 0.99, 1.).unwrap();
    let plan = compile(&p, 1);
    let report = plan
        .evaluate_vega_kt_gamma_recovery_audit(strikes(9), &policy)
        .unwrap();
    assert_eq!(
        (report.pilot_index_start, report.pilot_index_end_exclusive),
        (128, 256)
    );
    let pilot = &report.pilot_candidates[0].diagnostic;
    let val = plan
        .evaluate_vega_kt_gamma_recovery_diagnostic(strikes(9), 0.01)
        .unwrap();
    p["engine"]["independent_sampling_units"] = json!(256);
    let union = compile(&p, 1)
        .evaluate_vega_kt_gamma_recovery_diagnostic(strikes(9), 0.01)
        .unwrap();
    for ((a, b), c) in val.rows.iter().zip(&pilot.rows).zip(&union.rows) {
        for ((a, b), c) in a
            .local_vega
            .iter()
            .chain(&a.gamma_masses)
            .zip(b.local_vega.iter().chain(&b.gamma_masses))
            .zip(c.local_vega.iter().chain(&c.gamma_masses))
        {
            let mean = 0.5 * a.mean() + 0.5 * b.mean();
            let variance = (127. * a.sample_variance().unwrap()
                + 127. * b.sample_variance().unwrap()
                + 64. * (a.mean() - b.mean()).powi(2))
                / 255.;
            assert!((mean - c.mean()).abs() < 1e-11);
            assert!(
                (variance - c.sample_variance().unwrap()).abs() < 1e-8 * variance.abs().max(1.)
            );
        }
    }
    let choice = report
        .pilot_candidates
        .iter()
        .find(|c| c.passed)
        .map(|c| c.diagnostic.relative_regularization);
    assert_eq!(report.selected_regularization, choice);
    assert_eq!(
        report.initial,
        plan.evaluate_vega_kt_initial_gamma_diagnostic(strikes(9))
            .unwrap()
    );
    if let Some(v) = report.validation {
        assert_eq!(
            v.diagnostic,
            plan.evaluate_vega_kt_gamma_recovery_diagnostic(strikes(9), choice.unwrap())
                .unwrap()
        );
    }
}

#[test]
fn split_sample_policy_rejects_invalid_and_preserves_no_feasible_result() {
    use pricing::risk::{VegaKtGammaRecoveryPolicy, VegaKtGammaRecoveryStatus};
    for candidates in [
        vec![],
        vec![0.001, 0.01],
        vec![0.01, 0.01],
        vec![0.],
        vec![f64::NAN],
        vec![1.; 9],
    ] {
        assert!(VegaKtGammaRecoveryPolicy::new(candidates, 128, 0.03, 0.25, 2.).is_err());
    }
    for p in [
        (1, 0.03, 0.25, 2.),
        (128, 0., 0.25, 2.),
        (128, 0.03, 1., 2.),
        (128, 0.03, 0.25, f64::NAN),
        (128, 0.03, 0.25, 0.),
    ] {
        assert!(VegaKtGammaRecoveryPolicy::new(vec![0.01], p.0, p.1, p.2, p.3).is_err());
    }
    let policy = VegaKtGammaRecoveryPolicy::new(vec![0.01, 0.001], 128, 1e-12, 0.25, 2.).unwrap();
    let r = compile(&payload("asian"), 1)
        .evaluate_vega_kt_gamma_recovery_audit(strikes(9), &policy)
        .unwrap();
    assert_eq!(r.status, VegaKtGammaRecoveryStatus::NoPilotCandidate);
    assert!(r.validation.is_none());
    assert!(r.selected_regularization.is_none());
    assert!(r.pilot_candidates.iter().all(|c| !c.passed));
}

#[test]
fn conservative_inverse_matches_primal_kkt_and_preserves_signed_local_vega_integral() {
    let base: Value = serde_json::from_str(include_str!(
        "../../../fixtures/local-vol/convolution-gamma-v0.1.json"
    ))
    .unwrap();
    let fixture: Value = serde_json::from_str(include_str!(
        "../../../fixtures/local-vol/conservative-convolution-gamma-v0.1.json"
    ))
    .unwrap();
    let array = |key: &str| {
        base[key]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect::<Vec<_>>()
    };
    for case in fixture["cases"].as_array().unwrap() {
        let op = VegaKtConvolutionPlan::new(
            base["spot"].as_f64().unwrap(),
            base["source_variance"].as_f64().unwrap(),
            base["step_variance"].as_f64().unwrap(),
            base["sigma_dt"].as_f64().unwrap(),
            &array("source_log_nodes"),
            array("gamma_strikes"),
            case["relative_regularization"].as_f64().unwrap(),
        )
        .unwrap();
        assert!(!op.preserves_local_vega_total());
        let op = op.conserve_local_vega_total().unwrap();
        assert!(op.preserves_local_vega_total());
        assert_eq!(
            op.recovery_matrix(),
            op.clone()
                .conserve_local_vega_total()
                .unwrap()
                .recovery_matrix()
        );
        for (actual, reference) in op
            .recovery_matrix()
            .iter()
            .zip(case["recovery_matrix"].as_array().unwrap())
        {
            for (&a, r) in actual.iter().zip(reference.as_array().unwrap()) {
                assert!((a - r.as_f64().unwrap()).abs() < 2e-7);
            }
        }
        let mut zero_sum = vec![0.; 17];
        zero_sum[5] = 2.;
        zero_sum[10] = -2.;
        for b in [array("local_vega"), zero_sum, vec![0.; 17]] {
            let q = op.recover(&b).unwrap();
            let reconstructed = op.reconstruct(&q).unwrap();
            assert!((reconstructed.iter().sum::<f64>() - b.iter().sum::<f64>()).abs() < 1e-10);
            let scaled = op
                .recover(&b.iter().map(|v| -2.5 * v).collect::<Vec<_>>())
                .unwrap();
            for (a, b) in q.iter().zip(scaled) {
                assert!((b + 2.5 * a).abs() < 1e-9);
            }
        }
        // Conservation does not pretend the full input vector was matched.
        let b = array("local_vega");
        let q = op.recover(&b).unwrap();
        assert!(
            norm(
                &op.reconstruct(&q)
                    .unwrap()
                    .iter()
                    .zip(&b)
                    .map(|(a, b)| a - b)
                    .collect::<Vec<_>>()
            ) > 1e-4
        );
        let gain = op
            .recovery_matrix()
            .iter()
            .flatten()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();
        assert!((gain - op.inverse_noise_gain_bound()).abs() < 1e-9 * gain);
    }
}
