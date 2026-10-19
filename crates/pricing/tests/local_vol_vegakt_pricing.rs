use pricing::core::{Date, DayCountConvention};
use pricing::market::{
    ImpliedVarianceSurface, LocalVarianceGrid, MarketError, ThetaRegion, TotalVarianceDerivatives,
};
use pricing::mc::{
    DeterministicExecutor, ExecutionPolicy, LocalVolLogEulerPlan, LocalVolTimeGrid, RandomDomain,
};
use pricing::risk::{
    ReportingIvBasis, VegaKtChainEstimate, VegaKtChainPlan, VegaKtChainSample,
    VegaKtInitialGammaPlan, VegaKtStepMoment, VegaKtStepRecoveryPlan, VegaKtStepSamplingUnit,
};
use pricing::{JsonLimits, PricingPlan, PricingRequest, parse_request_json};
use serde_json::{Value, json};
const DATES: [&str; 3] = ["2026-12-04", "2027-03-04", "2027-09-04"];
fn times() -> Vec<f64> {
    let start: Date = "2026-09-04".parse().unwrap();
    DATES
        .iter()
        .map(|d| DayCountConvention::Act365F.year_fraction(start, d.parse().unwrap()))
        .collect()
}
fn nodes() -> Vec<f64> {
    vec![-0.5, -0.25, 0., 0.25, 0.5]
}
fn strikes() -> Vec<f64> {
    vec![60., 80., 100., 120., 160.]
}
fn payload() -> Value {
    let mut p: Value = serde_json::from_str(include_str!(
        "../../../fixtures/v3/pricing_request.golden.json"
    ))
    .unwrap();
    let ts = times();
    p["market"]["discount_curve"]["discount_factors"] = json!([1., 1.]);
    p["market"]["dividend_curve"]["discount_factors"] = json!([1., 1.]);
    p["model"] = json!({"type":"local_volatility","local_variance_grid":{"time_nodes":([vec![0.],ts.clone()].concat()),"log_forward_moneyness_nodes":nodes(),"shape":[4,5],"values":vec![0.04;20],"floor":1e-8,"cap":4.},"reporting_iv_basis":{"maturity_nodes":ts,"log_forward_moneyness_nodes":nodes(),"shape":[3,5],"implied_volatilities":vec![0.2;15]}});
    p["risk"] = json!({"delta":false,"vega":true,"vega_kt":{"maturity_nodes":DATES,"log_forward_moneyness_nodes":nodes(),"relative_density_threshold":1e-10,"full_bucket_covariance":false},"smile_dynamics":{"type":"sticky_log_moneyness"}});
    p["engine"]["independent_sampling_units"] = json!(64);
    p
}
fn compile(p: &Value, w: u32) -> PricingPlan {
    let r: PricingRequest =
        parse_request_json(&serde_json::to_vec(p).unwrap(), JsonLimits::DEFAULT).unwrap();
    PricingPlan::compile(&r, ExecutionPolicy::new(w, Some(16)).unwrap()).unwrap()
}
struct Flat;
impl ImpliedVarianceSurface for Flat {
    fn total_variance_derivatives(
        &self,
        t: f64,
        _: f64,
    ) -> Result<TotalVarianceDerivatives, MarketError> {
        Ok(TotalVarianceDerivatives {
            total_variance: 0.04 * t,
            log_moneyness_derivative: 0.,
            log_moneyness_second_derivative: 0.,
            time_derivative: 0.04,
            theta: 0.04 * t,
            theta_derivative: 0.04,
            theta_region: ThetaRegion::Interpolated,
        })
    }
}
fn low_level(
    initial_strikes: Vec<f64>,
    recovery_nodes: Vec<f64>,
) -> (VegaKtChainPlan, LocalVolLogEulerPlan) {
    let ts = [vec![0.], times()].concat();
    let dt = ts.windows(2).map(|v| v[1] - v[0]).fold(0., f64::max);
    let p = LocalVolLogEulerPlan::with_constant_forward(
        LocalVolTimeGrid::compile(ts.clone(), dt).unwrap(),
        100.,
    )
    .unwrap();
    let n = recovery_nodes.len();
    let g = LocalVarianceGrid::new(ts, recovery_nodes, vec![0.04; 4 * n], 1e-8, 4.).unwrap();
    let i = VegaKtInitialGammaPlan::new(p.clone(), g.clone(), initial_strikes, 0.04).unwrap();
    let s = VegaKtStepRecoveryPlan::new(p.clone(), g, &Flat, 1e-10).unwrap();
    (
        VegaKtChainPlan::new(
            i,
            s,
            ReportingIvBasis::new(times(), nodes(), vec![0.2; 15]).unwrap(),
            2,
        )
        .unwrap()
        .with_integrated_density_recovery()
        .unwrap(),
        p,
    )
}
fn payoff(kind: usize, s: &[f64]) -> (f64, Vec<f64>) {
    let mut ds = vec![0.; s.len()];
    let level = match kind {
        0 => {
            ds[3] = 1.;
            s[3]
        }
        1 => {
            ds[1] = 0.25;
            ds[2] = 0.25;
            ds[3] = 0.25;
            (100. + s[1] + s[2] + s[3]) / 4.
        }
        _ => {
            let (i, &v) = s[1..]
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap();
            ds[i + 1] = 1.;
            v
        }
    };
    if level <= 100. {
        ds.fill(0.);
    }
    ((level - 100.).max(0.), ds)
}
fn all(r: &VegaKtChainEstimate) -> Vec<VegaKtStepMoment> {
    let mut m = r.position_iv_buckets.clone();
    m.extend(&r.offsetting_hedge_quantities);
    m.extend(&r.initial_bump_bucket_differences);
    m.extend(&r.quadrature_bucket_differences);
    m.extend([
        r.total_position_iv_vega,
        r.total_initial_bump_difference,
        r.total_quadrature_difference,
        r.initial_reconstruction_difference,
        r.initial_bump_vega_difference,
    ]);
    m.extend(&r.excluded_local_vegas);
    m.extend(r.initial_tail_vegas);
    m.extend(&r.propagated_tail_iv_vegas);
    m.extend(r.reporting_edge_iv_vegas.iter().flatten());
    m
}
#[test]
fn compiled_call_asian_and_lookback_match_manual_payoff_seeds_and_random_streams() {
    for product in 0..3 {
        let mut p = payload();
        if product == 1 {
            p["product"] = json!({"type":"arithmetic_asian","underlying_id":1,"currency_id":2,"strike":100.,"notional":1.,"side":{"type":"call"},"observations":[{"date":"2026-09-03","weight":0.25,"value":{"type":"known","fixing":100.}},{"date":DATES[0],"weight":0.25,"value":{"type":"unknown"}},{"date":DATES[1],"weight":0.25,"value":{"type":"unknown"}},{"date":DATES[2],"weight":0.25,"value":{"type":"unknown"}}],"payment_date":DATES[2]});
        }
        if product == 2 {
            p["product"] = json!({"type":"fixed_lookback","underlying_id":1,"currency_id":2,"strike":100.,"notional":1.,"side":{"type":"call"},"monitoring_dates":DATES,"payment_date":DATES[2]});
        }
        for antithetic in [false, true] {
            for bridge in [false, true] {
                p["engine"]["variance_reduction"] =
                    json!({"antithetic":antithetic,"brownian_bridge":bridge});
                let plan = compile(&p, 3);
                let before = plan.evaluate().unwrap();
                let actual = plan
                    .evaluate_vega_kt_chain_diagnostic(strikes(), 0.04, 2)
                    .unwrap();
                let (low, sim) = low_level(
                    actual.initial_strikes.to_vec(),
                    actual.recovery_log_moneyness_nodes.to_vec(),
                );
                let kind = if antithetic {
                    VegaKtStepSamplingUnit::AntitheticPair
                } else {
                    VegaKtStepSamplingUnit::IndependentPath
                };
                let expected = low
                    .estimate(
                        &DeterministicExecutor::new(ExecutionPolicy::new(1, Some(16)).unwrap())
                            .unwrap(),
                        64,
                        kind,
                        |i| {
                            let z = sim.path_shocks(7, i, RandomDomain::Valuation, bridge)?;
                            let sample =
                                |z: &[f64]| low.sample_from_shocks(z, |s| Ok(payoff(product, s)));
                            let a = sample(&z)?;
                            if antithetic {
                                Ok(VegaKtChainSample::antithetic_pair(
                                    a,
                                    sample(&z.iter().map(|g| -g).collect::<Vec<_>>())?,
                                )?)
                            } else {
                                Ok(a)
                            }
                        },
                    )
                    .unwrap();
                for (a, b) in all(&actual.estimate).iter().zip(all(&expected)) {
                    for (v, w) in [
                        (a.mean(), b.mean()),
                        (
                            a.sample_variance().unwrap().sqrt(),
                            b.sample_variance().unwrap().sqrt(),
                        ),
                        (a.standard_error().unwrap(), b.standard_error().unwrap()),
                    ] {
                        assert!(
                            (v - w).abs() < 1e-8 * (1. + w.abs()),
                            "product={product}, pair={antithetic}, bridge={bridge}: {v} vs {w}"
                        );
                    }
                }
                assert_eq!(actual.request_fingerprint, plan.request_fingerprint());
                assert_eq!(actual.plan_fingerprint, plan.plan_fingerprint());
                assert_eq!(actual.brownian_bridge, bridge);
                assert_eq!(actual.evaluated_paths, if antithetic { 128 } else { 64 });
                assert_eq!(before, plan.evaluate().unwrap());
            }
        }
    }
}
#[test]
fn unsupported_scope_is_rejected_without_changing_the_ordinary_result() {
    let p = payload();
    let mut cases = Vec::new();
    let mut q = p.clone();
    q["model"] = json!({"type":"black_scholes","volatility":0.2});
    q["risk"].as_object_mut().unwrap().remove("vega_kt");
    cases.push(q);
    let mut q = p.clone();
    q["engine"] = json!({"type":"randomized_quasi_monte_carlo","points_per_scramble":64,"scramble_count":2,"master_scramble_seed":7,"variance_reduction":{"antithetic":true,"brownian_bridge":false}});
    cases.push(q);
    let mut q = p.clone();
    q["risk"]["vega_kt"]["full_bucket_covariance"] = json!(true);
    cases.push(q);
    let mut q = p.clone();
    q["market"]["discount_curve"]["discount_factors"] = json!([1., 0.98]);
    q["market"]["dividend_curve"]["discount_factors"] = json!([1., 0.98]);
    cases.push(q);
    let mut q = p.clone();
    q["risk"].as_object_mut().unwrap().remove("vega_kt");
    cases.push(q);
    let mut q = p.clone();
    q["product"] = json!({"type":"digital","underlying_id":1,"currency_id":2,"expiry":DATES[2],"strike":100.,"payout":1.,"side":{"type":"call"},"payout_kind":{"type":"cash"},"payment_date":DATES[2]});
    q["risk"] =
        json!({"delta":false,"vega":false,"smile_dynamics":{"type":"sticky_log_moneyness"}});
    cases.push(q);
    let mut american: Value = serde_json::from_str(include_str!(
        "../../../fixtures/v3/pricing_request_american.golden.json"
    ))
    .unwrap();
    american["model"] = p["model"].clone();
    american["market"] = p["market"].clone();
    american["risk"] = p["risk"].clone();
    cases.push(american);
    for q in cases {
        let plan = compile(&q, 1);
        assert!(
            plan.evaluate_vega_kt_chain_diagnostic(strikes(), 0.04, 2)
                .is_err()
        );
        assert!(plan.evaluate().is_ok());
    }
}
#[test]
fn parameters_are_auditable_worker_replay_is_stable_and_pricing_minimum_is_preserved() {
    let mut p = payload();
    p["engine"]["independent_sampling_units"] = json!(2);
    let one = compile(&p, 1);
    let four = compile(&p, 4);
    let a = one
        .evaluate_vega_kt_chain_diagnostic(strikes(), 0.04, 2)
        .unwrap();
    let b = four
        .evaluate_vega_kt_chain_diagnostic(strikes(), 0.04, 2)
        .unwrap();
    assert_eq!(a.estimate, b.estimate);
    assert_ne!(a.plan_fingerprint, b.plan_fingerprint);
    assert!(a.estimate.total_position_iv_vega.standard_error().is_some());
    let c = one
        .evaluate_vega_kt_chain_diagnostic(strikes(), 0.02, 4)
        .unwrap();
    assert_eq!(a.request_fingerprint, c.request_fingerprint);
    assert_ne!(a.relative_bump, c.relative_bump);
    assert_ne!(
        a.estimate.coarse_subdivisions,
        c.estimate.coarse_subdivisions
    );
    for (ks, h, q) in [
        (vec![100.], 0.04, 2),
        (strikes(), f64::NAN, 2),
        (strikes(), 0.04, 0),
    ] {
        assert!(one.evaluate_vega_kt_chain_diagnostic(ks, h, q).is_err());
    }
}

#[test]
fn terminal_measure_from_native_call_and_put_is_exactly_localized_and_does_not_mutate_pricing() {
    for side in ["call", "put"] {
        let mut p = payload();
        p["product"]["side"] = json!({"type":side});
        let plan = compile(&p, 1);
        let before = plan.evaluate().unwrap();
        let d = plan.evaluate_vega_kt_terminal_diagnostic().unwrap();
        assert_eq!(d.payoff.gamma().atoms().len(), 1);
        assert_eq!(d.payoff.gamma().atoms()[0].strike(), 100.);
        assert_eq!(d.payoff.gamma().atoms()[0].mass(), 1.);
        assert_eq!(
            d.projection.hedge_quantities(),
            &[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1., 0., 0.]
        );
        let total = d.projection.position_risk().scalar_vega();
        assert!(
            (total - 100. * (-0.005_f64).exp() / (2. * std::f64::consts::PI).sqrt()).abs() < 1e-12
        );
        assert!((d.expected_local_parallel_vega - total).abs() < 1e-12);
        let first = &d.local_rows[0];
        assert_eq!(first.source_maturity, 0.);
        assert_eq!(first.source_total_variance, 0.);
        assert_eq!(
            first
                .expected_source_hat_vegas
                .iter()
                .filter(|&&x| x != 0.)
                .count(),
            1
        );
        assert!((first.expected_source_hat_vegas[2] - total * times()[0]).abs() < 1e-12);
        assert_eq!(d.request_fingerprint, plan.request_fingerprint());
        assert_eq!(before, plan.evaluate().unwrap());
    }
}

#[test]
fn terminal_measure_handles_term_structure_without_conflating_local_and_iv_vega() {
    let mut p = payload();
    let vars = [0.01, 0.04, 0.09, 0.16];
    p["model"]["local_variance_grid"]["values"] =
        json!(vars.iter().flat_map(|&v| [v; 5]).collect::<Vec<_>>());
    let ts = [vec![0.], times()].concat();
    let mut v = 0.;
    let mut ivs = vec![];
    let mut sigma_dt = 0.;
    for i in 0..3 {
        v += vars[i] * (ts[i + 1] - ts[i]);
        sigma_dt += vars[i].sqrt() * (ts[i + 1] - ts[i]);
        ivs.extend([(v / ts[i + 1]).sqrt(); 5]);
    }
    p["model"]["reporting_iv_basis"]["implied_volatilities"] = json!(ivs);
    let d = compile(&p, 1)
        .evaluate_vega_kt_terminal_diagnostic()
        .unwrap();
    assert!((d.total_variance - v).abs() < 1e-15);
    let ratio = sigma_dt / (v * ts[3]).sqrt();
    assert!(
        (d.expected_local_parallel_vega / d.projection.position_risk().scalar_vega() - ratio).abs()
            < 1e-12
    );
    assert!((ratio - 1.).abs() > 0.01);
}

#[test]
fn terminal_measure_rejects_nonflat_mismatched_iv_and_path_observations() {
    let mut cases = vec![];
    let mut p = payload();
    p["model"]["local_variance_grid"]["values"][1] = json!(0.05);
    cases.push(p);
    let mut p = payload();
    p["model"]["reporting_iv_basis"]["implied_volatilities"][1] = json!(0.21);
    cases.push(p);
    let mut p = payload();
    p["market"]["discount_curve"]["discount_factors"] = json!([1., 0.98]);
    p["market"]["dividend_curve"]["discount_factors"] = json!([1., 0.98]);
    cases.push(p);
    let mut p = payload();
    p["product"] = json!({"type":"fixed_lookback","underlying_id":1,"currency_id":2,"strike":100.,"notional":1.,"side":{"type":"call"},"monitoring_dates":DATES,"payment_date":DATES[2]});
    cases.push(p);
    for p in cases {
        assert!(
            compile(&p, 1)
                .evaluate_vega_kt_terminal_diagnostic()
                .is_err()
        );
    }
}

#[test]
fn sparse_reporting_dates_preserve_terminal_samples_and_joint_statistics() {
    let start: Date = "2026-09-04".parse().unwrap();
    let reporting_dates = ["2026-10-20", DATES[2]];
    let reporting_times: Vec<_> = reporting_dates
        .iter()
        .map(|d| DayCountConvention::Act365F.year_fraction(start, d.parse().unwrap()))
        .collect();
    for term_structure in [false, true] {
        let mut p = payload();
        p["product"]["strike"] = json!(105.);
        p["risk"]["vega_kt"]["full_bucket_covariance"] = json!(true);
        let vars: [f64; 4] = if term_structure {
            [0.01, 0.04, 0.09, 0.16]
        } else {
            [0.04; 4]
        };
        p["model"]["local_variance_grid"]["values"] =
            json!(vars.iter().flat_map(|&v| [v; 5]).collect::<Vec<_>>());
        let ts = [vec![0.], times()].concat();
        let mut total = 0.;
        let mut ivs = vec![];
        for i in 0..3 {
            total += vars[i] * (ts[i + 1] - ts[i]);
            ivs.extend([(total / ts[i + 1]).sqrt(); 5]);
        }
        p["model"]["reporting_iv_basis"]["implied_volatilities"] = json!(ivs);
        let dense_plan = compile(&p, 1);
        let dense = dense_plan.evaluate().unwrap();
        let dense_diagnostic = dense_plan.evaluate_vega_kt_terminal_diagnostic().unwrap();
        p["model"]["reporting_iv_basis"]["maturity_nodes"] = json!(reporting_times);
        p["model"]["reporting_iv_basis"]["shape"] = json!([2, 5]);
        p["model"]["reporting_iv_basis"]["implied_volatilities"] =
            json!([vec![vars[0].sqrt(); 5], ivs[10..].to_vec()].concat());
        p["risk"]["vega_kt"]["maturity_nodes"] = json!(reporting_dates);
        let sparse_plan = compile(&p, 4);
        let sparse = sparse_plan.evaluate().unwrap();
        let a = dense.pricing_result.risks.vega_kt.as_ref().unwrap();
        let b = sparse.pricing_result.risks.vega_kt.as_ref().unwrap();
        assert_eq!(
            b.policy_label(),
            "flat_variance_terminal_measure_pathwise_v1"
        );
        assert_eq!(dense.pricing_result.value, sparse.pricing_result.value);
        assert_eq!(
            dense.pricing_result.risks.vega,
            sparse.pricing_result.risks.vega
        );
        assert_eq!(
            a.projection().pre_projection(),
            b.projection().pre_projection()
        );
        assert_eq!(b.projection().signed_residual().get(), 0.);
        for j in 0..5 {
            assert_eq!(b.raw_buckets()[j].get(), 0.);
            assert_eq!(a.raw_buckets()[10 + j], b.raw_buckets()[5 + j]);
            assert_eq!(a.estimates()[10 + j], b.estimates()[5 + j]);
            for k in 0..5 {
                assert_eq!(
                    a.full_bucket_covariance().unwrap()[(10 + j) * 15 + 10 + k],
                    b.full_bucket_covariance().unwrap()[(5 + j) * 10 + 5 + k],
                );
            }
        }
        let d = sparse_plan.evaluate_vega_kt_terminal_diagnostic().unwrap();
        assert_eq!(d.total_variance, dense_diagnostic.total_variance);
        assert_eq!(d.local_rows, dense_diagnostic.local_rows);
        assert_eq!(
            d.projection.position_risk().scalar_vega(),
            dense_diagnostic.projection.position_risk().scalar_vega(),
        );
        // The new interior reporting date is inside the first Euler interval.
        // Even though its risk is zero, an inconsistent quote must not activate
        // this model. In particular, do not check only terminal IV consistency.
        for j in 0..5 {
            p["model"]["reporting_iv_basis"]["implied_volatilities"][j] = json!(0.3);
        }
        assert!(
            compile(&p, 1)
                .evaluate_vega_kt_terminal_diagnostic()
                .is_err()
        );
    }
}

#[test]
fn sparse_reporting_scope_still_requires_terminal_quote_and_flat_variance() {
    let mut p = payload();
    p["model"]["reporting_iv_basis"]["maturity_nodes"] = json!(&times()[..2]);
    p["model"]["reporting_iv_basis"]["shape"] = json!([2, 5]);
    p["model"]["reporting_iv_basis"]["implied_volatilities"] = json!(vec![0.2; 10]);
    p["risk"]["vega_kt"]["maturity_nodes"] = json!(&DATES[..2]);
    assert!(
        compile(&p, 1)
            .evaluate_vega_kt_terminal_diagnostic()
            .is_err()
    );
    p["model"]["reporting_iv_basis"]["maturity_nodes"] = json!([times()[0], times()[2]]);
    p["risk"]["vega_kt"]["maturity_nodes"] = json!([DATES[0], DATES[2]]);
    p["model"]["local_variance_grid"]["values"][1] = json!(0.05);
    assert!(
        compile(&p, 1)
            .evaluate_vega_kt_terminal_diagnostic()
            .is_err()
    );
}

fn assert_near(a: f64, b: f64, tolerance: f64) {
    assert!(
        (a - b).abs() <= tolerance,
        "{a:.16e} != {b:.16e}; tolerance={tolerance}"
    );
}

#[test]
fn ordinary_terminal_buckets_and_joint_covariance_match_same_path_price_differences() {
    let ts = [vec![0.], times()].concat();
    let simulation = LocalVolLogEulerPlan::with_constant_forward(
        LocalVolTimeGrid::compile(
            ts.clone(),
            ts.windows(2).map(|t| t[1] - t[0]).fold(0., f64::max),
        )
        .unwrap(),
        100.,
    )
    .unwrap();
    for term_structure in [false, true] {
        let vars = if term_structure {
            vec![0.01, 0.04, 0.09, 0.16]
        } else {
            vec![0.04; 4]
        };
        let mut integrated = 0.;
        let mut ivs = vec![];
        for i in 0..3 {
            integrated += vars[i] * (ts[i + 1] - ts[i]);
            ivs.extend([(integrated / ts[i + 1]).sqrt(); 5]);
        }
        let terminal_iv = integrated.sqrt(); // final maturity is one year
        for antithetic in [false, true] {
            for bridge in [false, true] {
                for side in ["call", "put"] {
                    let mut p = payload();
                    p["product"]["strike"] = json!(105.);
                    p["product"]["side"] = json!({"type":side});
                    p["risk"]["vega_kt"]["full_bucket_covariance"] = json!(true);
                    p["model"]["local_variance_grid"]["values"] =
                        json!(vars.iter().flat_map(|&v| [v; 5]).collect::<Vec<_>>());
                    p["model"]["reporting_iv_basis"]["implied_volatilities"] = json!(ivs);
                    p["engine"]["variance_reduction"] =
                        json!({"antithetic":antithetic,"brownian_bridge":bridge});
                    let plan = compile(&p, 1);
                    let result = plan.evaluate().unwrap();
                    let kt = result.pricing_result.risks.vega_kt.as_ref().unwrap();
                    assert_eq!(
                        kt.policy_label(),
                        "flat_variance_terminal_measure_pathwise_v1"
                    );
                    let h = 1e-6;
                    let terminal = |z: &[f64], iv: f64| {
                        let scale = iv / terminal_iv;
                        let mut s = 100.;
                        for i in 0..3 {
                            let dt = ts[i + 1] - ts[i];
                            s *= (-0.5 * vars[i] * scale * scale * dt
                                + vars[i].sqrt() * scale * dt.sqrt() * z[i])
                                .exp();
                        }
                        s
                    };
                    let sample = |z: &[f64]| {
                        let s = terminal(z, terminal_iv);
                        let price = if side == "call" {
                            (s - 105.).max(0.)
                        } else {
                            (105. - s).max(0.)
                        };
                        // Terminal replication uses signed call masses; the affine put
                        // remainder has zero model Vega, not a noisy spot-path derivative.
                        let up = (terminal(z, terminal_iv + h) - 105.).max(0.);
                        let down = (terminal(z, terminal_iv - h) - 105.).max(0.);
                        (price, (up - down) / (2. * h))
                    };
                    let mut prices = vec![];
                    let mut risks = vec![];
                    for i in 0..64 {
                        let z = simulation
                            .path_shocks(7, i, RandomDomain::Valuation, bridge)
                            .unwrap();
                        let (mut price, mut risk) = sample(&z);
                        if antithetic {
                            let (p, r) = sample(&z.iter().map(|z| -z).collect::<Vec<_>>());
                            price = (price + p) / 2.;
                            risk = (risk + r) / 2.;
                        }
                        prices.push(price);
                        risks.push(risk);
                    }
                    let pm = prices.iter().sum::<f64>() / 64.;
                    let rm = risks.iter().sum::<f64>() / 64.;
                    let variance = risks.iter().map(|r| (r - rm).powi(2)).sum::<f64>() / 63.;
                    let covariance = prices
                        .iter()
                        .zip(&risks)
                        .map(|(p, r)| (p - pm) * (r - rm))
                        .sum::<f64>()
                        / 63.;
                    let w = (105_f64 / 100.).ln() / 0.25;
                    let mut weights = [0.; 15];
                    weights[12] = 1. - w;
                    weights[13] = w;
                    assert_near(result.pricing_result.value.value().get(), pm, 2e-12);
                    assert_near(kt.projection().pre_projection().get(), rm, 3e-8);
                    assert_eq!(kt.projection().signed_residual().get(), 0.);
                    assert_eq!(
                        kt.residual_diagnostics().excluded_probability_mass().get(),
                        0.
                    );
                    for (j, &weight) in weights.iter().enumerate() {
                        let estimate = kt.estimates()[j];
                        assert_near(estimate.raw_mean().get(), weight * rm, 3e-8);
                        assert_near(
                            estimate.sample_variance().unwrap().get(),
                            weight * weight * variance,
                            1e-5,
                        );
                        assert_near(
                            estimate.price_covariance().unwrap().get(),
                            weight * covariance,
                            1e-6,
                        );
                        for (k, &other) in weights.iter().enumerate() {
                            assert_near(
                                kt.full_bucket_covariance().unwrap()[j * 15 + k]
                                    .unwrap()
                                    .get(),
                                weight * other * variance,
                                1e-5,
                            );
                        }
                    }
                    // Execution/replay identity changes with workers, numerical results do not.
                    let parallel = compile(&p, 4).evaluate().unwrap();
                    assert_eq!(result.pricing_result.risks, parallel.pricing_result.risks);
                }
            }
        }
    }
}

#[test]
fn ordinary_terminal_rqmc_variance_and_smile_replication_are_distinct() {
    let mut p = payload();
    p["engine"] = json!({"type":"randomized_quasi_monte_carlo","points_per_scramble":128,"scramble_count":4,"master_scramble_seed":7,"variance_reduction":{"antithetic":true,"brownian_bridge":true}});
    p["risk"]["vega_kt"]["full_bucket_covariance"] = json!(true);
    let result = compile(&p, 1).evaluate().unwrap();
    let kt = result.pricing_result.risks.vega_kt.as_ref().unwrap();
    let vega = result.pricing_result.risks.vega.as_ref().unwrap().raw();
    assert_eq!(
        kt.policy_label(),
        "flat_variance_terminal_measure_pathwise_v1"
    );
    assert_near(kt.raw_buckets()[12].get(), vega.value().get(), 1e-12);
    assert_near(
        kt.estimates()[12].sample_variance().unwrap().get(),
        vega.standard_error().get().powi(2) * 4.,
        1e-12,
    );
    assert!(kt.estimates()[12].sample_variance().unwrap().get() > 0.);
    p["model"]["local_variance_grid"]["values"][1] = json!(0.05);
    let smile = compile(&p, 1).evaluate().unwrap();
    let kt = smile.pricing_result.risks.vega_kt.as_ref().unwrap();
    assert_eq!(kt.policy_label(), "terminal_payoff_replication_iv_v1");
    assert!(
        kt.estimates()
            .iter()
            .all(|e| e.sample_variance().is_none() && e.price_covariance().is_none())
    );
    assert!(
        kt.full_bucket_covariance()
            .unwrap()
            .iter()
            .all(Option::is_none)
    );
    p["risk"].as_object_mut().unwrap().remove("vega_kt");
    let without = compile(&p, 1).evaluate().unwrap();
    assert_eq!(smile.pricing_result.value, without.pricing_result.value);
    assert_eq!(
        smile.pricing_result.risks.delta,
        without.pricing_result.risks.delta
    );
    assert_eq!(
        smile.pricing_result.risks.vega,
        without.pricing_result.risks.vega
    );
}
