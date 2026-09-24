//! Continuous-time BS moment reference and common-noise price refinement.
//! The price panel compares finite grids, not an exact continuous-time oracle.
use pricing::mc::{
    DeterministicStatistics, LocalVolTimeGrid, Philox4x32, RandomCoordinate, RandomDomain,
};
use pricing::models::{Bergomi1Factor, Bergomi2Factor};
use pricing::stochastic_dividends::{BuehlerDividendModel, StochasticDividendPathPlan as Path};
use pricing::{JsonLimits, parse_request_json};
use serde_json::{Value, json};

#[test]
fn bs_split_second_moments_converge_to_continuous_time_ode() {
    let (k, alpha, nu, sigma, rho) = (0.7_f64, 0.6_f64, 0.35_f64, 0.2_f64, -0.25_f64);
    let c = 1.0 - alpha;
    let rhs = |m: [f64; 3]| {
        [
            sigma * sigma * m[0],
            k * alpha * m[0] + k * c + (sigma * nu * rho - k) * m[1],
            2.0 * k * alpha * m[1] + 2.0 * k * c + (nu * nu - 2.0 * k) * m[2],
        ]
    };
    // Independent fourth-order ODE integration of Ito second moments (f²,fY,Y²).
    let mut reference = [1.0; 3];
    let dt = 1.0 / 32768.0;
    for _ in 0..32768 {
        let a = rhs(reference);
        let b = rhs(std::array::from_fn(|i| reference[i] + 0.5 * dt * a[i]));
        let c = rhs(std::array::from_fn(|i| reference[i] + 0.5 * dt * b[i]));
        let d = rhs(std::array::from_fn(|i| reference[i] + dt * c[i]));
        for (i, value) in reference.iter_mut().enumerate() {
            *value += dt * (a[i] + 2.0 * b[i] + 2.0 * c[i] + d[i]) / 6.0;
        }
    }
    let mut errors = Vec::new();
    for n in [16, 32, 64, 128] {
        let h = 1.0 / n as f64;
        let a = (-0.5 * k * h).exp();
        let b = -(-0.5 * k * h).exp_m1();
        let mut m = [1.0; 3];
        for _ in 0..n {
            let hh = a * a * m[2]
                + b * b * (alpha * alpha * m[0] + 2.0 * alpha * c + c * c)
                + 2.0 * a * b * (alpha * m[1] + c);
            let fh = (a * m[1] + b * (alpha * m[0] + c)) * (sigma * nu * rho * h).exp();
            let ff = m[0] * (sigma * sigma * h).exp();
            m = [
                ff,
                a * fh + b * (alpha * ff + c),
                a * a * hh * (nu * nu * h).exp()
                    + b * b * (alpha * alpha * ff + 2.0 * alpha * c + c * c)
                    + 2.0 * a * b * (alpha * fh + c),
            ];
        }
        let error = (0..3)
            .map(|i| (m[i] - reference[i]).abs())
            .fold(0.0_f64, f64::max);
        println!("BS moment steps={n}, max_error={error}");
        errors.push(error);
    }
    assert!(errors[3] < 1e-5);
    for pair in errors.windows(2) {
        assert!(pair[0] / pair[1] > 3.5);
    }
}

fn b(k: f64, h: f64) -> f64 {
    if k == 0.0 { h } else { -(-k * h).exp_m1() / k }
}
#[allow(clippy::needless_range_loop)]
fn lower(family: usize, h: f64) -> Vec<Vec<f64>> {
    let dim = 2 + family;
    let rates = [0.0, 0.0, 0.8, 2.1];
    let rho = [
        [1.0, -0.25, -0.4, -0.2],
        [-0.25, 1.0, 0.15, -0.1],
        [-0.4, 0.15, 1.0, 0.3],
        [-0.2, -0.1, 0.3, 1.0],
    ];
    let mut l = vec![vec![0.0; dim]; dim];
    for i in 0..dim {
        for j in 0..=i {
            let c = rho[i][j] * b(rates[i] + rates[j], h)
                - (0..j).map(|q| l[i][q] * l[j][q]).sum::<f64>();
            l[i][j] = if i == j { c.sqrt() } else { c / l[j][j] };
        }
    }
    l
}
#[allow(clippy::needless_range_loop)]
fn coarse_normals(fine: &[Vec<f64>], l: &[Vec<f64>], n: usize) -> Vec<f64> {
    let block = fine.len() / n;
    let dim = l.len();
    let rates = [0.0, 0.0, 0.8, 2.1];
    let h = 1.0 / fine.len() as f64;
    let mut out = Vec::with_capacity(n * dim);
    for cells in fine.chunks_exact(block) {
        let mut increments = vec![0.0; dim];
        for cell in cells {
            for i in 0..dim {
                increments[i] = (-rates[i] * h).exp() * increments[i] + cell[i];
            }
        }
        let mut z = vec![0.0; dim];
        for i in 0..dim {
            z[i] = (increments[i] - (0..i).map(|j| l[i][j] * z[j]).sum::<f64>()) / l[i][i];
        }
        out.extend(z);
    }
    out
}
#[test]
#[ignore = "release-mode paired price refinement"]
fn coupled_bs_one_and_two_factor_price_refinement() {
    let mut v: Value = serde_json::from_str(include_str!(
        "../../../fixtures/v1/pricing_request.golden.json"
    ))
    .unwrap();
    v["market"]["discrete_dividends"] = json!([
        {"event_id":1,"ex_time":0.5,"quote":{"type":"fixed_cash","amount":5.0}},
        {"event_id":2,"ex_time":1.0,"quote":{"type":"fixed_cash","amount":3.0}},
        {"event_id":3,"ex_time":1.4,"quote":{"type":"fixed_cash","amount":12.0}}]);
    let r = parse_request_json(&serde_json::to_vec(&v).unwrap(), JsonLimits::DEFAULT).unwrap();
    let d = BuehlerDividendModel::new(0.7, 0.6, 0.35, -0.25).unwrap();
    let levels = [16_usize, 32, 64, 128, 256];
    for family in 0..3 {
        let dim = family + 2;
        let plans = levels
            .iter()
            .map(|&n| {
                let grid =
                    LocalVolTimeGrid::compile((1..=n).map(|i| i as f64 / n as f64).collect(), 1.0)
                        .unwrap();
                let m = r.market().equity().forward();
                match family {
                    0 => Path::compile(m, d, 0.2, &grid),
                    1 => Path::compile_bergomi(
                        m,
                        d,
                        0.2,
                        Bergomi1Factor::new(0.8, 0.3, -0.4).unwrap(),
                        0.15,
                        &grid,
                    ),
                    _ => Path::compile_bergomi_two_factor(
                        m,
                        d,
                        0.2,
                        Bergomi2Factor::new([0.8, 2.1], 0.3, 0.35, [-0.4, -0.2], 0.3).unwrap(),
                        [0.15, -0.1],
                        &grid,
                    ),
                }
                .unwrap()
            })
            .collect::<Vec<_>>();
        let lowers = levels
            .iter()
            .map(|n| lower(family, 1.0 / *n as f64))
            .collect::<Vec<_>>();
        for seed in [193_u64, 877] {
            let rng = Philox4x32::from_seed(seed);
            let mut differences = vec![Vec::new(); 4];
            for path in 0..4096_u64 {
                let mut fine = Vec::with_capacity(256);
                for step in 0..256 {
                    let z = (0..dim)
                        .map(|j| {
                            rng.standard_normal(RandomCoordinate::new(
                                path,
                                (step * dim + j) as u32,
                                RandomDomain::Valuation,
                            ))
                        })
                        .collect::<Vec<_>>();
                    fine.push(
                        (0..dim)
                            .map(|i| (0..=i).map(|j| lowers[4][i][j] * z[j]).sum())
                            .collect::<Vec<f64>>(),
                    );
                }
                let mut values = [0.0; 5];
                for (j, plan) in plans.iter().enumerate() {
                    let z = coarse_normals(&fine, &lowers[j], levels[j]);
                    for sign in [1.0, -1.0] {
                        let shocks = z.iter().map(|x| sign * x).collect::<Vec<_>>();
                        let states = plan.evolve_path(&shocks).unwrap();
                        let spot = plan
                            .nodes()
                            .last()
                            .unwrap()
                            .spots(*states.last().unwrap())
                            .unwrap()
                            .0;
                        values[j] += 0.5 * 0.95 * (spot - 100.0).max(0.0);
                    }
                }
                for j in 0..4 {
                    differences[j].push(values[j] - values[4]);
                }
            }
            for (j, values) in differences.iter().enumerate() {
                let stats = DeterministicStatistics::from_ordered_values_two_pass(values);
                let mean = stats.sum().total() / values.len() as f64;
                let se = (stats.moments().sample_variance().unwrap() / values.len() as f64).sqrt();
                println!(
                    "{}",
                    json!({"family":family,"seed":seed,"steps":levels[j],"reference_steps":256,"antithetic_units":4096,"paired_price_difference":mean,"paired_se":se})
                );
                // Chosen price-unit budget; sampling error does not widen it.
                if j == 3 {
                    assert!(
                        mean.abs() + 4.0 * se < 0.02,
                        "family={family}, seed={seed}, diff={mean}, SE={se}"
                    );
                }
            }
        }
    }
}
