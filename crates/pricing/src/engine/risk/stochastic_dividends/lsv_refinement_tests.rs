//! Pairwise common-Brownian refinement conditional on one calibrated surface.
//! Each coarse/fine pair has the exact joint Gaussian integral law. Pairs do
//! not define a joint law across all coarse levels; no cross-level SE is used.

use super::*;
use crate::models::RoughBergomi;
use crate::{JsonLimits, parse_request_json};
use serde_json::{Value, json};

const SD: f64 = -0.25;
const SV: f64 = -0.4;
const DV: f64 = 0.15;

fn vol_loading() -> [f64; 3] {
    let second = (DV - SD * SV) / (1.0 - SD * SD).sqrt();
    [SV, second, (1.0 - SV * SV - second * second).sqrt()]
}

// Covariance of a coarse newest-cell integral and a fine newest-cell integral,
// in units of fine_dt^(2H). offset is the distance between their right edges,
// measured in fine cells. Substitution x=t^(1/(H+1/2)) removes the singularity.
fn cross_kernel(h: f64, offset: usize) -> f64 {
    if offset == 0 {
        return 1.0;
    }
    let p = h + 0.5;
    let n = 8192;
    let integrand = |i: usize| (offset as f64 + (i as f64 / n as f64).powf(1.0 / p)).powf(h - 0.5);
    let mut sum = integrand(0) + integrand(n);
    for i in 1..n {
        sum += if i % 2 == 0 { 2.0 } else { 4.0 } * integrand(i);
    }
    2.0 * h * sum / (3.0 * n as f64 * p)
}

struct Coupling {
    block: usize,
    // Coarse independent residual normal in the basis of fine interleaved
    // normals plus one new normal. Coefficients are scale independent.
    residual: Vec<f64>,
}

impl Coupling {
    fn new(h: f64, block: usize) -> Self {
        assert!(h > 0.0 && h <= 0.5 && block.is_power_of_two());
        let mut residual = vec![0.0; 4 * block + 1];
        if h == 0.5 || block == 1 {
            // At H=1/2 the near-cell residual is unused by the path model.
            residual[if block == 1 { 3 } else { 4 * block }] = 1.0;
            return Self { block, residual };
        }
        let p = h + 0.5;
        let fine_average = (2.0 * h).sqrt() / p;
        let fine_residual = (0.5 - h) / p;
        let coarse_average = fine_average * (block as f64).powf(h - 0.5);
        let coarse_residual = fine_residual * (block as f64).powf(h);
        let loading = vol_loading();
        for cell in 0..block {
            let distance = (block - cell) as f64;
            let dw_cov = (2.0 * h).sqrt() / p * (distance.powf(p) - (distance - 1.0).powf(p));
            let near_cov = cross_kernel(h, block - cell - 1);
            for factor in 0..3 {
                residual[4 * cell + factor] =
                    (dw_cov - coarse_average) * loading[factor] / coarse_residual;
            }
            residual[4 * cell + 3] =
                (near_cov - fine_average * dw_cov) / (fine_residual * coarse_residual);
        }
        let explained = residual.iter().map(|v| v * v).sum::<f64>();
        // A negative conditional variance is an error, not a clipping policy.
        assert!(explained < 1.0, "H={h}, block={block}, norm2={explained}");
        residual[4 * block] = (1.0 - explained).sqrt();
        Self { block, residual }
    }

    fn coarsen(&self, fine: &[f64], extra: &[f64]) -> Vec<f64> {
        assert_eq!(fine.len(), 4 * self.block * extra.len());
        let mut coarse = Vec::with_capacity(4 * extra.len());
        for (cell, &independent) in fine.chunks_exact(4 * self.block).zip(extra) {
            for factor in 0..3 {
                coarse.push(
                    cell.iter().skip(factor).step_by(4).sum::<f64>() / (self.block as f64).sqrt(),
                );
            }
            coarse.push(
                cell.iter()
                    .zip(&self.residual)
                    .map(|(z, w)| z * w)
                    .sum::<f64>()
                    + independent * self.residual[4 * self.block],
            );
        }
        coarse
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "actual={actual:.16e}, expected={expected:.16e}, tolerance={tolerance}"
    );
}

#[test]
fn rough_cross_kernel_matches_independent_gaussian_quadrature() {
    // 512-point Gauss-Legendre after x=u^(1/H), independently of the
    // Simpson substitution above. Orders 256/512 differ by less than 2e-12.
    // Integrand: 2*u^(1/(2H))*(offset+u^(1/H))^(H-1/2), u in [0,1].
    for (h, reference) in [
        (
            0.01,
            [
                0.03459081320065839,
                0.021786620035437566,
                0.014777178079555687,
            ],
        ),
        (
            0.1,
            [0.2972848796035765, 0.20535642686670175, 0.14995909763209497],
        ),
        (
            0.3,
            [0.7004045779860347, 0.5861688788927809, 0.5020851560707129],
        ),
        (
            0.49,
            [0.9861019839611178, 0.9776162150380237, 0.9701636297406226],
        ),
    ] {
        for (offset, expected) in [1, 3, 7].into_iter().zip(reference) {
            close(cross_kernel(h, offset), expected, 3e-11);
        }
    }
}

#[test]
fn rough_grid_coupling_preserves_marginal_and_cross_covariances() {
    let loading = vol_loading();
    for h in [0.01, 0.1, 0.3, 0.49, 0.5] {
        for block in [1, 2, 4, 8] {
            let coupling = Coupling::new(h, block);
            let dim = 4 * block + 1;
            // Recover the implemented linear map with basis vectors, so this
            // also checks layout, normalization and the extra-normal index.
            let mut rows = vec![vec![0.0; dim]; 4];
            for coordinate in 0..dim {
                let mut z = vec![0.0; dim];
                z[coordinate] = 1.0;
                let out = coupling.coarsen(&z[..4 * block], &z[4 * block..]);
                for j in 0..4 {
                    rows[j][coordinate] = out[j];
                }
            }
            for i in 0..4 {
                for j in 0..4 {
                    close(dot(&rows[i], &rows[j]), f64::from(i == j), 3e-11);
                }
            }
            let p = h + 0.5;
            let average = (2.0 * h).sqrt() * (block as f64).powf(h - 0.5) / p;
            let residual = (block as f64).powf(h) * (0.5 - h) / p;
            let coarse_near = (0..dim)
                .map(|j| {
                    average
                        * (block as f64).sqrt()
                        * (0..3).map(|f| loading[f] * rows[f][j]).sum::<f64>()
                        + residual * rows[3][j]
                })
                .collect::<Vec<_>>();
            close(
                dot(&coarse_near, &coarse_near),
                (block as f64).powf(2.0 * h),
                3e-11,
            );
            for cell in 0..block {
                let distance = (block - cell) as f64;
                let dw_cov = (2.0 * h).sqrt() / p * (distance.powf(p) - (distance - 1.0).powf(p));
                let mut fine_near = vec![0.0; dim];
                for factor in 0..3 {
                    close(
                        coarse_near[4 * cell + factor],
                        dw_cov * loading[factor],
                        3e-11,
                    );
                    fine_near[4 * cell + factor] = (2.0 * h).sqrt() / p * loading[factor];
                }
                fine_near[4 * cell + 3] = (0.5 - h) / p;
                close(
                    dot(&coarse_near, &fine_near),
                    cross_kernel(h, block - cell - 1),
                    3e-11,
                );
            }
        }
    }
}

fn payload() -> Value {
    let mut v: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/v1/pricing_request.golden.json"
    )))
    .unwrap();
    v["model"] = json!({"type":"local_volatility", "local_variance_grid": {
        "time_nodes":[0.0,0.5,1.0], "log_forward_moneyness_nodes":[-0.5,0.0,0.5],
        "shape":[3,3], "values":[0.045,0.04,0.035,0.05,0.045,0.04,0.055,0.05,0.045],
        "floor":1e-8, "cap":4.0 }});
    v["market"]["discrete_dividends"] = json!([
        {"event_id":1,"ex_time":0.5,"quote":{"type":"fixed_cash","amount":5.0}},
        {"event_id":2,"ex_time":1.0,"quote":{"type":"fixed_cash","amount":3.0}},
        {"event_id":3,"ex_time":1.4,"quote":{"type":"fixed_cash","amount":12.0}}
    ]);
    v["engine"] = json!({"type":"pseudo_monte_carlo", "independent_sampling_units":32,
        "master_seed":193, "variance_reduction":{"antithetic":false,"brownian_bridge":false}});
    v
}

fn calibrated(h: f64) -> StochasticDividendPricingPlan {
    let request = parse_request_json(
        &serde_json::to_vec(&payload()).unwrap(),
        JsonLimits::DEFAULT,
    )
    .unwrap();
    StochasticDividendPricingPlan::compile_rough_bergomi_lsv(
        &request,
        BuehlerDividendModel::new(0.7, 0.6, 0.35, SD).unwrap(),
        RoughBergomi::new(h, 0.6, SV).unwrap(),
        DV,
        LsvParticleConfig::new(512, 42, 0.35, 5.0, false).unwrap(),
        1.0 / 16.0,
        ExecutionPolicy::new(1, Some(32)).unwrap(),
    )
    .unwrap()
}

fn path_estimates(path: &StochasticDividendPathPlan, z: &[f64]) -> [f64; 2] {
    let states = path.evolve_path(z).unwrap();
    let state = *states.last().unwrap();
    let stock = path.nodes().last().unwrap().spots(state).unwrap().0;
    // Normalized f/Y dynamics are invariant to physical Spot when leverage is
    // reanchored to funded residual equity. dS_T/dS0 = (0.98/0.95)*f_T.
    [
        0.95 * (stock - 100.0).max(0.0),
        if stock > 100.0 {
            0.98 * state.equity()
        } else {
            0.0
        },
    ]
}

fn antithetic_estimates(path: &StochasticDividendPathPlan, z: &[f64]) -> [f64; 2] {
    let a = path_estimates(path, z);
    let b = path_estimates(path, &z.iter().map(|v| -v).collect::<Vec<_>>());
    std::array::from_fn(|j| 0.5 * (a[j] + b[j]))
}

fn mean_se(values: &[f64]) -> (f64, f64) {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
        / ((values.len() - 1) * values.len()) as f64;
    (mean, variance.sqrt())
}

#[test]
fn rough_refinement_payoff_and_delta_match_public_estimator() {
    let base = calibrated(0.1);
    let rng = Philox4x32::from_seed(193);
    let mut values = [Vec::new(), Vec::new()];
    for p in 0..32 {
        let z = (0..base.path.random_dimension())
            .map(|d| rng.standard_normal(RandomCoordinate::new(p, d, RandomDomain::Valuation)))
            .collect::<Vec<_>>();
        for (out, x) in values.iter_mut().zip(path_estimates(&base.path, &z)) {
            out.push(x);
        }
    }
    let risk = base.evaluate_lsv_spot_risk().unwrap();
    for (values, (mean, se)) in values.iter().zip([
        (risk.price.value, risk.price.standard_error),
        (risk.delta, risk.delta_standard_error),
    ]) {
        let reference = mean_se(values);
        assert!(reference.0 > 0.0 && reference.1 > 0.0);
        close(reference.0, mean, 2e-11);
        close(reference.1, se, 2e-11);
    }
}

#[test]
#[ignore = "release-mode conditional rough-LSV price and Delta refinement"]
fn rough_lsv_fixed_surface_price_and_delta_refinement() {
    const LEVELS: [usize; 4] = [16, 32, 64, 128];
    const UNITS: u64 = 131072;
    let mut failures = Vec::new();
    for h in [0.1, 0.3] {
        let base = calibrated(h);
        let surface = base.path.lsv_surface().unwrap();
        let paths = LEVELS
            .iter()
            .map(|&n| {
                let grid =
                    LocalVolTimeGrid::compile((1..=n).map(|i| i as f64 / n as f64).collect(), 1.0)
                        .unwrap();
                let path = StochasticDividendPathPlan::compile(
                    &base.market,
                    BuehlerDividendModel::new(0.7, 0.6, 0.35, SD).unwrap(),
                    0.0,
                    &grid,
                )
                .unwrap()
                .with_rough_bergomi_lsv(RoughBergomi::new(h, 0.6, SV).unwrap(), DV, surface.clone())
                .unwrap();
                assert_eq!(path.times().len(), n + 1);
                assert_eq!(
                    path.lsv_surface().unwrap().squared_leverage(),
                    surface.squared_leverage()
                );
                path
            })
            .collect::<Vec<_>>();
        let fine = paths.last().unwrap();
        let fine_steps = *LEVELS.last().unwrap();
        let couplings = LEVELS[..3]
            .iter()
            .map(|n| Coupling::new(h, fine_steps / n))
            .collect::<Vec<_>>();
        for seed in [193, 877] {
            let rng = Philox4x32::from_seed(seed);
            let mut differences = vec![[Vec::new(), Vec::new()]; 3];
            let mut coarse_samples = vec![[Vec::new(), Vec::new()]; 3];
            let mut fine_samples = [Vec::new(), Vec::new()];
            for p in 0..UNITS {
                let normal =
                    |d| rng.standard_normal(RandomCoordinate::new(p, d, RandomDomain::Valuation));
                let z = (0..fine.random_dimension()).map(normal).collect::<Vec<_>>();
                let reference = antithetic_estimates(fine, &z);
                for j in 0..2 {
                    fine_samples[j].push(reference[j]);
                }
                for (level, coupling) in couplings.iter().enumerate() {
                    let extra = (0..LEVELS[level])
                        .map(|i| normal(fine.random_dimension() + i as u32))
                        .collect::<Vec<_>>();
                    let coarse = coupling.coarsen(&z, &extra);
                    let estimate = antithetic_estimates(&paths[level], &coarse);
                    for j in 0..2 {
                        coarse_samples[level][j].push(estimate[j]);
                        differences[level][j].push(estimate[j] - reference[j]);
                    }
                }
            }
            for (level, &steps) in LEVELS[..3].iter().enumerate() {
                for (j, label) in ["price", "delta"].iter().enumerate() {
                    let (gap, se) = mean_se(&differences[level][j]);
                    let coarse = mean_se(&coarse_samples[level][j]);
                    let reference = mean_se(&fine_samples[j]);
                    let unpaired_se = coarse.1.hypot(reference.1);
                    println!(
                        "{}",
                        json!({"hurst":h,"seed":seed,"steps":steps,"reference_steps":fine_steps,
                        "antithetic_units":UNITS,"quantity":label,"coarse":coarse.0,"reference":reference.0,
                        "paired_difference":gap,"paired_se":se,"unpaired_se":unpaired_se,
                        "calibration_steps":16,"calibration_particles":512,"calibration_seed":42,
                        "scope":"fixed_calibrated_surface"})
                    );
                    let coupling_reduces_noise = se < 0.75 * unpaired_se;
                    if !coupling_reduces_noise {
                        failures.push(format!("H={h}, seed={seed}, steps={steps}, {label}: coupling did not reduce noise"));
                    }
                    if level == 2 {
                        let budget = [0.05, 0.005][j];
                        let within_budget = gap.abs() + 4.0 * se < budget && se < [0.01, 0.001][j];
                        if !within_budget {
                            failures.push(format!(
                                "H={h}, seed={seed}, {label}: gap={gap}, SE={se}, budget={budget}"
                            ));
                        }
                    }
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
