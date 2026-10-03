//! Pairwise finite-grid checks, not continuous-time bias or convergence bounds.
//! The Gaussian coupling is test-only; production random layouts are unchanged.

use pricing::mc::{Philox4x32, RandomCoordinate, RandomDomain};
use pricing::rough_volatility::*;
use pricing_numerics::fractional_ou_correlation;
use serde_json::Value;

const FINE: usize = 128;
const LEVELS: [usize; 3] = [16, 32, 64];
const STANDARD_UNITS: usize = 32_768;
const ROUGH_UNITS: usize = 524_288;
const PRICE_BUDGET: f64 = 0.10;
const PAIRED_SE_CAP: f64 = 0.01;

fn fixture() -> Value {
    serde_json::from_str(include_str!(
        "../../../fixtures/rough-volatility/refinement.json"
    ))
    .unwrap()
}
fn numbers(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}
fn grid(n: usize) -> Vec<f64> {
    (0..=n).map(|j| j as f64 / n as f64).collect()
}
fn families(h: f64) -> Vec<RoughVolatilityModel> {
    let heston = RoughHeston::new(h, 0.04, 0.7, 0.055, 0.18, -0.65).unwrap();
    vec![
        heston.clone().into(),
        // Hold the chosen lift fixed when changing the time grid.
        LiftedHeston::from_rough(&heston, 20, 2.5).unwrap().into(),
        QuadraticRoughHeston::new(h, 0.15, 1.1, 0.5, 0.8, 0.25, 0.02)
            .unwrap()
            .into(),
        MixedRoughBergomi::new(
            h,
            -0.65,
            vec![0.35, 0.65],
            vec![0.45, 1.05],
            ForwardVarianceCurve::exponential(0.04, 0.08).unwrap(),
        )
        .unwrap()
        .into(),
        RoughSabr::new(
            h,
            0.75,
            -0.65,
            1.0,
            ForwardVarianceCurve::exponential(0.04, 0.06).unwrap(),
        )
        .unwrap()
        .into(),
        Rfsv::new(h, 1.3, 0.2, 0.2_f64.ln(), None).unwrap().into(),
    ]
}

#[derive(Clone)]
struct Cell {
    brownian: Vec<f64>,
    residual: Vec<f64>,
    extra: f64,
    rho: f64,
}
struct Coupling {
    fine: usize,
    coarse: usize,
    blocks: usize,
    fine_dimension: usize,
    coarse_dimension: usize,
    cell: Option<Cell>,
    rfsv_map: Option<Vec<Vec<f64>>>,
}
fn cholesky(matrix: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = matrix.len();
    let mut lower: Vec<Vec<f64>> = Vec::with_capacity(n);
    for (i, matrix_row) in matrix.iter().enumerate() {
        let mut row: Vec<f64> = Vec::with_capacity(n);
        for (j, &element) in matrix_row.iter().take(i + 1).enumerate() {
            let product = if i == j {
                row.iter().map(|x| x * x).sum::<f64>()
            } else {
                row.iter().zip(&lower[j]).map(|(a, b)| a * b).sum::<f64>()
            };
            let value = element - product;
            row.push(if i == j {
                assert!(value > 0.0, "test covariance not positive definite");
                value.sqrt()
            } else {
                value / lower[j][j]
            });
        }
        row.resize(n, 0.0);
        lower.push(row);
    }
    lower
}
fn rfsv_lower(model: &Rfsv, n: usize) -> Vec<Vec<f64>> {
    // Cache by integer lag, avoiding repeated numerical covariance integration.
    // This shares the already independently validated covariance routine, not
    // production matrix factorization or production path transforms.
    let correlation: Vec<f64> = (0..=n)
        .map(|j| {
            fractional_ou_correlation(model.hurst(), model.mean_reversion() * j as f64 / n as f64)
                .unwrap()
        })
        .collect();
    let matrix: Vec<Vec<f64>> = (0..=n)
        .map(|i| (0..=n).map(|j| correlation[i.abs_diff(j)]).collect())
        .collect();
    cholesky(&matrix)
}
impl Coupling {
    fn new(model: &RoughVolatilityModel, fine: usize, coarse: usize) -> Self {
        assert!(fine.is_multiple_of(coarse));
        let ratio = fine / coarse;
        let fp = RoughVolatilityPathPlan::compile(model.clone(), grid(fine)).unwrap();
        let cp = RoughVolatilityPathPlan::compile(model.clone(), grid(coarse)).unwrap();
        let blocks = match model {
            RoughVolatilityModel::Rfsv(_) | RoughVolatilityModel::QuadraticRoughHeston(_) => 1,
            _ => 2,
        };
        let power = match model {
            RoughVolatilityModel::RoughHeston(m) => Some((m.hurst(), m.correlation())),
            RoughVolatilityModel::QuadraticRoughHeston(m) => Some((m.hurst(), 1.0)),
            RoughVolatilityModel::MixedRoughBergomi(m) => Some((m.hurst(), m.correlation())),
            RoughVolatilityModel::RoughSabr(m) => Some((m.hurst(), m.correlation())),
            _ => None,
        };
        let cell = power.map(|(h, rho)| {
            let data = fixture();
            let row = data["cells"]
                .as_array()
                .unwrap()
                .iter()
                .find(|row| {
                    row["hurst"].as_f64().unwrap() == h
                        && row["ratio"].as_u64().unwrap() == ratio as u64
                })
                .unwrap();
            Cell {
                brownian: numbers(&row["brownian_coefficients"]),
                residual: numbers(&row["residual_coefficients"]),
                extra: row["extra_coefficient"].as_f64().unwrap(),
                rho,
            }
        });
        let rfsv_map = if let RoughVolatilityModel::Rfsv(m) = model {
            let fine_lower = rfsv_lower(m, fine);
            let coarse_lower = rfsv_lower(m, coarse);
            let mut map: Vec<Vec<f64>> = Vec::with_capacity(coarse + 1);
            for (i, lower_row) in coarse_lower.iter().enumerate() {
                let mut row = vec![0.0; fine + 1];
                for (k, value) in row.iter_mut().enumerate() {
                    let prior: f64 = map.iter().zip(lower_row).map(|(r, l)| r[k] * l).sum();
                    *value = (fine_lower[i * ratio][k] - prior) / lower_row[i];
                }
                map.push(row);
            }
            Some(map)
        } else {
            None
        };
        Self {
            fine,
            coarse,
            blocks,
            fine_dimension: fp.random_dimension() as usize,
            coarse_dimension: cp.random_dimension() as usize,
            cell,
            rfsv_map,
        }
    }
    fn dimension(&self) -> usize {
        self.fine_dimension + if self.cell.is_some() { self.coarse } else { 0 }
    }
    fn apply(&self, input: &[f64]) -> Vec<f64> {
        assert!(input.len() >= self.dimension());
        let m = self.fine / self.coarse;
        let mut out = vec![0.0; self.coarse_dimension];
        for block in 0..self.blocks {
            for (j, value) in out[block * self.coarse..(block + 1) * self.coarse]
                .iter_mut()
                .enumerate()
            {
                *value = input[block * self.fine + j * m..block * self.fine + (j + 1) * m]
                    .iter()
                    .sum::<f64>()
                    / (m as f64).sqrt();
            }
        }
        if let Some(cell) = &self.cell {
            for (j, value) in out[self.blocks * self.coarse..].iter_mut().enumerate() {
                *value = cell.extra * input[self.fine_dimension + j];
                for (k, (&a, &b)) in cell.brownian.iter().zip(&cell.residual).enumerate() {
                    let index = j * m + k;
                    let w = if self.blocks == 1 {
                        input[index]
                    } else {
                        cell.rho * input[index]
                            + (1.0 - cell.rho * cell.rho).sqrt() * input[self.fine + index]
                    };
                    *value += a * w + b * input[self.blocks * self.fine + index];
                }
            }
        }
        if let Some(map) = &self.rfsv_map {
            for (value, row) in out[self.coarse..].iter_mut().zip(map) {
                *value = row
                    .iter()
                    .zip(&input[self.fine..self.fine_dimension])
                    .map(|(a, z)| a * z)
                    .sum();
            }
        }
        out
    }
}

#[test]
fn coupled_normals_have_identity_marginals_for_every_family() {
    for h in [0.1, 0.3] {
        for model in families(h) {
            for coarse in [1, 2, 4] {
                let coupling = Coupling::new(&model, 8, coarse);
                let mut rows = vec![vec![0.0; coupling.dimension()]; coupling.coarse_dimension];
                for k in 0..coupling.dimension() {
                    let mut basis = vec![0.0; coupling.dimension()];
                    basis[k] = 1.0;
                    for (row, &value) in rows.iter_mut().zip(&coupling.apply(&basis)) {
                        row[k] = value;
                    }
                }
                for (i, row) in rows.iter().enumerate() {
                    for (j, other) in rows.iter().enumerate() {
                        let covariance: f64 = row.iter().zip(other).map(|(a, b)| a * b).sum();
                        let expected = f64::from(i == j);
                        assert!(
                            (covariance - expected).abs() < 2e-11,
                            "{} H={h} coarse={coarse} {i},{j}",
                            model.name()
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn coarse_newest_cells_have_the_retained_cross_grid_integral_covariance() {
    for cell in fixture()["cells"].as_array().unwrap() {
        let h = cell["hurst"].as_f64().unwrap();
        let m = cell["ratio"].as_u64().unwrap() as usize;
        let a = numbers(&cell["brownian_coefficients"]);
        let b = numbers(&cell["residual_coefficients"]);
        let cross = numbers(&cell["normalized_integral_cross_covariances"]);
        let sigma = (0.5 - h) / (0.5 + h);
        let fine_loading = (2.0 * h).sqrt() / (h + 0.5);
        let coarse_loading = fine_loading * (m as f64).powf(h - 0.5);
        let coarse_sigma = (m as f64).powf(h) * sigma;
        for ((&a, &b), &expected) in a.iter().zip(&b).zip(&cross) {
            let actual =
                (coarse_loading + coarse_sigma * a) * fine_loading + coarse_sigma * b * sigma;
            assert!((actual - expected).abs() < 2e-12);
        }
    }
}

#[test]
fn rfsv_coupling_preserves_shared_latent_nodes() {
    for h in [0.1, 0.3] {
        for initial in [None, Some(-1.4)] {
            let model: RoughVolatilityModel = Rfsv::new(h, 1.3, 0.2, 0.2_f64.ln(), initial)
                .unwrap()
                .into();
            let fine = RoughVolatilityPathPlan::compile(model.clone(), grid(16)).unwrap();
            for n in [2, 4, 8] {
                let coarse = RoughVolatilityPathPlan::compile(model.clone(), grid(n)).unwrap();
                let coupling = Coupling::new(&model, 16, n);
                let z: Vec<f64> = (0..coupling.dimension())
                    .map(|i| (i as f64 * 1.7).sin())
                    .collect();
                let fp = fine
                    .evolve_path(100.0, &z[..fine.random_dimension() as usize])
                    .unwrap();
                let cp = coarse.evolve_path(100.0, &coupling.apply(&z)).unwrap();
                for (i, &x) in cp.latent_states.iter().enumerate() {
                    assert!((x - fp.latent_states[i * 16 / n]).abs() < 2e-12);
                }
            }
        }
    }
}

#[test]
fn lift_kernel_refinement_matches_independent_positive_lag_reference() {
    for row in fixture()["lifts"].as_array().unwrap() {
        let h = row["hurst"].as_f64().unwrap();
        let n = row["factors"].as_u64().unwrap() as usize;
        let ratio = row["ratio"].as_f64().unwrap();
        let heston = RoughHeston::new(h, 0.04, 0.7, 0.055, 0.18, -0.65).unwrap();
        let lift = LiftedHeston::from_rough(&heston, n, ratio).unwrap();
        for (&t, &expected) in numbers(&row["lags"])
            .iter()
            .zip(&numbers(&row["kernel_values"]))
        {
            let actual: f64 = lift
                .weights()
                .iter()
                .zip(lift.rates())
                .map(|(&w, &x)| w * (-x * t).exp())
                .sum();
            assert!((actual - expected).abs() < 2e-11 * expected.abs().max(1.0));
        }
    }
}

fn stats(values: &[f64]) -> (f64, f64) {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let square = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>();
    (mean, (square / (n * (n - 1.0))).sqrt())
}
fn payoff(path: &RoughVolatilityPath) -> f64 {
    (path.forwards.last().unwrap() - 100.0).max(0.0)
}

#[test]
#[ignore = "multistep numerical acceptance; run explicitly in release"]
fn six_families_coupled_time_grid_price_refinement() {
    let mut failures = Vec::new();
    for h in [0.1, 0.3] {
        for model in families(h) {
            let fine = RoughVolatilityPathPlan::compile(model.clone(), grid(FINE)).unwrap();
            let coarse: Vec<_> = LEVELS
                .iter()
                .map(|&n| RoughVolatilityPathPlan::compile(model.clone(), grid(n)).unwrap())
                .collect();
            let couplings: Vec<_> = LEVELS
                .iter()
                .map(|&n| Coupling::new(&model, FINE, n))
                .collect();
            let dimension = couplings.iter().map(Coupling::dimension).max().unwrap();
            let units = if h == 0.1 {
                ROUGH_UNITS
            } else {
                STANDARD_UNITS
            };
            let chunk_size = units / 4;
            for seed in [91, 1973] {
                let rng = Philox4x32::from_seed(seed);
                let mut samples = vec![[0.0; 4]; units];
                std::thread::scope(|scope| {
                    for (chunk_id, chunk) in samples.chunks_mut(chunk_size).enumerate() {
                        let fine = &fine;
                        let coarse = &coarse;
                        let couplings = &couplings;
                        let rng = &rng;
                        scope.spawn(move || {
                            for (offset, sample) in chunk.iter_mut().enumerate() {
                                let unit = (chunk_id * chunk_size + offset) as u64;
                                let mut z: Vec<f64> = (0..dimension)
                                    .map(|d| {
                                        rng.standard_normal(RandomCoordinate::new(
                                            unit,
                                            d as u32,
                                            RandomDomain::Valuation,
                                        ))
                                    })
                                    .collect();
                                for _ in 0..2 {
                                    let fp = fine
                                        .evolve_path(100.0, &z[..fine.random_dimension() as usize])
                                        .unwrap();
                                    sample[0] += 0.5 * payoff(&fp);
                                    for (j, (plan, coupling)) in
                                        coarse.iter().zip(couplings).enumerate()
                                    {
                                        let cp =
                                            plan.evolve_path(100.0, &coupling.apply(&z)).unwrap();
                                        sample[j + 1] += 0.5 * payoff(&cp);
                                    }
                                    for value in &mut z {
                                        *value = -*value;
                                    }
                                }
                            }
                        });
                    }
                });
                let fine_values: Vec<_> = samples.iter().map(|v| v[0]).collect();
                let (fine_mean, fine_se) = stats(&fine_values);
                for (j, &n) in LEVELS.iter().enumerate() {
                    let coarse_values: Vec<_> = samples.iter().map(|v| v[j + 1]).collect();
                    let gaps: Vec<_> = samples.iter().map(|v| v[j + 1] - v[0]).collect();
                    let (coarse_mean, coarse_se) = stats(&coarse_values);
                    let (gap, se) = stats(&gaps);
                    let bound = gap.abs() + 4.0 * se;
                    let efficiency = se / (fine_se * fine_se + coarse_se * coarse_se).sqrt();
                    println!(
                        "REFINE {} H={h} seed={seed} grid={n}/{FINE} units={units} fine={fine_mean:.9} coarse={coarse_mean:.9} gap={gap:.9} paired_se={se:.9} bound={bound:.9} paired_unpaired={efficiency:.6}",
                        model.name()
                    );
                    assert!(se > 0.0 && se.is_finite());
                    if n == 64 && (bound > PRICE_BUDGET || se > PAIRED_SE_CAP) {
                        failures.push(format!(
                            "{} H={h} seed={seed}: bound={bound}, se={se}",
                            model.name()
                        ));
                    }
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "fixed 64/128 budgets failed: {failures:#?}"
    );
}

#[test]
fn multistep_public_prices_and_sampling_errors_match_primal_reconstruction() {
    use pricing::mc::ExecutionPolicy;
    use pricing::{JsonLimits, parse_request_json};
    use serde_json::json;
    for model in families(0.1) {
        for n in [64, 128] {
            let mut value: Value = serde_json::from_str(include_str!(
                "../../../fixtures/v2/pricing_request.golden.json"
            ))
            .unwrap();
            value["market"]["discount_curve"]["discount_factors"] = json!([1.0, 1.0]);
            value["market"]["dividend_curve"]["discount_factors"] = json!([1.0, 1.0]);
            value["engine"] = json!({"type":"pseudo_monte_carlo","independent_sampling_units":64,
                "master_seed":91,"variance_reduction":{"antithetic":true,"brownian_bridge":false}});
            let request =
                parse_request_json(&serde_json::to_vec(&value).unwrap(), JsonLimits::DEFAULT)
                    .unwrap();
            let plan = RoughVolatilityPricingPlan::compile(
                &request,
                model.clone(),
                1.0 / n as f64,
                ExecutionPolicy::new(2, Some(32)).unwrap(),
            )
            .unwrap();
            assert_eq!(plan.time_nodes(), grid(n));
            let path = plan.path_plan();
            let samples: Vec<f64> = (0..64)
                .map(|unit| {
                    let z = path.pseudo_shocks(91, unit, RandomDomain::Valuation);
                    let minus: Vec<f64> = z.iter().map(|x| -x).collect();
                    0.5 * (payoff(&path.evolve_path(100.0, &z).unwrap())
                        + payoff(&path.evolve_path(100.0, &minus).unwrap()))
                })
                .collect();
            let (mean, se) = stats(&samples);
            let actual = plan.evaluate().unwrap();
            assert_eq!(actual.independent_sampling_units, 64);
            assert_eq!(actual.evaluated_paths, 128);
            assert!(se > 0.0);
            assert!((actual.value - mean).abs() < 2e-12);
            assert!((actual.standard_error - se).abs() < 2e-12);
        }
    }
}
