//! Discounted-moment weak convergence and coupled finite-grid price/risk checks.
//! The finest option grid is a reference discretization, not an exact oracle.
use pricing::mc::{LocalVolTimeGrid, Philox4x32, RandomCoordinate, RandomDomain};
use pricing::models::{HullWhite1Factor, HybridCorrelation};
use pricing::stochastic_dividends::{
    BuehlerDividendModel as Model, BuehlerDividendState as Factors,
    StochasticDividendHullWhitePathPlan as Path, StochasticDividendHullWhiteState as State,
};
use pricing::{JsonLimits, PricingRequest, parse_request_json};
use serde_json::{Value, json};

type Matrix = [[f64; 4]; 4];
const LEVELS: [usize; 5] = [16, 32, 64, 128, 256];
const KNOTS: [f64; 4] = [0.0, 0.375, 0.75, 1.125];
const VOLS: [f64; 4] = [0.04, 0.065, 0.025, 0.09];
const RHO_FD: f64 = -0.25;
const RHO_FR: f64 = 0.25;
const RHO_DR: f64 = -0.2;
const SIGMA: f64 = 0.2;
const NU: f64 = 0.35;
const ALPHA: f64 = 0.6;

fn b(a: f64, t: f64) -> f64 {
    if a == 0.0 { t } else { -(-a * t).exp_m1() / a }
}
fn model(k: f64) -> Model {
    Model::new(k, ALPHA, NU, RHO_FD).unwrap()
}
fn rates(a: f64) -> HullWhite1Factor {
    HullWhite1Factor::new(a, KNOTS.to_vec(), VOLS.to_vec()).unwrap()
}
fn request(cash: &[(f64, f64)]) -> PricingRequest {
    let mut v: Value = serde_json::from_str(include_str!(
        "../../../fixtures/v1/pricing_request.golden.json"
    ))
    .unwrap();
    v["market"]["discrete_dividends"] = json!(
        cash.iter()
            .enumerate()
            .map(|(i, (t, amount))| json!({"event_id": i + 1, "ex_time": t,
            "quote": {"type": "fixed_cash", "amount": amount}}))
            .collect::<Vec<_>>()
    );
    parse_request_json(&serde_json::to_vec(&v).unwrap(), JsonLimits::DEFAULT).unwrap()
}
fn path(r: &PricingRequest, a: f64, k: f64, n: usize) -> Path {
    let grid =
        LocalVolTimeGrid::compile((1..=n).map(|i| i as f64 / n as f64).collect(), 1.0).unwrap();
    Path::compile_bs(
        r.market().equity().forward(),
        model(k),
        SIGMA,
        &rates(a),
        RHO_FR,
        RHO_DR,
        &grid,
    )
    .unwrap()
}
fn close(x: f64, y: f64, tolerance: f64) {
    assert!(
        (x - y).abs() < tolerance,
        "{x} != {y}; tolerance={tolerance}"
    );
}

// Independent elementary-kernel quadrature, including knots inside an interval.
// Coordinates are (W_f, W_D, x innovation, integral-x innovation).
fn covariance(a: f64, start: f64, end: f64) -> Matrix {
    let rho = [
        [1.0, RHO_FD, RHO_FR, RHO_FR],
        [RHO_FD, 1.0, RHO_DR, RHO_DR],
        [RHO_FR, RHO_DR, 1.0, 1.0],
        [RHO_FR, RHO_DR, 1.0, 1.0],
    ];
    let mut result = [[0.0; 4]; 4];
    for (index, &left) in KNOTS.iter().enumerate() {
        let lo = start.max(left);
        let hi = end.min(KNOTS.get(index + 1).copied().unwrap_or(end));
        if hi <= lo {
            continue;
        }
        let h = (hi - lo) / 64.0;
        for p in 0..=64 {
            let tau = end - (lo + p as f64 * h);
            let kernel = [
                1.0,
                1.0,
                VOLS[index] * (-a * tau).exp(),
                VOLS[index] * b(a, tau),
            ];
            let weight = h / 3.0
                * if p == 0 || p == 64 {
                    1.0
                } else if p % 2 == 0 {
                    2.0
                } else {
                    4.0
                };
            for (i, row) in result.iter_mut().enumerate() {
                for (j, value) in row.iter_mut().enumerate() {
                    *value += weight * rho[i][j] * kernel[i] * kernel[j];
                }
            }
        }
    }
    result
}
#[allow(clippy::needless_range_loop)]
fn lower(c: Matrix) -> Matrix {
    let mut l = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..=i {
            let x = c[i][j] - (0..j).map(|q| l[i][q] * l[j][q]).sum::<f64>();
            l[i][j] = if i == j { x.sqrt() } else { x / l[j][j] };
        }
        assert!(l[i][i].is_finite() && l[i][i] > 0.0);
    }
    l
}
fn loadings(a: f64, n: usize) -> Vec<Matrix> {
    (0..n)
        .map(|i| {
            lower(covariance(
                a,
                i as f64 / n as f64,
                (i + 1) as f64 / n as f64,
            ))
        })
        .collect()
}
fn fine_innovations(rng: &Philox4x32, unit: u64, lowers: &[Matrix]) -> Vec<[f64; 4]> {
    lowers
        .iter()
        .enumerate()
        .map(|(step, l)| {
            let z: [f64; 4] = std::array::from_fn(|j| {
                rng.standard_normal(RandomCoordinate::new(
                    unit,
                    (4 * step + j) as u32,
                    RandomDomain::Valuation,
                ))
            });
            std::array::from_fn(|i| (0..=i).map(|j| l[i][j] * z[j]).sum())
        })
        .collect()
}
fn coupled_normals(a: f64, fine: &[[f64; 4]], lowers: &[Matrix]) -> Vec<f64> {
    assert_eq!(fine.len() % lowers.len(), 0);
    let h = 1.0 / fine.len() as f64;
    let mut normals = Vec::with_capacity(4 * lowers.len());
    for (cells, l) in fine.chunks_exact(fine.len() / lowers.len()).zip(lowers) {
        let mut noise = [0.0; 4];
        for cell in cells {
            noise[0] += cell[0];
            noise[1] += cell[1];
            noise[3] += b(a, h) * noise[2] + cell[3];
            noise[2] = (-a * h).exp() * noise[2] + cell[2];
        }
        let mut z = [0.0; 4];
        for i in 0..4 {
            z[i] = (noise[i] - (0..i).map(|j| l[i][j] * z[j]).sum::<f64>()) / l[i][i];
        }
        normals.extend(z);
    }
    normals
}

// RK4 for the continuous-time T=1 Gaussian-tilted first moments. Select eta
// from the interval midpoint, so both sides of a knot use their own volatility.
fn continuous_moments(a: f64, steps: usize) -> [f64; 2] {
    let h = 1.0 / steps as f64;
    let mut m = [1.0; 2];
    for i in 0..steps {
        let t = i as f64 * h;
        let eta = VOLS[KNOTS.partition_point(|&s| s <= t + h * 0.5) - 1];
        let rhs = |s: f64, v: [f64; 2]| {
            let tilt = eta * b(a, 1.0 - s);
            [
                -SIGMA * RHO_FR * tilt * v[0],
                0.7 * ALPHA * v[0] + 0.7 * (1.0 - ALPHA) - (0.7 + NU * RHO_DR * tilt) * v[1],
            ]
        };
        let k1 = rhs(t, m);
        let k2 = rhs(t + h * 0.5, std::array::from_fn(|j| m[j] + h * 0.5 * k1[j]));
        let k3 = rhs(t + h * 0.5, std::array::from_fn(|j| m[j] + h * 0.5 * k2[j]));
        let k4 = rhs(t + h, std::array::from_fn(|j| m[j] + h * k3[j]));
        for j in 0..2 {
            m[j] += h / 6.0 * (k1[j] + 2.0 * k2[j] + 2.0 * k3[j] + k4[j]);
        }
    }
    m
}
fn split_moments(a: f64, steps: usize) -> [f64; 2] {
    // Positive half of eight-point standard-normal Gauss-Hermite quadrature.
    const GH: [(f64, f64); 4] = [
        (0.5390798113513752, 0.3730122576790775),
        (1.6365190424351082, 0.117239907661759),
        (2.802485861287542, 0.009635220120788263),
        (4.1445471861258945, 0.0001126145383753679),
    ];
    let rule: Vec<_> = GH.iter().flat_map(|&(z, w)| [(-z, w), (z, w)]).collect();
    let mut m = Factors::initial();
    for (i, l) in loadings(a, steps).iter().enumerate() {
        let remaining = 1.0 - (i + 1) as f64 / steps as f64;
        let shift: [f64; 2] = std::array::from_fn(|j| -l[3][j] - b(a, remaining) * l[2][j]);
        let mut next = [0.0; 2];
        for &(z0, w0) in &rule {
            for &(z1, w1) in &rule {
                let s = model(0.7)
                    .evolve(m, SIGMA, 1.0 / steps as f64, [z0 + shift[0], z1 + shift[1]])
                    .unwrap();
                next[0] += w0 * w1 * s.equity();
                next[1] += w0 * w1 * s.dividend();
            }
        }
        m = Factors::new(next[0], next[1]).unwrap();
    }
    [m.equity(), m.dividend()]
}

#[test]
fn discounted_dividend_moments_converge_to_independent_continuous_time_ode() {
    for a in [0.0, 0.4] {
        let reference = continuous_moments(a, 32768);
        for (x, y) in reference.into_iter().zip(continuous_moments(a, 16384)) {
            close(x, y, 2e-12);
        }
        let p = path(&request(&[(1.0, 3.0)]), a, 0.7, 16);
        close(p.initial_dividend_forwards()[0] / 3.0, reference[1], 2e-11);
        let mut errors = Vec::new();
        for n in LEVELS {
            let m = split_moments(a, n);
            close(m[0], reference[0], 2e-12);
            let error = (m[1] - reference[1]).abs();
            println!(
                "{}",
                json!({"check":"discounted_moment", "rate_reversion":a,
                "steps":n, "split_moments":m, "continuous_moments":reference, "dividend_error":error})
            );
            errors.push(error);
        }
        assert!(errors[4] < 5e-8);
        for pair in errors.windows(2) {
            assert!(pair[0] / pair[1] > 3.5, "{errors:?}");
        }
    }
}

#[test]
fn gaussian_coupling_preserves_shared_states_and_the_zero_reversion_limit() {
    let r = request(&[(0.5, 4.0), (1.0, 3.0), (1.5, 12.0)]);
    for a in [0.0, 0.4] {
        for (start, end) in [(0.0, 0.0625), (0.25, 1.0), (1.0, 1.5)] {
            let actual = rates(a)
                .transition(
                    start,
                    end,
                    0.0,
                    HybridCorrelation::new(RHO_FD, RHO_FR, RHO_DR).unwrap(),
                )
                .unwrap()
                .covariance;
            for (left, right) in actual
                .iter()
                .flatten()
                .zip(covariance(a, start, end).iter().flatten())
            {
                close(*left, *right, 2e-13);
            }
        }
        let lowers: Vec<_> = LEVELS.iter().map(|&n| loadings(a, n)).collect();
        let rng = Philox4x32::from_seed(421);
        for k in [0.0, 0.7] {
            let plans: Vec<_> = LEVELS.iter().map(|&n| path(&r, a, k, n)).collect();
            for unit in 0..8 {
                let fine = fine_innovations(&rng, unit, &lowers[4]);
                for sign in [-1.0, 1.0] {
                    let states: Vec<_> = plans
                        .iter()
                        .zip(&lowers)
                        .map(|(p, l)| {
                            let z: Vec<_> = coupled_normals(a, &fine, l)
                                .iter()
                                .map(|v| sign * v)
                                .collect();
                            p.evolve_path(&z).unwrap()
                        })
                        .collect();
                    for (level, &n) in LEVELS[..4].iter().enumerate() {
                        for (i, s) in states[level].iter().enumerate() {
                            let reference = states[4][i * 256 / n];
                            close(s.factors().equity(), reference.factors().equity(), 3e-12);
                            close(s.rate_factor(), reference.rate_factor(), 3e-12);
                            close(
                                s.integrated_rate_factor(),
                                reference.integrated_rate_factor(),
                                3e-12,
                            );
                            if k == 0.0 {
                                close(
                                    s.factors().dividend(),
                                    reference.factors().dividend(),
                                    3e-12,
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

struct Payoffs {
    growth: [f64; 2],
    discount: [f64; 2],
    duration: [f64; 2],
}
impl Payoffs {
    fn new(a: f64) -> Self {
        let hw = rates(a);
        Self {
            growth: [0.5, 1.0].map(|t| {
                (0.98_f64 / 0.95).powf(t) * (0.5 * hw.integrated_variance(t).unwrap()).exp()
            }),
            discount: [1.0, 1.25].map(|u| {
                0.95_f64.powf(u)
                    * hw.relative_discount(1.0, 0.0).unwrap()
                    * hw.relative_bond(1.0, u, 0.0).unwrap()
            }),
            duration: [0.0, b(a, 0.25)],
        }
    }
    // Each tuple is (price, Delta, Gamma(0.5), Gamma(1), Gamma(2)). Spot enters
    // only through the initially funded residual; every Q cash mean stays fixed.
    fn evaluate(&self, p: &Path, states: &[State]) -> [[f64; 5]; 2] {
        assert!(p.risky_spot() > 2.0);
        let n = states.len() - 1;
        let mut spot = [0.0; 2];
        let mut delta = [0.0; 2];
        for (j, i) in [n / 2, n].into_iter().enumerate() {
            spot[j] = p.spots(i, states[i]).unwrap().0;
            delta[j] = self.growth[j]
                * states[i].integrated_rate_factor().exp()
                * states[i].factors().equity();
        }
        let s = [spot[1], 0.4 * spot[0] + 0.6 * spot[1]];
        let d = [delta[1], 0.4 * delta[0] + 0.6 * delta[1]];
        std::array::from_fn(|product| {
            let df = self.discount[product]
                * (-states[n].integrated_rate_factor()
                    - self.duration[product] * states[n].rate_factor())
                .exp();
            let derivative = |shift: f64| {
                if s[product] + shift * d[product] > 100.0 {
                    df * d[product]
                } else {
                    0.0
                }
            };
            let mut sample = [
                df * (s[product] - 100.0).max(0.0),
                derivative(0.0),
                0.0,
                0.0,
                0.0,
            ];
            for (j, h) in [0.5, 1.0, 2.0].into_iter().enumerate() {
                sample[j + 2] = (derivative(h) - derivative(-h)) / (2.0 * h);
            }
            sample
        })
    }
}
fn moments(samples: &[f64]) -> (f64, f64) {
    let n = samples.len() as f64;
    let mean = samples.iter().sum::<f64>() / n;
    (
        mean,
        (samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n * (n - 1.0))).sqrt(),
    )
}

#[test]
#[ignore = "release-mode paired Hull-White price/Delta/Gamma refinement"]
fn coupled_hull_white_price_delta_and_gamma_refinement() {
    let r = request(&[(0.5, 4.0), (1.0, 3.0), (1.5, 12.0)]);
    for a in [0.0, 0.4] {
        let plans: Vec<_> = LEVELS.iter().map(|&n| path(&r, a, 0.7, n)).collect();
        let lowers: Vec<_> = LEVELS.iter().map(|&n| loadings(a, n)).collect();
        let payoffs = Payoffs::new(a);
        for seed in [421, 1607] {
            let rng = Philox4x32::from_seed(seed);
            let mut samples: [[Vec<[f64; 5]>; 2]; 5] =
                std::array::from_fn(|_| std::array::from_fn(|_| Vec::with_capacity(4096)));
            for unit in 0..4096 {
                let fine = fine_innovations(&rng, unit, &lowers[4]);
                for (j, (p, l)) in plans.iter().zip(&lowers).enumerate() {
                    let z = coupled_normals(a, &fine, l);
                    let mut pair = [[0.0; 5]; 2];
                    for sign in [-1.0, 1.0] {
                        let normals: Vec<_> = z.iter().map(|v| sign * v).collect();
                        let value = payoffs.evaluate(p, &p.evolve_path(&normals).unwrap());
                        for product in 0..2 {
                            for quantity in 0..5 {
                                pair[product][quantity] += 0.5 * value[product][quantity];
                            }
                        }
                    }
                    for (product, value) in pair.into_iter().enumerate() {
                        samples[j][product].push(value);
                    }
                }
            }
            for product in 0..2 {
                for (quantity, name) in ["price", "delta", "gamma_0.5", "gamma_1", "gamma_2"]
                    .iter()
                    .enumerate()
                {
                    for (j, &n) in LEVELS.iter().enumerate() {
                        let values: Vec<_> =
                            samples[j][product].iter().map(|s| s[quantity]).collect();
                        let differences: Vec<_> = samples[j][product]
                            .iter()
                            .zip(&samples[4][product])
                            .map(|(x, y)| x[quantity] - y[quantity])
                            .collect();
                        let (mean, se) = moments(&values);
                        let (difference, paired_se) = moments(&differences);
                        println!(
                            "{}",
                            json!({"check":"paired_refinement", "rate_reversion":a,
                            "seed":seed, "product":if product == 0 {"european"} else {"delayed_asian"},
                            "quantity":name, "steps":n, "reference_steps":256, "antithetic_units":4096,
                            "estimate":mean, "standard_error":se, "paired_difference":difference, "paired_se":paired_se})
                        );
                        if j == 3 {
                            let budget = if quantity == 0 { 0.02 } else { 0.002 };
                            assert!(
                                difference.abs() + 4.0 * paired_se < budget,
                                "a={a}, seed={seed}, product={product}, {name}: diff={difference}, SE={paired_se}, budget={budget}"
                            );
                        }
                    }
                }
            }
        }
    }
}
