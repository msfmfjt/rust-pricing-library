//! Conditional discounted cash for a Bergomi residual share.
//! Under the dividend-date forward measure E[Y(T)] = A(t,X) f + B(t) Y + C(t).
//! Only A depends on the volatility factor. Propagate its difference from the
//! constant-volatility A, retaining the exact existing B, C and Black limit.
use super::*;
use crate::models::{Bergomi1Factor, Bergomi2Factor};
use cash_volatility::OuStep;

const GRID_POINTS: usize = 257;
const PLANE_POINTS: usize = 65;
// Seven-point Gauss-Hermite quadrature for a standard normal.
const GAUSSIAN: [(f64, f64); 7] = [
    (-3.7504397177257425, 0.000548268855972218),
    (-2.3667594107345415, 0.0307571239675865),
    (-1.1544053947399682, 0.2401231786050127),
    (0., 0.4571428571428571),
    (1.1544053947399682, 0.2401231786050127),
    (2.3667594107345415, 0.0307571239675865),
    (3.7504397177257425, 0.000548268855972218),
];

#[derive(Clone, Copy, Debug)]
struct Axis {
    width: f64,
    scale: f64,
    last: usize,
}
impl Axis {
    fn new(width: f64, points: usize) -> Self {
        Self {
            width,
            scale: (points - 1) as f64 / (2. * width),
            last: points - 1,
        }
    }
    fn point(self, index: usize) -> f64 {
        index as f64 / self.scale - self.width
    }
    fn interval(self, x: f64) -> (usize, f64) {
        let pos = ((x + self.width) * self.scale).clamp(0., self.last as f64);
        let i = (pos as usize).min(self.last - 1);
        (i, pos - i as f64)
    }
}

#[derive(Clone, Copy, Debug)]
enum Grid {
    Line { axis: Axis, projection: [f64; 2] },
    Plane([Axis; 2]),
}

#[derive(Clone, Debug)]
pub(super) struct ReserveRow {
    grid: Grid,
    values: Box<[f64]>,
}

impl ReserveRow {
    fn line(&self, axis: Axis, x: f64) -> (f64, f64) {
        let (i, u) = axis.interval(x);
        let diff = self.values[i + 1] - self.values[i];
        (self.values[i] + u * diff, axis.scale * diff)
    }

    fn plane(&self, axes: [Axis; 2], x: [f64; 2]) -> (f64, [f64; 2]) {
        let (i, u) = axes[0].interval(x[0]);
        let (j, v) = axes[1].interval(x[1]);
        let stride = axes[1].last + 1;
        let a = i * stride + j;
        let dy0 = self.values[a + 1] - self.values[a];
        let dy1 = self.values[a + stride + 1] - self.values[a + stride];
        let lo = self.values[a] + v * dy0;
        let hi = self.values[a + stride] + v * dy1;
        (
            lo + u * (hi - lo),
            [
                (hi - lo) * axes[0].scale,
                (dy0 + u * (dy1 - dy0)) * axes[1].scale,
            ],
        )
    }

    // Repeated bilinear remapping diffuses a smooth conditional claim even as
    // the time step tends to zero. Use centered cubic Hermite interpolation
    // in the backward integral, with bilinear interpolation at the grid edges.
    fn plane_expectand(&self, axes: [Axis; 2], x: [f64; 2]) -> f64 {
        let (i, u) = axes[0].interval(x[0]);
        let (j, v) = axes[1].interval(x[1]);
        if i == 0 || j == 0 || i + 2 > axes[0].last || j + 2 > axes[1].last {
            return self.plane(axes, x).0;
        }
        let weights = |t: f64| {
            let p = t * (1. - t);
            [
                -0.5 * p * (1. - t),
                1. - t + p * (1. - 1.5 * t),
                t + p * (1.5 * t - 0.5),
                -0.5 * p * t,
            ]
        };
        let wx = weights(u);
        let wy = weights(v);
        let stride = axes[1].last + 1;
        let start = (i - 1) * stride + j - 1;
        wx.iter()
            .enumerate()
            .map(|(n, w)| {
                let row = &self.values[start + n * stride..][..4];
                w * (wy[0] * row[0] + wy[1] * row[1] + wy[2] * row[2] + wy[3] * row[3])
            })
            .sum()
    }

    pub(super) fn at(&self, x: [f64; 2]) -> Result<(f64, [f64; 2]), StochasticDividendError> {
        let outside = |x: f64, axis: Axis| !x.is_finite() || x.abs() > axis.width;
        match self.grid {
            Grid::Line { axis, projection } => {
                let y = projection[0] * x[0] + projection[1] * x[1];
                if !outside(y, axis) {
                    let (v, slope) = self.line(axis, y);
                    return Ok((v, projection.map(|w| w * slope)));
                }
            }
            Grid::Plane(axes) if !outside(x[0], axes[0]) && !outside(x[1], axes[1]) => {
                return Ok(self.plane(axes, x));
            }
            _ => {}
        }
        Err(invalid(
            "conditional dividend volatility factor outside reserve grid",
        ))
    }
}

/// A single effective OU is exact only with equal reversions or an inactive factor.
fn scalar_projection(f: Bergomi2Factor) -> Option<(f64, [f64; 2])> {
    let [a, b] = f.mean_reversions();
    let w = f.normalized_weights();
    if a == b || w[1] == 0. {
        Some((a, w))
    } else if w[0] == 0. {
        Some((b, w))
    } else {
        None
    }
}

pub(super) struct ReserveModel<'a> {
    pub factor: CashVolatility,
    pub snapshot: BergomiAsOf,
    pub initial_volatility: f64,
    pub dividend: BuehlerDividendModel,
    pub rates: &'a HullWhite1Factor,
    pub stock_rate: f64,
    pub dividend_rate: f64,
}

impl ReserveModel<'_> {
    pub(super) fn cells_per_row(&self) -> usize {
        match self.factor {
            CashVolatility::Two(f) if scalar_projection(f).is_none() => PLANE_POINTS * PLANE_POINTS,
            _ => GRID_POINTS,
        }
    }

    fn grid(&self, maturity: f64) -> Result<(CashVolatility, Grid), MonteCarloError> {
        let x = self.snapshot.x;
        let width = |v: f64, k: f64| v.abs() + 9. * b(2. * k, maturity).sqrt();
        let (f, projection) = match self.factor {
            CashVolatility::One(f) => (f, [1., 0.]),
            CashVolatility::Two(f) => {
                let Some((k, w)) = scalar_projection(f) else {
                    let k = f.mean_reversions();
                    let axes = std::array::from_fn(|i| Axis::new(width(x[i], k[i]), PLANE_POINTS));
                    return Ok((self.factor, Grid::Plane(axes)));
                };
                let r = f.spot_correlations();
                (
                    Bergomi1Factor::new(
                        k,
                        f.vol_of_vol(),
                        (w[0] * r[0] + w[1] * r[1]).clamp(-1., 1.),
                    )
                    .map_err(|_| invalid("projected Bergomi reserve factor"))?,
                    w,
                )
            }
        };
        let initial = projection[0] * x[0] + projection[1] * x[1];
        let axis = Axis::new(width(initial, f.mean_reversion()), GRID_POINTS);
        Ok((CashVolatility::One(f), Grid::Line { axis, projection }))
    }

    pub fn rows(
        &self,
        simulation: &[f64],
        maturity: f64,
    ) -> Result<Vec<Option<ReserveRow>>, MonteCarloError> {
        let count = simulation.partition_point(|&t| t < maturity);
        let mut rows = vec![None; count];
        let (factor, grid) = self.grid(maturity)?;
        let sigma0 = self.initial_volatility;
        let k = self.dividend.mean_reversion();
        let alpha = self.dividend.equity_linkage();
        let mut next = ReserveRow {
            grid,
            values: vec![0.; self.cells_per_row()].into_boxed_slice(),
        };
        let mut current = next.values.clone();
        let mut boundaries = simulation[..count].to_vec();
        boundaries.extend(
            self.rates
                .volatility_times()
                .iter()
                .copied()
                .filter(|&t| t > 0. && t < maturity),
        );
        boundaries.push(maturity);
        boundaries.sort_by(f64::total_cmp);
        boundaries.dedup();
        // Cover reserve cash beyond the last payoff fixing. Smaller simulation
        // steps are retained, and long gaps in the cash tail are subdivided.
        let mut baseline_a = 0.;
        let mut baseline_b = 1.;
        for pair in boundaries.windows(2).rev() {
            let steps = ((pair[1] - pair[0]) * 52.).ceil().max(1.);
            if steps > 1_000_000. {
                return Err(invalid("dividend reserve time grid limit").into());
            }
            let h = (pair[1] - pair[0]) / steps;
            let rate_sigma = self.rates.volatilities()[self
                .rates
                .volatility_times()
                .partition_point(|&t| t <= pair[0])
                - 1];
            let tr = factor
                .transition(h)
                .map_err(|_| invalid("Bergomi reserve OU transition"))?;
            let a_rate = self.rates.mean_reversion();
            let half_decay = (-0.5 * k * h).exp();
            let blend = -(-0.5 * k * h).exp_m1();
            for j in (0..steps as usize).rev() {
                let start = pair[0] + j as f64 * h;
                let tail = (maturity - start - h).max(0.);
                // Integral over this step of sigma_r B_r(u,T).
                let bond_loading = rate_sigma
                    * (crate::models::hull_white::integrated_b(a_rate, h)
                        + b(a_rate, tail) * b(a_rate, h));
                let stock_shift = -self.stock_rate * bond_loading;
                let dividend_mean =
                    (-self.dividend_rate * self.dividend.dividend_volatility() * bond_loading)
                        .exp();
                let baseline_mean = (sigma0 * stock_shift).exp();
                let carry = baseline_a + baseline_b * blend * alpha;
                let center = factor
                    .centering(self.snapshot.age + start)
                    .map_err(|_| invalid("Bergomi reserve variance centering"))?;
                // Tilt by the residual-share increment under the foreign cash-date
                // forward measure. Currency conversion is on the outer path.
                let update = |sigma: f64, expectation: f64| {
                    (sigma * stock_shift).exp() * expectation
                        + baseline_mean * ((sigma - sigma0) * stock_shift).exp_m1() * carry
                };
                match (tr, grid) {
                    (OuStep::One(t), Grid::Line { axis, .. }) => {
                        for (i, value) in current.iter_mut().enumerate() {
                            let x = axis.point(i);
                            let sigma = factor.volatility(sigma0, [x, 0.], center);
                            let mean = t.decay * x + t.covariance * (sigma + stock_shift / h);
                            let expectation = GAUSSIAN
                                .iter()
                                .map(|(z, w)| w * next.line(axis, mean + t.variance.sqrt() * z).0)
                                .sum();
                            *value = update(sigma, expectation);
                        }
                    }
                    (OuStep::Two(t), Grid::Plane(axes)) => {
                        // Three-point Gauss-Hermite per Gaussian dimension. The
                        // tilted OU covariance is unchanged; its mean shifts along
                        // the exact covariance with the stock Brownian increment.
                        let a = t.covariance[1][1].sqrt();
                        let b = t.covariance[1][2] / a;
                        let c = (t.covariance[2][2] - b * b).max(0.).sqrt();
                        let gaussian = [
                            (-3f64.sqrt(), 1. / 6.),
                            (0., 2. / 3.),
                            (3f64.sqrt(), 1. / 6.),
                        ];
                        let noise: [([f64; 2], f64); 9] = std::array::from_fn(|i| {
                            let (z1, w1) = gaussian[i / 3];
                            let (z2, w2) = gaussian[i % 3];
                            ([a * z1, b * z1 + c * z2], w1 * w2)
                        });
                        for (i, row) in current
                            .as_chunks_mut::<PLANE_POINTS>()
                            .0
                            .iter_mut()
                            .enumerate()
                        {
                            let x1 = axes[0].point(i);
                            for (j, value) in row.iter_mut().enumerate() {
                                let x2 = axes[1].point(j);
                                let sigma = factor.volatility(sigma0, [x1, x2], center);
                                let tilt = sigma + stock_shift / h;
                                let m1 = t.decay[0] * x1 + t.covariance[1][0] * tilt;
                                let m2 = t.decay[1] * x2 + t.covariance[2][0] * tilt;
                                let expectation = noise
                                    .iter()
                                    .map(|(d, w)| {
                                        w * next.plane_expectand(axes, [m1 + d[0], m2 + d[1]])
                                    })
                                    .sum();
                                *value = update(sigma, expectation);
                            }
                        }
                    }
                    _ => unreachable!("grid matches the reserve factor"),
                }
                if !current.iter().all(|v| v.is_finite()) {
                    return Err(invalid("nonfinite conditional dividend coefficient").into());
                }
                std::mem::swap(&mut next.values, &mut current);
                baseline_a =
                    baseline_mean * carry + baseline_b * blend * alpha * half_decay * dividend_mean;
                baseline_b *= half_decay * half_decay * dividend_mean;
            }
            if let Ok(i) = simulation[..count].binary_search_by(|t| t.total_cmp(&pair[0])) {
                rows[i] = Some(next.clone());
            }
        }
        Ok(rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mc::{Philox4x32, RandomCoordinate, RandomDomain};

    fn check_forward(factor: CashVolatility, x0: [f64; 2]) {
        // Independent forward Feynman–Kac integral for the residual-share
        // coefficient of the cash dividend, under the stock tilt.
        let (k, r, r12, w) = match factor {
            CashVolatility::One(f) => (
                [f.mean_reversion(), 0.],
                [f.correlation(), 0.],
                0.,
                [1., 0.],
            ),
            CashVolatility::Two(f) => (
                f.mean_reversions(),
                f.spot_correlations(),
                f.factor_correlation(),
                f.normalized_weights(),
            ),
        };
        let rate = HullWhite1Factor::new(0.32, vec![0.], vec![0.22]).unwrap();
        let model = ReserveModel {
            factor,
            snapshot: BergomiAsOf {
                age: 0.43,
                x: x0,
                ..Default::default()
            },
            initial_volatility: 0.45,
            dividend: BuehlerDividendModel::new(1.1, 0.8, 0.31, 0.45).unwrap(),
            rates: &rate,
            stock_rate: 0.72,
            dividend_rate: 0.72 * 0.45,
        };
        let t_end = 1.7;
        let row = model.rows(&[0., 0.28], t_end).unwrap().remove(0).unwrap();
        let (calculated, slope) = row.at(model.snapshot.x).unwrap();
        assert!(slope[0] < 0.);
        if factor.dimension() == 2 {
            assert!(slope[1] < 0.);
        }
        for i in 0..factor.dimension() {
            let mut up = x0;
            let mut down = x0;
            up[i] += 1e-6;
            down[i] -= 1e-6;
            let fd = (row.at(up).unwrap().0 - row.at(down).unwrap().0) / 2e-6;
            assert!((fd - slope[i]).abs() < 1e-8);
        }
        let n = 240;
        let dt = t_end / n as f64;
        let terms: Vec<_> = (0..n)
            .map(|j| {
                let t = j as f64 * dt;
                let remaining = t_end - t - 0.5 * dt;
                let age = model.snapshot.age + t;
                let variance = w[0] * w[0] * b(2. * k[0], age)
                    + w[1] * w[1] * b(2. * k[1], age)
                    + 2. * w[0] * w[1] * r12 * b(k[0] + k[1], age);
                let q = 0.22 * b(0.32, remaining);
                let by = (-1.1 * remaining
                    - model.dividend_rate
                        * 0.31
                        * 0.22
                        * crate::models::hull_white::integrated_b(0.32, remaining))
                .exp();
                (q, 0.8 * 1.1 * dt * by, factor.vol_of_vol() * variance)
            })
            .collect();
        let rng = Philox4x32::from_seed(9182);
        let mut sum = 0.;
        let mut sum2 = 0.;
        let paths = 4000;
        let sd1 = b(2. * k[0], dt).sqrt();
        let cross = r12 * b(k[0] + k[1], dt) / sd1;
        let sd2 = (b(2. * k[1], dt) - cross * cross).sqrt();
        let decay = k.map(|a| (-a * dt).exp());
        let shifts = [r[0] * b(k[0], dt), r[1] * b(k[1], dt)];
        for path in 0..paths {
            let mut x = [x0; 2];
            let mut weight = [1.; 2];
            let mut base = 1.;
            let mut integral = 0.;
            for (j, &(q, source, center)) in terms.iter().enumerate() {
                let normal = |i| {
                    rng.standard_normal(RandomCoordinate::new(
                        path,
                        2 * j as u32 + i,
                        RandomDomain::Valuation,
                    ))
                };
                let z1 = normal(0);
                let increments = [sd1 * z1, cross * z1 + sd2 * normal(1)];
                let ref_step = (-model.stock_rate * model.initial_volatility * q * dt).exp();
                for (i, sign) in [1., -1.].iter().enumerate() {
                    let vol = model.initial_volatility
                        * (factor.vol_of_vol() * (w[0] * x[i][0] + w[1] * x[i][1] - center)).exp();
                    let step = (-model.stock_rate * vol * q * dt).exp();
                    integral += 0.5 * source * (weight[i] * step.sqrt() - base * ref_step.sqrt());
                    weight[i] *= step;
                    for j in 0..2 {
                        x[i][j] = decay[j] * x[i][j]
                            + shifts[j] * (vol - model.stock_rate * q)
                            + sign * increments[j];
                    }
                }
                base *= ref_step;
            }
            sum += integral;
            sum2 += integral * integral;
        }
        let average = sum / paths as f64;
        let error = ((sum2 / paths as f64 - average * average) / (paths - 1) as f64).sqrt();
        assert!(
            (calculated - average).abs() < 5. * error + 0.00025,
            "backward {calculated}, forward {average} +/- {error}"
        );
    }

    #[test]
    fn conditional_cash_at_nonzero_volatility_state_against_forward_simulation() {
        let f = Bergomi1Factor::new(0.85, 0.82, -0.55).unwrap();
        check_forward(CashVolatility::One(f), [0.82, 0.]);
    }

    #[test]
    fn two_factor_cash_matches_forward_simulation_at_distinct_states() {
        let f = Bergomi2Factor::new([2.4, 0.45], 0.82, 0.4, [-0.55, -0.25], 0.25).unwrap();
        // Same current volatility, different future conditional distributions.
        for x in [[0.82, -0.31], [-0.18, 1.19]] {
            check_forward(CashVolatility::Two(f), x);
        }
    }
}
