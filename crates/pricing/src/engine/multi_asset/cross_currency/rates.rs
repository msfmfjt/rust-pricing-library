//! Exact joint Gaussian equity/FX Brownian increments, OU states and rate integrals.
use crate::engine::processes::hull_white::covariance_loading;
use crate::models::hull_white::{b, integrated_b, mixed_b};
use crate::models::{HullWhite1Factor, HybridCorrelation};
use crate::multi_asset::MultiAssetError as Error;

#[derive(Clone, Debug)]
pub struct CrossCurrencyHullWhite {
    pub(super) domestic: HullWhite1Factor,
    pub(super) foreign: HullWhite1Factor,
    pub(super) correlations: [[f64; 4]; 4],
}
impl CrossCurrencyHullWhite {
    /// Instantaneous Brownian correlation order: foreign equity, quoted FX,
    /// settlement-currency short rate, equity-currency short rate.
    pub fn new(
        domestic: HullWhite1Factor,
        foreign: HullWhite1Factor,
        correlations: [[f64; 4]; 4],
    ) -> Result<Self, Error> {
        for (i, row) in correlations.iter().enumerate() {
            for (j, &c) in row.iter().enumerate() {
                if !(-1.0..=1.0).contains(&c)
                    || (c - correlations[j][i]).abs() > 1e-12
                    || (i == j && c != 1.0)
                {
                    return Err(Error::Invalid(
                        "invalid equity/FX/domestic/foreign rate correlation matrix",
                    ));
                }
            }
        }
        covariance_loading(correlations).map_err(Error::numerical)?;
        Ok(Self {
            domestic,
            foreign,
            correlations,
        })
    }

    pub(super) fn independent(
        domestic: HullWhite1Factor,
        foreign: HullWhite1Factor,
        rho: f64,
    ) -> Result<Self, Error> {
        Self::new(
            domestic,
            foreign,
            [
                [1., rho, 0., 0.],
                [rho, 1., 0., 0.],
                [0., 0., 1., 0.],
                [0., 0., 0., 1.],
            ],
        )
    }

    /// Conditional on both rate Brownian increments, each OU endpoint and
    /// integral share one bridge residual: exp(-a*u) + a*B(a,u) = 1.
    /// Bridge residuals are independent of every stock and FX increment.
    pub(super) fn basket_steps(
        &self,
        times: &[f64],
        sigma_fx: f64,
    ) -> Result<Vec<BasketRateStep>, Error> {
        times
            .windows(2)
            .map(|w| {
                let h = w[1] - w[0];
                let [domestic, foreign] = [&self.domestic, &self.foreign];
                let reversion = [domestic.mean_reversion(), foreign.mean_reversion()];
                let vol = [domestic, foreign].map(|m| {
                    m.volatilities()[m.volatility_times().partition_point(|&t| t <= w[0]) - 1]
                });
                let loading = reversion.map(|a| b(a, h));
                let mean_integral = reversion.map(|a| integrated_b(a, h));
                let residual = std::array::from_fn(|i| {
                    std::array::from_fn(|j| {
                        vol[i]
                            * vol[j]
                            * self.correlations[i + 2][j + 2]
                            * centered_b_product(reversion[i], reversion[j], h)
                    })
                });
                let quanto = -sigma_fx * self.correlations[1][3] * vol[1];
                Ok(BasketRateStep {
                    residual: covariance_loading(residual).map_err(Error::numerical)?,
                    decay: reversion.map(|a| (-a * h).exp()),
                    loading,
                    reversion,
                    shift: [
                        domestic
                            .integrated_shift(w[0], w[1])
                            .map_err(Error::numerical)?,
                        foreign
                            .integrated_shift(w[0], w[1])
                            .map_err(Error::numerical)?,
                    ],
                    endpoint_weight: std::array::from_fn(|i| vol[i] * loading[i] / h.sqrt()),
                    integral_weight: std::array::from_fn(|i| vol[i] * mean_integral[i] / h.sqrt()),
                    foreign_quanto: [quanto * loading[1], quanto * mean_integral[1]],
                })
            })
            .collect()
    }

    pub(super) fn steps(&self, times: &[f64], sigma_fx: f64) -> Result<Vec<RateStep>, Error> {
        let zero = HybridCorrelation::new(0., 0., 0.).map_err(Error::numerical)?;
        times
            .windows(2)
            .map(|w| {
                let h = w[1] - w[0];
                let models = [&self.domestic, &self.foreign];
                let mut cov = [[0.; 6]; 6];
                cov[0][0] = h;
                cov[1][1] = h;
                cov[1][0] = h * self.correlations[0][1];
                let mut vols = [0.; 2];
                let mut decay = [0.; 2];
                let mut loading = [0.; 2];
                let mut shift = [0.; 2];
                for (i, m) in models.iter().enumerate() {
                    let col = 2 + 2 * i;
                    let a = m.mean_reversion();
                    let tr = m
                        .transition(w[0], w[1], 0., zero)
                        .map_err(Error::numerical)?;
                    vols[i] =
                        m.volatilities()[m.volatility_times().partition_point(|&t| t <= w[0]) - 1];
                    decay[i] = tr.rate_decay;
                    loading[i] = tr.integral_loading;
                    shift[i] = m.integrated_shift(w[0], w[1]).map_err(Error::numerical)?;
                    for (j, &rho) in self.correlations[i + 2][..2].iter().enumerate() {
                        cov[col][j] = rho * vols[i] * b(a, h);
                        cov[col + 1][j] = rho * vols[i] * integrated_b(a, h);
                    }
                    cov[col][col] = tr.covariance[2][2];
                    cov[col + 1][col] = tr.covariance[3][2];
                    cov[col + 1][col + 1] = tr.covariance[3][3];
                }
                let a = self.domestic.mean_reversion();
                let k = self.foreign.mean_reversion();
                let c = vols[0] * vols[1] * self.correlations[2][3];
                cov[4][2] = c * b(a + k, h);
                cov[5][2] = c * mixed_b(a, k, h);
                cov[4][3] = c * mixed_b(k, a, h);
                cov[5][3] = c * b_product(a, k, h);
                let foreign_quanto = [-sigma_fx * cov[4][1], -sigma_fx * cov[5][1]];
                let symmetric =
                    std::array::from_fn(|i| std::array::from_fn(|j| cov[i.max(j)][i.min(j)]));
                Ok(RateStep {
                    noise: covariance_loading(symmetric).map_err(Error::numerical)?,
                    decay,
                    loading,
                    shift,
                    foreign_quanto,
                })
            })
            .collect()
    }
}

// Integral of B(a,u)B(k,u) on [0,h]. For small mean reversions use the
// integrated power series to retain the Ho–Lee limit h^3/3.
fn b_product(a: f64, k: f64, h: f64) -> f64 {
    let (lo, hi) = (a.min(k), a.max(k));
    if hi * h >= 0.1 {
        return (integrated_b(lo, h) - mixed_b(hi, lo, h)) / hi;
    }
    let mut left = 1.;
    let mut sum = 0.;
    for i in 0..12 {
        let mut right = 1.;
        for j in 0..12 {
            sum += left * right / (i + j + 3) as f64;
            right *= -hi * h / (j + 2) as f64;
        }
        left *= -lo * h / (i + 2) as f64;
    }
    sum * h * h * h
}

pub(super) struct RateStep {
    noise: [[f64; 6]; 6],
    decay: [f64; 2],
    loading: [f64; 2],
    shift: [f64; 2],
    foreign_quanto: [f64; 2],
}
#[derive(Default)]
pub(super) struct RateState {
    pub(super) factors: [f64; 2],
    /// Integrated short rates in excess of their input forward curves.
    pub integrated: [f64; 2],
    pub fx_brownian: f64,
}

impl RateStep {
    pub(super) fn advance(&self, s: &mut RateState, z: [f64; 6]) {
        let shock = |i: usize| self.noise[i].iter().zip(z).map(|(a, b)| a * b).sum::<f64>();
        s.fx_brownian += shock(1);
        for i in 0..2 {
            let f = i == 1;
            s.integrated[i] += self.shift[i]
                + self.loading[i] * s.factors[i]
                + shock(3 + 2 * i)
                + if f { self.foreign_quanto[1] } else { 0. };
            s.factors[i] = self.decay[i] * s.factors[i]
                + shock(2 + 2 * i)
                + if f { self.foreign_quanto[0] } else { 0. };
        }
    }
}

// Integral of (B(a,u)-mean_a)*(B(k,u)-mean_k). For fast mean reversion
// remove the constant in B analytically before subtracting similar moments.
fn centered_b_product(a: f64, k: f64, h: f64) -> f64 {
    let (lo, hi) = (a.min(k), a.max(k));
    if lo * h >= 0.1 {
        (b(lo + hi, h) - b(lo, h) * b(hi, h) / h) / lo / hi
    } else if hi * h >= 0.1 {
        (b(hi, h) * integrated_b(lo, h) / h - mixed_b(hi, lo, h)) / hi
    } else {
        b_product(lo, hi, h) - integrated_b(lo, h) * integrated_b(hi, h) / h
    }
}

pub(super) struct BasketRateStep {
    residual: [[f64; 2]; 2],
    decay: [f64; 2],
    loading: [f64; 2],
    reversion: [f64; 2],
    shift: [f64; 2],
    endpoint_weight: [f64; 2],
    integral_weight: [f64; 2],
    foreign_quanto: [f64; 2],
}
impl BasketRateStep {
    pub(super) fn advance(&self, s: &mut RateState, rate_normals: [f64; 2], bridge: [f64; 2]) {
        for (i, z) in rate_normals.into_iter().enumerate() {
            let residual = self.residual[i][0] * bridge[0] + self.residual[i][1] * bridge[1];
            s.integrated[i] += self.shift[i]
                + self.loading[i] * s.factors[i]
                + self.integral_weight[i] * z
                + residual;
            s.factors[i] = self.decay[i] * s.factors[i] + self.endpoint_weight[i] * z
                - self.reversion[i] * residual;
        }
        s.factors[1] += self.foreign_quanto[0];
        s.integrated[1] += self.foreign_quanto[1];
    }
}

#[cfg(test)]
mod tests {
    use super::{b, centered_b_product};

    #[test]
    fn ou_bridge_integrals_match_centered_kernel_quadrature() {
        let h = 0.75;
        const CELLS: usize = 8192;
        for (a, k) in [
            (0., 0.),
            (0., 2.),
            (1e-6, 1e-3),
            (0.12, 0.17),
            (0.2, 2.),
            (3., 15.),
        ] {
            let cell = h / CELLS as f64;
            let kernels = (0..CELLS).map(|i| {
                let u = (i as f64 + 0.5) * cell;
                (b(a, u), b(k, u))
            });
            let (sum_a, sum_k) = kernels
                .clone()
                .fold((0., 0.), |(sa, sk), (x, y)| (sa + x, sk + y));
            let integral: f64 = kernels
                .map(|(x, y)| (x - sum_a / CELLS as f64) * (y - sum_k / CELLS as f64) * cell)
                .sum();
            assert!((centered_b_product(a, k, h) - integral).abs() < 2e-7 * integral);
        }
    }
}
