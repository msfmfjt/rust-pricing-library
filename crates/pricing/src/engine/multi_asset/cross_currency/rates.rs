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
