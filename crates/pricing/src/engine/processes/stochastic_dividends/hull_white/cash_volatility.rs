//! Joint residual-stock and volatility innovations under the stock currency measure.
use crate::models::hull_white::b;
use crate::models::{
    Bergomi1Factor, Bergomi2Factor, Bergomi2FactorTransition, BergomiError, BergomiTransition,
};

#[derive(Clone, Copy, Debug)]
pub(in crate::engine) enum CashVolatility {
    One(Bergomi1Factor),
    Two(Bergomi2Factor),
}

#[derive(Clone, Copy)]
pub(in crate::engine) enum OuStep {
    One(BergomiTransition),
    Two(Bergomi2FactorTransition),
}

impl CashVolatility {
    pub fn dimension(self) -> usize {
        match self {
            Self::One(_) => 1,
            Self::Two(_) => 2,
        }
    }
    pub fn vol_of_vol(self) -> f64 {
        match self {
            Self::One(f) => f.vol_of_vol(),
            Self::Two(f) => f.vol_of_vol(),
        }
    }
    pub fn weights(self) -> [f64; 2] {
        match self {
            Self::One(_) => [1., 0.],
            Self::Two(f) => f.normalized_weights(),
        }
    }
    pub fn transition(self, dt: f64) -> Result<OuStep, BergomiError> {
        Ok(match self {
            Self::One(f) => OuStep::One(f.transition(dt)?),
            Self::Two(f) => OuStep::Two(f.transition(dt)?),
        })
    }
    pub fn centering(self, age: f64) -> Result<f64, BergomiError> {
        if age == 0. || self.vol_of_vol() == 0. {
            return Ok(0.);
        }
        let variance = match self {
            Self::One(f) => b(2. * f.mean_reversion(), age),
            Self::Two(f) => {
                let t = f.transition(age)?;
                let w = f.normalized_weights();
                (0..3)
                    .map(|j| (w[0] * t.lower[1][j] + w[1] * t.lower[2][j]).powi(2))
                    .sum()
            }
        };
        Ok(self.vol_of_vol() * variance)
    }
    pub fn volatility(self, scale: f64, x: [f64; 2], centering: f64) -> f64 {
        let [a, b] = self.weights();
        scale * (self.vol_of_vol() * (a * x[0] + b * x[1] - centering)).exp()
    }

    /// Project the reserve's instantaneous OU gradient onto the shared stock
    /// driver and the standard deviation orthogonal to it (independent of cash noise).
    pub fn diffusion(self, [a, b]: [f64; 2]) -> [f64; 2] {
        let (r, s, c) = match self {
            Self::One(f) => (f.correlation(), 0., 0.),
            Self::Two(f) => {
                let [r, s] = f.spot_correlations();
                (r, s, f.factor_correlation())
            }
        };
        let shared = a * r + b * s;
        let independent = a * a * (1. - r * r) + b * b * (1. - s * s) + 2. * a * b * (c - r * s);
        [shared, independent.max(0.).sqrt()]
    }
}
impl OuStep {
    pub fn evolve(self, x: [f64; 2], spot: f64, orthogonal: &[f64], stride: usize) -> [f64; 2] {
        match self {
            Self::One(t) => [t.evolve(x[0], spot, orthogonal[0]), 0.],
            Self::Two(t) => t.evolve(x, [spot, orthogonal[0], orthogonal[stride]]),
        }
    }
}
