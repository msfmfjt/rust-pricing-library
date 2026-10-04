//! Inputs for the multi-marginal Bass local-volatility construction.

use pricing_numerics::standard_normal_cdf;
use std::{error::Error, fmt};

mod spec;
mod surface;
pub use spec::BassLvSpec;
pub use surface::{BassMarginalProjection, BassSurfaceDiagnostics, BassSurfaceProjectionConfig};

/// Compact triangular terminal-map perturbation, in latent Brownian units.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BassMappingBump {
    interval: usize,
    left: f64,
    center: f64,
    right: f64,
}
impl BassMappingBump {
    pub fn new(interval: usize, left: f64, center: f64, right: f64) -> Result<Self, BassError> {
        if [left, center, right].iter().any(|x| !x.is_finite()) || left >= center || center >= right
        {
            return Err(BassError::InvalidInput(
                "mapping hat requires finite left < center < right".into(),
            ));
        }
        Ok(Self {
            interval,
            left,
            center,
            right,
        })
    }
    pub fn interval(&self) -> usize {
        self.interval
    }
    pub fn left(&self) -> f64 {
        self.left
    }
    pub fn center(&self) -> f64 {
        self.center
    }
    pub fn right(&self) -> f64 {
        self.right
    }
    pub(crate) fn value(&self, w: f64) -> f64 {
        if w <= self.left || w >= self.right {
            0.0
        } else if w <= self.center {
            (w - self.left) / (self.center - self.left)
        } else {
            (self.right - w) / (self.right - self.center)
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum BassError {
    InvalidInput(String),
    ConvexOrder {
        interval: usize,
        strike: f64,
        gap: f64,
    },
    Calibration {
        interval: usize,
        iterations: usize,
        residual: f64,
    },
    GridTooNarrow {
        interval: usize,
        tail_probability: f64,
    },
    Numerical(String),
}

impl fmt::Display for BassError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(s) | Self::Numerical(s) => f.write_str(s),
            Self::ConvexOrder {
                interval,
                strike,
                gap,
            } => write!(
                f,
                "Bass marginals violate convex order on interval {interval} at strike {strike}: call gap {gap}"
            ),
            Self::Calibration {
                interval,
                iterations,
                residual,
            } => write!(
                f,
                "Bass calibration did not converge on interval {interval} after {iterations} iterations (CDF residual {residual})"
            ),
            Self::GridTooNarrow {
                interval,
                tail_probability,
            } => write!(
                f,
                "Bass Brownian grid is too narrow on interval {interval} (boundary tail {tail_probability}); increase grid_width and grid_points"
            ),
        }
    }
}
impl Error for BassError {}

/// A continuous distribution with piecewise constant density on finite support.
/// CDF is linear between knots, zero below support and one above it. No atoms.
#[derive(Clone, Debug)]
pub struct BassMarginal {
    expiry: f64,
    spots: Vec<f64>,
    probabilities: Vec<f64>,
    mean: f64,
    variance: f64,
}

impl BassMarginal {
    pub fn new(expiry: f64, spots: Vec<f64>, probabilities: Vec<f64>) -> Result<Self, BassError> {
        if !expiry.is_finite()
            || expiry <= 0.0
            || spots.len() < 3
            || spots.len() != probabilities.len()
            || spots.iter().any(|x| !x.is_finite() || *x < 0.0)
            || probabilities
                .iter()
                .any(|p| !p.is_finite() || !(0.0..=1.0).contains(p))
            || spots.windows(2).any(|x| x[0] >= x[1])
            || probabilities.windows(2).any(|p| p[0] >= p[1])
            || probabilities[0] != 0.0
            || probabilities[probabilities.len() - 1] != 1.0
        {
            return Err(BassError::InvalidInput("marginal requires positive finite expiry, at least three increasing nonnegative spots, and strictly increasing CDF knots from 0 to 1".into()));
        }
        let mean: f64 = spots
            .windows(2)
            .zip(probabilities.windows(2))
            .map(|(s, p)| (p[1] - p[0]) * (s[0] + s[1]) * 0.5)
            .sum();
        let variance: f64 = spots
            .windows(2)
            .zip(probabilities.windows(2))
            .map(|(s, p)| {
                let a = s[0] - mean;
                let b = s[1] - mean;
                (p[1] - p[0]) * (a * a + a * b + b * b) / 3.0
            })
            .sum();
        if !mean.is_finite() || mean <= 0.0 || !variance.is_finite() || variance <= 0.0 {
            return Err(BassError::InvalidInput(
                "marginal must have finite positive mean and variance".into(),
            ));
        }
        Ok(Self {
            expiry,
            spots,
            probabilities,
            mean,
            variance,
        })
    }

    /// Finite-support approximation to a lognormal distribution, with exact
    /// tabulated mean equal to `spot`. `tail_std` is the lognormal tail cutoff.
    pub fn lognormal(
        expiry: f64,
        spot: f64,
        volatility: f64,
        nodes: usize,
        tail_std: f64,
    ) -> Result<Self, BassError> {
        if !spot.is_finite()
            || spot <= 0.0
            || !volatility.is_finite()
            || volatility <= 0.0
            || !expiry.is_finite()
            || expiry <= 0.0
            || !(33..=100_001).contains(&nodes)
            || !tail_std.is_finite()
            || !(3.0..=7.5).contains(&tail_std)
        {
            return Err(BassError::InvalidInput(
                "invalid lognormal marginal parameters (nodes 33..100001, tail_std 3..7.5)".into(),
            ));
        }
        let sd = volatility * expiry.sqrt();
        let low = standard_normal_cdf(-tail_std);
        let mass = 1.0 - 2.0 * low;
        let mut s = Vec::with_capacity(nodes);
        let mut p = Vec::with_capacity(nodes);
        for i in 0..nodes {
            let z = -tail_std + 2.0 * tail_std * i as f64 / (nodes - 1) as f64;
            s.push(spot * (sd * z - 0.5 * sd * sd).exp());
            p.push(((standard_normal_cdf(z) - low) / mass).clamp(0.0, 1.0));
        }
        p[0] = 0.0;
        p[nodes - 1] = 1.0;
        let raw = Self::new(expiry, s, p)?;
        let scale = spot / raw.mean;
        Self::new(
            expiry,
            raw.spots.iter().map(|s| s * scale).collect(),
            raw.probabilities,
        )
    }

    pub fn expiry(&self) -> f64 {
        self.expiry
    }
    pub fn mean(&self) -> f64 {
        self.mean
    }
    pub fn variance(&self) -> f64 {
        self.variance
    }
    pub fn spots(&self) -> &[f64] {
        &self.spots
    }
    pub fn probabilities(&self) -> &[f64] {
        &self.probabilities
    }
    pub fn cdf(&self, spot: f64) -> f64 {
        interpolate(&self.spots, &self.probabilities, spot)
    }
    pub fn quantile(&self, probability: f64) -> Result<f64, BassError> {
        if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
            return Err(BassError::InvalidInput(
                "quantile probability must be in [0,1]".into(),
            ));
        }
        Ok(self.quantile_unchecked(probability))
    }
    pub(crate) fn quantile_unchecked(&self, p: f64) -> f64 {
        interpolate(&self.probabilities, &self.spots, p)
    }
    /// Undiscounted call expectation under the supplied marginal.
    pub fn call_price(&self, strike: f64) -> Result<f64, BassError> {
        if !strike.is_finite() {
            return Err(BassError::InvalidInput("strike must be finite".into()));
        }
        Ok(self.call_unchecked(strike))
    }
    pub(crate) fn call_unchecked(&self, k: f64) -> f64 {
        self.spots
            .windows(2)
            .zip(self.probabilities.windows(2))
            .map(|(s, p)| {
                let lo = s[0].max(k);
                let hi = s[1];
                if lo >= hi {
                    0.0
                } else {
                    (p[1] - p[0]) / (s[1] - s[0]) * (hi - lo) * ((hi + lo) * 0.5 - k)
                }
            })
            .sum()
    }
}

pub(crate) fn interpolate(x: &[f64], y: &[f64], value: f64) -> f64 {
    if value.is_nan() {
        return f64::NAN;
    }
    if value <= x[0] {
        return y[0];
    }
    let n = x.len();
    if value >= x[n - 1] {
        return y[n - 1];
    }
    let i = x.partition_point(|v| *v <= value) - 1;
    let a = (value - x[i]) / (x[i + 1] - x[i]);
    y[i] * (1.0 - a) + y[i + 1] * a
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BassLvConfig {
    pub grid_points: usize,
    /// Half width in units of an estimated latent standard deviation.
    pub grid_width: f64,
    pub max_iterations: usize,
    pub cdf_tolerance: f64,
    pub tail_tolerance: f64,
}
impl Default for BassLvConfig {
    fn default() -> Self {
        Self {
            grid_points: 801,
            grid_width: 8.0,
            max_iterations: 2000,
            cdf_tolerance: 1e-6,
            tail_tolerance: 1e-7,
        }
    }
}
impl BassLvConfig {
    pub(crate) fn validate(&self) -> Result<(), BassError> {
        if !(101..=8001).contains(&self.grid_points)
            || self.grid_points.is_multiple_of(2)
            || !self.grid_width.is_finite()
            || !(4.0..=20.0).contains(&self.grid_width)
            || self.max_iterations == 0
            || self.max_iterations > 100_000
            || !self.cdf_tolerance.is_finite()
            || !(1e-12..=1e-3).contains(&self.cdf_tolerance)
            || !self.tail_tolerance.is_finite()
            || !(1e-12..=1e-3).contains(&self.tail_tolerance)
        {
            return Err(BassError::InvalidInput("invalid Bass configuration: odd grid_points 101..8001, grid_width 4..20, positive iteration limit and tolerances 1e-12..1e-3".into()));
        }
        Ok(())
    }
}
