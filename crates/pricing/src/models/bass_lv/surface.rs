use super::{BassError, BassMarginal};
use crate::market::ImpliedVarianceSurface;
use pricing_numerics::{standard_normal_cdf as cdf, standard_normal_pdf as pdf};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BassSurfaceProjectionConfig {
    pub tail_probability_tolerance: f64,
    pub relative_mean_tolerance: f64,
}
impl Default for BassSurfaceProjectionConfig {
    fn default() -> Self {
        Self {
            tail_probability_tolerance: 1e-7,
            relative_mean_tolerance: 1e-4,
        }
    }
}

#[derive(Clone, Debug)]
pub struct BassSurfaceDiagnostics {
    pub lower_tail_probability: f64,
    pub upper_tail_probability: f64,
    pub retained_probability: f64,
    pub unscaled_mean: f64,
    pub mean_scale: f64,
    /// Maximum absolute difference against source calls at all supplied nodes.
    pub max_call_price_error: f64,
    pub retained_nodes: usize,
}

#[derive(Clone, Debug)]
pub struct BassMarginalProjection {
    pub marginal: BassMarginal,
    pub diagnostics: BassSurfaceDiagnostics,
}

impl BassMarginal {
    /// Build a marginal in a martingale coordinate with constant forward `spot`.
    /// The surface must be quoted in log(K/spot) for this coordinate. Raw spot
    /// smiles with carry or affine dividends must first be transformed.
    pub fn from_surface<S: ImpliedVarianceSurface + ?Sized>(
        expiry: f64,
        spot: f64,
        surface: &S,
        log_moneyness_nodes: &[f64],
        config: BassSurfaceProjectionConfig,
    ) -> Result<BassMarginalProjection, BassError> {
        if !expiry.is_finite()
            || expiry <= 0.0
            || !spot.is_finite()
            || spot <= 0.0
            || !(33..=100_001).contains(&log_moneyness_nodes.len())
            || log_moneyness_nodes.iter().any(|k| !k.is_finite())
            || log_moneyness_nodes.windows(2).any(|p| p[0] >= p[1])
            || log_moneyness_nodes[0] >= 0.0
            || *log_moneyness_nodes.last().unwrap() <= 0.0
            || !config.tail_probability_tolerance.is_finite()
            || !(1e-12..=1e-2).contains(&config.tail_probability_tolerance)
            || !config.relative_mean_tolerance.is_finite()
            || !(1e-12..=1e-2).contains(&config.relative_mean_tolerance)
        {
            return Err(BassError::InvalidInput("surface projection requires positive spot/expiry, 33..100001 ordered log-moneyness nodes spanning zero, and tolerances 1e-12..1e-2".into()));
        }
        let mut spots = Vec::with_capacity(log_moneyness_nodes.len());
        let mut probabilities = Vec::with_capacity(log_moneyness_nodes.len());
        let mut calls = Vec::with_capacity(log_moneyness_nodes.len());
        for &k in log_moneyness_nodes {
            // Also checks positive total variance and Durrleman density.
            let e = surface
                .forward_call_evaluation(expiry, k, spot)
                .map_err(|e| BassError::InvalidInput(e.to_string()))?;
            let w = e.variance.total_variance;
            // F(K) = 1 + dC/dK. The skew term is essential; Phi(-d2)
            // alone gives the wrong distribution for a non-flat smile.
            let skew = pdf(e.d2) * e.variance.log_moneyness_derivative / (2.0 * w.sqrt());
            let p = if e.d2 >= 0.0 {
                cdf(-e.d2) + skew
            } else {
                1.0 - (cdf(e.d2) - skew)
            };
            if !p.is_finite()
                || !(-1e-14..=1.0 + 1e-14).contains(&p)
                || !e.strike.is_finite()
                || e.strike <= 0.0
                || !e.undiscounted_price.is_finite()
            {
                return Err(BassError::InvalidInput(
                    "invalid surface-implied marginal CDF or strike".into(),
                ));
            }
            spots.push(e.strike);
            probabilities.push(p.clamp(0.0, 1.0));
            calls.push(e.undiscounted_price);
        }
        if probabilities.windows(2).any(|p| p[1] < p[0]) {
            return Err(BassError::InvalidInput(
                "surface CDF decreases on the supplied strike grid".into(),
            ));
        }
        let low = probabilities[0];
        let high = 1.0 - probabilities[probabilities.len() - 1];
        if low.max(high) > config.tail_probability_tolerance {
            return Err(BassError::InvalidInput(format!(
                "surface projection tails ({low}, {high}) exceed tolerance; widen the log-moneyness grid"
            )));
        }
        let retained = 1.0 - low - high;
        let mut s = vec![spots[0]];
        let mut p = vec![0.0];
        for i in 1..spots.len() - 1 {
            let probability = (probabilities[i] - low) / retained;
            // Roundoff can create repeated CDF values in remote tails. Remove
            // only duplicates, never smooth or repair a decreasing CDF.
            if probability > *p.last().unwrap() && probability < 1.0 {
                s.push(spots[i]);
                p.push(probability);
            }
        }
        s.push(*spots.last().unwrap());
        p.push(1.0);
        let raw = BassMarginal::new(expiry, s, p)?;
        let unscaled_mean = raw.mean();
        let mean_scale = spot / unscaled_mean;
        if (mean_scale - 1.0).abs() > config.relative_mean_tolerance {
            return Err(BassError::InvalidInput(format!(
                "surface marginal mean correction {} exceeds tolerance; widen or refine the grid",
                mean_scale - 1.0
            )));
        }
        let marginal = BassMarginal::new(
            expiry,
            raw.spots().iter().map(|s| s * mean_scale).collect(),
            raw.probabilities().to_vec(),
        )?;
        let max_call_price_error = spots
            .iter()
            .zip(&calls)
            .map(|(k, c)| (marginal.call_unchecked(*k) - c).abs())
            .fold(0.0, f64::max);
        let retained_nodes = marginal.spots().len();
        Ok(BassMarginalProjection {
            marginal,
            diagnostics: BassSurfaceDiagnostics {
                lower_tail_probability: low,
                upper_tail_probability: high,
                retained_probability: retained,
                unscaled_mean,
                mean_scale,
                max_call_price_error,
                retained_nodes,
            },
        })
    }
}
