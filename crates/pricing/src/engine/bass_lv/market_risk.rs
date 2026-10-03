//! Full quote bump/reprojection/recalibration with paired Monte Carlo errors.
use super::*;
use crate::market::MarketIvSurface;
use crate::models::bass_lv::{BassSurfaceDiagnostics, BassSurfaceProjectionConfig};
use crate::risk::VEGA_KT_MARKET_SCALE;

pub const BASS_VEGA_KT_METHOD: &str = "recalibrated-central-crn-v1";

/// Quotes are in the martingale coordinate log(K/spot), flattened time-major.
/// Every quote maturity becomes a calibrated Bass marginal.
#[derive(Clone, Debug)]
pub struct BassMarketIvModel {
    surface: MarketIvSurface,
    projection_nodes: Vec<f64>,
    config: BassLvConfig,
    projection_config: BassSurfaceProjectionConfig,
    model: BassLvModel,
    projection_diagnostics: Vec<BassSurfaceDiagnostics>,
}
impl BassMarketIvModel {
    pub fn calibrate(
        spot: f64,
        surface: MarketIvSurface,
        projection_nodes: Vec<f64>,
        config: BassLvConfig,
        projection_config: BassSurfaceProjectionConfig,
    ) -> Result<Self, BassError> {
        let mut marginals = Vec::new();
        let mut projection_diagnostics = Vec::new();
        for &time in surface.maturity_nodes() {
            let p = BassMarginal::from_surface(
                time,
                spot,
                &surface,
                &projection_nodes,
                projection_config,
            )?;
            marginals.push(p.marginal);
            projection_diagnostics.push(p.diagnostics);
        }
        let model = BassLvModel::calibrate(spot, marginals, config)?;
        Ok(Self {
            surface,
            projection_nodes,
            config,
            projection_config,
            model,
            projection_diagnostics,
        })
    }
    pub fn model(&self) -> &BassLvModel {
        &self.model
    }
    pub fn surface(&self) -> &MarketIvSurface {
        &self.surface
    }
    pub fn projection_diagnostics(&self) -> &[BassSurfaceDiagnostics] {
        &self.projection_diagnostics
    }

    /// Absolute IV shifts. Rebuilds both the marginal projection and calibration,
    /// including mean normalization, automatic Brownian bounds and initial W0.
    pub fn bumped(&self, shifts: &[f64]) -> Result<Self, BassError> {
        if shifts.len() != self.surface.implied_volatilities().len()
            || shifts.iter().any(|x| !x.is_finite())
        {
            return Err(BassError::InvalidInput(
                "market IV shifts must be finite and match the time-major quote array".into(),
            ));
        }
        let surface = MarketIvSurface::new(
            self.surface.maturity_nodes().to_vec(),
            self.surface.log_moneyness_nodes().to_vec(),
            self.surface
                .implied_volatilities()
                .iter()
                .zip(shifts)
                .map(|(q, s)| q + s)
                .collect(),
        )
        .map_err(|e| BassError::InvalidInput(e.to_string()))?;
        Self::calibrate(
            self.model.spot,
            surface,
            self.projection_nodes.clone(),
            self.config,
            self.projection_config,
        )
    }

    /// Precompute 2Q single-quote and two parallel scenarios, reusable by payoffs.
    /// No arbitrage repair, one-sided fallback, or automatic bump resizing.
    pub fn compile_vega_kt(
        &self,
        observation_times: Vec<f64>,
        bump_size: f64,
    ) -> Result<BassVegaKtRiskPlan, BassError> {
        if !bump_size.is_finite()
            || bump_size <= 0.0
            || !(2.0 * bump_size).is_finite()
            || self
                .surface
                .implied_volatilities()
                .iter()
                .any(|q| q - bump_size <= 0.0 || q + bump_size == *q || q - bump_size == *q)
        {
            return Err(BassError::InvalidInput(
                "VegaKT bump must be positive, representable, and smaller than every IV quote"
                    .into(),
            ));
        }
        let simulation = self.model.compile_simulation(observation_times.clone())?;
        let count = self.surface.implied_volatilities().len();
        let mut pairs = Vec::with_capacity(count + 1);
        let mut diagnostics = Vec::with_capacity(2 * (count + 1));
        for index in 0..=count {
            let quote_index = (index < count).then_some(index);
            let mut scenarios = Vec::with_capacity(2);
            for shift in [bump_size, -bump_size] {
                let mut shifts = vec![0.0; count];
                if let Some(j) = quote_index {
                    shifts[j] = shift;
                } else {
                    shifts.fill(shift);
                }
                let label = scenario_label(quote_index, shift);
                let model = self
                    .bumped(&shifts)
                    .map_err(|e| scenario_error(&label, e))?;
                let plan = model
                    .model
                    .compile_simulation(observation_times.clone())
                    .map_err(|e| scenario_error(&label, e))?;
                diagnostics.push(BassVegaKtScenarioDiagnostics {
                    quote_index,
                    shift,
                    calibration: model.model.diagnostics.clone(),
                    projection: model.projection_diagnostics,
                    initial_spot_error: model.model.initial_spot_error,
                });
                scenarios.push(plan);
            }
            let down = scenarios.pop().unwrap();
            let up = scenarios.pop().unwrap();
            pairs.push((up, down));
        }
        Ok(BassVegaKtRiskPlan {
            simulation,
            pairs,
            diagnostics,
            bump_size,
            maturity_nodes: self.surface.maturity_nodes().to_vec(),
            log_moneyness_nodes: self.surface.log_moneyness_nodes().to_vec(),
            implied_volatilities: self.surface.implied_volatilities().to_vec(),
        })
    }
}

#[derive(Clone, Debug)]
pub struct BassVegaKtScenarioDiagnostics {
    /// None denotes the simultaneous parallel bump. Others are time-major indices.
    pub quote_index: Option<usize>,
    pub shift: f64,
    pub calibration: Vec<BassCalibrationDiagnostics>,
    pub projection: Vec<BassSurfaceDiagnostics>,
    pub initial_spot_error: f64,
}

#[derive(Clone, Debug)]
pub struct BassVegaKtRisk {
    pub estimate: BassEstimate,
    /// Currency per unit absolute volatility, in time-major quote order.
    pub sensitivities: Vec<f64>,
    /// Paired-difference sampling errors; exclude bump and calibration bias.
    pub standard_errors: Vec<f64>,
    /// Independently revalued all-quotes-at-once central bump.
    pub parallel_sensitivity: f64,
    pub parallel_standard_error: f64,
    pub bucket_sum: f64,
    /// Includes cross-bucket covariance, estimated from each path's bucket sum.
    pub bucket_sum_standard_error: f64,
    pub bump_size: f64,
    pub maturity_nodes: Vec<f64>,
    pub log_moneyness_nodes: Vec<f64>,
    pub implied_volatilities: Vec<f64>,
}
impl BassVegaKtRisk {
    pub fn method(&self) -> &'static str {
        BASS_VEGA_KT_METHOD
    }
    pub fn vega_per_vol_point(&self) -> Vec<f64> {
        self.sensitivities
            .iter()
            .map(|v| v * VEGA_KT_MARKET_SCALE)
            .collect()
    }
    pub fn standard_errors_per_vol_point(&self) -> Vec<f64> {
        self.standard_errors
            .iter()
            .map(|v| v * VEGA_KT_MARKET_SCALE)
            .collect()
    }
}

#[derive(Clone, Debug)]
pub struct BassVegaKtRiskPlan {
    simulation: BassSimulationPlan,
    /// Last pair is the all-quotes parallel bump.
    pairs: Vec<(BassSimulationPlan, BassSimulationPlan)>,
    diagnostics: Vec<BassVegaKtScenarioDiagnostics>,
    bump_size: f64,
    maturity_nodes: Vec<f64>,
    log_moneyness_nodes: Vec<f64>,
    implied_volatilities: Vec<f64>,
}
impl BassVegaKtRiskPlan {
    pub(super) fn pairs(&self) -> &[(BassSimulationPlan, BassSimulationPlan)] {
        &self.pairs
    }
    pub fn simulation(&self) -> &BassSimulationPlan {
        &self.simulation
    }
    pub fn diagnostics(&self) -> &[BassVegaKtScenarioDiagnostics] {
        &self.diagnostics
    }
    pub fn bump_size(&self) -> f64 {
        self.bump_size
    }
    pub fn maturity_nodes(&self) -> &[f64] {
        &self.maturity_nodes
    }
    pub fn log_moneyness_nodes(&self) -> &[f64] {
        &self.log_moneyness_nodes
    }
    pub fn implied_volatilities(&self) -> &[f64] {
        &self.implied_volatilities
    }

    /// The callback must be a deterministic undiscounted path payoff.
    /// All scenarios share each path's Brownian increments. Spot, observation
    /// dates, quote axes, projection nodes and discount factor stay fixed.
    pub fn price<F: Fn(&[f64]) -> f64>(
        &self,
        paths: usize,
        seed: u64,
        discount_factor: f64,
        payoff: F,
    ) -> Result<BassVegaKtRisk, BassError> {
        if paths < 2 || !discount_factor.is_finite() || discount_factor <= 0.0 {
            return Err(BassError::InvalidInput(
                "pricing requires at least two paths and a positive finite discount factor".into(),
            ));
        }
        let count = self.implied_volatilities.len();
        let rng = Philox4x32::from_seed(seed);
        let mut normals = vec![0.0; self.simulation.normal_count()];
        let mut full = vec![0.0; self.simulation.time_nodes.len()];
        let mut output = vec![0.0; self.simulation.observation_times.len()];
        let mut price = Moments::default();
        let mut risks = vec![Moments::default(); count + 1];
        let mut sum = Moments::default();
        for p in 0..paths {
            for (d, z) in normals.iter_mut().enumerate() {
                *z = rng.standard_normal(RandomCoordinate::new(
                    p as u64,
                    d as u32,
                    RandomDomain::Valuation,
                ));
            }
            self.simulation
                .fill_path(|d| normals[d], &mut full, &mut output)?;
            price.push(discount_factor * payoff(&output), p + 1)?;
            let mut bucket_sum = 0.0;
            for (j, ((up, down), moments)) in self.pairs.iter().zip(&mut risks).enumerate() {
                let index = (j < count).then_some(j);
                up.fill_path(|d| normals[d], &mut full, &mut output)
                    .map_err(|e| scenario_error(&scenario_label(index, self.bump_size), e))?;
                let up_value = discount_factor * payoff(&output);
                down.fill_path(|d| normals[d], &mut full, &mut output)
                    .map_err(|e| scenario_error(&scenario_label(index, -self.bump_size), e))?;
                let down_value = discount_factor * payoff(&output);
                let derivative = (up_value - down_value) / (2.0 * self.bump_size);
                moments.push(derivative, p + 1)?;
                if j < count {
                    bucket_sum += derivative;
                }
            }
            sum.push(bucket_sum, p + 1)?;
        }
        Ok(BassVegaKtRisk {
            estimate: BassEstimate {
                price: price.mean,
                standard_error: price.error(paths),
                paths,
                seed,
            },
            sensitivities: risks[..count].iter().map(|r| r.mean).collect(),
            standard_errors: risks[..count].iter().map(|r| r.error(paths)).collect(),
            parallel_sensitivity: risks[count].mean,
            parallel_standard_error: risks[count].error(paths),
            bucket_sum: sum.mean,
            bucket_sum_standard_error: sum.error(paths),
            bump_size: self.bump_size,
            maturity_nodes: self.maturity_nodes.clone(),
            log_moneyness_nodes: self.log_moneyness_nodes.clone(),
            implied_volatilities: self.implied_volatilities.clone(),
        })
    }
    pub fn price_european(
        &self,
        strike: f64,
        is_call: bool,
        paths: usize,
        seed: u64,
        discount_factor: f64,
    ) -> Result<BassVegaKtRisk, BassError> {
        validate_strike(strike)?;
        let sign = if is_call { 1.0 } else { -1.0 };
        self.price(paths, seed, discount_factor, |s| {
            (sign * (s[s.len() - 1] - strike)).max(0.0)
        })
    }
    pub fn price_asian(
        &self,
        strike: f64,
        is_call: bool,
        paths: usize,
        seed: u64,
        discount_factor: f64,
    ) -> Result<BassVegaKtRisk, BassError> {
        validate_strike(strike)?;
        let sign = if is_call { 1.0 } else { -1.0 };
        self.price(paths, seed, discount_factor, |s| {
            (sign * (s.iter().sum::<f64>() / s.len() as f64 - strike)).max(0.0)
        })
    }
}

#[derive(Clone, Default, Debug)]
struct Moments {
    mean: f64,
    m2: f64,
}
impl Moments {
    fn push(&mut self, x: f64, n: usize) -> Result<(), BassError> {
        if !x.is_finite() {
            return Err(BassError::Numerical(
                "nonfinite Bass VegaKT payoff or difference".into(),
            ));
        }
        let delta = x - self.mean;
        self.mean += delta / n as f64;
        self.m2 += delta * (x - self.mean);
        if !self.mean.is_finite() || !self.m2.is_finite() {
            return Err(BassError::Numerical(
                "Bass VegaKT statistics overflowed".into(),
            ));
        }
        Ok(())
    }
    fn error(&self, n: usize) -> f64 {
        (self.m2 / ((n - 1) as f64 * n as f64)).sqrt()
    }
}
fn scenario_label(index: Option<usize>, shift: f64) -> String {
    match index {
        Some(i) => format!("quote[{i}] shift {shift:+}"),
        None => format!("parallel shift {shift:+}"),
    }
}
fn scenario_error(label: &str, e: BassError) -> BassError {
    BassError::Numerical(format!("Bass VegaKT {label}: {e}"))
}
fn validate_strike(strike: f64) -> Result<(), BassError> {
    if !strike.is_finite() || strike < 0.0 {
        Err(BassError::InvalidInput(
            "strike must be finite and nonnegative".into(),
        ))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests;
