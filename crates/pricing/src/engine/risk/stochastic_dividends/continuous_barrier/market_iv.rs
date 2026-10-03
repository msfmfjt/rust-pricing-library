//! Parallel bumps of retained residual-forward IV quotes, Dupire rebuild and
//! complete finite-particle LSV recalibration with common random numbers.
use super::sampling::BumpStatistics;
use super::*;
use crate::market::{MARKET_IV_INTERPOLATION, MarketIvSurface};

const SOURCE_POLICY: &str = "continuous-residual-market-iv-original-grid-dupire-v1";
const METHOD: &str =
    "buehler-rough-residual-lsv-continuous-bridge-parallel-market-iv-recalibrated-crn-v1";

/// Finite parallel quote-IV risk per unit absolute volatility. Each shifted
/// quote surface rebuilds the original Dupire grid and recalibrates leverage.
/// Inputs are residual-forward IV, not unconverted physical-Spot option quotes.
#[derive(Clone, Debug, PartialEq)]
pub struct StochasticDividendContinuousBarrierMarketIvRisk {
    pub price: StochasticDividendPrice,
    pub implied_volatility_bumps: [f64; 3],
    pub vega_estimates: [f64; 3],
    pub vega_standard_errors: [f64; 3],
    pub bump_differences: [f64; 2],
    pub bump_difference_standard_errors: [f64; 2],
    pub quote_maturity_nodes: Vec<f64>,
    pub quote_log_moneyness_nodes: Vec<f64>,
    pub implied_volatilities: Vec<f64>,
    pub scenario_evaluated_paths: u128,
    pub payoff_evaluations: u128,
    pub recalibration_count: usize,
    pub risk_fingerprint: Fingerprint,
    pub method: &'static str,
}
impl StochasticDividendContinuousBarrierMarketIvRisk {
    #[must_use]
    pub fn vega(&self) -> f64 {
        self.vega_estimates[1]
    }
    #[must_use]
    pub fn standard_error(&self) -> f64 {
        self.vega_standard_errors[1]
    }
    #[must_use]
    pub fn vega_per_vol_point(&self) -> f64 {
        0.01 * self.vega()
    }
    #[must_use]
    pub fn standard_error_per_vol_point(&self) -> f64 {
        0.01 * self.standard_error()
    }
    #[must_use]
    pub const fn interpolation(&self) -> &'static str {
        MARKET_IV_INTERPOLATION
    }
    #[must_use]
    pub const fn uncertainty_scope(&self) -> &'static str {
        "pricing_only_fixed_calibration_seed_grid_bridge_and_quote_interpolation"
    }
}
impl StochasticDividendContinuousBarrierPlan {
    /// Retain explicit residual-forward quote provenance. The surface must
    /// reproduce every ORIGINAL target variance exactly, without floor/cap repair.
    /// This consuming builder can bind a source once; it changes the fingerprint
    /// but not the calibrated surface or price. Reporting IV is never inferred.
    pub fn with_market_iv_surface(
        mut self,
        surface: MarketIvSurface,
    ) -> Result<Self, MonteCarloError> {
        if self.market_iv_surface.is_some() {
            return Err(invalid("continuous_market_iv_source_already_bound").into());
        }
        let target = self.original_local_variance_target()?;
        let rebuilt = surface.local_variance_grid(
            target.time_nodes().to_vec(),
            target.log_moneyness_nodes().to_vec(),
            target.floor(),
            target.cap(),
        )?;
        if !target.repairs().is_empty() || rebuilt.values() != target.values() {
            return Err(invalid("continuous_market_iv_original_target_mismatch").into());
        }
        let mut hash = blake3::Hasher::new();
        hash.update(self.plan_fingerprint().as_bytes());
        hash.update(SOURCE_POLICY.as_bytes());
        hash.update(MARKET_IV_INTERPOLATION.as_bytes());
        for axis in [
            surface.maturity_nodes(),
            surface.log_moneyness_nodes(),
            surface.implied_volatilities(),
        ] {
            hash.update(&(axis.len() as u64).to_le_bytes());
            for value in axis {
                hash.update(&value.to_bits().to_le_bytes());
            }
        }
        self.inner.fingerprint = Fingerprint::from_bytes(*hash.finalize().as_bytes());
        self.market_iv_surface = Some(surface);
        Ok(self)
    }

    #[must_use]
    pub const fn supports_market_iv_risk(&self) -> bool {
        self.market_iv_surface.is_some()
    }

    /// Shift every retained IV quote by +/- h/2, h and 2h. Reconstruct natural
    /// cubic total variance / linear time interpolation and original-grid Dupire
    /// variance, then refine and fully recalibrate each of the six scenarios.
    /// Reject nonpositive/unrepresentable quotes and any sampled arbitrage or
    /// variance repair before beginning scenario calibrations. No SSVI refit occurs.
    pub fn evaluate_parallel_market_iv_risk(
        &self,
        implied_volatility_bump: f64,
    ) -> Result<StochasticDividendContinuousBarrierMarketIvRisk, MonteCarloError> {
        let surface = self
            .market_iv_surface
            .as_ref()
            .ok_or_else(|| invalid("continuous_market_iv_source_not_retained"))?;
        let target = self.original_local_variance_target()?;
        let bumps = [
            0.5 * implied_volatility_bump,
            implied_volatility_bump,
            2.0 * implied_volatility_bump,
        ];
        let mut targets = Vec::with_capacity(6);
        for &h in &bumps {
            if !h.is_finite() || h <= 0.0 || !(2.0 * h).is_finite() {
                return Err(invalid("continuous_market_iv_bump").into());
            }
            for shift in [-h, h] {
                let quotes = surface
                    .implied_volatilities()
                    .iter()
                    .map(|&sigma| {
                        let bumped = sigma + shift;
                        if !bumped.is_finite()
                            || bumped <= 0.0
                            || bumped == sigma
                            || bumped * bumped == sigma * sigma
                        {
                            return Err(invalid("continuous_market_iv_shifted_quote"));
                        }
                        Ok(bumped)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let shifted = MarketIvSurface::new(
                    surface.maturity_nodes().to_vec(),
                    surface.log_moneyness_nodes().to_vec(),
                    quotes,
                )?;
                targets.push(shifted.local_variance_grid(
                    target.time_nodes().to_vec(),
                    target.log_moneyness_nodes().to_vec(),
                    target.floor(),
                    target.cap(),
                )?);
            }
        }
        let scenarios = self.recalibrated_target_scenarios(&targets)?;
        let BumpStatistics {
            values,
            errors,
            units,
            paths,
        } = self.bump_statistics(6, |z, bridge, antithetic, out| {
            self.local_volatility_sample(&scenarios, &bumps, z, bridge, antithetic, out)
        })?;
        let mut hash = blake3::Hasher::new();
        hash.update(METHOD.as_bytes());
        hash.update(self.plan_fingerprint().as_bytes());
        for h in bumps {
            hash.update(&h.to_bits().to_le_bytes());
        }
        Ok(StochasticDividendContinuousBarrierMarketIvRisk {
            price: StochasticDividendPrice {
                value: values[0],
                standard_error: errors[0],
                independent_sampling_units: units,
                evaluated_paths: paths,
                plan_fingerprint: self.plan_fingerprint(),
                scheme: SCHEME,
            },
            implied_volatility_bumps: bumps,
            vega_estimates: [values[1], values[2], values[3]],
            vega_standard_errors: [errors[1], errors[2], errors[3]],
            bump_differences: [values[4], values[5]],
            bump_difference_standard_errors: [errors[4], errors[5]],
            quote_maturity_nodes: surface.maturity_nodes().to_vec(),
            quote_log_moneyness_nodes: surface.log_moneyness_nodes().to_vec(),
            implied_volatilities: surface.implied_volatilities().to_vec(),
            scenario_evaluated_paths: 7 * paths,
            payoff_evaluations: 7 * paths,
            recalibration_count: 6,
            risk_fingerprint: Fingerprint::from_bytes(*hash.finalize().as_bytes()),
            method: METHOD,
        })
    }
}
