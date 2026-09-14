//! Discounted hybrid particle calibration.
//! Equity/Hull–White hybrid simulation and discounted-particle LSV calibration.
//! The default equity state is S0*S/F0(t), before proportional dividends.
//! Explicit escrowed cash mode simulates normalized residual equity and
//! calibrates a continuous target coordinate including the stochastic bond reserve.

use crate::mc::lsv::{LsvError, LsvLeverageSurface, LsvParticleConfig};
use crate::mc::{Philox4x32, RandomCoordinate, RandomDomain};
use crate::models::hull_white::hw_valid;
use crate::models::hull_white_dividends::HullWhiteDividendPlan;
use crate::models::{Bergomi1Factor, HullWhite1Factor, HybridCorrelation, RoughBergomi};
use pricing_numerics::NeumaierSum;

use crate::engine::processes::hull_white::*;
mod reverse;

#[derive(Clone, Debug)]
pub struct CalibratedHullWhiteLsv {
    pub surface: LsvLeverageSurface,
    pub diagnostics: Box<[HullWhiteCalibrationDiagnostics]>,
    /// Row-major discounted conditional second moment of exp(nu*X); in cash
    /// mode the loading also includes normalized residual equity / target F.
    pub conditional_second_moments: Box<[f64]>,
    /// Row-major variance correction subtracted from the deterministic Dupire target.
    pub rate_corrections: Box<[f64]>,
    /// Cash-dividend conditional equity/bond cross coefficient (before rho).
    pub conditional_cross_moments: Box<[f64]>,
    /// Cash-dividend conditional bond variance coefficient.
    pub conditional_rate_variances: Box<[f64]>,
    reverse_trace: Option<reverse::CalibrationTrace>,
}

pub fn calibrate_hull_white_lsv(
    target: &HullWhiteLsvTarget,
    factor: Bergomi1Factor,
    rates: &HullWhite1Factor,
    correlation: HybridCorrelation,
    initial_spot: f64,
    config: &LsvParticleConfig,
) -> Result<CalibratedHullWhiteLsv, HullWhiteMcError> {
    calibrate_hull_white_lsv_with_dividends(
        target,
        factor,
        rates,
        correlation,
        initial_spot,
        config,
        None,
    )
}

pub fn calibrate_hull_white_lsv_with_dividends(
    target: &HullWhiteLsvTarget,
    factor: Bergomi1Factor,
    rates: &HullWhite1Factor,
    correlation: HybridCorrelation,
    initial_spot: f64,
    config: &LsvParticleConfig,
    dividends: Option<&HullWhiteDividendPlan>,
) -> Result<CalibratedHullWhiteLsv, HullWhiteMcError> {
    calibrate_hybrid_lsv_with_dividends(
        target,
        factor.into(),
        rates,
        correlation,
        initial_spot,
        config,
        dividends,
    )
}

pub fn calibrate_rough_hull_white_lsv(
    target: &HullWhiteLsvTarget,
    factor: RoughBergomi,
    rates: &HullWhite1Factor,
    correlation: HybridCorrelation,
    initial_spot: f64,
    config: &LsvParticleConfig,
    dividends: Option<&HullWhiteDividendPlan>,
) -> Result<CalibratedHullWhiteLsv, HullWhiteMcError> {
    calibrate_hybrid_lsv_with_dividends(
        target,
        factor.into(),
        rates,
        correlation,
        initial_spot,
        config,
        dividends,
    )
}

pub fn calibrate_hybrid_lsv_with_dividends(
    target: &HullWhiteLsvTarget,
    factor: HybridVolatilityFactor,
    rates: &HullWhite1Factor,
    correlation: HybridCorrelation,
    initial_spot: f64,
    config: &LsvParticleConfig,
    dividends: Option<&HullWhiteDividendPlan>,
) -> Result<CalibratedHullWhiteLsv, HullWhiteMcError> {
    if factor.correlation() != correlation.equity_vol {
        return Err(invalid("equity_vol_correlation_mismatch", 0));
    }
    let grid = target.grid();
    let times = grid.time_nodes();
    if let Some(d) = dividends
        && (d.rates() != rates
            || d.initial_spot() != initial_spot
            || d.nodes().len() != times.len()
            || d.nodes().iter().zip(times).any(|(n, t)| n.time() != *t))
    {
        return Err(invalid("dividend_calibration_grid", 0));
    }
    let xs = grid.log_moneyness_nodes();
    let m = xs.len();
    let n = config.particle_count();
    let initial = HybridState::initial(initial_spot)?;
    let mut states = vec![initial; n];
    let kernels = times
        .windows(2)
        .map(|w| Kernel::new(rates, factor.mean_reversion(), correlation, w[0], w[1]))
        .collect::<Result<Vec<_>, _>>()?;
    let rough_driver = factor
        .rough()
        .map(|f| RoughBergomiDriverPlan::compile(f, rates, correlation, times))
        .transpose()?;
    let two_factor_driver = factor.two_factor_driver(rates, correlation, times)?;
    let rough_values = rough_driver
        .as_ref()
        .map(|driver| {
            (0..n)
                .map(|p| {
                    let shocks = hybrid_shocks(
                        config.seed(),
                        p as u64,
                        kernels.len(),
                        5,
                        RandomDomain::LsvCalibration,
                    )?;
                    driver.normalized(&shocks)
                })
                .collect::<Result<Vec<_>, HullWhiteMcError>>()
        })
        .transpose()?;
    let factor_values = if let Some(driver) = &two_factor_driver {
        Some(
            (0..n)
                .map(|p| {
                    let shocks = hybrid_shocks(
                        config.seed(),
                        p as u64,
                        kernels.len(),
                        5,
                        RandomDomain::LsvCalibration,
                    )?;
                    driver.evolve(&shocks)
                })
                .collect::<Result<Vec<_>, HullWhiteMcError>>()?,
        )
    } else {
        rough_values
    };
    let rng = Philox4x32::from_seed(config.seed());
    let steps = kernels.len();
    steps
        .checked_mul(if rough_driver.is_some() || two_factor_driver.is_some() {
            5
        } else {
            4
        })
        .and_then(|n| u32::try_from(n).ok())
        .ok_or_else(|| invalid("random_dimension", steps))?;
    let mut values = Vec::with_capacity(grid.values().len());
    let mut moments = Vec::with_capacity(values.capacity());
    let mut corrections = Vec::with_capacity(values.capacity());
    let mut crosses = Vec::with_capacity(values.capacity());
    let mut rate_variances = Vec::with_capacity(values.capacity());
    let mut diagnostics = Vec::new();
    let mut trace = config.retain_reverse_trace().then(Vec::new);
    for (r, &t) in times.iter().enumerate() {
        let shift = rates.rate_shift(t)?;
        let half_v = 0.5 * rates.integrated_variance(t)?;
        let mut sorted = states
            .iter()
            .map(|s| {
                let (f, rate_loading) = target_state(dividends, r, *s)?;
                let weight = (-s.integrated_rate_factor - half_v).exp();
                let a2 = (2.0 * factor.vol_of_vol() * s.volatility_factor).exp();
                let ratio = s.normalized_equity / f;
                let rate_ratio = rate_loading / f;
                Ok((
                    (f / initial_spot).ln(),
                    weight,
                    weight * (s.rate_factor + shift),
                    a2 * ratio * ratio,
                    f,
                    a2.sqrt() * ratio * rate_ratio,
                    rate_ratio * rate_ratio,
                ))
            })
            .collect::<Result<Vec<_>, HullWhiteMcError>>()?;
        if sorted.iter().any(|p| {
            !p.1.is_finite()
                || p.1 <= 0.0
                || !p.2.is_finite()
                || !p.3.is_finite()
                || p.3 <= 0.0
                || !p.5.is_finite()
                || !p.6.is_finite()
        }) {
            return Err(invalid("calibration_particle", r));
        }
        sorted.sort_by(|p, q| p.0.total_cmp(&q.0));
        let mut suffix_d = vec![0.0; n + 1];
        let mut suffix_y = vec![0.0; n + 1];
        let (mut sd, mut sy, mut sdf) =
            (NeumaierSum::new(), NeumaierSum::new(), NeumaierSum::new());
        for i in (0..n).rev() {
            sd.add(sorted[i].1);
            sy.add(sorted[i].2);
            sdf.add(sorted[i].1 * sorted[i].4);
            suffix_d[i] = sd.total();
            suffix_y[i] = sy.total();
        }
        let mut row = vec![None; m];
        let mut weight_sums = vec![0.0; m];
        let analytic = r == 0 || (rates.is_deterministic() && factor.vol_of_vol() == 0.0);
        for (j, &x) in xs.iter().enumerate() {
            if analytic {
                let loading = if r == 0 {
                    target_state(dividends, 0, initial)?.1 / (initial_spot * x.exp())
                } else {
                    0.0
                };
                row[j] = Some((1.0, 0.0, n as f64, loading, loading * loading));
                continue;
            }
            let h = config.log_bandwidth();
            let lo = sorted.partition_point(|p| p.0 <= x - h);
            let hi = sorted.partition_point(|p| p.0 < x + h);
            let (mut w, mut w2, mut wa2) =
                (NeumaierSum::new(), NeumaierSum::new(), NeumaierSum::new());
            let (mut wc, mut wr) = (NeumaierSum::new(), NeumaierSum::new());
            for p in &sorted[lo..hi] {
                let u = (p.0 - x) / h;
                let kw = (1.0 - u * u).powi(2) * p.1;
                w.add(kw);
                w2.add(kw * kw);
                wa2.add(kw * p.3);
                wc.add(kw * p.5);
                wr.add(kw * p.6);
            }
            let ess = w.total() * w.total() / w2.total();
            if !ess.is_finite() || ess < config.minimum_effective_samples() {
                continue;
            }
            let correction = if rates.is_deterministic() {
                0.0
            } else {
                let density = target.log_densities[r * m + j];
                if density == 0.0 {
                    continue;
                }
                let above = sorted.partition_point(|p| p.0 <= x);
                // E[Dbar*(r-f0)]=0 exactly. Empirical centering reduces noise;
                // its finite-population ratio bias is part of this versioned scheme.
                let q = (suffix_y[above] - suffix_d[above] / suffix_d[0] * suffix_y[0]) / n as f64;
                let h = dividends.map_or(0.0, |d| {
                    d.nodes()[r].deterministic_reserve() / d.nodes()[r].scale()
                });
                2.0 * q / density * (1.0 + h / (initial_spot * x.exp()))
            };
            let second = wa2.total() / w.total();
            if !second.is_finite() || second <= 0.0 || !correction.is_finite() {
                return Err(invalid("conditional_estimator", r * m + j));
            }
            row[j] = Some((
                second,
                correction,
                ess,
                wc.total() / w.total(),
                wr.total() / w.total(),
            ));
            weight_sums[j] = w.total();
        }
        let supported = row
            .iter()
            .enumerate()
            .filter_map(|(j, v)| v.map(|_| j))
            .collect::<Vec<_>>();
        if supported.is_empty() {
            return Err(invalid("no_supported_calibration_node", r));
        }
        let mut fallback = 0;
        let mut donors = Vec::with_capacity(m);
        let mut max_c: f64 = 0.0;
        let min_ess = supported
            .iter()
            .map(|&j| row[j].unwrap().2)
            .fold(f64::INFINITY, f64::min);
        for j in 0..m {
            let donor = if row[j].is_some() {
                j
            } else {
                fallback += 1;
                *supported
                    .iter()
                    .min_by(|&&p, &&q| {
                        (xs[p] - xs[j])
                            .abs()
                            .total_cmp(&(xs[q] - xs[j]).abs())
                            .then(p.cmp(&q))
                    })
                    .unwrap()
            };
            let (second, correction, _, cross, rate_variance) = row[donor].unwrap();
            donors.push(donor);
            let value = corrected_leverage(
                grid.values()[r * m + j] - correction,
                second,
                correlation.equity_rate * cross,
                rate_variance,
            )?;
            if !value.is_finite() || value <= 0.0 {
                return Err(invalid("non_positive_rate_corrected_variance", r * m + j));
            }
            values.push(value);
            moments.push(second);
            corrections.push(correction);
            crosses.push(cross);
            rate_variances.push(rate_variance);
            max_c = max_c.max(correction.abs());
        }
        diagnostics.push(HullWhiteCalibrationDiagnostics {
            time: t,
            minimum_effective_samples: min_ess,
            fallback_nodes: fallback,
            mean_relative_discount: sd.total() / n as f64,
            mean_discounted_normalized_equity: sdf.total() / n as f64,
            maximum_rate_correction: max_c,
        });
        if let Some(trace) = &mut trace {
            trace.push(reverse::TraceRow {
                states: states.clone().into(),
                weight_sums: weight_sums.into(),
                donors: donors.into(),
                analytic,
            });
        }
        if r < steps {
            for (p, state) in states.iter_mut().enumerate() {
                let x = (target_state(dividends, r, *state)?.0 / initial_spot).ln();
                let j = xs.partition_point(|v| *v <= x).saturating_sub(1).min(m - 2);
                let w = ((x - xs[j]) / (xs[j + 1] - xs[j])).clamp(0.0, 1.0);
                let l2 = (1.0 - w) * values[r * m + j] + w * values[r * m + j + 1];
                let z = std::array::from_fn(|d| {
                    rng.standard_normal(RandomCoordinate::new(
                        p as u64,
                        (d * steps + r) as u32,
                        RandomDomain::LsvCalibration,
                    ))
                });
                *state = kernels[r].advance(*state, l2, factor.vol_of_vol(), z, r)?;
                if let Some(values) = &factor_values {
                    state.volatility_factor = values[p][r + 1];
                }
            }
        }
    }
    let mut calibrated = CalibratedHullWhiteLsv {
        surface: LsvLeverageSurface::new(times.to_vec(), xs.to_vec(), values, initial_spot)?,
        diagnostics: diagnostics.into(),
        conditional_second_moments: moments.into(),
        rate_corrections: corrections.into(),
        conditional_cross_moments: crosses.into(),
        conditional_rate_variances: rate_variances.into(),
        reverse_trace: trace.map(|rows| reverse::CalibrationTrace {
            rows: rows.into(),
            target: target.clone(),
            factor,
            rates: rates.clone(),
            correlation,
            config: config.clone(),
            dividends: dividends.cloned(),
            kernels: kernels.into(),
            primal_fingerprint: [0; 32],
        }),
    };
    if config.retain_reverse_trace() {
        let fingerprint = calibrated.reverse_primal_fingerprint();
        calibrated
            .reverse_trace
            .as_mut()
            .expect("reverse trace requested")
            .primal_fingerprint = fingerprint;
    }
    Ok(calibrated)
}

// Solve A*L^2+2*B*L+C=target on the upper positive branch. Rationalize
// the subtraction for B>=0; the zero-bond limit preserves the v1 calculation.
fn corrected_leverage(target: f64, a: f64, b: f64, c: f64) -> Result<f64, HullWhiteMcError> {
    if b == 0.0 && c == 0.0 {
        return Ok(target / a);
    }
    let residual = target - c;
    let discriminant = b * b + a * residual;
    if !discriminant.is_finite() || discriminant < 0.0 {
        return Err(invalid("cash_dividend_leverage_discriminant", 0));
    }
    let root = discriminant.sqrt();
    let leverage = if b >= 0.0 {
        residual / (root + b)
    } else {
        (root - b) / a
    };
    if !leverage.is_finite() || leverage <= 0.0 {
        return Err(invalid("positive_cash_dividend_leverage", 0));
    }
    Ok(leverage * leverage)
}
