//! Parallel shifts of original residual Local-volatility nodes, followed by full
//! finite-particle recalibration. This coordinate is not quoted market-IV Vega.
use super::sampling::BumpStatistics;
use super::*;

const METHOD: &str =
    "buehler-rough-residual-lsv-continuous-bridge-parallel-local-vol-recalibrated-crn-v1";
const WIDTH: usize = 6;

/// Finite differences for a parallel absolute shift of sqrt(original target
/// Local variance). Every shifted target is refined and recalibrated with the
/// original calibration seed/configuration. Valuation normals are also shared.
/// Estimates are per unit absolute volatility; one vol point equals 0.01.
#[derive(Clone, Debug, PartialEq)]
pub struct StochasticDividendContinuousBarrierLocalVolatilityRisk {
    pub price: StochasticDividendPrice,
    pub local_volatility_bumps: [f64; 3],
    pub vega_estimates: [f64; 3],
    pub vega_standard_errors: [f64; 3],
    /// Vega(h/2)-Vega(h), Vega(h)-Vega(2h), paired before sampling reduction.
    pub bump_differences: [f64; 2],
    pub bump_difference_standard_errors: [f64; 2],
    /// Seven full valuation paths, including the base, per base evaluated path.
    /// Calibration particle paths are excluded.
    pub scenario_evaluated_paths: u128,
    pub payoff_evaluations: u128,
    pub recalibration_count: usize,
    pub risk_fingerprint: Fingerprint,
    pub method: &'static str,
}
impl StochasticDividendContinuousBarrierLocalVolatilityRisk {
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
    pub const fn uncertainty_scope(&self) -> &'static str {
        "pricing_only_fixed_calibration_seed_grid_bridge_and_bump"
    }
}

impl StochasticDividendContinuousBarrierPlan {
    /// Bump every original target node as v -> (sqrt(v) +/- h)^2, then refine
    /// onto the unchanged execution grid and recalibrate. Bump-before-refinement
    /// matters for a non-flat grid. All nodes must stay positive, representably
    /// shifted and inside the original variance floor/cap; no clipping occurs.
    /// The half/base/double ladder is validated in full before any recalibration.
    /// This is residual Local-volatility risk, not market-IV Vega or VegaKT.
    pub fn evaluate_parallel_local_volatility_risk(
        &self,
        local_volatility_bump: f64,
    ) -> Result<StochasticDividendContinuousBarrierLocalVolatilityRisk, MonteCarloError> {
        let bumps = [
            0.5 * local_volatility_bump,
            local_volatility_bump,
            2.0 * local_volatility_bump,
        ];
        let scenarios = self.local_volatility_scenarios(&bumps)?;
        let BumpStatistics {
            values,
            errors,
            units,
            paths,
        } = self.bump_statistics(WIDTH, |z, bridge, antithetic, out| {
            self.local_volatility_sample(&scenarios, &bumps, z, bridge, antithetic, out)
        })?;
        let mut hash = blake3::Hasher::new();
        hash.update(METHOD.as_bytes());
        hash.update(self.plan_fingerprint().as_bytes());
        for h in bumps {
            hash.update(&h.to_bits().to_le_bytes());
        }
        Ok(StochasticDividendContinuousBarrierLocalVolatilityRisk {
            price: StochasticDividendPrice {
                value: values[0],
                standard_error: errors[0],
                independent_sampling_units: units,
                evaluated_paths: paths,
                plan_fingerprint: self.plan_fingerprint(),
                scheme: SCHEME,
            },
            local_volatility_bumps: bumps,
            vega_estimates: [values[1], values[2], values[3]],
            vega_standard_errors: [errors[1], errors[2], errors[3]],
            bump_differences: [values[4], values[5]],
            bump_difference_standard_errors: [errors[4], errors[5]],
            scenario_evaluated_paths: 7 * paths,
            payoff_evaluations: 7 * paths,
            recalibration_count: 6,
            risk_fingerprint: Fingerprint::from_bytes(*hash.finalize().as_bytes()),
            method: METHOD,
        })
    }

    pub(super) fn local_volatility_scenarios(
        &self,
        bumps: &[f64; 3],
    ) -> Result<Vec<StochasticDividendPathPlan>, MonteCarloError> {
        self.local_volatility_scenarios_for_nodes(bumps, &[None])
    }

    pub(super) fn original_local_variance_target(
        &self,
    ) -> Result<&LocalVarianceGrid, MonteCarloError> {
        match self.inner.lsv.as_ref() {
            Some(StochasticDividendLsvCalibration::Rough {
                original_target, ..
            }) => Ok(original_target),
            _ => Err(invalid("continuous_local_volatility_calibration").into()),
        }
    }

    // None denotes the parallel shift; Some(index) denotes one original node.
    // Validate and refine every target before starting any particle calibration.
    pub(super) fn local_volatility_scenarios_for_nodes(
        &self,
        bumps: &[f64; 3],
        nodes: &[Option<usize>],
    ) -> Result<Vec<StochasticDividendPathPlan>, MonteCarloError> {
        let original_target = self.original_local_variance_target()?;
        let mut targets = Vec::new();
        for &node in nodes {
            for &h in bumps {
                if !h.is_finite() || h <= 0.0 || !(2.0 * h).is_finite() {
                    return Err(invalid("continuous_local_volatility_bump").into());
                }
                for shift in [-h, h] {
                    targets.push(shifted_target(original_target, shift, node)?);
                }
            }
        }
        self.recalibrated_target_scenarios(&targets)
    }

    // All original targets must validate before any refinement or calibration.
    // Refinement interpolates original VARIANCE nodes onto the fixed price grid.
    pub(super) fn recalibrated_target_scenarios(
        &self,
        original_targets: &[LocalVarianceGrid],
    ) -> Result<Vec<StochasticDividendPathPlan>, MonteCarloError> {
        let Some(StochasticDividendLsvCalibration::Rough {
            calibration,
            dividend_volatility_correlation,
            ..
        }) = self.inner.lsv.as_ref()
        else {
            return Err(invalid("continuous_local_volatility_calibration").into());
        };
        let mut targets = Vec::with_capacity(original_targets.len());
        for original in original_targets {
            let mut values = Vec::with_capacity(calibration.target().values().len());
            for &t in self.time_nodes() {
                for &x in original.log_moneyness_nodes() {
                    values.push(original.interpolate(t, x)?.value);
                }
            }
            targets.push(LocalVarianceGrid::new(
                self.time_nodes().to_vec(),
                original.log_moneyness_nodes().to_vec(),
                values,
                original.floor(),
                original.cap(),
            )?);
        }
        let executor = DeterministicExecutor::new(self.inner.policy)?;
        targets
            .iter()
            .map(|target| {
                let bumped = calibrate_rough_bergomi_lsv_parallel(
                    target,
                    calibration.model(),
                    self.risky_spot(),
                    calibration.config().clone(),
                    &executor,
                )?;
                let path = self.inner.path.clone().with_rough_bergomi_lsv(
                    calibration.model(),
                    *dividend_volatility_correlation,
                    bumped.surface().clone(),
                )?;
                if path.random_dimension() != self.inner.path.random_dimension()
                    || path.times() != self.time_nodes()
                {
                    return Err(invalid("continuous_local_volatility_scenario_grid").into());
                }
                Ok(path)
            })
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn local_volatility_sample(
        &self,
        scenarios: &[StochasticDividendPathPlan],
        bumps: &[f64; 3],
        mut z: Vec<f64>,
        bridge: Option<&BrownianBridgePlan>,
        antithetic: bool,
        out: &mut [f64],
    ) -> Result<(), MonteCarloError> {
        if let Some(bridge) = bridge {
            let count = self.random_factor_count();
            for factor in 0..count {
                let input = z
                    .iter()
                    .skip(factor)
                    .step_by(count)
                    .copied()
                    .collect::<Vec<_>>();
                let output = bridge
                    .apply_one_factor(&input)
                    .map_err(|e| MonteCarloError::LocalVol(e.into()))?;
                for (step, value) in output.into_iter().enumerate() {
                    z[count * step + factor] = value;
                }
            }
        }
        out.fill(0.0);
        for &sign in if antithetic {
            &[1.0, -1.0][..]
        } else {
            &[1.0][..]
        } {
            let shocks = z.iter().map(|v| sign * v).collect::<Vec<_>>();
            out[0] += self.path_payoff(&shocks)?;
            for (bucket, ladder) in scenarios.as_chunks::<6>().0.iter().enumerate() {
                for (j, (pair, h)) in ladder.as_chunks::<2>().0.iter().zip(bumps).enumerate() {
                    let payoff = |path: &StochasticDividendPathPlan| {
                        let (states, volatilities) =
                            path.evolve_rough_path_with_volatilities(&shocks)?;
                        self.payoff_from_trace(path, &states, &volatilities)
                    };
                    out[1 + 5 * bucket + j] += (payoff(&pair[1])? - payoff(&pair[0])?) / (2.0 * h);
                }
            }
        }
        if antithetic {
            for v in out.iter_mut() {
                *v *= 0.5;
            }
        }
        for row in out[1..].as_chunks_mut::<5>().0 {
            row[3] = row[0] - row[1];
            row[4] = row[1] - row[2];
        }
        if out.iter().any(|v| !v.is_finite()) {
            return Err(invalid("continuous_local_volatility_sample").into());
        }
        Ok(())
    }
}

fn shifted_target(
    target: &LocalVarianceGrid,
    shift: f64,
    node: Option<usize>,
) -> Result<LocalVarianceGrid, MonteCarloError> {
    let values = target
        .values()
        .iter()
        .enumerate()
        .map(|(index, &v)| {
            if node.is_some_and(|node| node != index) {
                return Ok(v);
            }
            let sigma = v.sqrt();
            let bumped = sigma + shift;
            let variance = bumped * bumped;
            if !bumped.is_finite()
                || bumped <= 0.0
                || bumped == sigma
                || !variance.is_finite()
                || (shift < 0.0 && variance >= v)
                || (shift > 0.0 && variance <= v)
            {
                return Err(invalid("continuous_local_volatility_shifted_node"));
            }
            Ok(variance)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(LocalVarianceGrid::new(
        target.time_nodes().to_vec(),
        target.log_moneyness_nodes().to_vec(),
        values,
        target.floor(),
        target.cap(),
    )?)
}
