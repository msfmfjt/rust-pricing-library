//! Paired finite differences of the continuous bridge price, including endpoint
//! and cash-jump branch changes. No pathwise derivative or zero-bump limit is claimed.
use super::*;
use crate::core::PositiveF64;
use crate::risk::SpotBump;

const METHOD: &str = "buehler-rough-residual-lsv-continuous-bridge-crn-spot-bump-v1";
const WIDTH: usize = 6; // Price, three finite-bump Deltas, two paired gaps.
const GAMMA_METHOD: &str = "buehler-rough-residual-lsv-continuous-bridge-crn-price-gamma-v1";
const GAMMA_WIDTH: usize = 11; // Also three price Gammas and two paired Gamma gaps.

/// Half/base/double Spot-bump estimates with common random numbers. The leverage
/// surface is re-anchored in residual equity; its calibrated values stay fixed.
/// Errors condition on calibration and the grid, and exclude bridge/bump bias.
/// Gaps diagnose bump dependence; they are not derivative error bounds.
#[derive(Clone, Debug, PartialEq)]
pub struct StochasticDividendContinuousBarrierSpotRisk {
    pub price: StochasticDividendPrice,
    pub spot: f64,
    pub spot_bumps: [f64; 3],
    pub delta_estimates: [f64; 3],
    pub delta_standard_errors: [f64; 3],
    /// Delta(h/2)-Delta(h), Delta(h)-Delta(2h), paired on every sampling unit.
    pub bump_differences: [f64; 2],
    pub bump_difference_standard_errors: [f64; 2],
    /// One normalized state/volatility trace supports seven physical payoffs.
    pub payoff_evaluations: u128,
    pub risk_fingerprint: Fingerprint,
    pub method: &'static str,
}
impl StochasticDividendContinuousBarrierSpotRisk {
    #[must_use]
    pub fn delta(&self) -> f64 {
        self.delta_estimates[1]
    }
    #[must_use]
    pub fn standard_error(&self) -> f64 {
        self.delta_standard_errors[1]
    }
    #[must_use]
    pub const fn uncertainty_scope(&self) -> &'static str {
        "sampling_only_fixed_calibration_grid_bridge_and_bump"
    }
}

/// Central second price differences on a half/base/double Spot-bump ladder.
/// Delta is the central price difference at the base bump. Both include changes
/// of endpoint/cash-jump branches, rather than differentiating a discrete graph.
/// All errors are paired sampling errors, conditional on calibration and the
/// finite grid/bridge/bump. Gamma gaps are diagnostics, not derivative error bounds.
#[derive(Clone, Debug, PartialEq)]
pub struct StochasticDividendContinuousBarrierGammaRisk {
    pub price: StochasticDividendPrice,
    pub delta: f64,
    pub delta_standard_error: f64,
    pub spot: f64,
    pub spot_bumps: [f64; 3],
    pub gamma_estimates: [f64; 3],
    pub gamma_standard_errors: [f64; 3],
    /// Gamma(h/2)-Gamma(h), Gamma(h)-Gamma(2h), paired per sampling unit.
    pub bump_differences: [f64; 2],
    pub bump_difference_standard_errors: [f64; 2],
    pub payoff_evaluations: u128,
    pub risk_fingerprint: Fingerprint,
    pub method: &'static str,
}
impl StochasticDividendContinuousBarrierGammaRisk {
    #[must_use]
    pub fn gamma(&self) -> f64 {
        self.gamma_estimates[1]
    }
    #[must_use]
    pub fn standard_error(&self) -> f64 {
        self.gamma_standard_errors[1]
    }
    #[must_use]
    pub const fn uncertainty_scope(&self) -> &'static str {
        "sampling_only_fixed_calibration_grid_bridge_and_bump"
    }
}

impl StochasticDividendContinuousBarrierPlan {
    /// Evaluate (P(S+h)-P(S-h))/(2h) for h/2, h and 2h with shared shocks.
    /// Historical hit state stays fixed, while current/future endpoint and cash
    /// hit branches are re-evaluated under each Spot. Even at a branch boundary
    /// this is a finite-bump number, not a claim that an exact Delta exists.
    /// All six shifted Spots must be representable and funded; no clamping or
    /// one-sided fallback is used. Generic risk flags remain unsupported.
    pub fn evaluate_spot_bump_risk(
        &self,
        bump: SpotBump,
    ) -> Result<StochasticDividendContinuousBarrierSpotRisk, MonteCarloError> {
        self.evaluate_spot_bump_orders(bump, false)
            .map(|(delta, _)| delta)
    }

    /// Common-noise (P(S+h)-2P(S)+P(S-h))/h² at h/2, h and 2h.
    /// Reuses the same seven physical payoffs as the finite-bump Delta ladder.
    /// Initial equality and cash-jump boundary crossings are included. This
    /// finite-bump quantity does not imply a zero-bump second derivative exists.
    /// No smoothing, extrapolation, or one-sided fallback is applied.
    pub fn evaluate_gamma_bump_risk(
        &self,
        bump: SpotBump,
    ) -> Result<StochasticDividendContinuousBarrierGammaRisk, MonteCarloError> {
        self.evaluate_spot_bump_orders(bump, true)
            .map(|(_, gamma)| gamma.expect("requested second price differences"))
    }

    fn evaluate_spot_bump_orders(
        &self,
        bump: SpotBump,
        include_gamma: bool,
    ) -> Result<
        (
            StochasticDividendContinuousBarrierSpotRisk,
            Option<StochasticDividendContinuousBarrierGammaRisk>,
        ),
        MonteCarloError,
    > {
        let width = if include_gamma { GAMMA_WIDTH } else { WIDTH };
        let spot = self.inner.market.spot().get();
        let (bump, convention) = match bump {
            SpotBump::Absolute(h) => (h.get(), b"absolute".as_slice()),
            SpotBump::Relative(h) => (h.get() * spot, b"relative".as_slice()),
        };
        let bumps = [0.5 * bump, bump, 2.0 * bump];
        let mut scenarios = Vec::with_capacity(6);
        for &h in &bumps {
            if !h.is_finite()
                || h <= 0.0
                || !(2.0 * h).is_finite()
                || spot - h >= spot
                || spot + h <= spot
            {
                return Err(invalid("continuous_barrier_spot_bump").into());
            }
            for shifted in [spot - h, spot + h] {
                let shifted = PositiveF64::new(shifted, "continuous_barrier_shifted_spot")
                    .map_err(|_| invalid("continuous_barrier_shifted_spot"))?;
                scenarios.push(self.inner.lsv_spot_path(shifted)?);
            }
        }
        let executor = DeterministicExecutor::new(self.inner.policy)?;
        let dimension = self.inner.path.random_dimension();
        let (statistics, units, paths) = match self.inner.engine {
            EngineConfig::PseudoMonteCarlo(config) => {
                let count = config.independent_sampling_units().get();
                let bridge = self.inner.bridge(config.variance_reduction())?;
                let rng = Philox4x32::from_seed(config.master_seed());
                let stats = executor.try_map_reduce_statistics_vector(count, width, |p, out| {
                    let z = (0..dimension)
                        .map(|d| {
                            rng.standard_normal(RandomCoordinate::new(
                                p,
                                d,
                                RandomDomain::Valuation,
                            ))
                        })
                        .collect();
                    self.spot_bump_sample(
                        &scenarios,
                        &bumps,
                        z,
                        bridge.as_ref(),
                        config.variance_reduction().antithetic(),
                        out,
                    )
                })?;
                (stats, count, config.evaluated_paths())
            }
            EngineConfig::RandomizedQuasiMonteCarlo(config) => {
                let qmc = RqmcPlan::compile(config, dimension)?;
                let bridge = self.inner.bridge(config.variance_reduction())?;
                let count = config.points_per_scramble().get();
                let mut means = Vec::with_capacity(config.scramble_count().get() as usize);
                for scramble in 0..config.scramble_count().get() {
                    let stats =
                        executor.try_map_reduce_statistics_vector(count, width, |p, out| {
                            let z = (0..dimension)
                                .map(|d| {
                                    let u = qmc
                                        .uniform(scramble, p, d)
                                        .map_err(|_| invalid("rqmc_uniform"))?;
                                    inverse_standard_normal(u).map_err(|_| invalid("rqmc_normal"))
                                })
                                .collect::<Result<Vec<_>, _>>()?;
                            self.spot_bump_sample(
                                &scenarios,
                                &bumps,
                                z,
                                bridge.as_ref(),
                                config.variance_reduction().antithetic(),
                                out,
                            )
                        })?;
                    means.push(
                        stats
                            .iter()
                            .map(|s| s.sum().total() / count as f64)
                            .collect::<Vec<_>>(),
                    );
                }
                let scrambles = u64::from(config.scramble_count().get());
                let stats = (0..width)
                    .map(|j| {
                        let values = means.iter().map(|v| v[j]).collect::<Vec<_>>();
                        DeterministicStatistics::from_ordered_values_two_pass(&values)
                    })
                    .collect::<Vec<_>>();
                (
                    stats,
                    scrambles,
                    u128::from(count)
                        * u128::from(scrambles)
                        * if config.variance_reduction().antithetic() {
                            2
                        } else {
                            1
                        },
                )
            }
        };
        let values = statistics
            .iter()
            .map(|s| s.sum().total() / units as f64)
            .collect::<Vec<_>>();
        let errors = statistics
            .iter()
            .map(|s| {
                let variance = s
                    .moments()
                    .sample_variance()
                    .ok_or(MonteCarloError::InsufficientSamplingUnits { count: units })?;
                Ok((variance / units as f64).sqrt())
            })
            .collect::<Result<Vec<_>, MonteCarloError>>()?;
        if values.iter().chain(&errors).any(|v| !v.is_finite()) {
            return Err(invalid("continuous_barrier_spot_bump_estimator").into());
        }
        let risk_fingerprint = |method: &str| {
            let mut hash = blake3::Hasher::new();
            hash.update(method.as_bytes());
            hash.update(self.plan_fingerprint().as_bytes());
            hash.update(convention);
            for h in bumps {
                hash.update(&h.to_bits().to_le_bytes());
            }
            Fingerprint::from_bytes(*hash.finalize().as_bytes())
        };
        let delta = StochasticDividendContinuousBarrierSpotRisk {
            price: StochasticDividendPrice {
                value: values[0],
                standard_error: errors[0],
                independent_sampling_units: units,
                evaluated_paths: paths,
                plan_fingerprint: self.plan_fingerprint(),
                scheme: SCHEME,
            },
            spot,
            spot_bumps: bumps,
            delta_estimates: [values[1], values[2], values[3]],
            delta_standard_errors: [errors[1], errors[2], errors[3]],
            bump_differences: [values[4], values[5]],
            bump_difference_standard_errors: [errors[4], errors[5]],
            payoff_evaluations: 7 * paths,
            risk_fingerprint: risk_fingerprint(METHOD),
            method: METHOD,
        };
        let gamma = include_gamma.then(|| StochasticDividendContinuousBarrierGammaRisk {
            price: delta.price.clone(),
            delta: delta.delta(),
            delta_standard_error: delta.standard_error(),
            spot,
            spot_bumps: bumps,
            gamma_estimates: [values[6], values[7], values[8]],
            gamma_standard_errors: [errors[6], errors[7], errors[8]],
            bump_differences: [values[9], values[10]],
            bump_difference_standard_errors: [errors[9], errors[10]],
            payoff_evaluations: delta.payoff_evaluations,
            risk_fingerprint: risk_fingerprint(GAMMA_METHOD),
            method: GAMMA_METHOD,
        });
        Ok((delta, gamma))
    }

    #[allow(clippy::too_many_arguments)]
    fn spot_bump_sample(
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
            let (states, volatilities) = self
                .inner
                .path
                .evolve_rough_path_with_volatilities(&shocks)?;
            let center = self.payoff_from_trace(&self.inner.path, &states, &volatilities)?;
            out[0] += center;
            for (j, (pair, h)) in scenarios.as_chunks::<2>().0.iter().zip(bumps).enumerate() {
                let down = self.payoff_from_trace(&pair[0], &states, &volatilities)?;
                let up = self.payoff_from_trace(&pair[1], &states, &volatilities)?;
                out[j + 1] += (up - down) / (2.0 * h);
                if out.len() == GAMMA_WIDTH {
                    // Subtract the center before adding; fixed rebates then
                    // cancel exactly. Sequential division avoids h² under/overflow.
                    out[j + 6] += ((up - center) + (down - center)) / h / h;
                }
            }
        }
        if antithetic {
            for value in out.iter_mut() {
                *value *= 0.5;
            }
        }
        out[4] = out[1] - out[2];
        out[5] = out[2] - out[3];
        if out.len() == GAMMA_WIDTH {
            out[9] = out[6] - out[7];
            out[10] = out[7] - out[8];
        }
        if out.iter().any(|v| !v.is_finite()) {
            return Err(invalid("continuous_barrier_spot_bump_sample").into());
        }
        Ok(())
    }
}
