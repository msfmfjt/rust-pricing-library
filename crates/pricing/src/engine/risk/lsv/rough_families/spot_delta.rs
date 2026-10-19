//! Physical Spot derivatives under two explicitly different smile conventions.
use super::*;
use pricing_numerics::NeumaierSum;

/// What is held fixed under a physical Spot move. Curves, dividend quotes,
/// model parameters, seeds, time nodes and bandwidth are always fixed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RoughFamilyLsvDeltaConvention {
    /// Calibrated leverage and its reference funded-forward anchor stay fixed.
    /// Initial residual equity changes, not the reference escrow observation map.
    FrozenLeverage,
    /// Relative Dupire variance values and log-moneyness axes stay fixed.
    /// Recalibration is scale equivariant: leverage values do not change,
    /// but the anchor follows Spot. NOT sticky-strike market-IV Delta.
    StickyRelativeLocalVariance,
}
impl RoughFamilyLsvDeltaConvention {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::FrozenLeverage => "frozen_leverage_in_reference_f",
            Self::StickyRelativeLocalVariance => "sticky_relative_local_variance",
        }
    }
    const fn method(self) -> &'static str {
        match self {
            Self::FrozenLeverage => "rough-family-lsv-frozen-leverage-spot-vjp-v1",
            Self::StickyRelativeLocalVariance => "rough-family-lsv-sticky-target-scale-spot-v1",
        }
    }
}

/// Physical Spot derivative per one currency unit. Sampling error is
/// conditional on one calibration, not calibration noise or total error.
#[derive(Clone, Debug, PartialEq)]
pub struct RoughFamilyLsvDelta {
    pub price: LsvPrice,
    pub delta: f64,
    pub delta_standard_error: f64,
    pub convention: RoughFamilyLsvDeltaConvention,
    pub method: &'static str,
}
#[derive(Clone, Copy)]
struct DeltaInputs {
    convention: RoughFamilyLsvDeltaConvention,
    spot: f64,
    residual: f64,
}
impl RoughFamilyLsvPricingPlan {
    /// Physical Spot Delta at fixed calibrated leverage in the reference
    /// funded-forward coordinate. Initial state changes by S_ref/(S_ref-A0)
    /// per unit physical Spot, while the reference escrow observation map stays
    /// fixed. Requires pathwise payoff or explicit smoothing, not a calibration
    /// reverse trace. Does not refit to a market-IV surface.
    pub fn evaluate_frozen_leverage_delta(&self) -> Result<RoughFamilyLsvDelta, MonteCarloError> {
        self.spot_delta(RoughFamilyLsvDeltaConvention::FrozenLeverage)
    }
    /// Physical Spot Delta at fixed relative Local Variance nodes/axes.
    /// Recalibration scales every particle equally, leaving squared leverage
    /// invariant and moving its anchor. Uses the discrete homogeneity identity,
    /// not bumps. NOT recalibration at fixed absolute-strike market IV quotes.
    pub fn evaluate_sticky_moneyness_delta(&self) -> Result<RoughFamilyLsvDelta, MonteCarloError> {
        self.spot_delta(RoughFamilyLsvDeltaConvention::StickyRelativeLocalVariance)
    }
    fn delta_inputs(
        &self,
        convention: RoughFamilyLsvDeltaConvention,
    ) -> Result<DeltaInputs, MonteCarloError> {
        let spot = self.core.base.spot;
        let runtime =
            self.core
                .base
                .local_volatility
                .as_ref()
                .ok_or(MonteCarloError::UnsupportedModel {
                    model: "rough-family LSV target",
                })?;
        // Includes future cash after payoff expiry and time-zero events.
        let reserve = runtime
            .dividends
            .as_ref()
            .map_or(0.0, |d| d.initial_reserve());
        let residual = spot - reserve;
        if !residual.is_finite() || residual <= 0.0 || !(spot / residual).is_finite() {
            return Err(LsvError::InvalidInput {
                field: "rough_lsv_spot_residual",
                index: 0,
            }
            .into());
        }
        Ok(DeltaInputs {
            convention,
            spot,
            residual,
        })
    }
    fn delta_sample(
        &self,
        z: &[f64],
        path: u64,
        anti: bool,
        input: DeltaInputs,
        out: &mut [f64],
    ) -> Result<(), MonteCarloError> {
        let signs = if anti { &[1.0, -1.0][..] } else { &[1.0][..] };
        let weight = if anti { 0.5 } else { 1.0 };
        for &sign in signs {
            let shocks = z.iter().map(|z| z * sign).collect::<Vec<_>>();
            let record = self.core.path_plan.evolve_path(input.spot, &shocks)?;
            let (price, seeds) =
                self.core
                    .base
                    .lsv_payoff(record.states(), PathIndex::new(path), true)?;
            let seeds = seeds.expect("payoff reverse was requested");
            let delta = match input.convention {
                RoughFamilyLsvDeltaConvention::FrozenLeverage => {
                    self.core.path_plan.validate_spot_delta_path(&record)?;
                    record.reverse(&seeds)?.initial_forward * (input.spot / input.residual)
                }
                RoughFamilyLsvDeltaConvention::StickyRelativeLocalVariance => {
                    // Joint initial-state / anchor move: f_j(S)=S*f_j(ref)/S_ref.
                    // Fixed-cash escrow gives dP/dS=sum_j P_fj*f_j/(S_ref-A0).
                    seeds
                        .iter()
                        .zip(record.states())
                        .map(|(&a, &f)| a * (f / input.residual))
                        .collect::<NeumaierSum>()
                        .total()
                }
            };
            if !delta.is_finite() {
                return Err(LsvError::InvalidInput {
                    field: "rough_lsv_spot_delta",
                    index: 0,
                }
                .into());
            }
            out[0] += weight * price;
            out[1] += weight * delta;
        }
        Ok(())
    }
    fn spot_delta(
        &self,
        convention: RoughFamilyLsvDeltaConvention,
    ) -> Result<RoughFamilyLsvDelta, MonteCarloError> {
        if !self.core.risk_supported {
            return Err(MonteCarloError::UnsupportedRiskForModel {
                model: "rough-family LSV Spot Delta requires pathwise payoff or explicit smoothing",
            });
        }
        let input = self.delta_inputs(convention)?;
        let core = &self.core;
        let executor = DeterministicExecutor::new(core.policy)?;
        let (stats, units, paths) = match core.engine {
            EngineConfig::PseudoMonteCarlo(c) => {
                let n = c.independent_sampling_units().get();
                let bridge = core.bridge(c.variance_reduction())?;
                let stats = executor.try_map_reduce_statistics_vector(n, 2, |i, out| {
                    let z =
                        core.path_plan
                            .pseudo_shocks(c.master_seed(), i, RandomDomain::Valuation);
                    let z = core.apply_bridge(z, bridge.as_ref())?;
                    self.delta_sample(&z, i, c.variance_reduction().antithetic(), input, out)
                })?;
                (stats, n, c.evaluated_paths())
            }
            EngineConfig::RandomizedQuasiMonteCarlo(c) => {
                let dimension = core.path_plan.random_dimension();
                let qmc = RqmcPlan::compile(c, dimension)?;
                let bridge = core.bridge(c.variance_reduction())?;
                let n = c.points_per_scramble().get();
                let count = c.scramble_count().get();
                let mut means = [
                    Vec::with_capacity(count as usize),
                    Vec::with_capacity(count as usize),
                ];
                for scramble in 0..count {
                    let stats = executor.try_map_reduce_statistics_vector(n, 2, |i, out| {
                        let z = (0..dimension)
                            .map(|d| {
                                let u = qmc.uniform(scramble, i, d).map_err(|_| {
                                    LsvError::InvalidInput {
                                        field: "rqmc_coordinate",
                                        index: d as usize,
                                    }
                                })?;
                                inverse_standard_normal(u).map_err(|_| LsvError::InvalidInput {
                                    field: "rqmc_normal",
                                    index: d as usize,
                                })
                            })
                            .collect::<Result<Vec<_>, LsvError>>()?;
                        let z = core.apply_bridge(z, bridge.as_ref())?;
                        self.delta_sample(
                            &z,
                            u64::from(scramble) * n + i,
                            c.variance_reduction().antithetic(),
                            input,
                            out,
                        )
                    })?;
                    for j in 0..2 {
                        means[j].push(stats[j].sum().total() / n as f64);
                    }
                }
                (
                    means
                        .iter()
                        .map(|v| DeterministicStatistics::from_ordered_values_two_pass(v))
                        .collect(),
                    u64::from(count),
                    n as u128
                        * u128::from(count)
                        * if c.variance_reduction().antithetic() {
                            2
                        } else {
                            1
                        },
                )
            }
        };
        let delta = stats[1].sum().total() / units as f64;
        if !delta.is_finite() {
            return Err(LsvError::InvalidInput {
                field: "rough_lsv_spot_delta_estimator",
                index: 0,
            }
            .into());
        }
        Ok(RoughFamilyLsvDelta {
            price: LsvPrice {
                value: stats[0].sum().total() / units as f64,
                standard_error: standard_error(stats[0], units)?,
                independent_sampling_units: units,
                evaluated_paths: paths,
                calibration_seed: core.calibration.config().seed(),
                plan_fingerprint: core.fingerprint,
                scheme: CalibratedRoughFamilyLsv::SCHEME,
            },
            delta,
            delta_standard_error: standard_error(stats[1], units)?,
            convention,
            method: convention.method(),
        })
    }
}
