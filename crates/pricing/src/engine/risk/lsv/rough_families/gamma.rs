//! Finite-bump physical Gamma under the parent's two explicit Spot conventions.
use super::*;
use crate::engine::risk::gamma_bump::{BumpGrid, GammaSampling, RoughGammaBump};

pub type RoughFamilyLsvGamma = RoughGammaBump<LsvPrice>;
impl RoughFamilyLsvPricingPlan {
    /// Physical Spot Gamma with leverage values, axes and reference anchor fixed.
    /// Returns finite second differences at h and h/2, NOT a branchwise Hessian.
    pub fn evaluate_frozen_leverage_gamma_bump(
        &self,
        spot_bump: f64,
    ) -> Result<RoughFamilyLsvGamma, MonteCarloError> {
        self.gamma_bump(spot_bump, RoughFamilyLsvDeltaConvention::FrozenLeverage)
    }
    /// Physical Spot Gamma with relative Local Variance nodes/axes fixed.
    /// Discrete scale equivariance eliminates extra particle calibrations. This
    /// is not a sticky-absolute-strike market-IV or stochastic-parameter risk.
    pub fn evaluate_sticky_moneyness_gamma_bump(
        &self,
        spot_bump: f64,
    ) -> Result<RoughFamilyLsvGamma, MonteCarloError> {
        self.gamma_bump(
            spot_bump,
            RoughFamilyLsvDeltaConvention::StickyRelativeLocalVariance,
        )
    }
    fn gamma_bump(
        &self,
        spot_bump: f64,
        convention: RoughFamilyLsvDeltaConvention,
    ) -> Result<RoughFamilyLsvGamma, MonteCarloError> {
        let core = &self.core;
        if !core.risk_supported {
            return Err(MonteCarloError::UnsupportedRiskForModel {
                model: "rough LSV Gamma requires a pathwise payoff or explicit smoothing",
            });
        }
        let runtime =
            core.base
                .local_volatility
                .as_ref()
                .ok_or(MonteCarloError::UnsupportedModel {
                    model: "rough-family LSV target",
                })?;
        let spot = core.base.spot;
        let reserve = runtime
            .dividends
            .as_ref()
            .map_or(0.0, |d| d.initial_reserve());
        let bump = BumpGrid::new(spot, spot - reserve, spot_bump)?;
        let run = GammaSampling {
            engine: core.engine,
            policy: core.policy,
            times: core.path_plan.times(),
            brownian_blocks: core.calibration.brownian_block_count(),
            dimension: core.path_plan.random_dimension(),
        }
        .evaluate(
            bump,
            |seed, i| {
                core.path_plan
                    .pseudo_shocks(seed, i, RandomDomain::Valuation)
            },
            |z, path| {
                let mut prices = [0.0; 5];
                let base = core.path_plan.evolve_path(spot, z)?;
                if convention == RoughFamilyLsvDeltaConvention::FrozenLeverage {
                    core.path_plan.validate_spot_delta_path(&base)?;
                }
                for (i, shift) in bump.shifts().into_iter().enumerate() {
                    let states = if i == 0 {
                        base.states().to_vec()
                    } else {
                        match convention {
                            RoughFamilyLsvDeltaConvention::FrozenLeverage => core
                                .path_plan
                                .evolve_path(spot + shift * spot / bump.residual, z)?
                                .states()
                                .to_vec(),
                            RoughFamilyLsvDeltaConvention::StickyRelativeLocalVariance => {
                                // Reference observation map stays fixed here;
                                // multiply f by the change in residual equity.
                                base.states()
                                    .iter()
                                    .map(|&f| f * (1.0 + shift / bump.residual))
                                    .collect()
                            }
                        }
                    };
                    prices[i] = core
                        .base
                        .lsv_payoff(&states, PathIndex::new(path), false)?
                        .0;
                }
                Ok(prices)
            },
        )?;
        let (value, standard_error) = run.moments(0)?;
        run.finish(
            LsvPrice {
                value,
                standard_error,
                independent_sampling_units: run.units,
                evaluated_paths: run.paths,
                calibration_seed: core.calibration.config().seed(),
                plan_fingerprint: core.fingerprint,
                scheme: CalibratedRoughFamilyLsv::SCHEME,
            },
            bump,
            core.fingerprint,
            convention.as_str(),
        )
    }
}
