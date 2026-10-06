//! Fixed-model physical Spot Gamma by explicitly sized central CRN bumps.
use super::*;
use crate::engine::risk::gamma_bump::{BumpGrid, GammaSampling, RoughGammaBump};

pub type RoughVolatilityGamma = RoughGammaBump<RoughVolatilityPrice>;
impl RoughVolatilityPricingPlan {
    /// Central physical-Spot second differences at h and h/2, where h is an
    /// absolute currency amount. Model, carry and full cash schedule are fixed.
    /// This is a finite-bump estimator, NOT pathwise second-order AAD. Hard
    /// discontinuous payoffs require explicit smoothing; SE excludes bump bias.
    pub fn evaluate_gamma_bump(
        &self,
        spot_bump: f64,
    ) -> Result<RoughVolatilityGamma, MonteCarloError> {
        if !self.risk_supported {
            return Err(MonteCarloError::UnsupportedRiskForModel {
                model: "rough Gamma requires a pathwise payoff or explicit smoothing",
            });
        }
        let bump = BumpGrid::new(self.initial_forward, self.risky_spot, spot_bump)?;
        let run = GammaSampling {
            engine: self.engine,
            policy: self.policy,
            times: self.time_nodes(),
            brownian_blocks: self.path.brownian_block_count(),
            dimension: self.random_dimension(),
        }
        .evaluate(
            bump,
            |seed, i| self.path.pseudo_shocks(seed, i, RandomDomain::Valuation),
            |z, _| {
                let mut prices = [0.0; 5];
                for (i, shift) in bump.shifts().into_iter().enumerate() {
                    // Recompile-equivalent residual-equity map: the pure model
                    // evolves with initial coordinate S+shift (important if beta<1).
                    let path = self.path.evolve_path(bump.spot + shift, z)?;
                    let scale_ratio = (1.0 + shift / bump.residual) / (1.0 + shift / bump.spot);
                    let spots = self
                        .observations
                        .iter()
                        .zip(&path.forwards)
                        .map(|(o, &f)| {
                            let post = (o.scale * scale_ratio) * f + o.reserve;
                            let pre = o.event.map(|e| (post + e.cash) / (1.0 - e.beta));
                            if !post.is_finite() || pre.is_some_and(|x| !x.is_finite()) {
                                return Err(invalid("rough_gamma_observation"));
                            }
                            Ok((post, pre))
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    prices[i] = self.base.hybrid_spot_payoff(self.time_nodes(), &spots)?;
                }
                Ok(prices)
            },
        )?;
        let (value, standard_error) = run.moments(0)?;
        run.finish(
            RoughVolatilityPrice {
                value,
                standard_error,
                independent_sampling_units: run.units,
                evaluated_paths: run.paths,
                plan_fingerprint: self.fingerprint,
                scheme: self.path.scheme(),
                calibration_method: None,
                calibration_seed: None,
                cash_dividend_model: Some(HULL_WHITE_CASH_DIVIDEND_MODEL),
            },
            bump,
            self.fingerprint,
            "fixed_model_fixed_curves_and_cash_dividends",
        )
    }
}
