//! Deterministic-rate pricing adapter for the additional rough model families.
//! Models carry their own volatility levels; the BlackScholes request is an
//! explicit carrier for market, payoff, dates and sampling settings only.

mod gamma;
pub use gamma::RoughVolatilityGamma;

mod heston_parameter;
pub use heston_parameter::HestonMcParameterRisk;

mod delta;
pub use delta::RoughVolatilityDelta;

use crate::core::DayCountConvention;
use crate::engine::processes::rough_volatility::RoughVolatilityPathPlan;
use crate::hull_white::HullWhitePrice;
use crate::mc::{
    BrownianBridgePlan, DeterministicExecutor, DeterministicStatistics, EngineConfig,
    ExecutionPolicy, LocalVolTimeGrid, RandomDomain, RqmcPlan, VarianceReduction,
    inverse_standard_normal,
};
use crate::models::hull_white_dividends::{HULL_WHITE_CASH_DIVIDEND_MODEL, HullWhiteDividendPlan};
use crate::models::rough_volatility::invalid;
use crate::models::{HullWhite1Factor, HullWhiteError, ModelSpec, RoughVolatilityModel};
use crate::{Fingerprint, MonteCarloError, PricingRequest, SimulationPlan};

/// Alias of the existing price/SE metadata contract. No stochastic-rate model
/// is implied by the underlying result type's historical name.
pub type RoughVolatilityPrice = HullWhitePrice;

#[derive(Clone, Copy, Debug)]
struct DividendJump {
    cash: f64,
    beta: f64,
}

/// Affine physical observation = scale * normalized funded forward + reserve.
/// The jump permits reconstructing the pre-dividend observation from the post value.
#[derive(Clone, Copy, Debug)]
struct PhysicalObservation {
    scale: f64,
    reserve: f64,
    event: Option<DividendJump>,
}

#[derive(Clone, Debug)]
pub struct RoughVolatilityPricingPlan {
    base: SimulationPlan,
    path: RoughVolatilityPathPlan,
    observations: Vec<PhysicalObservation>,
    engine: EngineConfig,
    policy: ExecutionPolicy,
    initial_forward: f64,
    risky_spot: f64,
    fingerprint: Fingerprint,
    dividends: HullWhiteDividendPlan,
    risk_supported: bool,
}
impl RoughVolatilityPricingPlan {
    /// Additive extension API. A price-only BlackScholes request supplies the
    /// contractual payoff, deterministic market curves and engine. The supplied
    /// rough model is authoritative, INCLUDING its variance level. The request's
    /// BlackScholes sigma is not a second rough-model parameter. Stable JSON model
    /// tags are unchanged; this method never silently replaces the stable engine.
    ///
    /// Rough SABR beta<1 applies to the normalized funded-forward coordinate,
    /// not physical Spot. beta=0 is a normal approximation which can be negative.
    pub fn compile(
        request: &PricingRequest,
        model: RoughVolatilityModel,
        maximum_step: f64,
        policy: ExecutionPolicy,
    ) -> Result<Self, MonteCarloError> {
        if !matches!(request.model(), ModelSpec::BlackScholes(_)) {
            return Err(HullWhiteError::Unsupported {
                feature: "rough extension requires a BlackScholes request carrier",
            }
            .into());
        }
        let risk = request.risk();
        if risk.delta() || risk.gamma().is_some() || risk.vega() || risk.vega_kt().is_some() {
            return Err(HullWhiteError::Unsupported {
                feature: "new rough families are price-only; no implicit AAD or market-vega contract",
            }.into());
        }
        let expiry = DayCountConvention::Act365F
            .year_fraction(request.valuation_date(), request.product().expiry());
        if !expiry.is_finite() || expiry <= 0.0 || !maximum_step.is_finite() || maximum_step <= 0.0
        {
            return Err(invalid("rough_positive_horizon_and_step").into());
        }
        let base = SimulationPlan::compile_hybrid_base(request, policy)?;
        let market = request.market().equity().forward();
        let mut events = base.hybrid_observation_times().to_vec();
        events.push(0.0);
        events.push(expiry);
        let dividends = market.discrete_dividends().map_or(&[][..], |d| d.events());
        events.extend(
            dividends
                .iter()
                .filter(|e| e.ex_time() <= expiry)
                .map(|e| e.ex_time()),
        );
        events.sort_by(f64::total_cmp);
        events.dedup_by(|a, b| a.to_bits() == b.to_bits());
        let limit = RoughVolatilityPathPlan::maximum_time_steps(&model);
        // Bound work BEFORE the shared grid constructor allocates all nodes.
        let estimated_steps: f64 = events
            .windows(2)
            .map(|w| ((w[1] - w[0]) / maximum_step).ceil())
            .sum();
        if !estimated_steps.is_finite() || estimated_steps > limit as f64 {
            return Err(invalid("rough_time_grid_resource_limit").into());
        }
        let grid = LocalVolTimeGrid::compile(events, maximum_step)
            .map_err(|e| MonteCarloError::HullWhite(e.into()))?;
        let path = RoughVolatilityPathPlan::compile(model, grid.nodes().to_vec())?;
        let rates = HullWhite1Factor::new(0.0, vec![0.0], vec![0.0])?;
        let dividend_plan = HullWhiteDividendPlan::new(&rates, market, grid.nodes())?;
        let observations = dividend_plan
            .nodes()
            .iter()
            .map(|node| {
                let event = dividends
                    .iter()
                    .find(|event| event.ex_time() == node.time())
                    .map(|event| DividendJump {
                        cash: event.fixed_cash(),
                        beta: event.beta(),
                    });
                Ok(PhysicalObservation {
                    scale: node.scale(),
                    reserve: node.reserve(0.0)?.0,
                    event,
                })
            })
            .collect::<Result<Vec<_>, HullWhiteError>>()?;
        let mut hash = blake3::Hasher::new();
        hash.update(b"rough-volatility-pricing/v1\0");
        hash.update(base.plan_fingerprint().as_bytes());
        hash.update(path.plan_fingerprint().as_bytes());
        for &PhysicalObservation {
            scale,
            reserve,
            event,
        } in &observations
        {
            hash.update(&scale.to_le_bytes());
            hash.update(&reserve.to_le_bytes());
            hash.update(&[u8::from(event.is_some())]);
            if let Some(DividendJump { cash, beta }) = event {
                hash.update(&cash.to_le_bytes());
                hash.update(&beta.to_le_bytes());
            }
        }
        let initial_forward = market.spot().get();
        hash.update(&initial_forward.to_le_bytes());
        let fingerprint = Fingerprint::from_bytes(*hash.finalize().as_bytes());
        Ok(Self {
            base,
            path,
            observations,
            engine: request.engine(),
            policy,
            initial_forward,
            risky_spot: dividend_plan.risky_spot(),
            dividends: dividend_plan,
            risk_supported: request.product().supports_pathwise_risk()
                || risk.payoff_smoothing().is_some(),
            fingerprint,
        })
    }
    #[must_use]
    pub fn path_plan(&self) -> &RoughVolatilityPathPlan {
        &self.path
    }
    #[must_use]
    pub fn plan_fingerprint(&self) -> Fingerprint {
        self.fingerprint
    }
    #[must_use]
    pub fn time_nodes(&self) -> &[f64] {
        self.path.time_nodes()
    }
    #[must_use]
    pub fn random_dimension(&self) -> u32 {
        self.path.random_dimension()
    }
    #[must_use]
    pub const fn risky_spot(&self) -> f64 {
        self.risky_spot
    }

    pub fn evaluate(&self) -> Result<RoughVolatilityPrice, MonteCarloError> {
        let executor = DeterministicExecutor::new(self.policy)?;
        let (statistics, units, paths) = match self.engine {
            EngineConfig::PseudoMonteCarlo(config) => {
                let count = config.independent_sampling_units().get();
                let bridge = self.bridge(config.variance_reduction())?;
                let statistics = executor.try_map_reduce_statistics(count, |p| {
                    let z =
                        self.path
                            .pseudo_shocks(config.master_seed(), p, RandomDomain::Valuation);
                    self.sample(z, bridge.as_ref(), config.variance_reduction().antithetic())
                })?;
                (statistics, count, config.evaluated_paths())
            }
            EngineConfig::RandomizedQuasiMonteCarlo(config) => {
                let dimension = self.path.random_dimension();
                let qmc = RqmcPlan::compile(config, dimension)?;
                let bridge = self.bridge(config.variance_reduction())?;
                let count = config.points_per_scramble().get();
                let mut means = Vec::with_capacity(config.scramble_count().get() as usize);
                for scramble in 0..config.scramble_count().get() {
                    let statistics = executor.try_map_reduce_statistics(count, |p| {
                        let z = (0..dimension)
                            .map(|d| {
                                let u = qmc
                                    .uniform(scramble, p, d)
                                    .map_err(|_| invalid("rough_rqmc_uniform"))?;
                                inverse_standard_normal(u).map_err(|_| invalid("rough_rqmc_normal"))
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        self.sample(z, bridge.as_ref(), config.variance_reduction().antithetic())
                    })?;
                    means.push(statistics.sum().total() / count as f64);
                }
                let scrambles = u64::from(config.scramble_count().get());
                (
                    DeterministicStatistics::from_ordered_values_two_pass(&means),
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
        let variance = statistics
            .moments()
            .sample_variance()
            .ok_or(MonteCarloError::InsufficientSamplingUnits { count: units })?;
        let value = statistics.sum().total() / units as f64;
        let standard_error = (variance / units as f64).sqrt();
        if !value.is_finite() || !standard_error.is_finite() {
            return Err(invalid("rough_price_estimator").into());
        }
        Ok(RoughVolatilityPrice {
            value,
            standard_error,
            independent_sampling_units: units,
            evaluated_paths: paths,
            plan_fingerprint: self.fingerprint,
            scheme: self.path.scheme(),
            calibration_method: None,
            calibration_seed: None,
            cash_dividend_model: Some(HULL_WHITE_CASH_DIVIDEND_MODEL),
        })
    }
    fn bridge(&self, vr: VarianceReduction) -> Result<Option<BrownianBridgePlan>, MonteCarloError> {
        if !vr.brownian_bridge() {
            return Ok(None);
        }
        BrownianBridgePlan::compile(self.path.time_nodes().to_vec(), 1)
            .map(Some)
            .map_err(|e| MonteCarloError::LocalVol(e.into()))
    }
    fn sample(
        &self,
        mut normals: Vec<f64>,
        bridge: Option<&BrownianBridgePlan>,
        antithetic: bool,
    ) -> Result<f64, MonteCarloError> {
        let n = self.time_nodes().len() - 1;
        if let Some(bridge) = bridge {
            // Do not reinterpret hybrid near residuals or fOU level coordinates
            // as Brownian increments. Only genuine Brownian blocks are bridged.
            for block in normals[..self.path.brownian_block_count() * n].chunks_exact_mut(n) {
                let transformed = bridge
                    .apply_one_factor(block)
                    .map_err(|e| MonteCarloError::LocalVol(e.into()))?;
                block.copy_from_slice(&transformed);
            }
        }
        let mut average = 0.0;
        let signs = if antithetic {
            &[1.0, -1.0][..]
        } else {
            &[1.0][..]
        };
        for &sign in signs {
            let z = normals.iter().map(|value| sign * value).collect::<Vec<_>>();
            let path = self.path.evolve_path(self.initial_forward, &z)?;
            let spots = self
                .observations
                .iter()
                .zip(&path.forwards)
                .map(
                    |(
                        &PhysicalObservation {
                            scale,
                            reserve,
                            event,
                        },
                        &f,
                    )| {
                        let post = scale * f + reserve;
                        let pre =
                            event.map(|DividendJump { cash, beta }| (post + cash) / (1.0 - beta));
                        if !post.is_finite() || pre.is_some_and(|s| !s.is_finite()) {
                            return Err(invalid("rough_observation_overflow"));
                        }
                        Ok((post, pre))
                    },
                )
                .collect::<Result<Vec<_>, _>>()?;
            // The shared graph discounts to PAYMENT date already.
            average +=
                self.base.hybrid_spot_payoff(self.time_nodes(), &spots)? / signs.len() as f64;
        }
        Ok(average)
    }
}
