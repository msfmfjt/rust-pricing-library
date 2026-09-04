use pricing_core::{Date, DayCountConvention, SchemaVersion, UnderlyingId};
use pricing_aad::{AadTilePolicy, CheckpointPolicy};
use pricing_market::{CurveRegion, DiscountCurve};
use pricing_mc::{
    DeterministicExecutor, DeterministicStatistics, EngineConfig, ExecutionPolicy, Philox4x32,
    PseudoMcConfig, RandomCoordinate, RandomDomain,
};
use pricing_models::ModelSpec;
use pricing_product::{CompiledPayoff, GraphFingerprint, GraphLimitPolicy, ProductSpec};
use pricing_risk::{GammaConfig, SpotBump};

use crate::{
    Diagnostics, Estimate, EstimatorKind, MonteCarloError, PricingRequest, PricingResult,
    PricingWarning, ReplayMetadata, ResultBuildError, RiskEstimate, RiskReport, RiskUnit,
    fingerprint_request,
};

const NORMAL_95: f64 = 1.959_963_984_540_054;
const PRICE: usize = 0;
const DELTA: usize = 1;
const VEGA: usize = 2;
const GAMMA: usize = 3;
const PATHWISE_COMPONENTS: usize = 4;

/// Immutable one-expiry plan for the European Black-Scholes pseudo-MC slice.
#[derive(Clone, Debug)]
pub struct SimulationPlan {
    valuation_date: Date,
    expiry: Date,
    underlying: UnderlyingId,
    time: f64,
    forward: f64,
    spot: f64,
    discount: f64,
    volatility: f64,
    total_variance: f64,
    standard_deviation: f64,
    payoff: CompiledPayoff,
    engine: PseudoMcConfig,
    execution_policy: ExecutionPolicy,
    aad_tile_policy: AadTilePolicy,
    checkpoint_policy: CheckpointPolicy,
    request_delta: bool,
    request_gamma: Option<GammaConfig>,
    request_vega: bool,
    request_fingerprint: [u8; 32],
    discount_region: CurveRegion,
    dividend_region: CurveRegion,
}

impl SimulationPlan {
    pub fn compile(
        request: &PricingRequest,
        execution_policy: ExecutionPolicy,
    ) -> Result<Self, MonteCarloError> {
        let engine = match request.engine() {
            EngineConfig::PseudoMonteCarlo(engine) => engine,
            EngineConfig::RandomizedQuasiMonteCarlo(_) => {
                return Err(MonteCarloError::UnsupportedEngine);
            }
        };
        let ProductSpec::EuropeanVanilla(product) = request.product();
        let ModelSpec::BlackScholes(model) = request.model();
        let time =
            DayCountConvention::Act365F.year_fraction(request.valuation_date(), product.expiry());
        let market_forward = request.market().equity().forward();
        let forward_evaluation = market_forward.evaluate(time)?;
        let discount = market_forward.discount_curve().discount(time)?;
        let spot = market_forward.spot().get();
        let volatility = model.volatility().get();
        let total_variance = volatility * volatility * time;
        if !total_variance.is_finite() {
            return Err(MonteCarloError::NonFiniteTotalVariance {
                bits: total_variance.to_bits(),
            });
        }
        if total_variance > 0.0 && engine.independent_sampling_units().get() < 2 {
            return Err(MonteCarloError::InsufficientSamplingUnits {
                count: engine.independent_sampling_units().get(),
            });
        }
        let payoff = product.source_graph()?.compile(GraphLimitPolicy::DEFAULT)?;
        let request_fingerprint = *fingerprint_request(request)?.as_bytes();
        let aad_tile_policy = AadTilePolicy::resolve(
            execution_policy.reduction_block_size().get(),
            request.risk().aad_tile_capacity(),
        )?;
        let checkpoint_policy = CheckpointPolicy::resolve(request.risk().checkpoint_interval());
        if let Some(gamma) = request.risk().gamma() {
            let bump = resolve_spot_bump(gamma, spot);
            if !bump.is_finite() || bump >= spot {
                return Err(MonteCarloError::InvalidGammaBump {
                    spot_bits: spot.to_bits(),
                    bump_bits: bump.to_bits(),
                });
            }
        }
        Ok(Self {
            valuation_date: request.valuation_date(),
            expiry: product.expiry(),
            underlying: product.underlying(),
            time,
            forward: forward_evaluation.forward,
            spot,
            discount,
            volatility,
            total_variance,
            standard_deviation: total_variance.sqrt(),
            payoff,
            engine,
            execution_policy,
            aad_tile_policy,
            checkpoint_policy,
            request_delta: request.risk().delta(),
            request_gamma: request.risk().gamma(),
            request_vega: request.risk().vega(),
            request_fingerprint,
            discount_region: forward_evaluation.discount_region,
            dividend_region: forward_evaluation.dividend_region,
        })
    }

    #[must_use]
    pub const fn valuation_date(&self) -> Date {
        self.valuation_date
    }

    #[must_use]
    pub const fn expiry(&self) -> Date {
        self.expiry
    }

    #[must_use]
    pub const fn time(&self) -> f64 {
        self.time
    }

    #[must_use]
    pub const fn forward(&self) -> f64 {
        self.forward
    }

    #[must_use]
    pub const fn discount(&self) -> f64 {
        self.discount
    }

    #[must_use]
    pub const fn total_variance(&self) -> f64 {
        self.total_variance
    }

    #[must_use]
    pub const fn payoff_fingerprint(&self) -> GraphFingerprint {
        self.payoff.tape_fingerprint()
    }

    #[must_use]
    pub const fn execution_policy(&self) -> ExecutionPolicy {
        self.execution_policy
    }

    pub fn execute(&self) -> Result<MonteCarloPrice, MonteCarloError> {
        let executor = DeterministicExecutor::new(self.execution_policy)?;
        let generator = Philox4x32::from_seed(self.engine.master_seed());
        let antithetic = self.engine.variance_reduction().antithetic();
        let statistics = if self.risk_enabled() {
            executor.try_map_reduce_statistics_array_tiled(
                self.engine.independent_sampling_units().get(),
                self.aad_tile_policy.resolved_capacity(),
                |sampling_unit| {
                    let normal = self.normal(&generator, sampling_unit);
                    let primary = self.pathwise_values(normal)?;
                    if antithetic {
                        let mate = self.pathwise_values(-normal)?;
                        Ok(std::array::from_fn(|component| {
                            (primary[component] + mate[component]) * 0.5
                        }))
                    } else {
                        Ok(primary)
                    }
                },
            )?
        } else {
            let price = executor.try_map_reduce_statistics(
                self.engine.independent_sampling_units().get(),
                |sampling_unit| {
                    let normal = self.normal(&generator, sampling_unit);
                    let primary = self.discounted_payoff(normal)?;
                    if antithetic {
                        let mate = self.discounted_payoff(-normal)?;
                        Ok((primary + mate) * 0.5)
                    } else {
                        Ok(primary)
                    }
                },
            )?;
            let mut values = [DeterministicStatistics::default(); PATHWISE_COMPONENTS];
            values[PRICE] = price;
            values
        };

        let independent_units = self.engine.independent_sampling_units().get();
        let price = statistics[PRICE].sum().total() / independent_units as f64;
        let sampling_variance = if self.total_variance == 0.0 {
            0.0
        } else {
            statistics[PRICE].moments().sample_variance().ok_or(
                MonteCarloError::InsufficientSamplingUnits {
                    count: independent_units,
                },
            )?
        };
        let estimator_variance = sampling_variance / independent_units as f64;
        let estimate = estimate_from_statistics(statistics[PRICE], independent_units, 1.0)?;
        debug_assert_eq!(estimate.value().get().to_bits(), price.to_bits());
        let risks = self.build_risk_report(&statistics, independent_units)?;
        let warnings = extrapolation_warnings(self.discount_region, self.dividend_region);
        let pricing_result = PricingResult {
            value: estimate,
            risks,
            diagnostics: Diagnostics::new(warnings),
            replay: ReplayMetadata::new(
                SchemaVersion::CURRENT,
                self.request_fingerprint,
                crate::version(),
                format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH),
            ),
        };
        Ok(MonteCarloPrice {
            pricing_result,
            sampling_variance,
            estimator_variance,
            independent_sampling_units: independent_units,
            evaluated_paths: self.engine.evaluated_paths(),
            diagnostics: MonteCarloDiagnostics {
                master_seed: self.engine.master_seed(),
                policy_version: self.execution_policy.version(),
                worker_threads: self.execution_policy.worker_threads().get(),
                reduction_block_size: self.execution_policy.reduction_block_size().get(),
                aad_tile_policy_version: self.aad_tile_policy.version(),
                aad_tile_capacity: self.aad_tile_policy.resolved_capacity().get(),
                checkpoint_policy_version: self.checkpoint_policy.version(),
                checkpoint_interval: self.checkpoint_policy.resolved_interval().get(),
                antithetic,
                discount_region: self.discount_region,
                dividend_region: self.dividend_region,
                payoff_fingerprint: self.payoff.tape_fingerprint(),
            },
        })
    }

    fn risk_enabled(&self) -> bool {
        self.request_delta || self.request_gamma.is_some() || self.request_vega
    }

    fn normal(&self, generator: &Philox4x32, sampling_unit: u64) -> f64 {
        if self.total_variance == 0.0 {
            0.0
        } else {
            generator.standard_normal(RandomCoordinate::new(
                sampling_unit,
                0,
                RandomDomain::Valuation,
            ))
        }
    }

    fn discounted_payoff(&self, normal: f64) -> Result<f64, pricing_product::GraphError> {
        let log_return = -0.5 * self.total_variance + self.standard_deviation * normal;
        let terminal = self.forward * log_return.exp();
        let outputs = self.payoff.evaluate(|underlying, date| {
            (underlying == self.underlying && date == self.expiry).then_some(terminal)
        })?;
        Ok(self.discount
            * outputs
                .first()
                .copied()
                .ok_or(pricing_product::GraphError::NoOutputs)?)
    }

    fn pathwise_values(
        &self,
        normal: f64,
    ) -> Result<[f64; PATHWISE_COMPONENTS], pricing_product::GraphError> {
        let (price, delta, vega) = self.pathwise_aad(normal, self.spot)?;
        let gamma = if let Some(gamma) = self.request_gamma {
            let bump = resolve_spot_bump(gamma, self.spot);
            let delta_down = self.pathwise_aad(normal, self.spot - bump)?.1;
            let delta_up = self.pathwise_aad(normal, self.spot + bump)?.1;
            (delta_up - delta_down) / (2.0 * bump)
        } else {
            0.0
        };
        Ok([price, delta, vega, gamma])
    }

    fn pathwise_aad(
        &self,
        normal: f64,
        spot: f64,
    ) -> Result<(f64, f64, f64), pricing_product::GraphError> {
        let log_return = -0.5 * self.total_variance + self.standard_deviation * normal;
        let bumped_forward = self.forward * (spot / self.spot);
        let terminal = bumped_forward * log_return.exp();
        let payoff = self
            .payoff
            .evaluate_single_with_terminal_adjoint(|underlying, date| {
                (underlying == self.underlying && date == self.expiry).then_some(terminal)
            })?;
        let terminal_adjoint = payoff
            .terminal_adjoints
            .iter()
            .filter(|adjoint| {
                adjoint.underlying == self.underlying && adjoint.observation_date == self.expiry
            })
            .fold(0.0, |total, adjoint| total + adjoint.value);
        let price = self.discount * payoff.value;
        let delta = self.discount * terminal_adjoint * terminal / spot;
        let terminal_vega = terminal * (-self.volatility * self.time + self.time.sqrt() * normal);
        let vega = self.discount * terminal_adjoint * terminal_vega;
        Ok((price, delta, vega))
    }

    fn build_risk_report(
        &self,
        statistics: &[DeterministicStatistics; PATHWISE_COMPONENTS],
        independent_units: u64,
    ) -> Result<RiskReport, MonteCarloError> {
        let delta = self
            .request_delta
            .then(|| {
                risk_estimate(
                    statistics[DELTA],
                    independent_units,
                    self.spot * 0.01,
                    RiskUnit::DeltaRaw,
                    RiskUnit::DeltaOnePercentSpot,
                )
            })
            .transpose()?;
        let gamma = self
            .request_gamma
            .map(|_| {
                risk_estimate(
                    statistics[GAMMA],
                    independent_units,
                    (self.spot * 0.01).powi(2),
                    RiskUnit::GammaRaw,
                    RiskUnit::GammaOnePercentSpotSquared,
                )
            })
            .transpose()?;
        let vega = self
            .request_vega
            .then(|| {
                risk_estimate(
                    statistics[VEGA],
                    independent_units,
                    0.01,
                    RiskUnit::VegaRaw,
                    RiskUnit::VegaOneVolPoint,
                )
            })
            .transpose()?;
        Ok(RiskReport { delta, gamma, vega })
    }
}

fn resolve_spot_bump(gamma: GammaConfig, spot: f64) -> f64 {
    match gamma.bump() {
        SpotBump::Absolute(value) => value.get(),
        SpotBump::Relative(value) => spot * value.get(),
    }
}

fn estimate_from_statistics(
    statistics: DeterministicStatistics,
    independent_units: u64,
    scale: f64,
) -> Result<Estimate, ResultBuildError> {
    let inverse_count = 1.0 / independent_units as f64;
    let value = statistics.sum().total() * inverse_count * scale;
    let sampling_variance = statistics.moments().sample_variance().unwrap_or(0.0);
    let standard_error = (sampling_variance * inverse_count).sqrt() * scale;
    let half_width = NORMAL_95 * standard_error;
    Estimate::new(
        value,
        standard_error,
        value - half_width,
        value + half_width,
        EstimatorKind::PseudoMonteCarlo,
        independent_units,
    )
}

fn risk_estimate(
    statistics: DeterministicStatistics,
    independent_units: u64,
    market_scale: f64,
    raw_unit: RiskUnit,
    market_scaled_unit: RiskUnit,
) -> Result<RiskEstimate, ResultBuildError> {
    Ok(RiskEstimate::new(
        estimate_from_statistics(statistics, independent_units, 1.0)?,
        estimate_from_statistics(statistics, independent_units, market_scale)?,
        raw_unit,
        market_scaled_unit,
    ))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MonteCarloDiagnostics {
    pub master_seed: u64,
    pub policy_version: u32,
    pub worker_threads: u32,
    pub reduction_block_size: u64,
    pub aad_tile_policy_version: u32,
    pub aad_tile_capacity: u32,
    pub checkpoint_policy_version: u32,
    pub checkpoint_interval: u32,
    pub antithetic: bool,
    pub discount_region: CurveRegion,
    pub dividend_region: CurveRegion,
    pub payoff_fingerprint: GraphFingerprint,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MonteCarloPrice {
    pub pricing_result: PricingResult,
    pub sampling_variance: f64,
    pub estimator_variance: f64,
    pub independent_sampling_units: u64,
    pub evaluated_paths: u128,
    pub diagnostics: MonteCarloDiagnostics,
}

pub fn price_pseudo_monte_carlo(
    request: &PricingRequest,
    execution_policy: ExecutionPolicy,
) -> Result<MonteCarloPrice, MonteCarloError> {
    SimulationPlan::compile(request, execution_policy)?.execute()
}

fn extrapolation_warnings(
    discount_region: CurveRegion,
    dividend_region: CurveRegion,
) -> Vec<PricingWarning> {
    let mut warnings = Vec::new();
    if discount_region.is_extrapolated() {
        warnings.push(PricingWarning::new(
            "discount_curve_extrapolation",
            "expiry uses flat-forward discount-curve extrapolation",
        ));
    }
    if dividend_region.is_extrapolated() {
        warnings.push(PricingWarning::new(
            "dividend_curve_extrapolation",
            "expiry uses flat-forward dividend-curve extrapolation",
        ));
    }
    warnings
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use pricing_core::{CurrencyId, CurveId, PositiveF64};
    use pricing_market::{EquityForward, EquityMarket, LogLinearDiscountCurve, MarketContext};
    use pricing_mc::{PseudoMcConfig, VarianceReduction};
    use pricing_models::BlackScholesSpec;
    use pricing_product::{EuropeanVanillaSpec, OptionSide};
    use pricing_risk::{GammaConfig, RiskRequest, SmileDynamics, SpotBump};

    use super::*;
    use crate::analytical::black_scholes_oracle;

    fn curve(id: u32, rate: f64) -> Arc<LogLinearDiscountCurve> {
        Arc::new(
            LogLinearDiscountCurve::new(CurveId::new(id), vec![0.0, 1.0], vec![1.0, (-rate).exp()])
                .expect("curve"),
        )
    }

    fn request(
        side: OptionSide,
        strike: f64,
        volatility: f64,
        sampling_units: u64,
        antithetic: bool,
    ) -> PricingRequest {
        request_with_spot_and_risk(
            side,
            strike,
            volatility,
            sampling_units,
            antithetic,
            100.0,
            RiskRequest::price_only(SmileDynamics::StickyLogMoneyness),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn request_with_spot_and_risk(
        side: OptionSide,
        strike: f64,
        volatility: f64,
        sampling_units: u64,
        antithetic: bool,
        spot: f64,
        risk: RiskRequest,
    ) -> PricingRequest {
        let underlying = UnderlyingId::new(1);
        let currency = CurrencyId::new(1);
        let product = ProductSpec::EuropeanVanilla(
            EuropeanVanillaSpec::new(
                underlying,
                currency,
                "2027-09-04".parse().expect("expiry"),
                strike,
                1.0,
                side,
            )
            .expect("product"),
        );
        let market = MarketContext::Equity(EquityMarket::new(
            currency,
            EquityForward::new(
                underlying,
                PositiveF64::new(spot, "spot").expect("spot"),
                curve(1, 0.05),
                curve(2, 0.02),
            ),
        ));
        let model = ModelSpec::BlackScholes(BlackScholesSpec::new(volatility).expect("model"));
        let engine = EngineConfig::PseudoMonteCarlo(
            PseudoMcConfig::new(
                0x0123_4567_89ab_cdef,
                sampling_units,
                VarianceReduction::new(antithetic, false),
            )
            .expect("engine"),
        );
        PricingRequest::new(
            "2026-09-04".parse().expect("valuation"),
            product,
            market,
            model,
            engine,
            risk,
        )
        .expect("request")
    }

    fn all_risks() -> RiskRequest {
        RiskRequest::new(
            true,
            Some(GammaConfig::new(
                SpotBump::relative(0.01).expect("gamma bump"),
            )),
            true,
            None,
            SmileDynamics::StickyLogMoneyness,
            Some(13),
            Some(64),
        )
        .expect("risk request")
    }

    fn policy(workers: u32) -> ExecutionPolicy {
        ExecutionPolicy::new(workers, Some(1024)).expect("policy")
    }

    #[test]
    fn mc_converges_to_the_analytical_oracle_with_reported_error() {
        let request = request(OptionSide::Call, 100.0, 0.2, 131_072, true);
        let oracle = black_scholes_oracle(&request).expect("oracle").price;
        let result = price_pseudo_monte_carlo(&request, policy(4)).expect("MC");
        let estimate = result.pricing_result.value;
        let error = (estimate.value().get() - oracle).abs();
        assert!(error <= 6.0 * estimate.standard_error().get());
        assert_eq!(result.independent_sampling_units, 131_072);
        assert_eq!(result.evaluated_paths, 262_144);
        assert_eq!(
            result.estimator_variance.sqrt().to_bits(),
            estimate.standard_error().get().to_bits()
        );
    }

    #[test]
    fn seeded_replay_is_bitwise_equal_across_worker_counts() {
        let request = request(OptionSide::Put, 105.0, 0.35, 10_003, true);
        let single = price_pseudo_monte_carlo(&request, policy(1)).expect("single");
        let parallel = price_pseudo_monte_carlo(&request, policy(4)).expect("parallel");
        assert_eq!(
            single.pricing_result.value.value().to_bits(),
            parallel.pricing_result.value.value().to_bits()
        );
        assert_eq!(
            single.pricing_result.value.standard_error().get().to_bits(),
            parallel
                .pricing_result
                .value
                .standard_error()
                .get()
                .to_bits()
        );
        assert_eq!(
            single.estimator_variance.to_bits(),
            parallel.estimator_variance.to_bits()
        );
    }

    #[test]
    fn zero_volatility_obeys_exact_forward_discounting() {
        let request = request(OptionSide::Call, 90.0, 0.0, 1, false);
        let plan = SimulationPlan::compile(&request, policy(2)).expect("plan");
        let expected = plan.discount() * (plan.forward() - 90.0);
        let result = plan.execute().expect("execution");
        assert_eq!(result.pricing_result.value.value().get(), expected);
        assert_eq!(result.sampling_variance.to_bits(), 0.0_f64.to_bits());
        assert_eq!(result.pricing_result.value.standard_error().get(), 0.0);
    }

    #[test]
    fn higher_call_strike_cannot_increase_seeded_pathwise_price() {
        let low = price_pseudo_monte_carlo(
            &request(OptionSide::Call, 90.0, 0.25, 8192, true),
            policy(3),
        )
        .expect("low strike");
        let high = price_pseudo_monte_carlo(
            &request(OptionSide::Call, 110.0, 0.25, 8192, true),
            policy(3),
        )
        .expect("high strike");
        assert!(low.pricing_result.value.value().get() >= high.pricing_result.value.value().get());
    }

    #[test]
    fn aad_delta_vega_and_bumped_aad_gamma_match_the_oracle() {
        let request = request_with_spot_and_risk(
            OptionSide::Call,
            100.0,
            0.2,
            131_072,
            true,
            100.0,
            all_risks(),
        );
        let oracle = black_scholes_oracle(&request).expect("oracle");
        let result = price_pseudo_monte_carlo(&request, policy(4)).expect("risk MC");
        let risks = &result.pricing_result.risks;
        for (estimate, expected) in [
            (risks.delta.expect("delta").raw(), oracle.delta),
            (risks.gamma.expect("gamma").raw(), oracle.gamma),
            (risks.vega.expect("vega").raw(), oracle.vega),
        ] {
            let error = (estimate.value().get() - expected).abs();
            assert!(error <= 6.0 * estimate.standard_error().get() + 2.0e-5);
        }
        assert_eq!(
            risks.delta.expect("delta").market_scaled().value().get(),
            risks.delta.expect("delta").raw().value().get()
        );
        assert_eq!(result.diagnostics.aad_tile_capacity, 64);
        assert_eq!(result.diagnostics.checkpoint_interval, 13);
    }

    #[test]
    fn adding_risks_does_not_change_seeded_price_bits() {
        let price_only = request(OptionSide::Put, 105.0, 0.35, 10_003, true);
        let with_risks = request_with_spot_and_risk(
            OptionSide::Put,
            105.0,
            0.35,
            10_003,
            true,
            100.0,
            all_risks(),
        );
        let price = price_pseudo_monte_carlo(&price_only, policy(3)).expect("price");
        let risk = price_pseudo_monte_carlo(&with_risks, policy(3)).expect("risk");
        assert_eq!(
            price.pricing_result.value.value().to_bits(),
            risk.pricing_result.value.value().to_bits()
        );
        assert_eq!(
            price
                .pricing_result
                .value
                .standard_error()
                .get()
                .to_bits(),
            risk.pricing_result
                .value
                .standard_error()
                .get()
                .to_bits()
        );
    }

    #[test]
    fn risk_replay_is_bitwise_equal_across_worker_counts() {
        let request = request_with_spot_and_risk(
            OptionSide::Call,
            97.0,
            0.31,
            10_003,
            true,
            100.0,
            all_risks(),
        );
        let single = price_pseudo_monte_carlo(&request, policy(1)).expect("single");
        let parallel = price_pseudo_monte_carlo(&request, policy(4)).expect("parallel");
        for (left, right) in [
            (
                single.pricing_result.risks.delta.expect("delta").raw(),
                parallel.pricing_result.risks.delta.expect("delta").raw(),
            ),
            (
                single.pricing_result.risks.gamma.expect("gamma").raw(),
                parallel.pricing_result.risks.gamma.expect("gamma").raw(),
            ),
            (
                single.pricing_result.risks.vega.expect("vega").raw(),
                parallel.pricing_result.risks.vega.expect("vega").raw(),
            ),
        ] {
            assert_eq!(left.value().to_bits(), right.value().to_bits());
            assert_eq!(
                left.standard_error().get().to_bits(),
                right.standard_error().get().to_bits()
            );
        }
    }

    #[test]
    fn aad_delta_and_vega_reconcile_with_common_random_number_bumps() {
        let units = 32_768;
        let base = request_with_spot_and_risk(
            OptionSide::Call,
            100.0,
            0.2,
            units,
            true,
            100.0,
            all_risks(),
        );
        let aad = price_pseudo_monte_carlo(&base, policy(4)).expect("AAD");
        let spot_bump = 0.01;
        let down_spot = price_pseudo_monte_carlo(
            &request_with_spot_and_risk(
                OptionSide::Call,
                100.0,
                0.2,
                units,
                true,
                100.0 - spot_bump,
                RiskRequest::price_only(SmileDynamics::StickyLogMoneyness),
            ),
            policy(4),
        )
        .expect("down spot");
        let up_spot = price_pseudo_monte_carlo(
            &request_with_spot_and_risk(
                OptionSide::Call,
                100.0,
                0.2,
                units,
                true,
                100.0 + spot_bump,
                RiskRequest::price_only(SmileDynamics::StickyLogMoneyness),
            ),
            policy(4),
        )
        .expect("up spot");
        let bump_delta = (up_spot.pricing_result.value.value().get()
            - down_spot.pricing_result.value.value().get())
            / (2.0 * spot_bump);
        let volatility_bump = 0.0001;
        let down_vol = price_pseudo_monte_carlo(
            &request(
                OptionSide::Call,
                100.0,
                0.2 - volatility_bump,
                units,
                true,
            ),
            policy(4),
        )
        .expect("down volatility");
        let up_vol = price_pseudo_monte_carlo(
            &request(
                OptionSide::Call,
                100.0,
                0.2 + volatility_bump,
                units,
                true,
            ),
            policy(4),
        )
        .expect("up volatility");
        let bump_vega = (up_vol.pricing_result.value.value().get()
            - down_vol.pricing_result.value.value().get())
            / (2.0 * volatility_bump);
        let risks = &aad.pricing_result.risks;
        assert!(
            (risks.delta.expect("delta").raw().value().get() - bump_delta).abs() < 5.0e-4
        );
        assert!((risks.vega.expect("vega").raw().value().get() - bump_vega).abs() < 5.0e-3);
    }

    #[test]
    fn gamma_bump_must_leave_a_positive_down_spot() {
        let risk = RiskRequest::new(
            false,
            Some(GammaConfig::new(
                SpotBump::absolute(100.0).expect("positive"),
            )),
            false,
            None,
            SmileDynamics::StickyStrike,
            None,
            None,
        )
        .expect("risk");
        let request = request_with_spot_and_risk(
            OptionSide::Call,
            100.0,
            0.2,
            1024,
            true,
            100.0,
            risk,
        );
        assert!(matches!(
            SimulationPlan::compile(&request, policy(2)),
            Err(MonteCarloError::InvalidGammaBump { .. })
        ));
    }
}
