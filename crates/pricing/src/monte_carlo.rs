use pricing_core::{Date, DayCountConvention, SchemaVersion, UnderlyingId};
use pricing_market::{CurveRegion, DiscountCurve};
use pricing_mc::{
    DeterministicExecutor, EngineConfig, ExecutionPolicy, Philox4x32, PseudoMcConfig,
    RandomCoordinate, RandomDomain,
};
use pricing_models::ModelSpec;
use pricing_product::{CompiledPayoff, GraphFingerprint, GraphLimitPolicy, ProductSpec};

use crate::{
    Diagnostics, Estimate, EstimatorKind, MonteCarloError, PricingRequest, PricingResult,
    PricingWarning, ReplayMetadata, RiskReport, fingerprint_request,
};

const NORMAL_95: f64 = 1.959_963_984_540_054;

/// Immutable one-expiry plan for the European Black-Scholes pseudo-MC slice.
#[derive(Clone, Debug)]
pub struct SimulationPlan {
    valuation_date: Date,
    expiry: Date,
    underlying: UnderlyingId,
    time: f64,
    forward: f64,
    discount: f64,
    total_variance: f64,
    standard_deviation: f64,
    payoff: CompiledPayoff,
    engine: PseudoMcConfig,
    execution_policy: ExecutionPolicy,
    request_fingerprint: [u8; 32],
    discount_region: CurveRegion,
    dividend_region: CurveRegion,
}

impl SimulationPlan {
    pub fn compile(
        request: &PricingRequest,
        execution_policy: ExecutionPolicy,
    ) -> Result<Self, MonteCarloError> {
        if request.risk().delta()
            || request.risk().gamma().is_some()
            || request.risk().vega()
            || request.risk().vega_kt().is_some()
        {
            return Err(MonteCarloError::RiskRequestNotPriceOnly);
        }
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
        Ok(Self {
            valuation_date: request.valuation_date(),
            expiry: product.expiry(),
            underlying: product.underlying(),
            time,
            forward: forward_evaluation.forward,
            discount,
            total_variance,
            standard_deviation: total_variance.sqrt(),
            payoff,
            engine,
            execution_policy,
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
        let statistics = executor.try_map_reduce_statistics(
            self.engine.independent_sampling_units().get(),
            |sampling_unit| {
                let normal = if self.total_variance == 0.0 {
                    0.0
                } else {
                    generator.standard_normal(RandomCoordinate::new(
                        sampling_unit,
                        0,
                        RandomDomain::Valuation,
                    ))
                };
                let primary = self.discounted_payoff(normal)?;
                if antithetic {
                    let mate = self.discounted_payoff(-normal)?;
                    Ok((primary + mate) * 0.5)
                } else {
                    Ok(primary)
                }
            },
        )?;

        let independent_units = self.engine.independent_sampling_units().get();
        let price = statistics.sum().total() / independent_units as f64;
        let sampling_variance = if self.total_variance == 0.0 {
            0.0
        } else {
            statistics.moments().sample_variance().ok_or(
                MonteCarloError::InsufficientSamplingUnits {
                    count: independent_units,
                },
            )?
        };
        let estimator_variance = sampling_variance / independent_units as f64;
        let standard_error = estimator_variance.sqrt();
        let half_width = NORMAL_95 * standard_error;
        let estimate = Estimate::new(
            price,
            standard_error,
            price - half_width,
            price + half_width,
            EstimatorKind::PseudoMonteCarlo,
            independent_units,
        )?;
        let warnings = extrapolation_warnings(self.discount_region, self.dividend_region);
        let pricing_result = PricingResult {
            value: estimate,
            risks: RiskReport::default(),
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
                antithetic,
                discount_region: self.discount_region,
                dividend_region: self.dividend_region,
                payoff_fingerprint: self.payoff.tape_fingerprint(),
            },
        })
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
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MonteCarloDiagnostics {
    pub master_seed: u64,
    pub policy_version: u32,
    pub worker_threads: u32,
    pub reduction_block_size: u64,
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
    use pricing_risk::SmileDynamics;

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
                PositiveF64::new(100.0, "spot").expect("spot"),
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
            pricing_risk::RiskRequest::price_only(SmileDynamics::StickyLogMoneyness),
        )
        .expect("request")
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
}
