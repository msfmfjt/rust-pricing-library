use pricing_mc::ExecutionPolicy;

use crate::{Fingerprint, MonteCarloError, MonteCarloPrice, PricingRequest, SimulationPlan};

/// Stable compiled-plan facade. All mutable validation and compilation work is
/// completed before this value can be evaluated.
#[derive(Clone, Debug)]
pub struct PricingPlan {
    simulation: SimulationPlan,
}

impl PricingPlan {
    pub fn compile(
        request: &PricingRequest,
        execution_policy: ExecutionPolicy,
    ) -> Result<Self, MonteCarloError> {
        Ok(Self {
            simulation: SimulationPlan::compile(request, execution_policy)?,
        })
    }

    pub fn evaluate(&self) -> Result<MonteCarloPrice, MonteCarloError> {
        self.simulation.execute()
    }

    #[must_use]
    pub const fn request_fingerprint(&self) -> Fingerprint {
        self.simulation.request_fingerprint()
    }

    #[must_use]
    pub const fn plan_fingerprint(&self) -> Fingerprint {
        self.simulation.plan_fingerprint()
    }

    #[must_use]
    pub const fn execution_policy(&self) -> ExecutionPolicy {
        self.simulation.execution_policy()
    }
}

pub fn compile(
    request: &PricingRequest,
    execution_policy: ExecutionPolicy,
) -> Result<PricingPlan, MonteCarloError> {
    PricingPlan::compile(request, execution_policy)
}

pub fn evaluate(plan: &PricingPlan) -> Result<MonteCarloPrice, MonteCarloError> {
    plan.evaluate()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use pricing_core::{CurrencyId, CurveId, PositiveF64, UnderlyingId};
    use pricing_market::{EquityForward, EquityMarket, LogLinearDiscountCurve, MarketContext};
    use pricing_mc::{EngineConfig, PseudoMcConfig, RqmcConfig, VarianceReduction};
    use pricing_models::{BlackScholesSpec, ModelSpec};
    use pricing_product::{EuropeanVanillaSpec, OptionSide, ProductSpec};
    use pricing_risk::{RiskRequest, SmileDynamics};

    use super::*;
    use crate::{JsonLimits, parse_request_json, request_to_json};

    fn curve(id: u32, rate: f64) -> Arc<LogLinearDiscountCurve> {
        Arc::new(
            LogLinearDiscountCurve::new(CurveId::new(id), vec![0.0, 1.0], vec![1.0, (-rate).exp()])
                .expect("curve"),
        )
    }

    fn request(engine: EngineConfig) -> PricingRequest {
        let underlying = UnderlyingId::new(1);
        let currency = CurrencyId::new(1);
        PricingRequest::new(
            "2026-09-04".parse().expect("valuation"),
            ProductSpec::EuropeanVanilla(
                EuropeanVanillaSpec::new(
                    underlying,
                    currency,
                    "2027-09-04".parse().expect("expiry"),
                    100.0,
                    1.0,
                    OptionSide::Call,
                )
                .expect("product"),
            ),
            MarketContext::Equity(EquityMarket::new(
                currency,
                EquityForward::new(
                    underlying,
                    PositiveF64::new(100.0, "spot").expect("spot"),
                    curve(1, 0.05),
                    curve(2, 0.02),
                ),
            )),
            ModelSpec::BlackScholes(BlackScholesSpec::new(0.2).expect("model")),
            engine,
            RiskRequest::price_only(SmileDynamics::StickyLogMoneyness),
        )
        .expect("request")
    }

    fn policy(workers: u32) -> ExecutionPolicy {
        ExecutionPolicy::new(workers, Some(1024)).expect("policy")
    }

    #[test]
    fn public_compile_and_evaluate_replay_pseudo_mc() {
        let request = request(EngineConfig::PseudoMonteCarlo(
            PseudoMcConfig::new(7, 4096, VarianceReduction::new(true, false)).expect("engine"),
        ));
        let first = compile(&request, policy(2)).expect("compile");
        let second = PricingPlan::compile(&request, policy(2)).expect("compile replay");
        assert_eq!(first.request_fingerprint(), second.request_fingerprint());
        assert_eq!(first.plan_fingerprint(), second.plan_fingerprint());
        assert_eq!(
            evaluate(&first).expect("evaluate"),
            second.evaluate().expect("replay")
        );
    }

    #[test]
    fn normalized_json_and_rust_requests_compile_to_the_same_plan() {
        let request = request(EngineConfig::RandomizedQuasiMonteCarlo(
            RqmcConfig::new(256, 4, 11, VarianceReduction::new(true, true)).expect("engine"),
        ));
        let json = request_to_json(&request).expect("serialize");
        let parsed = parse_request_json(json.as_bytes(), JsonLimits::DEFAULT).expect("parse");
        let rust_plan = compile(&request, policy(2)).expect("Rust plan");
        let json_plan = compile(&parsed, policy(2)).expect("JSON plan");
        assert_eq!(
            rust_plan.request_fingerprint(),
            json_plan.request_fingerprint()
        );
        assert_eq!(rust_plan.plan_fingerprint(), json_plan.plan_fingerprint());
        assert_eq!(
            rust_plan.evaluate().expect("Rust"),
            json_plan.evaluate().expect("JSON")
        );
    }

    #[test]
    fn execution_configuration_changes_plan_but_not_request_fingerprint() {
        let request = request(EngineConfig::PseudoMonteCarlo(
            PseudoMcConfig::new(7, 128, VarianceReduction::new(false, false)).expect("engine"),
        ));
        let one = compile(&request, policy(1)).expect("one worker");
        let four = compile(&request, policy(4)).expect("four workers");
        assert_eq!(one.request_fingerprint(), four.request_fingerprint());
        assert_ne!(one.plan_fingerprint(), four.plan_fingerprint());
    }
}
