use pricing_core::PositiveF64;
use pricing_mc::ExecutionPolicy;
use pricing_risk::PayoffSmoothing;

use crate::{
    Fingerprint, MigrationProvenance, MonteCarloError, MonteCarloPrice, PricingRequest,
    SimulationPlan,
};

/// Stable compiled-plan facade. All mutable validation and compilation work is
/// completed before this value can be evaluated.
#[derive(Clone, Debug)]
pub struct PricingPlan {
    simulation: SimulationPlan,
    width_ladder: Box<[(PositiveF64, SimulationPlan)]>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WidthLadderDifference {
    pub price: f64,
    pub delta: Option<f64>,
    pub gamma: Option<f64>,
    pub vega: Option<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct WidthLadderEntry {
    pub half_width: PositiveF64,
    pub result: MonteCarloPrice,
    pub adjacent_difference: Option<WidthLadderDifference>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct WidthLadderResult {
    pub primary: MonteCarloPrice,
    pub entries: Box<[WidthLadderEntry]>,
}

impl PricingPlan {
    pub fn compile(
        request: &PricingRequest,
        execution_policy: ExecutionPolicy,
    ) -> Result<Self, MonteCarloError> {
        let width_ladder = request.risk().payoff_smoothing_width_ladder().map_or_else(
            || Ok(Vec::new()),
            |ladder| {
                ladder
                    .half_widths()
                    .iter()
                    .copied()
                    .map(|half_width| {
                        let smoothing = PayoffSmoothing::compact_c2(half_width.get())?;
                        let risk = request
                            .risk()
                            .clone()
                            .without_payoff_smoothing_width_ladder()
                            .with_payoff_smoothing(smoothing);
                        let width_request = request.clone().replace_risk(risk);
                        Ok((
                            half_width,
                            SimulationPlan::compile(&width_request, execution_policy)?,
                        ))
                    })
                    .collect::<Result<Vec<_>, MonteCarloError>>()
            },
        )?;
        Ok(Self {
            simulation: SimulationPlan::compile(request, execution_policy)?,
            width_ladder: width_ladder.into_boxed_slice(),
        })
    }

    pub fn evaluate(&self) -> Result<MonteCarloPrice, MonteCarloError> {
        self.simulation.execute()
    }

    pub fn evaluate_width_ladder(&self) -> Result<WidthLadderResult, MonteCarloError> {
        let primary = self.simulation.execute()?;
        let mut entries = Vec::with_capacity(self.width_ladder.len());
        for (half_width, simulation) in &self.width_ladder {
            let result = simulation.execute()?;
            let adjacent_difference = entries
                .last()
                .map(|previous: &WidthLadderEntry| difference(&result, &previous.result));
            entries.push(WidthLadderEntry {
                half_width: *half_width,
                result,
                adjacent_difference,
            });
        }
        Ok(WidthLadderResult {
            primary,
            entries: entries.into_boxed_slice(),
        })
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

    #[must_use]
    pub const fn request_migration(&self) -> &MigrationProvenance {
        self.simulation.request_migration()
    }
}

fn difference(current: &MonteCarloPrice, previous: &MonteCarloPrice) -> WidthLadderDifference {
    let risk_difference = |current: Option<crate::RiskEstimate>,
                           previous: Option<crate::RiskEstimate>| {
        current
            .zip(previous)
            .map(|(current, previous)| current.raw().value().get() - previous.raw().value().get())
    };
    WidthLadderDifference {
        price: current.pricing_result.value.value().get()
            - previous.pricing_result.value.value().get(),
        delta: risk_difference(
            current.pricing_result.risks.delta,
            previous.pricing_result.risks.delta,
        ),
        gamma: risk_difference(
            current.pricing_result.risks.gamma,
            previous.pricing_result.risks.gamma,
        ),
        vega: risk_difference(
            current.pricing_result.risks.vega,
            previous.pricing_result.risks.vega,
        ),
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
    use pricing_product::{
        DigitalPayout, DigitalSpec, EuropeanVanillaSpec, OptionSide, ProductSpec,
    };
    use pricing_risk::{PayoffSmoothing, PayoffSmoothingWidthLadder, RiskRequest, SmileDynamics};

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

    #[test]
    fn width_ladder_preserves_order_primary_and_common_random_coordinates() {
        let engine = EngineConfig::PseudoMonteCarlo(
            PseudoMcConfig::new(7, 4096, VarianceReduction::new(true, false)).expect("engine"),
        );
        let base = request(engine);
        let risk = RiskRequest::new(
            true,
            None,
            true,
            None,
            SmileDynamics::StickyLogMoneyness,
            None,
            None,
        )
        .expect("risk")
        .with_payoff_smoothing(PayoffSmoothing::compact_c2(3.0).expect("primary"))
        .with_payoff_smoothing_width_ladder(
            PayoffSmoothingWidthLadder::new(vec![4.0, 2.0, 1.0]).expect("ladder"),
        );
        let request = PricingRequest::new(
            base.valuation_date(),
            ProductSpec::Digital(
                DigitalSpec::new(
                    UnderlyingId::new(1),
                    CurrencyId::new(1),
                    "2027-09-04".parse().expect("expiry"),
                    100.0,
                    10.0,
                    OptionSide::Call,
                    DigitalPayout::Cash,
                )
                .expect("Digital"),
            ),
            base.market().clone(),
            base.model().clone(),
            engine,
            risk,
        )
        .expect("request");
        let plan = PricingPlan::compile(&request, policy(2)).expect("plan");
        let standalone_primary = plan.evaluate().expect("primary");
        let ladder = plan.evaluate_width_ladder().expect("ladder");
        assert_eq!(ladder.primary, standalone_primary);
        assert_eq!(
            ladder
                .entries
                .iter()
                .map(|entry| entry.half_width.get())
                .collect::<Vec<_>>(),
            vec![4.0, 2.0, 1.0]
        );
        assert_eq!(
            ladder
                .primary
                .diagnostics
                .payoff_smoothing
                .expect("primary smoothing")
                .half_width
                .get(),
            3.0
        );
        for entry in &ladder.entries {
            assert_eq!(entry.result.diagnostics.master_seed, 7);
            assert_eq!(
                entry.result.evaluated_paths,
                standalone_primary.evaluated_paths
            );
            assert_eq!(
                entry
                    .result
                    .diagnostics
                    .payoff_smoothing
                    .expect("smoothing")
                    .half_width,
                entry.half_width
            );
        }
        assert!(ladder.entries[0].adjacent_difference.is_none());
        for pair in ladder.entries.windows(2) {
            let difference = pair[1].adjacent_difference.expect("difference");
            assert_eq!(
                difference.price,
                pair[1].result.pricing_result.value.value().get()
                    - pair[0].result.pricing_result.value.value().get()
            );
            assert_eq!(
                difference.delta,
                Some(
                    pair[1]
                        .result
                        .pricing_result
                        .risks
                        .delta
                        .expect("current Delta")
                        .raw()
                        .value()
                        .get()
                        - pair[0]
                            .result
                            .pricing_result
                            .risks
                            .delta
                            .expect("previous Delta")
                            .raw()
                            .value()
                            .get()
                )
            );
        }
    }
}
