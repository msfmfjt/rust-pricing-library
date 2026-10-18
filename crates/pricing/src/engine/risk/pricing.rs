use crate::core::PositiveF64;
use crate::mc::ExecutionPolicy;
use crate::risk::PayoffSmoothing;

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
    pub fn bass_calibration_diagnostics(
        &self,
    ) -> Option<&[crate::bass_lv::BassCalibrationDiagnostics]> {
        self.simulation.bass_calibration_diagnostics()
    }
    pub fn bass_projection_diagnostics(&self) -> Option<&[crate::bass_lv::BassSurfaceDiagnostics]> {
        self.simulation.bass_projection_diagnostics()
    }
    pub fn bass_vega_scenario_diagnostics(
        &self,
    ) -> Option<&[crate::bass_lv::BassVegaKtScenarioDiagnostics]> {
        self.simulation.bass_vega_scenario_diagnostics()
    }
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

    /// Deterministic signed terminal Gamma measure and equation-(9) projection
    /// for a compiled continuous piecewise-affine terminal payoff, spatially flat
    /// variance and zero carry. Expected source-hat local Vegas include t=0.
    /// Sampling settings are unused: no MC moments or covariance are produced.
    /// Does not replace evaluate() or certify the ordinary acceptance gate.
    pub fn evaluate_vega_kt_terminal_diagnostic(
        &self,
    ) -> Result<crate::risk::VegaKtTerminalDiagnostic, MonteCarloError> {
        self.simulation.evaluate_vega_kt_terminal_diagnostic()
    }

    /// Diagnose the flat-space, zero-carry weak Gamma chain using this compiled payoff,
    /// observation map, engine seed and sampling units. Does not replace evaluate().
    /// Rejects unsupported models, indicators/barriers, early exercise, smoothing,
    /// RQMC and full covariance requests instead of falling back to grid allocation.
    /// Initial strike anchors and the recovery log grid are subdivided with spacing
    /// bounded by min(relative_bump/2, sqrt(first-step variance)/2), up to 1025 nodes.
    /// The result records both requested anchors and actual numerical grids.
    pub fn evaluate_vega_kt_chain_diagnostic(
        &self,
        initial_strikes: Vec<f64>,
        relative_bump: f64,
        subdivisions: usize,
    ) -> Result<crate::risk::VegaKtChainDiagnostic, MonteCarloError> {
        self.simulation.evaluate_vega_kt_chain_diagnostic(
            initial_strikes,
            relative_bump,
            subdivisions,
        )
    }

    /// Full source-hat convolution recovery of positive-time Gamma from the
    /// compiled continuous path payoff's AD. Explicit regularization; initial
    /// risk stays unrecovered. Does not certify or replace ordinary IV buckets.
    pub fn evaluate_vega_kt_gamma_recovery_diagnostic(
        &self,
        gamma_strikes: Vec<f64>,
        relative_regularization: f64,
    ) -> Result<crate::risk::VegaKtConvolutionDiagnostic, MonteCarloError> {
        self.simulation
            .evaluate_vega_kt_gamma_recovery_diagnostic(gamma_strikes, relative_regularization)
    }

    /// Initial-boundary Gamma from conditional piecewise-affine payoff measures.
    /// Bins include both tails. First-step Vega uses original atom locations,
    /// independent of binning, with same-draw differences against native AD.
    /// Spatially flat, zero carry, continuous affine/min/max payoffs only.
    pub fn evaluate_vega_kt_initial_gamma_diagnostic(
        &self,
        gamma_bin_boundaries: Vec<f64>,
    ) -> Result<crate::risk::VegaKtInitialMeasureDiagnostic, MonteCarloError> {
        self.simulation
            .evaluate_vega_kt_initial_gamma_diagnostic(gamma_bin_boundaries)
    }

    /// Select regularization on disjoint pilot counters, then validate once on
    /// the original pricing counters. Includes the separate initial measure.
    /// A failed validation is retained, never used to choose another candidate.
    pub fn evaluate_vega_kt_gamma_recovery_audit(
        &self,
        gamma_strikes: Vec<f64>,
        policy: &crate::risk::VegaKtGammaRecoveryPolicy,
    ) -> Result<crate::risk::VegaKtGammaRecoveryAudit, MonteCarloError> {
        self.simulation
            .evaluate_vega_kt_gamma_recovery_audit(gamma_strikes, policy)
    }

    /// Same-path joint annual-variance readouts after both Gamma recovery gates.
    pub fn evaluate_vega_kt_joint_gamma_audit(
        &self,
        gamma_strikes: Vec<f64>,
        initial_gamma_bin_boundaries: Vec<f64>,
        policy: &crate::risk::VegaKtGammaRecoveryPolicy,
        directions: crate::risk::VegaKtJointGammaDirections,
        subdivisions: usize,
        relative_tolerance: f64,
    ) -> Result<crate::risk::VegaKtJointGammaAudit, MonteCarloError> {
        self.simulation.evaluate_vega_kt_joint_gamma_audit(
            gamma_strikes,
            initial_gamma_bin_boundaries,
            policy,
            directions,
            subdivisions,
            relative_tolerance,
        )
    }

    /// Refined initial continuation Gamma for two-step nonflat local variance.
    pub fn evaluate_vega_kt_initial_smile_gamma_audit(
        &self,
        gamma_bin_boundaries: Vec<f64>,
        subdivisions: usize,
        relative_tolerance: f64,
    ) -> Result<crate::risk::VegaKtInitialSmileGammaAudit, MonteCarloError> {
        self.simulation.evaluate_vega_kt_initial_smile_gamma_audit(
            gamma_bin_boundaries,
            subdivisions,
            relative_tolerance,
        )
    }

    /// Initial Gamma through both subsequent local-variance transitions.
    pub fn evaluate_vega_kt_three_step_initial_gamma_audit(
        &self,
        boundaries: Vec<f64>,
        subdivisions: usize,
        relative_tolerance: f64,
    ) -> Result<crate::risk::VegaKtThreeStepInitialGammaAudit, MonteCarloError> {
        self.simulation
            .evaluate_vega_kt_three_step_initial_gamma_audit(
                boundaries,
                subdivisions,
                relative_tolerance,
            )
    }

    /// History-conditional middle Gamma with separate deterministic and MC gates.
    pub fn evaluate_vega_kt_middle_smile_gamma_audit(
        &self,
        boundaries: Vec<f64>,
        subdivisions: usize,
        relative_tolerance: f64,
    ) -> Result<crate::risk::VegaKtMiddleSmileGammaAudit, MonteCarloError> {
        self.simulation.evaluate_vega_kt_middle_smile_gamma_audit(
            boundaries,
            subdivisions,
            relative_tolerance,
        )
    }

    /// Selected-row conditional Gamma with the complete history and future evolution.
    pub fn evaluate_vega_kt_multi_step_gamma_chain_audit(
        &self,
        rows: Vec<usize>,
        boundaries: Vec<f64>,
        directions: Vec<Vec<Vec<f64>>>,
        subdivisions: usize,
        relative_tolerance: f64,
        gamma_se_limit: f64,
    ) -> Result<crate::risk::VegaKtPathGammaChainAudit, MonteCarloError> {
        self.simulation
            .evaluate_vega_kt_multi_step_gamma_chain_audit(
                rows,
                boundaries,
                directions,
                subdivisions,
                relative_tolerance,
                gamma_se_limit,
            )
    }

    /// Same audit with optional counter-ordered readouts for portfolio covariance.
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_vega_kt_multi_step_gamma_chain_with_samples(
        &self,
        rows: Vec<usize>,
        boundaries: Vec<f64>,
        directions: Vec<Vec<Vec<f64>>>,
        subdivisions: usize,
        relative_tolerance: f64,
        gamma_se_limit: f64,
        retain_samples: bool,
    ) -> Result<crate::risk::VegaKtPathGammaChainAudit, MonteCarloError> {
        self.simulation
            .evaluate_vega_kt_multi_step_gamma_chain_with_samples(
                rows,
                boundaries,
                directions,
                subdivisions,
                relative_tolerance,
                gamma_se_limit,
                retain_samples,
            )
    }

    /// Last-row Gamma after 1..15 integrated earlier log-Euler transitions.
    pub fn evaluate_vega_kt_multi_step_terminal_gamma_audit(
        &self,
        strikes: Vec<f64>,
        policy: &crate::risk::VegaKtGammaRecoveryPolicy,
        directions: Vec<Vec<f64>>,
    ) -> Result<crate::risk::VegaKtMultiStepTerminalGammaAudit, MonteCarloError> {
        self.simulation
            .evaluate_vega_kt_multi_step_terminal_gamma_audit(strikes, policy, directions)
    }

    /// Third-row recovery under the actual two-transition source distribution.
    /// Initial and middle continuation Gamma are not inferred by this audit.
    pub fn evaluate_vega_kt_three_step_terminal_gamma_audit(
        &self,
        strikes: Vec<f64>,
        policy: &crate::risk::VegaKtGammaRecoveryPolicy,
        directions: Vec<Vec<f64>>,
    ) -> Result<crate::risk::VegaKtThreeStepTerminalGammaAudit, crate::MonteCarloError> {
        self.simulation
            .evaluate_vega_kt_three_step_terminal_gamma_audit(strikes, policy, directions)
    }

    /// Dependent terminal readouts certified against an independent constraint basis.
    pub fn evaluate_vega_kt_three_step_terminal_readout_audit(
        &self,
        strikes: Vec<f64>,
        policy: &crate::risk::VegaKtGammaRecoveryPolicy,
        directions: Vec<Vec<f64>>,
        readouts: Vec<Vec<f64>>,
    ) -> Result<crate::risk::VegaKtThreeStepTerminalGammaAudit, MonteCarloError> {
        self.simulation
            .evaluate_vega_kt_three_step_terminal_readout_audit(
                strikes, policy, directions, readouts,
            )
    }

    /// Two-step nonflat variance-node recovery with a fixed first Gaussian marginal.
    /// No initial Gamma or full IV-chain estimate is inferred from this audit.
    pub fn evaluate_vega_kt_two_step_smile_gamma_audit(
        &self,
        gamma_strikes: Vec<f64>,
        policy: &crate::risk::VegaKtGammaRecoveryPolicy,
        variance_conservation_directions: Vec<Vec<f64>>,
    ) -> Result<crate::risk::VegaKtTwoStepSmileGammaAudit, MonteCarloError> {
        self.simulation.evaluate_vega_kt_two_step_smile_gamma_audit(
            gamma_strikes,
            policy,
            variance_conservation_directions,
        )
    }

    /// Joint signed-atom equation-(4)/(9) IV diagnostic after held-out Gamma
    /// validation. Spatially flat calibrated IV, zero carry, state-only payoff.
    /// Failed recovery returns no estimate; does not promote ordinary IV risk.
    pub fn evaluate_vega_kt_atom_chain_diagnostic(
        &self,
        gamma_strikes: Vec<f64>,
        policy: &crate::risk::VegaKtGammaRecoveryPolicy,
    ) -> Result<crate::risk::VegaKtAtomChainDiagnostic, MonteCarloError> {
        self.simulation
            .evaluate_vega_kt_atom_chain_diagnostic(gamma_strikes, policy)
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

    use crate::core::{CurrencyId, CurveId, PositiveF64, UnderlyingId};
    use crate::market::{EquityForward, EquityMarket, LogLinearDiscountCurve, MarketContext};
    use crate::mc::{EngineConfig, PseudoMcConfig, RqmcConfig, VarianceReduction};
    use crate::models::{BlackScholesSpec, ModelSpec};
    use crate::product::{
        DigitalPayout, DigitalSpec, EuropeanVanillaSpec, OptionSide, ProductSpec,
    };
    use crate::risk::{PayoffSmoothing, PayoffSmoothingWidthLadder, RiskRequest, SmileDynamics};

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
