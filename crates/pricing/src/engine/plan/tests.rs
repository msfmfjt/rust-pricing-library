//! plan / tests implementation.

use crate::api::mc_result::BarrierBridgeDiagnostics;
use crate::api::mc_result::BarrierHitIndicatorMode;
use crate::api::mc_result::ExerciseStrategyRisk;
use crate::api::mc_result::PathStateDiagnostics;
use crate::api::mc_result::PayoffSmoothingKernel;
use crate::api::mc_result::PayoffSmoothingWidthUnit;
use crate::api::mc_result::PayoffValuationKind;
use crate::api::mc_result::RiskMethod;
use crate::api::mc_result::StoppingIndexRisk;
use crate::core::{Date, DayCountConvention, PathIndex, UnderlyingId};
use crate::engine::plan::simulation::ContinuousBarrierRuntime;
use crate::engine::plan::simulation::SMOOTHED_BARRIER_BRIDGE_ABI;
use crate::engine::plan::simulation::SimulationPlan;
use crate::engine::plan::simulation::VEGA;
use crate::engine::plan::simulation::bucket_sample_capacity;
use crate::engine::risk::valuation::price_monte_carlo;
use crate::engine::risk::valuation::price_pseudo_monte_carlo;
use crate::market::DiscountCurve;
use crate::mc::{
    BARRIER_BRIDGE_ABI, BarrierBridgeError, EngineConfig, ExecutionPolicy, LsmConfig,
    LsmStateVariable, Philox4x32, RandomDomain,
};
use crate::models::{LocalVolatilityReportingBasis, ModelSpec};
use crate::risk::PayoffSmoothing;
use crate::{EstimatorKind, MonteCarloError, PricingRequest, fingerprint_request};

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::core::{CurrencyId, CurveId, EventId, PositiveF64};
    use crate::market::{
        DividendEvent, DividendQuote, EquityForward, EquityMarket, LogLinearDiscountCurve,
        MarketContext,
    };
    use crate::mc::{
        CpqrConfig, PolynomialBasisSpec, PseudoMcConfig, RqmcConfig, VarianceReduction,
    };
    use crate::models::{Black76Spec, BlackScholesSpec, LocalVolatilitySpec};
    use crate::product::{
        AmericanVanillaSpec, ArithmeticAsianSpec, AsianObservation, BarrierDirection,
        BarrierMonitoring, BarrierSpec, BarrierStyle, DigitalPayout, DigitalSpec,
        EuropeanVanillaSpec, FixedLookbackSpec, OptionSide, ProductSpec,
    };
    use crate::risk::{GammaConfig, RiskRequest, SmileDynamics, SpotBump, VegaKtConfig};

    use super::*;
    use crate::analytical::{black_76_oracle, black_scholes_oracle};

    #[test]
    fn bucket_sample_capacity_rejects_overflow() {
        assert_eq!(bucket_sample_capacity(3, 4), 12);
        assert_eq!(bucket_sample_capacity(usize::MAX, 2), 0);
    }

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

    fn american_request(
        side: OptionSide,
        strike: f64,
        volatility: f64,
        training_units: u64,
        valuation_units: u64,
        antithetic: bool,
    ) -> PricingRequest {
        let underlying = UnderlyingId::new(1);
        let currency = CurrencyId::new(1);
        let exercise_dates = ["2026-12-04", "2027-03-04", "2027-06-04", "2027-09-04"]
            .map(|date| date.parse().expect("exercise date"));
        let product = ProductSpec::AmericanVanilla(
            AmericanVanillaSpec::new(
                underlying,
                currency,
                *exercise_dates.last().expect("exercise dates"),
                strike,
                1.0,
                side,
                exercise_dates.to_vec(),
            )
            .expect("American product"),
        );
        let market = MarketContext::Equity(EquityMarket::new(
            currency,
            EquityForward::new(
                underlying,
                PositiveF64::new(100.0, "spot").expect("spot"),
                curve(1, 0.05),
                curve(2, 0.0),
            ),
        ));
        let training_engine = EngineConfig::PseudoMonteCarlo(
            PseudoMcConfig::new(
                0x1020_3040_5060_7080,
                training_units,
                VarianceReduction::new(antithetic, false),
            )
            .expect("training engine"),
        );
        let lsm = LsmConfig::new(
            training_engine,
            vec![LsmStateVariable::Spot],
            PolynomialBasisSpec::new(1, 3, 8, 8).expect("basis"),
            0.0,
            CpqrConfig::new(1.0e-14, 1.0e-12).expect("CPQR config"),
            1_000_000,
        )
        .expect("LSM config");
        PricingRequest::new_with_lsm(
            "2026-09-04".parse().expect("valuation"),
            product,
            market,
            ModelSpec::BlackScholes(BlackScholesSpec::new(volatility).expect("model")),
            EngineConfig::PseudoMonteCarlo(
                PseudoMcConfig::new(
                    0x0123_4567_89ab_cdef,
                    valuation_units,
                    VarianceReduction::new(antithetic, false),
                )
                .expect("valuation engine"),
            ),
            RiskRequest::price_only(SmileDynamics::StickyLogMoneyness),
            Some(lsm),
        )
        .expect("American request")
    }

    fn american_rqmc_request(
        side: OptionSide,
        strike: f64,
        training_seed: u64,
        valuation_seed: u64,
    ) -> PricingRequest {
        let base = american_request(side, strike, 0.2, 2, 2, true);
        let training_engine = EngineConfig::RandomizedQuasiMonteCarlo(
            RqmcConfig::new(2048, 8, training_seed, VarianceReduction::new(true, true))
                .expect("training RQMC engine"),
        );
        let lsm = LsmConfig::new(
            training_engine,
            vec![LsmStateVariable::Spot],
            PolynomialBasisSpec::new(1, 3, 8, 8).expect("basis"),
            0.0,
            CpqrConfig::new(1.0e-14, 1.0e-12).expect("CPQR config"),
            1_000_000,
        )
        .expect("LSM config");
        PricingRequest::new_with_lsm(
            base.valuation_date(),
            base.product().clone(),
            base.market().clone(),
            base.model().clone(),
            EngineConfig::RandomizedQuasiMonteCarlo(
                RqmcConfig::new(4096, 16, valuation_seed, VarianceReduction::new(true, true))
                    .expect("valuation RQMC engine"),
            ),
            base.risk().clone(),
            Some(lsm),
        )
        .expect("American RQMC request")
    }

    fn with_constant_local_volatility(request: &PricingRequest) -> PricingRequest {
        let local_volatility = LocalVolatilitySpec::from_explicit_grid(
            vec![0.0, 1.0],
            vec![-1.0, 1.0],
            vec![0.04, 0.04, 0.04, 0.04],
            1.0e-8,
            1.0,
        )
        .expect("constant Local Volatility model");
        with_model(request, ModelSpec::LocalVolatility(local_volatility))
    }

    fn with_model(request: &PricingRequest, model: ModelSpec) -> PricingRequest {
        PricingRequest::new_with_lsm(
            request.valuation_date(),
            request.product().clone(),
            request.market().clone(),
            model,
            request.engine(),
            request.risk().clone(),
            request.lsm().cloned(),
        )
        .expect("Local Volatility American request")
    }

    fn with_risk(request: &PricingRequest, risk: RiskRequest) -> PricingRequest {
        PricingRequest::new_with_lsm(
            request.valuation_date(),
            request.product().clone(),
            request.market().clone(),
            request.model().clone(),
            request.engine(),
            risk,
            request.lsm().cloned(),
        )
        .expect("request with risk")
    }

    fn american_vega_kt_risk(full_bucket_covariance: bool) -> RiskRequest {
        RiskRequest::new(
            true,
            Some(GammaConfig::new(
                SpotBump::relative(0.01).expect("gamma bump"),
            )),
            true,
            Some(
                VegaKtConfig::new(
                    vec![
                        "2027-03-05".parse().expect("first maturity"),
                        "2027-09-04".parse().expect("second maturity"),
                    ],
                    vec![-1.0, 0.0, 1.0],
                    1.0e-8,
                    full_bucket_covariance,
                )
                .expect("VegaKT config"),
            ),
            SmileDynamics::StickyLogMoneyness,
            Some(16),
            Some(128),
        )
        .expect("American VegaKT risk")
    }

    fn small_american_rqmc_request() -> PricingRequest {
        let base = american_request(OptionSide::Put, 100.0, 0.2, 2, 2, true);
        let lsm = LsmConfig::new(
            EngineConfig::RandomizedQuasiMonteCarlo(
                RqmcConfig::new(128, 4, 0x7171, VarianceReduction::new(true, true))
                    .expect("training RQMC engine"),
            ),
            vec![LsmStateVariable::Spot],
            PolynomialBasisSpec::new(1, 3, 8, 8).expect("basis"),
            0.0,
            CpqrConfig::new(1.0e-14, 1.0e-12).expect("CPQR config"),
            1_000_000,
        )
        .expect("LSM config");
        PricingRequest::new_with_lsm(
            base.valuation_date(),
            base.product().clone(),
            base.market().clone(),
            base.model().clone(),
            EngineConfig::RandomizedQuasiMonteCarlo(
                RqmcConfig::new(256, 8, 0x8181, VarianceReduction::new(true, true))
                    .expect("valuation RQMC engine"),
            ),
            base.risk().clone(),
            Some(lsm),
        )
        .expect("small American RQMC request")
    }

    #[test]
    fn american_zero_volatility_exercises_at_deterministic_optimal_date() {
        let request = american_request(OptionSide::Put, 120.0, 0.0, 2, 2, true);
        let result = price_monte_carlo(&request, policy(2)).expect("American price");
        let diagnostics = result
            .early_exercise_diagnostics
            .as_ref()
            .expect("early-exercise diagnostics");
        let first_date = diagnostics.exercise_dates[0];
        let time = DayCountConvention::Act365F.year_fraction(request.valuation_date(), first_date);
        let forward = request
            .market()
            .equity()
            .forward()
            .evaluate(time)
            .expect("forward");
        let discount = request
            .market()
            .equity()
            .forward()
            .discount_curve()
            .evaluate(time)
            .expect("discount")
            .discount;
        let expected = discount * (120.0 - forward.forward).max(0.0);
        assert_eq!(
            result.pricing_result.value.value().get().to_bits(),
            expected.to_bits()
        );
        assert_eq!(diagnostics.exercise_counts, Box::from([4, 0, 0, 0]));
        assert_eq!(
            diagnostics.exercise_probabilities,
            Box::from([1.0, 0.0, 0.0, 0.0])
        );
        assert!(diagnostics.stopping_indices.iter().all(|index| *index == 0));
        assert_eq!(diagnostics.training_random_domain, RandomDomain::LsmTrain);
        assert_eq!(diagnostics.valuation_random_domain, RandomDomain::Valuation);
        assert_eq!(diagnostics.training_sampling_units, 2);
        assert_eq!(diagnostics.training_trajectories, 4);
        assert_eq!(diagnostics.valuation_sampling_units, 2);
        assert_eq!(diagnostics.valuation_trajectories, 4);
        assert_eq!(diagnostics.in_sample_value.to_bits(), expected.to_bits());
    }

    #[test]
    fn american_pseudo_replays_across_worker_counts() {
        let request = american_request(OptionSide::Put, 100.0, 0.2, 1024, 2048, true);
        let request_json = crate::request_to_json(&request).expect("serialize American request");
        let parsed = crate::parse_request_json(request_json.as_bytes(), crate::JsonLimits::DEFAULT)
            .expect("parse American request");
        assert_eq!(parsed, request);
        assert_eq!(fingerprint_request(&request), fingerprint_request(&request));
        let serial = price_monte_carlo(&request, policy(1)).expect("serial American price");
        let parallel = price_monte_carlo(&request, policy(4)).expect("parallel American price");
        assert_eq!(serial.pricing_result, parallel.pricing_result);
        assert_eq!(
            serial.early_exercise_diagnostics,
            parallel.early_exercise_diagnostics
        );
        let diagnostics = serial
            .early_exercise_diagnostics
            .expect("early-exercise diagnostics");
        assert_eq!(diagnostics.training_trajectories, 2048);
        assert_eq!(diagnostics.valuation_trajectories, 4096);
        assert_eq!(diagnostics.exercise_counts.iter().sum::<usize>(), 4096);
        assert_eq!(diagnostics.regression_diagnostics.len(), 3);
    }

    #[test]
    fn american_rqmc_uses_between_scramble_uncertainty_and_replays() {
        let request = american_rqmc_request(OptionSide::Put, 100.0, 0x1111, 0x2222);
        let serial = price_monte_carlo(&request, policy(1)).expect("serial American RQMC price");
        let parallel =
            price_monte_carlo(&request, policy(4)).expect("parallel American RQMC price");
        assert_eq!(serial.pricing_result, parallel.pricing_result);
        assert_eq!(
            serial.early_exercise_diagnostics,
            parallel.early_exercise_diagnostics
        );
        assert_eq!(serial.independent_sampling_units, 16);
        assert_eq!(serial.evaluated_paths, 131_072);
        assert_eq!(serial.diagnostics.scramble_count, Some(16));
        assert_eq!(
            serial.estimator_variance.sqrt().to_bits(),
            serial.pricing_result.value.standard_error().get().to_bits()
        );
        let diagnostics = serial
            .early_exercise_diagnostics
            .expect("early-exercise diagnostics");
        assert_eq!(
            diagnostics.training_random_domain,
            RandomDomain::RqmcScramble
        );
        assert_eq!(
            diagnostics.valuation_random_domain,
            RandomDomain::RqmcScramble
        );
        assert_eq!(diagnostics.training_sampling_units, 8);
        assert_eq!(diagnostics.training_trajectories, 32_768);
        assert_eq!(diagnostics.valuation_sampling_units, 16);
        assert_eq!(diagnostics.valuation_trajectories, 131_072);
        assert_eq!(
            diagnostics.training_direction_checksum,
            diagnostics.valuation_direction_checksum
        );
        assert_ne!(
            diagnostics.training_scramble_checksum,
            diagnostics.valuation_scramble_checksum
        );
        assert_eq!(diagnostics.exercise_counts.iter().sum::<usize>(), 131_072);
    }

    #[test]
    fn american_rqmc_seed_changes_policy_and_valuation_replicates() {
        let first = price_monte_carlo(
            &american_rqmc_request(OptionSide::Put, 100.0, 0x1111, 0x2222),
            policy(2),
        )
        .expect("first American RQMC price");
        let changed = price_monte_carlo(
            &american_rqmc_request(OptionSide::Put, 100.0, 0x3333, 0x4444),
            policy(2),
        )
        .expect("changed American RQMC price");
        assert_ne!(
            first.pricing_result.value.value().get().to_bits(),
            changed.pricing_result.value.value().get().to_bits()
        );
        assert_ne!(
            first
                .early_exercise_diagnostics
                .expect("first diagnostics")
                .policy_fingerprint,
            changed
                .early_exercise_diagnostics
                .expect("changed diagnostics")
                .policy_fingerprint
        );
    }

    #[test]
    fn local_vol_american_pseudo_replays_and_matches_constant_volatility() {
        let black_scholes = american_request(OptionSide::Put, 100.0, 0.2, 4096, 8192, true);
        let local_volatility = with_constant_local_volatility(&black_scholes);
        let serial =
            price_monte_carlo(&local_volatility, policy(1)).expect("serial Local Vol American");
        let parallel =
            price_monte_carlo(&local_volatility, policy(4)).expect("parallel Local Vol American");
        assert_eq!(serial.pricing_result, parallel.pricing_result);
        assert_eq!(
            serial.early_exercise_diagnostics,
            parallel.early_exercise_diagnostics
        );
        let constant =
            price_monte_carlo(&black_scholes, policy(4)).expect("constant-volatility American");
        let local_estimate = serial.pricing_result.value;
        let constant_estimate = constant.pricing_result.value;
        let combined_error = local_estimate
            .standard_error()
            .get()
            .hypot(constant_estimate.standard_error().get());
        assert!(
            (local_estimate.value().get() - constant_estimate.value().get()).abs()
                <= 6.0 * combined_error,
            "Local Vol={}, constant vol={}, combined_se={combined_error}",
            local_estimate.value().get(),
            constant_estimate.value().get(),
        );
        let diagnostics = serial
            .early_exercise_diagnostics
            .expect("early-exercise diagnostics");
        assert_eq!(diagnostics.training_random_domain, RandomDomain::LsmTrain);
        assert_eq!(diagnostics.valuation_random_domain, RandomDomain::Valuation);
        assert_eq!(diagnostics.training_trajectories, 8192);
        assert_eq!(diagnostics.valuation_trajectories, 16_384);
        assert_eq!(diagnostics.exercise_counts.iter().sum::<usize>(), 16_384);
    }

    #[test]
    fn local_vol_american_rqmc_replays_with_brownian_bridge() {
        let request = with_constant_local_volatility(&american_rqmc_request(
            OptionSide::Put,
            100.0,
            0x5555,
            0x6666,
        ));
        let serial =
            price_monte_carlo(&request, policy(1)).expect("serial Local Vol RQMC American");
        let parallel =
            price_monte_carlo(&request, policy(4)).expect("parallel Local Vol RQMC American");
        assert_eq!(serial.pricing_result, parallel.pricing_result);
        assert_eq!(
            serial.early_exercise_diagnostics,
            parallel.early_exercise_diagnostics
        );
        let diagnostics = serial
            .early_exercise_diagnostics
            .expect("early-exercise diagnostics");
        assert_eq!(diagnostics.training_sampling_units, 8);
        assert_eq!(diagnostics.training_trajectories, 32_768);
        assert_eq!(diagnostics.valuation_sampling_units, 16);
        assert_eq!(diagnostics.valuation_trajectories, 131_072);
        assert_eq!(
            diagnostics.training_direction_checksum,
            diagnostics.valuation_direction_checksum
        );
        assert_ne!(
            diagnostics.training_scramble_checksum,
            diagnostics.valuation_scramble_checksum
        );
        assert_eq!(diagnostics.exercise_counts.iter().sum::<usize>(), 131_072);
    }

    #[test]
    fn american_fixed_policy_risks_replay_and_preserve_stopping_indices() {
        let price_request = american_request(OptionSide::Put, 100.0, 0.2, 4096, 8192, true);
        let risk_request = with_risk(&price_request, all_risks());
        let price_only = price_monte_carlo(&price_request, policy(4)).expect("price only");
        let serial = price_monte_carlo(&risk_request, policy(1)).expect("serial fixed-policy risk");
        let parallel =
            price_monte_carlo(&risk_request, policy(4)).expect("parallel fixed-policy risk");
        assert_eq!(price_only.pricing_result.value, serial.pricing_result.value);
        assert_eq!(serial.pricing_result, parallel.pricing_result);
        assert_eq!(serial.risk_diagnostics, parallel.risk_diagnostics);
        assert_eq!(
            price_only
                .early_exercise_diagnostics
                .as_ref()
                .expect("price diagnostics")
                .stopping_indices,
            serial
                .early_exercise_diagnostics
                .as_ref()
                .expect("risk diagnostics")
                .stopping_indices
        );
        let risks = &serial.pricing_result.risks;
        for estimate in [
            risks.delta.expect("delta").raw(),
            risks.gamma.expect("gamma").raw(),
            risks.vega.expect("vega").raw(),
        ] {
            assert!(estimate.value().get().is_finite());
            assert!(estimate.standard_error().get().is_finite());
        }
        assert_eq!(
            serial.risk_diagnostics.methods.delta,
            Some(RiskMethod::AadReverse)
        );
        assert_eq!(
            serial.risk_diagnostics.methods.gamma,
            Some(RiskMethod::CentralBumpOfAadDelta)
        );
        assert_eq!(
            serial.risk_diagnostics.methods.vega,
            Some(RiskMethod::AadReverse)
        );
        assert_eq!(
            serial.risk_diagnostics.methods.exercise_strategy,
            Some(ExerciseStrategyRisk::FixedExerciseStrategy)
        );
        assert_eq!(
            serial.risk_diagnostics.methods.stopping_indices,
            Some(StoppingIndexRisk::FrozenStoppingIndices)
        );
        assert_eq!(
            serial.risk_diagnostics.methods.exercise_policy_fingerprint,
            Some(
                serial
                    .early_exercise_diagnostics
                    .as_ref()
                    .expect("early-exercise diagnostics")
                    .policy_fingerprint
            )
        );
        for validation in [
            serial
                .risk_diagnostics
                .delta_validation
                .expect("delta validation"),
            serial
                .risk_diagnostics
                .gamma_validation
                .expect("gamma validation"),
            serial
                .risk_diagnostics
                .vega_validation
                .expect("vega validation"),
        ] {
            assert!(validation.bump_and_revalue.value().get().is_finite());
            assert!(validation.bump_minus_primary.value().get().is_finite());
        }
        assert!(
            serial
                .risk_diagnostics
                .delta_validation
                .expect("delta validation")
                .bump_minus_primary
                .value()
                .get()
                .abs()
                < 5.0e-3
        );
        assert!(
            serial
                .risk_diagnostics
                .vega_validation
                .expect("vega validation")
                .bump_minus_primary
                .value()
                .get()
                .abs()
                < 5.0e-2
        );
    }

    #[test]
    fn local_vol_american_pseudo_vega_kt_replays_with_frozen_stopping_indices() {
        let base = american_request(OptionSide::Put, 100.0, 0.2, 256, 512, true);
        let price_request = with_model(&base, constant_local_vol_model_with_reporting_basis());
        let risk_request = with_risk(&price_request, american_vega_kt_risk(true));
        let price_only = price_monte_carlo(&price_request, policy(2)).expect("price only");
        let serial = price_monte_carlo(&risk_request, policy(1)).expect("serial VegaKT");
        let parallel = price_monte_carlo(&risk_request, policy(4)).expect("parallel VegaKT");

        assert_eq!(price_only.pricing_result.value, serial.pricing_result.value);
        assert_eq!(serial.pricing_result, parallel.pricing_result);
        assert_eq!(serial.risk_diagnostics, parallel.risk_diagnostics);
        assert_eq!(
            price_only
                .early_exercise_diagnostics
                .as_ref()
                .expect("price diagnostics")
                .stopping_indices,
            serial
                .early_exercise_diagnostics
                .as_ref()
                .expect("risk diagnostics")
                .stopping_indices
        );
        let report = serial.pricing_result.risks.vega_kt.expect("VegaKT");
        assert_eq!(report.coordinates().len(), 6);
        assert_eq!(report.estimates().len(), 6);
        assert_eq!(report.raw_buckets().len(), 6);
        assert_eq!(
            report.full_bucket_covariance().expect("covariance").len(),
            36
        );
        assert_eq!(
            report.projection().pre_projection().get().to_bits(),
            serial
                .pricing_result
                .risks
                .vega
                .expect("scalar Vega")
                .raw()
                .value()
                .get()
                .to_bits()
        );
        assert_eq!(
            serial.risk_diagnostics.methods.exercise_strategy,
            Some(ExerciseStrategyRisk::FixedExerciseStrategy)
        );
        assert_eq!(
            serial.risk_diagnostics.methods.stopping_indices,
            Some(StoppingIndexRisk::FrozenStoppingIndices)
        );
        assert_eq!(
            serial.risk_diagnostics.methods.exercise_policy_fingerprint,
            Some(
                serial
                    .early_exercise_diagnostics
                    .as_ref()
                    .expect("early-exercise diagnostics")
                    .policy_fingerprint
            )
        );
    }

    #[test]
    fn local_vol_american_rqmc_vega_kt_uses_scramble_uncertainty() {
        let base = small_american_rqmc_request();
        let price_request = with_model(&base, constant_local_vol_model_with_reporting_basis());
        let risk_request = with_risk(&price_request, american_vega_kt_risk(false));
        let price_only = price_monte_carlo(&price_request, policy(2)).expect("price only");
        let result = price_monte_carlo(&risk_request, policy(3)).expect("RQMC VegaKT");

        assert_eq!(price_only.pricing_result.value, result.pricing_result.value);
        assert_eq!(result.independent_sampling_units, 8);
        assert_eq!(
            price_only
                .early_exercise_diagnostics
                .as_ref()
                .expect("price diagnostics")
                .stopping_indices,
            result
                .early_exercise_diagnostics
                .as_ref()
                .expect("risk diagnostics")
                .stopping_indices
        );
        let report = result.pricing_result.risks.vega_kt.expect("VegaKT");
        assert_eq!(report.estimates().len(), 6);
        assert!(report.full_bucket_covariance().is_none());
        assert!(
            report
                .estimates()
                .iter()
                .all(|estimate| estimate.sample_variance().is_some())
        );
        assert_eq!(
            result.risk_diagnostics.methods.exercise_strategy,
            Some(ExerciseStrategyRisk::FixedExerciseStrategy)
        );
        assert_eq!(
            result.risk_diagnostics.methods.stopping_indices,
            Some(StoppingIndexRisk::FrozenStoppingIndices)
        );
    }

    #[test]
    fn american_rqmc_fixed_policy_risks_use_between_scramble_uncertainty() {
        let price_request = american_rqmc_request(OptionSide::Put, 100.0, 0x7777, 0x8888);
        let risk_request = with_risk(&price_request, all_risks());
        let price_only = price_monte_carlo(&price_request, policy(4)).expect("RQMC price only");
        let serial = price_monte_carlo(&risk_request, policy(1)).expect("serial RQMC risk");
        let parallel = price_monte_carlo(&risk_request, policy(4)).expect("parallel RQMC risk");
        assert_eq!(price_only.pricing_result.value, serial.pricing_result.value);
        assert_eq!(serial.pricing_result, parallel.pricing_result);
        assert_eq!(serial.risk_diagnostics, parallel.risk_diagnostics);
        assert_eq!(
            price_only
                .early_exercise_diagnostics
                .as_ref()
                .expect("price diagnostics")
                .stopping_indices,
            serial
                .early_exercise_diagnostics
                .as_ref()
                .expect("risk diagnostics")
                .stopping_indices
        );
        for estimate in [
            serial.pricing_result.risks.delta.expect("delta").raw(),
            serial.pricing_result.risks.gamma.expect("gamma").raw(),
            serial.pricing_result.risks.vega.expect("vega").raw(),
        ] {
            assert_eq!(
                estimate.estimator(),
                EstimatorKind::RandomizedQuasiMonteCarlo
            );
            assert_eq!(estimate.effective_sampling_units().get(), 16);
            assert!(estimate.standard_error().get().is_finite());
        }
        assert_eq!(
            serial.risk_diagnostics.methods.exercise_strategy,
            Some(ExerciseStrategyRisk::FixedExerciseStrategy)
        );
        assert_eq!(
            serial.risk_diagnostics.methods.stopping_indices,
            Some(StoppingIndexRisk::FrozenStoppingIndices)
        );
    }

    #[test]
    fn local_vol_american_fixed_policy_risks_replay_and_match_constant_variance() {
        let constant_price = american_request(OptionSide::Put, 100.0, 0.2, 4096, 8192, true);
        let constant_risk = with_risk(&constant_price, all_risks());
        let local_price = with_constant_local_volatility(&constant_price);
        let local_risk = with_risk(&local_price, all_risks());
        let price_only = price_monte_carlo(&local_price, policy(4)).expect("Local Vol price only");
        let serial = price_monte_carlo(&local_risk, policy(1)).expect("serial Local Vol risk");
        let parallel = price_monte_carlo(&local_risk, policy(4)).expect("parallel Local Vol risk");
        let constant =
            price_monte_carlo(&constant_risk, policy(4)).expect("constant-variance risk");
        assert_eq!(price_only.pricing_result.value, serial.pricing_result.value);
        assert_eq!(serial.pricing_result, parallel.pricing_result);
        assert_eq!(serial.risk_diagnostics, parallel.risk_diagnostics);
        assert_eq!(
            price_only
                .early_exercise_diagnostics
                .as_ref()
                .expect("price diagnostics")
                .stopping_indices,
            serial
                .early_exercise_diagnostics
                .as_ref()
                .expect("risk diagnostics")
                .stopping_indices
        );
        let local_risks = &serial.pricing_result.risks;
        let constant_risks = &constant.pricing_result.risks;
        for (local, constant) in [
            (
                local_risks.delta.expect("Local Vol delta").raw(),
                constant_risks.delta.expect("constant delta").raw(),
            ),
            (
                local_risks.gamma.expect("Local Vol gamma").raw(),
                constant_risks.gamma.expect("constant gamma").raw(),
            ),
            (
                local_risks.vega.expect("Local Vol vega").raw(),
                constant_risks.vega.expect("constant vega").raw(),
            ),
        ] {
            let combined_error = local
                .standard_error()
                .get()
                .hypot(constant.standard_error().get());
            assert!(
                (local.value().get() - constant.value().get()).abs()
                    <= 8.0 * combined_error + 5.0e-3,
                "Local Vol={}, constant={}, combined_se={combined_error}",
                local.value().get(),
                constant.value().get(),
            );
        }
        assert_eq!(
            serial.risk_diagnostics.methods.delta,
            Some(RiskMethod::CentralBump)
        );
        assert_eq!(
            serial.risk_diagnostics.methods.gamma,
            Some(RiskMethod::CentralBump)
        );
        assert_eq!(
            serial.risk_diagnostics.methods.vega,
            Some(RiskMethod::AadReverse)
        );
    }

    #[test]
    fn local_vol_american_rqmc_fixed_policy_risks_replay() {
        let price_request = with_constant_local_volatility(&american_rqmc_request(
            OptionSide::Put,
            100.0,
            0x9999,
            0xaaaa,
        ));
        let risk_request = with_risk(&price_request, all_risks());
        let price_only = price_monte_carlo(&price_request, policy(4)).expect("RQMC price only");
        let serial = price_monte_carlo(&risk_request, policy(1)).expect("serial RQMC risk");
        let parallel = price_monte_carlo(&risk_request, policy(4)).expect("parallel RQMC risk");
        assert_eq!(price_only.pricing_result.value, serial.pricing_result.value);
        assert_eq!(serial.pricing_result, parallel.pricing_result);
        assert_eq!(serial.risk_diagnostics, parallel.risk_diagnostics);
        assert_eq!(
            price_only
                .early_exercise_diagnostics
                .as_ref()
                .expect("price diagnostics")
                .stopping_indices,
            serial
                .early_exercise_diagnostics
                .as_ref()
                .expect("risk diagnostics")
                .stopping_indices
        );
        for estimate in [
            serial.pricing_result.risks.delta.expect("delta").raw(),
            serial.pricing_result.risks.gamma.expect("gamma").raw(),
            serial.pricing_result.risks.vega.expect("vega").raw(),
        ] {
            assert_eq!(
                estimate.estimator(),
                EstimatorKind::RandomizedQuasiMonteCarlo
            );
            assert_eq!(estimate.effective_sampling_units().get(), 16);
            assert!(estimate.standard_error().get().is_finite());
        }
    }

    #[test]
    fn no_dividend_american_call_agrees_with_european_call() {
        let american = american_request(OptionSide::Call, 100.0, 0.2, 16_384, 32_768, true);
        let ProductSpec::AmericanVanilla(american_product) = american.product() else {
            unreachable!("helper constructs an American option");
        };
        let european = PricingRequest::new(
            american.valuation_date(),
            ProductSpec::EuropeanVanilla(
                EuropeanVanillaSpec::new(
                    american_product.underlying(),
                    american_product.currency(),
                    american_product.expiry(),
                    american_product.strike().get(),
                    american_product.notional().get(),
                    american_product.side(),
                )
                .expect("European product"),
            ),
            american.market().clone(),
            american.model().clone(),
            american.engine(),
            american.risk().clone(),
        )
        .expect("European request");
        let american_result = price_monte_carlo(&american, policy(4)).expect("American price");
        let european_result = price_monte_carlo(&european, policy(4)).expect("European price");
        let american_estimate = american_result.pricing_result.value;
        let european_estimate = european_result.pricing_result.value;
        let combined_error = american_estimate
            .standard_error()
            .get()
            .hypot(european_estimate.standard_error().get());
        assert!(
            (american_estimate.value().get() - european_estimate.value().get()).abs()
                <= 6.0 * combined_error,
            "American={}, European={}, combined_se={combined_error}",
            american_estimate.value().get(),
            european_estimate.value().get(),
        );
    }

    #[test]
    fn american_put_respects_intrinsic_and_european_lower_bounds() {
        let american = american_request(OptionSide::Put, 105.0, 0.2, 16_384, 32_768, true);
        let ProductSpec::AmericanVanilla(american_product) = american.product() else {
            unreachable!("helper constructs an American option");
        };
        let european = PricingRequest::new(
            american.valuation_date(),
            ProductSpec::EuropeanVanilla(
                EuropeanVanillaSpec::new(
                    american_product.underlying(),
                    american_product.currency(),
                    american_product.expiry(),
                    american_product.strike().get(),
                    american_product.notional().get(),
                    american_product.side(),
                )
                .expect("European product"),
            ),
            american.market().clone(),
            american.model().clone(),
            american.engine(),
            american.risk().clone(),
        )
        .expect("European request");
        let european_value = black_scholes_oracle(&european)
            .expect("European oracle")
            .price;
        let result = price_monte_carlo(&american, policy(4)).expect("American price");
        let estimate = result.pricing_result.value;
        let lower_bound = european_value.max(5.0);
        assert!(
            estimate.value().get() + 6.0 * estimate.standard_error().get() >= lower_bound,
            "American={}, se={}, lower_bound={lower_bound}",
            estimate.value().get(),
            estimate.standard_error().get(),
        );
    }

    #[test]
    fn constant_vol_paths_separate_training_and_valuation_domains() {
        let plan =
            SimulationPlan::compile(&request(OptionSide::Put, 100.0, 0.2, 2, false), policy(1))
                .expect("plan");
        let generator = Philox4x32::from_seed(0x0123_4567_89ab_cdef);
        let valuation = plan.normals(&generator, 0, RandomDomain::Valuation);
        let training = plan.normals(&generator, 0, RandomDomain::LsmTrain);
        assert_ne!(valuation, training);
        assert_eq!(
            valuation,
            plan.normals(&generator, 0, RandomDomain::Valuation)
        );
        assert_eq!(
            training,
            plan.normals(&generator, 0, RandomDomain::LsmTrain)
        );
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

    fn rqmc_request(
        scramble_seed: u64,
        points: u64,
        scrambles: u32,
        antithetic: bool,
        risk: RiskRequest,
    ) -> PricingRequest {
        let underlying = UnderlyingId::new(1);
        let currency = CurrencyId::new(1);
        let product = ProductSpec::EuropeanVanilla(
            EuropeanVanillaSpec::new(
                underlying,
                currency,
                "2027-09-04".parse().expect("expiry"),
                100.0,
                1.0,
                OptionSide::Call,
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
        let model = ModelSpec::BlackScholes(BlackScholesSpec::new(0.2).expect("model"));
        let engine = EngineConfig::RandomizedQuasiMonteCarlo(
            RqmcConfig::new(
                points,
                scrambles,
                scramble_seed,
                VarianceReduction::new(antithetic, true),
            )
            .expect("RQMC engine"),
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

    fn policy(workers: u32) -> ExecutionPolicy {
        ExecutionPolicy::new(workers, Some(1024)).expect("policy")
    }

    fn local_vol_price_only_request(
        sampling_units: u64,
        antithetic: bool,
        risk: RiskRequest,
    ) -> PricingRequest {
        price_only_request_with_model(
            ModelSpec::LocalVolatility(
                LocalVolatilitySpec::from_explicit_grid(
                    vec![0.0, 1.0],
                    vec![-1.0, 1.0],
                    vec![0.04, 0.04, 0.04, 0.04],
                    1.0e-8,
                    1.0,
                )
                .expect("local volatility"),
            ),
            sampling_units,
            antithetic,
            risk,
        )
    }

    fn zero_carry_black_scholes_request(sampling_units: u64, antithetic: bool) -> PricingRequest {
        price_only_request_with_model(
            ModelSpec::BlackScholes(BlackScholesSpec::new(0.2).expect("model")),
            sampling_units,
            antithetic,
            RiskRequest::price_only(SmileDynamics::StickyLogMoneyness),
        )
    }

    fn zero_carry_black_76_request(sampling_units: u64, antithetic: bool) -> PricingRequest {
        price_only_request_with_model(
            ModelSpec::Black76(Black76Spec::new(0.2).expect("model")),
            sampling_units,
            antithetic,
            RiskRequest::price_only(SmileDynamics::StickyLogMoneyness),
        )
    }

    fn digital_zero_vol_request(
        side: OptionSide,
        strike: f64,
        payout_kind: DigitalPayout,
    ) -> PricingRequest {
        let underlying = UnderlyingId::new(1);
        let currency = CurrencyId::new(1);
        let product = ProductSpec::Digital(
            DigitalSpec::new(
                underlying,
                currency,
                "2027-09-04".parse().expect("expiry"),
                strike,
                10.0,
                side,
                payout_kind,
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
        let model = ModelSpec::BlackScholes(BlackScholesSpec::new(0.0).expect("model"));
        let engine = EngineConfig::PseudoMonteCarlo(
            PseudoMcConfig::new(
                0x0123_4567_89ab_cdef,
                1,
                VarianceReduction::new(false, false),
            )
            .expect("engine"),
        );
        PricingRequest::new(
            "2026-09-04".parse().expect("valuation"),
            product,
            market,
            model,
            engine,
            RiskRequest::price_only(SmileDynamics::StickyLogMoneyness),
        )
        .expect("request")
    }

    fn barrier_zero_vol_request(barrier: f64) -> PricingRequest {
        barrier_zero_vol_request_with_monitoring(barrier, BarrierMonitoring::Discrete)
    }

    fn barrier_zero_vol_request_with_monitoring(
        barrier: f64,
        monitoring: BarrierMonitoring,
    ) -> PricingRequest {
        let underlying = UnderlyingId::new(1);
        let currency = CurrencyId::new(1);
        let product = ProductSpec::Barrier(
            BarrierSpec::new(
                underlying,
                currency,
                "2027-09-04".parse().expect("expiry"),
                100.0,
                barrier,
                2.0,
                OptionSide::Call,
                BarrierDirection::Up,
                BarrierStyle::KnockOut,
                monitoring,
                vec![
                    "2027-03-05".parse().expect("first"),
                    "2027-09-04".parse().expect("second"),
                ],
                None,
                "2027-09-04".parse().expect("payment"),
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
        let model = ModelSpec::BlackScholes(BlackScholesSpec::new(0.0).expect("model"));
        let engine = EngineConfig::PseudoMonteCarlo(
            PseudoMcConfig::new(
                0x0123_4567_89ab_cdef,
                1,
                VarianceReduction::new(false, false),
            )
            .expect("engine"),
        );
        PricingRequest::new(
            "2026-09-04".parse().expect("valuation"),
            product,
            market,
            model,
            engine,
            RiskRequest::price_only(SmileDynamics::StickyLogMoneyness),
        )
        .expect("request")
    }

    fn barrier_dividend_jump_request(
        direction: BarrierDirection,
        style: BarrierStyle,
        volatility: f64,
        risk: RiskRequest,
    ) -> PricingRequest {
        barrier_dividend_jump_request_with_monitoring(
            direction,
            style,
            volatility,
            risk,
            BarrierMonitoring::Discrete,
            None,
        )
    }

    fn barrier_dividend_jump_request_with_monitoring(
        direction: BarrierDirection,
        style: BarrierStyle,
        volatility: f64,
        risk: RiskRequest,
        monitoring: BarrierMonitoring,
        barrier_override: Option<f64>,
    ) -> PricingRequest {
        let underlying = UnderlyingId::new(1);
        let currency = CurrencyId::new(1);
        let valuation_date: Date = "2026-09-04".parse().expect("valuation");
        let dividend_date: Date = "2027-03-05".parse().expect("dividend date");
        let expiry: Date = "2027-09-04".parse().expect("expiry");
        let barrier = barrier_override.unwrap_or(match direction {
            BarrierDirection::Up => 95.0,
            BarrierDirection::Down => 90.0,
        });
        let monitoring_dates = match monitoring {
            BarrierMonitoring::Discrete => vec![dividend_date, expiry],
            BarrierMonitoring::Continuous => vec![expiry],
        };
        let product = ProductSpec::Barrier(
            BarrierSpec::new(
                underlying,
                currency,
                expiry,
                80.0,
                barrier,
                1.0,
                OptionSide::Call,
                direction,
                style,
                monitoring,
                monitoring_dates,
                None,
                expiry,
            )
            .expect("product"),
        );
        let ex_time = DayCountConvention::Act365F.year_fraction(valuation_date, dividend_date);
        let event = EventId::new(1);
        let dividend_quote = match monitoring {
            BarrierMonitoring::Discrete => {
                DividendQuote::fixed_cash(15.0, event).expect("cash dividend")
            }
            BarrierMonitoring::Continuous => {
                DividendQuote::fixed_cash_and_proportional(15.0, 0.1, event)
                    .expect("affine dividend")
            }
        };
        let forward = EquityForward::with_discrete_dividends(
            underlying,
            PositiveF64::new(100.0, "spot").expect("spot"),
            curve(1, 0.0),
            curve(2, 0.0),
            vec![DividendEvent::new(event, ex_time, dividend_quote).expect("dividend")],
        )
        .expect("forward");
        let market = MarketContext::Equity(EquityMarket::new(currency, forward));
        let sampling_units = if volatility == 0.0 { 1 } else { 4_096 };
        PricingRequest::new(
            valuation_date,
            product,
            market,
            ModelSpec::BlackScholes(BlackScholesSpec::new(volatility).expect("model")),
            EngineConfig::PseudoMonteCarlo(
                PseudoMcConfig::new(
                    0x0123_4567_89ab_cdef,
                    sampling_units,
                    VarianceReduction::new(volatility != 0.0, false),
                )
                .expect("engine"),
            ),
            risk,
        )
        .expect("request")
    }

    fn continuous_barrier_conformance_request(
        direction: BarrierDirection,
        style: BarrierStyle,
        rebate: Option<f64>,
        engine: EngineConfig,
        risk: RiskRequest,
    ) -> PricingRequest {
        let barrier = match direction {
            BarrierDirection::Up => 130.0,
            BarrierDirection::Down => 70.0,
        };
        let base = barrier_dividend_jump_request_with_monitoring(
            direction,
            style,
            0.2,
            RiskRequest::price_only(SmileDynamics::StickyLogMoneyness),
            BarrierMonitoring::Continuous,
            Some(barrier),
        );
        let source = match base.product() {
            ProductSpec::Barrier(source) => source,
            _ => unreachable!("helper constructs a Barrier"),
        };
        let product = ProductSpec::Barrier(
            BarrierSpec::new(
                source.underlying(),
                source.currency(),
                source.expiry(),
                source.strike().get(),
                source.barrier().get(),
                source.notional().get(),
                source.side(),
                source.direction(),
                source.style(),
                source.monitoring(),
                source.monitoring_dates().to_vec(),
                rebate,
                source.payment_date(),
            )
            .expect("continuous Barrier product"),
        );
        PricingRequest::new(
            base.valuation_date(),
            product,
            base.market().clone(),
            base.model().clone(),
            engine,
            risk,
        )
        .expect("continuous Barrier conformance request")
    }

    fn asian_zero_vol_request() -> PricingRequest {
        let underlying = UnderlyingId::new(1);
        let currency = CurrencyId::new(1);
        let product = ProductSpec::ArithmeticAsian(
            ArithmeticAsianSpec::new(
                underlying,
                currency,
                100.0,
                2.0,
                OptionSide::Call,
                vec![
                    AsianObservation::unknown("2027-03-05".parse().expect("first"), 0.25)
                        .expect("first"),
                    AsianObservation::unknown("2027-09-04".parse().expect("second"), 0.75)
                        .expect("second"),
                ],
                "2027-09-04".parse().expect("payment"),
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
        let model = ModelSpec::BlackScholes(BlackScholesSpec::new(0.0).expect("model"));
        let engine = EngineConfig::PseudoMonteCarlo(
            PseudoMcConfig::new(
                0x0123_4567_89ab_cdef,
                1,
                VarianceReduction::new(false, false),
            )
            .expect("engine"),
        );
        PricingRequest::new(
            "2026-09-04".parse().expect("valuation"),
            product,
            market,
            model,
            engine,
            RiskRequest::price_only(SmileDynamics::StickyLogMoneyness),
        )
        .expect("request")
    }

    fn asian_risk_request(risk: RiskRequest) -> PricingRequest {
        let underlying = UnderlyingId::new(1);
        let currency = CurrencyId::new(1);
        let product = ProductSpec::ArithmeticAsian(
            ArithmeticAsianSpec::new(
                underlying,
                currency,
                100.0,
                1.5,
                OptionSide::Call,
                vec![
                    AsianObservation::unknown("2027-03-05".parse().expect("first"), 0.25)
                        .expect("first"),
                    AsianObservation::unknown("2027-09-04".parse().expect("second"), 0.75)
                        .expect("second"),
                ],
                "2027-09-04".parse().expect("payment"),
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
        let model = ModelSpec::BlackScholes(BlackScholesSpec::new(0.25).expect("model"));
        let engine = EngineConfig::PseudoMonteCarlo(
            PseudoMcConfig::new(
                0x0123_4567_89ab_cdef,
                16_384,
                VarianceReduction::new(true, false),
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

    fn asian_rqmc_risk_request() -> PricingRequest {
        let underlying = UnderlyingId::new(1);
        let currency = CurrencyId::new(1);
        let product = ProductSpec::ArithmeticAsian(
            ArithmeticAsianSpec::new(
                underlying,
                currency,
                100.0,
                1.5,
                OptionSide::Call,
                vec![
                    AsianObservation::unknown("2027-03-05".parse().expect("first"), 0.25)
                        .expect("first"),
                    AsianObservation::unknown("2027-09-04".parse().expect("second"), 0.75)
                        .expect("second"),
                ],
                "2027-09-04".parse().expect("payment"),
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
        let model = ModelSpec::BlackScholes(BlackScholesSpec::new(0.25).expect("model"));
        let engine = EngineConfig::RandomizedQuasiMonteCarlo(
            RqmcConfig::new(
                256,
                8,
                0xfedc_ba98_7654_3210,
                VarianceReduction::new(true, true),
            )
            .expect("RQMC engine"),
        );
        PricingRequest::new(
            "2026-09-04".parse().expect("valuation"),
            product,
            market,
            model,
            engine,
            all_risks(),
        )
        .expect("request")
    }

    fn lookback_zero_vol_request() -> PricingRequest {
        let underlying = UnderlyingId::new(1);
        let currency = CurrencyId::new(1);
        let product = ProductSpec::FixedLookback(
            FixedLookbackSpec::new(
                underlying,
                currency,
                100.0,
                2.0,
                OptionSide::Call,
                vec![
                    "2027-03-05".parse().expect("first"),
                    "2027-09-04".parse().expect("second"),
                ],
                None,
                "2027-09-04".parse().expect("payment"),
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
        let model = ModelSpec::BlackScholes(BlackScholesSpec::new(0.0).expect("model"));
        let engine = EngineConfig::PseudoMonteCarlo(
            PseudoMcConfig::new(
                0x0123_4567_89ab_cdef,
                1,
                VarianceReduction::new(false, false),
            )
            .expect("engine"),
        );
        PricingRequest::new(
            "2026-09-04".parse().expect("valuation"),
            product,
            market,
            model,
            engine,
            RiskRequest::price_only(SmileDynamics::StickyLogMoneyness),
        )
        .expect("request")
    }

    fn lookback_risk_request() -> PricingRequest {
        let underlying = UnderlyingId::new(1);
        let currency = CurrencyId::new(1);
        let product = ProductSpec::FixedLookback(
            FixedLookbackSpec::new(
                underlying,
                currency,
                100.0,
                1.5,
                OptionSide::Call,
                vec![
                    "2027-03-05".parse().expect("first"),
                    "2027-09-04".parse().expect("second"),
                ],
                None,
                "2027-09-04".parse().expect("payment"),
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
        let model = ModelSpec::BlackScholes(BlackScholesSpec::new(0.25).expect("model"));
        let engine = EngineConfig::PseudoMonteCarlo(
            PseudoMcConfig::new(
                0x0123_4567_89ab_cdef,
                16_384,
                VarianceReduction::new(true, false),
            )
            .expect("engine"),
        );
        PricingRequest::new(
            "2026-09-04".parse().expect("valuation"),
            product,
            market,
            model,
            engine,
            all_risks(),
        )
        .expect("request")
    }

    fn partially_fixed_asian_request(fixing: f64, model: ModelSpec) -> PricingRequest {
        let underlying = UnderlyingId::new(1);
        let currency = CurrencyId::new(1);
        let expiry: Date = "2027-09-04".parse().expect("expiry");
        let product = ProductSpec::ArithmeticAsian(
            ArithmeticAsianSpec::new(
                underlying,
                currency,
                1.0,
                1.0,
                OptionSide::Call,
                vec![
                    AsianObservation::known("2026-06-04".parse().expect("known date"), 0.4, fixing)
                        .expect("known fixing"),
                    AsianObservation::unknown(expiry, 0.6).expect("unknown fixing"),
                ],
                expiry,
            )
            .expect("Asian"),
        );
        let market = MarketContext::Equity(EquityMarket::new(
            currency,
            EquityForward::new(
                underlying,
                PositiveF64::new(100.0, "spot").expect("spot"),
                curve(1, 0.0),
                curve(2, 0.0),
            ),
        ));
        PricingRequest::new(
            "2026-09-04".parse().expect("valuation"),
            product,
            market,
            model,
            EngineConfig::PseudoMonteCarlo(
                PseudoMcConfig::new(7, 4_096, VarianceReduction::new(true, false)).expect("engine"),
            ),
            all_risks(),
        )
        .expect("partially fixed Asian request")
    }

    fn partially_fixed_lookback_request(
        historical_extremum: f64,
        model: ModelSpec,
    ) -> PricingRequest {
        let underlying = UnderlyingId::new(1);
        let currency = CurrencyId::new(1);
        let expiry: Date = "2027-09-04".parse().expect("expiry");
        let product = ProductSpec::FixedLookback(
            FixedLookbackSpec::new(
                underlying,
                currency,
                200.0,
                1.0,
                OptionSide::Put,
                vec!["2026-06-04".parse().expect("past date"), expiry],
                Some(historical_extremum),
                expiry,
            )
            .expect("Lookback"),
        );
        let market = MarketContext::Equity(EquityMarket::new(
            currency,
            EquityForward::new(
                underlying,
                PositiveF64::new(100.0, "spot").expect("spot"),
                curve(1, 0.0),
                curve(2, 0.0),
            ),
        ));
        PricingRequest::new(
            "2026-09-04".parse().expect("valuation"),
            product,
            market,
            model,
            EngineConfig::PseudoMonteCarlo(
                PseudoMcConfig::new(7, 4_096, VarianceReduction::new(true, false)).expect("engine"),
            ),
            all_risks(),
        )
        .expect("partially fixed Lookback request")
    }

    fn fully_fixed_asian_request(engine: EngineConfig) -> PricingRequest {
        fully_fixed_asian_request_with_risk(
            engine,
            RiskRequest::price_only(SmileDynamics::StickyLogMoneyness),
        )
    }

    fn fully_fixed_asian_request_with_risk(
        engine: EngineConfig,
        risk: RiskRequest,
    ) -> PricingRequest {
        let underlying = UnderlyingId::new(1);
        let currency = CurrencyId::new(1);
        let product = ProductSpec::ArithmeticAsian(
            ArithmeticAsianSpec::new(
                underlying,
                currency,
                100.0,
                2.0,
                OptionSide::Call,
                vec![
                    AsianObservation::known("2026-03-04".parse().expect("first"), 0.25, 95.0)
                        .expect("first"),
                    AsianObservation::known("2026-06-04".parse().expect("second"), 0.75, 115.0)
                        .expect("second"),
                ],
                "2027-09-04".parse().expect("payment"),
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
        let model = ModelSpec::BlackScholes(BlackScholesSpec::new(0.2).expect("model"));
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

    fn fully_fixed_lookback_request(engine: EngineConfig) -> PricingRequest {
        fully_fixed_lookback_request_with_risk(
            engine,
            RiskRequest::price_only(SmileDynamics::StickyLogMoneyness),
        )
    }

    fn fully_fixed_lookback_request_with_risk(
        engine: EngineConfig,
        risk: RiskRequest,
    ) -> PricingRequest {
        let underlying = UnderlyingId::new(1);
        let currency = CurrencyId::new(1);
        let product = ProductSpec::FixedLookback(
            FixedLookbackSpec::new(
                underlying,
                currency,
                100.0,
                2.0,
                OptionSide::Call,
                vec![
                    "2026-03-04".parse().expect("first"),
                    "2026-06-04".parse().expect("second"),
                ],
                Some(120.0),
                "2027-09-04".parse().expect("payment"),
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
        let model = ModelSpec::BlackScholes(BlackScholesSpec::new(0.2).expect("model"));
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

    fn fixed_payoff_pseudo_engine() -> EngineConfig {
        EngineConfig::PseudoMonteCarlo(
            PseudoMcConfig::new(
                0x0123_4567_89ab_cdef,
                16,
                VarianceReduction::new(true, false),
            )
            .expect("engine"),
        )
    }

    fn fixed_payoff_rqmc_engine() -> EngineConfig {
        EngineConfig::RandomizedQuasiMonteCarlo(
            RqmcConfig::new(
                16,
                4,
                0xfedc_ba98_7654_3210,
                VarianceReduction::new(true, false),
            )
            .expect("RQMC engine"),
        )
    }

    fn zero_carry_rqmc_request(model: ModelSpec) -> PricingRequest {
        zero_carry_rqmc_request_with_risk(
            model,
            RiskRequest::price_only(SmileDynamics::StickyLogMoneyness),
        )
    }

    fn zero_carry_rqmc_request_with_risk(model: ModelSpec, risk: RiskRequest) -> PricingRequest {
        let underlying = UnderlyingId::new(1);
        let currency = CurrencyId::new(1);
        let product = ProductSpec::EuropeanVanilla(
            EuropeanVanillaSpec::new(
                underlying,
                currency,
                "2027-09-04".parse().expect("expiry"),
                100.0,
                1.0,
                OptionSide::Call,
            )
            .expect("product"),
        );
        let market = MarketContext::Equity(EquityMarket::new(
            currency,
            EquityForward::new(
                underlying,
                PositiveF64::new(100.0, "spot").expect("spot"),
                curve(1, 0.0),
                curve(2, 0.0),
            ),
        ));
        let engine = EngineConfig::RandomizedQuasiMonteCarlo(
            RqmcConfig::new(
                256,
                16,
                0xfedc_ba98_7654_3210,
                VarianceReduction::new(true, true),
            )
            .expect("RQMC engine"),
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

    fn constant_local_vol_model() -> ModelSpec {
        constant_local_vol_model_with_volatility(0.2)
    }

    fn constant_local_vol_model_with_volatility(volatility: f64) -> ModelSpec {
        let variance = volatility * volatility;
        ModelSpec::LocalVolatility(
            LocalVolatilitySpec::from_explicit_grid(
                vec![0.0, 1.0],
                vec![-1.0, 1.0],
                vec![variance; 4],
                1.0e-8,
                1.0,
            )
            .expect("local volatility"),
        )
    }

    fn skewed_local_vol_model(time_nodes: Vec<f64>) -> ModelSpec {
        skewed_local_vol_model_with_parallel_shift(time_nodes, 0.0)
    }

    fn skewed_local_vol_model_with_parallel_shift(
        time_nodes: Vec<f64>,
        volatility_shift: f64,
    ) -> ModelSpec {
        let mut values = Vec::with_capacity(time_nodes.len() * 3);
        for _ in &time_nodes {
            values.extend(
                [0.15_f64, 0.2, 0.25].map(|volatility| (volatility + volatility_shift).powi(2)),
            );
        }
        ModelSpec::LocalVolatility(
            LocalVolatilitySpec::from_explicit_grid(
                time_nodes,
                vec![-1.0, 0.0, 1.0],
                values,
                1.0e-8,
                1.0,
            )
            .expect("skewed local volatility"),
        )
    }

    fn continuous_barrier_local_vol_request(
        model: ModelSpec,
        monitoring_dates: Vec<Date>,
        risk: RiskRequest,
    ) -> PricingRequest {
        let underlying = UnderlyingId::new(1);
        let currency = CurrencyId::new(1);
        let expiry: Date = "2027-09-04".parse().expect("expiry");
        let product = ProductSpec::Barrier(
            BarrierSpec::new(
                underlying,
                currency,
                expiry,
                100.0,
                130.0,
                1.0,
                OptionSide::Call,
                BarrierDirection::Up,
                BarrierStyle::KnockOut,
                BarrierMonitoring::Continuous,
                monitoring_dates,
                None,
                expiry,
            )
            .expect("continuous Barrier"),
        );
        let market = MarketContext::Equity(EquityMarket::new(
            currency,
            EquityForward::new(
                underlying,
                PositiveF64::new(100.0, "spot").expect("spot"),
                curve(1, 0.0),
                curve(2, 0.0),
            ),
        ));
        PricingRequest::new(
            "2026-09-04".parse().expect("valuation"),
            product,
            market,
            model,
            EngineConfig::PseudoMonteCarlo(
                PseudoMcConfig::new(
                    0x0123_4567_89ab_cdef,
                    4_096,
                    VarianceReduction::new(true, true),
                )
                .expect("engine"),
            ),
            risk,
        )
        .expect("request")
    }

    fn local_vol_barrier_dividend_jump_request(
        style: BarrierStyle,
        risk: RiskRequest,
    ) -> PricingRequest {
        let base = barrier_dividend_jump_request(BarrierDirection::Up, style, 0.2, risk);
        PricingRequest::new(
            base.valuation_date(),
            base.product().clone(),
            base.market().clone(),
            constant_local_vol_model(),
            base.engine(),
            base.risk().clone(),
        )
        .expect("Local Volatility Barrier request")
    }

    fn constant_local_vol_model_with_reporting_basis() -> ModelSpec {
        let valuation: Date = "2026-09-04".parse().expect("valuation");
        let first_maturity: Date = "2027-03-05".parse().expect("first maturity");
        let second_maturity: Date = "2027-09-04".parse().expect("second maturity");
        let maturity_nodes = vec![
            DayCountConvention::Act365F.year_fraction(valuation, first_maturity),
            DayCountConvention::Act365F.year_fraction(valuation, second_maturity),
        ];
        ModelSpec::LocalVolatility(
            LocalVolatilitySpec::from_explicit_grid(
                vec![0.0, maturity_nodes[0], maturity_nodes[1]],
                vec![-1.0, 0.0, 1.0],
                vec![0.04; 9],
                1.0e-8,
                1.0,
            )
            .expect("local volatility")
            .with_reporting_iv_basis(
                LocalVolatilityReportingBasis::new(
                    maturity_nodes,
                    vec![-1.0, 0.0, 1.0],
                    vec![0.2; 6],
                )
                .expect("reporting basis"),
            ),
        )
    }

    fn price_only_request_with_model(
        model: ModelSpec,
        sampling_units: u64,
        antithetic: bool,
        risk: RiskRequest,
    ) -> PricingRequest {
        let underlying = UnderlyingId::new(1);
        let currency = CurrencyId::new(1);
        let product = ProductSpec::EuropeanVanilla(
            EuropeanVanillaSpec::new(
                underlying,
                currency,
                "2027-09-04".parse().expect("expiry"),
                100.0,
                1.0,
                OptionSide::Call,
            )
            .expect("product"),
        );
        let market = MarketContext::Equity(EquityMarket::new(
            currency,
            EquityForward::new(
                underlying,
                PositiveF64::new(100.0, "spot").expect("spot"),
                curve(1, 0.0),
                curve(2, 0.0),
            ),
        ));
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
    fn black_76_mc_converges_to_the_analytical_oracle_with_reported_error() {
        let request = zero_carry_black_76_request(131_072, true);
        let oracle = black_76_oracle(&request).expect("oracle").price;
        let result = price_pseudo_monte_carlo(&request, policy(4)).expect("MC");
        let estimate = result.pricing_result.value;
        let error = (estimate.value().get() - oracle).abs();
        assert!(error <= 6.0 * estimate.standard_error().get());
    }

    #[test]
    fn local_vol_price_only_matches_constant_variance_black_scholes_limit() {
        let risk = RiskRequest::price_only(SmileDynamics::StickyLogMoneyness);
        let local_vol = local_vol_price_only_request(4096, true, risk);
        let black_scholes = zero_carry_black_scholes_request(4096, true);
        let local_result = price_pseudo_monte_carlo(&local_vol, policy(2)).expect("local vol");
        let bs_result = price_pseudo_monte_carlo(&black_scholes, policy(2)).expect("black scholes");
        assert_eq!(
            local_result.pricing_result.value.value().to_bits(),
            bs_result.pricing_result.value.value().to_bits()
        );
        assert_eq!(
            local_result
                .pricing_result
                .value
                .standard_error()
                .get()
                .to_bits(),
            bs_result
                .pricing_result
                .value
                .standard_error()
                .get()
                .to_bits()
        );
        assert_eq!(local_result.evaluated_paths, 8192);
    }

    #[test]
    fn local_vol_rqmc_price_only_matches_constant_variance_black_scholes_limit() {
        let local_vol = zero_carry_rqmc_request(constant_local_vol_model());
        let black_scholes = zero_carry_rqmc_request(ModelSpec::BlackScholes(
            BlackScholesSpec::new(0.2).expect("model"),
        ));
        let local_result = price_monte_carlo(&local_vol, policy(2)).expect("local vol");
        let bs_result = price_monte_carlo(&black_scholes, policy(2)).expect("black scholes");
        assert!(
            (local_result.pricing_result.value.value().get()
                - bs_result.pricing_result.value.value().get())
            .abs()
                < 1.0e-12
        );
        assert!(
            (local_result.pricing_result.value.standard_error().get()
                - bs_result.pricing_result.value.standard_error().get())
            .abs()
                < 1.0e-12
        );
        assert_eq!(local_result.independent_sampling_units, 16);
        assert_eq!(local_result.evaluated_paths, 8192);
        assert!(local_result.diagnostics.direction_checksum.is_some());
        assert!(local_result.diagnostics.scramble_checksum.is_some());
    }

    #[test]
    fn local_vol_delta_and_gamma_use_common_random_number_bumps() {
        let risk = RiskRequest::new(
            true,
            Some(GammaConfig::new(
                SpotBump::relative(0.01).expect("gamma bump"),
            )),
            false,
            None,
            SmileDynamics::StickyLogMoneyness,
            Some(16),
            Some(128),
        )
        .expect("risk");
        let request = local_vol_price_only_request(4096, true, risk);
        let result = price_pseudo_monte_carlo(&request, policy(2)).expect("local vol risks");
        assert!(result.pricing_result.risks.delta.is_some());
        assert!(result.pricing_result.risks.gamma.is_some());
        assert!(result.pricing_result.risks.vega.is_none());
        assert_eq!(
            result.risk_diagnostics.methods.delta,
            Some(RiskMethod::CentralBump)
        );
        assert_eq!(
            result.risk_diagnostics.methods.gamma,
            Some(RiskMethod::CentralBump)
        );
        assert_eq!(result.risk_diagnostics.methods.vega, None);
        assert_eq!(result.risk_diagnostics.delta_validation, None);
        assert_eq!(result.risk_diagnostics.gamma_validation, None);
    }

    #[test]
    fn local_vol_rqmc_delta_and_gamma_use_between_scramble_uncertainty() {
        let risk = RiskRequest::new(
            true,
            Some(GammaConfig::new(
                SpotBump::relative(0.01).expect("gamma bump"),
            )),
            false,
            None,
            SmileDynamics::StickyLogMoneyness,
            Some(16),
            Some(128),
        )
        .expect("risk");
        let request = zero_carry_rqmc_request_with_risk(constant_local_vol_model(), risk);
        let result = price_monte_carlo(&request, policy(2)).expect("local vol rqmc risks");
        let delta = result.pricing_result.risks.delta.expect("delta").raw();
        let gamma = result.pricing_result.risks.gamma.expect("gamma").raw();
        assert_eq!(delta.effective_sampling_units().get(), 16);
        assert_eq!(gamma.effective_sampling_units().get(), 16);
        assert_eq!(
            result.risk_diagnostics.methods.delta,
            Some(RiskMethod::CentralBump)
        );
        assert_eq!(
            result.risk_diagnostics.methods.gamma,
            Some(RiskMethod::CentralBump)
        );
    }

    #[test]
    fn local_vol_vega_and_vega_kt_are_reported() {
        let risk = RiskRequest::new(
            true,
            Some(GammaConfig::new(
                SpotBump::relative(0.01).expect("gamma bump"),
            )),
            true,
            Some(
                VegaKtConfig::new(
                    vec![
                        "2027-03-05".parse().expect("first maturity"),
                        "2027-09-04".parse().expect("second maturity"),
                    ],
                    vec![-1.0, 0.0, 1.0],
                    1.0e-8,
                    true,
                )
                .expect("vega kt"),
            ),
            SmileDynamics::StickyLogMoneyness,
            Some(16),
            Some(128),
        )
        .expect("risk");
        let request = price_only_request_with_model(
            constant_local_vol_model_with_reporting_basis(),
            1024,
            true,
            risk,
        );
        let result = price_pseudo_monte_carlo(&request, policy(2)).expect("local vol vega kt");
        let vega = result.pricing_result.risks.vega.expect("vega").raw();
        assert!(vega.value().get().is_finite());
        let report = result.pricing_result.risks.vega_kt.expect("vega kt");
        assert_eq!(report.coordinates().len(), 6);
        assert_eq!(report.estimates().len(), 6);
        assert_eq!(report.raw_buckets().len(), 6);
        assert_eq!(
            report.full_bucket_covariance().expect("covariance").len(),
            36
        );
        assert!(report.projection().scalar_vega().get().is_finite());
        assert_eq!(
            result.risk_diagnostics.methods.vega,
            Some(RiskMethod::AadReverse)
        );
    }

    #[test]
    fn local_vol_rqmc_vega_kt_uses_between_scramble_bucket_uncertainty() {
        let risk = RiskRequest::new(
            false,
            None,
            true,
            Some(
                VegaKtConfig::new(
                    vec![
                        "2027-03-05".parse().expect("first maturity"),
                        "2027-09-04".parse().expect("second maturity"),
                    ],
                    vec![-1.0, 0.0, 1.0],
                    1.0e-8,
                    false,
                )
                .expect("vega kt"),
            ),
            SmileDynamics::StickyLogMoneyness,
            Some(16),
            Some(128),
        )
        .expect("risk");
        let request = zero_carry_rqmc_request_with_risk(
            constant_local_vol_model_with_reporting_basis(),
            risk,
        );
        let result = price_monte_carlo(&request, policy(2)).expect("local vol rqmc vega kt");
        let vega = result.pricing_result.risks.vega.expect("vega").raw();
        assert_eq!(vega.effective_sampling_units().get(), 16);
        let report = result.pricing_result.risks.vega_kt.expect("vega kt");
        assert!(report.estimates()[0].raw_mean().get().is_finite());
        assert!(report.full_bucket_covariance().is_none());
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
    fn digital_zero_volatility_obeys_exact_indicator_payoff_discounting() {
        let cash_request = digital_zero_vol_request(OptionSide::Call, 100.0, DigitalPayout::Cash);
        let cash_plan = SimulationPlan::compile(&cash_request, policy(2)).expect("cash plan");
        let cash_expected = cash_plan.discount() * 10.0;
        let cash_result = cash_plan.execute().expect("cash execution");
        assert_eq!(
            cash_result.diagnostics.valuation_kind,
            PayoffValuationKind::ExactContractual
        );
        assert!(cash_result.diagnostics.payoff_smoothing.is_none());
        assert_eq!(
            cash_result.pricing_result.value.value().get(),
            cash_expected
        );
        assert_eq!(cash_result.sampling_variance.to_bits(), 0.0_f64.to_bits());

        let asset_request = digital_zero_vol_request(OptionSide::Call, 100.0, DigitalPayout::Asset);
        let asset_plan = SimulationPlan::compile(&asset_request, policy(2)).expect("asset plan");
        let asset_expected = asset_plan.discount() * asset_plan.forward() * 10.0;
        let asset_result = asset_plan.execute().expect("asset execution");
        assert!(
            (asset_result.pricing_result.value.value().get() - asset_expected).abs() <= 1.0e-12
        );
        assert_eq!(asset_result.sampling_variance.to_bits(), 0.0_f64.to_bits());

        let out_request = digital_zero_vol_request(OptionSide::Put, 100.0, DigitalPayout::Cash);
        let out_result = price_pseudo_monte_carlo(&out_request, policy(2)).expect("out");
        assert_eq!(out_result.pricing_result.value.value().get(), 0.0);
        assert_eq!(out_result.sampling_variance.to_bits(), 0.0_f64.to_bits());
    }

    #[test]
    fn smoothed_digital_reports_aad_risk_and_crn_validation() {
        let base = digital_zero_vol_request(OptionSide::Call, 100.0, DigitalPayout::Cash);
        let smoothing = PayoffSmoothing::compact_c2(2.0).expect("smoothing");
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
        .with_payoff_smoothing(smoothing);
        let request = PricingRequest::new(
            base.valuation_date(),
            base.product().clone(),
            base.market().clone(),
            ModelSpec::BlackScholes(BlackScholesSpec::new(0.2).expect("model")),
            EngineConfig::PseudoMonteCarlo(
                PseudoMcConfig::new(7, 4096, VarianceReduction::new(true, false)).expect("engine"),
            ),
            risk,
        )
        .expect("request");

        let result = price_pseudo_monte_carlo(&request, policy(2)).expect("price");
        assert!(result.pricing_result.risks.delta.is_some());
        assert!(result.pricing_result.risks.vega.is_some());
        assert!(result.risk_diagnostics.delta_validation.is_some());
        assert!(result.risk_diagnostics.vega_validation.is_some());
        assert_eq!(
            result.diagnostics.valuation_kind,
            PayoffValuationKind::SmoothedSurrogate
        );
        let diagnostics = result.diagnostics.payoff_smoothing.expect("diagnostics");
        assert_eq!(diagnostics.kernel, PayoffSmoothingKernel::CompactC2);
        assert_eq!(diagnostics.policy_version, PayoffSmoothing::POLICY_VERSION);
        assert_eq!(diagnostics.half_width.get(), 2.0);
        assert_eq!(diagnostics.full_transition_width.get(), 4.0);
        assert_eq!(diagnostics.width_unit, PayoffSmoothingWidthUnit::Spot);
        assert!(diagnostics.price_and_greeks_share_payoff);
        assert_eq!(diagnostics.endpoint_count, 1);
        assert_eq!(diagnostics.dividend_jump_count, 0);
    }

    #[test]
    fn digital_zero_volatility_discounts_to_explicit_payment_date() {
        let mut request = digital_zero_vol_request(OptionSide::Call, 100.0, DigitalPayout::Cash);
        request = PricingRequest::new(
            request.valuation_date(),
            ProductSpec::Digital(
                DigitalSpec::with_payment_date(
                    UnderlyingId::new(1),
                    CurrencyId::new(1),
                    "2027-09-04".parse().expect("expiry"),
                    100.0,
                    10.0,
                    OptionSide::Call,
                    DigitalPayout::Cash,
                    "2027-09-05".parse().expect("payment"),
                )
                .expect("digital"),
            ),
            request.market().clone(),
            request.model().clone(),
            request.engine(),
            request.risk().clone(),
        )
        .expect("request");
        let plan = SimulationPlan::compile(&request, policy(2)).expect("plan");
        let result = plan.execute().expect("execution");
        assert_eq!(
            result.pricing_result.value.value().get(),
            plan.discount() * 10.0
        );
        assert_eq!(result.sampling_variance.to_bits(), 0.0_f64.to_bits());
    }

    #[test]
    fn barrier_zero_volatility_uses_declared_monitoring_knock_out() {
        let live =
            SimulationPlan::compile(&barrier_zero_vol_request(200.0), policy(2)).expect("live");
        assert_eq!(live.observation_dates.len(), 2);
        let live_expected = live.discount() * (live.observation_forwards[1] - 100.0) * 2.0;
        let live_result = live.execute().expect("live execution");
        assert!((live_result.pricing_result.value.value().get() - live_expected).abs() <= 1.0e-12);
        assert_eq!(live_result.sampling_variance.to_bits(), 0.0_f64.to_bits());

        let knocked =
            SimulationPlan::compile(&barrier_zero_vol_request(50.0), policy(2)).expect("knocked");
        let knocked_result = knocked.execute().expect("knocked execution");
        assert_eq!(
            knocked_result.pricing_result.value.value().get().to_bits(),
            0.0_f64.to_bits()
        );
        assert_eq!(
            knocked_result.sampling_variance.to_bits(),
            0.0_f64.to_bits()
        );
    }

    #[test]
    fn continuous_barrier_zero_variance_has_deterministic_bridge_limits() {
        let live = SimulationPlan::compile(
            &barrier_zero_vol_request_with_monitoring(120.0, BarrierMonitoring::Continuous),
            policy(2),
        )
        .expect("continuous plan");
        assert!(live.continuous_barrier.is_some());
        let live_result = live.execute().expect("live execution");
        let expected = live.discount() * (live.observation_forwards[1] - 100.0) * 2.0;
        assert!((live_result.pricing_result.value.value().get() - expected).abs() <= 1.0e-12);
        let live_diagnostics = live_result
            .diagnostics
            .barrier_bridge
            .expect("bridge diagnostics");
        assert_eq!(live_diagnostics.endpoint_hit_fraction, 0.0);
        assert_eq!(live_diagnostics.dividend_jump_hit_fraction, 0.0);
        assert_eq!(live_diagnostics.mean_interval_count, 2.0);
        assert_eq!(live_diagnostics.mean_zero_variance_count, 2.0);

        let touched = SimulationPlan::compile(
            &barrier_zero_vol_request_with_monitoring(50.0, BarrierMonitoring::Continuous),
            policy(2),
        )
        .expect("touched plan");
        let touched_result = touched.execute().expect("touched execution");
        assert_eq!(touched_result.pricing_result.value.value().get(), 0.0);
        let touched_diagnostics = touched_result
            .diagnostics
            .barrier_bridge
            .expect("bridge diagnostics");
        assert_eq!(touched_diagnostics.endpoint_hit_fraction, 1.0);
        assert_eq!(touched_diagnostics.mean_interval_count, 0.0);
    }

    #[test]
    fn continuous_barrier_bridge_matches_independent_quadrature_reference() {
        let underlying = UnderlyingId::new(1);
        let currency = CurrencyId::new(1);
        let expiry: Date = "2027-09-04".parse().expect("expiry");
        let product = ProductSpec::Barrier(
            BarrierSpec::new(
                underlying,
                currency,
                expiry,
                100.0,
                120.0,
                1.0,
                OptionSide::Call,
                BarrierDirection::Up,
                BarrierStyle::KnockOut,
                BarrierMonitoring::Continuous,
                vec![expiry],
                None,
                expiry,
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
        let request = PricingRequest::new(
            "2026-09-04".parse().expect("valuation"),
            product,
            market,
            ModelSpec::BlackScholes(BlackScholesSpec::new(0.2).expect("model")),
            EngineConfig::RandomizedQuasiMonteCarlo(
                RqmcConfig::new(
                    8192,
                    16,
                    0x1234_5678_9abc_def0,
                    VarianceReduction::new(true, true),
                )
                .expect("engine"),
            ),
            RiskRequest::price_only(SmileDynamics::StickyLogMoneyness),
        )
        .expect("request");
        let risk_request = PricingRequest::new(
            request.valuation_date(),
            request.product().clone(),
            request.market().clone(),
            request.model().clone(),
            request.engine(),
            all_risks(),
        )
        .expect("continuous Barrier risk request");
        let plan = SimulationPlan::compile(&request, policy(2)).expect("plan");
        assert_eq!(plan.observation_times.len(), 1);
        let knock_out = plan.continuous_barrier.clone().expect("continuous Barrier");
        let knock_in = ContinuousBarrierRuntime {
            style: BarrierStyle::KnockIn,
            ..knock_out.clone()
        };
        for normal in [-1.0, 0.0, 0.5] {
            let observations = plan.path_observations_from_normals(&[normal], plan.spot, 0.2);
            let out = plan
                .continuous_barrier_discounted_payoff(&knock_out, &observations)
                .expect("knock out");
            let entered = plan
                .continuous_barrier_discounted_payoff(&knock_in, &observations)
                .expect("knock in");
            let vanilla = plan.discount
                * (observations[knock_out.expiry_observation_index].post_spot - 100.0).max(0.0);
            assert!((out + entered - vanilla).abs() < 1.0e-14);
        }
        for normal in [-1.0, -0.5, 0.0] {
            let normals = [normal];
            let analytic = plan
                .continuous_barrier_pathwise_aad(&knock_out, &normals, 100.0, 0.2)
                .expect("analytic");
            let spot_bump = 1.0e-4;
            let spot_down = plan
                .continuous_barrier_pathwise_aad(&knock_out, &normals, 100.0 - spot_bump, 0.2)
                .expect("spot down")
                .price;
            let spot_up = plan
                .continuous_barrier_pathwise_aad(&knock_out, &normals, 100.0 + spot_bump, 0.2)
                .expect("spot up")
                .price;
            let delta = (spot_up - spot_down) / (2.0 * spot_bump);
            let volatility_bump = 1.0e-5;
            let volatility_down = plan
                .continuous_barrier_pathwise_aad(&knock_out, &normals, 100.0, 0.2 - volatility_bump)
                .expect("volatility down")
                .price;
            let volatility_up = plan
                .continuous_barrier_pathwise_aad(&knock_out, &normals, 100.0, 0.2 + volatility_bump)
                .expect("volatility up")
                .price;
            let vega = (volatility_up - volatility_down) / (2.0 * volatility_bump);
            assert!((analytic.delta - delta).abs() < 2.0e-8);
            assert!((analytic.vega - vega).abs() < 2.0e-7);
        }
        let result = plan.execute().expect("execution");

        let volatility = 0.2;
        let time = plan.time;
        let standard_deviation = volatility * time.sqrt();
        let upper_normal = ((120.0 / plan.forward).ln() + 0.5 * volatility * volatility * time)
            / standard_deviation;
        let lower_normal = -10.0;
        let slices = 200_000_u32;
        let dz = (upper_normal - lower_normal) / f64::from(slices);
        let mut reference = pricing_numerics::NeumaierSum::default();
        for index in 0..slices {
            let normal = lower_normal + (f64::from(index) + 0.5) * dz;
            let terminal = plan.forward
                * (-0.5 * volatility * volatility * time + standard_deviation * normal).exp();
            let survival = 1.0
                - (-2.0 * (120.0_f64 / 100.0).ln() * (120.0 / terminal).ln()
                    / (volatility * volatility * time))
                    .exp();
            let density = (-0.5 * normal * normal).exp() / std::f64::consts::TAU.sqrt();
            reference.add((terminal - 100.0).max(0.0) * survival * density * dz);
        }
        let reference = plan.discount * reference.total();
        let estimate = &result.pricing_result.value;
        assert!(
            (estimate.value().get() - reference).abs()
                <= 8.0 * estimate.standard_error().get() + 5.0e-5,
            "estimate={}, standard_error={}, reference={reference}",
            estimate.value().get(),
            estimate.standard_error().get()
        );

        let risk_result = SimulationPlan::compile(&risk_request, policy(2))
            .expect("risk plan")
            .execute()
            .expect("risk execution");
        assert_eq!(
            risk_result.pricing_result.value.value().get().to_bits(),
            result.pricing_result.value.value().get().to_bits()
        );
        assert!(risk_result.pricing_result.risks.delta.is_some());
        assert!(risk_result.pricing_result.risks.gamma.is_some());
        assert!(risk_result.pricing_result.risks.vega.is_some());
        assert!(risk_result.risk_diagnostics.delta_validation.is_some());
        assert!(risk_result.risk_diagnostics.gamma_validation.is_some());
        assert!(risk_result.risk_diagnostics.vega_validation.is_some());
        let diagnostics = result
            .diagnostics
            .barrier_bridge
            .expect("bridge diagnostics");
        assert_eq!(diagnostics.abi, BARRIER_BRIDGE_ABI);
        assert_eq!(diagnostics.indicator_mode, BarrierHitIndicatorMode::Exact);
        assert_eq!(diagnostics.mean_interval_count, 1.0);
        assert!(diagnostics.endpoint_hit_fraction > 0.0);
        assert!(diagnostics.mean_conditional_bridge_hit_weight > 0.0);
        assert_eq!(risk_result.diagnostics.barrier_bridge, Some(diagnostics));
    }

    #[test]
    fn smoothed_continuous_barrier_reports_matched_aad_and_diagnostics() {
        let risk = all_risks().with_payoff_smoothing(
            PayoffSmoothing::compact_c2(2.0).expect("continuous Barrier smoothing"),
        );
        let request = continuous_barrier_conformance_request(
            BarrierDirection::Up,
            BarrierStyle::KnockOut,
            Some(7.5),
            EngineConfig::PseudoMonteCarlo(
                PseudoMcConfig::new(29, 4_096, VarianceReduction::new(true, true)).expect("engine"),
            ),
            risk,
        );
        let plan = SimulationPlan::compile(&request, policy(2)).expect("smoothed plan");
        let barrier = plan
            .continuous_barrier
            .as_ref()
            .expect("continuous Barrier");
        let normals = [0.35, -0.2];
        let analytic = plan
            .continuous_barrier_pathwise_aad(barrier, &normals, 100.0, 0.2)
            .expect("analytic");
        let spot_bump = 1.0e-4;
        let spot_down = plan
            .continuous_barrier_pathwise_aad(barrier, &normals, 100.0 - spot_bump, 0.2)
            .expect("spot down")
            .price;
        let spot_up = plan
            .continuous_barrier_pathwise_aad(barrier, &normals, 100.0 + spot_bump, 0.2)
            .expect("spot up")
            .price;
        let volatility_bump = 1.0e-5;
        let volatility_down = plan
            .continuous_barrier_pathwise_aad(barrier, &normals, 100.0, 0.2 - volatility_bump)
            .expect("volatility down")
            .price;
        let volatility_up = plan
            .continuous_barrier_pathwise_aad(barrier, &normals, 100.0, 0.2 + volatility_bump)
            .expect("volatility up")
            .price;
        assert!((analytic.delta - (spot_up - spot_down) / (2.0 * spot_bump)).abs() < 2.0e-8);
        assert!(
            (analytic.vega - (volatility_up - volatility_down) / (2.0 * volatility_bump)).abs()
                < 3.0e-7
        );

        let result = plan.execute().expect("execution");
        assert_eq!(
            result.diagnostics.valuation_kind,
            PayoffValuationKind::SmoothedSurrogate
        );
        let smoothing = result
            .diagnostics
            .payoff_smoothing
            .expect("smoothing diagnostics");
        assert!(smoothing.price_and_greeks_share_payoff);
        assert_eq!(smoothing.half_width.get(), 2.0);
        assert_eq!(smoothing.endpoint_count, 2);
        assert_eq!(smoothing.dividend_jump_count, 1);
        let bridge = result
            .diagnostics
            .barrier_bridge
            .expect("bridge diagnostics");
        assert_eq!(bridge.abi, SMOOTHED_BARRIER_BRIDGE_ABI);
        assert_eq!(
            bridge.policy_version,
            BarrierBridgeDiagnostics::SMOOTHED_POLICY_VERSION
        );
        assert_eq!(bridge.indicator_mode, BarrierHitIndicatorMode::CompactC2);
        assert!(bridge.mean_conditional_bridge_hit_weight > 0.0);
        assert!(result.pricing_result.risks.delta.is_some());
        assert!(result.pricing_result.risks.gamma.is_some());
        assert!(result.pricing_result.risks.vega.is_some());
        assert!(result.risk_diagnostics.delta_validation.is_some());
        assert!(result.risk_diagnostics.gamma_validation.is_some());
        assert!(result.risk_diagnostics.vega_validation.is_some());
    }

    #[test]
    fn smoothed_continuous_up_barrier_rejects_width_outside_log_domain() {
        let request =
            barrier_zero_vol_request_with_monitoring(120.0, BarrierMonitoring::Continuous)
                .replace_risk(
                    RiskRequest::price_only(SmileDynamics::StickyLogMoneyness)
                        .with_payoff_smoothing(
                            PayoffSmoothing::compact_c2(120.0).expect("smoothing"),
                        ),
                );
        let error = SimulationPlan::compile(&request, policy(2)).expect_err("invalid width");
        assert!(matches!(
            error,
            MonteCarloError::BarrierBridge(BarrierBridgeError::InvalidSmoothedSafeDistance { .. })
        ));
    }

    #[test]
    fn smoothed_continuous_barrier_local_vol_matches_constant_variance_limit() {
        let risk = all_risks().with_payoff_smoothing(
            PayoffSmoothing::compact_c2(2.0).expect("continuous Barrier smoothing"),
        );
        let black_scholes = barrier_dividend_jump_request_with_monitoring(
            BarrierDirection::Down,
            BarrierStyle::KnockOut,
            0.2,
            risk,
            BarrierMonitoring::Continuous,
            Some(70.0),
        );
        let local_vol = PricingRequest::new(
            black_scholes.valuation_date(),
            black_scholes.product().clone(),
            black_scholes.market().clone(),
            constant_local_vol_model(),
            black_scholes.engine(),
            black_scholes.risk().clone(),
        )
        .expect("Local Volatility smoothed continuous Barrier request");
        let black_scholes_result = SimulationPlan::compile(&black_scholes, policy(2))
            .expect("Black-Scholes plan")
            .execute()
            .expect("Black-Scholes execution");
        let local_vol_result = SimulationPlan::compile(&local_vol, policy(2))
            .expect("Local Volatility plan")
            .execute()
            .expect("Local Volatility execution");
        assert!(
            (local_vol_result.pricing_result.value.value().get()
                - black_scholes_result.pricing_result.value.value().get())
            .abs()
                < 1.0e-12
        );
        let black_scholes_vega = black_scholes_result
            .pricing_result
            .risks
            .vega
            .expect("Black-Scholes Vega")
            .raw()
            .value()
            .get();
        let local_vol_vega = local_vol_result
            .pricing_result
            .risks
            .vega
            .expect("Local Volatility Vega")
            .raw()
            .value()
            .get();
        assert!((local_vol_vega - black_scholes_vega).abs() < 1.0e-10);
        assert_eq!(
            local_vol_result
                .diagnostics
                .barrier_bridge
                .expect("bridge diagnostics")
                .indicator_mode,
            BarrierHitIndicatorMode::CompactC2
        );
    }

    #[test]
    fn continuous_barrier_in_out_parity_covers_directions_and_rebates() {
        let engine = EngineConfig::PseudoMonteCarlo(
            PseudoMcConfig::new(7, 16, VarianceReduction::new(true, true)).expect("engine"),
        );
        for smoothing_width in [None, Some(2.0)] {
            for direction in [BarrierDirection::Up, BarrierDirection::Down] {
                for rebate in [None, Some(7.5)] {
                    let risk = smoothing_width.map_or_else(
                        || RiskRequest::price_only(SmileDynamics::StickyLogMoneyness),
                        |width| {
                            RiskRequest::price_only(SmileDynamics::StickyLogMoneyness)
                                .with_payoff_smoothing(
                                    PayoffSmoothing::compact_c2(width).expect("smoothing"),
                                )
                        },
                    );
                    let request = continuous_barrier_conformance_request(
                        direction,
                        BarrierStyle::KnockOut,
                        rebate,
                        engine,
                        risk,
                    );
                    let plan = SimulationPlan::compile(&request, policy(2)).expect("plan");
                    let knock_out = plan.continuous_barrier.clone().expect("continuous Barrier");
                    let knock_in = ContinuousBarrierRuntime {
                        style: BarrierStyle::KnockIn,
                        ..knock_out.clone()
                    };

                    for normals in [[-1.25, 0.5], [0.0, 0.0], [0.75, -0.25], [1.5, 1.0]] {
                        let observations = plan.path_observations_from_normals(
                            &normals,
                            plan.spot,
                            plan.volatility,
                        );
                        let out = plan
                            .continuous_barrier_discounted_payoff(&knock_out, &observations)
                            .expect("knock out");
                        let entered = plan
                            .continuous_barrier_discounted_payoff(&knock_in, &observations)
                            .expect("knock in");
                        let terminal = observations[knock_out.expiry_observation_index].post_spot;
                        let vanilla = (terminal - knock_out.strike).max(0.0) * knock_out.notional;
                        let expected = plan.discount * (vanilla + rebate.unwrap_or(0.0));
                        assert!(
                            (out + entered - expected).abs() < 2.0e-13,
                            "smoothing_width={smoothing_width:?}, direction={direction:?}, rebate={rebate:?}, normals={normals:?}, out={out}, in={entered}, expected={expected}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn continuous_barrier_replays_across_worker_counts_for_pseudo_mc_and_rqmc() {
        let engines = [
            EngineConfig::PseudoMonteCarlo(
                PseudoMcConfig::new(0x1234_5678, 2_048, VarianceReduction::new(true, true))
                    .expect("pseudo-MC engine"),
            ),
            EngineConfig::RandomizedQuasiMonteCarlo(
                RqmcConfig::new(512, 8, 0x8765_4321, VarianceReduction::new(true, true))
                    .expect("RQMC engine"),
            ),
        ];
        for smoothing_width in [None, Some(2.0)] {
            for engine in engines {
                let risk = smoothing_width.map_or_else(all_risks, |width| {
                    all_risks().with_payoff_smoothing(
                        PayoffSmoothing::compact_c2(width).expect("smoothing"),
                    )
                });
                let request = continuous_barrier_conformance_request(
                    BarrierDirection::Down,
                    BarrierStyle::KnockIn,
                    Some(7.5),
                    engine,
                    risk,
                );
                let plan =
                    SimulationPlan::compile(&request, policy(1)).expect("single-worker plan");
                assert_eq!(plan.observation_times.len(), 2);
                let mut single = plan.execute().expect("single-worker execution");
                let parallel = SimulationPlan::compile(&request, policy(4))
                    .expect("parallel plan")
                    .execute()
                    .expect("parallel execution");
                assert_ne!(
                    single.diagnostics.worker_threads,
                    parallel.diagnostics.worker_threads
                );
                single.diagnostics.worker_threads = parallel.diagnostics.worker_threads;
                assert_eq!(single, parallel, "smoothing_width={smoothing_width:?}");
            }
        }
    }

    #[test]
    fn continuous_barrier_splits_affine_dividend_jump_without_extra_coordinate() {
        let deterministic = barrier_dividend_jump_request_with_monitoring(
            BarrierDirection::Down,
            BarrierStyle::KnockOut,
            0.0,
            RiskRequest::price_only(SmileDynamics::StickyLogMoneyness),
            BarrierMonitoring::Continuous,
            Some(90.0),
        );
        let deterministic_plan =
            SimulationPlan::compile(&deterministic, policy(2)).expect("deterministic plan");
        assert_eq!(
            deterministic_plan.observation_dates.as_ref(),
            [None, Some(deterministic.product().expiry())]
        );
        assert_eq!(deterministic_plan.observation_times.len(), 2);
        assert_eq!(
            deterministic_plan
                .continuous_barrier
                .as_ref()
                .expect("continuous Barrier")
                .bridge_observation_indices
                .len(),
            2
        );
        assert!(deterministic_plan.observation_pre_dividend_coordinates[0].is_some());
        let deterministic_result = deterministic_plan
            .execute()
            .expect("deterministic execution");
        assert_eq!(deterministic_result.pricing_result.value.value().get(), 0.0);
        let deterministic_diagnostics = deterministic_result
            .diagnostics
            .barrier_bridge
            .expect("bridge diagnostics");
        assert_eq!(deterministic_diagnostics.endpoint_hit_fraction, 0.0);
        assert_eq!(deterministic_diagnostics.dividend_jump_hit_fraction, 1.0);
        assert_eq!(
            deterministic_diagnostics.mean_conditional_bridge_hit_weight,
            0.0
        );

        let source = match deterministic.product() {
            ProductSpec::Barrier(source) => source,
            _ => unreachable!("helper constructs a Barrier"),
        };
        let valuation_monitoring_product = ProductSpec::Barrier(
            BarrierSpec::new(
                source.underlying(),
                source.currency(),
                source.expiry(),
                source.strike().get(),
                source.barrier().get(),
                source.notional().get(),
                source.side(),
                source.direction(),
                source.style(),
                BarrierMonitoring::Continuous,
                vec![deterministic.valuation_date(), source.expiry()],
                source.rebate().map(PositiveF64::get),
                source.payment_date(),
            )
            .expect("valuation-date monitoring product"),
        );
        let valuation_monitoring_request = PricingRequest::new(
            deterministic.valuation_date(),
            valuation_monitoring_product,
            deterministic.market().clone(),
            deterministic.model().clone(),
            deterministic.engine(),
            deterministic.risk().clone(),
        )
        .expect("valuation-date monitoring request");
        let valuation_monitoring_plan =
            SimulationPlan::compile(&valuation_monitoring_request, policy(2))
                .expect("valuation-date monitoring plan");
        assert_eq!(valuation_monitoring_plan.observation_times.len(), 2);
        assert_eq!(
            valuation_monitoring_plan
                .execute()
                .expect("valuation-date execution")
                .pricing_result
                .value
                .value()
                .get()
                .to_bits(),
            deterministic_result
                .pricing_result
                .value
                .value()
                .get()
                .to_bits()
        );

        let request = barrier_dividend_jump_request_with_monitoring(
            BarrierDirection::Down,
            BarrierStyle::KnockOut,
            0.2,
            all_risks(),
            BarrierMonitoring::Continuous,
            Some(70.0),
        );
        let plan = SimulationPlan::compile(&request, policy(2)).expect("stochastic plan");
        let barrier = plan
            .continuous_barrier
            .as_ref()
            .expect("continuous Barrier");
        let normals = [0.75, 0.25];
        let observations = plan.path_observations_from_normals(&normals, 100.0, 0.2);
        let ex_time = plan.observation_times[0];
        let first_state = observations[0].canonical_f;
        let terminal_state = observations[1].canonical_f;
        let reserve = 15.0 / 0.9;
        let transformed_pre_dividend_barrier = (70.0 - reserve) / (1.0 - reserve / 100.0);
        let transformed_post_dividend_barrier = 70.0 / 0.75;
        let first_survival = 1.0
            - (-2.0
                * (100.0_f64 / transformed_pre_dividend_barrier).ln()
                * (first_state / transformed_pre_dividend_barrier).ln()
                / (0.2_f64.powi(2) * ex_time))
                .exp();
        let second_survival = 1.0
            - (-2.0
                * (first_state / transformed_post_dividend_barrier).ln()
                * (terminal_state / transformed_post_dividend_barrier).ln()
                / (0.2_f64.powi(2) * (plan.time - ex_time)))
                .exp();
        let expected = plan.discount
            * (observations[1].post_spot - 80.0).max(0.0)
            * first_survival
            * second_survival;
        let actual = plan
            .continuous_barrier_discounted_payoff(barrier, &observations)
            .expect("payoff");
        assert!((actual - expected).abs() < 1.0e-13);

        let analytic = plan
            .continuous_barrier_pathwise_aad(barrier, &normals, 100.0, 0.2)
            .expect("analytic");
        let spot_bump = 1.0e-4;
        let spot_down = plan
            .continuous_barrier_pathwise_aad(barrier, &normals, 100.0 - spot_bump, 0.2)
            .expect("spot down")
            .price;
        let spot_up = plan
            .continuous_barrier_pathwise_aad(barrier, &normals, 100.0 + spot_bump, 0.2)
            .expect("spot up")
            .price;
        let volatility_bump = 1.0e-5;
        let volatility_down = plan
            .continuous_barrier_pathwise_aad(barrier, &normals, 100.0, 0.2 - volatility_bump)
            .expect("volatility down")
            .price;
        let volatility_up = plan
            .continuous_barrier_pathwise_aad(barrier, &normals, 100.0, 0.2 + volatility_bump)
            .expect("volatility up")
            .price;
        assert!((analytic.delta - (spot_up - spot_down) / (2.0 * spot_bump)).abs() < 2.0e-8);
        assert!(
            (analytic.vega - (volatility_up - volatility_down) / (2.0 * volatility_bump)).abs()
                < 2.0e-7
        );

        let result = plan.execute().expect("risk execution");
        assert!(result.pricing_result.risks.delta.is_some());
        assert!(result.pricing_result.risks.gamma.is_some());
        assert!(result.pricing_result.risks.vega.is_some());
    }

    #[test]
    fn continuous_barrier_local_vol_matches_constant_variance_limit() {
        let black_scholes = barrier_dividend_jump_request_with_monitoring(
            BarrierDirection::Down,
            BarrierStyle::KnockOut,
            0.2,
            all_risks(),
            BarrierMonitoring::Continuous,
            Some(70.0),
        );
        let local_vol = PricingRequest::new(
            black_scholes.valuation_date(),
            black_scholes.product().clone(),
            black_scholes.market().clone(),
            constant_local_vol_model(),
            black_scholes.engine(),
            black_scholes.risk().clone(),
        )
        .expect("Local Volatility continuous Barrier request");
        let black_scholes_result = SimulationPlan::compile(&black_scholes, policy(2))
            .expect("Black-Scholes plan")
            .execute()
            .expect("Black-Scholes execution");
        let local_vol_result = SimulationPlan::compile(&local_vol, policy(2))
            .expect("Local Volatility plan")
            .execute()
            .expect("Local Volatility execution");

        let black_scholes_price = black_scholes_result.pricing_result.value.value().get();
        let local_vol_price = local_vol_result.pricing_result.value.value().get();
        assert!((local_vol_price - black_scholes_price).abs() < 1.0e-12);
        let black_scholes_vega = black_scholes_result
            .pricing_result
            .risks
            .vega
            .as_ref()
            .expect("Black-Scholes Vega")
            .raw()
            .value()
            .get();
        let local_vol_vega = local_vol_result
            .pricing_result
            .risks
            .vega
            .as_ref()
            .expect("Local Volatility Vega")
            .raw()
            .value()
            .get();
        assert!((local_vol_vega - black_scholes_vega).abs() < 1.0e-10);
        assert!(local_vol_result.pricing_result.risks.delta.is_some());
        assert!(local_vol_result.pricing_result.risks.gamma.is_some());
    }

    #[test]
    fn continuous_barrier_local_vol_time_step_refinement_converges() {
        let expiry: Date = "2027-09-04".parse().expect("expiry");
        let value = |time_nodes: Vec<f64>| {
            let request = continuous_barrier_local_vol_request(
                skewed_local_vol_model(time_nodes),
                vec![expiry],
                RiskRequest::price_only(SmileDynamics::StickyLogMoneyness),
            );
            let plan = SimulationPlan::compile(&request, policy(2)).expect("plan");
            let local_volatility = plan.local_volatility.as_ref().expect("Local Volatility");
            let shocks = local_volatility
                .plan
                .time_grid()
                .nodes()
                .windows(2)
                .map(|times| 0.35 * (times[1] - times[0]).sqrt())
                .collect::<Vec<_>>();
            plan.local_vol_discounted_payoff_at_spot(
                local_volatility,
                plan.spot,
                &shocks,
                PathIndex::new(0),
            )
            .expect("path value")
        };
        let coarse = value(vec![0.0, 1.0]);
        let medium = value(vec![0.0, 0.5, 1.0]);
        let fine = value(vec![0.0, 0.25, 0.5, 0.75, 1.0]);
        let reference = value((0..=16).map(|index| f64::from(index) / 16.0).collect());
        let coarse_error = (coarse - reference).abs();
        let medium_error = (medium - reference).abs();
        let fine_error = (fine - reference).abs();
        assert!(
            medium_error < coarse_error && fine_error < medium_error,
            "coarse={coarse}, medium={medium}, fine={fine}, reference={reference}"
        );
    }

    #[test]
    fn continuous_barrier_local_vol_monitoring_refinement_converges() {
        let first: Date = "2026-12-04".parse().expect("first");
        let second: Date = "2027-03-05".parse().expect("second");
        let third: Date = "2027-06-04".parse().expect("third");
        let expiry: Date = "2027-09-04".parse().expect("expiry");
        let value = |monitoring_dates: Vec<Date>, model: ModelSpec| {
            let request = continuous_barrier_local_vol_request(
                model,
                monitoring_dates,
                RiskRequest::price_only(SmileDynamics::StickyLogMoneyness),
            );
            let plan = SimulationPlan::compile(&request, policy(2)).expect("plan");
            let local_volatility = plan.local_volatility.as_ref().expect("Local Volatility");
            let shocks = local_volatility
                .plan
                .time_grid()
                .nodes()
                .windows(2)
                .map(|times| 0.35 * (times[1] - times[0]).sqrt())
                .collect::<Vec<_>>();
            plan.local_vol_discounted_payoff_at_spot(
                local_volatility,
                plan.spot,
                &shocks,
                PathIndex::new(0),
            )
            .expect("path value")
        };
        let coarse = value(vec![expiry], skewed_local_vol_model(vec![0.0, 1.0]));
        let medium = value(vec![second, expiry], skewed_local_vol_model(vec![0.0, 1.0]));
        let fine = value(
            vec![first, second, third, expiry],
            skewed_local_vol_model(vec![0.0, 1.0]),
        );
        let reference = value(
            vec![expiry],
            skewed_local_vol_model((0..=16).map(|index| f64::from(index) / 16.0).collect()),
        );
        let coarse_error = (coarse - reference).abs();
        let medium_error = (medium - reference).abs();
        let fine_error = (fine - reference).abs();
        assert!(
            medium_error < coarse_error && fine_error < medium_error,
            "coarse={coarse}, medium={medium}, fine={fine}, reference={reference}"
        );
    }

    #[test]
    fn continuous_barrier_local_vol_reverse_matches_parallel_volatility_bump() {
        let expiry: Date = "2027-09-04".parse().expect("expiry");
        let risk = RiskRequest::new(
            false,
            None,
            true,
            None,
            SmileDynamics::StickyLogMoneyness,
            None,
            None,
        )
        .expect("risk");
        let time_nodes = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let request = |volatility_shift| {
            continuous_barrier_local_vol_request(
                skewed_local_vol_model_with_parallel_shift(time_nodes.clone(), volatility_shift),
                vec![expiry],
                risk.clone(),
            )
        };
        let base = SimulationPlan::compile(&request(0.0), policy(2)).expect("base plan");
        let bump = 1.0e-5;
        let down = SimulationPlan::compile(&request(-bump), policy(2)).expect("down plan");
        let up = SimulationPlan::compile(&request(bump), policy(2)).expect("up plan");
        let base_runtime = base.local_volatility.as_ref().expect("base runtime");
        let shocks = base_runtime
            .plan
            .time_grid()
            .nodes()
            .windows(2)
            .map(|times| 0.35 * (times[1] - times[0]).sqrt())
            .collect::<Vec<_>>();
        let analytic = base
            .local_vol_pathwise_values_and_buckets(base_runtime, None, &shocks, PathIndex::new(0))
            .expect("analytic")
            .values[VEGA];
        let down_runtime = down.local_volatility.as_ref().expect("down runtime");
        let down_value = down
            .local_vol_discounted_payoff_at_spot(
                down_runtime,
                down.spot,
                &shocks,
                PathIndex::new(0),
            )
            .expect("down value");
        let up_runtime = up.local_volatility.as_ref().expect("up runtime");
        let up_value = up
            .local_vol_discounted_payoff_at_spot(up_runtime, up.spot, &shocks, PathIndex::new(0))
            .expect("up value");
        let finite_difference = (up_value - down_value) / (2.0 * bump);
        assert!(
            (analytic - finite_difference).abs() < 2.0e-6,
            "analytic={analytic}, finite_difference={finite_difference}"
        );
    }

    #[test]
    fn continuous_barrier_local_vol_reports_vega_kt() {
        let expiry: Date = "2027-09-04".parse().expect("expiry");
        let risk = RiskRequest::new(
            false,
            None,
            true,
            Some(
                VegaKtConfig::new(
                    vec!["2027-03-05".parse().expect("first maturity"), expiry],
                    vec![-1.0, 0.0, 1.0],
                    1.0e-8,
                    false,
                )
                .expect("VegaKT"),
            ),
            SmileDynamics::StickyLogMoneyness,
            None,
            None,
        )
        .expect("risk");
        let request = continuous_barrier_local_vol_request(
            constant_local_vol_model_with_reporting_basis(),
            vec![expiry],
            risk,
        );
        let result = SimulationPlan::compile(&request, policy(2))
            .expect("plan")
            .execute()
            .expect("execution");
        assert!(result.pricing_result.risks.vega.is_some());
        let vega_kt = result.pricing_result.risks.vega_kt.expect("VegaKT");
        assert_eq!(vega_kt.coordinates().len(), 6);
        assert_eq!(vega_kt.raw_buckets().len(), 6);
        assert!(vega_kt.projection().scalar_vega().get().is_finite());
    }

    #[test]
    fn barrier_dividend_collision_uses_pre_and_post_spot_without_an_extra_dimension() {
        for direction in [BarrierDirection::Up, BarrierDirection::Down] {
            let knock_in = SimulationPlan::compile(
                &barrier_dividend_jump_request(
                    direction,
                    BarrierStyle::KnockIn,
                    0.0,
                    RiskRequest::price_only(SmileDynamics::StickyLogMoneyness),
                ),
                policy(2),
            )
            .expect("knock-in plan");
            let knock_out = SimulationPlan::compile(
                &barrier_dividend_jump_request(
                    direction,
                    BarrierStyle::KnockOut,
                    0.0,
                    RiskRequest::price_only(SmileDynamics::StickyLogMoneyness),
                ),
                policy(2),
            )
            .expect("knock-out plan");

            assert_eq!(knock_in.observation_dates.len(), 2);
            assert_eq!(knock_in.payoff.pre_dividend_observations().len(), 1);
            let observations = knock_in.path_observations_from_normals(&[0.0, 0.0], 100.0, 0.0);
            assert!(
                (observations[0].pre_dividend_spot.expect("pre spot") - 100.0).abs() <= 1.0e-12
            );
            assert!((observations[0].post_spot - 85.0).abs() <= 1.0e-12);

            let knock_in_value = knock_in
                .execute()
                .expect("knock-in execution")
                .pricing_result
                .value
                .value()
                .get();
            let knock_out_value = knock_out
                .execute()
                .expect("knock-out execution")
                .pricing_result
                .value
                .value()
                .get();
            assert!((knock_in_value - 5.0).abs() <= 1.0e-12);
            assert_eq!(knock_out_value.to_bits(), 0.0_f64.to_bits());
            assert!((knock_in_value + knock_out_value - 5.0).abs() <= 1.0e-12);
        }
    }

    #[test]
    fn smoothed_barrier_dividend_jump_aad_matches_common_random_number_bumps() {
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
        .with_payoff_smoothing(PayoffSmoothing::compact_c2(5.0).expect("smoothing"));
        let request =
            barrier_dividend_jump_request(BarrierDirection::Up, BarrierStyle::KnockIn, 0.2, risk);
        let plan = SimulationPlan::compile(&request, policy(2)).expect("plan");
        let normals = [0.0, 0.0];
        let base = plan
            .pathwise_aad(&normals, plan.spot, plan.volatility)
            .expect("base AAD");
        let spot_bump = 1.0e-4;
        let delta_fd = (plan
            .pathwise_aad(&normals, plan.spot + spot_bump, plan.volatility)
            .expect("up spot")
            .price
            - plan
                .pathwise_aad(&normals, plan.spot - spot_bump, plan.volatility)
                .expect("down spot")
                .price)
            / (2.0 * spot_bump);
        let volatility_bump = 1.0e-5;
        let vega_fd = (plan
            .pathwise_aad(&normals, plan.spot, plan.volatility + volatility_bump)
            .expect("up volatility")
            .price
            - plan
                .pathwise_aad(&normals, plan.spot, plan.volatility - volatility_bump)
                .expect("down volatility")
                .price)
            / (2.0 * volatility_bump);
        assert!((base.delta - delta_fd).abs() <= 1.0e-7);
        assert!((base.vega - vega_fd).abs() <= 1.0e-6);

        let result = plan.execute().expect("execution");
        let diagnostics = result.diagnostics.payoff_smoothing.expect("smoothing");
        assert_eq!(diagnostics.endpoint_count, 2);
        assert_eq!(diagnostics.dividend_jump_count, 1);
    }

    #[test]
    fn local_vol_exact_barrier_detects_pre_dividend_hit() {
        let risk = RiskRequest::price_only(SmileDynamics::StickyLogMoneyness);
        let knock_in = SimulationPlan::compile(
            &local_vol_barrier_dividend_jump_request(BarrierStyle::KnockIn, risk.clone()),
            policy(2),
        )
        .expect("knock-in plan");
        let knock_out = SimulationPlan::compile(
            &local_vol_barrier_dividend_jump_request(BarrierStyle::KnockOut, risk),
            policy(2),
        )
        .expect("knock-out plan");
        let local_volatility = knock_in
            .local_volatility
            .as_ref()
            .expect("Local Volatility");
        let shocks = vec![0.0; local_volatility.plan.time_grid().step_count()];
        assert_eq!(shocks.len(), 2);
        let path = local_volatility
            .plan
            .evolve_path_with_dividend_checks(
                &local_volatility.grid,
                knock_in.spot,
                &shocks,
                local_volatility.dividends.as_ref().expect("dividends"),
                local_volatility
                    .dividend_schedule
                    .as_ref()
                    .expect("dividend schedule"),
                PathIndex::new(0),
            )
            .expect("path");
        let observations = knock_in
            .local_vol_path_observations(local_volatility, &path, knock_in.spot)
            .expect("observations");
        assert!(observations[0].pre_dividend_spot.expect("pre spot") >= 95.0);
        assert!(
            observations
                .iter()
                .all(|observation| observation.post_spot < 95.0)
        );

        let knock_in_value = knock_in
            .local_vol_discounted_payoff_at_spot(
                local_volatility,
                knock_in.spot,
                &shocks,
                PathIndex::new(0),
            )
            .expect("knock-in value");
        let knock_out_runtime = knock_out
            .local_volatility
            .as_ref()
            .expect("Local Volatility");
        let knock_out_value = knock_out
            .local_vol_discounted_payoff_at_spot(
                knock_out_runtime,
                knock_out.spot,
                &shocks,
                PathIndex::new(0),
            )
            .expect("knock-out value");
        assert!(knock_in_value > 0.0);
        assert_eq!(knock_out_value.to_bits(), 0.0_f64.to_bits());
    }

    #[test]
    fn local_vol_barrier_dividend_jump_uses_event_node_and_reports_risks() {
        let risk = RiskRequest::new(
            true,
            Some(GammaConfig::new(
                SpotBump::relative(0.01).expect("gamma bump"),
            )),
            true,
            None,
            SmileDynamics::StickyLogMoneyness,
            None,
            None,
        )
        .expect("risk")
        .with_payoff_smoothing(PayoffSmoothing::compact_c2(5.0).expect("smoothing"));
        let request = local_vol_barrier_dividend_jump_request(BarrierStyle::KnockIn, risk);
        let plan = SimulationPlan::compile(&request, policy(2)).expect("plan");
        let local_volatility = plan.local_volatility.as_ref().expect("Local Volatility");
        let dividend_time = plan.observation_times[0];
        assert_eq!(local_volatility.grid.time_nodes(), [0.0, 1.0]);
        assert!(
            local_volatility
                .plan
                .time_grid()
                .node_index_for_time(dividend_time)
                .is_some()
        );
        assert_eq!(local_volatility.plan.time_grid().step_count(), 2);
        assert_eq!(plan.payoff.pre_dividend_observations().len(), 1);

        let result = plan.execute().expect("execution");
        assert!(result.pricing_result.risks.delta.is_some());
        assert!(result.pricing_result.risks.gamma.is_some());
        assert!(result.pricing_result.risks.vega.is_some());
        let diagnostics = result.diagnostics.payoff_smoothing.expect("smoothing");
        assert_eq!(diagnostics.endpoint_count, 2);
        assert_eq!(diagnostics.dividend_jump_count, 1);
    }

    #[test]
    fn smoothed_discrete_barrier_reports_all_risks_and_endpoint_diagnostics() {
        let base = barrier_zero_vol_request(120.0);
        let risk = RiskRequest::new(
            true,
            Some(GammaConfig::new(
                SpotBump::relative(0.01).expect("gamma bump"),
            )),
            true,
            None,
            SmileDynamics::StickyLogMoneyness,
            None,
            None,
        )
        .expect("risk")
        .with_payoff_smoothing(PayoffSmoothing::compact_c2(2.0).expect("smoothing"));
        let request = PricingRequest::new(
            base.valuation_date(),
            base.product().clone(),
            base.market().clone(),
            ModelSpec::BlackScholes(BlackScholesSpec::new(0.2).expect("model")),
            EngineConfig::PseudoMonteCarlo(
                PseudoMcConfig::new(7, 4096, VarianceReduction::new(true, false)).expect("engine"),
            ),
            risk,
        )
        .expect("request");

        let result = price_pseudo_monte_carlo(&request, policy(2)).expect("price");
        assert!(result.pricing_result.risks.delta.is_some());
        assert!(result.pricing_result.risks.gamma.is_some());
        assert!(result.pricing_result.risks.vega.is_some());
        assert!(result.risk_diagnostics.delta_validation.is_some());
        assert!(result.risk_diagnostics.gamma_validation.is_some());
        assert!(result.risk_diagnostics.vega_validation.is_some());
        let diagnostics = result.diagnostics.payoff_smoothing.expect("smoothing");
        assert_eq!(diagnostics.endpoint_count, 2);
        assert_eq!(diagnostics.dividend_jump_count, 0);
    }

    #[test]
    fn arithmetic_asian_zero_volatility_uses_all_declared_observation_dates() {
        let request = asian_zero_vol_request();
        let plan = SimulationPlan::compile(&request, policy(2)).expect("plan");
        assert_eq!(plan.observation_dates.len(), 2);
        let average = 0.25 * plan.observation_forwards[0] + 0.75 * plan.observation_forwards[1];
        let expected = plan.discount() * (average - 100.0) * 2.0;
        let result = plan.execute().expect("execution");
        assert!((result.pricing_result.value.value().get() - expected).abs() <= 1.0e-12);
        assert_eq!(result.sampling_variance.to_bits(), 0.0_f64.to_bits());
    }

    #[test]
    fn arithmetic_asian_price_respects_geometric_lower_and_convexity_upper_bounds() {
        let request =
            asian_risk_request(RiskRequest::price_only(SmileDynamics::StickyLogMoneyness));
        let plan = SimulationPlan::compile(&request, policy(2)).expect("plan");
        let weights = [0.25, 0.75];
        let weighted_log_mean = weights
            .iter()
            .zip(plan.observation_forwards.iter())
            .zip(plan.observation_times.iter())
            .map(|((&weight, &forward), &time)| {
                weight * (forward.ln() - 0.5 * plan.volatility.powi(2) * time)
            })
            .sum::<f64>();
        let mut weighted_minimum_time = 0.0;
        for (left, &left_weight) in weights.iter().enumerate() {
            for (right, &right_weight) in weights.iter().enumerate() {
                weighted_minimum_time += left_weight
                    * right_weight
                    * plan.observation_times[left].min(plan.observation_times[right]);
            }
        }
        let geometric_variance = plan.volatility.powi(2) * weighted_minimum_time;
        let geometric_forward = (weighted_log_mean + 0.5 * geometric_variance).exp();
        let geometric_lower = crate::analytical::evaluate_black_forward(
            crate::analytical::BlackForwardOracleInputs {
                side: OptionSide::Call,
                forward: geometric_forward,
                strike: 100.0,
                notional: 1.5,
                discount: plan.discount,
                volatility: geometric_variance.sqrt(),
                time: 1.0,
            },
        )
        .expect("geometric Asian lower bound")
        .price;
        let convexity_upper = weights
            .iter()
            .zip(plan.observation_forwards.iter())
            .zip(plan.observation_times.iter())
            .map(|((&weight, &forward), &time)| {
                weight
                    * crate::analytical::evaluate_black_forward(
                        crate::analytical::BlackForwardOracleInputs {
                            side: OptionSide::Call,
                            forward,
                            strike: 100.0,
                            notional: 1.5,
                            discount: plan.discount,
                            volatility: plan.volatility,
                            time,
                        },
                    )
                    .expect("European convexity upper bound")
                    .price
            })
            .sum::<f64>();

        let result = plan.execute().expect("Asian result");
        let estimate = result.pricing_result.value;
        let tolerance = 6.0 * estimate.standard_error().get() + 1.0e-12;
        assert!(estimate.value().get() + tolerance >= geometric_lower);
        assert!(estimate.value().get() - tolerance <= convexity_upper);
    }

    #[test]
    fn asian_and_lookback_dividend_collisions_observe_post_jump_spot() {
        let underlying = UnderlyingId::new(1);
        let currency = CurrencyId::new(1);
        let valuation_date: Date = "2026-09-04".parse().expect("valuation");
        let dividend_date: Date = "2027-03-05".parse().expect("dividend date");
        let expiry: Date = "2027-09-04".parse().expect("expiry");
        let event = EventId::new(1);
        let ex_time = DayCountConvention::Act365F.year_fraction(valuation_date, dividend_date);
        let forward = EquityForward::with_discrete_dividends(
            underlying,
            PositiveF64::new(100.0, "spot").expect("spot"),
            curve(1, 0.0),
            curve(2, 0.0),
            vec![
                DividendEvent::new(
                    event,
                    ex_time,
                    DividendQuote::fixed_cash(15.0, event).expect("cash dividend"),
                )
                .expect("dividend"),
            ],
        )
        .expect("forward");
        let market = MarketContext::Equity(EquityMarket::new(currency, forward));
        let products = [
            ProductSpec::ArithmeticAsian(
                ArithmeticAsianSpec::new(
                    underlying,
                    currency,
                    80.0,
                    1.0,
                    OptionSide::Call,
                    vec![
                        AsianObservation::unknown(dividend_date, 0.5).expect("first"),
                        AsianObservation::unknown(expiry, 0.5).expect("second"),
                    ],
                    expiry,
                )
                .expect("Asian"),
            ),
            ProductSpec::FixedLookback(
                FixedLookbackSpec::new(
                    underlying,
                    currency,
                    80.0,
                    1.0,
                    OptionSide::Call,
                    vec![dividend_date, expiry],
                    None,
                    expiry,
                )
                .expect("Lookback"),
            ),
        ];
        let engine = EngineConfig::PseudoMonteCarlo(
            PseudoMcConfig::new(7, 1, VarianceReduction::new(false, false)).expect("engine"),
        );
        for product in products {
            let request = PricingRequest::new(
                valuation_date,
                product,
                market.clone(),
                ModelSpec::BlackScholes(BlackScholesSpec::new(0.0).expect("model")),
                engine,
                RiskRequest::price_only(SmileDynamics::StickyLogMoneyness),
            )
            .expect("request");
            let plan = SimulationPlan::compile(&request, policy(2)).expect("plan");
            let observations = plan.path_observations_from_normals(&[0.0, 0.0], 100.0, 0.0);
            assert_eq!(observations.len(), 2);
            assert!((observations[0].post_spot - 85.0).abs() <= 1.0e-12);
            assert!(observations[0].pre_dividend_spot.is_none());
            let result = plan.execute().expect("execution");
            assert!((result.pricing_result.value.value().get() - 5.0).abs() <= 1.0e-12);
        }
    }

    #[test]
    fn fully_fixed_arithmetic_asian_prices_as_discounted_cashflow() {
        for engine in [fixed_payoff_pseudo_engine(), fixed_payoff_rqmc_engine()] {
            let request = fully_fixed_asian_request(engine);
            let plan = SimulationPlan::compile(&request, policy(2)).expect("plan");
            assert!(plan.observation_dates.is_empty());
            let result = plan.execute().expect("execution");
            let expected = plan.discount() * 20.0;
            assert!((result.pricing_result.value.value().get() - expected).abs() <= 1.0e-12);
            assert_eq!(result.sampling_variance.to_bits(), 0.0_f64.to_bits());
            assert_eq!(
                result.pricing_result.value.standard_error().get().to_bits(),
                0.0_f64.to_bits()
            );
        }
    }

    #[test]
    fn fixed_lookback_zero_volatility_uses_declared_monitoring_extremum() {
        let request = lookback_zero_vol_request();
        let plan = SimulationPlan::compile(&request, policy(2)).expect("plan");
        assert_eq!(plan.observation_dates.len(), 2);
        let maximum = plan
            .observation_forwards
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let expected = plan.discount() * (maximum - 100.0) * 2.0;
        let result = plan.execute().expect("execution");
        assert!((result.pricing_result.value.value().get() - expected).abs() <= 1.0e-12);
        assert_eq!(result.sampling_variance.to_bits(), 0.0_f64.to_bits());
    }

    #[test]
    fn fully_fixed_lookback_prices_as_discounted_cashflow() {
        for engine in [fixed_payoff_pseudo_engine(), fixed_payoff_rqmc_engine()] {
            let request = fully_fixed_lookback_request(engine);
            let plan = SimulationPlan::compile(&request, policy(2)).expect("plan");
            assert!(plan.observation_dates.is_empty());
            let result = plan.execute().expect("execution");
            let expected = plan.discount() * 40.0;
            assert!((result.pricing_result.value.value().get() - expected).abs() <= 1.0e-12);
            assert_eq!(result.sampling_variance.to_bits(), 0.0_f64.to_bits());
            assert_eq!(
                result.pricing_result.value.standard_error().get().to_bits(),
                0.0_f64.to_bits()
            );
        }
    }

    #[test]
    fn fully_fixed_asian_and_lookback_report_exact_zero_market_risks() {
        for engine in [fixed_payoff_pseudo_engine(), fixed_payoff_rqmc_engine()] {
            for request in [
                fully_fixed_asian_request_with_risk(engine, all_risks()),
                fully_fixed_lookback_request_with_risk(engine, all_risks()),
            ] {
                let result = price_monte_carlo(&request, policy(2)).expect("execution");
                assert_eq!(result.sampling_variance.to_bits(), 0.0_f64.to_bits());
                for estimate in [
                    result.pricing_result.risks.delta.expect("Delta").raw(),
                    result.pricing_result.risks.gamma.expect("Gamma").raw(),
                    result.pricing_result.risks.vega.expect("Vega").raw(),
                ] {
                    assert_eq!(estimate.value().get().to_bits(), 0.0_f64.to_bits());
                    assert_eq!(estimate.standard_error().get().to_bits(), 0.0_f64.to_bits());
                }
            }
        }
    }

    #[test]
    fn path_state_diagnostics_retain_fixed_asian_and_lookback_state() {
        let asian = price_monte_carlo(
            &partially_fixed_asian_request(
                90.0,
                ModelSpec::BlackScholes(BlackScholesSpec::new(0.2).expect("model")),
            ),
            policy(1),
        )
        .expect("Asian");
        assert_eq!(
            asian.diagnostics.path_state,
            Some(PathStateDiagnostics::ArithmeticAsian {
                known_observation_count: 1,
                unknown_observation_count: 1,
                known_weight_sum: 0.4,
                unknown_weight_sum: 0.6,
                weighted_known_fixing_sum: 36.0,
            })
        );

        let lookback = price_monte_carlo(
            &partially_fixed_lookback_request(
                92.0,
                ModelSpec::BlackScholes(BlackScholesSpec::new(0.2).expect("model")),
            ),
            policy(1),
        )
        .expect("Lookback");
        assert_eq!(
            lookback.diagnostics.path_state,
            Some(PathStateDiagnostics::FixedLookback {
                past_monitoring_count: 1,
                future_monitoring_count: 1,
                historical_extremum: Some(92.0),
            })
        );
    }

    #[test]
    fn partially_fixed_history_is_invariant_under_all_market_risks() {
        for model in [
            ModelSpec::BlackScholes(BlackScholesSpec::new(0.2).expect("model")),
            constant_local_vol_model(),
        ] {
            let request_pairs = [
                (
                    partially_fixed_asian_request(90.0, model.clone()),
                    partially_fixed_asian_request(110.0, model.clone()),
                    8.0,
                ),
                (
                    partially_fixed_lookback_request(1.0, model.clone()),
                    partially_fixed_lookback_request(2.0, model.clone()),
                    -1.0,
                ),
            ];
            for (first_request, second_request, expected_price_difference) in request_pairs {
                let first_plan =
                    SimulationPlan::compile(&first_request, policy(2)).expect("first plan");
                let second_plan =
                    SimulationPlan::compile(&second_request, policy(2)).expect("second plan");
                assert_eq!(first_plan.observation_dates.len(), 1);
                assert_eq!(second_plan.observation_dates.len(), 1);
                let first = first_plan.execute().expect("first execution");
                let second = second_plan.execute().expect("second execution");
                let price_difference = second.pricing_result.value.value().get()
                    - first.pricing_result.value.value().get();
                assert!((price_difference - expected_price_difference).abs() < 2.0e-12);

                let first_risks = [
                    first.pricing_result.risks.delta.expect("Delta").raw(),
                    first.pricing_result.risks.gamma.expect("Gamma").raw(),
                    first.pricing_result.risks.vega.expect("Vega").raw(),
                ];
                let second_risks = [
                    second.pricing_result.risks.delta.expect("Delta").raw(),
                    second.pricing_result.risks.gamma.expect("Gamma").raw(),
                    second.pricing_result.risks.vega.expect("Vega").raw(),
                ];
                for (first_risk, second_risk) in first_risks.into_iter().zip(second_risks) {
                    assert!((first_risk.value().get() - second_risk.value().get()).abs() < 2.0e-12);
                    assert!(
                        (first_risk.standard_error().get() - second_risk.standard_error().get())
                            .abs()
                            < 2.0e-14
                    );
                }
            }
        }
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
        let diagnostics = &result.risk_diagnostics;
        assert_eq!(diagnostics.methods.delta, Some(RiskMethod::AadReverse));
        assert_eq!(
            diagnostics.methods.gamma,
            Some(RiskMethod::CentralBumpOfAadDelta)
        );
        assert_eq!(diagnostics.methods.vega, Some(RiskMethod::AadReverse));
        assert_eq!(diagnostics.methods.gamma_spot_bump, Some(1.0));
        assert_eq!(diagnostics.methods.validation_spot_bump, Some(1.0));
        assert_eq!(diagnostics.methods.bump_policy_version, 1);
        for validation in [
            diagnostics.delta_validation.expect("delta validation"),
            diagnostics.gamma_validation.expect("gamma validation"),
            diagnostics.vega_validation.expect("vega validation"),
        ] {
            assert!(validation.bump_and_revalue.value().get().is_finite());
            assert!(validation.bump_minus_primary.value().get().is_finite());
            assert!(
                validation
                    .bump_minus_primary
                    .standard_error()
                    .get()
                    .is_finite()
            );
        }
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
            price.pricing_result.value.standard_error().get().to_bits(),
            risk.pricing_result.value.standard_error().get().to_bits()
        );
    }

    #[test]
    fn asian_risks_do_not_change_seeded_price_bits() {
        let price_only =
            asian_risk_request(RiskRequest::price_only(SmileDynamics::StickyLogMoneyness));
        let with_risks = asian_risk_request(all_risks());
        let price = price_pseudo_monte_carlo(&price_only, policy(3)).expect("price");
        let risk = price_pseudo_monte_carlo(&with_risks, policy(3)).expect("risk");
        assert_eq!(
            price.pricing_result.value.value().to_bits(),
            risk.pricing_result.value.value().to_bits()
        );
        assert_eq!(
            price.pricing_result.value.standard_error().get().to_bits(),
            risk.pricing_result.value.standard_error().get().to_bits()
        );
    }

    #[test]
    fn asian_and_lookback_pathwise_risks_are_reported() {
        for request in [asian_risk_request(all_risks()), lookback_risk_request()] {
            let result = price_pseudo_monte_carlo(&request, policy(4)).expect("risk");
            let risks = &result.pricing_result.risks;
            for estimate in [
                risks.delta.expect("delta").raw(),
                risks.gamma.expect("gamma").raw(),
                risks.vega.expect("vega").raw(),
            ] {
                assert!(estimate.value().get().is_finite());
                assert!(estimate.standard_error().get().is_finite());
            }
            let diagnostics = &result.risk_diagnostics;
            assert_eq!(diagnostics.methods.delta, Some(RiskMethod::AadReverse));
            assert_eq!(
                diagnostics.methods.gamma,
                Some(RiskMethod::CentralBumpOfAadDelta)
            );
            assert_eq!(diagnostics.methods.vega, Some(RiskMethod::AadReverse));
            assert!(
                diagnostics
                    .delta_validation
                    .expect("delta validation")
                    .bump_minus_primary
                    .value()
                    .get()
                    .abs()
                    < 2.0e-3
            );
            assert!(
                diagnostics
                    .vega_validation
                    .expect("vega validation")
                    .bump_minus_primary
                    .value()
                    .get()
                    .abs()
                    < 2.0e-2
            );
        }
    }

    #[test]
    fn asian_and_lookback_local_vol_risks_match_constant_variance_limit() {
        for black_scholes in [asian_risk_request(all_risks()), lookback_risk_request()] {
            let local_vol = PricingRequest::new(
                black_scholes.valuation_date(),
                black_scholes.product().clone(),
                black_scholes.market().clone(),
                constant_local_vol_model_with_volatility(0.25),
                black_scholes.engine(),
                black_scholes.risk().clone(),
            )
            .expect("Local Volatility request");
            let black_scholes_result =
                price_monte_carlo(&black_scholes, policy(4)).expect("Black-Scholes execution");
            let local_vol_result =
                price_monte_carlo(&local_vol, policy(4)).expect("Local Volatility execution");
            let black_scholes_values = [
                black_scholes_result.pricing_result.value.value().get(),
                black_scholes_result
                    .pricing_result
                    .risks
                    .delta
                    .expect("Delta")
                    .raw()
                    .value()
                    .get(),
                black_scholes_result
                    .pricing_result
                    .risks
                    .gamma
                    .expect("Gamma")
                    .raw()
                    .value()
                    .get(),
                black_scholes_result
                    .pricing_result
                    .risks
                    .vega
                    .expect("Vega")
                    .raw()
                    .value()
                    .get(),
            ];
            let local_vol_values = [
                local_vol_result.pricing_result.value.value().get(),
                local_vol_result
                    .pricing_result
                    .risks
                    .delta
                    .expect("Delta")
                    .raw()
                    .value()
                    .get(),
                local_vol_result
                    .pricing_result
                    .risks
                    .gamma
                    .expect("Gamma")
                    .raw()
                    .value()
                    .get(),
                local_vol_result
                    .pricing_result
                    .risks
                    .vega
                    .expect("Vega")
                    .raw()
                    .value()
                    .get(),
            ];
            for ((black_scholes_value, local_vol_value), tolerance) in black_scholes_values
                .into_iter()
                .zip(local_vol_values)
                .zip([2.0e-10, 2.0e-3, 2.0e-3, 2.0e-8])
            {
                assert!(
                    (local_vol_value - black_scholes_value).abs() < tolerance,
                    "Black-Scholes={black_scholes_value}, Local Volatility={local_vol_value}, tolerance={tolerance}"
                );
            }
        }
    }

    #[test]
    fn asian_pathwise_risks_work_under_rqmc() {
        let result = price_monte_carlo(&asian_rqmc_risk_request(), policy(4)).expect("RQMC risk");
        assert_eq!(
            result.pricing_result.value.estimator(),
            EstimatorKind::RandomizedQuasiMonteCarlo
        );
        assert_eq!(result.independent_sampling_units, 8);
        let risks = &result.pricing_result.risks;
        for estimate in [
            risks.delta.expect("delta").raw(),
            risks.gamma.expect("gamma").raw(),
            risks.vega.expect("vega").raw(),
        ] {
            assert!(estimate.value().get().is_finite());
            assert!(estimate.standard_error().get().is_finite());
            assert_eq!(
                estimate.estimator(),
                EstimatorKind::RandomizedQuasiMonteCarlo
            );
        }
    }

    #[test]
    fn price_only_result_has_no_risk_methods_or_validations() {
        let request = request(OptionSide::Call, 100.0, 0.2, 1024, true);
        let result = price_pseudo_monte_carlo(&request, policy(2)).expect("price");
        assert_eq!(result.risk_diagnostics.methods.delta, None);
        assert_eq!(result.risk_diagnostics.methods.gamma, None);
        assert_eq!(result.risk_diagnostics.methods.vega, None);
        assert_eq!(result.risk_diagnostics.delta_validation, None);
        assert_eq!(result.risk_diagnostics.gamma_validation, None);
        assert_eq!(result.risk_diagnostics.vega_validation, None);
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
            &request(OptionSide::Call, 100.0, 0.2 - volatility_bump, units, true),
            policy(4),
        )
        .expect("down volatility");
        let up_vol = price_pseudo_monte_carlo(
            &request(OptionSide::Call, 100.0, 0.2 + volatility_bump, units, true),
            policy(4),
        )
        .expect("up volatility");
        let bump_vega = (up_vol.pricing_result.value.value().get()
            - down_vol.pricing_result.value.value().get())
            / (2.0 * volatility_bump);
        let risks = &aad.pricing_result.risks;
        assert!((risks.delta.expect("delta").raw().value().get() - bump_delta).abs() < 5.0e-4);
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
        let request =
            request_with_spot_and_risk(OptionSide::Call, 100.0, 0.2, 1024, true, 100.0, risk);
        assert!(matches!(
            SimulationPlan::compile(&request, policy(2)),
            Err(MonteCarloError::InvalidGammaBump { .. })
        ));
    }

    #[test]
    fn rqmc_price_and_error_use_independent_scrambles() {
        let request = rqmc_request(
            0x8877_6655_4433_2211,
            4096,
            16,
            true,
            RiskRequest::price_only(SmileDynamics::StickyLogMoneyness),
        );
        let oracle = black_scholes_oracle(&request).expect("oracle").price;
        let result = price_monte_carlo(&request, policy(4)).expect("RQMC");
        let estimate = result.pricing_result.value;
        assert_eq!(
            estimate.estimator(),
            EstimatorKind::RandomizedQuasiMonteCarlo
        );
        assert_eq!(estimate.effective_sampling_units().get(), 16);
        assert_eq!(result.independent_sampling_units, 16);
        assert_eq!(result.evaluated_paths, 4096 * 16 * 2);
        assert_eq!(result.diagnostics.scramble_count, Some(16));
        assert!(result.diagnostics.direction_checksum.is_some());
        assert!(result.diagnostics.scramble_checksum.is_some());
        assert!(
            (estimate.value().get() - oracle).abs()
                <= 8.0 * estimate.standard_error().get() + 2.0e-5
        );
    }

    #[test]
    fn rqmc_risks_replay_across_worker_counts() {
        let request = rqmc_request(42, 1024, 8, true, all_risks());
        let single = price_monte_carlo(&request, policy(1)).expect("single");
        let parallel = price_monte_carlo(&request, policy(4)).expect("parallel");
        assert_eq!(single.pricing_result, parallel.pricing_result);
        assert_eq!(
            single.estimator_variance.to_bits(),
            parallel.estimator_variance.to_bits()
        );
        assert_eq!(
            single.diagnostics.scramble_checksum,
            parallel.diagnostics.scramble_checksum
        );
        for estimate in [
            single.pricing_result.risks.delta.expect("delta").raw(),
            single.pricing_result.risks.gamma.expect("gamma").raw(),
            single.pricing_result.risks.vega.expect("vega").raw(),
        ] {
            assert_eq!(
                estimate.estimator(),
                EstimatorKind::RandomizedQuasiMonteCarlo
            );
            assert_eq!(estimate.effective_sampling_units().get(), 8);
        }
    }

    #[test]
    fn rqmc_seed_changes_randomization_but_replays_exactly() {
        let risk = RiskRequest::price_only(SmileDynamics::StickyLogMoneyness);
        let first = price_monte_carlo(&rqmc_request(1, 1024, 4, false, risk.clone()), policy(2))
            .expect("first");
        let replay = price_monte_carlo(&rqmc_request(1, 1024, 4, false, risk.clone()), policy(2))
            .expect("replay");
        let changed = price_monte_carlo(&rqmc_request(2, 1024, 4, false, risk), policy(2))
            .expect("changed seed");
        assert_eq!(first.pricing_result, replay.pricing_result);
        assert_eq!(
            first.diagnostics.scramble_checksum,
            replay.diagnostics.scramble_checksum
        );
        assert_ne!(
            first.diagnostics.scramble_checksum,
            changed.diagnostics.scramble_checksum
        );
        assert_ne!(
            first.pricing_result.value.value().to_bits(),
            changed.pricing_result.value.value().to_bits()
        );
    }

    #[test]
    fn pseudo_only_entry_point_rejects_rqmc() {
        let request = rqmc_request(
            7,
            8,
            2,
            false,
            RiskRequest::price_only(SmileDynamics::StickyLogMoneyness),
        );
        assert!(matches!(
            price_pseudo_monte_carlo(&request, policy(1)),
            Err(MonteCarloError::UnsupportedEngine)
        ));
    }
}
