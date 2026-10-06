use super::*;

use crate::mc::{PseudoMcConfig, RqmcConfig, VarianceReduction};

fn training_metadata(trajectory_count: u64) -> ExercisePolicyTrainingMetadata {
    ExercisePolicyTrainingMetadata::new(
        [0x11; 32],
        [0x22; 32],
        7,
        trajectory_count,
        trajectory_count,
    )
    .expect("training metadata")
}

fn lsm_config(training_engine: EngineConfig, max_degree: u32) -> LsmConfig {
    LsmConfig::new(
        training_engine,
        vec![LsmStateVariable::Spot],
        PolynomialBasisSpec::new(1, max_degree, 16, 16).expect("basis"),
        1.0e-12,
        CpqrConfig::new(1.0e-14, 1.0e-12).expect("CPQR config"),
        1_000_000,
    )
    .expect("LSM config")
}

#[test]
fn lsm_config_preserves_training_count_semantics() {
    let pseudo = lsm_config(
        EngineConfig::PseudoMonteCarlo(
            PseudoMcConfig::new(7, 100, VarianceReduction::new(true, false))
                .expect("pseudo config"),
        ),
        2,
    );
    assert_eq!(pseudo.training_seed(), 7);
    assert_eq!(pseudo.training_effective_sampling_units(), 100);
    assert_eq!(pseudo.training_trajectory_count(), 200);
    assert_eq!(
        pseudo.training_random_domain(),
        crate::mc::RandomDomain::LsmTrain
    );

    let rqmc = lsm_config(
        EngineConfig::RandomizedQuasiMonteCarlo(
            RqmcConfig::new(1024, 8, 11, VarianceReduction::new(true, true)).expect("RQMC config"),
        ),
        2,
    );
    assert_eq!(rqmc.training_seed(), 11);
    assert_eq!(rqmc.training_effective_sampling_units(), 8);
    assert_eq!(rqmc.training_trajectory_count(), 16_384);
    assert_eq!(
        rqmc.training_random_domain(),
        crate::mc::RandomDomain::RqmcScramble
    );
    let metadata = rqmc.training_metadata([0x42; 32]).expect("metadata");
    assert_eq!(metadata.seed(), 11);
    assert_eq!(metadata.sampling_units(), 8);
    assert_eq!(metadata.trajectory_count(), 16_384);
    assert_eq!(
        metadata.random_domain(),
        crate::mc::RandomDomain::RqmcScramble
    );
}

#[test]
fn lsm_config_validates_state_variable_contract() {
    let engine = EngineConfig::PseudoMonteCarlo(
        PseudoMcConfig::new(7, 100, VarianceReduction::new(false, false)).expect("pseudo config"),
    );
    let basis = PolynomialBasisSpec::new(1, 2, 16, 16).expect("basis");
    assert!(matches!(
        LsmConfig::new(
            engine,
            Vec::new(),
            basis.clone(),
            0.0,
            CpqrConfig::new(0.0, 0.0).expect("CPQR config"),
            16,
        ),
        Err(LsmNumericalError::EmptyStateVariables)
    ));
    assert!(matches!(
        LsmConfig::new(
            engine,
            vec![LsmStateVariable::Spot, LsmStateVariable::Spot],
            basis,
            0.0,
            CpqrConfig::new(0.0, 0.0).expect("CPQR config"),
            16,
        ),
        Err(LsmNumericalError::DuplicateStateVariable)
    ));
}

#[test]
fn training_metadata_rejects_non_training_random_domains() {
    assert!(matches!(
        ExercisePolicyTrainingMetadata::new_with_random_domain(
            [0x11; 32],
            [0x22; 32],
            7,
            2,
            2,
            crate::mc::RandomDomain::Valuation,
        ),
        Err(LsmNumericalError::InvalidTrainingRandomDomain {
            domain: crate::mc::RandomDomain::Valuation
        })
    ));
}

#[test]
fn lsm_config_rejects_overlapping_rqmc_scrambles() {
    let training = lsm_config(
        EngineConfig::RandomizedQuasiMonteCarlo(
            RqmcConfig::new(1024, 8, 11, VarianceReduction::new(false, true)).expect("RQMC config"),
        ),
        2,
    );
    let overlapping = EngineConfig::RandomizedQuasiMonteCarlo(
        RqmcConfig::new(2048, 16, 11, VarianceReduction::new(true, true)).expect("RQMC config"),
    );
    let independent = EngineConfig::RandomizedQuasiMonteCarlo(
        RqmcConfig::new(2048, 16, 12, VarianceReduction::new(true, true)).expect("RQMC config"),
    );
    assert_eq!(
        training.validate_independent_from(overlapping),
        Err(LsmNumericalError::TrainingValuationPathOverlap {
            master_scramble_seed: 11
        })
    );
    assert_eq!(training.validate_independent_from(independent), Ok(()));

    let pseudo_training = lsm_config(
        EngineConfig::PseudoMonteCarlo(
            PseudoMcConfig::new(11, 100, VarianceReduction::new(false, false))
                .expect("pseudo config"),
        ),
        2,
    );
    let pseudo_valuation = EngineConfig::PseudoMonteCarlo(
        PseudoMcConfig::new(11, 200, VarianceReduction::new(false, false)).expect("pseudo config"),
    );
    assert_eq!(
        pseudo_training.validate_independent_from(pseudo_valuation),
        Ok(())
    );
}

#[test]
fn lsm_configuration_fingerprint_is_canonical_and_complete() {
    let engine = EngineConfig::PseudoMonteCarlo(
        PseudoMcConfig::new(7, 100, VarianceReduction::new(true, false)).expect("pseudo config"),
    );
    let first = lsm_config(engine, 2);
    let replay = lsm_config(engine, 2);
    let changed_basis = lsm_config(engine, 3);
    let changed_seed = lsm_config(
        EngineConfig::PseudoMonteCarlo(
            PseudoMcConfig::new(8, 100, VarianceReduction::new(true, false))
                .expect("pseudo config"),
        ),
        2,
    );
    assert_eq!(first.fingerprint(), replay.fingerprint());
    assert_ne!(first.fingerprint(), changed_basis.fingerprint());
    assert_ne!(first.fingerprint(), changed_seed.fingerprint());
    assert_eq!(
        first.fingerprint().to_string(),
        "blake3-256:9c729020cfd000b95c4dd11e46faffdd1e9a8eaf3fdf4e2788e9346cb131cfd2"
    );
}

struct CpqrCase<'a> {
    matrix: &'a [f64],
    rows: usize,
    columns: usize,
    target: &'a [f64],
    pivots: &'a [usize],
    rank: usize,
    coefficients: &'a [f64],
    residual: f64,
}

#[test]
fn polynomial_basis_matches_e0_order_and_evaluation() {
    let basis = PolynomialBasisSpec::new(2, 2, 16, 32).expect("basis");
    assert_eq!(basis.feature_count(), 2);
    assert_eq!(basis.max_degree(), 2);
    assert_eq!(
        basis.exponents(),
        [
            Box::from([0, 0]),
            Box::from([1, 0]),
            Box::from([0, 1]),
            Box::from([2, 0]),
            Box::from([1, 1]),
            Box::from([0, 2]),
        ]
    );
    assert_eq!(
        basis.evaluate(&[2.0, 3.0]).expect("basis values").as_ref(),
        [1.0, 2.0, 3.0, 4.0, 6.0, 9.0]
    );
}

#[test]
fn polynomial_basis_checks_resource_limits_and_inputs() {
    assert!(matches!(
        PolynomialBasisSpec::new(2, 2, 5, 32),
        Err(LsmNumericalError::BasisLimitExceeded {
            resource: "columns",
            requested: 6,
            maximum: 5,
        })
    ));
    assert!(matches!(
        PolynomialBasisSpec::new(2, 2, 6, 11),
        Err(LsmNumericalError::BasisLimitExceeded {
            resource: "exponents",
            requested: 12,
            maximum: 11,
        })
    ));
    let constant = PolynomialBasisSpec::new(0, 4, 1, 1).expect("constant basis");
    assert_eq!(constant.exponents(), [Box::<[u32]>::default()]);
    assert_eq!(
        constant.evaluate(&[]).expect("constant value").as_ref(),
        [1.0]
    );
}

#[test]
fn feature_scaling_matches_e0_fixture() {
    let scaling = FeatureScaling::fit(&[-1.0, 0.0, 1.0]).expect("scaling");
    assert_eq!(scaling.mean(), 0.0);
    assert_eq!(scaling.population_variance(), 2.0 / 3.0);
    assert_eq!(scaling.scale(), (2.0_f64 / 3.0).sqrt());
    assert_eq!(scaling.zero_scale_threshold(), 64.0 * f64::EPSILON);
    assert!(!scaling.inactive());
    assert_eq!(
        scaling.standardize(1.0).expect("standardized"),
        1.0 / (2.0_f64 / 3.0).sqrt()
    );

    let constant = FeatureScaling::fit(&[5.0, 5.0, 5.0]).expect("constant");
    assert!(constant.inactive());
    assert_eq!(constant.scale(), 0.0);
    assert_eq!(
        constant.standardize(5.0),
        Err(LsmNumericalError::InactiveFeature)
    );
}

#[test]
fn exercise_and_itm_comparisons_are_strict_and_finite() {
    assert!(!should_exercise(2.0, 2.0).expect("tie"));
    assert!(should_exercise(2.0001, 2.0).expect("exercise"));
    assert!(!is_training_itm(0.01, 0.01).expect("ITM boundary"));
    assert!(is_training_itm(0.0101, 0.01).expect("ITM"));
    assert!(matches!(
        should_exercise(f64::NAN, 1.0),
        Err(LsmNumericalError::NonFiniteDecisionValue { .. })
    ));
    assert!(matches!(
        is_training_itm(1.0, f64::INFINITY),
        Err(LsmNumericalError::InvalidTolerance { .. })
    ));
}

#[test]
fn cpqr_matches_e0_reference_cases() {
    let exact = CpqrConfig::new(0.0, 0.0).expect("config");
    let cases = [
        CpqrCase {
            matrix: &[1.0, -1.0, 1.0, 0.0, 1.0, 1.0],
            rows: 3,
            columns: 2,
            target: &[1.0, 2.0, 3.0],
            pivots: &[0, 1],
            rank: 2,
            coefficients: &[2.0, 1.0],
            residual: 0.0,
        },
        CpqrCase {
            matrix: &[1.0, 2.0, 0.0, 0.0],
            rows: 2,
            columns: 2,
            target: &[4.0, 0.0],
            pivots: &[1, 0],
            rank: 1,
            coefficients: &[0.0, 2.0],
            residual: 0.0,
        },
        CpqrCase {
            matrix: &[1.0, 0.0, 0.0, 1.0],
            rows: 2,
            columns: 2,
            target: &[2.0, 3.0],
            pivots: &[0, 1],
            rank: 2,
            coefficients: &[2.0, 3.0],
            residual: 0.0,
        },
    ];
    for case in cases {
        let fit = fit_cpqr(case.matrix, case.rows, case.columns, case.target, exact).expect("fit");
        assert_eq!(fit.pivot_order(), case.pivots);
        assert_eq!(fit.rank(), case.rank);
        for (&actual, &expected) in fit.coefficients().iter().zip(case.coefficients) {
            assert!((actual - expected).abs() < 1.0e-13);
        }
        assert!((fit.residual_sum_squares() - case.residual).abs() < 1.0e-26);
    }

    let fit = fit_cpqr(
        &[2.0, 0.0, 0.0, 0.0001],
        2,
        2,
        &[4.0, 1.0],
        CpqrConfig::new(0.001, 0.0).expect("rank config"),
    )
    .expect("rank-deficient fit");
    assert_eq!(fit.pivot_order(), [0, 1]);
    assert_eq!(fit.rank(), 1);
    assert_eq!(fit.coefficients(), [2.0, 0.0]);
    assert_eq!(fit.residual_sum_squares(), 1.0);
}

#[test]
fn cpqr_rejects_invalid_shapes_values_and_tolerances() {
    assert!(matches!(
        CpqrConfig::new(-1.0, 0.0),
        Err(LsmNumericalError::InvalidTolerance { .. })
    ));
    let config = CpqrConfig::new(0.0, 0.0).expect("config");
    assert_eq!(
        fit_cpqr(&[], 0, 1, &[], config),
        Err(LsmNumericalError::ZeroMatrixRows)
    );
    assert!(matches!(
        fit_cpqr(&[f64::NAN], 1, 1, &[1.0], config),
        Err(LsmNumericalError::NonFiniteMatrixValue { .. })
    ));
    assert!(matches!(
        fit_cpqr(&[1.0], 1, 1, &[f64::INFINITY], config),
        Err(LsmNumericalError::NonFiniteTargetValue { .. })
    ));
}

#[test]
fn polynomial_regression_scales_features_and_predicts_in_original_basis_order() {
    let basis = PolynomialBasisSpec::new(1, 1, 4, 4).expect("basis");
    let model = fit_polynomial_regression(
        basis,
        &[1.0, 2.0, 3.0],
        3,
        &[3.0, 5.0, 7.0],
        CpqrConfig::new(0.0, 0.0).expect("config"),
        16,
    )
    .expect("model");
    assert_eq!(model.active_basis_columns(), [0, 1]);
    assert!(model.pre_excluded_basis_columns().is_empty());
    assert_eq!(model.rank(), 2);
    assert!(model.rank_excluded_basis_columns().is_empty());
    assert!((model.coefficients()[0] - 5.0).abs() < 1.0e-13);
    assert!((model.coefficients()[1] - 2.0 * (2.0_f64 / 3.0).sqrt()).abs() < 1.0e-13);
    assert!(model.residual_sum_squares() < 1.0e-26);
    assert!((model.predict(&[4.0]).expect("prediction") - 9.0).abs() < 1.0e-13);
}

#[test]
fn polynomial_regression_pre_excludes_inactive_feature_columns() {
    let basis = PolynomialBasisSpec::new(1, 2, 4, 4).expect("basis");
    let model = fit_polynomial_regression(
        basis,
        &[5.0, 5.0, 5.0],
        3,
        &[1.0, 2.0, 3.0],
        CpqrConfig::new(0.0, 0.0).expect("config"),
        16,
    )
    .expect("model");
    assert_eq!(model.active_basis_columns(), [0]);
    assert_eq!(model.pre_excluded_basis_columns(), [1, 2]);
    assert_eq!(model.pivot_order(), [0]);
    assert_eq!(model.rank(), 1);
    assert_eq!(model.coefficients(), [2.0, 0.0, 0.0]);
    assert_eq!(model.residual_sum_squares(), 2.0);
    assert_eq!(model.predict(&[5.0]).expect("prediction"), 2.0);
}

#[test]
fn polynomial_regression_checks_matrix_shape_and_limit_before_allocation() {
    let basis = PolynomialBasisSpec::new(1, 1, 4, 4).expect("basis");
    assert_eq!(
        fit_polynomial_regression(
            basis.clone(),
            &[1.0],
            2,
            &[1.0, 2.0],
            CpqrConfig::new(0.0, 0.0).expect("config"),
            4,
        ),
        Err(LsmNumericalError::FeatureMatrixLengthMismatch {
            expected: 2,
            actual: 1,
        })
    );
    assert_eq!(
        fit_polynomial_regression(
            basis,
            &[1.0, 2.0],
            2,
            &[1.0, 2.0],
            CpqrConfig::new(0.0, 0.0).expect("config"),
            3,
        ),
        Err(LsmNumericalError::MatrixElementLimitExceeded {
            requested: 4,
            maximum: 3,
        })
    );
    let constant = PolynomialBasisSpec::new(0, 0, 1, 1).expect("constant basis");
    assert_eq!(
        fit_polynomial_regression(
            constant,
            &[],
            4,
            &[1.0, 1.0, 1.0, 1.0],
            CpqrConfig::new(0.0, 0.0).expect("config"),
            3,
        ),
        Err(LsmNumericalError::MatrixElementLimitExceeded {
            requested: 4,
            maximum: 3,
        })
    );
}

#[test]
fn exercise_decision_fits_only_strictly_itm_training_rows() {
    let fit = fit_exercise_decision(
        PolynomialBasisSpec::new(1, 1, 4, 4).expect("basis"),
        &[100.0, 90.0, 80.0, 110.0],
        4,
        &[0.0, 10.0, 20.0, 0.0],
        &[1.0, 9.0, 15.0, 2.0],
        0.0,
        CpqrConfig::new(0.0, 0.0).expect("config"),
        16,
    )
    .expect("date-local fit");
    assert_eq!(fit.diagnostics().candidate_rows(), 4);
    assert_eq!(fit.diagnostics().itm_rows(), 2);
    assert_eq!(fit.diagnostics().feature_count(), 1);
    assert!(fit.diagnostics().warnings().is_empty());
    let ExerciseDecisionModel::Regression(model) = fit.decision() else {
        panic!("expected regression");
    };
    assert_eq!(model.feature_scalings()[0].mean(), 85.0);
    assert!((model.predict(&[90.0]).expect("prediction") - 9.0).abs() < 1.0e-13);
    assert!((model.predict(&[80.0]).expect("prediction") - 15.0).abs() < 1.0e-13);
}

#[test]
fn exercise_decision_stores_continue_all_for_zero_itm_rows() {
    let fit = fit_exercise_decision(
        PolynomialBasisSpec::new(1, 2, 4, 4).expect("basis"),
        &[90.0, 100.0, 110.0],
        3,
        &[0.0, 0.01, 0.009],
        &[1.0, 2.0, 3.0],
        0.01,
        CpqrConfig::new(0.0, 0.0).expect("config"),
        16,
    )
    .expect("continue-all fit");
    assert_eq!(fit.diagnostics().itm_rows(), 0);
    assert_eq!(
        fit.decision(),
        &ExerciseDecisionModel::ContinueAll {
            reason: ContinueAllReason::ZeroItmTrainingPaths,
        }
    );
    assert!(
        !fit.decision()
            .should_exercise(100.0, &[100.0])
            .expect("ContinueAll decision")
    );
    assert_eq!(
        fit.diagnostics().warnings(),
        [LsmWarning::ZeroItmTrainingPaths]
    );
}

#[test]
fn exercise_decision_reports_inactive_features_and_rank_exclusions() {
    let fit = fit_exercise_decision(
        PolynomialBasisSpec::new(2, 2, 8, 16).expect("basis"),
        &[5.0, 1.0, 5.0, 2.0],
        2,
        &[1.0, 1.0],
        &[2.0, 3.0],
        0.0,
        CpqrConfig::new(0.0, 0.0).expect("config"),
        16,
    )
    .expect("rank-deficient fit");
    assert_eq!(
        fit.diagnostics().warnings()[0],
        LsmWarning::InactiveFeature { feature: 0 }
    );
    assert!(matches!(
        fit.diagnostics().warnings()[1],
        LsmWarning::RankExcludedBasisColumn { .. }
    ));
}

#[test]
fn exercise_decision_application_uses_strict_comparison() {
    let fit = fit_exercise_decision(
        PolynomialBasisSpec::new(1, 0, 1, 1).expect("basis"),
        &[90.0, 80.0],
        2,
        &[10.0, 20.0],
        &[12.0, 12.0],
        0.0,
        CpqrConfig::new(0.0, 0.0).expect("config"),
        4,
    )
    .expect("fit");
    let ExerciseDecisionModel::Regression(model) = fit.decision() else {
        panic!("expected regression");
    };
    let continuation = model.predict(&[85.0]).expect("continuation");
    assert!(
        !fit.decision()
            .should_exercise(continuation, &[85.0])
            .expect("tie continues")
    );
    assert!(
        fit.decision()
            .should_exercise(continuation + 0.0001, &[85.0])
            .expect("strict exercise")
    );
    assert!(matches!(
        fit.decision().should_exercise(f64::NAN, &[85.0]),
        Err(LsmNumericalError::NonFiniteDecisionValue { .. })
    ));
}

#[test]
fn exercise_decision_validates_every_candidate_before_zero_itm_shortcut() {
    assert!(matches!(
        fit_exercise_decision(
            PolynomialBasisSpec::new(1, 1, 4, 4).expect("basis"),
            &[1.0, f64::NAN],
            2,
            &[0.0, 0.0],
            &[1.0, 1.0],
            0.0,
            CpqrConfig::new(0.0, 0.0).expect("config"),
            8,
        ),
        Err(LsmNumericalError::NonFiniteFeature { index: 1, .. })
    ));
}

#[test]
fn policy_training_runs_backward_and_preserves_ascending_decisions() {
    let dates = [
        "2027-01-02".parse().expect("date"),
        "2027-02-02".parse().expect("date"),
        "2027-03-02".parse().expect("date"),
    ];
    let outcome = train_exercise_policy(
        &dates,
        PolynomialBasisSpec::new(1, 0, 1, 1).expect("basis"),
        &[90.0, 110.0, 92.0, 112.0],
        2,
        &[10.0, 0.0, 8.0, 0.0, 0.0, 10.0],
        &[1.0, 0.9, 0.8],
        0.0,
        CpqrConfig::new(0.0, 0.0).expect("config"),
        16,
        training_metadata(2),
    )
    .expect("policy");
    assert_eq!(outcome.policy().exercise_dates(), dates);
    assert_eq!(outcome.policy().decisions().len(), 2);
    assert_eq!(outcome.policy().diagnostics()[0].itm_rows(), 1);
    assert_eq!(outcome.policy().diagnostics()[1].itm_rows(), 1);
    let ExerciseDecisionModel::Regression(first) = &outcome.policy().decisions()[0] else {
        panic!("first decision must be a regression");
    };
    assert!((first.coefficients()[0] - 7.2).abs() < 1.0e-13);
    assert_eq!(outcome.realized_cashflows(), [10.0, 10.0]);
    assert_eq!(outcome.stopping_indices(), [0, 2]);
}

#[test]
fn policy_training_does_not_reuse_a_later_model_at_zero_itm_date() {
    let dates = [
        "2027-01-02".parse().expect("date"),
        "2027-02-02".parse().expect("date"),
        "2027-03-02".parse().expect("date"),
    ];
    let outcome = train_exercise_policy(
        &dates,
        PolynomialBasisSpec::new(1, 0, 1, 1).expect("basis"),
        &[90.0, 95.0],
        1,
        &[9.0, 0.0, 10.0],
        &[1.0, 0.9, 0.8],
        0.0,
        CpqrConfig::new(0.0, 0.0).expect("config"),
        8,
        training_metadata(1),
    )
    .expect("policy");
    assert!(matches!(
        outcome.policy().decisions()[0],
        ExerciseDecisionModel::Regression(_)
    ));
    assert_eq!(
        outcome.policy().decisions()[1],
        ExerciseDecisionModel::ContinueAll {
            reason: ContinueAllReason::ZeroItmTrainingPaths,
        }
    );
    assert_eq!(outcome.realized_cashflows(), [9.0]);
    assert_eq!(outcome.stopping_indices(), [0]);
}

#[test]
fn policy_training_supports_terminal_only_schedule_and_validates_shapes() {
    let expiry = ["2027-03-02".parse().expect("date")];
    let outcome = train_exercise_policy(
        &expiry,
        PolynomialBasisSpec::new(1, 2, 4, 4).expect("basis"),
        &[],
        2,
        &[0.0, 10.0],
        &[0.8],
        0.0,
        CpqrConfig::new(0.0, 0.0).expect("config"),
        8,
        training_metadata(2),
    )
    .expect("terminal policy");
    assert!(outcome.policy().decisions().is_empty());
    assert_eq!(outcome.realized_cashflows(), [0.0, 10.0]);
    assert_eq!(outcome.stopping_indices(), [0, 0]);
    assert_eq!(
        outcome.policy().fingerprint().to_string(),
        "blake3-256:c26c8476b3d6f3f9ff053a8175f721e0b31f78f3113884348b8aa561d405a30e"
    );
    let changed_seed = ExercisePolicyTrainingMetadata::new([0x11; 32], [0x22; 32], 8, 2, 2)
        .expect("changed metadata");
    let changed = train_exercise_policy(
        &expiry,
        PolynomialBasisSpec::new(1, 2, 4, 4).expect("basis"),
        &[],
        2,
        &[0.0, 10.0],
        &[0.8],
        0.0,
        CpqrConfig::new(0.0, 0.0).expect("config"),
        8,
        changed_seed,
    )
    .expect("changed policy");
    assert_ne!(
        outcome.policy().fingerprint(),
        changed.policy().fingerprint()
    );

    assert!(matches!(
        train_exercise_policy(
            &expiry,
            PolynomialBasisSpec::new(1, 1, 2, 2).expect("basis"),
            &[],
            2,
            &[0.0],
            &[0.8],
            0.0,
            CpqrConfig::new(0.0, 0.0).expect("config"),
            8,
            training_metadata(2),
        ),
        Err(LsmNumericalError::ImmediateValueMatrixLengthMismatch { .. })
    ));
}

#[test]
fn frozen_policy_values_independent_paths_and_records_stopping_indices() {
    let dates = [
        "2027-01-02".parse().expect("date"),
        "2027-02-02".parse().expect("date"),
        "2027-03-02".parse().expect("date"),
    ];
    let training = train_exercise_policy(
        &dates,
        PolynomialBasisSpec::new(1, 0, 1, 1).expect("basis"),
        &[90.0, 110.0, 92.0, 112.0],
        2,
        &[10.0, 0.0, 8.0, 0.0, 0.0, 10.0],
        &[1.0, 0.9, 0.8],
        0.0,
        CpqrConfig::new(0.0, 0.0).expect("config"),
        16,
        training_metadata(2),
    )
    .expect("policy");
    let valuation = value_exercise_policy(
        training.policy(),
        &[91.0, 101.0, 111.0, 93.0, 103.0, 113.0],
        3,
        &[8.0, 0.0, 0.0, 9.0, 5.0, 0.0, 20.0, 20.0, 10.0],
        &[1.0, 0.9, 0.8],
    )
    .expect("valuation");
    assert_eq!(
        valuation.policy_fingerprint(),
        training.policy().fingerprint()
    );
    assert_eq!(valuation.realized_cashflows(), [8.0, 5.0, 10.0]);
    assert_eq!(valuation.discounted_cashflows(), [8.0, 4.5, 8.0]);
    assert_eq!(valuation.stopping_indices(), [0, 1, 2]);
    assert_eq!(valuation.exercise_counts(), [1, 1, 1]);
}

#[test]
fn frozen_policy_honors_continue_all_on_out_of_sample_itm_path() {
    let dates = [
        "2027-01-02".parse().expect("date"),
        "2027-02-02".parse().expect("date"),
        "2027-03-02".parse().expect("date"),
    ];
    let training = train_exercise_policy(
        &dates,
        PolynomialBasisSpec::new(1, 0, 1, 1).expect("basis"),
        &[90.0, 95.0],
        1,
        &[9.0, 0.0, 10.0],
        &[1.0, 0.9, 0.8],
        0.0,
        CpqrConfig::new(0.0, 0.0).expect("config"),
        8,
        training_metadata(1),
    )
    .expect("policy");
    let valuation = value_exercise_policy(
        training.policy(),
        &[90.0, 95.0],
        1,
        &[0.0, 100.0, 10.0],
        &[1.0, 0.9, 0.8],
    )
    .expect("valuation");
    assert_eq!(valuation.realized_cashflows(), [10.0]);
    assert_eq!(valuation.stopping_indices(), [2]);
    assert_eq!(valuation.exercise_counts(), [0, 0, 1]);
}
