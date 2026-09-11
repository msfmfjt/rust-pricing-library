use std::error::Error;
use std::fmt;

use pricing_core::Date;
use pricing_numerics::NeumaierSum;

use crate::EngineConfig;

pub const LSM_BASIS_ABI: &str = "polynomial-total-degree-v1";
pub const LSM_REGRESSION_ABI: &str = "cpqr-householder-v1";
pub const LSM_POLICY_ABI: &str = "early_exercise_v1";

#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum LsmNumericalError {
    ZeroResourceLimit {
        resource: &'static str,
    },
    BasisCountOverflow,
    BasisLimitExceeded {
        resource: &'static str,
        requested: usize,
        maximum: usize,
    },
    AllocationFailed {
        resource: &'static str,
        requested: usize,
    },
    EmptyFeatureSample,
    NonFiniteFeature {
        index: usize,
        bits: u64,
    },
    FeatureCountMismatch {
        expected: usize,
        actual: usize,
    },
    FeatureMatrixLengthMismatch {
        expected: usize,
        actual: usize,
    },
    ImmediateValueLengthMismatch {
        expected: usize,
        actual: usize,
    },
    ContinuationTargetLengthMismatch {
        expected: usize,
        actual: usize,
    },
    InactiveFeature,
    NonFiniteDecisionValue {
        name: &'static str,
        bits: u64,
    },
    InvalidTolerance {
        name: &'static str,
        bits: u64,
    },
    EmptyStateVariables,
    DuplicateStateVariable,
    BasisStateVariableCountMismatch {
        basis: u32,
        state_variables: usize,
    },
    ZeroMatrixRows,
    ZeroMatrixColumns,
    MatrixShapeOverflow,
    MatrixElementLimitExceeded {
        requested: usize,
        maximum: usize,
    },
    MatrixLengthMismatch {
        expected: usize,
        actual: usize,
    },
    TargetLengthMismatch {
        expected: usize,
        actual: usize,
    },
    NonFiniteMatrixValue {
        row: usize,
        column: usize,
        bits: u64,
    },
    NonFiniteTargetValue {
        row: usize,
        bits: u64,
    },
    NonFiniteImmediateValue {
        row: usize,
        bits: u64,
    },
    NonFiniteContinuationTarget {
        row: usize,
        bits: u64,
    },
    EmptyExerciseSchedule,
    InvalidExerciseDateOrder {
        index: usize,
    },
    ImmediateValueMatrixLengthMismatch {
        expected: usize,
        actual: usize,
    },
    TrainingFeatureMatrixLengthMismatch {
        expected: usize,
        actual: usize,
    },
    ValuationFeatureMatrixLengthMismatch {
        expected: usize,
        actual: usize,
    },
    DiscountFactorLengthMismatch {
        expected: usize,
        actual: usize,
    },
    InvalidDiscountFactor {
        date_index: usize,
        bits: u64,
    },
    NegativeImmediateValue {
        date_index: usize,
        path: usize,
        bits: u64,
    },
    ZeroTrainingCount {
        field: &'static str,
    },
    TrainingTrajectoryCountMismatch {
        metadata: u64,
        actual: usize,
    },
    InvalidTrainingRandomDomain {
        domain: crate::RandomDomain,
    },
    FingerprintCountOverflow {
        field: &'static str,
    },
    NonFiniteIntermediate {
        stage: &'static str,
    },
}

impl fmt::Display for LsmNumericalError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroResourceLimit { resource } => {
                write!(formatter, "LSM resource limit {resource} must be positive")
            }
            Self::BasisCountOverflow => write!(formatter, "polynomial basis count overflowed"),
            Self::BasisLimitExceeded {
                resource,
                requested,
                maximum,
            } => write!(
                formatter,
                "polynomial basis needs {requested} {resource}, exceeding limit {maximum}"
            ),
            Self::AllocationFailed {
                resource,
                requested,
            } => write!(
                formatter,
                "failed to allocate {requested} elements for LSM {resource}"
            ),
            Self::EmptyFeatureSample => {
                write!(
                    formatter,
                    "feature scaling requires at least one training row"
                )
            }
            Self::NonFiniteFeature { index, bits } => write!(
                formatter,
                "feature value {index} is non-finite: 0x{bits:016x}"
            ),
            Self::FeatureCountMismatch { expected, actual } => write!(
                formatter,
                "polynomial basis needs {expected} features; received {actual}"
            ),
            Self::FeatureMatrixLengthMismatch { expected, actual } => write!(
                formatter,
                "LSM feature matrix needs {expected} values; received {actual}"
            ),
            Self::ImmediateValueLengthMismatch { expected, actual } => write!(
                formatter,
                "LSM exercise date needs {expected} immediate values; received {actual}"
            ),
            Self::ContinuationTargetLengthMismatch { expected, actual } => write!(
                formatter,
                "LSM exercise date needs {expected} continuation targets; received {actual}"
            ),
            Self::InactiveFeature => {
                write!(formatter, "an inactive feature has no standardized value")
            }
            Self::NonFiniteDecisionValue { name, bits } => write!(
                formatter,
                "LSM decision value {name} is non-finite: 0x{bits:016x}"
            ),
            Self::InvalidTolerance { name, bits } => write!(
                formatter,
                "LSM tolerance {name} must be finite and non-negative: 0x{bits:016x}"
            ),
            Self::EmptyStateVariables => {
                write!(formatter, "LSM requires at least one state variable")
            }
            Self::DuplicateStateVariable => {
                write!(formatter, "LSM state variables must be unique")
            }
            Self::BasisStateVariableCountMismatch {
                basis,
                state_variables,
            } => write!(
                formatter,
                "LSM basis declares {basis} features for {state_variables} state variables"
            ),
            Self::ZeroMatrixRows => write!(formatter, "LSM regression matrix has zero rows"),
            Self::ZeroMatrixColumns => {
                write!(formatter, "LSM regression matrix has zero columns")
            }
            Self::MatrixShapeOverflow => write!(formatter, "LSM regression shape overflowed"),
            Self::MatrixElementLimitExceeded { requested, maximum } => write!(
                formatter,
                "LSM regression matrix needs {requested} elements, exceeding limit {maximum}"
            ),
            Self::MatrixLengthMismatch { expected, actual } => write!(
                formatter,
                "LSM regression matrix needs {expected} values; received {actual}"
            ),
            Self::TargetLengthMismatch { expected, actual } => write!(
                formatter,
                "LSM regression target needs {expected} values; received {actual}"
            ),
            Self::NonFiniteMatrixValue { row, column, bits } => write!(
                formatter,
                "LSM matrix value ({row}, {column}) is non-finite: 0x{bits:016x}"
            ),
            Self::NonFiniteTargetValue { row, bits } => write!(
                formatter,
                "LSM target value {row} is non-finite: 0x{bits:016x}"
            ),
            Self::NonFiniteImmediateValue { row, bits } => write!(
                formatter,
                "LSM immediate value {row} is non-finite: 0x{bits:016x}"
            ),
            Self::NonFiniteContinuationTarget { row, bits } => write!(
                formatter,
                "LSM continuation target {row} is non-finite: 0x{bits:016x}"
            ),
            Self::EmptyExerciseSchedule => write!(formatter, "LSM exercise schedule is empty"),
            Self::InvalidExerciseDateOrder { index } => write!(
                formatter,
                "LSM exercise date {index} is not later than its predecessor"
            ),
            Self::ImmediateValueMatrixLengthMismatch { expected, actual } => write!(
                formatter,
                "LSM immediate-value matrix needs {expected} values; received {actual}"
            ),
            Self::TrainingFeatureMatrixLengthMismatch { expected, actual } => write!(
                formatter,
                "LSM training-feature matrix needs {expected} values; received {actual}"
            ),
            Self::ValuationFeatureMatrixLengthMismatch { expected, actual } => write!(
                formatter,
                "LSM valuation-feature matrix needs {expected} values; received {actual}"
            ),
            Self::DiscountFactorLengthMismatch { expected, actual } => write!(
                formatter,
                "LSM exercise schedule needs {expected} discount factors; received {actual}"
            ),
            Self::InvalidDiscountFactor { date_index, bits } => write!(
                formatter,
                "LSM discount factor {date_index} must be finite and positive: 0x{bits:016x}"
            ),
            Self::NegativeImmediateValue {
                date_index,
                path,
                bits,
            } => write!(
                formatter,
                "LSM immediate value ({date_index}, {path}) is negative: 0x{bits:016x}"
            ),
            Self::ZeroTrainingCount { field } => {
                write!(formatter, "LSM training {field} must be positive")
            }
            Self::TrainingTrajectoryCountMismatch { metadata, actual } => write!(
                formatter,
                "LSM metadata declares {metadata} trajectories; received {actual} training paths"
            ),
            Self::InvalidTrainingRandomDomain { domain } => write!(
                formatter,
                "random domain {domain:?} is not valid for LSM policy training"
            ),
            Self::FingerprintCountOverflow { field } => {
                write!(
                    formatter,
                    "LSM policy fingerprint {field} does not fit in u64"
                )
            }
            Self::NonFiniteIntermediate { stage } => {
                write!(formatter, "LSM regression produced a non-finite {stage}")
            }
        }
    }
}

impl Error for LsmNumericalError {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PolynomialBasisSpec {
    feature_count: u32,
    max_degree: u32,
    exponents: Box<[Box<[u32]>]>,
}

impl PolynomialBasisSpec {
    pub fn new(
        feature_count: u32,
        max_degree: u32,
        max_columns: usize,
        max_total_exponents: usize,
    ) -> Result<Self, LsmNumericalError> {
        if max_columns == 0 {
            return Err(LsmNumericalError::ZeroResourceLimit {
                resource: "basis_columns",
            });
        }
        if max_total_exponents == 0 {
            return Err(LsmNumericalError::ZeroResourceLimit {
                resource: "basis_exponents",
            });
        }
        let column_count = basis_column_count(feature_count, max_degree)?;
        if column_count > max_columns {
            return Err(LsmNumericalError::BasisLimitExceeded {
                resource: "columns",
                requested: column_count,
                maximum: max_columns,
            });
        }
        let total_exponents = column_count
            .checked_mul(feature_count as usize)
            .ok_or(LsmNumericalError::BasisCountOverflow)?;
        if total_exponents > max_total_exponents {
            return Err(LsmNumericalError::BasisLimitExceeded {
                resource: "exponents",
                requested: total_exponents,
                maximum: max_total_exponents,
            });
        }
        let mut exponents = Vec::new();
        exponents.try_reserve_exact(column_count).map_err(|_| {
            LsmNumericalError::AllocationFailed {
                resource: "basis columns",
                requested: column_count,
            }
        })?;
        if feature_count == 0 {
            exponents.push(Vec::new().into_boxed_slice());
        } else {
            let mut current = Vec::new();
            current
                .try_reserve_exact(feature_count as usize)
                .map_err(|_| LsmNumericalError::AllocationFailed {
                    resource: "basis enumeration row",
                    requested: feature_count as usize,
                })?;
            current.resize(feature_count as usize, 0_u32);
            for degree in 0..=max_degree {
                enumerate_degree(&mut exponents, &mut current, 0, degree)?;
            }
        }
        debug_assert_eq!(exponents.len(), column_count);
        Ok(Self {
            feature_count,
            max_degree,
            exponents: exponents.into_boxed_slice(),
        })
    }

    #[must_use]
    pub const fn feature_count(&self) -> u32 {
        self.feature_count
    }

    #[must_use]
    pub const fn max_degree(&self) -> u32 {
        self.max_degree
    }

    #[must_use]
    pub fn exponents(&self) -> &[Box<[u32]>] {
        &self.exponents
    }

    pub fn evaluate(&self, features: &[f64]) -> Result<Box<[f64]>, LsmNumericalError> {
        if features.len() != self.feature_count as usize {
            return Err(LsmNumericalError::FeatureCountMismatch {
                expected: self.feature_count as usize,
                actual: features.len(),
            });
        }
        for (index, &feature) in features.iter().enumerate() {
            if !feature.is_finite() {
                return Err(LsmNumericalError::NonFiniteFeature {
                    index,
                    bits: feature.to_bits(),
                });
            }
        }
        let mut values = Vec::new();
        values
            .try_reserve_exact(self.exponents.len())
            .map_err(|_| LsmNumericalError::AllocationFailed {
                resource: "basis values",
                requested: self.exponents.len(),
            })?;
        for exponents in &self.exponents {
            let mut value = 1.0;
            for (&feature, &exponent) in features.iter().zip(exponents.iter()) {
                for _ in 0..exponent {
                    value *= feature;
                    if !value.is_finite() {
                        return Err(LsmNumericalError::NonFiniteIntermediate {
                            stage: "basis value",
                        });
                    }
                }
            }
            values.push(value);
        }
        Ok(values.into_boxed_slice())
    }
}

fn basis_column_count(feature_count: u32, max_degree: u32) -> Result<usize, LsmNumericalError> {
    if feature_count == 0 {
        return Ok(1);
    }
    let n = u64::from(feature_count)
        .checked_add(u64::from(max_degree))
        .ok_or(LsmNumericalError::BasisCountOverflow)?;
    let k = u64::from(max_degree).min(u64::from(feature_count));
    let mut value = 1_u128;
    for index in 1..=k {
        value = value
            .checked_mul(u128::from(n - k + index))
            .ok_or(LsmNumericalError::BasisCountOverflow)?
            / u128::from(index);
    }
    usize::try_from(value).map_err(|_| LsmNumericalError::BasisCountOverflow)
}

fn enumerate_degree(
    result: &mut Vec<Box<[u32]>>,
    current: &mut [u32],
    feature: usize,
    remaining: u32,
) -> Result<(), LsmNumericalError> {
    if feature + 1 == current.len() {
        current[feature] = remaining;
        let mut copy = Vec::new();
        copy.try_reserve_exact(current.len())
            .map_err(|_| LsmNumericalError::AllocationFailed {
                resource: "basis exponent row",
                requested: current.len(),
            })?;
        copy.extend_from_slice(current);
        result.push(copy.into_boxed_slice());
        return Ok(());
    }
    for exponent in (0..=remaining).rev() {
        current[feature] = exponent;
        enumerate_degree(result, current, feature + 1, remaining - exponent)?;
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FeatureScaling {
    mean: f64,
    population_variance: f64,
    scale: f64,
    zero_scale_threshold: f64,
    inactive: bool,
}

impl FeatureScaling {
    pub fn fit(values: &[f64]) -> Result<Self, LsmNumericalError> {
        if values.is_empty() {
            return Err(LsmNumericalError::EmptyFeatureSample);
        }
        let mut sum = NeumaierSum::new();
        let mut feature_scale = 1.0_f64;
        for (index, &value) in values.iter().enumerate() {
            if !value.is_finite() {
                return Err(LsmNumericalError::NonFiniteFeature {
                    index,
                    bits: value.to_bits(),
                });
            }
            sum.add(value);
            feature_scale = feature_scale.max(value.abs());
        }
        let mean = sum.total() / values.len() as f64;
        let mut squared_deviations = NeumaierSum::new();
        for &value in values {
            let deviation = value - mean;
            squared_deviations.add(deviation * deviation);
        }
        let population_variance = squared_deviations.total() / values.len() as f64;
        if !mean.is_finite() || !population_variance.is_finite() || population_variance < 0.0 {
            return Err(LsmNumericalError::NonFiniteIntermediate {
                stage: "feature scaling",
            });
        }
        let scale = population_variance.sqrt();
        let zero_scale_threshold = (64.0 * f64::EPSILON) * feature_scale;
        Ok(Self {
            mean,
            population_variance,
            scale,
            zero_scale_threshold,
            inactive: scale <= zero_scale_threshold,
        })
    }

    #[must_use]
    pub const fn mean(self) -> f64 {
        self.mean
    }

    #[must_use]
    pub const fn population_variance(self) -> f64 {
        self.population_variance
    }

    #[must_use]
    pub const fn scale(self) -> f64 {
        self.scale
    }

    #[must_use]
    pub const fn zero_scale_threshold(self) -> f64 {
        self.zero_scale_threshold
    }

    #[must_use]
    pub const fn inactive(self) -> bool {
        self.inactive
    }

    pub fn standardize(self, value: f64) -> Result<f64, LsmNumericalError> {
        if !value.is_finite() {
            return Err(LsmNumericalError::NonFiniteFeature {
                index: 0,
                bits: value.to_bits(),
            });
        }
        if self.inactive {
            return Err(LsmNumericalError::InactiveFeature);
        }
        let standardized = (value - self.mean) / self.scale;
        if !standardized.is_finite() {
            return Err(LsmNumericalError::NonFiniteIntermediate {
                stage: "standardized feature",
            });
        }
        Ok(standardized)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PolynomialRegressionModel {
    basis: PolynomialBasisSpec,
    feature_scalings: Box<[FeatureScaling]>,
    active_basis_columns: Box<[usize]>,
    pre_excluded_basis_columns: Box<[usize]>,
    pivot_order: Box<[usize]>,
    diagonal_abs: Box<[f64]>,
    rank_threshold: f64,
    rank: usize,
    rank_excluded_basis_columns: Box<[usize]>,
    coefficients: Box<[f64]>,
    residual_sum_squares: f64,
}

impl PolynomialRegressionModel {
    #[must_use]
    pub const fn basis(&self) -> &PolynomialBasisSpec {
        &self.basis
    }

    #[must_use]
    pub fn feature_scalings(&self) -> &[FeatureScaling] {
        &self.feature_scalings
    }

    #[must_use]
    pub fn active_basis_columns(&self) -> &[usize] {
        &self.active_basis_columns
    }

    #[must_use]
    pub fn pre_excluded_basis_columns(&self) -> &[usize] {
        &self.pre_excluded_basis_columns
    }

    #[must_use]
    pub fn pivot_order(&self) -> &[usize] {
        &self.pivot_order
    }

    #[must_use]
    pub fn diagonal_abs(&self) -> &[f64] {
        &self.diagonal_abs
    }

    #[must_use]
    pub const fn rank_threshold(&self) -> f64 {
        self.rank_threshold
    }

    #[must_use]
    pub const fn rank(&self) -> usize {
        self.rank
    }

    #[must_use]
    pub fn rank_excluded_basis_columns(&self) -> &[usize] {
        &self.rank_excluded_basis_columns
    }

    #[must_use]
    pub fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }

    #[must_use]
    pub const fn residual_sum_squares(&self) -> f64 {
        self.residual_sum_squares
    }

    pub fn predict(&self, features: &[f64]) -> Result<f64, LsmNumericalError> {
        let standardized = standardize_features(features, &self.feature_scalings)?;
        let basis_values = self.basis.evaluate(&standardized)?;
        let mut prediction = NeumaierSum::new();
        for (&basis_value, &coefficient) in basis_values.iter().zip(&self.coefficients) {
            prediction.add(basis_value * coefficient);
        }
        let prediction = prediction.total();
        if !prediction.is_finite() {
            return Err(LsmNumericalError::NonFiniteIntermediate {
                stage: "regression prediction",
            });
        }
        Ok(prediction)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ContinueAllReason {
    ZeroItmTrainingPaths,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ExerciseDecisionModel {
    Regression(PolynomialRegressionModel),
    ContinueAll { reason: ContinueAllReason },
}

impl ExerciseDecisionModel {
    pub fn should_exercise(
        &self,
        immediate_value: f64,
        features: &[f64],
    ) -> Result<bool, LsmNumericalError> {
        validate_decision_value("immediate_value", immediate_value)?;
        match self {
            Self::Regression(model) => {
                let continuation_value = model.predict(features)?;
                should_exercise(immediate_value, continuation_value)
            }
            Self::ContinueAll { .. } => Ok(false),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum LsmWarning {
    ZeroItmTrainingPaths,
    InactiveFeature { feature: usize },
    RankExcludedBasisColumn { column: usize },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExerciseRegressionDiagnostics {
    candidate_rows: usize,
    itm_rows: usize,
    feature_count: usize,
    warnings: Box<[LsmWarning]>,
}

impl ExerciseRegressionDiagnostics {
    #[must_use]
    pub const fn candidate_rows(&self) -> usize {
        self.candidate_rows
    }

    #[must_use]
    pub const fn itm_rows(&self) -> usize {
        self.itm_rows
    }

    #[must_use]
    pub const fn feature_count(&self) -> usize {
        self.feature_count
    }

    #[must_use]
    pub fn warnings(&self) -> &[LsmWarning] {
        &self.warnings
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct DateLocalExerciseFit {
    decision: ExerciseDecisionModel,
    diagnostics: ExerciseRegressionDiagnostics,
}

impl DateLocalExerciseFit {
    #[must_use]
    pub const fn decision(&self) -> &ExerciseDecisionModel {
        &self.decision
    }

    #[must_use]
    pub const fn diagnostics(&self) -> &ExerciseRegressionDiagnostics {
        &self.diagnostics
    }

    #[must_use]
    pub fn into_decision(self) -> ExerciseDecisionModel {
        self.decision
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ExercisePolicyFingerprint([u8; 32]);

impl ExercisePolicyFingerprint {
    #[must_use]
    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u8)]
pub enum LsmStateVariable {
    Spot = 0,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct LsmConfigurationFingerprint([u8; 32]);

impl LsmConfigurationFingerprint {
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl fmt::Display for LsmConfigurationFingerprint {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "blake3-256:")?;
        for byte in self.0 {
            write!(formatter, "{byte:02x}")?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct LsmConfig {
    training_engine: EngineConfig,
    state_variables: Box<[LsmStateVariable]>,
    basis: PolynomialBasisSpec,
    itm_abs_tolerance: f64,
    cpqr_config: CpqrConfig,
    max_matrix_elements: usize,
    fingerprint: LsmConfigurationFingerprint,
}

impl LsmConfig {
    pub fn new(
        training_engine: EngineConfig,
        state_variables: Vec<LsmStateVariable>,
        basis: PolynomialBasisSpec,
        itm_abs_tolerance: f64,
        cpqr_config: CpqrConfig,
        max_matrix_elements: usize,
    ) -> Result<Self, LsmNumericalError> {
        validate_tolerance("itm_abs_tolerance", itm_abs_tolerance)?;
        if state_variables.is_empty() {
            return Err(LsmNumericalError::EmptyStateVariables);
        }
        for (index, state_variable) in state_variables.iter().enumerate() {
            if state_variables[..index].contains(state_variable) {
                return Err(LsmNumericalError::DuplicateStateVariable);
            }
        }
        if basis.feature_count() as usize != state_variables.len() {
            return Err(LsmNumericalError::BasisStateVariableCountMismatch {
                basis: basis.feature_count(),
                state_variables: state_variables.len(),
            });
        }
        if max_matrix_elements == 0 {
            return Err(LsmNumericalError::ZeroResourceLimit {
                resource: "regression_matrix_elements",
            });
        }
        let mut config = Self {
            training_engine,
            state_variables: state_variables.into_boxed_slice(),
            basis,
            itm_abs_tolerance,
            cpqr_config,
            max_matrix_elements,
            fingerprint: LsmConfigurationFingerprint([0; 32]),
        };
        config.fingerprint = fingerprint_lsm_configuration(&config)?;
        Ok(config)
    }

    #[must_use]
    pub const fn training_engine(&self) -> EngineConfig {
        self.training_engine
    }

    #[must_use]
    pub fn state_variables(&self) -> &[LsmStateVariable] {
        &self.state_variables
    }

    #[must_use]
    pub const fn basis(&self) -> &PolynomialBasisSpec {
        &self.basis
    }

    #[must_use]
    pub const fn itm_abs_tolerance(&self) -> f64 {
        self.itm_abs_tolerance
    }

    #[must_use]
    pub const fn cpqr_config(&self) -> CpqrConfig {
        self.cpqr_config
    }

    #[must_use]
    pub const fn max_matrix_elements(&self) -> usize {
        self.max_matrix_elements
    }

    #[must_use]
    pub const fn fingerprint(&self) -> LsmConfigurationFingerprint {
        self.fingerprint
    }

    #[must_use]
    pub const fn training_seed(&self) -> u64 {
        match self.training_engine {
            EngineConfig::PseudoMonteCarlo(config) => config.master_seed(),
            EngineConfig::RandomizedQuasiMonteCarlo(config) => config.master_scramble_seed(),
        }
    }

    #[must_use]
    pub const fn training_effective_sampling_units(&self) -> u64 {
        match self.training_engine {
            EngineConfig::PseudoMonteCarlo(config) => config.independent_sampling_units().get(),
            EngineConfig::RandomizedQuasiMonteCarlo(config) => config.scramble_count().get() as u64,
        }
    }

    #[must_use]
    pub const fn training_trajectory_count(&self) -> u128 {
        match self.training_engine {
            EngineConfig::PseudoMonteCarlo(config) => config.evaluated_paths(),
            EngineConfig::RandomizedQuasiMonteCarlo(config) => {
                let antithetic_multiplier = if config.variance_reduction().antithetic() {
                    2
                } else {
                    1
                };
                config.points_per_scramble().get() as u128
                    * config.scramble_count().get() as u128
                    * antithetic_multiplier
            }
        }
    }

    #[must_use]
    pub const fn training_random_domain(&self) -> crate::RandomDomain {
        match self.training_engine {
            EngineConfig::PseudoMonteCarlo(_) => crate::RandomDomain::LsmTrain,
            EngineConfig::RandomizedQuasiMonteCarlo(_) => crate::RandomDomain::RqmcScramble,
        }
    }

    pub fn training_metadata(
        &self,
        product_fingerprint: [u8; 32],
    ) -> Result<ExercisePolicyTrainingMetadata, LsmNumericalError> {
        let trajectory_count = u64::try_from(self.training_trajectory_count()).map_err(|_| {
            LsmNumericalError::FingerprintCountOverflow {
                field: "training_trajectory_count",
            }
        })?;
        ExercisePolicyTrainingMetadata::new_with_random_domain(
            product_fingerprint,
            *self.fingerprint.as_bytes(),
            self.training_seed(),
            self.training_effective_sampling_units(),
            trajectory_count,
            self.training_random_domain(),
        )
    }
}

impl fmt::Display for ExercisePolicyFingerprint {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "blake3-256:")?;
        for byte in self.0 {
            write!(formatter, "{byte:02x}")?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ExercisePolicyTrainingMetadata {
    product_fingerprint: [u8; 32],
    training_configuration_fingerprint: [u8; 32],
    seed: u64,
    sampling_units: u64,
    trajectory_count: u64,
    random_domain: crate::RandomDomain,
}

impl ExercisePolicyTrainingMetadata {
    pub fn new(
        product_fingerprint: [u8; 32],
        training_configuration_fingerprint: [u8; 32],
        seed: u64,
        sampling_units: u64,
        trajectory_count: u64,
    ) -> Result<Self, LsmNumericalError> {
        Self::new_with_random_domain(
            product_fingerprint,
            training_configuration_fingerprint,
            seed,
            sampling_units,
            trajectory_count,
            crate::RandomDomain::LsmTrain,
        )
    }

    pub fn new_with_random_domain(
        product_fingerprint: [u8; 32],
        training_configuration_fingerprint: [u8; 32],
        seed: u64,
        sampling_units: u64,
        trajectory_count: u64,
        random_domain: crate::RandomDomain,
    ) -> Result<Self, LsmNumericalError> {
        if sampling_units == 0 {
            return Err(LsmNumericalError::ZeroTrainingCount {
                field: "sampling_units",
            });
        }
        if trajectory_count == 0 {
            return Err(LsmNumericalError::ZeroTrainingCount {
                field: "trajectory_count",
            });
        }
        if !matches!(
            random_domain,
            crate::RandomDomain::LsmTrain | crate::RandomDomain::RqmcScramble
        ) {
            return Err(LsmNumericalError::InvalidTrainingRandomDomain {
                domain: random_domain,
            });
        }
        Ok(Self {
            product_fingerprint,
            training_configuration_fingerprint,
            seed,
            sampling_units,
            trajectory_count,
            random_domain,
        })
    }

    #[must_use]
    pub const fn product_fingerprint(self) -> [u8; 32] {
        self.product_fingerprint
    }

    #[must_use]
    pub const fn training_configuration_fingerprint(self) -> [u8; 32] {
        self.training_configuration_fingerprint
    }

    #[must_use]
    pub const fn seed(self) -> u64 {
        self.seed
    }

    #[must_use]
    pub const fn sampling_units(self) -> u64 {
        self.sampling_units
    }

    #[must_use]
    pub const fn trajectory_count(self) -> u64 {
        self.trajectory_count
    }

    #[must_use]
    pub const fn random_domain(self) -> crate::RandomDomain {
        self.random_domain
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExercisePolicy {
    exercise_dates: Box<[Date]>,
    basis: PolynomialBasisSpec,
    itm_abs_tolerance: f64,
    cpqr_config: CpqrConfig,
    max_matrix_elements: usize,
    training_metadata: ExercisePolicyTrainingMetadata,
    decisions: Box<[ExerciseDecisionModel]>,
    diagnostics: Box<[ExerciseRegressionDiagnostics]>,
    fingerprint: ExercisePolicyFingerprint,
}

impl ExercisePolicy {
    #[must_use]
    pub fn exercise_dates(&self) -> &[Date] {
        &self.exercise_dates
    }

    #[must_use]
    pub const fn basis(&self) -> &PolynomialBasisSpec {
        &self.basis
    }

    #[must_use]
    pub const fn itm_abs_tolerance(&self) -> f64 {
        self.itm_abs_tolerance
    }

    #[must_use]
    pub const fn cpqr_config(&self) -> CpqrConfig {
        self.cpqr_config
    }

    #[must_use]
    pub const fn max_matrix_elements(&self) -> usize {
        self.max_matrix_elements
    }

    #[must_use]
    pub const fn training_metadata(&self) -> ExercisePolicyTrainingMetadata {
        self.training_metadata
    }

    #[must_use]
    pub fn decisions(&self) -> &[ExerciseDecisionModel] {
        &self.decisions
    }

    #[must_use]
    pub fn diagnostics(&self) -> &[ExerciseRegressionDiagnostics] {
        &self.diagnostics
    }

    #[must_use]
    pub const fn fingerprint(&self) -> ExercisePolicyFingerprint {
        self.fingerprint
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExercisePolicyTrainingOutcome {
    policy: ExercisePolicy,
    realized_cashflows: Box<[f64]>,
    stopping_indices: Box<[usize]>,
}

impl ExercisePolicyTrainingOutcome {
    #[must_use]
    pub const fn policy(&self) -> &ExercisePolicy {
        &self.policy
    }

    #[must_use]
    pub fn realized_cashflows(&self) -> &[f64] {
        &self.realized_cashflows
    }

    #[must_use]
    pub fn stopping_indices(&self) -> &[usize] {
        &self.stopping_indices
    }

    #[must_use]
    pub fn into_policy(self) -> ExercisePolicy {
        self.policy
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExercisePolicyValuationOutcome {
    policy_fingerprint: ExercisePolicyFingerprint,
    realized_cashflows: Box<[f64]>,
    discounted_cashflows: Box<[f64]>,
    stopping_indices: Box<[usize]>,
    exercise_counts: Box<[usize]>,
}

impl ExercisePolicyValuationOutcome {
    #[must_use]
    pub const fn policy_fingerprint(&self) -> ExercisePolicyFingerprint {
        self.policy_fingerprint
    }

    #[must_use]
    pub fn realized_cashflows(&self) -> &[f64] {
        &self.realized_cashflows
    }

    #[must_use]
    pub fn discounted_cashflows(&self) -> &[f64] {
        &self.discounted_cashflows
    }

    #[must_use]
    pub fn stopping_indices(&self) -> &[usize] {
        &self.stopping_indices
    }

    #[must_use]
    pub fn exercise_counts(&self) -> &[usize] {
        &self.exercise_counts
    }
}

/// Applies a frozen policy to independent, date-major valuation paths.
pub fn value_exercise_policy(
    policy: &ExercisePolicy,
    valuation_features: &[f64],
    valuation_paths: usize,
    immediate_values: &[f64],
    discount_factors: &[f64],
) -> Result<ExercisePolicyValuationOutcome, LsmNumericalError> {
    if valuation_paths == 0 {
        return Err(LsmNumericalError::ZeroMatrixRows);
    }
    let date_count = policy.exercise_dates.len();
    if discount_factors.len() != date_count {
        return Err(LsmNumericalError::DiscountFactorLengthMismatch {
            expected: date_count,
            actual: discount_factors.len(),
        });
    }
    for (date_index, &discount_factor) in discount_factors.iter().enumerate() {
        if !discount_factor.is_finite() || discount_factor <= 0.0 {
            return Err(LsmNumericalError::InvalidDiscountFactor {
                date_index,
                bits: discount_factor.to_bits(),
            });
        }
    }
    let immediate_count = date_count
        .checked_mul(valuation_paths)
        .ok_or(LsmNumericalError::MatrixShapeOverflow)?;
    if immediate_values.len() != immediate_count {
        return Err(LsmNumericalError::ImmediateValueMatrixLengthMismatch {
            expected: immediate_count,
            actual: immediate_values.len(),
        });
    }
    for date_index in 0..date_count {
        for path in 0..valuation_paths {
            let value = immediate_values[date_index * valuation_paths + path];
            if !value.is_finite() {
                return Err(LsmNumericalError::NonFiniteImmediateValue {
                    row: date_index * valuation_paths + path,
                    bits: value.to_bits(),
                });
            }
            if value < 0.0 {
                return Err(LsmNumericalError::NegativeImmediateValue {
                    date_index,
                    path,
                    bits: value.to_bits(),
                });
            }
        }
    }
    let feature_count = policy.basis.feature_count as usize;
    let non_terminal_dates = date_count - 1;
    let features_per_date = valuation_paths
        .checked_mul(feature_count)
        .ok_or(LsmNumericalError::MatrixShapeOverflow)?;
    let expected_features = non_terminal_dates
        .checked_mul(features_per_date)
        .ok_or(LsmNumericalError::MatrixShapeOverflow)?;
    if valuation_features.len() != expected_features {
        return Err(LsmNumericalError::ValuationFeatureMatrixLengthMismatch {
            expected: expected_features,
            actual: valuation_features.len(),
        });
    }
    for (index, &feature) in valuation_features.iter().enumerate() {
        if !feature.is_finite() {
            return Err(LsmNumericalError::NonFiniteFeature {
                index,
                bits: feature.to_bits(),
            });
        }
    }

    let expiry_index = date_count - 1;
    let expiry_start = expiry_index * valuation_paths;
    let mut realized_cashflows = copied_values(
        &immediate_values[expiry_start..expiry_start + valuation_paths],
        "valuation realized cash flows",
    )?;
    let mut stopping_indices = Vec::new();
    stopping_indices
        .try_reserve_exact(valuation_paths)
        .map_err(|_| LsmNumericalError::AllocationFailed {
            resource: "valuation stopping indices",
            requested: valuation_paths,
        })?;
    stopping_indices.resize(valuation_paths, expiry_index);

    for date_index in 0..non_terminal_dates {
        let immediate_start = date_index * valuation_paths;
        let date_feature_start = date_index * features_per_date;
        for path in 0..valuation_paths {
            if stopping_indices[path] != expiry_index {
                continue;
            }
            let feature_start = date_feature_start + path * feature_count;
            let immediate_value = immediate_values[immediate_start + path];
            if policy.decisions[date_index].should_exercise(
                immediate_value,
                &valuation_features[feature_start..feature_start + feature_count],
            )? {
                realized_cashflows[path] = immediate_value;
                stopping_indices[path] = date_index;
            }
        }
    }

    let mut discounted_cashflows = zeroed_values(valuation_paths, "discounted cash flows")?;
    let mut exercise_counts = Vec::new();
    exercise_counts.try_reserve_exact(date_count).map_err(|_| {
        LsmNumericalError::AllocationFailed {
            resource: "exercise counts",
            requested: date_count,
        }
    })?;
    exercise_counts.resize(date_count, 0_usize);
    for path in 0..valuation_paths {
        let stopping_index = stopping_indices[path];
        exercise_counts[stopping_index] += 1;
        discounted_cashflows[path] = realized_cashflows[path] * discount_factors[stopping_index];
        if !discounted_cashflows[path].is_finite() {
            return Err(LsmNumericalError::NonFiniteIntermediate {
                stage: "discounted valuation cash flow",
            });
        }
    }
    Ok(ExercisePolicyValuationOutcome {
        policy_fingerprint: policy.fingerprint,
        realized_cashflows: realized_cashflows.into_boxed_slice(),
        discounted_cashflows: discounted_cashflows.into_boxed_slice(),
        stopping_indices: stopping_indices.into_boxed_slice(),
        exercise_counts: exercise_counts.into_boxed_slice(),
    })
}

/// Trains a policy from date-major immediate values and non-terminal features.
#[allow(clippy::too_many_arguments)]
pub fn train_exercise_policy(
    exercise_dates: &[Date],
    basis: PolynomialBasisSpec,
    training_features: &[f64],
    training_paths: usize,
    immediate_values: &[f64],
    discount_factors: &[f64],
    itm_abs_tolerance: f64,
    config: CpqrConfig,
    max_matrix_elements: usize,
    training_metadata: ExercisePolicyTrainingMetadata,
) -> Result<ExercisePolicyTrainingOutcome, LsmNumericalError> {
    validate_tolerance("itm_abs_tolerance", itm_abs_tolerance)?;
    if exercise_dates.is_empty() {
        return Err(LsmNumericalError::EmptyExerciseSchedule);
    }
    for (index, pair) in exercise_dates.windows(2).enumerate() {
        if pair[0] >= pair[1] {
            return Err(LsmNumericalError::InvalidExerciseDateOrder { index: index + 1 });
        }
    }
    let mut policy_dates = Vec::new();
    policy_dates
        .try_reserve_exact(exercise_dates.len())
        .map_err(|_| LsmNumericalError::AllocationFailed {
            resource: "exercise policy dates",
            requested: exercise_dates.len(),
        })?;
    policy_dates.extend_from_slice(exercise_dates);
    if training_paths == 0 {
        return Err(LsmNumericalError::ZeroMatrixRows);
    }
    if max_matrix_elements == 0 {
        return Err(LsmNumericalError::ZeroResourceLimit {
            resource: "regression_matrix_elements",
        });
    }
    if training_paths > max_matrix_elements {
        return Err(LsmNumericalError::MatrixElementLimitExceeded {
            requested: training_paths,
            maximum: max_matrix_elements,
        });
    }
    let actual_trajectory_count =
        u64::try_from(training_paths).map_err(|_| LsmNumericalError::FingerprintCountOverflow {
            field: "training_paths",
        })?;
    if training_metadata.trajectory_count != actual_trajectory_count {
        return Err(LsmNumericalError::TrainingTrajectoryCountMismatch {
            metadata: training_metadata.trajectory_count,
            actual: training_paths,
        });
    }
    if discount_factors.len() != exercise_dates.len() {
        return Err(LsmNumericalError::DiscountFactorLengthMismatch {
            expected: exercise_dates.len(),
            actual: discount_factors.len(),
        });
    }
    for (date_index, &discount_factor) in discount_factors.iter().enumerate() {
        if !discount_factor.is_finite() || discount_factor <= 0.0 {
            return Err(LsmNumericalError::InvalidDiscountFactor {
                date_index,
                bits: discount_factor.to_bits(),
            });
        }
    }

    let immediate_count = exercise_dates
        .len()
        .checked_mul(training_paths)
        .ok_or(LsmNumericalError::MatrixShapeOverflow)?;
    if immediate_values.len() != immediate_count {
        return Err(LsmNumericalError::ImmediateValueMatrixLengthMismatch {
            expected: immediate_count,
            actual: immediate_values.len(),
        });
    }
    for date_index in 0..exercise_dates.len() {
        for path in 0..training_paths {
            let value = immediate_values[date_index * training_paths + path];
            if !value.is_finite() {
                return Err(LsmNumericalError::NonFiniteImmediateValue {
                    row: date_index * training_paths + path,
                    bits: value.to_bits(),
                });
            }
            if value < 0.0 {
                return Err(LsmNumericalError::NegativeImmediateValue {
                    date_index,
                    path,
                    bits: value.to_bits(),
                });
            }
        }
    }

    let feature_count = basis.feature_count as usize;
    let non_terminal_dates = exercise_dates.len() - 1;
    let feature_count_per_date = training_paths
        .checked_mul(feature_count)
        .ok_or(LsmNumericalError::MatrixShapeOverflow)?;
    let expected_features = non_terminal_dates
        .checked_mul(feature_count_per_date)
        .ok_or(LsmNumericalError::MatrixShapeOverflow)?;
    if training_features.len() != expected_features {
        return Err(LsmNumericalError::TrainingFeatureMatrixLengthMismatch {
            expected: expected_features,
            actual: training_features.len(),
        });
    }

    let expiry_index = exercise_dates.len() - 1;
    let expiry_start = expiry_index * training_paths;
    let mut realized_cashflows = copied_values(
        &immediate_values[expiry_start..expiry_start + training_paths],
        "training realized cash flows",
    )?;
    let mut stopping_indices = Vec::new();
    stopping_indices
        .try_reserve_exact(training_paths)
        .map_err(|_| LsmNumericalError::AllocationFailed {
            resource: "training stopping indices",
            requested: training_paths,
        })?;
    stopping_indices.resize(training_paths, expiry_index);
    let mut reverse_fits = Vec::new();
    reverse_fits
        .try_reserve_exact(non_terminal_dates)
        .map_err(|_| LsmNumericalError::AllocationFailed {
            resource: "exercise date fits",
            requested: non_terminal_dates,
        })?;
    let mut continuation_targets = zeroed_values(training_paths, "continuation targets")?;

    for date_index in (0..non_terminal_dates).rev() {
        for path in 0..training_paths {
            let stopping_index = stopping_indices[path];
            continuation_targets[path] = realized_cashflows[path]
                * discount_factors[stopping_index]
                / discount_factors[date_index];
            if !continuation_targets[path].is_finite() {
                return Err(LsmNumericalError::NonFiniteContinuationTarget {
                    row: path,
                    bits: continuation_targets[path].to_bits(),
                });
            }
        }
        let immediate_start = date_index * training_paths;
        let feature_start = date_index * feature_count_per_date;
        let fit = fit_exercise_decision(
            basis.clone(),
            &training_features[feature_start..feature_start + feature_count_per_date],
            training_paths,
            &immediate_values[immediate_start..immediate_start + training_paths],
            &continuation_targets,
            itm_abs_tolerance,
            config,
            max_matrix_elements,
        )?;
        for path in 0..training_paths {
            let immediate_value = immediate_values[immediate_start + path];
            let feature_start = feature_start + path * feature_count;
            if fit.decision.should_exercise(
                immediate_value,
                &training_features[feature_start..feature_start + feature_count],
            )? {
                realized_cashflows[path] = immediate_value;
                stopping_indices[path] = date_index;
            }
        }
        reverse_fits.push(fit);
    }
    reverse_fits.reverse();
    let mut decisions = Vec::new();
    let mut diagnostics = Vec::new();
    decisions
        .try_reserve_exact(non_terminal_dates)
        .map_err(|_| LsmNumericalError::AllocationFailed {
            resource: "exercise decisions",
            requested: non_terminal_dates,
        })?;
    diagnostics
        .try_reserve_exact(non_terminal_dates)
        .map_err(|_| LsmNumericalError::AllocationFailed {
            resource: "exercise diagnostics",
            requested: non_terminal_dates,
        })?;
    for fit in reverse_fits {
        decisions.push(fit.decision);
        diagnostics.push(fit.diagnostics);
    }
    let mut policy = ExercisePolicy {
        exercise_dates: policy_dates.into_boxed_slice(),
        basis,
        itm_abs_tolerance,
        cpqr_config: config,
        max_matrix_elements,
        training_metadata,
        decisions: decisions.into_boxed_slice(),
        diagnostics: diagnostics.into_boxed_slice(),
        fingerprint: ExercisePolicyFingerprint::from_bytes([0; 32]),
    };
    policy.fingerprint = fingerprint_exercise_policy(&policy)?;
    Ok(ExercisePolicyTrainingOutcome {
        policy,
        realized_cashflows: realized_cashflows.into_boxed_slice(),
        stopping_indices: stopping_indices.into_boxed_slice(),
    })
}

fn fingerprint_exercise_policy(
    policy: &ExercisePolicy,
) -> Result<ExercisePolicyFingerprint, LsmNumericalError> {
    let mut hasher = blake3::Hasher::new();
    hash_bytes(&mut hasher, b"pricing/exercise-policy")?;
    hash_bytes(&mut hasher, LSM_POLICY_ABI.as_bytes())?;
    hash_bytes(&mut hasher, LSM_BASIS_ABI.as_bytes())?;
    hash_bytes(&mut hasher, LSM_REGRESSION_ABI.as_bytes())?;
    hasher.update(&policy.training_metadata.product_fingerprint);
    hasher.update(&policy.training_metadata.training_configuration_fingerprint);
    hasher.update(&policy.training_metadata.seed.to_be_bytes());
    hasher.update(&policy.training_metadata.sampling_units.to_be_bytes());
    hasher.update(&policy.training_metadata.trajectory_count.to_be_bytes());
    hasher.update(&policy.training_metadata.random_domain.id().to_be_bytes());
    hash_usize(&mut hasher, policy.exercise_dates.len(), "exercise_dates")?;
    for date in &policy.exercise_dates {
        hasher.update(&date.year().to_be_bytes());
        hasher.update(&[date.month(), date.day()]);
    }
    hasher.update(&policy.basis.feature_count.to_be_bytes());
    hasher.update(&policy.basis.max_degree.to_be_bytes());
    hash_usize(&mut hasher, policy.basis.exponents.len(), "basis_columns")?;
    for exponents in &policy.basis.exponents {
        hash_usize(&mut hasher, exponents.len(), "basis_exponents")?;
        for &exponent in exponents {
            hasher.update(&exponent.to_be_bytes());
        }
    }
    hasher.update(&policy.itm_abs_tolerance.to_bits().to_be_bytes());
    hasher.update(
        &policy
            .cpqr_config
            .abs_rank_tolerance
            .to_bits()
            .to_be_bytes(),
    );
    hasher.update(
        &policy
            .cpqr_config
            .rel_rank_tolerance
            .to_bits()
            .to_be_bytes(),
    );
    hash_usize(
        &mut hasher,
        policy.max_matrix_elements,
        "max_matrix_elements",
    )?;
    hash_usize(&mut hasher, policy.decisions.len(), "decisions")?;
    for decision in &policy.decisions {
        match decision {
            ExerciseDecisionModel::Regression(model) => {
                hasher.update(&[0]);
                hash_regression_model(&mut hasher, model)?;
            }
            ExerciseDecisionModel::ContinueAll { reason } => {
                hasher.update(&[1]);
                match reason {
                    ContinueAllReason::ZeroItmTrainingPaths => hasher.update(&[0]),
                };
            }
        }
    }
    hash_usize(&mut hasher, policy.diagnostics.len(), "diagnostics")?;
    for diagnostics in &policy.diagnostics {
        hash_usize(&mut hasher, diagnostics.candidate_rows, "candidate_rows")?;
        hash_usize(&mut hasher, diagnostics.itm_rows, "itm_rows")?;
        hash_usize(&mut hasher, diagnostics.feature_count, "feature_count")?;
        hash_usize(&mut hasher, diagnostics.warnings.len(), "warnings")?;
        for warning in &diagnostics.warnings {
            match warning {
                LsmWarning::ZeroItmTrainingPaths => {
                    hasher.update(&[0]);
                }
                LsmWarning::InactiveFeature { feature } => {
                    hasher.update(&[1]);
                    hash_usize(&mut hasher, *feature, "inactive_feature")?;
                }
                LsmWarning::RankExcludedBasisColumn { column } => {
                    hasher.update(&[2]);
                    hash_usize(&mut hasher, *column, "rank_excluded_column")?;
                }
            }
        }
    }
    Ok(ExercisePolicyFingerprint::from_bytes(
        *hasher.finalize().as_bytes(),
    ))
}

fn fingerprint_lsm_configuration(
    config: &LsmConfig,
) -> Result<LsmConfigurationFingerprint, LsmNumericalError> {
    let mut hasher = blake3::Hasher::new();
    hash_bytes(&mut hasher, b"pricing/lsm-configuration")?;
    hash_bytes(&mut hasher, LSM_POLICY_ABI.as_bytes())?;
    hash_bytes(&mut hasher, LSM_BASIS_ABI.as_bytes())?;
    hash_bytes(&mut hasher, LSM_REGRESSION_ABI.as_bytes())?;
    match config.training_engine {
        EngineConfig::PseudoMonteCarlo(engine) => {
            hasher.update(&[0]);
            hasher.update(&engine.master_seed().to_be_bytes());
            hasher.update(&engine.independent_sampling_units().get().to_be_bytes());
            hash_variance_reduction(&mut hasher, engine.variance_reduction());
        }
        EngineConfig::RandomizedQuasiMonteCarlo(engine) => {
            hasher.update(&[1]);
            hasher.update(&engine.points_per_scramble().get().to_be_bytes());
            hasher.update(&engine.scramble_count().get().to_be_bytes());
            hasher.update(&engine.master_scramble_seed().to_be_bytes());
            hash_variance_reduction(&mut hasher, engine.variance_reduction());
        }
    }
    hasher.update(&config.training_random_domain().id().to_be_bytes());
    hash_usize(&mut hasher, config.state_variables.len(), "state_variables")?;
    for state_variable in &config.state_variables {
        hasher.update(&[*state_variable as u8]);
    }
    hasher.update(&config.basis.feature_count.to_be_bytes());
    hasher.update(&config.basis.max_degree.to_be_bytes());
    hash_usize(&mut hasher, config.basis.exponents.len(), "basis_columns")?;
    for exponents in &config.basis.exponents {
        hash_usize(&mut hasher, exponents.len(), "basis_exponents")?;
        for &exponent in exponents {
            hasher.update(&exponent.to_be_bytes());
        }
    }
    hasher.update(&config.itm_abs_tolerance.to_bits().to_be_bytes());
    hasher.update(
        &config
            .cpqr_config
            .abs_rank_tolerance
            .to_bits()
            .to_be_bytes(),
    );
    hasher.update(
        &config
            .cpqr_config
            .rel_rank_tolerance
            .to_bits()
            .to_be_bytes(),
    );
    hash_usize(
        &mut hasher,
        config.max_matrix_elements,
        "max_matrix_elements",
    )?;
    Ok(LsmConfigurationFingerprint(*hasher.finalize().as_bytes()))
}

fn hash_variance_reduction(hasher: &mut blake3::Hasher, value: crate::VarianceReduction) {
    hasher.update(&[
        u8::from(value.antithetic()),
        u8::from(value.brownian_bridge()),
    ]);
}

fn hash_regression_model(
    hasher: &mut blake3::Hasher,
    model: &PolynomialRegressionModel,
) -> Result<(), LsmNumericalError> {
    hash_usize(hasher, model.feature_scalings.len(), "feature_scalings")?;
    for scaling in &model.feature_scalings {
        hasher.update(&scaling.mean.to_bits().to_be_bytes());
        hasher.update(&scaling.population_variance.to_bits().to_be_bytes());
        hasher.update(&scaling.scale.to_bits().to_be_bytes());
        hasher.update(&scaling.zero_scale_threshold.to_bits().to_be_bytes());
        hasher.update(&[u8::from(scaling.inactive)]);
    }
    hash_usize_slice(hasher, &model.active_basis_columns, "active_basis_columns")?;
    hash_usize_slice(
        hasher,
        &model.pre_excluded_basis_columns,
        "pre_excluded_basis_columns",
    )?;
    hash_usize_slice(hasher, &model.pivot_order, "pivot_order")?;
    hash_f64_slice(hasher, &model.diagonal_abs, "diagonal_abs")?;
    hasher.update(&model.rank_threshold.to_bits().to_be_bytes());
    hash_usize(hasher, model.rank, "rank")?;
    hash_usize_slice(
        hasher,
        &model.rank_excluded_basis_columns,
        "rank_excluded_basis_columns",
    )?;
    hash_f64_slice(hasher, &model.coefficients, "coefficients")?;
    hasher.update(&model.residual_sum_squares.to_bits().to_be_bytes());
    Ok(())
}

fn hash_bytes(hasher: &mut blake3::Hasher, bytes: &[u8]) -> Result<(), LsmNumericalError> {
    hash_usize(hasher, bytes.len(), "byte_string")?;
    hasher.update(bytes);
    Ok(())
}

fn hash_usize(
    hasher: &mut blake3::Hasher,
    value: usize,
    field: &'static str,
) -> Result<(), LsmNumericalError> {
    let value =
        u64::try_from(value).map_err(|_| LsmNumericalError::FingerprintCountOverflow { field })?;
    hasher.update(&value.to_be_bytes());
    Ok(())
}

fn hash_usize_slice(
    hasher: &mut blake3::Hasher,
    values: &[usize],
    field: &'static str,
) -> Result<(), LsmNumericalError> {
    hash_usize(hasher, values.len(), field)?;
    for &value in values {
        hash_usize(hasher, value, field)?;
    }
    Ok(())
}

fn hash_f64_slice(
    hasher: &mut blake3::Hasher,
    values: &[f64],
    field: &'static str,
) -> Result<(), LsmNumericalError> {
    hash_usize(hasher, values.len(), field)?;
    for value in values {
        hasher.update(&value.to_bits().to_be_bytes());
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn fit_exercise_decision(
    basis: PolynomialBasisSpec,
    candidate_features: &[f64],
    candidate_rows: usize,
    immediate_values: &[f64],
    continuation_targets: &[f64],
    itm_abs_tolerance: f64,
    config: CpqrConfig,
    max_matrix_elements: usize,
) -> Result<DateLocalExerciseFit, LsmNumericalError> {
    validate_tolerance("itm_abs_tolerance", itm_abs_tolerance)?;
    if candidate_rows == 0 {
        return Err(LsmNumericalError::ZeroMatrixRows);
    }
    if max_matrix_elements == 0 {
        return Err(LsmNumericalError::ZeroResourceLimit {
            resource: "regression_matrix_elements",
        });
    }
    let feature_count = basis.feature_count as usize;
    let expected_features = candidate_rows
        .checked_mul(feature_count)
        .ok_or(LsmNumericalError::MatrixShapeOverflow)?;
    if candidate_features.len() != expected_features {
        return Err(LsmNumericalError::FeatureMatrixLengthMismatch {
            expected: expected_features,
            actual: candidate_features.len(),
        });
    }
    if immediate_values.len() != candidate_rows {
        return Err(LsmNumericalError::ImmediateValueLengthMismatch {
            expected: candidate_rows,
            actual: immediate_values.len(),
        });
    }
    if continuation_targets.len() != candidate_rows {
        return Err(LsmNumericalError::ContinuationTargetLengthMismatch {
            expected: candidate_rows,
            actual: continuation_targets.len(),
        });
    }
    for row in 0..candidate_rows {
        if !immediate_values[row].is_finite() {
            return Err(LsmNumericalError::NonFiniteImmediateValue {
                row,
                bits: immediate_values[row].to_bits(),
            });
        }
        if !continuation_targets[row].is_finite() {
            return Err(LsmNumericalError::NonFiniteContinuationTarget {
                row,
                bits: continuation_targets[row].to_bits(),
            });
        }
        for feature in 0..feature_count {
            let value = candidate_features[row * feature_count + feature];
            if !value.is_finite() {
                return Err(LsmNumericalError::NonFiniteFeature {
                    index: row * feature_count + feature,
                    bits: value.to_bits(),
                });
            }
        }
    }

    let itm_rows = immediate_values
        .iter()
        .filter(|&&value| value > itm_abs_tolerance)
        .count();
    if itm_rows == 0 {
        return Ok(DateLocalExerciseFit {
            decision: ExerciseDecisionModel::ContinueAll {
                reason: ContinueAllReason::ZeroItmTrainingPaths,
            },
            diagnostics: ExerciseRegressionDiagnostics {
                candidate_rows,
                itm_rows,
                feature_count,
                warnings: Box::new([LsmWarning::ZeroItmTrainingPaths]),
            },
        });
    }

    let itm_feature_elements = itm_rows
        .checked_mul(feature_count)
        .ok_or(LsmNumericalError::MatrixShapeOverflow)?;
    if itm_feature_elements > max_matrix_elements {
        return Err(LsmNumericalError::MatrixElementLimitExceeded {
            requested: itm_feature_elements,
            maximum: max_matrix_elements,
        });
    }
    let mut itm_features = zeroed_values(itm_feature_elements, "ITM feature matrix")?;
    let mut itm_targets = zeroed_values(itm_rows, "ITM continuation targets")?;
    let mut itm_index = 0;
    for row in 0..candidate_rows {
        if immediate_values[row] <= itm_abs_tolerance {
            continue;
        }
        let source_start = row * feature_count;
        let target_start = itm_index * feature_count;
        itm_features[target_start..target_start + feature_count]
            .copy_from_slice(&candidate_features[source_start..source_start + feature_count]);
        itm_targets[itm_index] = continuation_targets[row];
        itm_index += 1;
    }
    debug_assert_eq!(itm_index, itm_rows);

    let model = fit_polynomial_regression(
        basis,
        &itm_features,
        itm_rows,
        &itm_targets,
        config,
        max_matrix_elements,
    )?;
    let mut warnings = Vec::new();
    for (feature, scaling) in model.feature_scalings().iter().enumerate() {
        if scaling.inactive() {
            warnings.push(LsmWarning::InactiveFeature { feature });
        }
    }
    warnings.extend(
        model
            .rank_excluded_basis_columns()
            .iter()
            .map(|&column| LsmWarning::RankExcludedBasisColumn { column }),
    );
    Ok(DateLocalExerciseFit {
        decision: ExerciseDecisionModel::Regression(model),
        diagnostics: ExerciseRegressionDiagnostics {
            candidate_rows,
            itm_rows,
            feature_count,
            warnings: warnings.into_boxed_slice(),
        },
    })
}

pub fn fit_polynomial_regression(
    basis: PolynomialBasisSpec,
    features: &[f64],
    rows: usize,
    target: &[f64],
    config: CpqrConfig,
    max_matrix_elements: usize,
) -> Result<PolynomialRegressionModel, LsmNumericalError> {
    if rows == 0 {
        return Err(LsmNumericalError::ZeroMatrixRows);
    }
    if max_matrix_elements == 0 {
        return Err(LsmNumericalError::ZeroResourceLimit {
            resource: "regression_matrix_elements",
        });
    }
    if rows > max_matrix_elements {
        return Err(LsmNumericalError::MatrixElementLimitExceeded {
            requested: rows,
            maximum: max_matrix_elements,
        });
    }
    let feature_count = basis.feature_count as usize;
    let expected_features = rows
        .checked_mul(feature_count)
        .ok_or(LsmNumericalError::MatrixShapeOverflow)?;
    if features.len() != expected_features {
        return Err(LsmNumericalError::FeatureMatrixLengthMismatch {
            expected: expected_features,
            actual: features.len(),
        });
    }
    if target.len() != rows {
        return Err(LsmNumericalError::TargetLengthMismatch {
            expected: rows,
            actual: target.len(),
        });
    }
    for (row, &value) in target.iter().enumerate() {
        if !value.is_finite() {
            return Err(LsmNumericalError::NonFiniteTargetValue {
                row,
                bits: value.to_bits(),
            });
        }
    }

    let mut feature_scalings = Vec::new();
    feature_scalings
        .try_reserve_exact(feature_count)
        .map_err(|_| LsmNumericalError::AllocationFailed {
            resource: "feature scalings",
            requested: feature_count,
        })?;
    let mut feature_column = if feature_count == 0 {
        Vec::new()
    } else {
        zeroed_values(rows, "feature scaling column")?
    };
    for feature in 0..feature_count {
        for row in 0..rows {
            feature_column[row] = features[row * feature_count + feature];
        }
        feature_scalings.push(FeatureScaling::fit(&feature_column)?);
    }

    let mut active_basis_columns = Vec::new();
    let mut pre_excluded_basis_columns = Vec::new();
    active_basis_columns
        .try_reserve_exact(basis.exponents.len())
        .map_err(|_| LsmNumericalError::AllocationFailed {
            resource: "active basis columns",
            requested: basis.exponents.len(),
        })?;
    pre_excluded_basis_columns
        .try_reserve_exact(basis.exponents.len())
        .map_err(|_| LsmNumericalError::AllocationFailed {
            resource: "pre-excluded basis columns",
            requested: basis.exponents.len(),
        })?;
    for (column, exponents) in basis.exponents.iter().enumerate() {
        let depends_on_inactive = exponents
            .iter()
            .zip(&feature_scalings)
            .any(|(&exponent, scaling)| exponent != 0 && scaling.inactive);
        if depends_on_inactive {
            pre_excluded_basis_columns.push(column);
        } else {
            active_basis_columns.push(column);
        }
    }
    debug_assert!(!active_basis_columns.is_empty());
    let matrix_elements = rows
        .checked_mul(active_basis_columns.len())
        .ok_or(LsmNumericalError::MatrixShapeOverflow)?;
    if matrix_elements > max_matrix_elements {
        return Err(LsmNumericalError::MatrixElementLimitExceeded {
            requested: matrix_elements,
            maximum: max_matrix_elements,
        });
    }
    let mut design = zeroed_values(matrix_elements, "polynomial design matrix")?;
    let mut raw_row = zeroed_values(feature_count, "raw feature row")?;
    for row in 0..rows {
        for feature in 0..feature_count {
            raw_row[feature] = features[row * feature_count + feature];
        }
        let standardized = standardize_features(&raw_row, &feature_scalings)?;
        for (active_column, &basis_column) in active_basis_columns.iter().enumerate() {
            design[row * active_basis_columns.len() + active_column] =
                evaluate_monomial(&standardized, &basis.exponents[basis_column])?;
        }
    }

    let fit = fit_cpqr(&design, rows, active_basis_columns.len(), target, config)?;
    let mut pivot_order = Vec::new();
    pivot_order
        .try_reserve_exact(fit.pivot_order.len())
        .map_err(|_| LsmNumericalError::AllocationFailed {
            resource: "original basis pivot order",
            requested: fit.pivot_order.len(),
        })?;
    pivot_order.extend(
        fit.pivot_order
            .iter()
            .map(|&active_column| active_basis_columns[active_column]),
    );
    let mut rank_excluded_basis_columns = Vec::new();
    let rank_excluded_count = pivot_order.len().saturating_sub(fit.rank);
    rank_excluded_basis_columns
        .try_reserve_exact(rank_excluded_count)
        .map_err(|_| LsmNumericalError::AllocationFailed {
            resource: "rank-excluded basis columns",
            requested: rank_excluded_count,
        })?;
    rank_excluded_basis_columns.extend_from_slice(&pivot_order[fit.rank..]);
    let mut coefficients = zeroed_values(basis.exponents.len(), "basis coefficients")?;
    for (active_column, &basis_column) in active_basis_columns.iter().enumerate() {
        coefficients[basis_column] = fit.coefficients[active_column];
    }
    Ok(PolynomialRegressionModel {
        basis,
        feature_scalings: feature_scalings.into_boxed_slice(),
        active_basis_columns: active_basis_columns.into_boxed_slice(),
        pre_excluded_basis_columns: pre_excluded_basis_columns.into_boxed_slice(),
        pivot_order: pivot_order.into_boxed_slice(),
        diagonal_abs: fit.diagonal_abs,
        rank_threshold: fit.rank_threshold,
        rank: fit.rank,
        rank_excluded_basis_columns: rank_excluded_basis_columns.into_boxed_slice(),
        coefficients: coefficients.into_boxed_slice(),
        residual_sum_squares: fit.residual_sum_squares,
    })
}

fn standardize_features(
    features: &[f64],
    scalings: &[FeatureScaling],
) -> Result<Vec<f64>, LsmNumericalError> {
    if features.len() != scalings.len() {
        return Err(LsmNumericalError::FeatureCountMismatch {
            expected: scalings.len(),
            actual: features.len(),
        });
    }
    let mut standardized = zeroed_values(features.len(), "standardized feature row")?;
    for (index, (&feature, &scaling)) in features.iter().zip(scalings).enumerate() {
        if !feature.is_finite() {
            return Err(LsmNumericalError::NonFiniteFeature {
                index,
                bits: feature.to_bits(),
            });
        }
        standardized[index] = if scaling.inactive {
            0.0
        } else {
            scaling.standardize(feature)?
        };
    }
    Ok(standardized)
}

fn evaluate_monomial(features: &[f64], exponents: &[u32]) -> Result<f64, LsmNumericalError> {
    let mut value = 1.0;
    for (&feature, &exponent) in features.iter().zip(exponents) {
        for _ in 0..exponent {
            value *= feature;
            if !value.is_finite() {
                return Err(LsmNumericalError::NonFiniteIntermediate {
                    stage: "basis value",
                });
            }
        }
    }
    Ok(value)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CpqrConfig {
    abs_rank_tolerance: f64,
    rel_rank_tolerance: f64,
}

impl CpqrConfig {
    pub fn new(
        abs_rank_tolerance: f64,
        rel_rank_tolerance: f64,
    ) -> Result<Self, LsmNumericalError> {
        validate_tolerance("abs_rank_tolerance", abs_rank_tolerance)?;
        validate_tolerance("rel_rank_tolerance", rel_rank_tolerance)?;
        Ok(Self {
            abs_rank_tolerance,
            rel_rank_tolerance,
        })
    }

    #[must_use]
    pub const fn abs_rank_tolerance(self) -> f64 {
        self.abs_rank_tolerance
    }

    #[must_use]
    pub const fn rel_rank_tolerance(self) -> f64 {
        self.rel_rank_tolerance
    }
}

fn validate_tolerance(name: &'static str, value: f64) -> Result<(), LsmNumericalError> {
    if !value.is_finite() || value < 0.0 {
        return Err(LsmNumericalError::InvalidTolerance {
            name,
            bits: value.to_bits(),
        });
    }
    Ok(())
}

pub fn is_training_itm(
    immediate_value: f64,
    itm_abs_tolerance: f64,
) -> Result<bool, LsmNumericalError> {
    validate_decision_value("immediate_value", immediate_value)?;
    validate_tolerance("itm_abs_tolerance", itm_abs_tolerance)?;
    Ok(immediate_value > itm_abs_tolerance)
}

pub fn should_exercise(
    immediate_value: f64,
    continuation_value: f64,
) -> Result<bool, LsmNumericalError> {
    validate_decision_value("immediate_value", immediate_value)?;
    validate_decision_value("continuation_value", continuation_value)?;
    Ok(immediate_value > continuation_value)
}

fn validate_decision_value(name: &'static str, value: f64) -> Result<(), LsmNumericalError> {
    if !value.is_finite() {
        return Err(LsmNumericalError::NonFiniteDecisionValue {
            name,
            bits: value.to_bits(),
        });
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq)]
pub struct CpqrFit {
    pivot_order: Box<[usize]>,
    diagonal_abs: Box<[f64]>,
    rank_threshold: f64,
    rank: usize,
    coefficients: Box<[f64]>,
    residual_sum_squares: f64,
}

impl CpqrFit {
    #[must_use]
    pub fn pivot_order(&self) -> &[usize] {
        &self.pivot_order
    }

    #[must_use]
    pub fn diagonal_abs(&self) -> &[f64] {
        &self.diagonal_abs
    }

    #[must_use]
    pub const fn rank_threshold(&self) -> f64 {
        self.rank_threshold
    }

    #[must_use]
    pub const fn rank(&self) -> usize {
        self.rank
    }

    #[must_use]
    pub fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }

    #[must_use]
    pub const fn residual_sum_squares(&self) -> f64 {
        self.residual_sum_squares
    }
}

pub fn fit_cpqr(
    matrix: &[f64],
    rows: usize,
    columns: usize,
    target: &[f64],
    config: CpqrConfig,
) -> Result<CpqrFit, LsmNumericalError> {
    if rows == 0 {
        return Err(LsmNumericalError::ZeroMatrixRows);
    }
    if columns == 0 {
        return Err(LsmNumericalError::ZeroMatrixColumns);
    }
    let matrix_length = rows
        .checked_mul(columns)
        .ok_or(LsmNumericalError::MatrixShapeOverflow)?;
    if matrix.len() != matrix_length {
        return Err(LsmNumericalError::MatrixLengthMismatch {
            expected: matrix_length,
            actual: matrix.len(),
        });
    }
    if target.len() != rows {
        return Err(LsmNumericalError::TargetLengthMismatch {
            expected: rows,
            actual: target.len(),
        });
    }
    for row in 0..rows {
        for column in 0..columns {
            let value = matrix[row * columns + column];
            if !value.is_finite() {
                return Err(LsmNumericalError::NonFiniteMatrixValue {
                    row,
                    column,
                    bits: value.to_bits(),
                });
            }
        }
        if !target[row].is_finite() {
            return Err(LsmNumericalError::NonFiniteTargetValue {
                row,
                bits: target[row].to_bits(),
            });
        }
    }
    let mut qr = copied_values(matrix, "QR matrix")?;
    let original = copied_values(matrix, "original matrix")?;
    let mut transformed_target = copied_values(target, "QR target")?;
    let mut permutation = Vec::new();
    permutation
        .try_reserve_exact(columns)
        .map_err(|_| LsmNumericalError::AllocationFailed {
            resource: "QR permutation",
            requested: columns,
        })?;
    permutation.extend(0..columns);
    let diagonal_count = rows.min(columns);
    let mut diagonal_abs = Vec::new();
    diagonal_abs
        .try_reserve_exact(diagonal_count)
        .map_err(|_| LsmNumericalError::AllocationFailed {
            resource: "QR diagonal",
            requested: diagonal_count,
        })?;

    for pivot_position in 0..diagonal_count {
        let mut selected = pivot_position;
        let mut selected_norm = -1.0_f64;
        for column in pivot_position..columns {
            let squared_norm = column_squared_norm(&qr, rows, columns, pivot_position, column)?;
            if squared_norm > selected_norm
                || (squared_norm == selected_norm && permutation[column] < permutation[selected])
            {
                selected = column;
                selected_norm = squared_norm;
            }
        }
        if selected != pivot_position {
            for row in 0..rows {
                qr.swap(row * columns + pivot_position, row * columns + selected);
            }
            permutation.swap(pivot_position, selected);
        }

        let norm = column_norm(&qr, rows, columns, pivot_position, pivot_position)?;
        if norm == 0.0 {
            qr[pivot_position * columns + pivot_position] = 0.0;
            diagonal_abs.push(0.0);
            continue;
        }
        let diagonal_index = pivot_position * columns + pivot_position;
        let leading = qr[diagonal_index];
        let alpha = if leading >= 0.0 { -norm } else { norm };
        let denominator = leading - alpha;
        let tau = (alpha - leading) / alpha;
        if !denominator.is_finite() || !tau.is_finite() {
            return Err(LsmNumericalError::NonFiniteIntermediate {
                stage: "Householder reflector",
            });
        }
        qr[diagonal_index] = alpha;
        for row in pivot_position + 1..rows {
            let index = row * columns + pivot_position;
            qr[index] /= denominator;
        }
        for column in pivot_position + 1..columns {
            let mut dot = NeumaierSum::new();
            dot.add(qr[pivot_position * columns + column]);
            for row in pivot_position + 1..rows {
                dot.add(qr[row * columns + pivot_position] * qr[row * columns + column]);
            }
            let projection = tau * dot.total();
            qr[pivot_position * columns + column] -= projection;
            for row in pivot_position + 1..rows {
                let index = row * columns + column;
                qr[index] -= qr[row * columns + pivot_position] * projection;
            }
        }
        let mut target_dot = NeumaierSum::new();
        target_dot.add(transformed_target[pivot_position]);
        for row in pivot_position + 1..rows {
            target_dot.add(qr[row * columns + pivot_position] * transformed_target[row]);
        }
        let target_projection = tau * target_dot.total();
        transformed_target[pivot_position] -= target_projection;
        for row in pivot_position + 1..rows {
            transformed_target[row] -= qr[row * columns + pivot_position] * target_projection;
        }
        if qr.iter().any(|value| !value.is_finite())
            || transformed_target.iter().any(|value| !value.is_finite())
        {
            return Err(LsmNumericalError::NonFiniteIntermediate {
                stage: "Householder update",
            });
        }
        diagonal_abs.push(alpha.abs());
    }

    let first_diagonal = diagonal_abs.first().copied().unwrap_or(0.0);
    let rank_threshold = config
        .abs_rank_tolerance
        .max(config.rel_rank_tolerance * first_diagonal);
    if !rank_threshold.is_finite() {
        return Err(LsmNumericalError::NonFiniteIntermediate {
            stage: "rank threshold",
        });
    }
    let rank = diagonal_abs
        .iter()
        .take_while(|&&diagonal| diagonal > rank_threshold)
        .count();
    let mut pivot_coefficients = zeroed_values(columns, "pivot coefficients")?;
    for row in (0..rank).rev() {
        let mut upper_product = NeumaierSum::new();
        for column in row + 1..rank {
            upper_product.add(qr[row * columns + column] * pivot_coefficients[column]);
        }
        let coefficient =
            (transformed_target[row] - upper_product.total()) / qr[row * columns + row];
        if !coefficient.is_finite() {
            return Err(LsmNumericalError::NonFiniteIntermediate {
                stage: "back substitution",
            });
        }
        pivot_coefficients[row] = coefficient;
    }
    let mut coefficients = zeroed_values(columns, "original-order coefficients")?;
    for (pivot_position, &original_column) in permutation.iter().enumerate() {
        coefficients[original_column] = pivot_coefficients[pivot_position];
    }
    let mut residual_sum = NeumaierSum::new();
    for row in 0..rows {
        let mut prediction = NeumaierSum::new();
        for column in 0..columns {
            prediction.add(original[row * columns + column] * coefficients[column]);
        }
        let residual = target[row] - prediction.total();
        residual_sum.add(residual * residual);
    }
    let residual_sum_squares = residual_sum.total();
    if !residual_sum_squares.is_finite() || residual_sum_squares < 0.0 {
        return Err(LsmNumericalError::NonFiniteIntermediate {
            stage: "residual sum of squares",
        });
    }
    Ok(CpqrFit {
        pivot_order: permutation.into_boxed_slice(),
        diagonal_abs: diagonal_abs.into_boxed_slice(),
        rank_threshold,
        rank,
        coefficients: coefficients.into_boxed_slice(),
        residual_sum_squares,
    })
}

fn copied_values(values: &[f64], resource: &'static str) -> Result<Vec<f64>, LsmNumericalError> {
    let mut copied = Vec::new();
    copied
        .try_reserve_exact(values.len())
        .map_err(|_| LsmNumericalError::AllocationFailed {
            resource,
            requested: values.len(),
        })?;
    copied.extend_from_slice(values);
    Ok(copied)
}

fn zeroed_values(length: usize, resource: &'static str) -> Result<Vec<f64>, LsmNumericalError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|_| LsmNumericalError::AllocationFailed {
            resource,
            requested: length,
        })?;
    values.resize(length, 0.0);
    Ok(values)
}

fn column_squared_norm(
    matrix: &[f64],
    rows: usize,
    columns: usize,
    start_row: usize,
    column: usize,
) -> Result<f64, LsmNumericalError> {
    let mut norm = NeumaierSum::new();
    for row in start_row..rows {
        let value = matrix[row * columns + column];
        norm.add(value * value);
    }
    let squared_norm = norm.total();
    if !squared_norm.is_finite() || squared_norm < 0.0 {
        return Err(LsmNumericalError::NonFiniteIntermediate {
            stage: "pivot column norm",
        });
    }
    Ok(squared_norm)
}

fn column_norm(
    matrix: &[f64],
    rows: usize,
    columns: usize,
    start_row: usize,
    column: usize,
) -> Result<f64, LsmNumericalError> {
    let mut scale = 0.0_f64;
    let mut sum_squares = 1.0_f64;
    for row in start_row..rows {
        let value = matrix[row * columns + column].abs();
        if value == 0.0 {
            continue;
        }
        if scale < value {
            let ratio = scale / value;
            sum_squares = 1.0 + sum_squares * (ratio * ratio);
            scale = value;
        } else {
            let ratio = value / scale;
            sum_squares += ratio * ratio;
        }
    }
    let norm = if scale == 0.0 {
        0.0
    } else {
        scale * sum_squares.sqrt()
    };
    if !norm.is_finite() {
        return Err(LsmNumericalError::NonFiniteIntermediate {
            stage: "Householder column norm",
        });
    }
    Ok(norm)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::{PseudoMcConfig, RqmcConfig, VarianceReduction};

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
            crate::RandomDomain::LsmTrain
        );

        let rqmc = lsm_config(
            EngineConfig::RandomizedQuasiMonteCarlo(
                RqmcConfig::new(1024, 8, 11, VarianceReduction::new(true, true))
                    .expect("RQMC config"),
            ),
            2,
        );
        assert_eq!(rqmc.training_seed(), 11);
        assert_eq!(rqmc.training_effective_sampling_units(), 8);
        assert_eq!(rqmc.training_trajectory_count(), 16_384);
        assert_eq!(
            rqmc.training_random_domain(),
            crate::RandomDomain::RqmcScramble
        );
        let metadata = rqmc.training_metadata([0x42; 32]).expect("metadata");
        assert_eq!(metadata.seed(), 11);
        assert_eq!(metadata.sampling_units(), 8);
        assert_eq!(metadata.trajectory_count(), 16_384);
        assert_eq!(metadata.random_domain(), crate::RandomDomain::RqmcScramble);
    }

    #[test]
    fn lsm_config_validates_state_variable_contract() {
        let engine = EngineConfig::PseudoMonteCarlo(
            PseudoMcConfig::new(7, 100, VarianceReduction::new(false, false))
                .expect("pseudo config"),
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
                crate::RandomDomain::Valuation,
            ),
            Err(LsmNumericalError::InvalidTrainingRandomDomain {
                domain: crate::RandomDomain::Valuation
            })
        ));
    }

    #[test]
    fn lsm_configuration_fingerprint_is_canonical_and_complete() {
        let engine = EngineConfig::PseudoMonteCarlo(
            PseudoMcConfig::new(7, 100, VarianceReduction::new(true, false))
                .expect("pseudo config"),
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
            let fit =
                fit_cpqr(case.matrix, case.rows, case.columns, case.target, exact).expect("fit");
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
}
