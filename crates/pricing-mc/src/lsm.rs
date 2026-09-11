use std::error::Error;
use std::fmt;

use pricing_numerics::NeumaierSum;

pub const LSM_BASIS_ABI: &str = "polynomial-total-degree-v1";
pub const LSM_REGRESSION_ABI: &str = "cpqr-householder-v1";

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
    InactiveFeature,
    NonFiniteDecisionValue {
        name: &'static str,
        bits: u64,
    },
    InvalidTolerance {
        name: &'static str,
        bits: u64,
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
}
