//! Regularized inversion of the full equation-(3) source-hat operator.
use crate::risk::{RiskConfigError, VegaKtGammaAtoms, VegaKtStepMoment};
use pricing_numerics::NeumaierSum;

mod constraints;
mod gaussian_variance;
mod multi_transition_source;
mod two_transition_source;
pub use gaussian_variance::VegaKtVarianceOperatorAccuracy;
pub(crate) use gaussian_variance::legendre as gaussian_legendre;
pub use multi_transition_source::{VegaKtMultiTransitionSourceAccuracy, VegaKtSourceStepAccuracy};
pub use two_transition_source::VegaKtTwoTransitionSourceAccuracy;

/// A fixed linear inverse. Masses are signed Gamma measures, not nodal densities.
/// The normalization and regularization do not depend on the observed adjoints.
#[derive(Clone, Debug)]
pub struct VegaKtConvolutionPlan {
    pub(crate) forward: Vec<Vec<f64>>,
    pub(crate) inverse: Vec<Vec<f64>>,
    pub(crate) strikes: Vec<f64>,
    pub(crate) operator_scale: f64,
    pub(crate) relative_regularization: f64,
    pub(crate) preserves_local_vega_total: bool,
    pub(crate) conservation_directions: Vec<Vec<f64>>,
    /// Frobenius norm bounds amplification of Euclidean local-Vega noise.
    pub(crate) inverse_noise_gain_bound: f64,
    pub(crate) resolution_trace: f64,
}

fn invalid(quantity: &'static str) -> RiskConfigError {
    RiskConfigError::VegaKtChainInput { quantity }
}
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(a, b)| a * b)
        .collect::<NeumaierSum>()
        .total()
}
fn apply(a: &[Vec<f64>], b: &[f64]) -> Result<Vec<f64>, RiskConfigError> {
    if b.len() != a[0].len() || b.iter().any(|v| !v.is_finite()) {
        return Err(invalid(
            "finite convolution input with matching dimension required",
        ));
    }
    let out: Vec<_> = a.iter().map(|row| dot(row, b)).collect();
    if out.iter().any(|v| !v.is_finite()) {
        return Err(invalid("nonfinite convolution output"));
    }
    Ok(out)
}

impl VegaKtConvolutionPlan {
    /// Flat spatial variance and zero carry. Integrates the source density and
    /// source log hats analytically, including boundary flat extrapolation.
    /// Each column is the equation-(3) local Vega of one unit destination atom.
    /// Time zero is rank one and is deliberately rejected, not inverted.
    /// Solves min ||B*q-b||² + (lambda*||B||_F)² ||q||² with signed q.
    pub fn new(
        spot: f64,
        source_variance: f64,
        step_variance: f64,
        sigma_dt: f64,
        source_log_nodes: &[f64],
        gamma_strikes: Vec<f64>,
        relative_regularization: f64,
    ) -> Result<Self, RiskConfigError> {
        if !source_variance.is_finite() || source_variance <= 0. {
            return Err(invalid(
                "positive source variance required; initial Gamma is not identifiable",
            ));
        }
        if !(1e-6..=1.).contains(&relative_regularization) {
            return Err(invalid(
                "relative regularization must be finite and in [1e-6, 1]",
            ));
        }
        let (m, n) = (source_log_nodes.len(), gamma_strikes.len());
        if !(2..=129).contains(&m)
            || !(2..=129).contains(&n)
            || gamma_strikes.windows(2).any(|p| p[0] >= p[1])
        {
            return Err(invalid(
                "2..129 strictly increasing convolution nodes required",
            ));
        }
        let mut forward = vec![vec![0.; n]; m];
        for (j, &k) in gamma_strikes.iter().enumerate() {
            let column = VegaKtGammaAtoms::new(vec![(k, 1.)])?.expected_source_hat_vegas(
                spot,
                source_variance,
                step_variance,
                sigma_dt,
                source_log_nodes,
            )?;
            for (row, v) in forward.iter_mut().zip(column) {
                row[j] = v;
            }
        }
        Self::from_operator(forward, gamma_strikes, relative_regularization)
    }

    fn from_operator(
        forward: Vec<Vec<f64>>,
        gamma_strikes: Vec<f64>,
        relative_regularization: f64,
    ) -> Result<Self, RiskConfigError> {
        let (m, n) = (forward.len(), gamma_strikes.len());
        if !(2..=129).contains(&m)
            || !(2..=129).contains(&n)
            || !(1e-6..=1.).contains(&relative_regularization)
            || forward
                .iter()
                .any(|r| r.len() != n || r.iter().any(|v| !v.is_finite()))
        {
            return Err(invalid("invalid convolution matrix"));
        }
        let scale = forward
            .iter()
            .flat_map(|r| r.iter())
            .map(|v| v * v)
            .collect::<NeumaierSum>()
            .total()
            .sqrt();
        if !scale.is_finite() || scale <= 0. {
            return Err(invalid("degenerate convolution operator"));
        }
        let a: Vec<Vec<f64>> = forward
            .iter()
            .map(|r| r.iter().map(|v| v / scale).collect())
            .collect();
        // Positive ridge and normalized matrix bound conditioning. No external
        // inverse, positive-mass fit, residual normalization or threshold tuning.
        let mut lower = vec![vec![0.; m]; m];
        for i in 0..m {
            for j in 0..=i {
                let mut value = dot(&a[i], &a[j]) - dot(&lower[i][..j], &lower[j][..j]);
                if i == j {
                    value += relative_regularization.powi(2);
                    if !value.is_finite() || value <= 0. {
                        return Err(invalid("convolution factorization failed"));
                    }
                    lower[i][j] = value.sqrt();
                } else {
                    lower[i][j] = value / lower[j][j];
                }
            }
        }
        let mut columns = vec![vec![0.; n]; m];
        for (column, output) in columns.iter_mut().enumerate() {
            let mut y = vec![0.; m];
            for i in 0..m {
                y[i] = (f64::from(i == column) - dot(&lower[i][..i], &y[..i])) / lower[i][i];
            }
            let mut x = vec![0.; m];
            for i in (0..m).rev() {
                let sum = (i + 1..m)
                    .map(|j| lower[j][i] * x[j])
                    .collect::<NeumaierSum>()
                    .total();
                x[i] = (y[i] - sum) / lower[i][i];
            }
            for j in 0..n {
                output[j] = (0..m)
                    .map(|i| a[i][j] * x[i])
                    .collect::<NeumaierSum>()
                    .total()
                    / scale;
            }
        }
        let inverse: Vec<Vec<f64>> = (0..n)
            .map(|j| columns.iter().map(|c| c[j]).collect())
            .collect();
        let gain = inverse
            .iter()
            .flat_map(|r| r.iter())
            .map(|v| v * v)
            .collect::<NeumaierSum>()
            .total()
            .sqrt();
        let trace = (0..n)
            .map(|j| {
                (0..m)
                    .map(|i| inverse[j][i] * forward[i][j])
                    .collect::<NeumaierSum>()
                    .total()
            })
            .collect::<NeumaierSum>()
            .total();
        if !gain.is_finite() || !trace.is_finite() {
            return Err(invalid("nonfinite convolution inverse"));
        }
        Ok(Self {
            forward,
            inverse,
            strikes: gamma_strikes,
            operator_scale: scale,
            relative_regularization,
            preserves_local_vega_total: false,
            conservation_directions: Vec::new(),
            inverse_noise_gain_bound: gain,
            resolution_trace: trace,
        })
    }
    /// Equality-constrained Tikhonov: preserve the integral of the observed
    /// local Vega, 1'Bq = 1'b. This constructs a fixed linear inverse before
    /// seeing ANY payoff samples. It neither rescales fitted masses per sample
    /// nor removes the spatial residual or claims unique Gamma identification.
    pub fn conserve_local_vega_total(mut self) -> Result<Self, RiskConfigError> {
        if self.preserves_local_vega_total {
            return Ok(self);
        }
        if !self.conservation_directions.is_empty() {
            return Err(invalid(
                "compile all Gamma conservation constraints together",
            ));
        }
        let n = self.inverse.len();
        let m = self.forward.len();
        let c = (0..n)
            .map(|j| {
                self.forward
                    .iter()
                    .map(|r| r[j])
                    .collect::<NeumaierSum>()
                    .total()
            })
            .collect::<Vec<_>>();
        // R1 = (B'B + alpha^2 I)^(-1) B'1. The rank-one update is
        // the exact Lagrange-multiplier solution of the constrained objective.
        let direction = self
            .inverse
            .iter()
            .map(|r| r.iter().copied().collect::<NeumaierSum>().total())
            .collect::<Vec<_>>();
        let denominator = dot(&c, &direction);
        if !denominator.is_finite() || denominator <= 0. {
            return Err(invalid("unidentifiable local Vega conservation constraint"));
        }
        let residual = (0..m)
            .map(|i| {
                1. - (0..n)
                    .map(|j| c[j] * self.inverse[j][i])
                    .collect::<NeumaierSum>()
                    .total()
            })
            .collect::<Vec<_>>();
        for (row, d) in self.inverse.iter_mut().zip(direction) {
            for (v, &e) in row.iter_mut().zip(&residual) {
                *v += d / denominator * e;
                if !v.is_finite() {
                    return Err(invalid("nonfinite conservative inverse"));
                }
            }
        }
        self.inverse_noise_gain_bound = self
            .inverse
            .iter()
            .flatten()
            .fold(0., |n, &v| f64::hypot(n, v));
        self.resolution_trace = (0..n)
            .map(|j| {
                (0..m)
                    .map(|i| self.inverse[j][i] * self.forward[i][j])
                    .collect::<NeumaierSum>()
                    .total()
            })
            .collect::<NeumaierSum>()
            .total();
        if !self.inverse_noise_gain_bound.is_finite() || !self.resolution_trace.is_finite() {
            return Err(invalid("nonfinite conservative inverse diagnostics"));
        }
        self.preserves_local_vega_total = true;
        Ok(self)
    }
    #[must_use]
    pub const fn preserves_local_vega_total(&self) -> bool {
        self.preserves_local_vega_total
    }
    #[must_use]
    pub fn conservation_directions(&self) -> &[Vec<f64>] {
        &self.conservation_directions
    }

    #[must_use]
    pub fn gamma_strikes(&self) -> &[f64] {
        &self.strikes
    }
    #[must_use]
    pub const fn operator_scale(&self) -> f64 {
        self.operator_scale
    }
    #[must_use]
    pub const fn relative_regularization(&self) -> f64 {
        self.relative_regularization
    }
    #[must_use]
    pub const fn inverse_noise_gain_bound(&self) -> f64 {
        self.inverse_noise_gain_bound
    }
    #[must_use]
    pub const fn resolution_trace(&self) -> f64 {
        self.resolution_trace
    }
    #[must_use]
    pub fn forward_matrix(&self) -> &[Vec<f64>] {
        &self.forward
    }
    #[must_use]
    pub fn recovery_matrix(&self) -> &[Vec<f64>] {
        &self.inverse
    }
    pub fn recover(&self, local_vega: &[f64]) -> Result<Vec<f64>, RiskConfigError> {
        apply(&self.inverse, local_vega)
    }
    pub fn reconstruct(&self, masses: &[f64]) -> Result<Vec<f64>, RiskConfigError> {
        apply(&self.forward, masses)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct VegaKtConvolutionRow {
    pub source_maturity: f64,
    pub recovered_maturity: f64,
    pub source_total_variance: f64,
    pub step_variance: f64,
    pub operator_scale: f64,
    pub inverse_noise_gain_bound: f64,
    pub resolution_trace: f64,
    pub gamma_masses: Vec<VegaKtStepMoment>,
    pub local_vega: Vec<VegaKtStepMoment>,
    /// Count nonzero pair/path contributions, not an assertion of tail coverage.
    pub local_vega_nonzero_sampling_units: Vec<u64>,
    /// Same-draw B*q-b statistics preserve covariance; this is NOT IV residual.
    pub reconstruction_difference: Vec<VegaKtStepMoment>,
    pub local_vega_total: VegaKtStepMoment,
    pub reconstructed_total: VegaKtStepMoment,
    pub total_difference: VegaKtStepMoment,
    /// Same-sample signed projections; covariance within each direction is kept.
    pub local_vega_directions: Vec<VegaKtStepMoment>,
    pub reconstructed_directions: Vec<VegaKtStepMoment>,
    pub direction_differences: Vec<VegaKtStepMoment>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VegaKtConvolutionDiagnostic {
    pub policy_label: &'static str,
    pub request_fingerprint: crate::Fingerprint,
    pub plan_fingerprint: crate::Fingerprint,
    pub gamma_strikes: Vec<f64>,
    pub source_log_nodes: Vec<f64>,
    pub relative_regularization: f64,
    pub preserves_local_vega_total: bool,
    pub conservation_directions: Vec<Vec<Vec<f64>>>,
    pub independent_sampling_units: u64,
    pub antithetic: bool,
    pub unrecovered_initial_local_vega: VegaKtStepMoment,
    pub rows: Vec<VegaKtConvolutionRow>,
}
