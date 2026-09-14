//! Deterministic, unpivoted factorization of positive-semidefinite correlations.
use std::{error::Error, fmt};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CorrelationToleranceConfig {
    pub symmetry_abs_tol: f64,
    pub diagonal_abs_tol: f64,
    pub psd_abs_tol: f64,
    pub psd_rel_tol: f64,
    pub zero_pivot_abs_tol: f64,
    pub zero_pivot_rel_tol: f64,
}
#[derive(Clone, Debug, PartialEq)]
pub enum CorrelationError {
    InvalidTolerance,
    Shape,
    NonFinite,
    Asymmetric,
    Diagonal,
    OutOfBounds,
    NotPsd {
        column: usize,
        pivot: f64,
    },
    SingularResidual {
        row: usize,
        column: usize,
        residual: f64,
    },
}
impl fmt::Display for CorrelationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid correlation: {self:?}")
    }
}
impl Error for CorrelationError {}

#[derive(Clone, Debug, PartialEq)]
pub struct CorrelationFactor {
    dimension: usize,
    raw: Box<[f64]>,
    canonical: Box<[f64]>,
    lower: Box<[f64]>,
    diagnostics: CorrelationDiagnostics,
}
#[derive(Clone, Debug, PartialEq)]
pub struct CorrelationDiagnostics {
    pub maximum_asymmetry: f64,
    pub maximum_diagonal_deviation: f64,
    pub maximum_adjustment: f64,
    pub scale: f64,
    pub psd_threshold: f64,
    pub zero_threshold: f64,
    pub pivots: Vec<f64>,
    pub zero_pivots: Vec<usize>,
    pub rank: usize,
    pub maximum_checked_residual: f64,
}
impl CorrelationFactor {
    pub const ABI: &'static str = "correlation-unpivoted-psd-v1";
    pub fn compile(
        matrix: Vec<Vec<f64>>,
        t: CorrelationToleranceConfig,
    ) -> Result<Self, CorrelationError> {
        let tolerances = [
            t.symmetry_abs_tol,
            t.diagonal_abs_tol,
            t.psd_abs_tol,
            t.psd_rel_tol,
            t.zero_pivot_abs_tol,
            t.zero_pivot_rel_tol,
        ];
        if tolerances.iter().any(|v| !v.is_finite() || *v < 0.0) {
            return Err(CorrelationError::InvalidTolerance);
        }
        let n = matrix.len();
        if n == 0 || matrix.iter().any(|r| r.len() != n) || n.checked_mul(n).is_none() {
            return Err(CorrelationError::Shape);
        }
        let raw: Vec<_> = matrix.into_iter().flatten().collect();
        if raw.iter().any(|x| !x.is_finite()) {
            return Err(CorrelationError::NonFinite);
        }
        let mut c = raw.clone();
        let mut asymmetry: f64 = 0.0;
        let mut diagonal: f64 = 0.0;
        for i in 0..n {
            diagonal = diagonal.max((raw[i * n + i] - 1.0).abs());
            if diagonal > t.diagonal_abs_tol {
                return Err(CorrelationError::Diagonal);
            }
            c[i * n + i] = 1.0;
            for j in i + 1..n {
                asymmetry = asymmetry.max((raw[i * n + j] - raw[j * n + i]).abs());
                if asymmetry > t.symmetry_abs_tol {
                    return Err(CorrelationError::Asymmetric);
                }
                let avg = (raw[i * n + j] + raw[j * n + i]) * 0.5;
                if !(-1.0..=1.0).contains(&avg) {
                    return Err(CorrelationError::OutOfBounds);
                }
                c[i * n + j] = avg;
                c[j * n + i] = avg;
            }
        }
        let scale = c
            .chunks_exact(n)
            .map(|r| r.iter().fold(0.0, |s, x| s + x.abs()))
            .fold(1.0, f64::max);
        let psd_threshold = t.psd_abs_tol.max(t.psd_rel_tol * scale);
        let zero_threshold = t.zero_pivot_abs_tol.max(t.zero_pivot_rel_tol * scale);
        if !psd_threshold.is_finite() || !zero_threshold.is_finite() {
            return Err(CorrelationError::InvalidTolerance);
        }
        let mut lower = vec![0.0; n * n];
        let mut pivots = Vec::with_capacity(n);
        let mut zero_pivots = Vec::new();
        let mut maximum_checked_residual: f64 = 0.0;
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..j {
                sum += lower[j * n + k] * lower[j * n + k];
            }
            let pivot = c[j * n + j] - sum;
            pivots.push(pivot);
            if !pivot.is_finite() || pivot < -psd_threshold {
                return Err(CorrelationError::NotPsd { column: j, pivot });
            }
            if pivot > zero_threshold {
                lower[j * n + j] = pivot.sqrt();
            } else {
                zero_pivots.push(j);
            }
            for i in j + 1..n {
                let mut sum = 0.0;
                for k in 0..j {
                    sum += lower[i * n + k] * lower[j * n + k];
                }
                let residual = c[i * n + j] - sum;
                if pivot > zero_threshold {
                    let value = residual / lower[j * n + j];
                    if !value.is_finite() {
                        return Err(CorrelationError::NonFinite);
                    }
                    lower[i * n + j] = value;
                } else {
                    maximum_checked_residual = maximum_checked_residual.max(residual.abs());
                    if !residual.is_finite() || residual.abs() > zero_threshold {
                        return Err(CorrelationError::SingularResidual {
                            row: i,
                            column: j,
                            residual,
                        });
                    }
                }
            }
        }
        let diagnostics = CorrelationDiagnostics {
            maximum_asymmetry: asymmetry,
            maximum_diagonal_deviation: diagonal,
            maximum_adjustment: raw
                .iter()
                .zip(&c)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max),
            scale,
            psd_threshold,
            zero_threshold,
            rank: n - zero_pivots.len(),
            pivots,
            zero_pivots,
            maximum_checked_residual,
        };
        Ok(Self {
            dimension: n,
            raw: raw.into(),
            canonical: c.into(),
            lower: lower.into(),
            diagnostics,
        })
    }
    pub fn dimension(&self) -> usize {
        self.dimension
    }
    pub fn raw(&self) -> &[f64] {
        &self.raw
    }
    pub fn canonical(&self) -> &[f64] {
        &self.canonical
    }
    pub fn lower(&self) -> &[f64] {
        &self.lower
    }
    pub fn diagnostics(&self) -> &CorrelationDiagnostics {
        &self.diagnostics
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn t() -> CorrelationToleranceConfig {
        CorrelationToleranceConfig {
            symmetry_abs_tol: 1e-12,
            diagonal_abs_tol: 1e-12,
            psd_abs_tol: 1e-12,
            psd_rel_tol: 1e-12,
            zero_pivot_abs_tol: 1e-12,
            zero_pivot_rel_tol: 1e-12,
        }
    }
    #[test]
    fn reconstructs_full_rank_and_singular_without_reordering() {
        for (c, rank) in [
            (
                vec![
                    vec![1.0, 0.2, -0.1],
                    vec![0.2, 1.0, 0.4],
                    vec![-0.1, 0.4, 1.0],
                ],
                3,
            ),
            (
                vec![
                    vec![1.0, 1.0, -1.0],
                    vec![1.0, 1.0, -1.0],
                    vec![-1.0, -1.0, 1.0],
                ],
                1,
            ),
        ] {
            let f = CorrelationFactor::compile(c, t()).unwrap();
            assert_eq!(f.diagnostics().rank, rank);
            for i in 0..3 {
                for j in 0..3 {
                    let actual = (0..3)
                        .map(|k| f.lower()[i * 3 + k] * f.lower()[j * 3 + k])
                        .sum::<f64>();
                    assert!((actual - f.canonical()[i * 3 + j]).abs() < 1e-14);
                }
            }
            for &j in &f.diagnostics().zero_pivots {
                for i in j..3 {
                    assert_eq!(f.lower()[i * 3 + j].to_bits(), 0f64.to_bits());
                }
            }
        }
    }
    #[test]
    fn records_canonicalization_and_never_clamps_bounds() {
        let f =
            CorrelationFactor::compile(vec![vec![1.0 + 1e-13, 0.5 + 1e-13], vec![0.5, 1.0]], t())
                .unwrap();
        assert_eq!(f.canonical()[0], 1.0);
        assert_eq!(f.canonical()[1], f.canonical()[2]);
        assert!(f.diagnostics().maximum_adjustment > 0.0);
        assert!(f.diagnostics().maximum_asymmetry > 0.0);
        assert_eq!(
            CorrelationFactor::compile(vec![vec![1.0, 1.0 + 1e-14], vec![1.0 + 1e-14, 1.0]], t()),
            Err(CorrelationError::OutOfBounds)
        );
    }
    #[test]
    fn rejects_shape_nonfinite_asymmetry_diagonal_and_non_psd() {
        for c in [
            vec![],
            vec![vec![1.0, 0.0]],
            vec![vec![f64::NAN]],
            vec![vec![2.0]],
            vec![vec![1.0, 0.1], vec![0.2, 1.0]],
            vec![
                vec![1.0, 0.9, 0.9],
                vec![0.9, 1.0, -0.9],
                vec![0.9, -0.9, 1.0],
            ],
        ] {
            assert!(CorrelationFactor::compile(c, t()).is_err());
        }
        let c = vec![
            vec![1.0, 1.0, 0.0],
            vec![1.0, 1.0, 0.2],
            vec![0.0, 0.2, 1.0],
        ];
        assert!(matches!(
            CorrelationFactor::compile(c, t()),
            Err(CorrelationError::SingularResidual { .. })
        ));
        let mut tol = t();
        tol.psd_rel_tol = f64::INFINITY;
        assert_eq!(
            CorrelationFactor::compile(vec![vec![1.0]], tol),
            Err(CorrelationError::InvalidTolerance)
        );
    }
}
