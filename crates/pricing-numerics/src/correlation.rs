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
    fn preserves_inputs_and_reconstructs_the_original_four_dimensional_matrix() {
        let matrix = vec![
            vec![1.0, 0.25, -0.5, 0.125],
            vec![0.25, 1.0, 0.375, -0.25],
            vec![-0.5, 0.375, 1.0, 0.125],
            vec![0.125, -0.25, 0.125, 1.0],
        ];
        let expected: Vec<_> = matrix.iter().flatten().copied().collect();
        let factor = CorrelationFactor::compile(matrix, t()).unwrap();
        assert_eq!(factor.dimension(), 4);
        assert_eq!(factor.raw(), expected);
        assert_eq!(factor.canonical(), expected);
        assert_eq!(factor.diagnostics().rank, 4);
        assert_eq!(factor.diagnostics().scale, 2.0);
        assert_eq!(factor.diagnostics().maximum_adjustment, 0.0);
        assert!(factor.diagnostics().zero_pivots.is_empty());
        for i in 0..4 {
            for j in 0..4 {
                let reconstructed: f64 = (0..4)
                    .map(|k| factor.lower()[i * 4 + k] * factor.lower()[j * 4 + k])
                    .sum();
                // Compare with the input, independently of canonicalization.
                assert!((reconstructed - expected[i * 4 + j]).abs() < 1e-14);
                if j > i {
                    assert_eq!(factor.lower()[i * 4 + j], 0.0);
                }
            }
        }
    }

    #[test]
    fn canonicalization_and_diagnostic_thresholds_are_exact() {
        // Dyadic values make the inclusive tolerance boundaries exact in f64.
        let tolerance = CorrelationToleranceConfig {
            symmetry_abs_tol: 0.25,
            diagonal_abs_tol: 0.125,
            psd_abs_tol: 0.03125,
            psd_rel_tol: 0.125,
            zero_pivot_abs_tol: 0.125,
            zero_pivot_rel_tol: 0.0625,
        };
        let factor =
            CorrelationFactor::compile(vec![vec![1.125, 0.25], vec![0.5, 1.0]], tolerance).unwrap();
        assert_eq!(factor.raw(), [1.125, 0.25, 0.5, 1.0]);
        assert_eq!(factor.canonical(), [1.0, 0.375, 0.375, 1.0]);
        assert_eq!(factor.lower(), [1.0, 0.0, 0.375, 0.859375_f64.sqrt()]);
        assert_eq!(
            *factor.diagnostics(),
            CorrelationDiagnostics {
                maximum_asymmetry: 0.25,
                maximum_diagonal_deviation: 0.125,
                maximum_adjustment: 0.125,
                scale: 1.375,
                psd_threshold: 0.171875,
                zero_threshold: 0.125,
                pivots: vec![1.0, 0.859375],
                zero_pivots: vec![],
                rank: 2,
                maximum_checked_residual: 0.0,
            }
        );
        assert_eq!(
            CorrelationFactor::compile(
                vec![vec![1.125_f64.next_up(), 0.25], vec![0.5, 1.0]],
                tolerance,
            ),
            Err(CorrelationError::Diagonal)
        );
        assert_eq!(
            CorrelationFactor::compile(
                vec![vec![1.0, 0.25], vec![0.5_f64.next_up(), 1.0]],
                tolerance,
            ),
            Err(CorrelationError::Asymmetric)
        );
        let alternate = CorrelationFactor::compile(
            vec![vec![1.125, 0.25], vec![0.5, 1.0]],
            CorrelationToleranceConfig {
                psd_abs_tol: 0.25,
                psd_rel_tol: 0.0625,
                zero_pivot_abs_tol: 0.03125,
                zero_pivot_rel_tol: 0.125,
                ..tolerance
            },
        )
        .unwrap();
        assert_eq!(alternate.diagnostics().psd_threshold, 0.25);
        assert_eq!(alternate.diagnostics().zero_threshold, 0.171875);
    }

    #[test]
    fn validates_each_tolerance_and_independent_threshold_overflow() {
        for index in 0..6 {
            for invalid in [-1.0, f64::NEG_INFINITY, f64::INFINITY, f64::NAN] {
                let mut tolerance = t();
                let fields = [
                    &mut tolerance.symmetry_abs_tol,
                    &mut tolerance.diagonal_abs_tol,
                    &mut tolerance.psd_abs_tol,
                    &mut tolerance.psd_rel_tol,
                    &mut tolerance.zero_pivot_abs_tol,
                    &mut tolerance.zero_pivot_rel_tol,
                ];
                *fields[index] = invalid;
                assert_eq!(
                    CorrelationFactor::compile(vec![vec![1.0]], tolerance),
                    Err(CorrelationError::InvalidTolerance),
                    "tolerance {index}: {invalid}"
                );
            }
        }
        for (psd_rel_tol, zero_pivot_rel_tol) in [(f64::MAX, 0.0), (0.0, f64::MAX)] {
            assert_eq!(
                CorrelationFactor::compile(
                    vec![vec![1.0, 1.0], vec![1.0, 1.0]],
                    CorrelationToleranceConfig {
                        psd_rel_tol,
                        zero_pivot_rel_tol,
                        ..t()
                    },
                ),
                Err(CorrelationError::InvalidTolerance)
            );
        }
    }

    #[test]
    fn zero_pivots_and_psd_and_residual_boundaries_are_inclusive() {
        let exact = CorrelationToleranceConfig {
            symmetry_abs_tol: 0.0,
            diagonal_abs_tol: 0.0,
            psd_abs_tol: 0.0,
            psd_rel_tol: 0.0,
            zero_pivot_abs_tol: 0.0,
            zero_pivot_rel_tol: 0.0,
        };
        let factor = CorrelationFactor::compile(vec![vec![1.0; 3]; 3], exact).unwrap();
        assert_eq!(factor.diagnostics().rank, 1);
        assert_eq!(factor.diagnostics().pivots, [1.0, 0.0, 0.0]);
        assert_eq!(factor.diagnostics().zero_pivots, [1, 2]);
        assert_eq!(
            factor.lower(),
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        );

        let indefinite = vec![vec![1.0, 0.0, 1.0], vec![0.0, 1.0, 1.0], vec![1.0; 3]];
        let factor = CorrelationFactor::compile(
            indefinite.clone(),
            CorrelationToleranceConfig {
                psd_abs_tol: 1.0,
                ..exact
            },
        )
        .unwrap();
        assert_eq!(factor.diagnostics().pivots, [1.0, 1.0, -1.0]);
        assert_eq!(factor.diagnostics().zero_pivots, [2]);
        assert_eq!(factor.diagnostics().rank, 2);
        assert_eq!(
            CorrelationFactor::compile(
                indefinite,
                CorrelationToleranceConfig {
                    psd_abs_tol: 1.0_f64.next_down(),
                    ..exact
                },
            ),
            Err(CorrelationError::NotPsd {
                column: 2,
                pivot: -1.0
            })
        );

        let singular = |residual| {
            vec![
                vec![1.0, 1.0, 0.0],
                vec![1.0, 1.0, residual],
                vec![0.0, residual, 1.0],
            ]
        };
        let tolerance = CorrelationToleranceConfig {
            zero_pivot_abs_tol: 0.25,
            ..exact
        };
        let factor = CorrelationFactor::compile(singular(0.25), tolerance).unwrap();
        assert_eq!(factor.diagnostics().maximum_checked_residual, 0.25);
        assert_eq!(factor.diagnostics().zero_pivots, [1]);
        let residual = 0.25_f64.next_up();
        assert_eq!(
            CorrelationFactor::compile(singular(residual), tolerance),
            Err(CorrelationError::SingularResidual {
                row: 2,
                column: 1,
                residual
            })
        );
    }

    #[test]
    fn errors_identify_the_failed_contract() {
        assert_eq!(
            CorrelationError::Shape.to_string(),
            "invalid correlation: Shape"
        );
        let error = CorrelationError::NotPsd {
            column: 2,
            pivot: -1.0,
        };
        assert_eq!(
            error.to_string(),
            "invalid correlation: NotPsd { column: 2, pivot: -1.0 }"
        );
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
