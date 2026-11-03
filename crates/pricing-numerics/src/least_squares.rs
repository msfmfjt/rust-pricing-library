//! Small dense box-constrained least squares with analytic Jacobians.
//!
//! A scaled, damped Gauss--Newton step is projected onto the box. Only strict
//! objective reductions are accepted; callback errors at trial points reject
//! that trial. This is not a global optimizer or a rank/identifiability test.
use std::{error::Error, fmt};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LeastSquaresVariable {
    pub lower: f64,
    pub upper: f64,
    /// Change in the physical variable corresponding to one solver-coordinate unit.
    pub scale: f64,
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LeastSquaresOptions {
    pub max_iterations: usize,
    pub max_evaluations: usize,
    pub residual_tolerance: f64,
    pub gradient_tolerance: f64,
    pub step_tolerance: f64,
    pub initial_damping: f64,
}
impl Default for LeastSquaresOptions {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            max_evaluations: 150,
            residual_tolerance: 1e-6,
            gradient_tolerance: 1e-10,
            step_tolerance: 1e-12,
            initial_damping: 1e-3,
        }
    }
}
impl LeastSquaresOptions {
    pub fn validate(self) -> Result<(), &'static str> {
        if !(1..=10_000).contains(&self.max_iterations)
            || !(1..=10_001).contains(&self.max_evaluations)
        {
            return Err("least-squares iteration/evaluation limits");
        }
        if [
            self.residual_tolerance,
            self.gradient_tolerance,
            self.step_tolerance,
            self.initial_damping,
        ]
        .iter()
        .any(|x| !x.is_finite() || *x <= 0.0)
            || self.initial_damping > 1e12
        {
            return Err("least-squares tolerances/damping must be positive and finite");
        }
        Ok(())
    }
}
#[derive(Clone, Debug, PartialEq)]
pub struct LeastSquaresEvaluation {
    pub residuals: Vec<f64>,
    /// Row-major derivatives of residuals with respect to PHYSICAL parameters.
    pub jacobian: Vec<f64>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LeastSquaresTermination {
    ResidualTolerance,
    ProjectedGradientTolerance,
    StepTolerance,
    MaxIterations,
    MaxEvaluations,
    NoProgress,
}
#[derive(Clone, Debug, PartialEq)]
pub struct LeastSquaresResult {
    pub parameters: Vec<f64>,
    pub evaluation: LeastSquaresEvaluation,
    pub termination: LeastSquaresTermination,
    pub objective: f64,
    pub projected_gradient_norm: f64,
    pub iterations: usize,
    pub evaluations: usize,
    pub rejected_evaluations: usize,
    pub invalid_evaluations: usize,
    /// Initial and strictly improving accepted objectives, not trial values.
    pub accepted_objectives: Vec<f64>,
}
#[derive(Debug)]
pub enum LeastSquaresError<E> {
    InvalidInput(&'static str),
    InitialEvaluation(E),
    InvalidEvaluation(&'static str),
}
impl<E: fmt::Display> fmt::Display for LeastSquaresError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(s) | Self::InvalidEvaluation(s) => write!(f, "{s}"),
            Self::InitialEvaluation(e) => write!(f, "initial least-squares evaluation: {e}"),
        }
    }
}
impl<E: Error + 'static> Error for LeastSquaresError<E> {}

fn checked_cost(
    e: &LeastSquaresEvaluation,
    n: usize,
    rows: Option<usize>,
) -> Result<f64, &'static str> {
    if e.residuals.is_empty()
        || e.residuals.len() > 4096
        || e.jacobian.len() != e.residuals.len() * n
        || rows.is_some_and(|m| m != e.residuals.len())
    {
        return Err("inconsistent least-squares residual/Jacobian shape");
    }
    if e.residuals
        .iter()
        .chain(&e.jacobian)
        .any(|x| !x.is_finite())
    {
        return Err("nonfinite least-squares residual/Jacobian");
    }
    let cost = e.residuals.iter().map(|r| 0.5 * r * r).sum::<f64>();
    if !cost.is_finite() {
        return Err("least-squares objective overflow");
    }
    Ok(cost)
}
fn gradient(e: &LeastSquaresEvaluation, vars: &[LeastSquaresVariable]) -> Vec<f64> {
    (0..vars.len())
        .map(|j| {
            e.residuals
                .iter()
                .enumerate()
                .map(|(i, r)| e.jacobian[i * vars.len() + j] * vars[j].scale * r)
                .sum()
        })
        .collect()
}
fn blocked(x: f64, v: LeastSquaresVariable, g: f64) -> bool {
    (x <= v.lower && g > 0.0) || (x >= v.upper && g < 0.0)
}
fn projected_norm(x: &[f64], vars: &[LeastSquaresVariable], g: &[f64]) -> f64 {
    x.iter()
        .zip(vars)
        .zip(g)
        .map(|((&x, &v), &g)| if blocked(x, v, g) { 0.0 } else { g.abs() })
        .fold(0.0, f64::max)
}
// Positive damping regularizes normal equations. This does not diagnose rank;
// loss of positive definiteness rejects the step instead of returning NaNs.
fn cholesky_solve(mut a: Vec<f64>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for i in 0..n {
        for j in 0..=i {
            let mut x = a[i * n + j];
            for k in 0..j {
                x -= a[i * n + k] * a[j * n + k];
            }
            a[i * n + j] = if i == j {
                if x <= 0.0 || !x.is_finite() {
                    return None;
                }
                x.sqrt()
            } else {
                x / a[j * n + j]
            };
        }
    }
    for i in 0..n {
        for j in 0..i {
            b[i] -= a[i * n + j] * b[j];
        }
        b[i] /= a[i * n + i];
    }
    for i in (0..n).rev() {
        for j in i + 1..n {
            b[i] -= a[j * n + i] * b[j];
        }
        b[i] /= a[i * n + i];
    }
    b.iter().all(|x| x.is_finite()).then_some(b)
}

/// Fit up to 32 variables / 4096 residuals. Trial callback errors mean an invalid
/// model-domain point and are counted/rejected. Wrong shapes or nonfinite data
/// are programming/numerical errors, never a silently accepted missing quote.
/// Only `ResidualTolerance` means all residuals met the requested tolerance.
pub fn bounded_least_squares<E, F>(
    initial: &[f64],
    vars: &[LeastSquaresVariable],
    options: LeastSquaresOptions,
    mut evaluate: F,
) -> Result<LeastSquaresResult, LeastSquaresError<E>>
where
    F: FnMut(&[f64]) -> Result<LeastSquaresEvaluation, E>,
{
    options
        .validate()
        .map_err(LeastSquaresError::InvalidInput)?;
    let n = initial.len();
    if n == 0 || n > 32 || n != vars.len() {
        return Err(LeastSquaresError::InvalidInput(
            "least-squares variable count",
        ));
    }
    for (&x, &v) in initial.iter().zip(vars) {
        if !x.is_finite()
            || !v.lower.is_finite()
            || !v.upper.is_finite()
            || !v.scale.is_finite()
            || v.scale <= 0.0
            || v.lower >= v.upper
            || x < v.lower
            || x > v.upper
            || !((v.upper - v.lower) / v.scale).is_finite()
        {
            return Err(LeastSquaresError::InvalidInput(
                "invalid least-squares variable bounds/scale/start",
            ));
        }
    }
    let mut x = initial.to_vec();
    let mut current = evaluate(&x).map_err(LeastSquaresError::InitialEvaluation)?;
    let mut cost = checked_cost(&current, n, None).map_err(LeastSquaresError::InvalidEvaluation)?;
    let mut history = vec![cost];
    let mut evaluations = 1;
    let mut iterations = 0;
    let mut rejected = 0;
    let mut invalid = 0;
    let mut damping = options.initial_damping;
    let termination = loop {
        if current
            .residuals
            .iter()
            .all(|r| r.abs() <= options.residual_tolerance)
        {
            break LeastSquaresTermination::ResidualTolerance;
        }
        let g = gradient(&current, vars);
        if g.iter().any(|v| !v.is_finite()) {
            return Err(LeastSquaresError::InvalidEvaluation(
                "scaled Jacobian/gradient overflow",
            ));
        }
        if projected_norm(&x, vars, &g) <= options.gradient_tolerance {
            break LeastSquaresTermination::ProjectedGradientTolerance;
        }
        if evaluations >= options.max_evaluations {
            break LeastSquaresTermination::MaxEvaluations;
        }
        if iterations >= options.max_iterations {
            break LeastSquaresTermination::MaxIterations;
        }
        if damping > 1e16 {
            break LeastSquaresTermination::NoProgress;
        }
        iterations += 1;
        let mut a = vec![0.0; n * n];
        let active: Vec<_> = (0..n).map(|j| blocked(x[j], vars[j], g[j])).collect();
        for row in current.jacobian.chunks_exact(n) {
            for i in 0..n {
                for j in 0..=i {
                    if !active[i] && !active[j] {
                        a[i * n + j] += (row[i] * vars[i].scale) * (row[j] * vars[j].scale);
                    }
                }
            }
        }
        if a.iter().any(|v| !v.is_finite()) {
            return Err(LeastSquaresError::InvalidEvaluation(
                "normal-equation overflow",
            ));
        }
        let diagonal_scale = (0..n).map(|i| a[i * n + i]).fold(1.0, f64::max);
        for i in 0..n {
            a[i * n + i] += damping.max(1e-14) * diagonal_scale;
        }
        let rhs = (0..n)
            .map(|i| if active[i] { 0.0 } else { -g[i] })
            .collect();
        let Some(step) = cholesky_solve(a, rhs) else {
            damping *= 10.0;
            continue;
        };
        let trial: Vec<_> = (0..n)
            .map(|j| (x[j] + vars[j].scale * step[j]).clamp(vars[j].lower, vars[j].upper))
            .collect();
        let step_norm = (0..n)
            .map(|j| ((trial[j] - x[j]) / vars[j].scale).abs())
            .fold(0.0, f64::max);
        if step_norm <= options.step_tolerance {
            break LeastSquaresTermination::StepTolerance;
        }
        evaluations += 1;
        match evaluate(&trial) {
            Err(_) => {
                invalid += 1;
                rejected += 1;
                damping *= 10.0;
            }
            Ok(e) => {
                let new_cost = checked_cost(&e, n, Some(current.residuals.len()))
                    .map_err(LeastSquaresError::InvalidEvaluation)?;
                if new_cost < cost {
                    x = trial;
                    current = e;
                    cost = new_cost;
                    history.push(cost);
                    damping = (damping / 3.0).max(1e-14);
                } else {
                    rejected += 1;
                    damping *= 10.0;
                }
            }
        }
    };
    let g = gradient(&current, vars);
    if g.iter().any(|v| !v.is_finite()) {
        return Err(LeastSquaresError::InvalidEvaluation(
            "final gradient overflow",
        ));
    }
    let pg = projected_norm(&x, vars, &g);
    Ok(LeastSquaresResult {
        parameters: x,
        evaluation: current,
        termination,
        objective: cost,
        projected_gradient_norm: pg,
        iterations,
        evaluations,
        rejected_evaluations: rejected,
        invalid_evaluations: invalid,
        accepted_objectives: history,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_zero_gradient_at_a_bound_can_move_inward_through_coupling() {
        for sign in [-1.0, 1.0] {
            let variables = [
                LeastSquaresVariable {
                    lower: if sign > 0.0 { 0.0 } else { -3.0 },
                    upper: if sign > 0.0 { 3.0 } else { 0.0 },
                    scale: 1.0,
                },
                LeastSquaresVariable {
                    lower: -3.0,
                    upper: 3.0,
                    scale: 1.0,
                },
            ];
            // The unconstrained solution is (sign, 0). At (0, 2*sign),
            // gradient[0] is exactly zero, but the coupled Newton step moves
            // that coordinate inward. Freezing it leaves an O(1) residual.
            let result = bounded_least_squares(
                &[0.0, 2.0 * sign],
                &variables,
                LeastSquaresOptions {
                    max_evaluations: 2,
                    ..Default::default()
                },
                |x| {
                    Ok::<_, ()>(LeastSquaresEvaluation {
                        residuals: vec![x[0] + x[1] - sign, x[0] - sign],
                        jacobian: vec![1.0, 1.0, 1.0, 0.0],
                    })
                },
            )
            .unwrap();
            assert_eq!(result.termination, LeastSquaresTermination::MaxEvaluations);
            assert!(result.objective < 1e-4);
            assert!(sign * result.parameters[0] > 0.9);
            assert!(result.parameters[1].abs() < 0.1);
        }
    }

    #[test]
    fn an_outward_active_variable_does_not_couple_into_the_free_step() {
        let variables = [
            LeastSquaresVariable {
                lower: -3.0,
                upper: 1.0,
                scale: 1.0,
            },
            LeastSquaresVariable {
                lower: -10.0,
                upper: 10.0,
                scale: 1.0,
            },
        ];
        // On x <= 1 the constrained optimum is (1, 3), with cost 1/2.
        // One free-coordinate step must approach it without retrying an
        // indefinite matrix that incorrectly retains the active coupling.
        let result = bounded_least_squares(
            &[1.0, 0.0],
            &variables,
            LeastSquaresOptions {
                max_iterations: 1,
                ..Default::default()
            },
            |x| {
                Ok::<_, ()>(LeastSquaresEvaluation {
                    residuals: vec![x[0] + x[1] - 4.0, x[0] - 2.0],
                    jacobian: vec![1.0, 1.0, 1.0, 0.0],
                })
            },
        )
        .unwrap();
        assert_eq!(result.termination, LeastSquaresTermination::MaxIterations);
        assert_eq!(result.parameters[0], 1.0);
        assert!((result.parameters[1] - 3.0).abs() < 0.01);
        assert!(result.objective < 0.501);
        assert_eq!(result.evaluations, 2);
    }

    #[test]
    fn damped_step_matches_an_independent_diagonal_ridge_problem() {
        // J=diag(2,3), r=(2,3), damping=1: lambda=max diag(J'J)=9.
        // Minimizing ||r+J*s||^2 + 9||s||^2 gives s=(-4/13,-1/2),
        // hence x_new=(9/13,1/2). This checks the numerical step itself,
        // independently of later iterations compensating for a wrong solve.
        let result = bounded_least_squares(
            &[1.0, 1.0],
            &vars(2),
            LeastSquaresOptions {
                initial_damping: 1.0,
                max_evaluations: 2,
                ..Default::default()
            },
            |x| {
                Ok::<_, ()>(LeastSquaresEvaluation {
                    residuals: vec![2.0 * x[0], 3.0 * x[1]],
                    jacobian: vec![2.0, 0.0, 0.0, 3.0],
                })
            },
        )
        .unwrap();
        assert_eq!(result.termination, LeastSquaresTermination::MaxEvaluations);
        assert!((result.parameters[0] - 9.0 / 13.0).abs() < 5e-16);
        assert!((result.parameters[1] - 0.5).abs() < 5e-16);
    }

    #[test]
    fn failed_factorizations_exhaust_damping_without_model_trials() {
        let mut calls = 0;
        // The finite Gram diagonal is 1e300. Multiplying it by the allowed
        // initial damping overflows; each factorization must fail safely.
        let result = bounded_least_squares(
            &[0.0],
            &vars(1),
            LeastSquaresOptions {
                initial_damping: 1e12,
                ..Default::default()
            },
            |_| {
                calls += 1;
                Ok::<_, ()>(LeastSquaresEvaluation {
                    residuals: vec![1.0],
                    jacobian: vec![1e150],
                })
            },
        )
        .unwrap();
        assert_eq!(result.termination, LeastSquaresTermination::NoProgress);
        assert_eq!(result.parameters, [0.0]);
        assert_eq!(result.objective, 0.5);
        assert_eq!(result.iterations, 5);
        assert_eq!(result.evaluations, 1);
        assert_eq!(calls, 1);
        assert_eq!(result.rejected_evaluations, 0);
        assert_eq!(result.invalid_evaluations, 0);
    }
    fn vars(n: usize) -> Vec<LeastSquaresVariable> {
        vec![
            LeastSquaresVariable {
                lower: -3.0,
                upper: 3.0,
                scale: 1.0
            };
            n
        ]
    }

    #[test]
    fn option_domains_and_inclusive_limits_are_validated_independently() {
        for (max_iterations, max_evaluations) in [(0, 150), (10_001, 150), (100, 0), (100, 10_002)]
        {
            assert!(
                LeastSquaresOptions {
                    max_iterations,
                    max_evaluations,
                    ..Default::default()
                }
                .validate()
                .is_err()
            );
        }
        for (max_iterations, max_evaluations) in [(1, 1), (10_000, 10_001)] {
            assert!(
                LeastSquaresOptions {
                    max_iterations,
                    max_evaluations,
                    ..Default::default()
                }
                .validate()
                .is_ok()
            );
        }
        for index in 0..4 {
            for invalid in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                let mut options = LeastSquaresOptions::default();
                let fields = [
                    &mut options.residual_tolerance,
                    &mut options.gradient_tolerance,
                    &mut options.step_tolerance,
                    &mut options.initial_damping,
                ];
                *fields[index] = invalid;
                assert!(options.validate().is_err(), "option {index}: {invalid}");
            }
        }
        assert!(
            LeastSquaresOptions {
                initial_damping: 1e12,
                ..Default::default()
            }
            .validate()
            .is_ok()
        );
        assert!(
            LeastSquaresOptions {
                initial_damping: 1e12_f64.next_up(),
                ..Default::default()
            }
            .validate()
            .is_err()
        );
    }

    #[test]
    fn invalid_variables_fail_before_evaluating_the_model() {
        let valid = vars(1)[0];
        let mut cases = vec![
            (-3.0_f64.next_up(), valid),
            (3.0_f64.next_up(), valid),
            (
                0.0,
                LeastSquaresVariable {
                    lower: 1.0,
                    upper: 1.0,
                    ..valid
                },
            ),
            (
                0.0,
                LeastSquaresVariable {
                    lower: 2.0,
                    upper: 1.0,
                    ..valid
                },
            ),
            (
                0.0,
                LeastSquaresVariable {
                    scale: 0.0,
                    ..valid
                },
            ),
            (
                0.0,
                LeastSquaresVariable {
                    scale: -1.0,
                    ..valid
                },
            ),
            (
                0.0,
                LeastSquaresVariable {
                    scale: f64::MIN_POSITIVE,
                    ..valid
                },
            ),
            (
                0.0,
                LeastSquaresVariable {
                    lower: -f64::MAX,
                    upper: f64::MAX,
                    ..valid
                },
            ),
        ];
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            cases.push((invalid, valid));
            cases.push((
                0.0,
                LeastSquaresVariable {
                    lower: invalid,
                    ..valid
                },
            ));
            cases.push((
                0.0,
                LeastSquaresVariable {
                    upper: invalid,
                    ..valid
                },
            ));
            cases.push((
                0.0,
                LeastSquaresVariable {
                    scale: invalid,
                    ..valid
                },
            ));
        }
        for (initial, variable) in cases {
            let result = bounded_least_squares(
                &[initial],
                &[variable],
                LeastSquaresOptions::default(),
                |_| -> Result<LeastSquaresEvaluation, ()> {
                    panic!("invalid input reached the model: {initial}, {variable:?}")
                },
            );
            assert!(matches!(result, Err(LeastSquaresError::InvalidInput(_))));
        }
        for (initial, variables) in [
            (vec![], vars(0)),
            (vec![0.0; 33], vars(33)),
            (vec![0.0], vars(2)),
        ] {
            let result = bounded_least_squares(
                &initial,
                &variables,
                LeastSquaresOptions::default(),
                |_| -> Result<LeastSquaresEvaluation, ()> {
                    panic!("invalid variable count reached the model")
                },
            );
            assert!(matches!(result, Err(LeastSquaresError::InvalidInput(_))));
        }
    }

    #[test]
    fn inclusive_input_limits_and_finite_scaled_widths_are_accepted() {
        for (initial, variables) in [
            (vec![-3.0], vars(1)),
            (vec![3.0], vars(1)),
            (vec![0.0; 32], vars(32)),
            (
                vec![0.5],
                vec![LeastSquaresVariable {
                    lower: 0.0,
                    upper: 1.0,
                    scale: 1.0,
                }],
            ),
            (
                vec![0.0],
                vec![LeastSquaresVariable {
                    scale: f64::MAX,
                    ..vars(1)[0]
                }],
            ),
            (
                vec![1e308],
                vec![LeastSquaresVariable {
                    lower: 1e308,
                    upper: 1.1e308,
                    scale: 1e308,
                }],
            ),
        ] {
            let n = initial.len();
            let result =
                bounded_least_squares(&initial, &variables, LeastSquaresOptions::default(), |_| {
                    Ok::<_, ()>(LeastSquaresEvaluation {
                        residuals: vec![0.0; 4096],
                        jacobian: vec![0.0; 4096 * n],
                    })
                })
                .unwrap();
            assert_eq!(
                result.termination,
                LeastSquaresTermination::ResidualTolerance
            );
            assert_eq!(result.parameters, initial);
            assert_eq!(result.evaluations, 1);
            assert_eq!(result.iterations, 0);
        }
    }

    #[test]
    fn evaluation_errors_preserve_their_category() {
        for evaluation in [
            LeastSquaresEvaluation {
                residuals: vec![],
                jacobian: vec![],
            },
            LeastSquaresEvaluation {
                residuals: vec![0.0; 4097],
                jacobian: vec![0.0; 4097],
            },
            LeastSquaresEvaluation {
                residuals: vec![1.0],
                jacobian: vec![],
            },
            LeastSquaresEvaluation {
                residuals: vec![f64::NAN],
                jacobian: vec![1.0],
            },
            LeastSquaresEvaluation {
                residuals: vec![1.0],
                jacobian: vec![f64::INFINITY],
            },
            LeastSquaresEvaluation {
                residuals: vec![f64::MAX],
                jacobian: vec![1.0],
            },
            LeastSquaresEvaluation {
                residuals: vec![2.0],
                jacobian: vec![f64::MAX],
            },
            LeastSquaresEvaluation {
                residuals: vec![1.0],
                jacobian: vec![1e200],
            },
        ] {
            let result =
                bounded_least_squares(&[0.0], &vars(1), LeastSquaresOptions::default(), |_| {
                    Ok::<_, ()>(evaluation.clone())
                });
            assert!(matches!(
                result,
                Err(LeastSquaresError::InvalidEvaluation(_))
            ));
        }
        let result =
            bounded_least_squares(&[0.0], &vars(1), LeastSquaresOptions::default(), |_| {
                Err("model domain")
            });
        assert!(matches!(
            &result,
            Err(LeastSquaresError::InitialEvaluation("model domain"))
        ));
        assert_eq!(
            result.unwrap_err().to_string(),
            "initial least-squares evaluation: model domain"
        );
        for error in [
            LeastSquaresError::<&str>::InvalidInput("bad bounds"),
            LeastSquaresError::InvalidEvaluation("bad residual"),
        ] {
            assert!(!error.to_string().is_empty());
        }
        let mut calls = 0;
        let result =
            bounded_least_squares(&[0.0], &vars(1), LeastSquaresOptions::default(), |_| {
                calls += 1;
                let rows = if calls == 1 { 1 } else { 2 };
                Ok::<_, ()>(LeastSquaresEvaluation {
                    residuals: vec![1.0; rows],
                    jacobian: vec![1.0; rows],
                })
            });
        assert_eq!(calls, 2);
        assert!(matches!(
            result,
            Err(LeastSquaresError::InvalidEvaluation(_))
        ));
    }

    #[test]
    fn solves_a_coupled_three_variable_problem_with_known_solution() {
        // Four independent linear observations of (1, -1, 2).
        let mut variables = vars(3);
        variables[1].scale = 2.0;
        variables[2].scale = 0.5;
        let mut calls = 0;
        let result =
            bounded_least_squares(&[0.0; 3], &variables, LeastSquaresOptions::default(), |x| {
                calls += 1;
                Ok::<_, ()>(LeastSquaresEvaluation {
                    residuals: vec![
                        2.0 * x[0] + x[1] + x[2] - 3.0,
                        x[0] + 3.0 * x[1] - x[2] + 4.0,
                        x[0] + 2.0 * x[2] - 5.0,
                        2.0 * x[0] - x[1] + x[2] - 5.0,
                    ],
                    jacobian: vec![2.0, 1.0, 1.0, 1.0, 3.0, -1.0, 1.0, 0.0, 2.0, 2.0, -1.0, 1.0],
                })
            })
            .unwrap();
        assert_eq!(
            result.termination,
            LeastSquaresTermination::ResidualTolerance
        );
        for (actual, expected) in result.parameters.iter().zip([1.0, -1.0, 2.0]) {
            assert!((actual - expected).abs() < 1e-6);
        }
        assert!(result.evaluation.residuals.iter().all(|r| r.abs() <= 1e-6));
        assert_eq!(result.accepted_objectives[0], 37.5);
        assert!(result.accepted_objectives.windows(2).all(|w| w[1] < w[0]));
        assert_eq!(result.evaluations, calls);
        assert_eq!(result.evaluations, result.iterations + 1);
        assert_eq!(result.rejected_evaluations, 0);
        assert_eq!(result.invalid_evaluations, 0);
    }

    #[test]
    fn linear_solve_matches_an_independent_system_and_rejects_invalid_pivots() {
        // A * [1, 2, 3] = [-20, -43, 192]. Checking the solve directly catches
        // wrong steps that an outer optimizer can compensate for later.
        let solution = cholesky_solve(
            vec![4.0, 12.0, -16.0, 12.0, 37.0, -43.0, -16.0, -43.0, 98.0],
            vec![-20.0, -43.0, 192.0],
        )
        .unwrap();
        for (actual, expected) in solution.iter().zip([1.0, 2.0, 3.0]) {
            assert!((actual - expected).abs() < 1e-13);
        }
        for pivot in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(cholesky_solve(vec![pivot], vec![1.0]), None);
        }
        assert_eq!(
            cholesky_solve(vec![1.0, 2.0, 2.0, 1.0], vec![1.0, 1.0]),
            None
        );
    }

    #[test]
    fn persistently_rejected_trials_terminate_without_claiming_a_fit() {
        for domain_errors in [false, true] {
            let mut calls = 0;
            let result = bounded_least_squares(
                &[0.0],
                &vars(1),
                LeastSquaresOptions {
                    initial_damping: 1e12,
                    step_tolerance: 1e-30,
                    ..Default::default()
                },
                |_| {
                    calls += 1;
                    if domain_errors && calls > 1 {
                        return Err(());
                    }
                    Ok(LeastSquaresEvaluation {
                        residuals: vec![1.0],
                        jacobian: vec![1.0],
                    })
                },
            )
            .unwrap();
            assert_eq!(result.termination, LeastSquaresTermination::NoProgress);
            assert_eq!(result.parameters, [0.0]);
            assert_eq!(result.accepted_objectives, [0.5]);
            // The final allowed damping is 1e16; five failed trials starting
            // from 1e12 exhaust that range, independently of the model domain.
            assert_eq!(result.iterations, 5);
            assert_eq!(result.evaluations, 6);
            assert_eq!(calls, 6);
            assert_eq!(result.rejected_evaluations, 5);
            assert_eq!(
                result.invalid_evaluations,
                if domain_errors { 5 } else { 0 }
            );
        }
    }

    #[test]
    fn iteration_and_evaluation_budgets_stop_after_the_promised_work() {
        for (max_iterations, max_evaluations, termination) in [
            (1, 5, LeastSquaresTermination::MaxIterations),
            (5, 2, LeastSquaresTermination::MaxEvaluations),
        ] {
            let mut calls = 0;
            let result = bounded_least_squares(
                &[0.0],
                &vars(1),
                LeastSquaresOptions {
                    max_iterations,
                    max_evaluations,
                    ..Default::default()
                },
                |x| {
                    calls += 1;
                    Ok::<_, ()>(LeastSquaresEvaluation {
                        residuals: vec![x[0] - 1.0],
                        jacobian: vec![1.0],
                    })
                },
            )
            .unwrap();
            assert_eq!(result.termination, termination);
            assert_eq!(result.iterations, 1);
            assert_eq!(result.evaluations, 2);
            assert_eq!(calls, 2);
        }
    }

    #[test]
    fn equal_cost_and_invalid_trials_are_rejected_and_counted() {
        for domain_errors in [false, true] {
            let mut calls = 0;
            let result = bounded_least_squares(
                &[0.0],
                &vars(1),
                LeastSquaresOptions {
                    max_evaluations: 4,
                    ..Default::default()
                },
                |_| {
                    calls += 1;
                    if domain_errors && calls > 1 {
                        return Err(());
                    }
                    Ok(LeastSquaresEvaluation {
                        residuals: vec![1.0],
                        jacobian: vec![1.0],
                    })
                },
            )
            .unwrap();
            assert_eq!(result.termination, LeastSquaresTermination::MaxEvaluations);
            assert_eq!(result.parameters, [0.0]);
            assert_eq!(result.objective, 0.5);
            assert_eq!(result.accepted_objectives, [0.5]);
            assert_eq!(result.iterations, 3);
            assert_eq!(result.evaluations, 4);
            assert_eq!(calls, 4);
            assert_eq!(result.rejected_evaluations, 3);
            assert_eq!(
                result.invalid_evaluations,
                if domain_errors { 3 } else { 0 }
            );
        }
    }

    #[test]
    fn lower_bound_stationarity_and_scaled_step_tolerance_are_distinct() {
        let variable = LeastSquaresVariable {
            lower: -1.0,
            ..vars(1)[0]
        };
        let result =
            bounded_least_squares(&[0.0], &[variable], LeastSquaresOptions::default(), |x| {
                Ok::<_, ()>(LeastSquaresEvaluation {
                    residuals: vec![x[0] + 2.0],
                    jacobian: vec![1.0],
                })
            })
            .unwrap();
        assert_eq!(
            result.termination,
            LeastSquaresTermination::ProjectedGradientTolerance
        );
        assert_eq!(result.parameters, [-1.0]);
        assert_eq!(result.objective, 0.5);
        assert_eq!(result.projected_gradient_norm, 0.0);

        for (step_tolerance, termination, evaluations, parameter) in [
            (0.25, LeastSquaresTermination::StepTolerance, 1, 1.0),
            (
                0.25_f64.next_down(),
                LeastSquaresTermination::MaxEvaluations,
                2,
                1.5,
            ),
        ] {
            let result = bounded_least_squares(
                &[1.0],
                &[LeastSquaresVariable {
                    scale: 2.0,
                    ..variable
                }],
                LeastSquaresOptions {
                    initial_damping: 1.0,
                    step_tolerance,
                    max_evaluations: 2,
                    ..Default::default()
                },
                |x| {
                    Ok::<_, ()>(LeastSquaresEvaluation {
                        residuals: vec![x[0] - 2.0],
                        jacobian: vec![1.0],
                    })
                },
            )
            .unwrap();
            assert_eq!(result.termination, termination);
            assert_eq!(result.evaluations, evaluations);
            assert_eq!(result.parameters, [parameter]);
        }
    }
    #[test]
    fn rosenbrock_and_scaled_coordinates() {
        let run = |scale: f64| {
            let mut v = vars(2);
            v[0] = LeastSquaresVariable {
                lower: -3.0 * scale,
                upper: 3.0 * scale,
                scale,
            };
            bounded_least_squares(
                &[-1.2 * scale, 1.0],
                &v,
                LeastSquaresOptions::default(),
                |x| {
                    let a = x[0] / scale;
                    Ok::<_, ()>(LeastSquaresEvaluation {
                        residuals: vec![10.0 * (x[1] - a * a), 1.0 - a],
                        jacobian: vec![-20.0 * a / scale, 10.0, -1.0 / scale, 0.0],
                    })
                },
            )
            .unwrap()
        };
        for s in [1.0, 1e-6, 1e6] {
            let r = run(s);
            assert_eq!(r.termination, LeastSquaresTermination::ResidualTolerance);
            assert!((r.parameters[0] / s - 1.0).abs() < 1e-5);
            assert!(r.accepted_objectives.windows(2).all(|w| w[1] < w[0]));
        }
    }
    #[test]
    fn bound_stationarity_is_not_a_fit_and_budget_is_honest() {
        let mut v = vars(1);
        v[0].upper = 1.0;
        let f = |x: &[f64]| {
            Ok::<_, ()>(LeastSquaresEvaluation {
                residuals: vec![x[0] - 2.0],
                jacobian: vec![1.0],
            })
        };
        let r = bounded_least_squares(&[0.0], &v, LeastSquaresOptions::default(), f).unwrap();
        assert_eq!(r.parameters, [1.0]);
        assert_eq!(
            r.termination,
            LeastSquaresTermination::ProjectedGradientTolerance
        );
        assert_eq!(r.objective, 0.5);
        let r = bounded_least_squares(
            &[0.0],
            &v,
            LeastSquaresOptions {
                max_evaluations: 1,
                ..Default::default()
            },
            f,
        )
        .unwrap();
        assert_eq!(r.termination, LeastSquaresTermination::MaxEvaluations);
        assert_eq!(r.evaluations, 1);
    }
    #[test]
    fn deficient_jacobian_and_trial_domain_errors() {
        let r = bounded_least_squares(&[0.0, 0.0], &vars(2), LeastSquaresOptions::default(), |x| {
            Ok::<_, ()>(LeastSquaresEvaluation {
                residuals: vec![x[0] + x[1] - 1.0],
                jacobian: vec![1.0, 1.0],
            })
        })
        .unwrap();
        assert_eq!(r.termination, LeastSquaresTermination::ResidualTolerance);
        let r = bounded_least_squares(&[0.1], &vars(1), LeastSquaresOptions::default(), |x| {
            if x[0] > 0.8 {
                return Err(());
            }
            Ok(LeastSquaresEvaluation {
                residuals: vec![x[0] * x[0] - 0.25],
                jacobian: vec![2.0 * x[0]],
            })
        })
        .unwrap();
        assert!(r.invalid_evaluations > 0);
        assert!((r.parameters[0] - 0.5).abs() < 1e-5);
    }
    #[test]
    fn malformed_inputs_and_callback_shape_are_errors() {
        let f = |_: &[f64]| {
            Ok::<_, ()>(LeastSquaresEvaluation {
                residuals: vec![1.0],
                jacobian: vec![],
            })
        };
        assert!(
            bounded_least_squares(&[0.0], &vars(1), LeastSquaresOptions::default(), f).is_err()
        );
        assert!(
            bounded_least_squares(&[f64::NAN], &vars(1), LeastSquaresOptions::default(), f)
                .is_err()
        );
        assert!(
            bounded_least_squares(
                &[0.0],
                &vars(1),
                LeastSquaresOptions {
                    residual_tolerance: f64::NAN,
                    ..Default::default()
                },
                f
            )
            .is_err()
        );
    }
}
