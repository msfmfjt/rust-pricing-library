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
