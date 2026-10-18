//! Certified linear growth envelope for continuous piecewise-affine payoffs.
use super::invalid;
use crate::core::{Date, UnderlyingId};
use crate::product::{CompiledOpcode as Op, CompiledPayoff};
use crate::risk::RiskConfigError;
use std::collections::BTreeMap;

#[derive(Clone, Copy)]
struct Bound {
    coefficients: [f64; 17],
    constant: Option<f64>,
}
impl Bound {
    fn literal(c: f64) -> Self {
        Self {
            coefficients: {
                let mut a = [0.; 17];
                a[0] = c.abs();
                a
            },
            constant: Some(c),
        }
    }
    fn scaled(self, c: f64) -> Self {
        Self {
            coefficients: self.coefficients.map(|x| x * c.abs()),
            constant: self.constant.map(|x| x * c),
        }
    }
}
pub(super) fn growth_bound(
    payoff: &CompiledPayoff,
    mut observation: impl FnMut(UnderlyingId, Date) -> Option<usize>,
    spot: f64,
) -> Result<[f64; 17], RiskConfigError> {
    if payoff.output_slots().len() != 1 {
        return Err(invalid("single output required for path Gamma chain"));
    }
    let mut slots = BTreeMap::<u32, Bound>::new();
    for &op in payoff.opcodes() {
        let get = |i: u32| {
            slots
                .get(&i)
                .copied()
                .ok_or_else(|| invalid("missing growth-bound operand"))
        };
        let (out, b) = match op {
            Op::Literal { value, output } => (output, Bound::literal(value)),
            Op::TerminalSpot {
                underlying,
                observation_date,
                output,
            } => {
                let i = observation(underlying, observation_date)
                    .ok_or_else(|| invalid("known simulation observation required"))?;
                let b = if i == 0 {
                    Bound::literal(spot)
                } else if i <= 16 {
                    let mut coefficients = [0.; 17];
                    coefficients[i] = 1.;
                    Bound {
                        coefficients,
                        constant: None,
                    }
                } else {
                    return Err(invalid("at most sixteen observation intervals required"));
                };
                (output, b)
            }
            Op::Add {
                left,
                right,
                output,
            }
            | Op::Subtract {
                left,
                right,
                output,
            }
            | Op::Minimum {
                left,
                right,
                output,
            }
            | Op::Maximum {
                left,
                right,
                output,
            } => {
                let a = get(left)?;
                let b = get(right)?;
                let operation = |x: f64, y: f64| match op {
                    Op::Add { .. } => x + y,
                    Op::Subtract { .. } => x - y,
                    Op::Minimum { .. } => x.min(y),
                    _ => x.max(y),
                };
                let c = match (a.constant, b.constant) {
                    (Some(a), Some(b)) => Some(operation(a, b)),
                    _ => None,
                };
                let coeff = std::array::from_fn(|i| {
                    if matches!(op, Op::Minimum { .. } | Op::Maximum { .. }) {
                        a.coefficients[i].max(b.coefficients[i])
                    } else {
                        a.coefficients[i] + b.coefficients[i]
                    }
                });
                (
                    output,
                    c.map(Bound::literal).unwrap_or(Bound {
                        coefficients: coeff,
                        constant: None,
                    }),
                )
            }
            Op::Multiply {
                left,
                right,
                output,
            } => {
                let a = get(left)?;
                let b = get(right)?;
                let value = if let Some(c) = a.constant {
                    b.scaled(c)
                } else if let Some(c) = b.constant {
                    a.scaled(c)
                } else {
                    return Err(invalid("global piecewise-affine growth envelope required"));
                };
                (output, value)
            }
            Op::Divide {
                numerator,
                denominator,
                output,
            } => {
                let c = get(denominator)?
                    .constant
                    .filter(|c| *c != 0.)
                    .ok_or_else(|| invalid("constant nonzero divisor required"))?;
                (output, get(numerator)?.scaled(1. / c))
            }
            Op::Negate { input, output } => (output, get(input)?.scaled(-1.)),
            _ => {
                return Err(invalid(
                    "continuous piecewise-affine growth envelope required",
                ));
            }
        };
        if b.coefficients.iter().any(|x| !x.is_finite())
            || b.constant.is_some_and(|v| !v.is_finite())
        {
            return Err(invalid("nonfinite payoff growth bound"));
        }
        slots.insert(out, b);
    }
    slots
        .get(&payoff.output_slots()[0])
        .map(|b| b.coefficients)
        .ok_or_else(|| invalid("missing growth-bound output"))
}
