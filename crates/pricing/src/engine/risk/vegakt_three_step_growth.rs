//! Certified linear growth envelope for continuous piecewise-affine payoffs.
use crate::core::{Date, UnderlyingId};
use crate::product::{CompiledOpcode as Op, CompiledPayoff};
use crate::risk::RiskConfigError;
use std::collections::BTreeMap;

fn invalid(s: &'static str) -> RiskConfigError {
    RiskConfigError::VegaKtChainInput { quantity: s }
}

#[derive(Clone, Copy)]
struct Bound {
    coefficients: [f64; 4],
    constant: Option<f64>,
}
impl Bound {
    fn literal(c: f64) -> Self {
        Self {
            coefficients: [c.abs(), 0., 0., 0.],
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
) -> Result<[f64; 4], RiskConfigError> {
    if payoff.output_slots().len() != 1 {
        return Err(invalid("single output required for middle smile Gamma"));
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
                    .ok_or_else(|| invalid("three-step observation required"))?;
                let b = match i {
                    0 => Bound::literal(spot),
                    1 => Bound {
                        coefficients: [0., 1., 0., 0.],
                        constant: None,
                    },
                    2 => Bound {
                        coefficients: [0., 0., 1., 0.],
                        constant: None,
                    },
                    3 => Bound {
                        coefficients: [0., 0., 0., 1.],
                        constant: None,
                    },
                    _ => return Err(invalid("three-step observation required")),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::product::{GraphLimitPolicy, SourceGraphBuilder, SourceOpcode};

    fn spot(builder: &mut SourceGraphBuilder, underlying: u32) -> crate::core::NodeId {
        builder
            .push(SourceOpcode::TerminalSpot {
                underlying: UnderlyingId::new(underlying),
                observation_date: "2027-01-01".parse().unwrap(),
            })
            .unwrap()
    }

    #[test]
    fn initial_spot_and_three_future_observations_keep_distinct_coefficients() {
        let mut builder = SourceGraphBuilder::new();
        let s = spot(&mut builder, 1);
        let payoff = builder
            .finish(vec![s])
            .compile(GraphLimitPolicy::DEFAULT)
            .unwrap();
        for i in 0..4 {
            let mut expected = [0.; 4];
            expected[i] = if i == 0 { 100. } else { 1. };
            assert_eq!(
                growth_bound(&payoff, |_, _| Some(i), 100.).unwrap(),
                expected
            );
        }
        for index in [None, Some(4)] {
            assert!(matches!(
                growth_bound(&payoff, |_, _| index, 100.),
                Err(RiskConfigError::VegaKtChainInput {
                    quantity: "three-step observation required"
                })
            ));
        }
    }

    #[test]
    fn signed_piecewise_affine_payoff_keeps_certified_envelope() {
        let mut b = SourceGraphBuilder::new();
        let s1 = spot(&mut b, 1);
        let s2 = spot(&mut b, 2);
        let s3 = spot(&mut b, 3);
        let minus_two = b.literal(-2.).unwrap();
        let three = b.literal(3.).unwrap();
        let two = b.literal(2.).unwrap();
        let seven = b.literal(7.).unwrap();
        let left = b
            .push(SourceOpcode::Multiply {
                left: minus_two,
                right: s1,
            })
            .unwrap();
        let right = b
            .push(SourceOpcode::Multiply {
                left: three,
                right: s2,
            })
            .unwrap();
        let sum = b.push(SourceOpcode::Add { left, right }).unwrap();
        let average = b
            .push(SourceOpcode::Divide {
                numerator: sum,
                denominator: two,
            })
            .unwrap();
        let shifted = b
            .push(SourceOpcode::Subtract {
                left: s3,
                right: seven,
            })
            .unwrap();
        let result = b
            .push(SourceOpcode::Maximum {
                left: average,
                right: shifted,
            })
            .unwrap();
        let payoff = b
            .finish(vec![result])
            .compile(GraphLimitPolicy::DEFAULT)
            .unwrap();
        // max((-2*S1 + 3*S2)/2, S3 - 7) is bounded by 7 + S1 + 1.5*S2 + S3.
        assert_eq!(
            growth_bound(&payoff, |u, _| Some(u.get() as usize), 100.).unwrap(),
            [7., 1., 1.5, 1.]
        );
    }

    #[test]
    fn nonlinear_product_and_variable_divisor_remain_rejected() {
        for divide in [false, true] {
            let mut b = SourceGraphBuilder::new();
            let left = spot(&mut b, 1);
            let right = spot(&mut b, 2);
            let (op, message) = if divide {
                (
                    SourceOpcode::Divide {
                        numerator: left,
                        denominator: right,
                    },
                    "constant nonzero divisor required",
                )
            } else {
                (
                    SourceOpcode::Multiply { left, right },
                    "global piecewise-affine growth envelope required",
                )
            };
            let out = b.push(op).unwrap();
            let payoff = b
                .finish(vec![out])
                .compile(GraphLimitPolicy::DEFAULT)
                .unwrap();
            assert!(matches!(
                growth_bound(&payoff, |u, _| Some(u.get() as usize), 100.),
                Err(RiskConfigError::VegaKtChainInput { quantity }) if quantity == message
            ));
        }
    }
}
