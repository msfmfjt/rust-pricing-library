//! Compile continuous piecewise-affine terminal payoffs to signed Gamma atoms.
use super::vegakt_measure::invalid;
use crate::core::{Date, FiniteF64, UnderlyingId};
use crate::product::{CompiledOpcode, CompiledPayoff};
use crate::risk::{RiskConfigError, VegaKtGammaAtoms, VegaKtTerminalPayoff};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq)]
struct Line {
    cash: f64,
    slope: f64,
}
#[derive(Clone)]
struct Piecewise {
    knots: Vec<f64>,
    lines: Vec<Line>,
}
impl Piecewise {
    fn affine(cash: f64, slope: f64) -> Self {
        Self {
            knots: vec![],
            lines: vec![Line { cash, slope }],
        }
    }
    fn constant(&self) -> Option<f64> {
        (self.knots.is_empty() && self.lines[0].slope == 0.).then_some(self.lines[0].cash)
    }
    fn at(&self, left: f64) -> Line {
        self.lines[self.knots.partition_point(|&k| k <= left)]
    }
    fn append(&mut self, left: f64, line: Line) -> Result<(), RiskConfigError> {
        FiniteF64::new(line.cash, "terminal_affine_cash")?;
        FiniteF64::new(line.slope, "terminal_affine_slope")?;
        if self.lines.last() != Some(&line) {
            if !self.lines.is_empty() {
                self.knots.push(left);
            }
            self.lines.push(line);
        }
        if self.knots.len() > 4096 {
            return Err(invalid("terminal payoff exceeds 4096-knot budget"));
        }
        Ok(())
    }
    fn scaled(&self, scale: f64) -> Result<Self, RiskConfigError> {
        FiniteF64::new(scale, "terminal_affine_scale")?;
        let mut out = Self {
            knots: vec![],
            lines: vec![],
        };
        for (left, l) in std::iter::once(0.)
            .chain(self.knots.iter().copied())
            .zip(&self.lines)
        {
            out.append(
                left,
                Line {
                    cash: l.cash * scale,
                    slope: l.slope * scale,
                },
            )?;
        }
        Ok(out)
    }
    // mode: 0 add, 1 subtract, 2 minimum, 3 maximum.
    fn combine(&self, rhs: &Self, mode: u8) -> Result<Self, RiskConfigError> {
        let mut knots = self
            .knots
            .iter()
            .chain(&rhs.knots)
            .copied()
            .collect::<Vec<_>>();
        knots.push(0.);
        knots.sort_by(f64::total_cmp);
        knots.dedup();
        knots.push(f64::INFINITY);
        let mut out = Self {
            knots: vec![],
            lines: vec![],
        };
        for pair in knots.windows(2) {
            let [lo, hi] = [pair[0], pair[1]];
            let a = self.at(lo);
            let b = rhs.at(lo);
            if mode < 2 {
                let sign = if mode == 0 { 1. } else { -1. };
                out.append(
                    lo,
                    Line {
                        cash: a.cash + sign * b.cash,
                        slope: a.slope + sign * b.slope,
                    },
                )?;
                continue;
            }
            let dc = FiniteF64::new(a.cash - b.cash, "terminal_intersection_cash")?.get();
            let ds = FiniteF64::new(a.slope - b.slope, "terminal_intersection_slope")?.get();
            let pick = |positive: bool| if positive == (mode == 3) { a } else { b };
            if ds == 0. {
                out.append(lo, pick(dc >= 0.))?;
                continue;
            }
            let root = -dc / ds;
            if root == f64::INFINITY {
                return Err(invalid("unrepresentable positive terminal payoff kink"));
            }
            if root > lo && root < hi {
                out.append(lo, pick(ds < 0.))?;
                out.append(root, pick(ds > 0.))?;
            } else {
                out.append(lo, pick(if root <= lo { ds > 0. } else { ds < 0. }))?;
            }
        }
        Ok(out)
    }

    fn into_payoff(self) -> Result<VegaKtTerminalPayoff, RiskConfigError> {
        let first = self.lines[0];
        let mut atoms = Vec::new();
        for (i, &k) in self.knots.iter().enumerate() {
            let a = self.lines[i];
            let b = self.lines[i + 1];
            let jump = FiniteF64::new(b.slope - a.slope, "terminal_gamma_jump")?.get();
            // A jump in value needs a derivative of a Dirac, not a Gamma mass.
            // Reject any non-roundoff discontinuity in this representation.
            let lhs = (b.cash - a.cash) + jump * k;
            let scale = a.cash.abs() + b.cash.abs() + (jump * k).abs();
            if !lhs.is_finite() || !scale.is_finite() || lhs.abs() > 64. * f64::EPSILON * scale {
                return Err(invalid("terminal payoff continuity could not be certified"));
            }
            atoms.push((k, jump));
        }
        Ok(VegaKtTerminalPayoff {
            cash: first.cash,
            slope: first.slope,
            gamma: VegaKtGammaAtoms::new(atoms)?,
        })
    }
}

/// Derives Gamma from the compiled expression, not from the product name or an
/// analytic Vega oracle. Supports affine arithmetic, min/max and multiplication
/// or division by a constant, with one expected terminal observation. Indicators,
/// smoothing, multiple observation dates and spot-dependent products/divisors fail.
pub fn terminal_payoff_gamma_measure(
    payoff: &CompiledPayoff,
    underlying: UnderlyingId,
    expiry: Date,
) -> Result<VegaKtTerminalPayoff, RiskConfigError> {
    try_terminal_payoff_gamma_measure(payoff, underlying, expiry)?
        .ok_or_else(|| invalid("continuous piecewise-affine single-terminal payoff required"))
}

/// None denotes unsupported mathematical scope only. Numerical failures must
/// propagate, never silently select the provisional node-allocation method.
pub(in crate::engine) fn try_terminal_payoff_gamma_measure(
    payoff: &CompiledPayoff,
    underlying: UnderlyingId,
    expiry: Date,
) -> Result<Option<VegaKtTerminalPayoff>, RiskConfigError> {
    try_affine_observation_payoff_gamma_measure(payoff, |u, date| {
        (u == underlying && date == expiry).then_some((0., 1.))
    })
}

/// Conditional continuation: each observation is cash + slope * first state.
/// Constants include known history; future observations retain their own dates.
/// Only the resolver changes; the signed, continuous payoff algebra is shared.
pub(super) fn try_affine_observation_payoff_gamma_measure(
    payoff: &CompiledPayoff,
    observation: impl FnMut(UnderlyingId, Date) -> Option<(f64, f64)>,
) -> Result<Option<VegaKtTerminalPayoff>, RiskConfigError> {
    compile_affine_observation_payoff(payoff, observation, None)
}

/// Retain intermediate branch boundaries even when output slope jumps cancel.
/// They are needed to integrate observation AD, not only the payoff value.
pub(super) fn conditional_payoff_branch_boundaries(
    payoff: &CompiledPayoff,
    observation: impl FnMut(UnderlyingId, Date) -> Option<(f64, f64)>,
) -> Result<Vec<f64>, RiskConfigError> {
    let mut boundaries = Vec::new();
    if compile_affine_observation_payoff(payoff, observation, Some(&mut boundaries))?.is_none() {
        return Err(invalid("piecewise affine conditional AD payoff required"));
    }
    boundaries.sort_by(f64::total_cmp);
    boundaries.dedup();
    if boundaries.len() > 256 {
        return Err(invalid("conditional AD exceeds 256-branch-boundary budget"));
    }
    Ok(boundaries)
}

fn compile_affine_observation_payoff(
    payoff: &CompiledPayoff,
    mut observation: impl FnMut(UnderlyingId, Date) -> Option<(f64, f64)>,
    mut boundaries: Option<&mut Vec<f64>>,
) -> Result<Option<VegaKtTerminalPayoff>, RiskConfigError> {
    if payoff.output_slots().len() != 1 {
        return Ok(None);
    }
    let mut slots = BTreeMap::<u32, Piecewise>::new();
    let mut line_count = 0_usize;
    for &op in payoff.opcodes() {
        let get = |id: u32| {
            slots
                .get(&id)
                .ok_or_else(|| invalid("missing compiled payoff operand"))
        };
        let (output, value) = match op {
            CompiledOpcode::Literal { value, output } => (output, Piecewise::affine(value, 0.)),
            CompiledOpcode::TerminalSpot {
                underlying: u,
                observation_date,
                output,
            } => {
                let Some((cash, slope)) = observation(u, observation_date) else {
                    return Ok(None);
                };
                FiniteF64::new(cash, "conditional_observation_cash")?;
                FiniteF64::new(slope, "conditional_observation_slope")?;
                (output, Piecewise::affine(cash, slope))
            }
            CompiledOpcode::Add {
                left,
                right,
                output,
            } => (output, get(left)?.combine(get(right)?, 0)?),
            CompiledOpcode::Subtract {
                left,
                right,
                output,
            } => (output, get(left)?.combine(get(right)?, 1)?),
            CompiledOpcode::Minimum {
                left,
                right,
                output,
            } => (output, get(left)?.combine(get(right)?, 2)?),
            CompiledOpcode::Maximum {
                left,
                right,
                output,
            } => (output, get(left)?.combine(get(right)?, 3)?),
            CompiledOpcode::Multiply {
                left,
                right,
                output,
            } => {
                let a = get(left)?;
                let b = get(right)?;
                let value = if let Some(c) = a.constant() {
                    b.scaled(c)?
                } else if let Some(c) = b.constant() {
                    a.scaled(c)?
                } else {
                    return Ok(None);
                };
                (output, value)
            }
            CompiledOpcode::Divide {
                numerator,
                denominator,
                output,
            } => {
                let Some(c) = get(denominator)?.constant() else {
                    return Ok(None);
                };
                if c == 0. {
                    return Err(invalid("nonzero constant payoff divisor required"));
                }
                (output, get(numerator)?.scaled(1. / c)?)
            }
            CompiledOpcode::Negate { input, output } => (output, get(input)?.scaled(-1.)?),
            _ => return Ok(None),
        };
        line_count += value.lines.len();
        if line_count > 65536 {
            return Err(invalid(
                "terminal payoff exceeds 65536-line compilation budget",
            ));
        }
        if let Some(cuts) = &mut boundaries {
            cuts.extend(&value.knots);
        }
        slots.insert(output, value);
    }
    slots
        .remove(&payoff.output_slots()[0])
        .ok_or_else(|| invalid("missing terminal output"))?
        .into_payoff()
        .map(Some)
}

#[cfg(test)]
mod conditional_tests {
    use super::*;
    use crate::product::{GraphLimitPolicy, SourceGraphBuilder, SourceOpcode as Op};

    #[test]
    fn conditional_ad_partition_keeps_canceled_output_kinks() {
        let mut b = SourceGraphBuilder::new();
        let u = UnderlyingId::new(1);
        let first: Date = "2026-12-04".parse().unwrap();
        let middle: Date = "2027-01-04".parse().unwrap();
        let last: Date = "2027-03-04".parse().unwrap();
        let a = b
            .push(Op::TerminalSpot {
                underlying: u,
                observation_date: first,
            })
            .unwrap();
        let c = b
            .push(Op::TerminalSpot {
                underlying: u,
                observation_date: middle,
            })
            .unwrap();
        let s = b
            .push(Op::TerminalSpot {
                underlying: u,
                observation_date: last,
            })
            .unwrap();
        let zero = b.literal(0.).unwrap();
        let sa = b.push(Op::Subtract { left: s, right: a }).unwrap();
        let sc = b.push(Op::Subtract { left: s, right: c }).unwrap();
        let pa = b
            .push(Op::Maximum {
                left: sa,
                right: zero,
            })
            .unwrap();
        let pc = b
            .push(Op::Maximum {
                left: sc,
                right: zero,
            })
            .unwrap();
        let difference = b
            .push(Op::Subtract {
                left: pa,
                right: pc,
            })
            .unwrap();
        let graph = b
            .finish(vec![difference])
            .compile(GraphLimitPolicy::DEFAULT)
            .unwrap();
        let resolve = |_, date| Some(if date == last { (0., 1.) } else { (100., 0.) });
        assert!(
            try_affine_observation_payoff_gamma_measure(&graph, resolve)
                .unwrap()
                .unwrap()
                .gamma()
                .atoms()
                .is_empty()
        );
        assert_eq!(
            conditional_payoff_branch_boundaries(&graph, resolve).unwrap(),
            vec![100.]
        );
    }

    #[test]
    fn conditional_compiler_keeps_history_signed_atoms_and_observation_dates() {
        let mut b = SourceGraphBuilder::new();
        let u = UnderlyingId::new(1);
        let first: Date = "2026-12-04".parse().unwrap();
        let last: Date = "2027-03-04".parse().unwrap();
        let s1 = b
            .push(Op::TerminalSpot {
                underlying: u,
                observation_date: first,
            })
            .unwrap();
        let s2 = b
            .push(Op::TerminalSpot {
                underlying: u,
                observation_date: last,
            })
            .unwrap();
        let half = b.literal(0.5).unwrap();
        let sum = b
            .push(Op::Add {
                left: s1,
                right: s2,
            })
            .unwrap();
        let avg = b
            .push(Op::Multiply {
                left: sum,
                right: half,
            })
            .unwrap();
        let k = b.literal(100.).unwrap();
        let diff = b
            .push(Op::Subtract {
                left: avg,
                right: k,
            })
            .unwrap();
        let zero = b.literal(0.).unwrap();
        let call = b
            .push(Op::Maximum {
                left: diff,
                right: zero,
            })
            .unwrap();
        let cap = b.literal(10.).unwrap();
        let capped = b
            .push(Op::Minimum {
                left: call,
                right: cap,
            })
            .unwrap();
        let graph = b
            .finish(vec![capped])
            .compile(GraphLimitPolicy::DEFAULT)
            .unwrap();
        // First fixing is now known at 80; last fixing varies as 1.2*x.
        let p = try_affine_observation_payoff_gamma_measure(&graph, |_, date| {
            Some(if date == first { (80., 0.) } else { (0., 1.2) })
        })
        .unwrap()
        .unwrap();
        let a = p.gamma().atoms();
        assert_eq!(a.len(), 2);
        assert!((a[0].strike() - 100.).abs() < 1e-12);
        assert!((a[1].strike() - 70. / 0.6).abs() < 1e-12);
        assert_eq!(a[0].mass(), 0.6);
        assert_eq!(a[1].mass(), -0.6);
        for x in [1., 99., 100., 110., 120., 1000.] {
            let actual = graph
                .evaluate(|_, date| Some(if date == first { 80. } else { 1.2 * x }))
                .unwrap()[0];
            assert!((p.value(x).unwrap() - actual).abs() < 1e-11);
        }
        assert!(
            try_terminal_payoff_gamma_measure(&graph, u, last)
                .unwrap()
                .is_none()
        );
        assert!(
            try_affine_observation_payoff_gamma_measure(&graph, |_, _| None)
                .unwrap()
                .is_none()
        );
        assert!(
            try_affine_observation_payoff_gamma_measure(&graph, |_, _| Some((0., f64::NAN)))
                .is_err()
        );
    }

    #[test]
    fn nonlinear_conditional_observation_products_are_not_treated_as_affine() {
        let mut b = SourceGraphBuilder::new();
        let s = b
            .push(Op::TerminalSpot {
                underlying: UnderlyingId::new(1),
                observation_date: "2027-03-04".parse().unwrap(),
            })
            .unwrap();
        let square = b.push(Op::Multiply { left: s, right: s }).unwrap();
        let graph = b
            .finish(vec![square])
            .compile(GraphLimitPolicy::DEFAULT)
            .unwrap();
        assert!(
            try_affine_observation_payoff_gamma_measure(&graph, |_, _| Some((0., 1.2)))
                .unwrap()
                .is_none()
        );
    }
}
