use super::ContractError;
use crate::core::Date;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Comparison {
    Less,
    LessEqual,
    Equal,
    NotEqual,
    GreaterEqual,
    Greater,
}
impl Comparison {
    pub(crate) fn test<T: PartialOrd + PartialEq>(self, a: T, b: T) -> bool {
        match self {
            Self::Less => a < b,
            Self::LessEqual => a <= b,
            Self::Equal => a == b,
            Self::NotEqual => a != b,
            Self::GreaterEqual => a >= b,
            Self::Greater => a > b,
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Aggregate {
    ArithmeticAverage,
    GeometricAverage,
    RunningMinimum,
    RunningMaximum,
}

/// Typed source expressions. Observation references are validated against the
/// contract's observation registry; dates and predicates cannot be numeric operands.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum NumericExpr {
    Constant {
        value: f64,
    },
    Observation {
        id: String,
    },
    Add {
        left: Box<Self>,
        right: Box<Self>,
    },
    Subtract {
        left: Box<Self>,
        right: Box<Self>,
    },
    Multiply {
        left: Box<Self>,
        right: Box<Self>,
    },
    Divide {
        left: Box<Self>,
        right: Box<Self>,
    },
    Minimum {
        left: Box<Self>,
        right: Box<Self>,
    },
    Maximum {
        left: Box<Self>,
        right: Box<Self>,
    },
    Abs {
        input: Box<Self>,
    },
    Negate {
        input: Box<Self>,
    },
    Conditional {
        condition: Box<BooleanExpr>,
        then_value: Box<Self>,
        else_value: Box<Self>,
    },
    Aggregate {
        operation: Aggregate,
        values: Vec<Self>,
    },
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum BooleanExpr {
    Constant {
        value: bool,
    },
    Compare {
        operation: Comparison,
        left: Box<NumericExpr>,
        right: Box<NumericExpr>,
    },
    DateCompare {
        operation: Comparison,
        left: Box<DateExpr>,
        right: Box<DateExpr>,
    },
    And {
        left: Box<Self>,
        right: Box<Self>,
    },
    Or {
        left: Box<Self>,
        right: Box<Self>,
    },
    Not {
        input: Box<Self>,
    },
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum DateExpr {
    Constant {
        #[serde(with = "super::wire::date")]
        value: Date,
    },
    Conditional {
        condition: Box<BooleanExpr>,
        then_value: Box<Self>,
        else_value: Box<Self>,
    },
}

impl From<f64> for NumericExpr {
    fn from(value: f64) -> Self {
        Self::Constant { value }
    }
}
impl From<bool> for BooleanExpr {
    fn from(value: bool) -> Self {
        Self::Constant { value }
    }
}
impl From<Date> for DateExpr {
    fn from(value: Date) -> Self {
        Self::Constant { value }
    }
}
macro_rules! binary_numeric {
    ($($method:ident => $variant:ident),*) => { $(pub fn $method(self, right: Self) -> Self {
        Self::$variant { left: Box::new(self), right: Box::new(right) }
    })* };
}
impl NumericExpr {
    /// Evaluate with explicit fixing values; unresolved observations are errors.
    pub fn evaluate(&self, fixings: &BTreeMap<String, f64>) -> Result<f64, ContractError> {
        let mut refs = BTreeSet::new();
        self.refs(&mut refs, 0, &mut 0)?;
        let reduced = self.reduce(fixings)?;
        reduced
            .constant()
            .ok_or_else(|| ContractError::MissingFixing("expression remains stochastic".into()))
    }

    pub fn observation(id: impl Into<String>) -> Self {
        Self::Observation { id: id.into() }
    }
    binary_numeric!(plus => Add, minus => Subtract, times => Multiply, divided_by => Divide, minimum => Minimum, maximum => Maximum);
    pub fn abs(self) -> Self {
        Self::Abs {
            input: Box::new(self),
        }
    }
    pub fn compare(self, operation: Comparison, right: Self) -> BooleanExpr {
        BooleanExpr::Compare {
            operation,
            left: Box::new(self),
            right: Box::new(right),
        }
    }
    pub fn select(condition: BooleanExpr, then_value: Self, else_value: Self) -> Self {
        Self::Conditional {
            condition: Box::new(condition),
            then_value: Box::new(then_value),
            else_value: Box::new(else_value),
        }
    }
    pub(crate) fn constant(&self) -> Option<f64> {
        if let Self::Constant { value } = self {
            Some(*value)
        } else {
            None
        }
    }
    pub(crate) fn reduce(&self, fixings: &BTreeMap<String, f64>) -> Result<Self, ContractError> {
        let binary = |a: &Self,
                      b: &Self,
                      op: fn(f64, f64) -> f64,
                      build: fn(Box<Self>, Box<Self>) -> Self| {
            let a = a.reduce(fixings)?;
            let b = b.reduce(fixings)?;
            if let (Some(x), Some(y)) = (a.constant(), b.constant()) {
                finite(op(x, y)).map(Self::from)
            } else {
                Ok(build(Box::new(a), Box::new(b)))
            }
        };
        match self {
            Self::Constant { value } => finite(*value).map(Self::from),
            Self::Observation { id } => match fixings.get(id) {
                Some(value) => finite(*value).map(Self::from),
                None => Ok(self.clone()),
            },
            Self::Add { left, right } => binary(
                left,
                right,
                |a, b| a + b,
                |left, right| Self::Add { left, right },
            ),
            Self::Subtract { left, right } => binary(
                left,
                right,
                |a, b| a - b,
                |left, right| Self::Subtract { left, right },
            ),
            Self::Multiply { left, right } => binary(
                left,
                right,
                |a, b| a * b,
                |left, right| Self::Multiply { left, right },
            ),
            Self::Divide { left, right } => binary(
                left,
                right,
                |a, b| a / b,
                |left, right| Self::Divide { left, right },
            ),
            Self::Minimum { left, right } => binary(left, right, f64::min, |left, right| {
                Self::Minimum { left, right }
            }),
            Self::Maximum { left, right } => binary(left, right, f64::max, |left, right| {
                Self::Maximum { left, right }
            }),
            Self::Abs { input } | Self::Negate { input } => {
                let input = input.reduce(fixings)?;
                if let Some(v) = input.constant() {
                    Ok(Self::from(if matches!(self, Self::Abs { .. }) {
                        v.abs()
                    } else {
                        -v
                    }))
                } else if matches!(self, Self::Abs { .. }) {
                    Ok(input.abs())
                } else {
                    Ok(Self::Negate {
                        input: Box::new(input),
                    })
                }
            }
            Self::Conditional {
                condition,
                then_value,
                else_value,
            } => {
                let condition = condition.reduce(fixings)?;
                if let Some(v) = condition.constant() {
                    if v { then_value } else { else_value }.reduce(fixings)
                } else {
                    Ok(Self::select(
                        condition,
                        then_value.reduce(fixings)?,
                        else_value.reduce(fixings)?,
                    ))
                }
            }
            Self::Aggregate { operation, values } => {
                let values = values
                    .iter()
                    .map(|v| v.reduce(fixings))
                    .collect::<Result<Vec<_>, _>>()?;
                if values.is_empty() {
                    return Err(ContractError::Invalid(
                        "aggregation requires observations".into(),
                    ));
                }
                if let Some(v) = values
                    .iter()
                    .map(Self::constant)
                    .collect::<Option<Vec<_>>>()
                {
                    let n = v.len() as f64;
                    let result = match operation {
                        Aggregate::ArithmeticAverage => v.iter().map(|x| x / n).sum(),
                        Aggregate::GeometricAverage => {
                            if v.iter().any(|x| *x <= 0.0) {
                                return Err(ContractError::Invalid(
                                    "geometric average requires strictly positive values".into(),
                                ));
                            }
                            v.iter().map(|x| x.ln() / n).sum::<f64>().exp()
                        }
                        Aggregate::RunningMinimum => {
                            v.iter().copied().fold(f64::INFINITY, f64::min)
                        }
                        Aggregate::RunningMaximum => {
                            v.iter().copied().fold(f64::NEG_INFINITY, f64::max)
                        }
                    };
                    finite(result).map(Self::from)
                } else {
                    Ok(Self::Aggregate {
                        operation: *operation,
                        values,
                    })
                }
            }
        }
    }
    pub(crate) fn refs(
        &self,
        out: &mut BTreeSet<String>,
        depth: usize,
        count: &mut usize,
    ) -> Result<(), ContractError> {
        budget(depth, count)?;
        match self {
            Self::Constant { value } => {
                finite(*value)?;
            }
            Self::Observation { id } => {
                out.insert(id.clone());
            }
            Self::Add { left, right }
            | Self::Subtract { left, right }
            | Self::Multiply { left, right }
            | Self::Divide { left, right }
            | Self::Minimum { left, right }
            | Self::Maximum { left, right } => {
                left.refs(out, depth + 1, count)?;
                right.refs(out, depth + 1, count)?;
            }
            Self::Abs { input } | Self::Negate { input } => input.refs(out, depth + 1, count)?,
            Self::Conditional {
                condition,
                then_value,
                else_value,
            } => {
                condition.refs(out, depth + 1, count)?;
                then_value.refs(out, depth + 1, count)?;
                else_value.refs(out, depth + 1, count)?;
            }
            Self::Aggregate { values, .. } => {
                if values.is_empty() {
                    return Err(ContractError::Invalid("empty aggregation".into()));
                }
                for v in values {
                    v.refs(out, depth + 1, count)?;
                }
            }
        }
        Ok(())
    }
}
impl BooleanExpr {
    /// Evaluate with explicit fixing values; unresolved observations are errors.
    pub fn evaluate(&self, fixings: &BTreeMap<String, f64>) -> Result<bool, ContractError> {
        let mut refs = BTreeSet::new();
        self.refs(&mut refs, 0, &mut 0)?;
        let reduced = self.reduce(fixings)?;
        reduced
            .constant()
            .ok_or_else(|| ContractError::MissingFixing("expression remains stochastic".into()))
    }

    pub fn and(self, right: Self) -> Self {
        Self::And {
            left: Box::new(self),
            right: Box::new(right),
        }
    }
    pub fn or(self, right: Self) -> Self {
        Self::Or {
            left: Box::new(self),
            right: Box::new(right),
        }
    }
    pub fn negate(self) -> Self {
        Self::Not {
            input: Box::new(self),
        }
    }
    pub(crate) fn constant(&self) -> Option<bool> {
        if let Self::Constant { value } = self {
            Some(*value)
        } else {
            None
        }
    }
    pub(crate) fn reduce(&self, fixings: &BTreeMap<String, f64>) -> Result<Self, ContractError> {
        Ok(match self {
            Self::Constant { .. } => self.clone(),
            Self::Compare {
                operation,
                left,
                right,
            } => {
                let a = left.reduce(fixings)?;
                let b = right.reduce(fixings)?;
                if let (Some(a), Some(b)) = (a.constant(), b.constant()) {
                    Self::from(operation.test(a, b))
                } else {
                    Self::Compare {
                        operation: *operation,
                        left: Box::new(a),
                        right: Box::new(b),
                    }
                }
            }
            Self::DateCompare {
                operation,
                left,
                right,
            } => {
                let a = left.reduce(fixings)?;
                let b = right.reduce(fixings)?;
                if let (Some(a), Some(b)) = (a.constant(), b.constant()) {
                    Self::from(operation.test(a, b))
                } else {
                    Self::DateCompare {
                        operation: *operation,
                        left: Box::new(a),
                        right: Box::new(b),
                    }
                }
            }
            Self::Not { input } => {
                let a = input.reduce(fixings)?;
                a.constant().map_or_else(|| a.negate(), |v| Self::from(!v))
            }
            Self::And { left, right } | Self::Or { left, right } => {
                let a = left.reduce(fixings)?;
                let and = matches!(self, Self::And { .. });
                if a.constant() == Some(!and) {
                    Self::from(!and)
                } else {
                    let b = right.reduce(fixings)?;
                    if a.constant() == Some(and) {
                        b
                    } else if b.constant() == Some(!and) {
                        Self::from(!and)
                    } else if b.constant() == Some(and) {
                        a
                    } else if and {
                        a.and(b)
                    } else {
                        a.or(b)
                    }
                }
            }
        })
    }
    pub(crate) fn refs(
        &self,
        out: &mut BTreeSet<String>,
        depth: usize,
        count: &mut usize,
    ) -> Result<(), ContractError> {
        budget(depth, count)?;
        match self {
            Self::Constant { .. } => {}
            Self::Compare { left, right, .. } => {
                left.refs(out, depth + 1, count)?;
                right.refs(out, depth + 1, count)?;
            }
            Self::DateCompare { left, right, .. } => {
                left.refs(out, depth + 1, count)?;
                right.refs(out, depth + 1, count)?;
            }
            Self::And { left, right } | Self::Or { left, right } => {
                left.refs(out, depth + 1, count)?;
                right.refs(out, depth + 1, count)?;
            }
            Self::Not { input } => input.refs(out, depth + 1, count)?,
        }
        Ok(())
    }
}
impl DateExpr {
    /// Evaluate with explicit fixing values; unresolved observations are errors.
    pub fn evaluate(&self, fixings: &BTreeMap<String, f64>) -> Result<Date, ContractError> {
        let mut refs = BTreeSet::new();
        self.refs(&mut refs, 0, &mut 0)?;
        let reduced = self.reduce(fixings)?;
        reduced
            .constant()
            .ok_or_else(|| ContractError::MissingFixing("expression remains stochastic".into()))
    }

    pub(crate) fn constant(&self) -> Option<Date> {
        if let Self::Constant { value } = self {
            Some(*value)
        } else {
            None
        }
    }
    pub(crate) fn reduce(&self, f: &BTreeMap<String, f64>) -> Result<Self, ContractError> {
        match self {
            Self::Constant { .. } => Ok(self.clone()),
            Self::Conditional {
                condition,
                then_value,
                else_value,
            } => {
                let c = condition.reduce(f)?;
                if let Some(v) = c.constant() {
                    if v { then_value } else { else_value }.reduce(f)
                } else {
                    Ok(Self::Conditional {
                        condition: Box::new(c),
                        then_value: Box::new(then_value.reduce(f)?),
                        else_value: Box::new(else_value.reduce(f)?),
                    })
                }
            }
        }
    }
    pub(crate) fn refs(
        &self,
        out: &mut BTreeSet<String>,
        depth: usize,
        count: &mut usize,
    ) -> Result<(), ContractError> {
        budget(depth, count)?;
        if let Self::Conditional {
            condition,
            then_value,
            else_value,
        } = self
        {
            condition.refs(out, depth + 1, count)?;
            then_value.refs(out, depth + 1, count)?;
            else_value.refs(out, depth + 1, count)?;
        }
        Ok(())
    }
}
pub(crate) fn finite(v: f64) -> Result<f64, ContractError> {
    if v.is_finite() {
        Ok(v)
    } else {
        Err(ContractError::Invalid("nonfinite expression value".into()))
    }
}
pub(crate) fn budget(depth: usize, count: &mut usize) -> Result<(), ContractError> {
    *count += 1;
    if depth > 64 || *count > 100_000 {
        Err(ContractError::Invalid(
            "contract resource limit exceeded".into(),
        ))
    } else {
        Ok(())
    }
}
