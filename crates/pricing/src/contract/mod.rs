//! Model-independent contracts, lifecycle reduction and lowering to native MC graphs.
mod expression;
mod source;
mod wire;

pub use expression::{Aggregate, BooleanExpr, Comparison, DateExpr, NumericExpr};
pub use source::{
    BarrierDirection, Cashflow, Contract, ContractAnalysis, ContractError, Observation,
    ObservationKind, PaymentDirection, Term,
};
