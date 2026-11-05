//! Model-independent contracts, lifecycle reduction and lowering to native MC graphs.
mod expression;
mod source;
mod wire;

pub use expression::{Aggregate, BooleanExpr, Comparison, DateExpr, NumericExpr};
pub use source::{
    BarrierDirection, Cashflow, Contract, ContractAnalysis, ContractError, Observation,
    ObservationKind, PaymentDirection, Term,
};

mod lifecycle;
pub use lifecycle::{DeterminedCashflow, Event, LifecycleEvent, LifecycleResult, advance};

pub use crate::engine::payoff::contract::{CompiledContract, CompiledCurrencyContract};

mod builders;
pub use builders::{AutocallPeriod, DiscreteBarrierTerms, InstrumentBuilder, SwapPeriod};

pub use lifecycle::parse_events_json;
