//! Fundamental domain types shared by the pricing workspace.

#![forbid(unsafe_code)]

mod calendar;
mod date;
mod error;
mod ids;
mod number;
mod pointer;

pub use calendar::{BusinessDayAdjustment, Calendar};
pub use date::{Date, DayCountConvention, Weekday};
pub use error::{
    CoreError, DocumentKind, ValidationErrors, ValidationIssue, ValidationPhase,
};
pub use ids::{
    CurrencyId, CurveId, EventId, NodeId, PathIndex, SchemaVersion, TimeIndex,
    UnderlyingId,
};
pub use number::{FiniteF64, NonNegativeF64, PositiveF64};
pub use pointer::JsonPointer;

/// Returns the crate's stable workspace role name.
#[must_use]
pub const fn role() -> &'static str {
    "core"
}
