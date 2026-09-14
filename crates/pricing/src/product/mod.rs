//! Product specifications and Event/Payoff graph compilation.

#![forbid(unsafe_code)]

pub(crate) mod graph;
mod smoothing;
mod spec;

pub use crate::engine::payoff::tape::{
    CompiledOpcode, CompiledPayoff, PayoffEvaluation, PreDividendAdjoint, TerminalAdjoint,
};
pub use crate::product::graph::{
    GraphError, GraphFingerprint, GraphLimitPolicy, SourceGraph, SourceGraphBuilder, SourceNode,
    SourceOpcode,
};
pub use smoothing::{CompactC2Smoothing, SmoothingBinaryDerivatives, SmoothingDerivatives};
pub use spec::{
    AmericanVanillaSpec, ArithmeticAsianSpec, AsianObservation, AsianObservationValue,
    BarrierDirection, BarrierMonitoring, BarrierSpec, BarrierStyle, DigitalPayout, DigitalSpec,
    EuropeanVanillaSpec, ExerciseObservationTiming, FixedLookbackSpec, NormalizedExerciseEvent,
    OptionSide, ProductSpec, every_business_day_exercise_schedule,
};

/// Returns the domain foundation role.
#[must_use]
pub const fn foundation_role() -> &'static str {
    crate::core::role()
}

pub mod multi_asset;
