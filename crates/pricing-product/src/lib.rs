//! Product specifications and Event/Payoff graph compilation.

#![forbid(unsafe_code)]

mod graph;
mod smoothing;
mod spec;

pub use graph::{
    CompiledOpcode, CompiledPayoff, GraphError, GraphFingerprint, GraphLimitPolicy,
    PayoffEvaluation, PreDividendAdjoint, SourceGraph, SourceGraphBuilder, SourceNode,
    SourceOpcode, TerminalAdjoint,
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
    pricing_core::role()
}
