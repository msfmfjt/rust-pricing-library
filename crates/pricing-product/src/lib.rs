//! Product specifications and Event/Payoff graph compilation.

#![forbid(unsafe_code)]

mod graph;
mod smoothing;
mod spec;

pub use graph::{
    CompiledOpcode, CompiledPayoff, GraphError, GraphFingerprint, GraphLimitPolicy,
    PayoffEvaluation, SourceGraph, SourceGraphBuilder, SourceNode, SourceOpcode, TerminalAdjoint,
};
pub use smoothing::{CompactC2Smoothing, SmoothingBinaryDerivatives, SmoothingDerivatives};
pub use spec::{
    ArithmeticAsianSpec, AsianObservation, AsianObservationValue, BarrierDirection, BarrierSpec,
    BarrierStyle, DigitalPayout, DigitalSpec, EuropeanVanillaSpec, FixedLookbackSpec, OptionSide,
    ProductSpec,
};

/// Returns the domain foundation role.
#[must_use]
pub const fn foundation_role() -> &'static str {
    pricing_core::role()
}
