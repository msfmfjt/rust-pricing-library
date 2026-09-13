//! Payoff implementation.
pub(crate) mod barrier;
pub(crate) mod barrier_bridge;
pub(crate) mod compile;
#[cfg(test)]
#[path = "tests.rs"]
mod conformance;
pub(crate) mod tape;
pub(crate) mod valuation;
