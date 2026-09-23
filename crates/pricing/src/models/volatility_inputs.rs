//! Borrowed volatility inputs. These types are internal despite their nominal
//! visibility: the sealed Bergomi supertrait uses them without exporting them.

#[derive(Clone, Copy, Debug)]
pub struct OrthogonalNormals<'a> {
    values: &'a [f64],
    stride: usize,
}

impl<'a> OrthogonalNormals<'a> {
    /// The caller validates the full factor-major buffer before taking a step.
    pub(crate) fn new(values: &'a [f64], stride: usize) -> Self {
        Self { values, stride }
    }
    pub(crate) fn get(self, factor: usize) -> f64 {
        self.values[factor * self.stride]
    }
}

/// Already correlated OU increments, with their transition variance included.
/// They must not pass through the independent-normal loading a second time.
#[derive(Clone, Copy, Debug)]
pub struct OuInnovations<'a> {
    values: &'a [f64],
    stride: usize,
}

impl<'a> OuInnovations<'a> {
    pub(crate) fn new(values: &'a [f64], stride: usize) -> Self {
        Self { values, stride }
    }
    pub(crate) fn get(self, factor: usize) -> f64 {
        self.values[factor * self.stride]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InnovationSource {
    IndependentNormals,
    JointOuIncrements,
}

impl InnovationSource {
    pub(crate) fn from_external(external: bool) -> Self {
        if external {
            Self::JointOuIncrements
        } else {
            Self::IndependentNormals
        }
    }
}

/// The coordinate adjoints have the concrete model's factor count, including
/// reserved coordinates whose loadings happen to be zero.
pub struct FactorAdjoints<S, C> {
    pub(crate) state: S,
    pub(crate) spot: f64,
    pub(crate) volatility: C,
}

/// Complete Brownian and newest-cell histories for a Volterra preparation.
/// This is deliberately distinct from a Markov step's OU increments.
#[derive(Clone, Copy, Debug)]
pub(crate) struct HistoryInnovations<'a> {
    pub(crate) increments: &'a [f64],
    pub(crate) near_cell: &'a [f64],
}
