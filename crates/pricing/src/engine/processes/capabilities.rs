//! Optional reverse of a recorded path. A price-only state evolution need not
//! implement this capability or allocate a record. Each record retains its
//! concrete adjoint payload and the numerical model's fixed-parameter policy.

use super::hull_white::{HullWhiteMcError, HullWhitePathAdjoints, HullWhiteRecordedPath};
use super::lsv::{BergomiLsvPath, LsvError, LsvPathAdjoints};
use super::rough_lsv::RoughBergomiLsvPath;
use crate::models::BergomiDynamics;

pub(crate) trait PathReverse {
    type Adjoints;
    type Error;
    fn path_pullback(&self, seeds: &[f64]) -> Result<Self::Adjoints, Self::Error>;
}

impl<F: BergomiDynamics> PathReverse for BergomiLsvPath<F> {
    type Adjoints = LsvPathAdjoints<F::State>;
    type Error = LsvError;
    fn path_pullback(&self, seeds: &[f64]) -> Result<Self::Adjoints, LsvError> {
        self.reverse(seeds)
    }
}

impl PathReverse for RoughBergomiLsvPath {
    type Adjoints = LsvPathAdjoints<()>;
    type Error = LsvError;
    fn path_pullback(&self, seeds: &[f64]) -> Result<Self::Adjoints, LsvError> {
        self.reverse(seeds)
    }
}

impl PathReverse for HullWhiteRecordedPath<'_> {
    type Adjoints = HullWhitePathAdjoints;
    type Error = HullWhiteMcError;
    fn path_pullback(&self, seeds: &[f64]) -> Result<Self::Adjoints, HullWhiteMcError> {
        self.reverse(seeds)
    }
}
