//! Date-based, right-continuous correlations in explicit asset order.
use crate::core::{Date, UnderlyingId};
use crate::multi_asset::MultiAssetError;
pub use pricing_numerics::{
    CorrelationDiagnostics, CorrelationError, CorrelationFactor, CorrelationToleranceConfig,
};
#[derive(Clone, Debug, PartialEq)]
pub struct CorrelationTermStructure {
    underlyings: Vec<UnderlyingId>,
    entries: Vec<(Date, CorrelationFactor)>,
    tolerances: CorrelationToleranceConfig,
}
impl CorrelationTermStructure {
    pub fn new(
        underlyings: Vec<UnderlyingId>,
        entries: Vec<(Date, Vec<Vec<f64>>)>,
        tolerances: CorrelationToleranceConfig,
    ) -> Result<Self, MultiAssetError> {
        if underlyings.is_empty()
            || underlyings
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len()
                != underlyings.len()
        {
            return Err(MultiAssetError::Invalid(
                "correlation underlyings must be nonempty and unique",
            ));
        }
        if entries.is_empty() || entries.windows(2).any(|w| w[0].0 >= w[1].0) {
            return Err(MultiAssetError::Invalid(
                "correlation dates must be strictly increasing",
            ));
        }
        let entries = entries
            .into_iter()
            .map(|(date, matrix)| {
                if matrix.len() != underlyings.len() {
                    return Err(MultiAssetError::Invalid(
                        "correlation dimension does not match underlyings",
                    ));
                }
                Ok((date, CorrelationFactor::compile(matrix, tolerances)?))
            })
            .collect::<Result<Vec<_>, MultiAssetError>>()?;
        Ok(Self {
            underlyings,
            entries,
            tolerances,
        })
    }
    pub fn underlyings(&self) -> &[UnderlyingId] {
        &self.underlyings
    }
    pub fn entries(&self) -> &[(Date, CorrelationFactor)] {
        &self.entries
    }
    pub fn tolerances(&self) -> CorrelationToleranceConfig {
        self.tolerances
    }
}
