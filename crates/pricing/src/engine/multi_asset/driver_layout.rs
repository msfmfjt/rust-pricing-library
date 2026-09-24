//! Compiled coordinate ranges for existing multi-asset driver kernels.
//! Correlation and covariance kernels retain ownership of their numeric loadings.

use crate::multi_asset::MultiAssetError;
use std::ops::Range;

#[derive(Clone, Debug)]
pub(super) struct DriverLayout {
    volatility: Box<[Range<usize>]>,
    base_factor_count: usize,
}

impl DriverLayout {
    /// Coordinates remain ordered as spots, per-asset volatility factors,
    /// rate state/integral, then history auxiliaries. Zero-loading coordinates
    /// remain reserved. Counts describe innovations, not Brownian correlations.
    pub(super) fn compile(
        factor_counts: impl ExactSizeIterator<Item = usize>,
        rate_coordinates: usize,
        history_coordinates: usize,
    ) -> Result<Self, MultiAssetError> {
        let mut next = factor_counts.len();
        let mut volatility = Vec::with_capacity(next);
        for count in factor_counts {
            let end = next
                .checked_add(count)
                .ok_or(MultiAssetError::Invalid("random factor count overflow"))?;
            volatility.push(next..end);
            next = end;
        }
        let base_factor_count = next
            .checked_add(rate_coordinates)
            .and_then(|n| n.checked_add(history_coordinates))
            .ok_or(MultiAssetError::Invalid("random factor count overflow"))?;
        Ok(Self {
            volatility: volatility.into_boxed_slice(),
            base_factor_count,
        })
    }

    pub(super) fn volatility(&self, asset: usize) -> Range<usize> {
        self.volatility[asset].clone()
    }

    pub(super) fn base_factor_count(&self) -> usize {
        self.base_factor_count
    }

    /// Local correlation samples two endpoint blocks. Its calibration receives
    /// the base plan before those blocks are attached, so block count is explicit.
    pub(super) fn dimension(
        &self,
        step_count: usize,
        endpoint_blocks: usize,
    ) -> Result<u32, MultiAssetError> {
        self.base_factor_count
            .checked_mul(endpoint_blocks)
            .and_then(|n| n.checked_mul(step_count))
            .and_then(|n| u32::try_from(n).ok())
            .ok_or(MultiAssetError::Invalid("random dimension overflow"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mixed_assets_keep_spot_volatility_rate_and_history_order() {
        // Four spots, 2F/BS/1F/rough volatility, HW state/integral,
        // and the rough newest-cell auxiliary. Inactive loadings still count.
        let layout = DriverLayout::compile([2, 0, 1, 1].into_iter(), 2, 1).unwrap();
        assert_eq!(layout.volatility(0), 4..6);
        assert_eq!(layout.volatility(1), 6..6);
        assert_eq!(layout.volatility(2), 6..7);
        assert_eq!(layout.volatility(3), 7..8);
        assert_eq!(layout.base_factor_count(), 11);
        assert_eq!(layout.dimension(5, 1).unwrap(), 55);
        assert_eq!(layout.dimension(5, 2).unwrap(), 110);
    }

    #[test]
    fn layout_does_not_assume_one_or_two_volatility_factors() {
        // A descriptor-only extension probe; no new pricing model is registered.
        let layout = DriverLayout::compile([3, 0].into_iter(), 0, 0).unwrap();
        assert_eq!(layout.volatility(0), 2..5);
        assert_eq!(layout.volatility(1), 5..5);
        assert_eq!(layout.dimension(7, 1).unwrap(), 35);
    }

    #[test]
    fn coordinate_and_dimension_overflow_are_rejected() {
        assert!(DriverLayout::compile([usize::MAX].into_iter(), 0, 0).is_err());
        assert!(DriverLayout::compile([0].into_iter(), usize::MAX, 0).is_err());
        assert!(DriverLayout::compile([0].into_iter(), 0, usize::MAX).is_err());
        let layout = DriverLayout::compile([1].into_iter(), 0, 0).unwrap();
        assert!(layout.dimension(usize::MAX, 1).is_err());
        assert!(layout.dimension(1, usize::MAX).is_err());
        assert!(layout.dimension(u32::MAX as usize, 1).is_err());
    }
}
