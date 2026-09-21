//! Compile-time selection of the existing marginal and shared-rate adapters.
//! Validation stays staged: request shape precedes product checks, while joint
//! covariance and paired-target checks require the compiled contractual grid.

use super::*;
use crate::market::EquityMarket;
use crate::mc::LocalVolTimeGrid;
use crate::multi_asset::MultiAssetError as E;
use std::sync::Arc;

pub(super) enum RateComposition {
    Deterministic,
    HullWhite(MultiAssetHullWhiteConfig),
}

impl From<Option<MultiAssetHullWhiteConfig>> for RateComposition {
    fn from(config: Option<MultiAssetHullWhiteConfig>) -> Self {
        config.map_or(Self::Deterministic, Self::HullWhite)
    }
}

pub(super) struct ModelComposition {
    pub marginals: Vec<Option<MultiAssetBergomiLsvConfig>>,
    pub driver_correlations: Option<Vec<Vec<Vec<f64>>>>,
    pub rates: RateComposition,
    pub local_correlation: Option<(LocalCorrelationConfig, LocalCorrelationExtensions)>,
}

/// Selected once per asset at compilation. The public plan keeps its existing
/// storage and path dispatch, including HW's reserved zero-loading coordinates.
pub(super) enum CompiledMarginal {
    Direct,
    Lsv(Arc<lsv::LsvAsset>),
    HullWhite(Arc<hull_white::HwAsset>),
}

impl ModelComposition {
    pub fn validate(&self, markets: &[EquityMarket], models: &[ModelSpec]) -> Result<(), E> {
        if markets.is_empty() || markets.len() != models.len() {
            return Err(E::Invalid("market/model dimensions differ"));
        }
        if self.marginals.len() != models.len() {
            return Err(E::Invalid("LSV configuration count must equal asset count"));
        }
        if matches!(self.rates, RateComposition::Deterministic) && self.rough_count() > 0 {
            return Err(E::Invalid(
                "rough-LSV requires the shared HW adapter and paired targets; use a zero-volatility HW model for deterministic rates",
            ));
        }
        for (config, model) in self.marginals.iter().zip(models) {
            if config.is_some() && !matches!(model, ModelSpec::LocalVolatility(_)) {
                return Err(E::Invalid(
                    "each LSV asset requires a LocalVolatility target model",
                ));
            }
        }
        Ok(())
    }

    fn rough_count(&self) -> usize {
        self.marginals
            .iter()
            .flatten()
            .filter(|c| c.rough().is_some())
            .count()
    }

    pub fn driver_layout(&self) -> Result<driver_layout::DriverLayout, E> {
        driver_layout::DriverLayout::compile(
            self.marginals.iter().map(|c| {
                c.as_ref()
                    .map_or(0, MultiAssetBergomiLsvConfig::factor_count)
            }),
            match self.rates {
                RateComposition::Deterministic => 0,
                RateComposition::HullWhite(_) => 2,
            },
            self.rough_count(),
        )
    }

    pub fn compile_drivers(
        &mut self,
        correlation: &CorrelationTermStructure,
        times: &[f64],
        intervals: &[usize],
    ) -> Result<Option<lsv::LsvDrivers>, E> {
        match &self.rates {
            RateComposition::Deterministic => lsv::LsvDrivers::compile(
                correlation,
                times,
                intervals,
                &self.marginals,
                self.driver_correlations.take(),
            ),
            RateComposition::HullWhite(hw) => hull_white::compile_drivers(
                correlation,
                times,
                intervals,
                &self.marginals,
                self.driver_correlations.take(),
                hw,
            )
            .map(Some),
        }
    }

    pub fn compile_marginal(
        &self,
        market: &EquityForward,
        model: &ModelSpec,
        grid: &LocalVolTimeGrid,
        asset: usize,
        driver_offset: usize,
    ) -> Result<CompiledMarginal, E> {
        let config = self.marginals[asset].clone();
        match &self.rates {
            RateComposition::HullWhite(hw) => hull_white::HwAsset::compile(
                market,
                model,
                grid,
                config,
                hw,
                asset,
                driver_offset,
            )
            .map(|a| CompiledMarginal::HullWhite(Arc::new(a))),
            RateComposition::Deterministic => match config {
                Some(config) => {
                    let ModelSpec::LocalVolatility(lv) = model else {
                        unreachable!("validated LSV target")
                    };
                    lsv::LsvAsset::compile(lv.local_variance_grid(), grid, config)
                        .map(|a| CompiledMarginal::Lsv(Arc::new(a)))
                }
                None => Ok(CompiledMarginal::Direct),
            },
        }
    }
}
