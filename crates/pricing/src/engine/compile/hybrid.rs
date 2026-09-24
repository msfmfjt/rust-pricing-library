//! Paired HW calibration inputs and their concrete pricing-volatility lowering.
//! The facades retain their own target-grid acceptance and error conventions.

use crate::mc::hull_white::{
    CalibratedHullWhiteLsv, HullWhiteLsvTarget, HullWhiteMcError, HybridEquityVolatility,
    HybridVolatilityFactor, calibrate_hybrid_lsv_with_dividends,
};
use crate::mc::lsv::LsvParticleConfig;
use crate::models::hull_white_dividends::HullWhiteDividendPlan;
use crate::models::{HullWhite1Factor, HybridCorrelation};

pub(crate) struct HybridLsvComposition<'a> {
    pub target: &'a HullWhiteLsvTarget,
    pub factor: HybridVolatilityFactor,
    pub rates: &'a HullWhite1Factor,
    pub correlation: HybridCorrelation,
    pub particles: &'a LsvParticleConfig,
}

pub(crate) struct CalibratedHybridLsv {
    factor: HybridVolatilityFactor,
    calibration: CalibratedHullWhiteLsv,
}

impl HybridLsvComposition<'_> {
    pub fn calibrate(
        self,
        initial_spot: f64,
        dividends: &HullWhiteDividendPlan,
    ) -> Result<CalibratedHybridLsv, HullWhiteMcError> {
        let calibration = calibrate_hybrid_lsv_with_dividends(
            self.target,
            self.factor,
            self.rates,
            self.correlation,
            initial_spot,
            self.particles,
            Some(dividends),
        )?;
        Ok(CalibratedHybridLsv {
            factor: self.factor,
            calibration,
        })
    }
}

impl CalibratedHybridLsv {
    /// The factor and leverage surface are selected together. In particular,
    /// retain the second volatility/rate correlation and the rough convention.
    pub fn into_parts(self) -> (HybridEquityVolatility, CalibratedHullWhiteLsv) {
        let leverage = self.calibration.surface.clone();
        let volatility = match self.factor {
            HybridVolatilityFactor::Bergomi(factor) => {
                HybridEquityVolatility::BergomiLsv { factor, leverage }
            }
            HybridVolatilityFactor::BergomiTwoFactor {
                factor,
                second_vol_rate_correlation,
            } => HybridEquityVolatility::Bergomi2FactorLsv {
                factor,
                second_vol_rate_correlation,
                leverage,
            },
            HybridVolatilityFactor::Rough(factor) => {
                HybridEquityVolatility::RoughBergomiLsv { factor, leverage }
            }
        };
        (volatility, self.calibration)
    }
}
