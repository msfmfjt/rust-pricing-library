use super::*;
use crate::models::{HullWhite1Factor, HullWhiteError, HybridCorrelation};

impl GaussianRateCovariance for HullWhite1Factor {
    type Error = HullWhiteError;
    fn rate_covariance(&self, start: f64, end: f64) -> Result<RateCovariance, HullWhiteError> {
        let c = self
            .transition(start, end, 0.0, HybridCorrelation::new(0.0, 0.0, 0.0)?)?
            .covariance;
        Ok(RateCovariance {
            factor_variance: c[2][2],
            integral_variance: c[3][3],
            factor_integral: c[2][3],
        })
    }
    fn ou_rate_covariance(
        &self,
        start: f64,
        end: f64,
        reversion: f64,
    ) -> Result<OuRateCovariance, HullWhiteError> {
        let c = self
            .transition(start, end, reversion, HybridCorrelation::new(0.0, 0.0, 1.0)?)?
            .covariance;
        Ok(OuRateCovariance {
            factor: c[1][2],
            integral: c[1][3],
        })
    }
}

impl HullWhite1Factor {
    pub(crate) fn bond_transition(
        &self,
        time: f64,
        maturity: f64,
    ) -> Result<GaussianBondTransition, HullWhiteError> {
        let tr = self.transition(time, maturity, 0.0, HybridCorrelation::new(0.0, 0.0, 0.0)?)?;
        Ok(GaussianBondTransition {
            loading: BondStateLoading::new(tr.integral_loading),
            integral_variance: tr.covariance[3][3],
        })
    }
    pub(crate) fn bond_exposure(&self, time: f64, maturity: f64) -> Result<GaussianBond, HullWhiteError> {
        let tr = self.bond_transition(time, maturity)?;
        Ok(GaussianBond {
            loading: tr.loading,
            integrated_shift: self.integrated_shift(time, maturity)?,
            integral_variance: tr.integral_variance,
        })
    }
}
