//! Nonuniform kappa=1 Volterra hybrid scheme. The newest cell is exact;
//! older cells use their L2-optimal kernel averages. No future noise is used.

use super::*;

pub const ROUGH_BERGOMI_SCHEME: &str = "rough-bergomi-hw-hybrid-kappa1-log-euler-v1";
pub const ROUGH_LSV_CALIBRATION: &str = "rough-lsv-hw-discounted-quartic-v1";
pub const ROUGH_CASH_LSV_CALIBRATION: &str = "rough-lsv-hw-escrowed-quadratic-v1";

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum HybridVolatilityFactor {
    Bergomi(Bergomi1Factor),
    Rough(RoughBergomi),
}
impl From<Bergomi1Factor> for HybridVolatilityFactor {
    fn from(v: Bergomi1Factor) -> Self {
        Self::Bergomi(v)
    }
}
impl From<RoughBergomi> for HybridVolatilityFactor {
    fn from(v: RoughBergomi) -> Self {
        Self::Rough(v)
    }
}
impl HybridVolatilityFactor {
    #[must_use]
    pub fn mean_reversion(self) -> f64 {
        match self {
            Self::Bergomi(v) => v.mean_reversion(),
            Self::Rough(_) => 0.0,
        }
    }
    /// Coefficient of log volatility, half eta for rough Bergomi.
    #[must_use]
    pub fn vol_of_vol(self) -> f64 {
        match self {
            Self::Bergomi(v) => v.vol_of_vol(),
            Self::Rough(v) => 0.5 * v.vol_of_vol(),
        }
    }
    #[must_use]
    pub fn correlation(self) -> f64 {
        match self {
            Self::Bergomi(v) => v.correlation(),
            Self::Rough(v) => v.correlation(),
        }
    }
    #[must_use]
    pub fn rough(self) -> Option<RoughBergomi> {
        match self {
            Self::Rough(v) => Some(v),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct RoughBergomiDriverPlan {
    model: RoughBergomi,
    times: Box<[f64]>,
    weights: Box<[Box<[f64]>]>,
    variances: Box<[f64]>,
    vol_loadings: Box<[[f64; 5]]>,
    near_loadings: Box<[[f64; 5]]>,
}
impl RoughBergomiDriverPlan {
    pub fn new(
        model: RoughBergomi,
        rates: &HullWhite1Factor,
        correlation: HybridCorrelation,
        grid: &LocalVolTimeGrid,
    ) -> Result<Self, HullWhiteMcError> {
        Self::compile(model, rates, correlation, grid.nodes())
    }
    pub(in crate::engine) fn compile(
        model: RoughBergomi,
        rates: &HullWhite1Factor,
        correlation: HybridCorrelation,
        times: &[f64],
    ) -> Result<Self, HullWhiteMcError> {
        let mut vol_loadings = Vec::new();
        let mut near_loadings = Vec::new();
        for pair in times.windows(2) {
            let loading = covariance_loading(model.hybrid_covariance(
                rates,
                pair[0],
                pair[1],
                correlation,
            )?)?;
            vol_loadings.push(loading[1]);
            near_loadings.push(if model.hurst() == 0.5 {
                loading[1]
            } else {
                loading[4]
            });
        }
        let mut weights = vec![Vec::new().into_boxed_slice()];
        let mut variances = vec![0.0];
        for i in 1..times.len() {
            let mut row = Vec::with_capacity(i - 1);
            let mut variance = NeumaierSum::new();
            variance.add((times[i] - times[i - 1]).powf(2.0 * model.hurst()));
            for j in 0..i - 1 {
                let weight = model.average_kernel(times[i] - times[j + 1], times[i] - times[j])?;
                row.push(weight);
                variance.add(weight * weight * (times[j + 1] - times[j]));
            }
            hw_valid(variance.total(), "rough_driver_variance", i, true)?;
            weights.push(row.into_boxed_slice());
            variances.push(variance.total());
        }
        Ok(Self {
            model,
            times: times.into(),
            weights: weights.into(),
            variances: variances.into(),
            vol_loadings: vol_loadings.into(),
            near_loadings: near_loadings.into(),
        })
    }
    #[must_use]
    pub fn times(&self) -> &[f64] {
        &self.times
    }
    #[must_use]
    pub fn variances(&self) -> &[f64] {
        &self.variances
    }
    /// Raw Gaussian X_H at the time nodes. Input has five factor-major blocks.
    pub fn evolve(&self, shocks: &[f64]) -> Result<Vec<f64>, HullWhiteMcError> {
        let n = self.times.len() - 1;
        if n.checked_mul(5) != Some(shocks.len()) {
            return Err(invalid("rough_shock_count", shocks.len()));
        }
        for &v in shocks {
            hw_valid(v, "rough_shock", 0, false)?;
        }
        let mut dw = Vec::with_capacity(n);
        let mut near = Vec::with_capacity(n);
        for i in 0..n {
            let z = std::array::from_fn::<_, 5, _>(|j| shocks[j * n + i]);
            dw.push(
                self.vol_loadings[i]
                    .iter()
                    .zip(z)
                    .map(|(a, z)| a * z)
                    .sum::<f64>(),
            );
            near.push(
                self.near_loadings[i]
                    .iter()
                    .zip(z)
                    .map(|(a, z)| a * z)
                    .sum::<f64>(),
            );
        }
        let mut values = vec![0.0];
        for i in 1..=n {
            let mut value = NeumaierSum::new();
            value.add(near[i - 1]);
            for (&w, &z) in self.weights[i].iter().zip(&dw) {
                value.add(w * z);
            }
            hw_valid(value.total(), "rough_driver_state", i, false)?;
            values.push(value.total());
        }
        Ok(values)
    }
    /// Store Y=X_H-eta*Var_hybrid/2 so exp(eta*Y) has unit Gaussian mean.
    pub(in crate::engine) fn normalized(
        &self,
        shocks: &[f64],
    ) -> Result<Vec<f64>, HullWhiteMcError> {
        let mut values = self.evolve(shocks)?;
        for (i, (x, &var)) in values.iter_mut().zip(&self.variances).enumerate() {
            *x -= 0.5 * self.model.vol_of_vol() * var;
            hw_valid(*x, "rough_normalized_state", i, false)?;
        }
        Ok(values)
    }
}
