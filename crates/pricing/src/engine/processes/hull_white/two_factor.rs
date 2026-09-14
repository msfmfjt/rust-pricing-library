//! Exact joint OU innovations for two-factor Bergomi and Hull–White.
//! The first four coordinates retain the existing HW ordering; the second
//! volatility innovation is appended so the one-factor limit keeps its draws.
use super::*;
use crate::models::Bergomi2Factor;
use crate::models::hull_white::b;

pub const BERGOMI_TWO_FACTOR_HW_SCHEME: &str = "bergomi-two-factor-hw-joint-gaussian-log-euler-v1";

#[derive(Clone, Debug)]
pub struct Bergomi2FactorHullWhiteDriverPlan {
    model: Bergomi2Factor,
    times: Box<[f64]>,
    decays: Box<[[f64; 2]]>,
    loadings: Box<[[[f64; 5]; 5]]>,
}

impl Bergomi2FactorHullWhiteDriverPlan {
    pub fn new(
        model: Bergomi2Factor,
        rates: &HullWhite1Factor,
        equity_rate: f64,
        vol_rate: [f64; 2],
        grid: &LocalVolTimeGrid,
    ) -> Result<Self, HullWhiteMcError> {
        Self::compile(model, rates, equity_rate, vol_rate, grid.nodes())
    }

    pub(in crate::engine) fn compile(
        model: Bergomi2Factor,
        rates: &HullWhite1Factor,
        equity_rate: f64,
        vol_rate: [f64; 2],
        times: &[f64],
    ) -> Result<Self, HullWhiteMcError> {
        let mut loadings = Vec::new();
        let mut decays = Vec::new();
        for pair in times.windows(2) {
            loadings.push(covariance_loading(Self::covariance(
                model,
                rates,
                equity_rate,
                vol_rate,
                pair[0],
                pair[1],
            )?)?);
            decays.push(
                model
                    .mean_reversions()
                    .map(|k| (-k * (pair[1] - pair[0])).exp()),
            );
        }
        Ok(Self {
            model,
            times: times.into(),
            decays: decays.into(),
            loadings: loadings.into(),
        })
    }

    /// Covariance order: [dW_S, OU_V1, OU_r, integrated_OU_r, OU_V2].
    /// All six Brownian correlations are explicit and jointly PSD checked.
    pub fn covariance(
        model: Bergomi2Factor,
        rates: &HullWhite1Factor,
        equity_rate: f64,
        vol_rate: [f64; 2],
        start: f64,
        end: f64,
    ) -> Result<[[f64; 5]; 5], HullWhiteMcError> {
        let [s1, s2] = model.spot_correlations();
        let v12 = model.factor_correlation();
        let [r1, r2] = vol_rate;
        let brownian = [
            [1.0, s1, equity_rate, s2],
            [s1, 1.0, r1, v12],
            [equity_rate, r1, 1.0, r2],
            [s2, v12, r2, 1.0],
        ];
        if brownian
            .iter()
            .flatten()
            .any(|v| !v.is_finite() || v.abs() > 1.0)
        {
            return Err(HullWhiteError::InvalidCorrelation.into());
        }
        covariance_loading(brownian)?;
        let [k1, k2] = model.mean_reversions();
        let c1 = rates
            .transition(start, end, k1, HybridCorrelation::new(s1, equity_rate, r1)?)?
            .covariance;
        let c2 = rates
            .transition(start, end, k2, HybridCorrelation::new(s2, equity_rate, r2)?)?
            .covariance;
        let mut c = [[0.0; 5]; 5];
        for i in 0..4 {
            c[i][..4].copy_from_slice(&c1[i]);
        }
        c[4] = [
            c2[1][0],
            v12 * b(k1 + k2, end - start),
            c2[1][2],
            c2[1][3],
            c2[1][1],
        ];
        let last = c[4];
        for (i, row) in c.iter_mut().take(4).enumerate() {
            row[4] = last[i];
        }
        Ok(c)
    }

    /// The weighted OU state used by exp(nu * state), with zero initial factors.
    pub fn evolve(&self, shocks: &[f64]) -> Result<Vec<f64>, HullWhiteMcError> {
        let n = self.decays.len();
        if shocks.len() != 5 * n || shocks.iter().any(|v| !v.is_finite()) {
            return Err(invalid("two_factor_hw_shocks", shocks.len()));
        }
        let weights = self.model.normalized_weights();
        let mut x = [0.0; 2];
        let mut values = vec![0.0];
        for i in 0..n {
            for (factor, row) in [1, 4].into_iter().enumerate() {
                let noise: f64 = (0..=row)
                    .map(|j| self.loadings[i][row][j] * shocks[j * n + i])
                    .sum();
                x[factor] = self.decays[i][factor] * x[factor] + noise;
            }
            let value = weights[0] * x[0] + weights[1] * x[1];
            hw_valid(value, "two_factor_hw_state", i, false)?;
            values.push(value);
        }
        Ok(values)
    }

    pub fn times(&self) -> &[f64] {
        &self.times
    }
}
