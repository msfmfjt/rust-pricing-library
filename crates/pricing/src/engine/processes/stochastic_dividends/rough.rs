//! Joint f/dividend/variance Brownian law plus an exact newest Volterra cell.
//! Older cells reuse the library's L2-average hybrid weights on the actual grid.
//! The log-variance centering uses that grid's variance, not t^(2H).
pub(super) mod correlation_reverse;
pub(super) mod reverse;
use super::*;
use crate::models::BERGOMI_TWO_FACTOR_CORRELATION_TOLERANCES;
use pricing_numerics::{CorrelationFactor, NeumaierSum};

pub(super) const SCHEME: &str = "buehler-rough-bergomi-joint-hybrid-positive-split-v1";
pub(super) const LSV_SCHEME: &str =
    "buehler-rough-bergomi-residual-lsv-joint-hybrid-positive-split-v1";
// A dense triangular history with 4096 steps needs about 64 MiB of scalar weights.
// Reject before constructing it; do not silently switch to a Markovian surrogate.
const MAX_STEPS: usize = 4096;

#[derive(Clone, Debug)]
pub(super) struct RoughDividendKernel {
    factor: RoughBergomi,
    dividend_volatility_correlation: f64,
    vol_loading: [f64; 3],
    weights: Box<[Box<[f64]>]>,
    variances: Box<[f64]>,
    steps: Box<[NearCell]>,
    leverage: Option<LsvLeverageSurface>,
}
#[derive(Clone, Copy, Debug)]
struct NearCell {
    root_dt: f64,
    average: f64,
    residual: f64,
}
impl NearCell {
    fn compile(factor: RoughBergomi, dt: f64) -> Result<Self, StochasticDividendError> {
        let h = factor.hurst();
        let p = h + 0.5;
        // Residual variance fraction = (1/2-H)^2/(H+1/2)^2.
        // This avoids subtracting nearly equal values as H tends to 1/2.
        let value = Self {
            root_dt: dt.sqrt(),
            average: (2.0 * h).sqrt() * dt.powf(h - 0.5) / p,
            residual: dt.powf(h) * (0.5 - h) / p,
        };
        positive(value.root_dt, "rough_step_length")?;
        positive(value.average, "rough_near_average")?;
        nonnegative(value.residual, "rough_near_residual")?;
        Ok(value)
    }
}
impl RoughDividendKernel {
    pub(super) fn compile(
        dividend: BuehlerDividendModel,
        factor: RoughBergomi,
        rho_dv: f64,
        times: &[f64],
    ) -> Result<Self, StochasticDividendError> {
        if times.len() < 2 || times.len() - 1 > MAX_STEPS {
            return Err(invalid("rough_grid_step_limit_4096"));
        }
        let sd = dividend.equity_dividend_correlation();
        let sv = factor.correlation();
        let matrix = vec![
            vec![1.0, sd, sv],
            vec![sd, 1.0, rho_dv],
            vec![sv, rho_dv, 1.0],
        ];
        // Enforce the full instantaneous PSD law, even at zero eta/sigma/nu_D.
        let correlation =
            CorrelationFactor::compile(matrix, BERGOMI_TWO_FACTOR_CORRELATION_TOLERANCES)
                .map_err(|_| invalid("rough_equity_dividend_volatility_correlation_matrix"))?;
        let vol_loading = [
            correlation.lower()[6],
            correlation.lower()[7],
            correlation.lower()[8],
        ];
        let (weights, variances) = factor
            .volterra_weights(times)
            .map_err(|_| invalid("rough_volterra_weights"))?;
        let steps = times
            .windows(2)
            .map(|w| NearCell::compile(factor, w[1] - w[0]))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            factor,
            dividend_volatility_correlation: rho_dv,
            vol_loading,
            weights: weights.into_boxed_slice(),
            variances: variances.into_boxed_slice(),
            steps: steps.into_boxed_slice(),
            leverage: None,
        })
    }

    pub(super) fn with_leverage(
        mut self,
        surface: LsvLeverageSurface,
        times: &[f64],
    ) -> Result<Self, StochasticDividendError> {
        crate::engine::processes::lsv::step_rows(&surface, times)
            .map_err(|_| invalid("rough_lsv_leverage_surface"))?;
        self.leverage = Some(surface);
        Ok(self)
    }

    pub(super) fn is_lsv(&self) -> bool {
        self.leverage.is_some()
    }

    pub(super) fn lsv_surface(&self) -> Option<&LsvLeverageSurface> {
        self.leverage.as_ref()
    }

    pub(super) fn scheme(&self) -> &'static str {
        if self.leverage.is_some() {
            LSV_SCHEME
        } else {
            SCHEME
        }
    }

    pub(super) fn hash_parameters(&self, hash: &mut blake3::Hasher) {
        hash.update(self.scheme().as_bytes());
        for x in [
            self.factor.hurst(),
            self.factor.vol_of_vol(),
            self.factor.correlation(),
            self.dividend_volatility_correlation,
        ] {
            hash.update(&x.to_bits().to_le_bytes());
        }
        if let Some(surface) = &self.leverage {
            hash.update(b"residual-lsv-squared-leverage-v1");
            hash.update(&surface.initial_f().to_bits().to_le_bytes());
            for &x in surface.times() {
                hash.update(&x.to_bits().to_le_bytes());
            }
            for &x in surface.log_nodes() {
                hash.update(&x.to_bits().to_le_bytes());
            }
            for &x in surface.squared_leverage() {
                hash.update(&x.to_bits().to_le_bytes());
            }
        }
    }
    fn innovations(&self, step: usize, z: &[f64; 4]) -> (f64, f64) {
        let s = self.steps[step];
        let dw = s.root_dt
            * self
                .vol_loading
                .iter()
                .zip(z)
                .map(|(l, z)| l * z)
                .sum::<f64>();
        (dw, s.average * dw + s.residual * z[3])
    }
    // Replay the very same causal history and compensated-sum order as evolve.
    // Risk never reconstructs it from rounded physical-stock observations.
    fn driver_path(
        &self,
        normals: &[f64],
    ) -> Result<(Vec<f64>, Vec<f64>), StochasticDividendError> {
        if normals.len() != 4 * self.steps.len() || normals.iter().any(|z| !z.is_finite()) {
            return Err(invalid("rough_reverse_normals"));
        }
        let mut increments = Vec::with_capacity(self.steps.len());
        let mut drivers = Vec::with_capacity(self.steps.len() + 1);
        drivers.push(0.0);
        for (i, z) in normals.as_chunks::<4>().0.iter().enumerate() {
            let (dw, near) = self.innovations(i, z);
            let mut sum = NeumaierSum::new();
            sum.add(near);
            for (&weight, &dw) in self.weights[i + 1].iter().zip(&increments) {
                sum.add(weight * dw);
            }
            let driver = sum.total();
            if !driver.is_finite() {
                return Err(invalid("rough_reverse_history"));
            }
            drivers.push(driver);
            increments.push(dw);
        }
        Ok((drivers, increments))
    }

    pub(super) fn volatility_loadings(
        &self,
        normals: &[f64],
    ) -> Result<Vec<f64>, StochasticDividendError> {
        let (drivers, _) = self.driver_path(normals)?;
        let eta = self.factor.vol_of_vol();
        drivers[..self.steps.len()]
            .iter()
            .zip(self.variances.iter())
            .map(|(&x, &v)| {
                let value = (0.5 * eta * x - 0.25 * eta * eta * v).exp();
                positive(value, "rough_reverse_volatility_loading")?;
                Ok(value)
            })
            .collect()
    }

    pub(super) fn evolve(
        &self,
        model: BuehlerDividendModel,
        sigma0: f64,
        times: &[f64],
        normals: &[f64],
    ) -> Result<Vec<BuehlerDividendState>, StochasticDividendError> {
        let mut increments = Vec::with_capacity(self.steps.len());
        let mut state = BuehlerDividendState::initial();
        let mut states = Vec::with_capacity(times.len());
        states.push(state);
        let mut driver = 0.0;
        let eta = self.factor.vol_of_vol();
        for (i, z) in normals.as_chunks::<4>().0.iter().enumerate() {
            // Use the PREVIOUS node's Volterra history. The next history is
            // built only after evolving f/Y, so no same-step look-ahead occurs.
            let multiplier = (0.5 * eta * driver - 0.25 * eta * eta * self.variances[i]).exp();
            positive(multiplier, "rough_volatility_multiplier")?;
            let sigma = if let Some(surface) = &self.leverage {
                let residual_f = surface.initial_f() * state.equity();
                positive(residual_f, "rough_lsv_residual_equity")?;
                let leverage_squared = surface
                    .squared_leverage_at(times[i], residual_f)
                    .map_err(|_| invalid("rough_lsv_leverage_lookup"))?;
                let sigma = leverage_squared.sqrt() * multiplier;
                positive(sigma, "rough_lsv_equity_volatility")?;
                sigma
            } else if sigma0 == 0.0 || eta == 0.0 {
                sigma0
            } else {
                let sigma = sigma0 * multiplier;
                positive(sigma, "rough_equity_volatility")?;
                sigma
            };
            state = model.evolve(state, sigma, times[i + 1] - times[i], [z[0], z[1]])?;
            states.push(state);
            let (dw, near) = self.innovations(i, z);
            let mut sum = NeumaierSum::new();
            sum.add(near);
            for (&weight, &dw) in self.weights[i + 1].iter().zip(&increments) {
                sum.add(weight * dw);
            }
            driver = sum.total();
            if !driver.is_finite() {
                return Err(invalid("rough_volterra_history"));
            }
            increments.push(dw);
        }
        Ok(states)
    }

    pub(super) fn lsv_leverage_pullback(
        &self,
        model: BuehlerDividendModel,
        times: &[f64],
        normals: &[f64],
        states: &[BuehlerDividendState],
        equity_seeds: &[f64],
        dividend_seeds: &[f64],
    ) -> Result<Vec<f64>, StochasticDividendError> {
        let surface = self
            .leverage
            .as_ref()
            .ok_or(StochasticDividendError::Unsupported {
                feature: "local-variance risk requires a rough residual-equity LSV plan",
            })?;
        let n = self.steps.len();
        if times.len() != n + 1
            || states.len() != n + 1
            || equity_seeds.len() != n + 1
            || dividend_seeds.len() != n + 1
            || normals.len() != 4 * n
        {
            return Err(invalid("rough_lsv_leverage_reverse_shape"));
        }
        if equity_seeds
            .iter()
            .chain(dividend_seeds)
            .any(|x| !x.is_finite())
        {
            return Err(invalid("rough_lsv_leverage_reverse_seed"));
        }

        // The rough Volterra driver depends only on the fixed Brownian history,
        // not on the Local-variance target. Replay it to recover left-endpoint
        // volatility multipliers while reversing only the Buehler state/leverage
        // dependence.
        let (drivers, _) = self.driver_path(normals)?;
        let eta = self.factor.vol_of_vol();
        let mut leverage_bar = vec![0.0; surface.squared_leverage().len()];
        let mut f_bar = equity_seeds[n];
        let mut y_bar = dividend_seeds[n];
        let alpha = model.equity_linkage();
        let rho = model.equity_dividend_correlation();
        let rho_root = ((1.0 - rho) * (1.0 + rho)).sqrt();

        for i in (1..=n).rev() {
            let step_index = i - 1;
            let old = states[step_index];
            let new = states[i];
            let dt = times[i] - times[step_index];
            let root = dt.sqrt();
            let z = &normals[4 * step_index..4 * i];

            let (a, b) = super::decay(model.mean_reversion(), 0.5 * dt);
            let next_f_bar = f_bar + b * alpha * y_bar;
            let v = model.dividend_volatility() * root;
            let z_dividend = rho * z[0] + rho_root * z[1];
            let dividend_exponential = (-0.5 * v * v + v * z_dividend).exp();
            let half_bar = a * y_bar * dividend_exponential;

            let multiplier = (0.5 * eta * drivers[step_index]
                - 0.25 * eta * eta * self.variances[step_index])
                .exp();
            positive(multiplier, "rough_lsv_reverse_volatility_multiplier")?;
            let residual_f = surface.initial_f() * old.equity;
            let lookup = surface
                .lookup(times[step_index], residual_f)
                .map_err(|_| invalid("rough_lsv_leverage_lookup"))?;
            let sigma = lookup.value.sqrt() * multiplier;
            positive(sigma, "rough_lsv_reverse_equity_volatility")?;
            let u = sigma * root;

            let sigma_bar = next_f_bar * new.equity * (z[0] - u) * root;
            let l_bar = sigma_bar * sigma / (2.0 * lookup.value);
            lookup.transpose(l_bar, &mut leverage_bar);
            let lookup_state_bar = l_bar * lookup.derivative_log_f / old.equity;

            let equity_exponential = new.equity / old.equity;
            f_bar = equity_seeds[step_index]
                + next_f_bar * equity_exponential
                + half_bar * b * alpha
                + lookup_state_bar;
            y_bar = dividend_seeds[step_index] + half_bar * a;
        }

        if leverage_bar.iter().any(|x| !x.is_finite()) {
            return Err(invalid("rough_lsv_leverage_reverse_result"));
        }
        Ok(leverage_bar)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn near_cell_joint_covariance_matches_independent_power_integrals() {
        for h in [0.01, 0.1, 0.49, 0.5] {
            for dt in [0.0001, 0.17, 1.3] {
                let sd = -0.25;
                let sv = -0.4;
                let dv = 0.15;
                let dividend = BuehlerDividendModel::new(0.7, 0.6, 0.35, sd).unwrap();
                let k = RoughDividendKernel::compile(
                    dividend,
                    RoughBergomi::new(h, 0.6, sv).unwrap(),
                    dv,
                    &[0.0, dt],
                )
                .unwrap();
                // Reconstruct full law on independent standard-normal basis.
                let mut rows = [[0.0; 4]; 4];
                rows[0][0] = dt.sqrt();
                rows[1][0] = sd * dt.sqrt();
                rows[1][1] = ((1.0 - sd) * (1.0 + sd)).sqrt() * dt.sqrt();
                let basis: [(f64, f64); 4] = std::array::from_fn(|j| {
                    let mut z = [0.0; 4];
                    z[j] = 1.0;
                    k.innovations(0, &z)
                });
                rows[2] = basis.map(|(dw, _)| dw);
                rows[3] = basis.map(|(_, near)| near);
                let c = |i: usize, j: usize| {
                    rows[i].iter().zip(rows[j]).map(|(x, y)| x * y).sum::<f64>()
                };
                let power = (2.0 * h).sqrt() * dt.powf(h + 0.5) / (h + 0.5);
                for (i, j, expected) in [
                    (0, 1, sd * dt),
                    (0, 2, sv * dt),
                    (1, 2, dv * dt),
                    (0, 3, sv * power),
                    (1, 3, dv * power),
                    (2, 3, power),
                    (2, 2, dt),
                    (3, 3, dt.powf(2.0 * h)),
                ] {
                    assert!(
                        (c(i, j) - expected).abs() < 2e-13,
                        "H={h} dt={dt} ({i},{j})"
                    );
                }
                if h == 0.5 {
                    assert_eq!(k.steps[0].residual, 0.0);
                }
            }
        }
    }
    #[test]
    fn discrete_history_variance_and_centering_match_normal_basis() {
        let times = [0.0, 0.07, 0.21, 0.64, 1.0];
        for h in [0.03, 0.1, 0.5] {
            let k = RoughDividendKernel::compile(
                BuehlerDividendModel::new(0.7, 0.6, 0.35, -0.25).unwrap(),
                RoughBergomi::new(h, 0.6, -0.4).unwrap(),
                0.15,
                &times,
            )
            .unwrap();
            for (i, &time) in times.iter().enumerate().skip(1) {
                let mut variance = 0.0;
                for cell in 0..i {
                    for j in 0..4 {
                        let mut z = [0.0; 4];
                        z[j] = 1.0;
                        let (dw, near) = k.innovations(cell, &z);
                        let coefficient = if cell == i - 1 {
                            near
                        } else {
                            k.weights[i][cell] * dw
                        };
                        variance += coefficient * coefficient;
                    }
                }
                assert!((variance - k.variances[i]).abs() < 3e-13);
                // E[exp(eta X - eta^2 V/2)] = 1 on this finite grid.
                let expected = (0.5 * 0.6_f64.powi(2) * (variance - k.variances[i])).exp();
                assert!((expected - 1.0).abs() < 2e-13);
                if h < 0.5 && i > 1 {
                    assert!(k.variances[i] < time.powf(2.0 * h));
                }
            }
        }
    }
}
