//! Joint cash-dividend Brownian and exact Bergomi OU innovations.
//! Covariances are integrated before factorization, not set equal to the
//! instantaneous Brownian correlations. Only compilation allocates kernels.

pub(super) mod correlation_reverse;
pub(super) mod parameter_reverse;

use super::*;
use crate::mc::lsv::LsvLeverageSurface;
use crate::models::{BERGOMI_TWO_FACTOR_CORRELATION_TOLERANCES, ou_kernel_correlation};
use pricing_numerics::CorrelationFactor;

#[derive(Clone, Debug)]
pub(super) enum BergomiDividendKernel {
    One(Kernel<1, 3>),
    Two(Kernel<2, 4>),
}

#[derive(Clone, Debug)]
pub(super) struct Kernel<const N: usize, const D: usize> {
    weights: [f64; N],
    vol_of_vol: f64,
    steps: Box<[Step<N, D>]>,
    centering: Box<[f64]>,
    identity: Box<[f64]>,
    /// Optional calibrated squared leverage in the funded residual-equity coordinate.
    /// `None` preserves the original pure-Bergomi arithmetic bit for bit.
    leverage: Option<LsvLeverageSurface>,
}
#[derive(Clone, Debug)]
struct Step<const N: usize, const D: usize> {
    decay: [f64; N],
    lower: [[f64; D]; N],
}

impl BergomiDividendKernel {
    pub(super) fn one(
        dividend: BuehlerDividendModel,
        factor: Bergomi1Factor,
        rho_dv: f64,
        times: &[f64],
    ) -> Result<Self, StochasticDividendError> {
        let sd = dividend.equity_dividend_correlation();
        let sv = factor.correlation();
        let matrix = [[1.0, sd, sv], [sd, 1.0, rho_dv], [sv, rho_dv, 1.0]];
        Ok(Self::One(Kernel::compile(
            [factor.mean_reversion()],
            [1.0],
            factor.vol_of_vol(),
            matrix,
            times,
            vec![factor.mean_reversion(), factor.vol_of_vol(), sv, rho_dv],
        )?))
    }
    pub(super) fn two(
        dividend: BuehlerDividendModel,
        factor: Bergomi2Factor,
        rho_dv: [f64; 2],
        times: &[f64],
    ) -> Result<Self, StochasticDividendError> {
        let sd = dividend.equity_dividend_correlation();
        let sv = factor.spot_correlations();
        let vv = factor.factor_correlation();
        let matrix = [
            [1.0, sd, sv[0], sv[1]],
            [sd, 1.0, rho_dv[0], rho_dv[1]],
            [sv[0], rho_dv[0], 1.0, vv],
            [sv[1], rho_dv[1], vv, 1.0],
        ];
        let k = factor.mean_reversions();
        Ok(Self::Two(Kernel::compile(
            k,
            factor.normalized_weights(),
            factor.vol_of_vol(),
            matrix,
            times,
            vec![
                k[0],
                k[1],
                factor.vol_of_vol(),
                factor.mixing_weight(),
                sv[0],
                sv[1],
                vv,
                rho_dv[0],
                rho_dv[1],
            ],
        )?))
    }
    pub(super) fn with_leverage(
        mut self,
        surface: LsvLeverageSurface,
        times: &[f64],
    ) -> Result<Self, StochasticDividendError> {
        crate::engine::processes::lsv::step_rows(&surface, times)
            .map_err(|_| invalid("lsv_leverage_surface"))?;
        match &mut self {
            Self::One(k) => k.leverage = Some(surface),
            Self::Two(k) => k.leverage = Some(surface),
        }
        Ok(self)
    }

    pub(super) fn is_lsv(&self) -> bool {
        match self {
            Self::One(k) => k.leverage.is_some(),
            Self::Two(k) => k.leverage.is_some(),
        }
    }

    pub(super) fn lsv_surface(&self) -> Option<&LsvLeverageSurface> {
        match self {
            Self::One(k) => k.leverage.as_ref(),
            Self::Two(k) => k.leverage.as_ref(),
        }
    }

    pub(super) fn volatility_loadings(
        &self,
        z: &[f64],
    ) -> Result<Vec<f64>, StochasticDividendError> {
        if self.is_lsv() {
            return Err(StochasticDividendError::Unsupported {
                feature: "residual-equity LSV reverse; use price-only evaluation",
            });
        }
        match self {
            Self::One(k) => k.volatility_loadings(z),
            Self::Two(k) => k.volatility_loadings(z),
        }
    }
    pub(super) const fn factor_count(&self) -> usize {
        match self {
            Self::One(_) => 3,
            Self::Two(_) => 4,
        }
    }
    pub(super) fn scheme(&self) -> &'static str {
        match self {
            Self::One(k) if k.leverage.is_some() => {
                "buehler-bergomi-1f-residual-lsv-joint-ou-positive-split-v1"
            }
            Self::Two(k) if k.leverage.is_some() => {
                "buehler-bergomi-2f-residual-lsv-joint-ou-positive-split-v1"
            }
            Self::One(_) => "buehler-bergomi-1f-joint-ou-positive-split-v1",
            Self::Two(_) => "buehler-bergomi-2f-joint-ou-positive-split-v1",
        }
    }
    pub(super) fn hash_parameters(&self, hash: &mut blake3::Hasher) {
        hash.update(self.scheme().as_bytes());
        let parameters = match self {
            Self::One(k) => &k.identity,
            Self::Two(k) => &k.identity,
        };
        for x in parameters.iter() {
            hash.update(&x.to_bits().to_le_bytes());
        }
        if let Some(surface) = self.lsv_surface() {
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
    pub(super) fn evolve(
        &self,
        model: BuehlerDividendModel,
        sigma0: f64,
        times: &[f64],
        z: &[f64],
    ) -> Result<Vec<BuehlerDividendState>, StochasticDividendError> {
        match self {
            Self::One(k) => k.evolve(model, sigma0, times, z),
            Self::Two(k) => k.evolve(model, sigma0, times, z),
        }
    }
}

impl<const N: usize, const D: usize> Kernel<N, D> {
    fn compile(
        k: [f64; N],
        weights: [f64; N],
        vol_of_vol: f64,
        matrix: [[f64; D]; D],
        times: &[f64],
        identity: Vec<f64>,
    ) -> Result<Self, StochasticDividendError> {
        // Even when the factors have zero loading, require the underlying
        // Brownian model to be valid. Integrated covariance alone is not enough.
        CorrelationFactor::compile(
            matrix.iter().map(|r| r.to_vec()).collect(),
            BERGOMI_TWO_FACTOR_CORRELATION_TOLERANCES,
        )
        .map_err(|_| invalid("equity_dividend_volatility_correlation_matrix"))?;
        let steps = times
            .windows(2)
            .map(|t| Step::compile(k, matrix, t[1] - t[0]))
            .collect::<Result<Vec<_>, _>>()?;
        let centering = times[..times.len() - 1]
            .iter()
            .map(|&t| {
                if t == 0.0 {
                    return Ok(0.0);
                }
                let step = Step::<N, D>::compile(k, matrix, t)?;
                // A sum of squares stays nonnegative near singular factor mixtures.
                let variance: f64 = (0..D)
                    .map(|j| {
                        weights
                            .iter()
                            .enumerate()
                            .map(|(i, w)| w * step.lower[i][j])
                            .sum::<f64>()
                            .powi(2)
                    })
                    .sum();
                let shift = vol_of_vol * variance;
                nonnegative(shift, "bergomi_centering")?;
                Ok(shift)
            })
            .collect::<Result<Vec<_>, StochasticDividendError>>()?;
        Ok(Self {
            weights,
            vol_of_vol,
            steps: steps.into_boxed_slice(),
            centering: centering.into_boxed_slice(),
            identity: identity.into_boxed_slice(),
            leverage: None,
        })
    }
    // Identical OU recurrence to `evolve`; correlation/OU parameters are fixed
    // in this reverse scope. In particular sigma0=0 never requires sigma/sigma0.
    fn volatility_loadings(&self, normals: &[f64]) -> Result<Vec<f64>, StochasticDividendError> {
        let mut x = [0.0; N];
        let mut out = Vec::with_capacity(self.steps.len());
        for (i, (step, z)) in self
            .steps
            .iter()
            .zip(normals.as_chunks::<D>().0)
            .enumerate()
        {
            let factor: f64 = self.weights.iter().zip(x).map(|(w, x)| w * x).sum();
            let loading = (self.vol_of_vol * (factor - self.centering[i])).exp();
            positive(loading, "reverse_volatility_loading")?;
            out.push(loading);
            for (j, value) in x.iter_mut().enumerate() {
                *value = step.decay[j] * *value
                    + step.lower[j].iter().zip(z).map(|(l, z)| l * z).sum::<f64>();
                if !value.is_finite() {
                    return Err(invalid("bergomi_state"));
                }
            }
        }
        Ok(out)
    }
    fn evolve(
        &self,
        model: BuehlerDividendModel,
        sigma0: f64,
        times: &[f64],
        normals: &[f64],
    ) -> Result<Vec<BuehlerDividendState>, StochasticDividendError> {
        let mut x = [0.0; N];
        let mut state = BuehlerDividendState::initial();
        let mut states = Vec::with_capacity(times.len());
        states.push(state);
        for (i, (step, z)) in self
            .steps
            .iter()
            .zip(normals.as_chunks::<D>().0)
            .enumerate()
        {
            let factor: f64 = self.weights.iter().zip(x).map(|(w, x)| w * x).sum();
            let sigma = if let Some(surface) = &self.leverage {
                // The calibrated surface lives in the *funded residual-equity*
                // coordinate F_res = F_res(0) * f. It is deliberately not a
                // local-volatility fit to physical stock S=a*f+b*Y+c.
                let residual_f = surface.initial_f() * state.equity();
                positive(residual_f, "lsv_residual_equity")?;
                let leverage_squared = surface
                    .squared_leverage_at(times[i], residual_f)
                    .map_err(|_| invalid("lsv_leverage_lookup"))?;
                let multiplier = (self.vol_of_vol * (factor - self.centering[i])).exp();
                let sigma = leverage_squared.sqrt() * multiplier;
                positive(sigma, "lsv_equity_volatility")?;
                sigma
            } else if sigma0 == 0.0 {
                // Preserve the original pure-Bergomi zero-volatility branch:
                // do not evaluate the stochastic-volatility exponential.
                0.0
            } else {
                let sigma =
                    sigma0 * (self.vol_of_vol * (factor - self.centering[i])).exp();
                positive(sigma, "bergomi_volatility")?;
                sigma
            };
            // Freeze variance at the LEFT endpoint. Correlating the updated OU
            // state with this same equity increment would destroy its mean.
            state = model.evolve(state, sigma, times[i + 1] - times[i], [z[0], z[1]])?;
            for (j, value) in x.iter_mut().enumerate() {
                *value = step.decay[j] * *value
                    + step.lower[j].iter().zip(z).map(|(l, z)| l * z).sum::<f64>();
                if !value.is_finite() {
                    return Err(invalid("bergomi_state"));
                }
            }
            states.push(state);
        }
        Ok(states)
    }
}

impl<const N: usize, const D: usize> Step<N, D> {
    fn compile(
        k: [f64; N],
        matrix: [[f64; D]; D],
        dt: f64,
    ) -> Result<Self, StochasticDividendError> {
        positive(dt, "bergomi_step_length")?;
        let kernels: [f64; D] = std::array::from_fn(|i| if i < 2 { 0.0 } else { k[i - 2] * dt });
        let mut correlation = vec![vec![0.0; D]; D];
        for i in 0..D {
            for j in 0..D {
                correlation[i][j] = matrix[i][j]
                    * ou_kernel_correlation(kernels[i], kernels[j])
                        .map_err(|_| invalid("bergomi_ou_covariance"))?;
            }
        }
        let factor =
            CorrelationFactor::compile(correlation, BERGOMI_TWO_FACTOR_CORRELATION_TOLERANCES)
                .map_err(|_| invalid("integrated_equity_dividend_volatility_covariance"))?;
        let mut lower = [[0.0; D]; N];
        let mut decay = [0.0; N];
        for i in 0..N {
            let marginal = Bergomi1Factor::new(k[i], 0.0, 0.0)
                .and_then(|m| m.transition(dt))
                .map_err(|_| invalid("bergomi_ou_variance"))?;
            decay[i] = marginal.decay;
            for (j, value) in lower[i].iter_mut().enumerate() {
                *value = marginal.variance.sqrt() * factor.lower()[(i + 2) * D + j];
            }
        }
        Ok(Self { decay, lower })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn joint_ou_covariance_matches_direct_integrals() {
        for k in [0.0, 1e-12, 0.8, 20.0] {
            let rates = [0.0, 0.0, k, 2.1];
            let matrix = [
                [1.0, -0.25, -0.4, -0.2],
                [-0.25, 1.0, 0.15, -0.1],
                [-0.4, 0.15, 1.0, 0.3],
                [-0.2, -0.1, 0.3, 1.0],
            ];
            for dt in [0.03_f64, 0.5, 1.7] {
                let step = Step::<2, 4>::compile([k, 2.1], matrix, dt).unwrap();
                let mut l = [[0.0; 4]; 4];
                l[0][0] = dt.sqrt();
                l[1][0] = -0.25 * dt.sqrt();
                l[1][1] = (1.0_f64 - 0.25 * 0.25).sqrt() * dt.sqrt();
                l[2] = step.lower[0];
                l[3] = step.lower[1];
                for i in 0..4 {
                    for j in 0..4 {
                        // Independent composite Simpson integration of exp kernels.
                        let n = 2048;
                        let integral: f64 = (0..=n)
                            .map(|m| {
                                let weight = if m == 0 || m == n {
                                    1.0
                                } else if m % 2 == 0 {
                                    2.0
                                } else {
                                    4.0
                                };
                                weight * (-(rates[i] + rates[j]) * dt * m as f64 / n as f64).exp()
                            })
                            .sum::<f64>()
                            * dt
                            / (3 * n) as f64;
                        let actual: f64 = (0..4).map(|c| l[i][c] * l[j][c]).sum();
                        assert!(
                            (actual - matrix[i][j] * integral).abs() < 2e-10,
                            "k={k}, dt={dt}, ({i},{j}), {actual} vs {}",
                            matrix[i][j] * integral
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn centering_equals_unconditional_weighted_ou_variance() {
        let d = BuehlerDividendModel::new(0.7, 0.6, 0.35, -0.25).unwrap();
        let factor = Bergomi2Factor::new([0.8, 2.1], 0.3, 0.35, [-0.4, -0.2], 0.3).unwrap();
        let times = [0.0, 0.13, 0.5, 1.0];
        let BergomiDividendKernel::Two(kernel) =
            BergomiDividendKernel::two(d, factor, [0.15, -0.1], &times).unwrap()
        else {
            unreachable!()
        };
        let w = factor.normalized_weights();
        for (i, &t) in times[..3].iter().enumerate() {
            let b = |k: f64| -(-k * t).exp_m1() / k;
            let variance =
                w[0] * w[0] * b(1.6) + w[1] * w[1] * b(4.2) + 2.0 * w[0] * w[1] * 0.3 * b(2.9);
            assert!((kernel.centering[i] - 0.3 * variance).abs() < 2e-15);
        }
    }
    #[test]
    fn constant_residual_lsv_matches_flat_bergomi_volatility() {
        let dividend = BuehlerDividendModel::new(0.7, 0.6, 0.35, -0.25).unwrap();
        let factor = Bergomi1Factor::new(0.8, 0.0, -0.4).unwrap();
        let times = [0.0, 0.5, 1.0];
        let surface =
            LsvLeverageSurface::new(times.to_vec(), vec![-1.0, 1.0], vec![0.04; 6], 90.0).unwrap();
        let normals = [0.2, -0.7, 1.1, -0.3, 0.5, -1.4];

        let plain = BergomiDividendKernel::one(dividend, factor, 0.15, &times)
            .unwrap()
            .evolve(dividend, 0.2, &times, &normals)
            .unwrap();
        let lsv_kernel = BergomiDividendKernel::one(dividend, factor, 0.15, &times)
            .unwrap()
            .with_leverage(surface, &times)
            .unwrap();
        assert_eq!(
            lsv_kernel.scheme(),
            "buehler-bergomi-1f-residual-lsv-joint-ou-positive-split-v1"
        );
        let lsv = lsv_kernel.evolve(dividend, 0.0, &times, &normals).unwrap();

        for (a, b) in plain.iter().zip(lsv.iter()) {
            assert!((a.equity() - b.equity()).abs() < 2e-15);
            assert!((a.dividend() - b.dividend()).abs() < 2e-15);
        }
    }
}
