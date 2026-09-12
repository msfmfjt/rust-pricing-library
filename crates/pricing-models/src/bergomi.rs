use pricing_core::{CoreError, NonNegativeF64};

/// One-factor Bergomi volatility multiplier `a(t) = exp(nu * X(t))`,
/// `dX = -k X dt + dW_vol`, with `corr(dW_spot, dW_vol) = rho`.
/// Deterministic normalization of `a` cancels in particle-calibrated LSV.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bergomi1Factor {
    mean_reversion: NonNegativeF64,
    vol_of_vol: NonNegativeF64,
    correlation: f64,
}

impl Bergomi1Factor {
    pub fn new(mean_reversion: f64, vol_of_vol: f64, correlation: f64) -> Result<Self, CoreError> {
        if !correlation.is_finite() || !(-1.0..=1.0).contains(&correlation) {
            return Err(CoreError::NumberNotPositive {
                field: "bergomi_one_minus_absolute_correlation",
                bits: (1.0 - correlation.abs()).to_bits(),
            });
        }
        Ok(Self {
            mean_reversion: NonNegativeF64::new(mean_reversion, "bergomi_mean_reversion")?,
            vol_of_vol: NonNegativeF64::new(vol_of_vol, "bergomi_vol_of_vol")?,
            correlation,
        })
    }

    #[must_use]
    pub const fn mean_reversion(self) -> f64 {
        self.mean_reversion.get()
    }

    #[must_use]
    pub const fn vol_of_vol(self) -> f64 {
        self.vol_of_vol.get()
    }

    #[must_use]
    pub const fn correlation(self) -> f64 {
        self.correlation
    }

    /// Exact joint law of an OU increment and the spot Brownian increment.
    /// Reusing `rho` as their Gaussian correlation is incorrect when `k > 0`.
    pub fn transition(self, dt: f64) -> Result<BergomiTransition, CoreError> {
        let dt = pricing_core::PositiveF64::new(dt, "bergomi_delta_t")?.get();
        let kd = self.mean_reversion() * dt;
        let decay = (-kd).exp();
        // Stable limits also avoid 0/0 when k*dt underflows.
        let phi1 = if kd == 0.0 { 1.0 } else { -(-kd).exp_m1() / kd };
        let phi2 = if kd == 0.0 {
            1.0
        } else {
            -(-2.0 * kd).exp_m1() / (2.0 * kd)
        };
        let variance = dt * phi2;
        let covariance = self.correlation * dt * phi1;
        let spot_loading = covariance / dt.sqrt();
        let residual_variance = (variance - spot_loading * spot_loading).max(0.0);
        if !variance.is_finite() || variance <= 0.0 || !covariance.is_finite() {
            return Err(CoreError::NumberNotPositive {
                field: "bergomi_ou_increment_variance",
                bits: variance.to_bits(),
            });
        }
        Ok(BergomiTransition {
            decay,
            variance,
            covariance,
            spot_loading,
            orthogonal_loading: residual_variance.sqrt(),
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BergomiTransition {
    pub decay: f64,
    pub variance: f64,
    /// Covariance of the OU innovation with `sqrt(dt) * z_spot`.
    pub covariance: f64,
    pub spot_loading: f64,
    pub orthogonal_loading: f64,
}

impl BergomiTransition {
    #[must_use]
    pub fn evolve(self, x: f64, z_spot: f64, z_orthogonal: f64) -> f64 {
        self.decay * x + self.spot_loading * z_spot + self.orthogonal_loading * z_orthogonal
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_ou_moments_and_brownian_limit() {
        for k in [0.0, 1e-12, 3.0] {
            let t = Bergomi1Factor::new(k, 1.0, -0.86)
                .unwrap()
                .transition(0.25)
                .unwrap();
            let actual = t.spot_loading.powi(2) + t.orthogonal_loading.powi(2);
            assert!((actual - t.variance).abs() < 1e-15);
            assert!((t.spot_loading * 0.5 - t.covariance).abs() < 1e-15);
            if k == 0.0 {
                assert_eq!(t.variance, 0.25);
                assert_eq!(t.covariance, -0.215);
            }
        }
    }

    #[test]
    fn parameter_boundaries_are_explicit() {
        for rho in [-1.0, 0.0, 1.0] {
            assert!(
                Bergomi1Factor::new(0.0, 0.0, rho)
                    .unwrap()
                    .transition(0.1)
                    .is_ok()
            );
        }
        for bad in [-0.1, f64::NAN, f64::INFINITY] {
            assert!(Bergomi1Factor::new(bad, 1.0, 0.0).is_err());
            assert!(Bergomi1Factor::new(1.0, bad, 0.0).is_err());
        }
        assert!(Bergomi1Factor::new(1.0, 1.0, 1.01).is_err());
    }
}
