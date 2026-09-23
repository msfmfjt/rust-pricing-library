//! Internal rate capabilities. Evolution, discounting and exact Gaussian
//! covariance are separate so a future rate model need not support all three.
//! Deterministic forward/discount curves remain owned by the market layer.

mod bonds;
mod hull_white;
#[cfg(test)]
mod tests;

pub(crate) use bonds::{
    BondStateLoading, GaussianBond, GaussianBondTransition, GaussianConditionalDiscount,
};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct CenteredRateState {
    pub factor: f64,
    pub integral: f64,
}

/// Jointly sampled rate-state and integrated-rate innovations, not independent
/// standard normals. A Gaussian model may need two coordinates for one driver.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RateInnovations {
    pub factor: f64,
    pub integral: f64,
}

pub(crate) struct RateAdvance<S> {
    pub state: S,
    pub step_integral: f64,
    pub integrated_shift: f64,
}

pub(crate) trait RateEvolution {
    type State;
    type Innovations;
    fn advance(
        &self,
        state: Self::State,
        innovations: Self::Innovations,
    ) -> RateAdvance<Self::State>;
}

/// Relative to deterministic carry, there is no rate state or innovation.
/// This adapter does not remove any coordinates from a zero-volatility HW plan.
#[derive(Clone, Copy, Debug)]
pub(crate) struct DeterministicRates;

impl RateEvolution for DeterministicRates {
    type State = ();
    type Innovations = ();
    fn advance(&self, (): (), (): ()) -> RateAdvance<()> {
        RateAdvance {
            state: (),
            step_integral: 0.0,
            integrated_shift: 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct GaussianRateStep {
    decay: f64,
    integral_loading: f64,
    integrated_shift: f64,
}

pub(crate) struct RateStepAdjoints {
    pub state: CenteredRateState,
    pub innovations: RateInnovations,
}

impl GaussianRateStep {
    pub fn new(decay: f64, integral_loading: f64, integrated_shift: f64) -> Self {
        Self {
            decay,
            integral_loading,
            integrated_shift,
        }
    }
    /// The step-integral seed includes equity carry; the accumulated-integral
    /// seed includes future discounting. Preserve their original addition order.
    pub fn pullback(&self, next: CenteredRateState, step_integral: f64) -> RateStepAdjoints {
        let integral_bar = step_integral + next.integral;
        RateStepAdjoints {
            state: CenteredRateState {
                factor: integral_bar * self.integral_loading + next.factor * self.decay,
                integral: next.integral,
            },
            innovations: RateInnovations {
                factor: next.factor,
                integral: integral_bar,
            },
        }
    }
}

impl RateEvolution for GaussianRateStep {
    type State = CenteredRateState;
    type Innovations = RateInnovations;
    fn advance(
        &self,
        state: CenteredRateState,
        noise: RateInnovations,
    ) -> RateAdvance<CenteredRateState> {
        let integral = self.integral_loading * state.factor + noise.integral;
        RateAdvance {
            state: CenteredRateState {
                factor: self.decay * state.factor + noise.factor,
                integral: state.integral + integral,
            },
            step_integral: integral,
            integrated_shift: self.integrated_shift,
        }
    }
}

/// Relative discounting only; the caller applies the deterministic curve and
/// retains its existing validation and curve-adjoint accumulation policy.
pub(crate) trait RateDiscount {
    type State;
    fn relative_discount(&self, state: Self::State) -> f64;
}

impl RateDiscount for DeterministicRates {
    type State = ();
    fn relative_discount(&self, (): ()) -> f64 {
        1.0
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct GaussianDiscount {
    half_variance: f64,
}

impl GaussianDiscount {
    pub fn new(integrated_variance: f64) -> Self {
        Self {
            half_variance: 0.5 * integrated_variance,
        }
    }
}

impl RateDiscount for GaussianDiscount {
    type State = f64;
    fn relative_discount(&self, integrated_factor: f64) -> f64 {
        (-integrated_factor - self.half_variance).exp()
    }
}

pub(crate) struct RateCovariance {
    pub factor_variance: f64,
    pub integral_variance: f64,
    pub factor_integral: f64,
}

pub(crate) struct OuRateCovariance {
    pub factor: f64,
    pub integral: f64,
}

/// Optional exact Gaussian capability, separate from rate evolution itself.
/// Joint sampling, covariance assembly and PSD factorization stay with the
/// composition driver. OU cross moments use unit Brownian correlation.
pub(crate) trait GaussianRateCovariance {
    type Error;
    fn rate_covariance(&self, start: f64, end: f64) -> Result<RateCovariance, Self::Error>;
    fn ou_rate_covariance(
        &self,
        start: f64,
        end: f64,
        reversion: f64,
    ) -> Result<OuRateCovariance, Self::Error>;
}
