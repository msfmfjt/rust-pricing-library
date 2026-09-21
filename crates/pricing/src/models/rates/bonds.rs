use super::{CenteredRateState, RateDiscount};

/// Rate-state loading after the deterministic/convexity amount is factored out.
#[derive(Clone, Copy, Debug)]
pub(crate) struct BondStateLoading {
    duration: f64,
}

impl BondStateLoading {
    pub fn new(duration: f64) -> Self {
        Self { duration }
    }
    pub fn duration(self) -> f64 {
        self.duration
    }
    pub fn multiplier(self, factor: f64) -> f64 {
        (-self.duration * factor).exp()
    }
    pub fn rate_derivative(self, amount: f64, factor: f64) -> f64 {
        -self.duration * amount * self.multiplier(factor)
    }
    pub fn rate_second_derivative(self, amount: f64, factor: f64) -> f64 {
        self.duration * self.duration * amount * self.multiplier(factor)
    }
}

/// Conditional Gaussian bond moments, before applying the integrated convexity
/// shift or the payment-date total variance. Keep these compile steps separate
/// so callers can retain their curve-validation order.
pub(crate) struct GaussianBondTransition {
    pub loading: BondStateLoading,
    pub integral_variance: f64,
}

impl GaussianBondTransition {
    pub fn conditional_discount(&self, payment_variance: f64) -> GaussianConditionalDiscount {
        GaussianConditionalDiscount {
            loading: self.loading,
            log_constant: -0.5 * payment_variance + 0.5 * self.integral_variance,
        }
    }
}

/// P(t,T) divided by the initial curve ratio. Do not precombine the shift and
/// variance: the public HW adapter retains its historical arithmetic order.
pub(crate) struct GaussianBond {
    pub loading: BondStateLoading,
    pub integrated_shift: f64,
    pub integral_variance: f64,
}

impl GaussianBond {
    pub fn relative_price(&self, factor: f64) -> f64 {
        (-self.loading.duration() * factor - self.integrated_shift + 0.5 * self.integral_variance)
            .exp()
    }
}

/// Relative conditional discount to a payment after the last observation.
/// This preserves the multi-asset combined-exponential convention. Single-asset
/// discount times bond pricing keeps its separate exponentials in its adapter.
#[derive(Clone, Copy, Debug)]
pub(crate) struct GaussianConditionalDiscount {
    pub loading: BondStateLoading,
    pub log_constant: f64,
}

impl RateDiscount for GaussianConditionalDiscount {
    type State = CenteredRateState;
    fn relative_discount(&self, state: CenteredRateState) -> f64 {
        (self.log_constant - state.integral - self.loading.duration() * state.factor).exp()
    }
}
