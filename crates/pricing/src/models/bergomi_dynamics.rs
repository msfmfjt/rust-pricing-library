//! Sealed kernels shared by the one- and two-factor LSV algorithms.
use super::{
    Bergomi1Factor, Bergomi2Factor, Bergomi2FactorTransition, BergomiError, BergomiTransition,
};
use std::fmt::Debug;

mod sealed {
    pub trait Sealed {}
    impl Sealed for super::Bergomi1Factor {}
    impl Sealed for super::Bergomi2Factor {}
}

/// Implemented by the supported Bergomi factors. This sealed interface keeps
/// the particle calibration and its reverse identical across factor counts.
pub trait BergomiDynamics: sealed::Sealed + Copy + Debug + PartialEq + Send + Sync {
    type State: Copy + Default + Debug + PartialEq + Send + Sync;
    type Transition: Copy + Debug + Send + Sync;
    const FACTOR_COUNT: usize;
    fn vol_of_vol(self) -> f64;
    fn parameters(self) -> Vec<f64>;
    fn transition(self, dt: f64) -> Result<Self::Transition, BergomiError>;
    fn multiplier(self, x: Self::State) -> f64;
    fn multiplier_squared(self, x: Self::State) -> f64;
    fn finite(x: Self::State) -> bool;
    fn evolve(self, t: Self::Transition, x: Self::State, z: f64, orth: [f64; 2]) -> Self::State;
    fn evolve_ou(self, t: Self::Transition, x: Self::State, innovations: [f64; 2]) -> Self::State;
    /// Return the previous factor adjoint and adjoints of supplied Gaussian
    /// coordinates. External innovations have an identity loading to the OU state.
    fn reverse_factor(
        self,
        t: Self::Transition,
        next: Self::State,
        vbar: f64,
        variance: f64,
        external: bool,
    ) -> (Self::State, f64, [f64; 2]);
}
impl BergomiDynamics for Bergomi1Factor {
    type State = f64;
    type Transition = BergomiTransition;
    const FACTOR_COUNT: usize = 1;
    fn vol_of_vol(self) -> f64 {
        self.vol_of_vol()
    }
    fn parameters(self) -> Vec<f64> {
        vec![self.mean_reversion(), self.vol_of_vol(), self.correlation()]
    }
    fn transition(self, dt: f64) -> Result<Self::Transition, BergomiError> {
        Ok(self.transition(dt)?)
    }
    fn multiplier(self, x: f64) -> f64 {
        (self.vol_of_vol() * x).exp()
    }
    fn multiplier_squared(self, x: f64) -> f64 {
        (2.0 * self.vol_of_vol() * x).exp()
    }
    fn finite(x: f64) -> bool {
        x.is_finite()
    }
    fn evolve(self, t: BergomiTransition, x: f64, z: f64, orth: [f64; 2]) -> f64 {
        t.evolve(x, z, orth[0])
    }
    fn evolve_ou(self, t: BergomiTransition, x: f64, v: [f64; 2]) -> f64 {
        t.decay * x + v[0]
    }
    fn reverse_factor(
        self,
        t: BergomiTransition,
        next: f64,
        vbar: f64,
        variance: f64,
        external: bool,
    ) -> (f64, f64, [f64; 2]) {
        (
            next * t.decay + vbar * 2.0 * self.vol_of_vol() * variance,
            next * if external { 0.0 } else { t.spot_loading },
            [
                next * if external { 1.0 } else { t.orthogonal_loading },
                0.0,
            ],
        )
    }
}
impl BergomiDynamics for Bergomi2Factor {
    type State = [f64; 2];
    type Transition = Bergomi2FactorTransition;
    const FACTOR_COUNT: usize = 2;
    fn vol_of_vol(self) -> f64 {
        self.vol_of_vol()
    }
    fn parameters(self) -> Vec<f64> {
        let k = self.mean_reversions();
        let r = self.spot_correlations();
        vec![
            k[0],
            k[1],
            self.vol_of_vol(),
            self.mixing_weight(),
            r[0],
            r[1],
            self.factor_correlation(),
        ]
    }
    fn transition(self, dt: f64) -> Result<Self::Transition, BergomiError> {
        self.transition(dt)
    }
    fn multiplier(self, x: [f64; 2]) -> f64 {
        let w = self.normalized_weights();
        (self.vol_of_vol() * (w[0] * x[0] + w[1] * x[1])).exp()
    }
    fn multiplier_squared(self, x: [f64; 2]) -> f64 {
        let w = self.normalized_weights();
        (2.0 * self.vol_of_vol() * (w[0] * x[0] + w[1] * x[1])).exp()
    }
    fn finite(x: [f64; 2]) -> bool {
        x.iter().all(|x| x.is_finite())
    }
    fn evolve(self, t: Self::Transition, x: [f64; 2], z: f64, orth: [f64; 2]) -> [f64; 2] {
        t.evolve(x, [z, orth[0], orth[1]])
    }
    fn evolve_ou(self, t: Self::Transition, x: [f64; 2], v: [f64; 2]) -> [f64; 2] {
        std::array::from_fn(|i| t.decay[i] * x[i] + v[i])
    }
    fn reverse_factor(
        self,
        t: Self::Transition,
        next: [f64; 2],
        vbar: f64,
        variance: f64,
        external: bool,
    ) -> ([f64; 2], f64, [f64; 2]) {
        let w = self.normalized_weights();
        let prev = std::array::from_fn(|i| {
            next[i] * t.decay[i] + vbar * 2.0 * self.vol_of_vol() * w[i] * variance
        });
        if external {
            (prev, 0.0, next)
        } else {
            (
                prev,
                next[0] * t.lower[1][0] + next[1] * t.lower[2][0],
                [
                    next[0] * t.lower[1][1] + next[1] * t.lower[2][1],
                    next[1] * t.lower[2][2],
                ],
            )
        }
    }
}
