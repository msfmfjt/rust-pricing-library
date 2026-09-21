//! Test-only registration of an alternative state/transition representation.
//! Y = nu X has the same law as 1F Bergomi, but neither the state nor its
//! adjoint is the existing scalar X. No production model is registered.

use super::*;
use crate::lsv::BergomiLsvPricingPlan;
use crate::mc::ExecutionPolicy;
use crate::mc::lsv::LsvParticleConfig;
use crate::{JsonLimits, PricingRequest, parse_request_json};
use serde_json::{Value, json};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct LogState {
    value: f64,
}

#[derive(Clone, Copy, Debug)]
struct LogTransition {
    decay: f64,
    spot: f64,
    orthogonal: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct LogCoordinate(Bergomi1Factor);

impl sealed::Sealed for LogCoordinate {
    type Coordinates = [f64; 1];
    fn coordinates(read: impl FnMut(usize) -> f64) -> Self::Coordinates {
        std::array::from_fn(read)
    }
    fn evolve_normals(
        self,
        t: LogTransition,
        x: LogState,
        spot: f64,
        orthogonal: OrthogonalNormals<'_>,
    ) -> LogState {
        LogState {
            value: t.decay * x.value + t.spot * spot + t.orthogonal * orthogonal.get(0),
        }
    }
    fn evolve_increments(self, t: LogTransition, x: LogState, v: OuInnovations<'_>) -> LogState {
        LogState {
            value: t.decay * x.value + self.vol_of_vol() * v.get(0),
        }
    }
    fn pullback(
        self,
        t: LogTransition,
        next: LogState,
        vbar: f64,
        variance: f64,
        source: InnovationSource,
    ) -> FactorAdjoints<LogState, Self::Coordinates> {
        let external = source == InnovationSource::JointOuIncrements;
        FactorAdjoints {
            state: LogState {
                value: next.value * t.decay + vbar * 2.0 * variance,
            },
            spot: next.value * if external { 0.0 } else { t.spot },
            volatility: [next.value
                * if external {
                    self.vol_of_vol()
                } else {
                    t.orthogonal
                }],
        }
    }
}

impl BergomiDynamics for LogCoordinate {
    type State = LogState;
    type Transition = LogTransition;
    const FACTOR_COUNT: usize = 1;
    fn vol_of_vol(self) -> f64 {
        self.0.vol_of_vol()
    }
    fn parameters(self) -> Vec<f64> {
        // Distinguish this test representation in the existing plan fingerprint.
        vec![
            self.0.mean_reversion(),
            self.vol_of_vol(),
            self.0.correlation(),
            601.0,
        ]
    }
    fn transition(self, dt: f64) -> Result<LogTransition, BergomiError> {
        let t = self.0.transition(dt)?;
        Ok(LogTransition {
            decay: t.decay,
            spot: self.vol_of_vol() * t.spot_loading,
            orthogonal: self.vol_of_vol() * t.orthogonal_loading,
        })
    }
    fn multiplier(self, x: LogState) -> f64 {
        x.value.exp()
    }
    fn multiplier_squared(self, x: LogState) -> f64 {
        (2.0 * x.value).exp()
    }
    fn finite(x: LogState) -> bool {
        x.value.is_finite()
    }
    fn evolve(self, t: LogTransition, x: LogState, z: f64, orth: [f64; 2]) -> LogState {
        self.evolve_normals(t, x, z, OrthogonalNormals::new(&orth, 1))
    }
    fn evolve_ou(self, t: LogTransition, x: LogState, increments: [f64; 2]) -> LogState {
        self.evolve_increments(t, x, OuInnovations::new(&increments, 1))
    }
    fn reverse_factor(
        self,
        t: LogTransition,
        next: LogState,
        vbar: f64,
        variance: f64,
        external: bool,
    ) -> (LogState, f64, [f64; 2]) {
        let a = self.pullback(
            t,
            next,
            vbar,
            variance,
            InnovationSource::from_external(external),
        );
        (a.state, a.spot, [a.volatility[0], 0.0])
    }
}

fn request(qmc: bool, bump: f64) -> PricingRequest {
    let mut v: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/v1/pricing_request.golden.json"
    )))
    .unwrap();
    let mut values = vec![0.04; 9];
    values[4] += bump;
    v["model"] = json!({"type":"local_volatility","local_variance_grid":{
        "time_nodes":[0.0,0.5,1.0],"log_forward_moneyness_nodes":[-0.8,0.0,0.8],
        "shape":[3,3],"values":values,"floor":1e-8,"cap":4.0}});
    v["engine"]["independent_sampling_units"] = json!(64);
    if qmc {
        v["engine"] = json!({"type":"randomized_quasi_monte_carlo",
            "points_per_scramble":32,"scramble_count":4,"master_scramble_seed":871,
            "variance_reduction":{"antithetic":true,"brownian_bridge":true}});
    }
    parse_request_json(&serde_json::to_vec(&v).unwrap(), JsonLimits::DEFAULT).unwrap()
}

fn close(a: f64, b: f64, tolerance: f64) {
    assert!((a - b).abs() < tolerance * (1.0 + b.abs()), "{a} != {b}");
}

#[test]
fn alternate_state_uses_shared_calibration_payoff_and_recalibrated_reverse() {
    let factor = Bergomi1Factor::new(0.7, 0.35, -0.4).unwrap();
    for qmc in [false, true] {
        for workers in [1, 2] {
            let policy = ExecutionPolicy::new(workers, Some(16)).unwrap();
            let particles = || LsvParticleConfig::new(128, 93, 0.6, 2.0, true).unwrap();
            let req = request(qmc, 0.0);
            let original =
                BergomiLsvPricingPlan::compile(&req, factor, particles(), policy).unwrap();
            let alternate =
                BergomiLsvPricingPlan::compile(&req, LogCoordinate(factor), particles(), policy)
                    .unwrap();
            assert_ne!(original.plan_fingerprint(), alternate.plan_fingerprint());
            for (&a, &b) in alternate
                .calibration()
                .surface()
                .squared_leverage()
                .iter()
                .zip(original.calibration().surface().squared_leverage())
            {
                close(a, b, 3e-13);
            }
            let risk = alternate.evaluate_local_variance_risk().unwrap();
            let reference = original.evaluate_local_variance_risk().unwrap();
            close(risk.price.value, reference.price.value, 3e-12);
            assert_eq!(alternate.evaluate().unwrap().value, risk.price.value);
            for (&a, &b) in risk.node_adjoints.iter().zip(&reference.node_adjoints) {
                close(a, b, 3e-11);
            }
            let value = |bump| {
                BergomiLsvPricingPlan::compile(
                    &request(qmc, bump),
                    LogCoordinate(factor),
                    particles(),
                    policy,
                )
                .unwrap()
                .evaluate()
                .unwrap()
                .value
            };
            close(
                risk.node_adjoints[4],
                (value(1e-6) - value(-1e-6)) / 2e-6,
                2e-6,
            );
        }
    }
}

#[test]
fn alternate_state_pullback_handles_normals_and_supplied_ou_increments() {
    let factor = LogCoordinate(Bergomi1Factor::new(0.7, 0.35, -0.4).unwrap());
    let t = factor.transition(0.3).unwrap();
    for source in [
        InnovationSource::IndependentNormals,
        InnovationSource::JointOuIncrements,
    ] {
        let loss = |v: &[f64]| {
            let state = LogState { value: v[0] };
            let next = if source == InnovationSource::IndependentNormals {
                factor.evolve_normals(t, state, v[1], OrthogonalNormals::new(&v[2..], 1))
            } else {
                factor.evolve_increments(t, state, OuInnovations::new(&v[2..], 1))
            };
            0.7 * next.value + 0.3 * 0.04 * factor.multiplier_squared(state)
        };
        let input = [0.17, -0.3, 0.2];
        let variance = 0.04 * factor.multiplier_squared(LogState { value: input[0] });
        let adj = factor.pullback(t, LogState { value: 0.7 }, 0.3, variance, source);
        for (i, bar) in [adj.state.value, adj.spot, adj.volatility[0]]
            .into_iter()
            .enumerate()
        {
            let mut up = input;
            up[i] += 1e-6;
            let mut down = input;
            down[i] -= 1e-6;
            close(bar, (loss(&up) - loss(&down)) / 2e-6, 2e-9);
        }
    }
}
