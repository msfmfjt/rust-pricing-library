use super::*;
use crate::{JsonLimits, parse_request_json};
use serde_json::{Value, json};

fn request(qmc: bool) -> PricingRequest {
    let mut v: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/v1/pricing_request.golden.json"
    )))
    .unwrap();
    v["model"] = json!({"type":"local_volatility","local_variance_grid":{
        "time_nodes":[0.0,0.5,1.0],"log_forward_moneyness_nodes":[-0.8,0.0,0.8],
        "shape":[3,3],"values":vec![0.04;9],"floor":1e-8,"cap":4.0}});
    v["engine"]["independent_sampling_units"] = json!(32);
    if qmc {
        v["engine"] = json!({"type":"randomized_quasi_monte_carlo",
            "points_per_scramble":16,"scramble_count":4,"master_scramble_seed":819,
            "variance_reduction":{"antithetic":true,"brownian_bridge":true}});
    }
    parse_request_json(&serde_json::to_vec(&v).unwrap(), JsonLimits::DEFAULT).unwrap()
}

fn particles(trace: bool) -> LsvParticleConfig {
    LsvParticleConfig::new(64, 71, 0.5, 2.0, trace).unwrap()
}

fn plan(qmc: bool, trace: bool) -> BergomiLsvPricingPlan {
    BergomiLsvPricingPlan::compile(
        &request(qmc),
        Bergomi1Factor::new(0.7, 0.3, -0.4).unwrap(),
        particles(trace),
        ExecutionPolicy::new(1, Some(16)).unwrap(),
    )
    .unwrap()
}

// These adapters intentionally implement only pricing. This is a compile-time
// witness that the shared price core does not require either reverse trait.
// They reuse the existing numerical model; they do not add another scheme.
#[derive(Clone, Debug)]
struct PriceModel(CalibratedBergomiLsv);
#[derive(Clone, Debug)]
struct PricePath(BergomiLsvPlan);

impl CalibratedModel for PriceModel {
    type Plan = PricePath;
    const SCHEME: &'static str = BERGOMI_LSV_SCHEME;
    const RANDOM_BLOCKS: usize = 2;
    fn surface(&self) -> &LsvLeverageSurface {
        self.0.surface()
    }
    fn config(&self) -> &LsvParticleConfig {
        self.0.config()
    }
    fn target(&self) -> &LocalVarianceGrid {
        self.0.target()
    }
    fn pricing_plan(&self, grid: &LocalVolTimeGrid) -> Result<PricePath, LsvError> {
        self.0.pricing_plan(grid).map(PricePath)
    }
}

impl PathModel for PricePath {
    fn times(&self) -> &[f64] {
        self.0.times()
    }
    fn pseudo_shocks(
        &self,
        seed: u64,
        path: u64,
        domain: RandomDomain,
    ) -> Result<Vec<f64>, LsvError> {
        self.0.pseudo_shocks(seed, path, domain)
    }
    fn evolve_states(
        &self,
        initial: f64,
        shocks: &[f64],
        states: &mut Vec<f64>,
    ) -> Result<(), LsvError> {
        self.0.evolve_states(initial, shocks, states)
    }
}

#[test]
fn price_only_core_needs_no_recording_or_reverse_implementation() {
    for qmc in [false, true] {
        let p = plan(qmc, false);
        let expected = p.evaluate().unwrap();
        let c = p.core;
        let price_only = LsvPricingCore {
            base: c.base,
            calibration: PriceModel(c.calibration),
            path_plan: PricePath(c.path_plan),
            original_target: c.original_target,
            engine: c.engine,
            policy: c.policy,
            risk_supported: c.risk_supported,
            fingerprint: c.fingerprint,
        };
        assert_eq!(price_only.evaluate().unwrap(), expected);
    }
}

#[test]
fn risk_selection_preserves_price_and_trace_error_precedence() {
    let p = plan(false, false);
    let expected = p.evaluate().unwrap();
    assert_eq!(
        expected.value.to_bits(),
        plan(false, true).evaluate().unwrap().value.to_bits()
    );
    assert!(matches!(
        p.evaluate_local_variance_risk(),
        Err(MonteCarloError::Lsv(LsvError::ReverseTraceNotRetained))
    ));
    // An unsupported payoff still takes precedence over missing trace, as in
    // the public facade before this capability extraction.
    let mut core = p.core;
    core.risk_supported = false;
    assert!(matches!(
        core.evaluate_local_variance_risk(),
        Err(MonteCarloError::UnsupportedRiskForModel { .. })
    ));
    let rough = RoughBergomiLsvPricingPlan::compile(
        &request(false),
        RoughBergomi::new(0.2, 0.5, -0.4).unwrap(),
        particles(false),
        ExecutionPolicy::new(1, Some(16)).unwrap(),
    )
    .unwrap();
    assert!(rough.evaluate().is_ok());
    assert!(matches!(
        rough.evaluate_local_variance_risk(),
        Err(MonteCarloError::Lsv(LsvError::ReverseTraceNotRetained))
    ));
}
