//! Release-mode sensitivity acceptance through public pricing plans.
//! Every finite difference recompiles and recalibrates with the same calibration
//! and pricing streams as AAD. Error bars never relax the derivative budget.

use pricing::mc::{ExecutionPolicy, RqmcConfig, VarianceReduction};
use serde::Serialize;
use serde_json::json;

#[path = "cases/extended_risk_multi.rs"]
mod multi;
#[path = "cases/extended_risk_single.rs"]
mod single;

const SEEDS: [u64; 3] = [1709, 2903, 4001];
const BUMPS: [f64; 4] = [1e-5, 1e-6, 1e-7, 1e-8];
const REQUIRED_BUMPS: [usize; 2] = [2, 3];
const ABS_TOL: f64 = 2e-5;
const REL_TOL: f64 = 1e-4;

#[derive(Clone, Copy, Debug, Serialize)]
struct Scenario {
    calibration_seed: u64,
    pricing_seed: u64,
    particles: usize,
    points: u64,
    scrambles: u32,
    steps: usize,
    refined: bool,
}
impl Scenario {
    fn all() -> impl Iterator<Item = Self> {
        [false, true].into_iter().flat_map(|refined| {
            SEEDS.into_iter().map(move |seed| Self {
                calibration_seed: seed,
                pricing_seed: seed ^ 0xd1b5_4a32_d192_ed03,
                particles: if refined { 2048 } else { 1024 },
                points: if refined { 512 } else { 256 },
                scrambles: 4,
                steps: if refined { 8 } else { 4 },
                refined,
            })
        })
    }
    fn engine(self) -> pricing::mc::EngineConfig {
        pricing::mc::EngineConfig::RandomizedQuasiMonteCarlo(
            RqmcConfig::new(
                self.points,
                self.scrambles,
                self.pricing_seed,
                VarianceReduction::new(true, true),
            )
            .unwrap(),
        )
    }
    fn policy(self) -> ExecutionPolicy {
        ExecutionPolicy::new(2, Some(128)).unwrap()
    }
    fn particles(self, trace: bool) -> pricing::mc::lsv::LsvParticleConfig {
        pricing::mc::lsv::LsvParticleConfig::new(
            self.particles,
            self.calibration_seed,
            0.35,
            5.0,
            trace,
        )
        .unwrap()
    }
}

#[derive(Clone, Copy, Debug, Serialize)]
enum Factor {
    One,
    Two,
    Rough,
}
impl Factor {
    fn one(self) -> pricing::models::Bergomi1Factor {
        pricing::models::Bergomi1Factor::new(1.3, 0.45, -0.4).unwrap()
    }
    fn two(self) -> pricing::models::Bergomi2Factor {
        pricing::models::Bergomi2Factor::new([0.4, 2.0], 0.45, 0.35, [-0.4, -0.2], 0.25).unwrap()
    }
    fn rough(self) -> pricing::models::RoughBergomi {
        pricing::models::RoughBergomi::new(0.16, 0.7, -0.4).unwrap()
    }
    fn multi(self, s: Scenario, trace: bool) -> pricing::multi_asset::MultiAssetBergomiLsvConfig {
        use pricing::multi_asset::*;
        let particles = s.particles(trace);
        match self {
            Self::One => MultiAssetLsvConfig {
                factor: self.one(),
                particles,
            }
            .into(),
            Self::Two => MultiAssetLsv2FactorConfig {
                factor: self.two(),
                particles,
            }
            .into(),
            Self::Rough => MultiAssetRoughLsvConfig {
                factor: self.rough(),
                particles,
            }
            .into(),
        }
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

fn derivative_failures(aad: f64, fd: &[f64; BUMPS.len()]) -> Vec<String> {
    let mut out = Vec::new();
    if !aad.is_finite() || fd.iter().any(|v| !v.is_finite()) {
        out.push("nonfinite derivative".into());
        return out;
    }
    // The two finest bumps must both agree. Coarser bumps are retained as
    // truncation/branch diagnostic, never selected to rescue a failing case.
    for i in REQUIRED_BUMPS {
        let tolerance = ABS_TOL + REL_TOL * aad.abs().max(fd[i].abs());
        if (aad - fd[i]).abs() > tolerance {
            out.push(format!(
                "bump {}: AAD/FD error exceeds {tolerance}",
                BUMPS[i]
            ));
        }
    }
    let [a, b] = REQUIRED_BUMPS;
    let plateau = 2.0 * (ABS_TOL + REL_TOL * fd[a].abs().max(fd[b].abs()));
    if (fd[a] - fd[b]).abs() > plateau {
        out.push("finest finite differences have no accepted plateau".into());
    }
    out
}

fn sweep(
    case: &str,
    scenario: Scenario,
    parameter: &str,
    aad: f64,
    mut price: impl FnMut(f64) -> f64,
) -> usize {
    let mut prices = [[0.0; 2]; BUMPS.len()];
    let mut fd = [0.0; BUMPS.len()];
    for (i, h) in BUMPS.into_iter().enumerate() {
        prices[i] = [price(h), price(-h)];
        assert!(prices[i].iter().all(|v| v.is_finite()));
        fd[i] = (prices[i][0] - prices[i][1]) / (2.0 * h);
    }
    let failures = derivative_failures(aad, &fd);
    println!(
        "EXTENDED_RISK_SWEEP {}",
        json!({"case":case,"scenario":scenario,"parameter":parameter,"aad":aad,
            "bumps":BUMPS,"prices_up_down":prices,"finite_differences":fd,
            "absolute_errors":fd.map(|v|(v-aad).abs()),"absolute_tolerance":ABS_TOL,
            "relative_tolerance":REL_TOL,"required_bump_indices":REQUIRED_BUMPS,"failures":failures})
    );
    usize::from(!failures.is_empty())
}

fn conditional_errors(errors: &[f64], expected: usize) {
    assert_eq!(errors.len(), expected);
    assert!(errors.iter().all(|e| e.is_finite() && *e >= 0.0));
}

#[test]
fn derivative_gate_rejects_missing_feedback_nonfinite_and_unstable_differences() {
    assert!(derivative_failures(3.0, &[3.1, 3.01, 3.00001, 3.0]).is_empty());
    assert!(!derivative_failures(0.0, &[3.0; BUMPS.len()]).is_empty());
    assert!(!derivative_failures(f64::NAN, &[3.0; BUMPS.len()]).is_empty());
    assert!(!derivative_failures(3.0, &[3.0, 3.0, f64::INFINITY, 3.0]).is_empty());
    assert!(!derivative_failures(3.0, &[3.0, 3.0, 3.1, 2.9]).is_empty());
}

macro_rules! acceptance {
    ($name:ident, $body:expr) => {
        #[test]
        #[ignore = "release-mode multi-seed recalibrated sensitivity acceptance"]
        fn $name() {
            $body
        }
    };
}
acceptance!(
    one_factor_calibrated_local_variance,
    single::deterministic(Factor::One)
);
acceptance!(
    two_factor_calibrated_local_variance,
    single::deterministic(Factor::Two)
);
acceptance!(
    rough_calibrated_local_variance,
    single::deterministic(Factor::Rough)
);
acceptance!(one_factor_hw_aad_and_vegakt, single::hybrid(Factor::One));
acceptance!(two_factor_hw_aad_and_vegakt, single::hybrid(Factor::Two));
acceptance!(rough_hw_aad_and_vegakt, single::hybrid(Factor::Rough));
acceptance!(mixed_bergomi_hw_vegakt, multi::fixed(false));
acceptance!(mixed_rough_hw_vegakt, multi::fixed(true));
acceptance!(
    joint_two_factor_local_correlation,
    multi::joint(Factor::Two, false)
);
acceptance!(
    joint_two_factor_hw_local_correlation,
    multi::joint(Factor::Two, true)
);
acceptance!(
    joint_rough_hw_local_correlation,
    multi::joint(Factor::Rough, true)
);
