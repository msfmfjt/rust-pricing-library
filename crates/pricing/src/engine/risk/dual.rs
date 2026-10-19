//! Policy-based nested simulation: Andersen and Broadie (2004), §3.

use std::{error::Error, fmt, num::NonZeroU32};

use pricing_numerics::NeumaierSum;

use crate::engine::plan::simulation::{EarlyExerciseRuntime, PathObservation};
use crate::engine::risk::report::estimate_from_statistics;
use crate::mc::{
    DeterministicExecutor, EngineConfig, ExecutionPolicy, ExercisePolicy,
    ExercisePolicyFingerprint, LsmNumericalError, Philox4x32, PseudoMcConfig, RandomDomain,
    inverse_standard_normal, open_unit_interval, train_exercise_policy,
};
use crate::models::ModelSpec;
use crate::product::{AmericanVanillaSpec, OptionSide, ProductSpec};
use crate::{
    Estimate, EstimatorKind, Fingerprint, MonteCarloError, PricingRequest, SimulationPlan,
};

pub const ANDERSEN_BROADIE_ABI: &str = "andersen-broadie-policy-nested-v1";
// A distinct Philox counter namespace. Existing RandomDomain IDs remain unchanged.
const INNER_DOMAIN: u32 = 0x4455_414c;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AndersenBroadieConfig {
    continuation_inner_paths: NonZeroU32,
    exercise_inner_paths: NonZeroU32,
    inner_seed: u64,
}

impl AndersenBroadieConfig {
    /// N2 and N3 in §3 of the paper. Inner paths are independent, not antithetic.
    pub fn new(
        continuation_inner_paths: u32,
        exercise_inner_paths: u32,
        inner_seed: u64,
    ) -> Result<Self, AndersenBroadieError> {
        Ok(Self {
            continuation_inner_paths: NonZeroU32::new(continuation_inner_paths).ok_or(
                AndersenBroadieError::ZeroInnerPaths {
                    region: "continuation",
                },
            )?,
            exercise_inner_paths: NonZeroU32::new(exercise_inner_paths)
                .ok_or(AndersenBroadieError::ZeroInnerPaths { region: "exercise" })?,
            inner_seed,
        })
    }

    #[must_use]
    pub const fn continuation_inner_paths(self) -> u32 {
        self.continuation_inner_paths.get()
    }

    #[must_use]
    pub const fn exercise_inner_paths(self) -> u32 {
        self.exercise_inner_paths.get()
    }

    #[must_use]
    pub const fn inner_seed(self) -> u64 {
        self.inner_seed
    }

    fn path_stride(self) -> u64 {
        u64::from(
            self.continuation_inner_paths()
                .max(self.exercise_inner_paths()),
        )
    }
}

#[derive(Debug)]
#[non_exhaustive]
pub enum AndersenBroadieError {
    ZeroInnerPaths { region: &'static str },
    Unsupported { feature: &'static str },
    RandomCoordinateOverflow,
    MonteCarlo(MonteCarloError),
}

impl fmt::Display for AndersenBroadieError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroInnerPaths { region } => {
                write!(f, "{region} inner-path count must be positive")
            }
            Self::Unsupported { feature } => {
                write!(f, "Andersen–Broadie does not support {feature}")
            }
            Self::RandomCoordinateOverflow => {
                write!(f, "Andersen–Broadie random-coordinate count overflowed")
            }
            Self::MonteCarlo(error) => error.fmt(f),
        }
    }
}

impl Error for AndersenBroadieError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::MonteCarlo(error) => Some(error),
            _ => None,
        }
    }
}

impl From<MonteCarloError> for AndersenBroadieError {
    fn from(error: MonteCarloError) -> Self {
        Self::MonteCarlo(error)
    }
}

/// Estimates are conditional on the fitted policy. The confidence bracket is
/// asymptotic, with no adjustment for continuous-exercise discretization error.
#[derive(Clone, Debug, PartialEq)]
pub struct AndersenBroadieResult {
    pub lower_bound: Estimate,
    pub upper_bound: Estimate,
    pub duality_gap: Estimate,
    /// [lower estimate - 1.96 SE_lower, upper estimate + 1.96 SE_upper].
    pub price_confidence_interval_95: [f64; 2],
    pub policy_fingerprint: ExercisePolicyFingerprint,
    pub plan_fingerprint: Fingerprint,
    pub config: AndersenBroadieConfig,
    pub outer_trajectories: u64,
    pub exercise_date_count: usize,
}

/// Immutable, opt-in adapter. The existing PricingPlan and wire ABI are unchanged.
#[derive(Clone, Debug)]
pub struct AndersenBroadiePlan {
    simulation: SimulationPlan,
    product: AmericanVanillaSpec,
    config: AndersenBroadieConfig,
    training_engine: PseudoMcConfig,
    outer_engine: PseudoMcConfig,
    fingerprint: Fingerprint,
}

impl AndersenBroadiePlan {
    pub fn compile(
        request: &PricingRequest,
        execution: ExecutionPolicy,
        config: AndersenBroadieConfig,
    ) -> Result<Self, AndersenBroadieError> {
        let unsupported = |feature| AndersenBroadieError::Unsupported { feature };
        let ProductSpec::AmericanVanilla(product) = request.product() else {
            return Err(unsupported("products other than AmericanVanilla"));
        };
        if !matches!(
            request.model(),
            ModelSpec::BlackScholes(_) | ModelSpec::Black76(_)
        ) {
            return Err(unsupported("models other than Black–Scholes/Black-76"));
        }
        let risk = request.risk();
        if risk.delta() || risk.gamma().is_some() || risk.vega() || risk.vega_kt().is_some() {
            return Err(unsupported(
                "Greeks (use a separate fixed-policy risk request)",
            ));
        }
        let simulation = SimulationPlan::compile(request, execution)?;
        let runtime = simulation
            .early_exercise
            .as_ref()
            .expect("American plan has runtime");
        let (
            EngineConfig::PseudoMonteCarlo(training_engine),
            EngineConfig::PseudoMonteCarlo(outer_engine),
        ) = (runtime.config.training_engine(), simulation.engine)
        else {
            return Err(unsupported("RQMC or mixed training/valuation engines"));
        };
        // Flatten (outer unit, antithetic lane, inner path) without collisions;
        // (branch date, future observation) occupies the dimension coordinate.
        let trajectories = outer_engine
            .independent_sampling_units()
            .get()
            .checked_mul(if outer_engine.variance_reduction().antithetic() {
                2
            } else {
                1
            })
            .ok_or(AndersenBroadieError::RandomCoordinateOverflow)?;
        trajectories
            .checked_mul(config.path_stride())
            .ok_or(AndersenBroadieError::RandomCoordinateOverflow)?;
        let dimensions = runtime
            .exercise_dates
            .len()
            .checked_mul(simulation.observation_times.len())
            .ok_or(AndersenBroadieError::RandomCoordinateOverflow)?;
        u32::try_from(dimensions).map_err(|_| AndersenBroadieError::RandomCoordinateOverflow)?;
        let mut hash = blake3::Hasher::new();
        hash.update(ANDERSEN_BROADIE_ABI.as_bytes());
        hash.update(simulation.plan_fingerprint().as_bytes());
        hash.update(&config.continuation_inner_paths().to_be_bytes());
        hash.update(&config.exercise_inner_paths().to_be_bytes());
        hash.update(&config.inner_seed.to_be_bytes());
        Ok(Self {
            simulation,
            product: product.clone(),
            config,
            training_engine,
            outer_engine,
            fingerprint: Fingerprint::from_bytes(*hash.finalize().as_bytes()),
        })
    }

    #[must_use]
    pub const fn fingerprint(&self) -> Fingerprint {
        self.fingerprint
    }

    pub fn evaluate(&self) -> Result<AndersenBroadieResult, AndersenBroadieError> {
        self.evaluate_inner().map_err(Into::into)
    }

    fn evaluate_inner(&self) -> Result<AndersenBroadieResult, MonteCarloError> {
        let runtime = self
            .simulation
            .early_exercise
            .as_ref()
            .expect("compiled American runtime");
        let training = self.simulation.pseudo_lsm_path_matrices(
            self.training_engine,
            RandomDomain::LsmTrain,
            runtime,
        )?;
        let fitted = train_exercise_policy(
            &runtime.exercise_dates,
            runtime.config.basis().clone(),
            &training.features,
            training.path_count,
            &training.immediate_values,
            &runtime.discount_factors,
            runtime.config.itm_abs_tolerance(),
            runtime.config.cpqr_config(),
            runtime.config.max_matrix_elements(),
            runtime
                .config
                .training_metadata(*self.simulation.payoff.source_fingerprint().as_bytes())?,
        )?;
        drop(training);
        let policy = fitted.into_policy();
        let generator = Philox4x32::from_seed(self.outer_engine.master_seed());
        let executor = DeterministicExecutor::new(self.simulation.execution_policy)?;
        let antithetic = self.outer_engine.variance_reduction().antithetic();
        let lanes = if antithetic { 2 } else { 1 };
        let count = self.outer_engine.independent_sampling_units().get();
        let statistics = executor.try_map_reduce_statistics_array_tiled(
            count,
            NonZeroU32::new(1).expect("one"),
            |unit| {
                let mut normals =
                    self.simulation
                        .normals(&generator, unit, RandomDomain::Valuation);
                let primary = self.outer_sample(&normals, unit * lanes, &policy, runtime)?;
                if antithetic {
                    for normal in &mut normals {
                        *normal = -*normal;
                    }
                    let mate = self.outer_sample(&normals, unit * lanes + 1, &policy, runtime)?;
                    Ok::<[f64; 3], MonteCarloError>(std::array::from_fn(|i| {
                        0.5 * primary[i] + 0.5 * mate[i]
                    }))
                } else {
                    Ok(primary)
                }
            },
        )?;
        let lower_bound =
            estimate_from_statistics(statistics[0], count, 1.0, EstimatorKind::PseudoMonteCarlo)?;
        let upper_bound =
            estimate_from_statistics(statistics[1], count, 1.0, EstimatorKind::PseudoMonteCarlo)?;
        let duality_gap =
            estimate_from_statistics(statistics[2], count, 1.0, EstimatorKind::PseudoMonteCarlo)?;
        Ok(AndersenBroadieResult {
            lower_bound,
            upper_bound,
            duality_gap,
            price_confidence_interval_95: [
                lower_bound.confidence_interval().lower().get(),
                upper_bound.confidence_interval().upper().get(),
            ],
            policy_fingerprint: policy.fingerprint(),
            plan_fingerprint: self.fingerprint,
            config: self.config,
            outer_trajectories: count * lanes,
            exercise_date_count: runtime.exercise_dates.len(),
        })
    }

    fn outer_sample(
        &self,
        normals: &[f64],
        trajectory: u64,
        policy: &ExercisePolicy,
        runtime: &EarlyExerciseRuntime,
    ) -> Result<[f64; 3], MonteCarloError> {
        let observations = self.simulation.path_observations_from_normals(
            normals,
            self.simulation.spot,
            self.simulation.volatility,
        );
        // Reuse the contractual graph for outer cashflows; direct vanilla inner
        // payoff below has identical operation order and is restricted at compile.
        let payoffs = self
            .simulation
            .payoff_outputs_from_observations(&observations)?;
        let mut lower = None;
        let mut dual = DualPath::new();
        for (date, &observation_index) in runtime.observation_indices.iter().enumerate() {
            let state = observations[observation_index];
            let terminal = date + 1 == runtime.exercise_dates.len();
            let exercise = terminal
                || policy.decisions()[date].should_exercise(payoffs[date], &[state.post_spot])?;
            let discounted = finite(
                payoffs[date] * runtime.discount_factors[date],
                "dual discounted payoff",
            )?;
            if exercise && lower.is_none() {
                lower = Some(discounted);
            }
            let continuation = if terminal {
                0.0
            } else {
                self.rollout_mean(date, state, trajectory, exercise, policy, runtime)?
            };
            dual.observe(discounted, continuation, exercise, terminal)?;
        }
        let lower = lower.expect("terminal exercise");
        Ok([
            lower,
            finite(lower + dual.gap, "dual upper sample")?,
            dual.gap,
        ])
    }

    fn rollout_mean(
        &self,
        branch_date: usize,
        state: PathObservation,
        trajectory: u64,
        exercise: bool,
        policy: &ExercisePolicy,
        runtime: &EarlyExerciseRuntime,
    ) -> Result<f64, MonteCarloError> {
        let configured = if exercise {
            self.config.exercise_inner_paths()
        } else {
            self.config.continuation_inner_paths()
        };
        // Exact deterministic transition: repeated identical paths add no information.
        let paths = if self.simulation.volatility == 0.0 {
            1
        } else {
            configured
        };
        let generator = Philox4x32::from_seed(self.config.inner_seed);
        let mut sum = NeumaierSum::new();
        for inner in 0..paths {
            let random_path = trajectory * self.config.path_stride() + u64::from(inner);
            let mut brownian = state.brownian;
            let mut previous_time =
                self.simulation.observation_times[runtime.observation_indices[branch_date]];
            for date in branch_date + 1..runtime.exercise_dates.len() {
                let index = runtime.observation_indices[date];
                let time = self.simulation.observation_times[index];
                let dimension =
                    (branch_date * self.simulation.observation_times.len() + index) as u32;
                let normal = if self.simulation.volatility == 0.0 {
                    0.0
                } else {
                    inner_normal(generator, random_path, dimension)
                };
                brownian += (time - previous_time).sqrt() * normal;
                previous_time = time;
                let variance = self.simulation.volatility * self.simulation.volatility * time;
                let canonical = self.simulation.observation_forwards[index]
                    * (-0.5 * variance + self.simulation.volatility * brownian).exp();
                let state = self.simulation.path_observation(
                    index,
                    canonical,
                    brownian,
                    self.simulation.spot,
                );
                let immediate = self.immediate(state.post_spot)?;
                if date + 1 == runtime.exercise_dates.len()
                    || policy.decisions()[date].should_exercise(immediate, &[state.post_spot])?
                {
                    sum.add(finite(
                        immediate * runtime.discount_factors[date],
                        "dual inner payoff",
                    )?);
                    break;
                }
            }
        }
        finite(sum.total() / f64::from(paths), "dual inner mean")
    }

    fn immediate(&self, spot: f64) -> Result<f64, MonteCarloError> {
        finite(spot, "dual inner spot")?;
        let intrinsic = match self.product.side() {
            OptionSide::Call => spot - self.product.strike().get(),
            OptionSide::Put => self.product.strike().get() - spot,
        };
        finite(
            self.product.notional().get() * intrinsic.max(0.0),
            "dual inner intrinsic",
        )
    }
}

fn inner_normal(generator: Philox4x32, path: u64, dimension: u32) -> f64 {
    let counter = [
        path as u32,
        (path >> 32) as u32,
        dimension / 4,
        INNER_DOMAIN,
    ];
    let word = generator.generate(counter)[(dimension % 4) as usize];
    inverse_standard_normal(open_unit_interval(word)).expect("open Philox midpoint")
}

fn finite(value: f64, stage: &'static str) -> Result<f64, MonteCarloError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(LsmNumericalError::NonFiniteIntermediate { stage }.into())
    }
}

// Equation (9) telescopes to pi_k = L_k/B_k + A_k, where A_k is the sum
// of discounted exercise payoff minus conditional continuation at PRIOR
// exercise recommendations. The policy is evaluated even after first exercise.
struct DualPath {
    correction: NeumaierSum,
    gap: f64,
}

impl DualPath {
    fn new() -> Self {
        Self {
            correction: NeumaierSum::new(),
            gap: f64::NEG_INFINITY,
        }
    }

    fn observe(
        &mut self,
        payoff: f64,
        continuation: f64,
        exercise: bool,
        terminal: bool,
    ) -> Result<(), MonteCarloError> {
        let value = if exercise { payoff } else { continuation };
        let candidate = finite(
            (payoff - value) - self.correction.total(),
            "dual gap candidate",
        )?;
        self.gap = self.gap.max(candidate);
        if exercise && !terminal {
            self.correction
                .add(finite(payoff - continuation, "dual exercise correction")?);
            finite(self.correction.total(), "dual accumulated correction")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exhaustive_tree_bounds_include_poor_policies_and_exact_optimal_policy() {
        // All numbers are in time-zero units. Each child has probability 1/2.
        let payoff: [f64; 16] = [
            0., 5., 2., 9., 0., 4., 8., 14., 1., 0., 3., 6., 5., 12., 8., 20.,
        ];
        let mut optimal = payoff;
        let mut optimal_exercise = [true; 16];
        for node in (1..8).rev() {
            let continuation = 0.5 * (optimal[2 * node] + optimal[2 * node + 1]);
            optimal_exercise[node] = payoff[node] > continuation;
            optimal[node] = payoff[node].max(continuation);
        }
        for choice in 0..3 {
            let mut exercise = optimal_exercise;
            for flag in &mut exercise[1..8] {
                if choice < 2 {
                    *flag = choice == 1;
                }
            }
            let mut policy_value = payoff;
            let mut continuation = [0.0; 16];
            for node in (1..8).rev() {
                continuation[node] = 0.5 * (policy_value[2 * node] + policy_value[2 * node + 1]);
                policy_value[node] = if exercise[node] {
                    payoff[node]
                } else {
                    continuation[node]
                };
            }
            let mut lower_mean = 0.0;
            let mut upper_mean = 0.0;
            for path in 0..8 {
                let mut node = 1;
                let mut lower = None;
                let mut dual = DualPath::new();
                for date in 0..4 {
                    if exercise[node] && lower.is_none() {
                        lower = Some(payoff[node]);
                    }
                    dual.observe(payoff[node], continuation[node], exercise[node], date == 3)
                        .unwrap();
                    if date < 3 {
                        node = 2 * node + ((path >> (2 - date)) & 1);
                    }
                }
                lower_mean += lower.unwrap() / 8.0;
                upper_mean += (lower.unwrap() + dual.gap) / 8.0;
            }
            assert_eq!(lower_mean, policy_value[1]);
            assert!(lower_mean <= optimal[1]);
            assert!(upper_mean >= optimal[1]);
            if choice == 2 {
                assert_eq!(lower_mean, optimal[1]);
                assert_eq!(upper_mean, optimal[1]);
            }
        }
    }

    #[test]
    fn finite_inner_noise_preserves_upper_expectation_in_both_regions() {
        // Exercise now pays 1; terminal payoff is 0/4 with equal probabilities.
        // N_inner=1 gives continuation estimates 0/4, independently of outer.
        // Enumerate the entire joint distribution, without a statistical test.
        for exercise_now in [false, true] {
            let mut upper = 0.0;
            for terminal in [0.0, 4.0] {
                for inner_continuation in [0.0, 4.0] {
                    let lower = if exercise_now { 1.0 } else { terminal };
                    let mut dual = DualPath::new();
                    dual.observe(1.0, inner_continuation, exercise_now, false)
                        .unwrap();
                    dual.observe(terminal, 0.0, true, true).unwrap();
                    upper += (lower + dual.gap) / 4.0;
                }
            }
            assert_eq!(upper, 2.5);
            assert!(upper >= 2.0);
        }
    }

    #[test]
    fn private_inner_namespace_is_separate_from_training_and_valuation() {
        let generator = Philox4x32::from_seed(7);
        for path in 0..8 {
            for dimension in 0..8 {
                let inner = inner_normal(generator, path, dimension);
                for domain in [
                    RandomDomain::Valuation,
                    RandomDomain::LsmTrain,
                    RandomDomain::Diagnostics,
                ] {
                    let other = generator
                        .standard_normal(crate::mc::RandomCoordinate::new(path, dimension, domain));
                    assert_ne!(inner.to_bits(), other.to_bits());
                }
            }
        }
    }
}
