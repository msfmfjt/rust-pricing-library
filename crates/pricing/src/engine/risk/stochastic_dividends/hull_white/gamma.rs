//! Common-noise central differences of HW/Buehler AAD Delta. Cash claims,
//! rates, model inputs, curves, dates, grid and smoothing width remain fixed.
use super::super::StochasticDividendGammaRisk;
use super::*;
use crate::core::PositiveF64;
use crate::engine::processes::stochastic_dividends::hull_white::StochasticDividendHullWhiteState;
use crate::risk::{GammaConfig, SpotBump};

const METHOD: &str = "buehler-bs-hw-common-noise-aad-delta-gamma-v1";
const WIDTH: usize = 7; // Price, Delta, three Gammas, two paired bump gaps.

impl StochasticDividendHullWhitePricingPlan {
    /// Central-bump AAD Delta on a half/base/double ladder with common normals.
    /// All six shifted Spots must be representable and leave positive funded
    /// residual equity; no clamping, one-sided fallback or adaptive bump is used.
    /// Vanilla kinks are handled by bumping Delta. Discontinuous payoffs still
    /// require explicit smoothing, whose width remains fixed across scenarios.
    pub fn evaluate_gamma(
        &self,
        config: GammaConfig,
    ) -> Result<StochasticDividendGammaRisk, MonteCarloError> {
        if !self.risk_supported {
            return Err(MonteCarloError::UnsupportedRiskForModel {
                model: "stochastic-dividend discontinuous payoff requires explicit smoothing",
            });
        }
        let spot = self.market.spot().get();
        let (bump, convention) = match config.bump() {
            SpotBump::Absolute(h) => (h.get(), b"absolute".as_slice()),
            SpotBump::Relative(h) => (h.get() * spot, b"relative".as_slice()),
        };
        let bumps = [0.5 * bump, bump, 2.0 * bump];
        let mut scenarios = Vec::with_capacity(6);
        // Validate and prepare every bump before constructing an executor.
        for &h in &bumps {
            if !h.is_finite()
                || h <= 0.0
                || !(2.0 * h).is_finite()
                || spot - h >= spot
                || spot + h <= spot
            {
                return Err(invalid("gamma_spot_bump").into());
            }
            for shifted_spot in [spot - h, spot + h] {
                let shifted_spot = PositiveF64::new(shifted_spot, "gamma_shifted_spot")
                    .map_err(|_| invalid("gamma_shifted_spot"))?;
                scenarios.push(self.path.risky_spot_at(shifted_spot.get())?);
            }
        }
        let executor = DeterministicExecutor::new(self.policy)?;
        let dimension = self.path.random_dimension();
        let (statistics, units, paths) = match self.engine {
            EngineConfig::PseudoMonteCarlo(config) => {
                let count = config.independent_sampling_units().get();
                let bridge = self.bridge(config.variance_reduction())?;
                let rng = Philox4x32::from_seed(config.master_seed());
                let stats = executor.try_map_reduce_statistics_vector(count, WIDTH, |p, out| {
                    let z = (0..dimension)
                        .map(|d| {
                            rng.standard_normal(RandomCoordinate::new(
                                p,
                                d,
                                RandomDomain::Valuation,
                            ))
                        })
                        .collect();
                    self.gamma_sample(
                        &scenarios,
                        &bumps,
                        z,
                        bridge.as_ref(),
                        config.variance_reduction().antithetic(),
                        out,
                    )
                })?;
                (stats, count, config.evaluated_paths())
            }
            EngineConfig::RandomizedQuasiMonteCarlo(config) => {
                let qmc = RqmcPlan::compile(config, dimension)?;
                let bridge = self.bridge(config.variance_reduction())?;
                let count = config.points_per_scramble().get();
                let mut means = Vec::new();
                for scramble in 0..config.scramble_count().get() {
                    let stats =
                        executor.try_map_reduce_statistics_vector(count, WIDTH, |p, out| {
                            let z = (0..dimension)
                                .map(|d| {
                                    let u = qmc
                                        .uniform(scramble, p, d)
                                        .map_err(|_| invalid("rqmc_uniform"))?;
                                    inverse_standard_normal(u).map_err(|_| invalid("rqmc_normal"))
                                })
                                .collect::<Result<Vec<_>, _>>()?;
                            self.gamma_sample(
                                &scenarios,
                                &bumps,
                                z,
                                bridge.as_ref(),
                                config.variance_reduction().antithetic(),
                                out,
                            )
                        })?;
                    means.push(
                        stats
                            .iter()
                            .map(|s| s.sum().total() / count as f64)
                            .collect::<Vec<_>>(),
                    );
                }
                let scrambles = u64::from(config.scramble_count().get());
                let stats = (0..WIDTH)
                    .map(|j| {
                        let values = means.iter().map(|v| v[j]).collect::<Vec<_>>();
                        DeterministicStatistics::from_ordered_values_two_pass(&values)
                    })
                    .collect();
                (
                    stats,
                    scrambles,
                    u128::from(count)
                        * u128::from(scrambles)
                        * if config.variance_reduction().antithetic() {
                            2
                        } else {
                            1
                        },
                )
            }
        };
        let values = statistics
            .iter()
            .map(|s| s.sum().total() / units as f64)
            .collect::<Vec<_>>();
        let errors = statistics
            .iter()
            .map(|s| {
                let variance = s
                    .moments()
                    .sample_variance()
                    .ok_or(MonteCarloError::InsufficientSamplingUnits { count: units })?;
                Ok((variance / units as f64).sqrt())
            })
            .collect::<Result<Vec<_>, MonteCarloError>>()?;
        if values.iter().chain(&errors).any(|v| !v.is_finite()) {
            return Err(invalid("gamma_estimator").into());
        }
        let mut hash = blake3::Hasher::new();
        hash.update(METHOD.as_bytes());
        hash.update(self.fingerprint.as_bytes());
        hash.update(convention);
        for h in bumps {
            hash.update(&h.to_bits().to_le_bytes());
        }
        Ok(StochasticDividendGammaRisk {
            price: StochasticDividendPrice {
                value: values[0],
                standard_error: errors[0],
                independent_sampling_units: units,
                evaluated_paths: paths,
                plan_fingerprint: self.fingerprint,
                scheme: self.scheme(),
            },
            delta: values[1],
            delta_standard_error: errors[1],
            spot,
            spot_bumps: bumps,
            gamma_estimates: [values[2], values[3], values[4]],
            gamma_standard_errors: [errors[2], errors[3], errors[4]],
            bump_differences: [values[5], values[6]],
            bump_difference_standard_errors: [errors[5], errors[6]],
            payoff_evaluations: 7 * paths,
            risk_fingerprint: Fingerprint::from_bytes(*hash.finalize().as_bytes()),
            method: METHOD,
        })
    }

    fn gamma_sample(
        &self,
        scenarios: &[f64],
        bumps: &[f64; 3],
        mut z: Vec<f64>,
        bridge: Option<&BrownianBridgePlan>,
        antithetic: bool,
        out: &mut [f64],
    ) -> Result<(), MonteCarloError> {
        if let Some(bridge) = bridge {
            let count = self.random_factor_count();
            for factor in 0..count {
                let input = z
                    .iter()
                    .skip(factor)
                    .step_by(count)
                    .copied()
                    .collect::<Vec<_>>();
                let output = bridge
                    .apply_one_factor(&input)
                    .map_err(|e| MonteCarloError::LocalVol(e.into()))?;
                for (step, value) in output.into_iter().enumerate() {
                    z[count * step + factor] = value;
                }
            }
        }
        out.fill(0.0);
        for &sign in if antithetic {
            &[1.0, -1.0][..]
        } else {
            &[1.0][..]
        } {
            let shocks = z.iter().map(|v| sign * v).collect::<Vec<_>>();
            // The normalized state evolution is exactly independent of S0.
            // Cash claims stay fixed; each scenario funds its own risky Spot.
            let states = self.path.evolve_path(&shocks)?;
            let terminal = states.last().ok_or(invalid("terminal_state"))?;
            let relative = self.payment_constant
                * (-terminal.integrated_rate_factor()
                    - self.payment_duration * terminal.rate_factor())
                .exp();
            positive(relative, "relative_payment_discount")?;
            let (price, delta) = self.spot_adjoint(self.path.risky_spot(), &states, relative)?;
            out[0] += price;
            out[1] += delta;
            for (j, (pair, h)) in scenarios.as_chunks::<2>().0.iter().zip(bumps).enumerate() {
                let down = self.spot_adjoint(pair[0], &states, relative)?.1;
                let up = self.spot_adjoint(pair[1], &states, relative)?.1;
                out[2 + j] += (up - down) / (2.0 * h);
            }
        }
        if antithetic {
            for value in out.iter_mut() {
                *value *= 0.5;
            }
        }
        out[5] = out[2] - out[3];
        out[6] = out[3] - out[4];
        if out.iter().any(|v| !v.is_finite()) {
            return Err(invalid("gamma_sample").into());
        }
        Ok(())
    }

    fn spot_adjoint(
        &self,
        risky_spot: f64,
        states: &[StochasticDividendHullWhiteState],
        relative: f64,
    ) -> Result<(f64, f64), MonteCarloError> {
        let spots = states
            .iter()
            .enumerate()
            .map(|(i, state)| self.path.spots_with_risky_spot(i, *state, risky_spot))
            .collect::<Result<Vec<_>, _>>()?;
        let (payoff, seeds) = self
            .base
            .hybrid_spot_payoff_adjoints(self.path.times(), &spots)?;
        // Discount each seed before adding pre/post terms, exactly as basic AAD.
        // The stochastic payment discount itself is invariant under S0 bumps.
        let discounted_seeds = seeds
            .iter()
            .map(|(post, pre)| (relative * post, relative * pre))
            .collect::<Vec<_>>();
        Ok((
            relative * payoff,
            self.path.spot_pullback(states, &discounted_seeds)?,
        ))
    }
}
