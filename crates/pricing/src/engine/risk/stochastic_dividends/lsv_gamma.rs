//! Scale-invariant Spot Gamma for stochastic-dividend residual LSV.
//!
//! Spot bumps re-anchor funded residual equity and the leverage surface initial_f,
//! while leaving every calibrated squared-leverage value unchanged. Normalized
//! f/Y states can therefore be reused across the whole common-noise bump ladder.

use super::*;
use crate::core::PositiveF64;
use crate::engine::processes::stochastic_dividends::reverse::ReverseContext;
use crate::mc::lsv::LsvLeverageSurface;
use crate::models::BuehlerDividendState;
use crate::risk::{GammaConfig, SpotBump};

const METHOD: &str = "buehler-residual-lsv-scale-invariant-aad-delta-gamma-v1";
const WIDTH: usize = 7;

struct Scenario {
    path: StochasticDividendPathPlan,
    reverse: ReverseContext,
}

impl StochasticDividendPricingPlan {
    /// Common-noise central differences of the exact scale-invariant LSV Spot
    /// Delta on a half/base/double bump ladder. The residual-LSV surface is
    /// re-anchored for every Spot scenario; its leverage values are not refit.
    pub fn evaluate_lsv_gamma(
        &self,
        config: GammaConfig,
    ) -> Result<StochasticDividendGammaRisk, MonteCarloError> {
        if self.lsv.is_none() || !self.path.is_lsv() {
            return Err(MonteCarloError::UnsupportedRiskForModel {
                model: "LSV Gamma requires a stochastic-dividend residual LSV plan",
            });
        }
        if !self.risk_supported {
            return Err(MonteCarloError::UnsupportedRiskForModel {
                model: "stochastic-dividend LSV discontinuous payoff requires explicit smoothing",
            });
        }

        let spot = self.market.spot().get();
        let (bump, convention) = match config.bump() {
            SpotBump::Absolute(h) => (h.get(), b"absolute".as_slice()),
            SpotBump::Relative(h) => (h.get() * spot, b"relative".as_slice()),
        };
        let bumps = [0.5 * bump, bump, 2.0 * bump];
        let mut scenarios = Vec::with_capacity(6);
        for &h in &bumps {
            if !h.is_finite()
                || h <= 0.0
                || !(2.0 * h).is_finite()
                || spot - h >= spot
                || spot + h <= spot
            {
                return Err(invalid("lsv_gamma_spot_bump").into());
            }
            for shifted_spot in [spot - h, spot + h] {
                let shifted_spot = PositiveF64::new(shifted_spot, "lsv_gamma_shifted_spot")
                    .map_err(|_| invalid("lsv_gamma_shifted_spot"))?;
                scenarios.push(self.lsv_spot_scenario(shifted_spot)?);
            }
        }

        let reverse = ReverseContext::new(&self.path, &self.market, self.payment_time)?;
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
                    self.sample_lsv_gamma(
                        &reverse,
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
                let mut means = Vec::with_capacity(config.scramble_count().get() as usize);
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
                            self.sample_lsv_gamma(
                                &reverse,
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
                    .collect::<Vec<_>>();
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
            .map(|s| estimator_error(*s, units))
            .collect::<Result<Vec<_>, _>>()?;
        if values.iter().chain(&errors).any(|value| !value.is_finite()) {
            return Err(invalid("lsv_gamma_estimator").into());
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

    fn lsv_spot_scenario(&self, shifted_spot: PositiveF64) -> Result<Scenario, MonteCarloError> {
        let lsv = self
            .lsv
            .as_ref()
            .ok_or(MonteCarloError::UnsupportedRiskForModel {
                model: "LSV Gamma requires a stochastic-dividend residual LSV calibration",
            })?;
        let market = self.market.with_spot(shifted_spot)?;
        let grid = LocalVolTimeGrid::compile(
            self.path.times().to_vec(),
            self.path.times()[self.path.times().len() - 1],
        )?;
        let base = StochasticDividendPathPlan::compile(&market, self.path.model(), 0.0, &grid)?;
        let path = match lsv {
            StochasticDividendLsvCalibration::One {
                calibration,
                dividend_volatility_correlation,
                ..
            } => {
                let surface = reanchored_surface(calibration.surface(), base.risky_spot())?;
                base.with_bergomi_lsv(
                    calibration.factor(),
                    *dividend_volatility_correlation,
                    surface,
                )?
            }
            StochasticDividendLsvCalibration::Two {
                calibration,
                dividend_volatility_correlations,
                ..
            } => {
                let surface = reanchored_surface(calibration.surface(), base.risky_spot())?;
                base.with_bergomi_two_factor_lsv(
                    calibration.factor(),
                    *dividend_volatility_correlations,
                    surface,
                )?
            }
        };
        if path.random_dimension() != self.path.random_dimension() {
            return Err(invalid("lsv_gamma_random_dimension").into());
        }
        let reverse = ReverseContext::new(&path, &market, self.payment_time)?;
        Ok(Scenario { path, reverse })
    }

    #[allow(clippy::too_many_arguments)]
    fn sample_lsv_gamma(
        &self,
        reverse: &ReverseContext,
        scenarios: &[Scenario],
        bumps: &[f64; 3],
        mut z: Vec<f64>,
        bridge: Option<&BrownianBridgePlan>,
        antithetic: bool,
        out: &mut [f64],
    ) -> Result<(), MonteCarloError> {
        if out.len() != WIDTH || scenarios.len() != 6 {
            return Err(invalid("lsv_gamma_shape").into());
        }
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
                    .map_err(|error| MonteCarloError::LocalVol(error.into()))?;
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
            let shocks = z.iter().map(|value| sign * value).collect::<Vec<_>>();
            let states = self.path.evolve_path(&shocks)?;
            let (price, delta) = self.lsv_spot_adjoint(&self.path, reverse, &states)?;
            out[0] += price;
            out[1] += delta;
            for (j, (pair, h)) in scenarios.as_chunks::<2>().0.iter().zip(bumps).enumerate() {
                let down = self
                    .lsv_spot_adjoint(&pair[0].path, &pair[0].reverse, &states)?
                    .1;
                let up = self
                    .lsv_spot_adjoint(&pair[1].path, &pair[1].reverse, &states)?
                    .1;
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
        if out.iter().any(|value| !value.is_finite()) {
            return Err(invalid("lsv_gamma_sample").into());
        }
        Ok(())
    }

    fn lsv_spot_adjoint(
        &self,
        path: &StochasticDividendPathPlan,
        reverse: &ReverseContext,
        states: &[BuehlerDividendState],
    ) -> Result<(f64, f64), MonteCarloError> {
        let spots = path
            .nodes()
            .iter()
            .zip(states)
            .map(|(node, state)| node.spots(*state))
            .collect::<Result<Vec<_>, _>>()?;
        let (payoff, seeds) = self
            .base
            .hybrid_spot_payoff_adjoints(path.times(), &spots)?;
        Ok((payoff, reverse.spot_pullback(states, &seeds)?))
    }
}

fn reanchored_surface(
    surface: &LsvLeverageSurface,
    initial_f: f64,
) -> Result<LsvLeverageSurface, MonteCarloError> {
    Ok(LsvLeverageSurface::new(
        surface.times().to_vec(),
        surface.log_nodes().to_vec(),
        surface.squared_leverage().to_vec(),
        initial_f,
    )?)
}

fn estimator_error(
    statistics: DeterministicStatistics,
    units: u64,
) -> Result<f64, MonteCarloError> {
    let variance = statistics
        .moments()
        .sample_variance()
        .ok_or(MonteCarloError::InsufficientSamplingUnits { count: units })?;
    let error = (variance / units as f64).sqrt();
    if !error.is_finite() {
        return Err(invalid("lsv_gamma_standard_error").into());
    }
    Ok(error)
}
