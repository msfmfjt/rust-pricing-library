//! Integration with request/payoff/sampling/result contracts. Normalized Bass
//! paths have mean one; existing affine coordinates reconstruct physical spot.
use super::*;
use crate::core::{DayCountConvention, PositiveF64};
use crate::engine::risk::report::{estimate_from_statistics, extrapolation_warnings};
use crate::mc::{
    BrownianBridgePlan, DeterministicExecutor, DeterministicStatistics, EngineConfig, RqmcPlan,
    inverse_standard_normal,
};
use crate::models::ModelSpec;
use crate::product::{BarrierMonitoring, ProductSpec};
use crate::risk::{SmileDynamics, vega_kt_bucket_estimates, vega_kt_full_bucket_covariance};
use crate::*;
use rayon::prelude::*;

#[derive(Clone, Debug)]
pub(crate) struct BassRuntime {
    market: BassMarketIvModel,
    simulation: BassSimulationPlan,
    risk: Option<BassVegaKtRiskPlan>,
    report_buckets: bool,
    full_covariance: bool,
    iv_bump: f64,
    qmc: Option<RqmcPlan>,
    bridge: Option<BrownianBridgePlan>,
}
impl BassRuntime {
    pub(crate) fn compile(
        request: &PricingRequest,
        times: &[f64],
    ) -> Result<Option<Self>, MonteCarloError> {
        let ModelSpec::BassLocalVolatility(spec) = request.model() else {
            return Ok(None);
        };
        if matches!(request.product(), ProductSpec::AmericanVanilla(_))
            || matches!(request.product(),ProductSpec::Barrier(b) if b.monitoring()==BarrierMonitoring::Continuous)
        {
            return Err(MonteCarloError::UnsupportedModel {
                model: "Bass continuous barriers and early exercise",
            });
        }
        if request.risk().smile_dynamics() != SmileDynamics::StickyLogMoneyness {
            return Err(MonteCarloError::UnsupportedRiskForModel {
                model: "Bass requires sticky log moneyness in the residual-equity coordinate",
            });
        }
        if let Some(kt) = request.risk().vega_kt() {
            let times: Vec<_> = kt
                .maturity_nodes()
                .iter()
                .map(|d| DayCountConvention::Act365F.year_fraction(request.valuation_date(), *d))
                .collect();
            let strikes: Vec<_> = kt
                .log_forward_moneyness_nodes()
                .iter()
                .map(|k| k.get())
                .collect();
            let same = |a: &[f64], b: &[f64]| {
                a.len() == b.len()
                    && a.iter().zip(b).all(|(a, b)| {
                        (a - b).abs() <= 8.0 * f64::EPSILON * a.abs().max(b.abs()).max(1.0)
                    })
            };
            if !same(&times, spec.surface().maturity_nodes())
                || !same(&strikes, spec.surface().log_moneyness_nodes())
            {
                return Err(BassError::InvalidInput(
                    "Bass VegaKT reporting axes must match the market IV quote axes".into(),
                )
                .into());
            }
        }
        let (units, vr) = match request.engine() {
            EngineConfig::PseudoMonteCarlo(c) => {
                (c.independent_sampling_units().get(), c.variance_reduction())
            }
            EngineConfig::RandomizedQuasiMonteCarlo(c) => {
                (u64::from(c.scramble_count().get()), c.variance_reduction())
            }
        };
        if units < 2 {
            return Err(MonteCarloError::InsufficientSamplingUnits { count: units });
        }
        let market = BassMarketIvModel::calibrate(
            1.0,
            spec.surface().clone(),
            spec.projection_nodes().to_vec(),
            spec.config(),
            spec.projection_config(),
        )?;
        let observations = if times.is_empty() {
            vec![0.0]
        } else {
            times.to_vec()
        };
        let simulation = market.model().compile_simulation(observations.clone())?;
        let risk = if request.risk().vega() || request.risk().vega_kt().is_some() {
            Some(market.compile_vega_kt(observations, spec.iv_bump())?)
        } else {
            None
        };
        let dimension = simulation.normal_count();
        let bridge = if vr.brownian_bridge() && dimension > 0 {
            Some(
                BrownianBridgePlan::compile(simulation.time_nodes().to_vec(), 1)
                    .map_err(|e| MonteCarloError::LocalVol(e.into()))?,
            )
        } else {
            None
        };
        let qmc = match request.engine() {
            EngineConfig::RandomizedQuasiMonteCarlo(c) if dimension > 0 => {
                Some(RqmcPlan::compile(c, dimension as u32)?)
            }
            _ => None,
        };
        Ok(Some(Self {
            market,
            simulation,
            risk,
            report_buckets: request.risk().vega_kt().is_some(),
            full_covariance: request
                .risk()
                .vega_kt()
                .is_some_and(|k| k.full_bucket_covariance()),
            iv_bump: spec.iv_bump(),
            qmc,
            bridge,
        }))
    }

    fn payoff(
        &self,
        base: &SimulationPlan,
        path: &[f64],
        spot: f64,
    ) -> Result<f64, MonteCarloError> {
        let physical = |u, date, pre| {
            if u != base.underlying {
                return None;
            }
            let i = base
                .observation_dates
                .iter()
                .position(|d| *d == Some(date))?;
            let coordinate = if pre {
                base.observation_pre_dividend_coordinates[i]?
            } else {
                base.observation_affine_coordinates[i]
            };
            let c = coordinate.at_spot(base.spot, spot);
            Some(c.reconstruct_spot(
                PositiveF64::new(spot, "spot").ok()?,
                base.observation_forwards[i] * (spot / base.spot) * path[i],
            ))
        };
        let values = base.payoff.evaluate_with_pre_dividend_spots(
            |u, d| physical(u, d, false),
            |u, d| physical(u, d, true),
        )?;
        let value = base.discount * values[0];
        if !value.is_finite() {
            return Err(BassError::Numerical("nonfinite Bass discounted payoff".into()).into());
        }
        Ok(value)
    }

    // Columns: price, Delta, Gamma, parallel Vega, then requested quote Vegas.
    fn sample(
        &self,
        base: &SimulationPlan,
        mut z: Vec<f64>,
        antithetic: bool,
    ) -> Result<Vec<f64>, MonteCarloError> {
        if let Some(bridge) = &self.bridge {
            z = bridge
                .apply_one_factor(&z)
                .map_err(|e| MonteCarloError::LocalVol(e.into()))?;
        }
        let n = if self.report_buckets {
            self.market.surface().implied_volatilities().len()
        } else {
            0
        };
        let mut result = vec![0.0; 4 + n];
        for sign in if antithetic {
            &[1.0, -1.0][..]
        } else {
            &[1.0][..]
        } {
            let shocks: Vec<_> = z.iter().map(|v| sign * v).collect();
            let path = self.simulation.path_from_normals(&shocks)?;
            let value = self.payoff(base, &path, base.spot)?;
            result[0] += value;
            if base.request_delta || base.request_gamma.is_some() {
                let h = base.validation_spot_bump;
                let up = self.payoff(base, &path, base.spot + h)?;
                let down = self.payoff(base, &path, base.spot - h)?;
                result[1] += (up - down) / (2.0 * h);
                result[2] += (up - 2.0 * value + down) / (h * h);
            }
            if let Some(risk) = &self.risk {
                for (j, (up, down)) in risk.pairs().iter().enumerate() {
                    let parallel = j + 1 == risk.pairs().len();
                    if !parallel && !self.report_buckets {
                        continue;
                    }
                    let up = self.payoff(base, &up.path_from_normals(&shocks)?, base.spot)?;
                    let down = self.payoff(base, &down.path_from_normals(&shocks)?, base.spot)?;
                    result[if parallel { 3 } else { 4 + j }] += (up - down) / (2.0 * self.iv_bump);
                }
            }
        }
        for v in &mut result {
            *v /= if antithetic { 2.0 } else { 1.0 };
        }
        if result.iter().any(|v| !v.is_finite()) {
            return Err(BassError::Numerical("nonfinite Bass risk sample".into()).into());
        }
        Ok(result)
    }

    pub(crate) fn execute(
        &self,
        base: &SimulationPlan,
    ) -> Result<MonteCarloPrice, MonteCarloError> {
        let executor = DeterministicExecutor::new(base.execution_policy)?;
        let dimension = self.simulation.normal_count();
        let columns = 4 + if self.report_buckets {
            self.market.surface().implied_volatilities().len()
        } else {
            0
        };
        let collect = |count: u64,
                       normal: &(dyn Fn(u64) -> Result<Vec<f64>, MonteCarloError> + Sync),
                       anti: bool| {
            let size = usize::try_from(count)
                .ok()
                .and_then(|n| n.checked_mul(columns));
            if size.is_none_or(|n| n > 100_000_000) {
                return Err(MonteCarloError::Bass(BassError::InvalidInput("Bass sampling matrix exceeds 100 million values; reduce sampling units or IV nodes".into())));
            }
            executor.install(|| {
                (0..count)
                    .into_par_iter()
                    .map(|p| self.sample(base, normal(p)?, anti))
                    .collect::<Result<Vec<_>, _>>()
            })
        };
        let (rows, estimator, seed, scrambles, antithetic, paths) = match base.engine {
            EngineConfig::PseudoMonteCarlo(c) => {
                let rng = Philox4x32::from_seed(c.master_seed());
                let rows = collect(
                    c.independent_sampling_units().get(),
                    &|p| {
                        Ok((0..dimension)
                            .map(|d| {
                                rng.standard_normal(RandomCoordinate::new(
                                    p,
                                    d as u32,
                                    RandomDomain::Valuation,
                                ))
                            })
                            .collect())
                    },
                    c.variance_reduction().antithetic(),
                )?;
                (
                    rows,
                    EstimatorKind::PseudoMonteCarlo,
                    c.master_seed(),
                    None,
                    c.variance_reduction().antithetic(),
                    c.evaluated_paths(),
                )
            }
            EngineConfig::RandomizedQuasiMonteCarlo(c) => {
                let mut rows = Vec::new();
                for scramble in 0..c.scramble_count().get() {
                    let samples = collect(
                        c.points_per_scramble().get(),
                        &|p| {
                            Ok((0..dimension)
                                .map(|d| {
                                    let u = self
                                        .qmc
                                        .as_ref()
                                        .expect("positive dimension")
                                        .uniform(scramble, p, d as u32)
                                        .expect("compiled coordinates");
                                    inverse_standard_normal(u)
                                        .expect("Sobol midpoint strictly in (0,1)")
                                })
                                .collect())
                        },
                        c.variance_reduction().antithetic(),
                    )?;
                    rows.push(
                        (0..columns)
                            .map(|j| {
                                DeterministicStatistics::from_ordered_values_two_pass(
                                    &samples.iter().map(|r| r[j]).collect::<Vec<_>>(),
                                )
                                .sum()
                                .total()
                                    / samples.len() as f64
                            })
                            .collect(),
                    );
                }
                let paths = u128::from(c.points_per_scramble().get())
                    * u128::from(c.scramble_count().get())
                    * if c.variance_reduction().antithetic() {
                        2
                    } else {
                        1
                    };
                (
                    rows,
                    EstimatorKind::RandomizedQuasiMonteCarlo,
                    c.master_scramble_seed(),
                    Some(c.scramble_count().get()),
                    c.variance_reduction().antithetic(),
                    paths,
                )
            }
        };
        let n = rows.len() as u64;
        let statistics: Vec<_> = (0..columns)
            .map(|j| {
                DeterministicStatistics::from_ordered_values_two_pass(
                    &rows.iter().map(|r| r[j]).collect::<Vec<_>>(),
                )
            })
            .collect();
        let estimate = estimate_from_statistics(statistics[0], n, 1.0, estimator)?;
        let risk_estimate = |j, scale, raw, scaled| -> Result<RiskEstimate, MonteCarloError> {
            Ok(RiskEstimate::new(
                estimate_from_statistics(statistics[j], n, 1.0, estimator)?,
                estimate_from_statistics(statistics[j], n, scale, estimator)?,
                raw,
                scaled,
            ))
        };
        let risks = RiskReport {
            delta: base
                .request_delta
                .then(|| {
                    risk_estimate(
                        1,
                        base.spot * 0.01,
                        RiskUnit::DeltaRaw,
                        RiskUnit::DeltaOnePercentSpot,
                    )
                })
                .transpose()?,
            gamma: base
                .request_gamma
                .map(|_| {
                    risk_estimate(
                        2,
                        (base.spot * 0.01).powi(2),
                        RiskUnit::GammaRaw,
                        RiskUnit::GammaOnePercentSpotSquared,
                    )
                })
                .transpose()?,
            vega: base
                .request_vega
                .then(|| risk_estimate(3, 0.01, RiskUnit::VegaRaw, RiskUnit::VegaOneVolPoint))
                .transpose()?,
            vega_kt: if self.report_buckets {
                Some(self.bucket_report(&rows, &statistics)?)
            } else {
                None
            },
        };
        let mut warnings = extrapolation_warnings(base.discount_region, base.dividend_region);
        warnings.push(PricingWarning::new("bass_finite_grid","Bass uses finite grids and central CRN differences; sampling errors exclude calibration, grid and bump bias."));
        Ok(MonteCarloPrice {
            pricing_result: PricingResult {
                value: estimate,
                risks,
                diagnostics: Diagnostics::new(warnings),
                replay: base.replay_metadata(),
            },
            sampling_variance: statistics[0].moments().sample_variance().unwrap_or(0.0),
            estimator_variance: estimate.standard_error().get().powi(2),
            independent_sampling_units: n,
            evaluated_paths: paths,
            risk_diagnostics: RiskDiagnostics {
                methods: RiskMethodMetadata {
                    delta: base.request_delta.then_some(RiskMethod::CentralBump),
                    gamma: base.request_gamma.map(|_| RiskMethod::CentralBump),
                    vega: base.request_vega.then_some(RiskMethod::CentralBump),
                    smile_dynamics: base.smile_dynamics,
                    gamma_spot_bump: base.request_gamma.map(|_| base.validation_spot_bump),
                    validation_spot_bump: (base.request_delta || base.request_gamma.is_some())
                        .then_some(base.validation_spot_bump),
                    validation_volatility_bump: self.risk.as_ref().map(|_| self.iv_bump),
                    bump_policy_version: 1,
                    exercise_strategy: None,
                    stopping_indices: None,
                    exercise_policy_fingerprint: None,
                },
                delta_validation: None,
                gamma_validation: None,
                vega_validation: None,
            },
            diagnostics: MonteCarloDiagnostics {
                master_seed: seed,
                estimator,
                scramble_count: scrambles,
                direction_checksum: self.qmc.as_ref().map(RqmcPlan::direction_checksum),
                scramble_checksum: self.qmc.as_ref().map(RqmcPlan::scramble_checksum),
                policy_version: base.execution_policy.version(),
                worker_threads: base.execution_policy.worker_threads().get(),
                reduction_block_size: base.execution_policy.reduction_block_size().get(),
                aad_tile_policy_version: base.aad_tile_policy.version(),
                aad_tile_capacity: base.aad_tile_policy.resolved_capacity().get(),
                checkpoint_policy_version: base.checkpoint_policy.version(),
                checkpoint_interval: base.checkpoint_policy.resolved_interval().get(),
                antithetic,
                discount_region: base.discount_region,
                dividend_region: base.dividend_region,
                payoff_fingerprint: base.payoff.tape_fingerprint(),
                valuation_kind: if base.payoff_smoothing.is_some() {
                    PayoffValuationKind::SmoothedSurrogate
                } else {
                    PayoffValuationKind::ExactContractual
                },
                payoff_smoothing: base.payoff_smoothing_diagnostics(),
                path_state: base.path_state_diagnostics,
                barrier_bridge: None,
            },
            early_exercise_diagnostics: None,
        })
    }

    fn bucket_report(
        &self,
        rows: &[Vec<f64>],
        stats: &[DeterministicStatistics],
    ) -> Result<VegaKtResult, MonteCarloError> {
        let surface = self.market.surface();
        let count = surface.implied_volatilities().len();
        let prices: Vec<_> = rows.iter().map(|r| r[0]).collect();
        let samples: Vec<_> = rows.iter().flat_map(|r| r[4..].iter().copied()).collect();
        let estimates = vega_kt_bucket_estimates(&prices, &samples, count)?;
        let coordinates = surface
            .maturity_nodes()
            .iter()
            .enumerate()
            .flat_map(|(i, &t)| {
                surface
                    .log_moneyness_nodes()
                    .iter()
                    .enumerate()
                    .map(move |(j, &k)| {
                        VegaKtResultCoordinate::new(
                            t,
                            k,
                            surface.implied_volatilities()
                                [i * surface.log_moneyness_nodes().len() + j],
                        )
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let values: Vec<_> = estimates.iter().map(|e| e.raw_mean()).collect();
        let bucket_estimates = estimates
            .iter()
            .map(|e| {
                VegaKtResultBucketEstimate::new(
                    e.raw_mean(),
                    e.market_scaled_mean(),
                    e.sample_variance(),
                    e.price_covariance(),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let parallel = stats[3].sum().total() / rows.len() as f64;
        let sum = values.iter().sum::<f64>();
        let reporting = VegaKtResultReportingStats::new(0, 0, 0.0, 0.0)?;
        let forward = surface
            .log_moneyness_nodes()
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
            .unwrap()
            .0;
        Ok(VegaKtResult::new(
            coordinates,
            bucket_estimates,
            values,
            if self.full_covariance {
                Some(vega_kt_full_bucket_covariance(&samples, count)?)
            } else {
                None
            },
            if self.full_covariance {
                VegaKtResultCovarianceLayout::FullBucketMatrixRowMajor
            } else {
                VegaKtResultCovarianceLayout::PriceAndBucketVarianceOnly
            },
            VegaKtResultProjection::new(parallel, parallel - sum, sum, reporting)?,
            VegaKtResultResidualDiagnostics::new(
                0,
                surface.log_moneyness_nodes().len() - 1,
                forward,
                0.0,
                parallel - sum,
                sum,
                reporting,
            )?,
            VegaKtResultUnit::CurrencyPerUnitAbsoluteVolatility,
            VegaKtResultUnit::CurrencyPerVolatilityPoint,
            BASS_VEGA_KT_METHOD,
            "central_difference_grid_dependent",
        )?)
    }
}

impl SimulationPlan {
    pub fn bass_vega_scenario_diagnostics(&self) -> Option<&[BassVegaKtScenarioDiagnostics]> {
        self.bass
            .as_ref()
            .and_then(|b| b.risk.as_ref().map(|r| r.diagnostics()))
    }
    pub fn bass_calibration_diagnostics(&self) -> Option<&[BassCalibrationDiagnostics]> {
        self.bass.as_ref().map(|b| b.market.model().diagnostics())
    }
    pub fn bass_projection_diagnostics(&self) -> Option<&[crate::bass_lv::BassSurfaceDiagnostics]> {
        self.bass
            .as_ref()
            .map(|b| b.market.projection_diagnostics())
    }
}
