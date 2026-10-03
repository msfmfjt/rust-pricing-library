//! Common deterministic MC/RQMC driver for continuous-bridge paired risks.
use super::*;

pub(super) struct BumpStatistics {
    pub values: Vec<f64>,
    pub errors: Vec<f64>,
    pub units: u64,
    pub paths: u128,
}
pub(super) struct JointBumpStatistics {
    pub marginal: BumpStatistics,
    pub covariance: Option<Vec<Vec<f64>>>,
}
impl StochasticDividendContinuousBarrierPlan {
    pub(super) fn bump_statistics(
        &self,
        width: usize,
        sample: impl Fn(
            Vec<f64>,
            Option<&BrownianBridgePlan>,
            bool,
            &mut [f64],
        ) -> Result<(), MonteCarloError>
        + Sync,
    ) -> Result<BumpStatistics, MonteCarloError> {
        self.bump_statistics_with_covariance(width, false, sample)
            .map(|s| s.marginal)
    }

    // The matrix is covariance of ESTIMATED MEANS, not of individual paths.
    pub(super) fn bump_statistics_with_covariance(
        &self,
        width: usize,
        full_covariance: bool,
        sample: impl Fn(
            Vec<f64>,
            Option<&BrownianBridgePlan>,
            bool,
            &mut [f64],
        ) -> Result<(), MonteCarloError>
        + Sync,
    ) -> Result<JointBumpStatistics, MonteCarloError> {
        let executor = DeterministicExecutor::new(self.inner.policy)?;
        let dimension = self.inner.path.random_dimension();
        let (statistics, upper_covariances, units, paths) = match self.inner.engine {
            EngineConfig::PseudoMonteCarlo(config) => {
                let count = config.independent_sampling_units().get();
                let bridge = self.inner.bridge(config.variance_reduction())?;
                let rng = Philox4x32::from_seed(config.master_seed());
                let stats = executor.try_map_reduce_statistics_vector_with_covariance(
                    count,
                    width,
                    full_covariance,
                    |p, out| {
                        let z = (0..dimension)
                            .map(|d| {
                                rng.standard_normal(RandomCoordinate::new(
                                    p,
                                    d,
                                    RandomDomain::Valuation,
                                ))
                            })
                            .collect();
                        sample(
                            z,
                            bridge.as_ref(),
                            config.variance_reduction().antithetic(),
                            out,
                        )
                    },
                )?;
                let covariances = stats
                    .covariances
                    .map(|rows| {
                        rows.into_iter()
                            .map(|row| {
                                row.into_iter()
                                    .map(|c| {
                                        c.sample_covariance().ok_or(
                                            MonteCarloError::InsufficientSamplingUnits { count },
                                        )
                                    })
                                    .collect::<Result<Vec<_>, _>>()
                            })
                            .collect::<Result<Vec<_>, _>>()
                    })
                    .transpose()?;
                (
                    stats.components,
                    covariances,
                    count,
                    config.evaluated_paths(),
                )
            }
            EngineConfig::RandomizedQuasiMonteCarlo(config) => {
                let qmc = RqmcPlan::compile(config, dimension)?;
                let bridge = self.inner.bridge(config.variance_reduction())?;
                let count = config.points_per_scramble().get();
                let mut means = Vec::with_capacity(config.scramble_count().get() as usize);
                for scramble in 0..config.scramble_count().get() {
                    let stats =
                        executor.try_map_reduce_statistics_vector(count, width, |p, out| {
                            let z = (0..dimension)
                                .map(|d| {
                                    let u = qmc
                                        .uniform(scramble, p, d)
                                        .map_err(|_| invalid("rqmc_uniform"))?;
                                    inverse_standard_normal(u).map_err(|_| invalid("rqmc_normal"))
                                })
                                .collect::<Result<Vec<_>, _>>()?;
                            sample(
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
                let stats = (0..width)
                    .map(|j| {
                        let values = means.iter().map(|v| v[j]).collect::<Vec<_>>();
                        DeterministicStatistics::from_ordered_values_two_pass(&values)
                    })
                    .collect::<Vec<_>>();
                // RQMC independent units are scramble means, never within-scramble points.
                let covariances = if full_covariance {
                    if scrambles < 2 {
                        return Err(MonteCarloError::InsufficientSamplingUnits {
                            count: scrambles,
                        });
                    }
                    let centers = stats
                        .iter()
                        .map(|s| s.sum().total() / scrambles as f64)
                        .collect::<Vec<_>>();
                    Some(
                        (0..width)
                            .map(|i| {
                                (i..width)
                                    .map(|j| {
                                        means
                                            .iter()
                                            .map(|row| {
                                                (row[i] - centers[i]) * (row[j] - centers[j])
                                            })
                                            .collect::<pricing_numerics::NeumaierSum>()
                                            .total()
                                            / (scrambles - 1) as f64
                                    })
                                    .collect::<Vec<_>>()
                            })
                            .collect::<Vec<_>>(),
                    )
                } else {
                    None
                };
                (
                    stats,
                    covariances,
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
            return Err(invalid("continuous_barrier_bump_estimator").into());
        }
        let covariance = upper_covariances.map(|rows| {
            let mut matrix = vec![vec![0.0; width]; width];
            for (i, row) in rows.into_iter().enumerate() {
                for (offset, covariance) in row.into_iter().enumerate() {
                    let j = i + offset;
                    let covariance = covariance / units as f64;
                    matrix[i][j] = covariance;
                    matrix[j][i] = covariance;
                }
            }
            matrix
        });
        if covariance
            .iter()
            .flatten()
            .flatten()
            .any(|v| !v.is_finite())
        {
            return Err(invalid("continuous_barrier_bump_covariance").into());
        }
        Ok(JointBumpStatistics {
            marginal: BumpStatistics {
                values,
                errors,
                units,
                paths,
            },
            covariance,
        })
    }
}
