use crate::engine::risk::report::estimate_from_statistics;
use crate::mc::{
    BrownianBridgePlan, DeterministicExecutor, DeterministicStatistics, EngineConfig,
    ExecutionPolicy, Philox4x32, RandomCoordinate, RandomDomain, RqmcPlan, inverse_standard_normal,
};
use crate::multi_asset::MultiAssetError as Error;
use crate::{Estimate, EstimatorKind};

/// Factor-major equity Brownian blocks precede the model's orthogonal residuals.
/// Additional currency and rate shocks may extend beyond the last equity fixing.
#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct Layout {
    pub equity_steps: usize,
    pub equity_dimension: usize,
    pub brownian_blocks: usize,
    pub extra_blocks: usize,
}
pub(super) fn estimate(
    times: &[f64],
    layout: Layout,
    engine: EngineConfig,
    policy: ExecutionPolicy,
    value: impl Fn(&[f64]) -> Result<f64, Error> + Sync,
) -> Result<Estimate, Error> {
    if times.len() == 1 {
        let exact = value(&[])?;
        return Estimate::new(exact, 0., exact, exact, EstimatorKind::Analytical, 1)
            .map_err(Error::numerical);
    }
    let steps = times.len() - 1;
    let Layout {
        equity_steps,
        equity_dimension,
        brownian_blocks,
        extra_blocks,
    } = layout;
    let dimension = u32::try_from(equity_dimension + extra_blocks * steps)
        .map_err(|_| Error::Invalid("equity/FX random dimension is too large"))?;
    let (vr, units) = match engine {
        EngineConfig::PseudoMonteCarlo(c) => {
            (c.variance_reduction(), c.independent_sampling_units().get())
        }
        EngineConfig::RandomizedQuasiMonteCarlo(c) => {
            (c.variance_reduction(), u64::from(c.scramble_count().get()))
        }
    };
    if units < 2 {
        return Err(Error::Invalid(
            "at least two independent sampling units are required",
        ));
    }
    let bridge_for = |active: bool, nodes: &[f64]| {
        (active && vr.brownian_bridge())
            .then(|| BrownianBridgePlan::compile(nodes.to_vec(), 1))
            .transpose()
            .map_err(Error::numerical)
    };
    let equity_bridge = bridge_for(equity_steps > 0, &times[..=equity_steps])?;
    let extra_bridge = bridge_for(extra_blocks > 0, times)?;
    let sample = |mut normals: Vec<f64>| {
        if let Some(b) = &equity_bridge {
            for block in normals[..brownian_blocks * equity_steps].chunks_exact_mut(equity_steps) {
                block.copy_from_slice(&b.apply_one_factor(block).map_err(Error::numerical)?);
            }
        }
        if let Some(b) = &extra_bridge {
            for block in normals[equity_dimension..].chunks_exact_mut(steps) {
                block.copy_from_slice(&b.apply_one_factor(block).map_err(Error::numerical)?);
            }
        }
        let first = value(&normals)?;
        let result = if vr.antithetic() {
            for z in &mut normals {
                *z = -*z;
            }
            0.5 * first + 0.5 * value(&normals)?
        } else {
            first
        };
        if !result.is_finite() {
            return Err(Error::Invalid("nonfinite equity/FX payoff"));
        }
        Ok(result)
    };
    let executor = DeterministicExecutor::new(policy).map_err(Error::numerical)?;
    let (stats, kind) = match engine {
        EngineConfig::PseudoMonteCarlo(c) => {
            let rng = Philox4x32::from_seed(c.master_seed());
            let stats = executor
                .try_map_reduce_statistics(units, |p| {
                    sample(
                        (0..dimension)
                            .map(|d| {
                                rng.standard_normal(RandomCoordinate::new(
                                    p,
                                    d,
                                    RandomDomain::Valuation,
                                ))
                            })
                            .collect(),
                    )
                })
                .map_err(Error::numerical)?;
            (stats, EstimatorKind::PseudoMonteCarlo)
        }
        EngineConfig::RandomizedQuasiMonteCarlo(c) => {
            let qmc = RqmcPlan::compile(c, dimension).map_err(Error::numerical)?;
            let count = c.points_per_scramble().get();
            let mut means = Vec::new();
            for scramble in 0..c.scramble_count().get() {
                let stats = executor
                    .try_map_reduce_statistics(count, |p| {
                        let normals = (0..dimension)
                            .map(|d| {
                                let u = qmc.uniform(scramble, p, d).map_err(Error::numerical)?;
                                inverse_standard_normal(u).map_err(Error::numerical)
                            })
                            .collect::<Result<Vec<_>, Error>>()?;
                        sample(normals)
                    })
                    .map_err(Error::numerical)?;
                means.push(stats.sum().total() / count as f64);
            }
            (
                DeterministicStatistics::from_ordered_values_two_pass(&means),
                EstimatorKind::RandomizedQuasiMonteCarlo,
            )
        }
    };
    estimate_from_statistics(stats, units, 1., kind).map_err(Error::numerical)
}
