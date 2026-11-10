use crate::engine::risk::report::estimate_from_statistics;
use crate::mc::{
    BrownianBridgePlan, DeterministicExecutor, DeterministicStatistics, EngineConfig,
    ExecutionPolicy, Philox4x32, RandomCoordinate, RandomDomain, RqmcPlan, inverse_standard_normal,
};
use crate::multi_asset::MultiAssetError as Error;
use crate::{Estimate, EstimatorKind};

/// The equity kernels have factor-major Brownian blocks followed by model-specific
/// Gaussian residuals. The optional independent currency Brownian is appended last.
pub(super) fn estimate(
    times: &[f64],
    equity_dimension: usize,
    brownian_blocks: usize,
    extra_fx: bool,
    engine: EngineConfig,
    policy: ExecutionPolicy,
    value: impl Fn(&[f64]) -> Result<f64, Error> + Sync,
) -> Result<Estimate, Error> {
    let steps = times.len() - 1;
    let dimension = u32::try_from(equity_dimension + if extra_fx { steps } else { 0 })
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
    let bridge = vr
        .brownian_bridge()
        .then(|| BrownianBridgePlan::compile(times.to_vec(), 1))
        .transpose()
        .map_err(Error::numerical)?;
    let sample = |mut normals: Vec<f64>| {
        if let Some(b) = &bridge {
            for block in normals[..brownian_blocks * steps].chunks_exact_mut(steps) {
                block.copy_from_slice(&b.apply_one_factor(block).map_err(Error::numerical)?);
            }
            if extra_fx {
                let block = &mut normals[equity_dimension..];
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
