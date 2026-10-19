//! Fixed-block moments and selected same-sampling-unit covariances.
use crate::MonteCarloError;
use crate::core::FiniteF64;
use crate::mc::{DeterministicExecutor, ExecutionPolicy};
use crate::risk::{RiskConfigError, VegaKtStepMoment};
use pricing_numerics::{
    CenteredCovariance, CenteredMoment, NeumaierSum, reduce_covariances, reduce_moments,
    reduce_sums,
};
use rayon::prelude::*;
use std::ops::Range;

fn finite(x: f64) -> Result<f64, MonteCarloError> {
    Ok(FiniteF64::new(x, "path_gamma_chain_statistic")
        .map_err(RiskConfigError::from)?
        .get())
}

/// The callback has already averaged each antithetic pair. Preserve the existing
/// sum/moment block order while accumulating only the requested covariance block.
#[cfg(test)]
pub(super) fn evaluate(
    policy: ExecutionPolicy,
    count: u64,
    components: usize,
    covariance_range: Range<usize>,
    sample: impl Fn(u64) -> Result<Vec<f64>, MonteCarloError> + Sync + Send,
) -> Result<(Vec<VegaKtStepMoment>, Vec<f64>), MonteCarloError> {
    let (moments, covariance, _) =
        evaluate_retaining(policy, count, components, covariance_range, &[], sample)?;
    Ok((moments, covariance))
}

type RetainedStatistics = (Vec<VegaKtStepMoment>, Vec<f64>, Vec<Vec<f64>>);

/// Retain requested readouts in counter order, after antithetic averaging.
pub(super) fn evaluate_retaining(
    policy: ExecutionPolicy,
    count: u64,
    components: usize,
    covariance_range: Range<usize>,
    retained_indices: &[usize],
    sample: impl Fn(u64) -> Result<Vec<f64>, MonteCarloError> + Sync + Send,
) -> Result<RetainedStatistics, MonteCarloError> {
    if retained_indices.iter().any(|i| *i >= components)
        || count as u128 * retained_indices.len() as u128 > 2_000_000
    {
        return Err(super::invalid("retained Gamma sample budget exceeded").into());
    }
    let dimension = covariance_range.len();
    let pairs: Vec<_> = (0..dimension)
        .flat_map(|i| (i..dimension).map(move |j| (i, j)))
        .collect();
    let block_size = policy.reduction_block_size().get();
    let executor = DeterministicExecutor::new(policy)?;
    let partials = executor
        .install(|| {
            (0..count.div_ceil(block_size))
                .into_par_iter()
                .map(|block| {
                    let mut sums = vec![NeumaierSum::new(); components];
                    let mut moments = vec![CenteredMoment::new(); components];
                    let mut covariance = vec![CenteredCovariance::new(); pairs.len()];
                    let begin = block * block_size;
                    let mut retained = Vec::new();
                    for index in begin..count.min(begin.saturating_add(block_size)) {
                        let values = sample(index)?;
                        if !retained_indices.is_empty() {
                            retained.push(
                                retained_indices
                                    .iter()
                                    .map(|i| finite(values[*i]))
                                    .collect::<Result<Vec<_>, _>>()?,
                            );
                        }
                        for ((s, m), &v) in sums.iter_mut().zip(&mut moments).zip(&values) {
                            s.add(v);
                            m.add(v);
                        }
                        let selected = &values[covariance_range.clone()];
                        for (c, &(i, j)) in covariance.iter_mut().zip(&pairs) {
                            c.add(selected[i], selected[j]);
                        }
                    }
                    Ok::<_, MonteCarloError>((sums, moments, covariance, retained))
                })
                .collect::<Vec<_>>()
        })
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?;
    let moments = (0..components)
        .map(|i| {
            let mean = finite(
                reduce_sums(partials.iter().map(|p| p.0[i]).collect()).total() / count as f64,
            )?;
            let sample_variance = reduce_moments(partials.iter().map(|p| p.1[i]).collect())
                .sample_variance()
                .map(finite)
                .transpose()?;
            let standard_error = sample_variance
                .map(|v| finite(v.sqrt() / (count as f64).sqrt()))
                .transpose()?;
            Ok(VegaKtStepMoment {
                mean,
                sample_variance,
                standard_error,
            })
        })
        .collect::<Result<Vec<_>, MonteCarloError>>()?;
    let mut covariance = vec![0.; dimension * dimension];
    for (index, &(i, j)) in pairs.iter().enumerate() {
        let value = finite(
            reduce_covariances(partials.iter().map(|p| p.2[index]).collect())
                .sample_covariance()
                .expect("validated at least two sampling units"),
        )?;
        covariance[i * dimension + j] = value;
        covariance[j * dimension + i] = value;
    }
    let retained = partials.into_iter().flat_map(|p| p.3).collect();
    Ok((moments, covariance, retained))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn centered_covariance_matches_independent_two_pass_and_prior_moments() {
        // Nonzero offset, dependent and zero coordinates, several partial blocks.
        let rows: Vec<_> = (0..19)
            .map(|i| {
                let x = (i as f64 - 7.) * 0.25;
                let y = ((i * 7) % 11) as f64 - 3.;
                vec![91., 1e6 + x, y, 1e6 + x - 2. * y, 0.]
            })
            .collect();
        let run = |workers| {
            let policy = ExecutionPolicy::new(workers, Some(4)).unwrap();
            evaluate(policy, 19, 5, 1..5, |i| Ok(rows[i as usize].clone())).unwrap()
        };
        let (moments, covariance) = run(1);
        assert_eq!((moments.clone(), covariance.clone()), run(4));
        let policy = ExecutionPolicy::new(1, Some(4)).unwrap();
        let previous = DeterministicExecutor::new(policy)
            .unwrap()
            .try_map_reduce_statistics_vector(19, 5, |i, out| {
                out.copy_from_slice(&rows[i as usize]);
                Ok::<_, MonteCarloError>(())
            })
            .unwrap();
        for (before, after) in previous.into_iter().zip(&moments) {
            assert_eq!(
                super::super::super::vegakt_steps::moment(before, 19).unwrap(),
                *after
            );
        }
        for i in 0..4 {
            for j in 0..4 {
                let mi = rows.iter().map(|r| r[1 + i]).sum::<f64>() / 19.;
                let mj = rows.iter().map(|r| r[1 + j]).sum::<f64>() / 19.;
                let expected = rows
                    .iter()
                    .map(|r| (r[1 + i] - mi) * (r[1 + j] - mj))
                    .sum::<f64>()
                    / 18.;
                assert!((covariance[4 * i + j] - expected).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn covariance_uses_pair_averages_and_rejects_nonfinite_statistics() {
        let policy = ExecutionPolicy::new(1, Some(2)).unwrap();
        let mates = [[1., 9.], [2., -6.], [8., 4.]];
        let (_, covariance) = evaluate(policy, 3, 2, 0..2, |i| {
            let pair = mates[i as usize];
            let mean = 0.5 * pair[0] + 0.5 * pair[1];
            Ok(vec![mean, -2. * mean])
        })
        .unwrap();
        let variance = 19.;
        assert_eq!(
            covariance,
            vec![variance, -2. * variance, -2. * variance, 4. * variance]
        );
        assert!(evaluate(policy, 3, 1, 0..1, |_| Ok(vec![f64::INFINITY])).is_err());
    }
}
