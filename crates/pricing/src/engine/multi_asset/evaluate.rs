use super::path::AssetPath;
use super::*;
use crate::engine::risk::report::estimate_from_statistics;
use crate::mc::{DeterministicExecutor, DeterministicStatistics};
use crate::multi_asset::MultiAssetError as E;
use crate::{EstimatorKind, product::CompiledOpcode};

struct Layout {
    vega: Vec<(usize, usize)>,
    gamma: usize,
    boundary: usize,
    total: usize,
}
impl Layout {
    fn new(plan: &MultiAssetPricingPlan, risk: Option<MultiAssetRiskConfig>) -> Self {
        let n = plan.assets.len();
        let mut offset = 1 + if risk.is_some() { n } else { 0 };
        let mut vega = Vec::new();
        if risk.is_some() {
            for a in &plan.assets {
                let count = match &a.model {
                    ModelSpec::LocalVolatility(v) => v.local_variance_grid().values().len(),
                    _ => 1,
                };
                vega.push((offset, count));
                offset += count;
            }
        }
        let gamma = offset;
        if risk.is_some_and(|r| r.gamma_relative_bump.is_some()) {
            offset += n * n;
        }
        let boundary = offset;
        offset += n;
        Self {
            vega,
            gamma,
            boundary,
            total: offset,
        }
    }
}
impl MultiAssetPricingPlan {
    pub fn evaluate(&self) -> Result<MultiAssetPrice, E> {
        self.run(None)
    }
    pub fn evaluate_aad(&self, config: MultiAssetRiskConfig) -> Result<MultiAssetPrice, E> {
        if config
            .gamma_relative_bump
            .is_some_and(|b| !b.is_finite() || b <= 0.0 || b >= 1.0)
        {
            return Err(E::Invalid("Gamma relative bump must lie in (0,1)"));
        }
        if self
            .payoff
            .opcodes()
            .iter()
            .any(|op| matches!(op, CompiledOpcode::Indicator { .. }))
        {
            return Err(E::Invalid(
                "AAD of discontinuous payoffs requires explicit smoothing",
            ));
        }
        self.run(Some(config))
    }
    fn run(&self, risk: Option<MultiAssetRiskConfig>) -> Result<MultiAssetPrice, E> {
        let layout = Layout::new(self, risk);
        let executor = DeterministicExecutor::new(self.execution).map_err(E::numerical)?;
        let reduce = |scramble, units, antithetic| {
            executor
                .try_map_reduce_statistics_vector(units, layout.total, |point, output| {
                    let shocks = self.shocks(scramble, point)?;
                    let first = self.path_values(&self.paths(&shocks)?, risk, &layout)?;
                    if antithetic {
                        let shocks: Vec<_> = shocks
                            .iter()
                            .map(|z| z.iter().map(|v| -v).collect())
                            .collect();
                        let second = self.path_values(&self.paths(&shocks)?, risk, &layout)?;
                        for ((o, a), b) in output.iter_mut().zip(first).zip(second) {
                            *o = (a + b) * 0.5;
                        }
                    } else {
                        output.copy_from_slice(&first);
                    }
                    if output.iter().any(|v| !v.is_finite()) {
                        return Err(E::Invalid("nonfinite price or risk sample"));
                    }
                    Ok::<(), E>(())
                })
                .map_err(E::numerical)
        };
        let (statistics, units, estimator, paths) = match self.engine {
            EngineConfig::PseudoMonteCarlo(c) => {
                let units = c.independent_sampling_units().get();
                (
                    reduce(None, units, c.variance_reduction().antithetic())?,
                    units,
                    EstimatorKind::PseudoMonteCarlo,
                    c.evaluated_paths(),
                )
            }
            EngineConfig::RandomizedQuasiMonteCarlo(c) => {
                let points = c.points_per_scramble().get();
                let scrambles = c.scramble_count().get();
                let mut means = vec![Vec::with_capacity(scrambles as usize); layout.total];
                for s in 0..scrambles {
                    let stats = reduce(Some(s), points, c.variance_reduction().antithetic())?;
                    for (out, stat) in means.iter_mut().zip(stats) {
                        out.push(stat.sum().total() / points as f64);
                    }
                }
                let stats = means
                    .iter()
                    .map(|m| DeterministicStatistics::from_ordered_values_two_pass(m))
                    .collect();
                let paths = u128::from(points)
                    * u128::from(scrambles)
                    * if c.variance_reduction().antithetic() {
                        2
                    } else {
                        1
                    };
                (
                    stats,
                    u64::from(scrambles),
                    EstimatorKind::RandomizedQuasiMonteCarlo,
                    paths,
                )
            }
        };
        let estimate = |index, scale| {
            estimate_from_statistics(statistics[index], units, scale, estimator)
                .map_err(E::numerical)
        };
        let mut risks = Vec::new();
        let mut gamma = Vec::new();
        if let Some(config) = risk {
            for (i, a) in self.assets.iter().enumerate() {
                let (offset, count) = layout.vega[i];
                let (bs_vega, bs_scaled, local) = match a.model {
                    ModelSpec::BlackScholes(_) => (
                        Some(estimate(offset, 1.0)?),
                        Some(estimate(offset, 0.01)?),
                        Vec::new(),
                    ),
                    _ => (
                        None,
                        None,
                        (offset..offset + count)
                            .map(|k| estimate(k, 1.0))
                            .collect::<Result<Vec<_>, _>>()?,
                    ),
                };
                risks.push(MultiAssetRisk {
                    underlying: a.forward.underlying(),
                    delta: estimate(1 + i, 1.0)?,
                    delta_per_one_percent_spot: estimate(1 + i, 0.01 * a.forward.spot().get())?,
                    bs_vega,
                    bs_vega_per_vol_point: bs_scaled,
                    local_variance: local,
                    local_variance_time_nodes: match &a.model {
                        ModelSpec::LocalVolatility(lv) => {
                            lv.local_variance_grid().time_nodes().to_vec()
                        }
                        _ => Vec::new(),
                    },
                    local_variance_log_moneyness_nodes: match &a.model {
                        ModelSpec::LocalVolatility(lv) => {
                            lv.local_variance_grid().log_moneyness_nodes().to_vec()
                        }
                        _ => Vec::new(),
                    },
                });
            }
            if config.gamma_relative_bump.is_some() {
                for i in 0..self.assets.len() {
                    gamma.push(
                        (0..self.assets.len())
                            .map(|j| estimate(layout.gamma + i * self.assets.len() + j, 1.0))
                            .collect::<Result<Vec<_>, _>>()?,
                    );
                }
            }
        }
        let mut hash = blake3::Hasher::new();
        hash.update(self.fingerprint.as_bytes());
        match risk {
            None => {
                hash.update(b"price");
            }
            Some(r) => {
                hash.update(b"aad-fixed-log-forward-grid");
                if let Some(b) = r.gamma_relative_bump {
                    hash.update(b"gamma-central-delta");
                    hash.update(&b.to_bits().to_be_bytes());
                }
            }
        }
        Ok(MultiAssetPrice {
            price: estimate(0, 1.0)?,
            risks,
            gamma,
            gamma_relative_bump: risk.and_then(|r| r.gamma_relative_bump),
            underlyings: self.underlyings(),
            fingerprint: hash.finalize().to_hex().to_string(),
            evaluated_paths: paths,
            worker_threads: self.execution.worker_threads().get(),
            reduction_block_size: self.execution.reduction_block_size().get(),
            direction_checksum: self.qmc.as_ref().map(|q| q.direction_checksum()),
            scramble_checksum: self.qmc.as_ref().map(|q| q.scramble_checksum()),
            local_variance_boundary_counts: (0..self.assets.len())
                .map(|i| statistics[layout.boundary + i].sum().total() / units as f64)
                .collect(),
        })
    }
    fn path_values(
        &self,
        paths: &[AssetPath],
        risk: Option<MultiAssetRiskConfig>,
        layout: &Layout,
    ) -> Result<Vec<f64>, E> {
        let mut values = vec![0.0; layout.total];
        for (i, p) in paths.iter().enumerate() {
            values[layout.boundary + i] = p
                .local
                .as_ref()
                .map_or(0.0, |p| p.boundary_stats().total_flat_count() as f64);
        }
        let Some(config) = risk else {
            values[0] = self.payoff.evaluate_with_pre_dividend_spots(
                |u, d| self.observe(paths, u, d, false, None),
                |u, d| self.observe(paths, u, d, true, None),
            )?[0];
            return Ok(values);
        };
        let payoff = self.payoff_adjoints(paths, None)?;
        values[0] = payoff.value;
        let deltas = self.delta(paths, &payoff);
        values[1..1 + paths.len()].copy_from_slice(&deltas);
        let mut seeds = vec![vec![0.0; self.times.len()]; paths.len()];
        for (pre, u, d, value) in payoff
            .terminal_adjoints
            .iter()
            .map(|s| (false, s.underlying, s.observation_date, s.value))
            .chain(
                payoff
                    .pre_dividend_adjoints
                    .iter()
                    .map(|s| (true, s.underlying, s.observation_date, s.value)),
            )
        {
            let (i, j) = self.indices(u, d).expect("validated observation");
            let a = &self.assets[i];
            seeds[i][j] += value
                * if pre {
                    a.pre_coordinates[j].b()
                } else {
                    a.coordinates[j].b()
                };
        }
        for (i, (p, a)) in paths.iter().zip(&self.assets).enumerate() {
            let (offset, count) = layout.vega[i];
            match &a.model {
                ModelSpec::BlackScholes(_) => {
                    values[offset] = seeds[i].iter().zip(&p.bs_vega).map(|(s, v)| s * v).sum();
                }
                ModelSpec::LocalVolatility(lv) => {
                    let g = lv.local_variance_grid();
                    let adj = p
                        .local
                        .as_ref()
                        .expect("LV path")
                        .reverse_state_adjoints(
                            &seeds[i],
                            g.values().len(),
                            g.log_moneyness_nodes().len(),
                        )
                        .map_err(E::numerical)?;
                    values[offset..offset + count]
                        .copy_from_slice(adj.local_variance_value_adjoints());
                }
                _ => unreachable!("validated model"),
            }
        }
        if let Some(relative) = config.gamma_relative_bump {
            for (j, a) in self.assets.iter().enumerate() {
                let h = relative * a.forward.spot().get();
                if !h.is_finite()
                    || h <= 0.0
                    || a.forward.spot().get() + h == a.forward.spot().get()
                    || a.forward.spot().get() - h == a.forward.spot().get()
                {
                    return Err(E::Invalid("unrepresentable Gamma Spot bump"));
                }
                let up = self.delta(paths, &self.payoff_adjoints(paths, Some((j, h)))?);
                let down = self.delta(paths, &self.payoff_adjoints(paths, Some((j, -h)))?);
                for i in 0..paths.len() {
                    values[layout.gamma + i * paths.len() + j] = (up[i] - down[i]) / (2.0 * h);
                }
            }
        }
        Ok(values)
    }
}
