//! Conditional continuation Gamma on selected rows; joint same-path readouts.
use super::vegakt_terminal::try_affine_observation_payoff_gamma_measure;
use crate::MonteCarloError;
use crate::engine::plan::simulation::SimulationPlan;
use crate::engine::sampling::path_normals::local_vol_rqmc_shocks;
use crate::mc::{
    BrownianBridgePlan, DeterministicExecutor, EngineConfig, RandomDomain, RqmcConfig, RqmcPlan,
};
use crate::risk::{VegaKtPathGammaChainAudit, VegaKtPathGammaRow, VegaKtStepMoment};
use pricing_numerics::NeumaierSum;
use pricing_numerics::standard_normal_pdf as pdf;
use rayon::prelude::*;
mod bound;
mod conditional_ad;
mod measure;
mod statistics;
use measure::{CurveInput, Slice, call, invalid, sum};

fn upper(m: &VegaKtStepMoment) -> f64 {
    m.mean().abs() + 6. * m.standard_error().unwrap()
}
fn norm(v: impl Iterator<Item = f64>) -> f64 {
    v.fold(0., f64::hypot)
}

impl SimulationPlan {
    pub(in crate::engine) fn evaluate_vega_kt_multi_step_gamma_chain_audit(
        &self,
        rows: Vec<usize>,
        boundaries: Vec<f64>,
        directions: Vec<Vec<Vec<f64>>>,
        subdivisions: usize,
        relative_tolerance: f64,
        gamma_se_limit: f64,
    ) -> Result<VegaKtPathGammaChainAudit, MonteCarloError> {
        self.evaluate_vega_kt_multi_step_gamma_chain_with_samples(
            rows,
            boundaries,
            directions,
            subdivisions,
            relative_tolerance,
            gamma_se_limit,
            false,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::engine) fn evaluate_vega_kt_multi_step_gamma_chain_with_samples(
        &self,
        rows: Vec<usize>,
        boundaries: Vec<f64>,
        directions: Vec<Vec<Vec<f64>>>,
        subdivisions: usize,
        relative_tolerance: f64,
        gamma_se_limit: f64,
        retain_samples: bool,
    ) -> Result<VegaKtPathGammaChainAudit, MonteCarloError> {
        let lv = self.state_only_path_gamma_runtime()?;
        let times = lv.plan.time_grid().nodes();
        let nodes = lv.grid.log_moneyness_nodes();
        let n = times.len() - 1;
        let m = nodes.len();
        let b = boundaries.len() + 1;
        let d = directions.len();
        if !(2..=16).contains(&n)
            || times != lv.grid.time_nodes()
            || !(2..=33).contains(&m)
            || rows.is_empty()
            || rows.len() > n
            || rows.iter().any(|r| *r >= n)
            || rows.windows(2).any(|p| p[0] >= p[1])
            || !(16..=1024).contains(&subdivisions)
            || !(1e-10..=1e-4).contains(&relative_tolerance)
            || !(0.01..=0.5).contains(&gamma_se_limit)
            || boundaries.is_empty()
            || boundaries.len() > 33
            || boundaries.iter().any(|x| !x.is_finite() || *x <= 0.)
            || boundaries.windows(2).any(|p| p[0] >= p[1])
            || !(1..=32).contains(&d)
            || directions.iter().any(|w| {
                w.len() != rows.len()
                    || w.iter()
                        .any(|v| v.len() != m || v.iter().any(|x| !x.is_finite()))
            })
        {
            return Err(invalid(
                "invalid conditional path Gamma grid, rows, directions or precision policy",
            )
            .into());
        }
        let (count, points, reduction) = match self.engine {
            EngineConfig::PseudoMonteCarlo(engine) => (
                engine.independent_sampling_units().get(),
                1,
                engine.variance_reduction(),
            ),
            EngineConfig::RandomizedQuasiMonteCarlo(engine) => (
                u64::from(engine.scramble_count().get()),
                engine.points_per_scramble().get(),
                engine.variance_reduction(),
            ),
        };
        let antithetic = reduction.antithetic();
        let physical_units = u128::from(count) * u128::from(points);
        let width = 8 + 2 * b + 8 * m;
        let components = 1 + rows.len() * width + 4 * d;
        let covariance_dimension = 3 * d;
        let covariance_components = covariance_dimension * (covariance_dimension + 1) / 2;
        let work = physical_units
            * (1 + u128::from(antithetic))
            * rows.iter().map(|r| (n - r) as u128).sum::<u128>()
            * (4 * (subdivisions + m + b + 3)) as u128
            * self.payoff.opcodes().len() as u128;
        if !(2..=262144).contains(&count)
            || physical_units > 262144
            || (matches!(self.engine, EngineConfig::RandomizedQuasiMonteCarlo(_))
                && u128::from(count) * components as u128 > 4_000_000)
            || count.div_ceil(self.execution_policy.reduction_block_size().get()) as u128
                * (components + covariance_components) as u128
                > 4_000_000
            || work > 10_000_000_000
        {
            return Err(invalid(
                "conditional Gamma chain sample, storage or continuation budget exceeded",
            )
            .into());
        }
        let observations =
            self.local_vol_state_observations(lv, &vec![self.spot; n + 1], self.spot)?;
        let node_for = |u, date| {
            if u != self.underlying {
                None
            } else {
                self.observation_dates
                    .iter()
                    .position(|d| *d == Some(date))
                    .map(|i| observations[i].node_index)
            }
        };
        let envelope = bound::growth_bound(&self.payoff, node_for, self.spot)?;
        let scale = envelope[0] + self.spot * sum(envelope[1..].iter().copied());
        if !scale.is_finite() || scale <= 0. {
            return Err(invalid("positive finite payoff growth scale required").into());
        }
        let tolerance = scale * relative_tolerance;
        let minmax = lv
            .grid
            .values()
            .chunks_exact(m)
            .take(n)
            .map(|v| {
                (
                    v.iter().copied().fold(f64::INFINITY, f64::min),
                    v.iter().copied().fold(0., f64::max),
                )
            })
            .collect::<Vec<_>>();
        let terminal_payoff = |states: &[f64]| {
            try_affine_observation_payoff_gamma_measure(&self.payoff, |u, date| {
                let i = node_for(u, date)?;
                if i == n {
                    Some((0., 1.))
                } else {
                    Some((states[i], 0.))
                }
            })?
            .ok_or_else(|| invalid("piecewise affine terminal payoff required"))
        };
        let integrate_native = |path: &crate::mc::LocalVolPath| {
            let boundaries_ad = super::vegakt_terminal::conditional_payoff_branch_boundaries(
                &self.payoff,
                |u, date| {
                    let i = node_for(u, date)?;
                    Some(if i == n {
                        (0., 1.)
                    } else {
                        (path.states()[i], 0.)
                    })
                },
            )?;
            let mut conditional_states = path.states().to_vec();
            conditional_ad::integrate(path, lv.grid.values().len(), m, &boundaries_ad, |last| {
                conditional_states[n] = last;
                self.chain_discounted_payoff_and_seeds(lv, &conditional_states)
                    .map(|(_, seeds)| seeds)
            })
        };
        let sample = |z: &[f64]| -> Result<Vec<f64>, MonteCarloError> {
            let path = lv.plan.evolve_path(&lv.grid, self.spot, z)?;
            let (price, seeds) = self.chain_discounted_payoff_and_seeds(lv, path.states())?;
            let ad = path.reverse_state_adjoints(&seeds, lv.grid.values().len(), m)?;
            let conditional = integrate_native(&path)?;
            let mut out = vec![price];
            let mut risks = Vec::new();
            let mut errors = Vec::new();
            for &row in &rows {
                let source = path.states()[row];
                let x = (source / self.spot).ln();
                let dt = times[row + 1] - times[row];
                let variance = lv.grid.interpolate(times[row], x)?.value * dt;
                if !variance.is_finite() || variance <= 0. || variance.sqrt() > 4. {
                    return Err(invalid("conditional variance outside precision domain").into());
                }
                let mut conditional_ad_variance = 0.;
                let (coarse, fine) = if row == n - 1 {
                    let payoff = terminal_payoff(path.states())?;
                    let mut masses = vec![0.; b];
                    let mut bins = vec![0.; b];
                    let mut value = payoff.cash() + payoff.slope() * source;
                    let mut negative = 0.;
                    for atom in payoff.gamma().atoms() {
                        let i = boundaries.partition_point(|v| *v <= atom.strike());
                        masses[i] += atom.mass();
                        bins[i] += atom.mass()
                            * source
                            * pdf((source.ln() - atom.strike().ln()) / variance.sqrt()
                                + variance.sqrt() / 2.)
                            * dt
                            / (2. * variance.sqrt());
                        value += atom.mass() * call(source, atom.strike(), variance);
                        if atom.mass() < 0. {
                            negative += 1.;
                        }
                    }
                    let slice = Slice {
                        price: value,
                        risk: sum(bins.iter().copied()),
                        masses,
                        bin_risks: bins,
                        tail_price: 0.,
                        tail_risk: 0.,
                        negative,
                    };
                    conditional_ad_variance =
                        measure::terminal_ad_variance(&payoff, source, variance, dt, slice.risk)?;
                    (slice.clone(), slice)
                } else {
                    let a = envelope[0] + sum((1..=row).map(|i| envelope[i] * path.states()[i]));
                    let mut growth = envelope[row + 1];
                    let mut multiplier = 1.;
                    for j in row + 1..n {
                        // The last transition is integrated analytically: E[S_N|S_(N-1)]=S_(N-1).
                        if j < n - 1 {
                            let delta = times[j + 1] - times[j];
                            let sd = z[j]
                                .clamp((minmax[j].0 * delta).sqrt(), (minmax[j].1 * delta).sqrt());
                            multiplier *= (-sd * sd / 2. + sd * z[j]).exp();
                        }
                        growth += envelope[j + 1] * multiplier;
                    }
                    if !growth.is_finite() {
                        return Err(invalid("nonfinite conditional growth envelope").into());
                    }
                    let mut states = path.states().to_vec();
                    let continuation = |s: f64| -> Result<f64, MonteCarloError> {
                        states[row + 1] = s;
                        for j in row + 1..n - 1 {
                            let v = lv
                                .grid
                                .interpolate(times[j], (states[j] / self.spot).ln())?
                                .value
                                * (times[j + 1] - times[j]);
                            states[j + 1] = states[j] * (-v / 2. + v.sqrt() * z[j]).exp();
                            if !states[j + 1].is_finite() || states[j + 1] <= 0. {
                                return Err(
                                    invalid("nonfinite conditional continuation state").into()
                                );
                            }
                        }
                        let payoff = terminal_payoff(&states)?;
                        let s = states[n - 1];
                        let v = lv
                            .grid
                            .interpolate(times[n - 1], (s / self.spot).ln())?
                            .value
                            * (times[n] - times[n - 1]);
                        let value = sum(std::iter::once(payoff.cash())
                            .chain(std::iter::once(payoff.slope() * s))
                            .chain(
                                payoff
                                    .gamma()
                                    .atoms()
                                    .iter()
                                    .map(|a| a.mass() * call(s, a.strike(), v)),
                            ));
                        if !value.is_finite() {
                            return Err(invalid("nonfinite conditional continuation price").into());
                        }
                        Ok(value)
                    };
                    measure::recover(
                        &CurveInput {
                            source,
                            variance,
                            dt,
                            growth: [a, growth],
                            spot: self.spot,
                            nodes,
                            boundaries: &boundaries,
                            subdivisions,
                        },
                        continuation,
                    )?
                };
                let i = nodes
                    .partition_point(|v| *v <= x)
                    .saturating_sub(1)
                    .min(m - 2);
                let h = ((x - nodes[i]) / (nodes[i + 1] - nodes[i])).clamp(0., 1.);
                let mut risk = vec![0.; m];
                risk[i] = (1. - h) * fine.risk;
                risk[i + 1] = h * fine.risk;
                let mut error = vec![0.; m];
                let refinement = (fine.risk - coarse.risk).abs();
                error[i] = (1. - h) * (refinement + fine.tail_risk);
                error[i + 1] = h * (refinement + fine.tail_risk);
                let native = &ad.local_variance_value_adjoints()[row * m..(row + 1) * m];
                out.extend([
                    fine.price,
                    fine.price - price,
                    refinement,
                    (fine.price - coarse.price).abs(),
                    fine.tail_price,
                    fine.tail_risk,
                    fine.negative,
                    norm(fine.masses.iter().zip(&coarse.masses).map(|(a, b)| a - b)),
                ]);
                out.extend(&fine.masses);
                out.extend(&fine.bin_risks);
                out.extend(&risk);
                out.extend(native);
                out.extend(risk.iter().zip(native).map(|(a, b)| a - b));
                let mut conditional_variances = vec![0.; m];
                conditional_variances[i] = (1. - h).powi(2) * conditional_ad_variance;
                conditional_variances[i + 1] = h * h * conditional_ad_variance;
                out.extend(conditional_variances);
                let integrated = &conditional.means[row * m..(row + 1) * m];
                out.extend(integrated);
                out.extend(risk.iter().zip(integrated).map(|(a, b)| a - b));
                out.extend(&conditional.variances[row * m..(row + 1) * m]);
                out.extend(native.iter().zip(integrated).map(|(a, b)| a - b));
                risks.push(risk);
                errors.push(error);
            }
            let mut native = Vec::new();
            let mut recovered = Vec::new();
            let mut allowance = Vec::new();
            for direction in &directions {
                native.push(sum(direction.iter().zip(&rows).flat_map(|(v, row)| {
                    v.iter()
                        .zip(&ad.local_variance_value_adjoints()[row * m..(row + 1) * m])
                        .map(|(a, b)| a * b)
                })));
                recovered.push(sum(direction
                    .iter()
                    .zip(&risks)
                    .flat_map(|(v, risk)| v.iter().zip(risk).map(|(a, b)| a * b))));
                allowance
                    .push(sum(direction.iter().zip(&errors).flat_map(|(v, error)| {
                        v.iter().zip(error).map(|(a, b)| a.abs() * b)
                    })));
            }
            out.extend(&native);
            out.extend(&recovered);
            out.extend(recovered.iter().zip(native).map(|(a, b)| a - b));
            out.extend(allowance);
            Ok(out)
        };
        let direction_start = 1 + rows.len() * width;
        let retained_indices = if retain_samples {
            std::iter::once(0)
                .chain(direction_start + d..direction_start + 2 * d)
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        let paired = |z: &[f64]| -> Result<Vec<f64>, MonteCarloError> {
            let mut values = sample(z)?;
            if antithetic {
                let mate = sample(&z.iter().map(|x| -x).collect::<Vec<_>>())?;
                for (a, b) in values.iter_mut().zip(mate) {
                    *a = 0.5 * *a + 0.5 * b;
                }
            }
            Ok(values)
        };
        // Increase only the cheap conditional-AD reference resolution. Gamma
        // recovery and all tolerances are unchanged. Pair both estimators by
        // scramble, so their covariance remains empirical across independent
        // replicates even when their nested Sobol prefixes differ in length.
        let conditional_points = if rows.iter().all(|row| *row == n - 1) {
            points
        } else {
            points * (1_u64 << (262144 / count / points).min(64).ilog2())
        };
        if matches!(self.engine, EngineConfig::RandomizedQuasiMonteCarlo(_))
            && u128::from(count)
                * u128::from(conditional_points)
                * (1 + u128::from(antithetic))
                * (n * m * self.payoff.opcodes().len()) as u128
                > 10_000_000_000
        {
            return Err(invalid("conditional AD reference work budget exceeded").into());
        }
        // A scramble, not an individual Sobol point, is the independent unit.
        // Compute scrambles in parallel; keep their index order for covariance.
        let rqmc_values = if let EngineConfig::RandomizedQuasiMonteCarlo(engine) = self.engine {
            let qmc = RqmcPlan::compile(
                RqmcConfig::new(
                    conditional_points,
                    engine.scramble_count().get(),
                    engine.master_scramble_seed(),
                    reduction,
                )
                .map_err(|_| invalid("invalid conditional AD reference configuration"))?,
                n as u32,
            )?;
            let bridge = if reduction.brownian_bridge() {
                Some(
                    BrownianBridgePlan::compile(times.to_vec(), 1)
                        .map_err(|error| MonteCarloError::LocalVol(error.into()))?,
                )
            } else {
                None
            };
            let executor = DeterministicExecutor::new(self.execution_policy)?;
            Some(
                executor
                    .install(|| {
                        (0..engine.scramble_count().get())
                            .into_par_iter()
                            .map(|scramble| {
                                let mut totals = vec![NeumaierSum::new(); components];
                                for point in 0..points {
                                    let z = local_vol_rqmc_shocks(
                                        &qmc,
                                        bridge.as_ref(),
                                        scramble,
                                        point,
                                    )?;
                                    for (total, value) in totals.iter_mut().zip(paired(&z)?) {
                                        total.add(value);
                                    }
                                }
                                let mut means = totals
                                    .into_iter()
                                    .map(|total| total.total() / points as f64)
                                    .collect::<Vec<_>>();
                                if conditional_points > points {
                                    let mut reference =
                                        vec![NeumaierSum::new(); 2 * rows.len() * m];
                                    for point in 0..conditional_points {
                                        let z = local_vol_rqmc_shocks(
                                            &qmc,
                                            bridge.as_ref(),
                                            scramble,
                                            point,
                                        )?;
                                        let path = lv.plan.evolve_path(&lv.grid, self.spot, &z)?;
                                        let first = integrate_native(&path)?;
                                        let mate = if antithetic {
                                            let z = z.iter().map(|v| -v).collect::<Vec<_>>();
                                            Some(integrate_native(
                                                &lv.plan.evolve_path(&lv.grid, self.spot, &z)?,
                                            )?)
                                        } else {
                                            None
                                        };
                                        for (j, &row) in rows.iter().enumerate() {
                                            for k in 0..m {
                                                let i = row * m + k;
                                                let (mean, variance) = if let Some(mate) = &mate {
                                                    (
                                                        0.5 * first.means[i] + 0.5 * mate.means[i],
                                                        0.5 * first.variances[i]
                                                            + 0.5 * mate.variances[i],
                                                    )
                                                } else {
                                                    (first.means[i], first.variances[i])
                                                };
                                                reference[2 * j * m + k].add(mean);
                                                reference[(2 * j + 1) * m + k].add(variance);
                                            }
                                        }
                                    }
                                    for j in 0..rows.len() {
                                        // At the terminal row Gamma and AD integrate the
                                        // identical last shock. Compare those two formulas
                                        // path by path on the same prefix; different sample
                                        // sizes would add noise to an exact conditional identity.
                                        if rows[j] == n - 1 {
                                            continue;
                                        }
                                        let offset = 1 + j * width + 8 + 2 * b;
                                        for k in 0..m {
                                            let mean = reference[2 * j * m + k].total()
                                                / conditional_points as f64;
                                            means[offset + 4 * m + k] = mean;
                                            means[offset + 5 * m + k] = means[offset + k] - mean;
                                            means[offset + 6 * m + k] =
                                                reference[(2 * j + 1) * m + k].total()
                                                    / conditional_points as f64;
                                            means[offset + 7 * m + k] =
                                                means[offset + m + k] - mean;
                                        }
                                    }
                                }
                                Ok::<_, MonteCarloError>(means)
                            })
                            .collect::<Vec<_>>()
                    })
                    .into_iter()
                    .collect::<Result<Vec<_>, _>>()?,
            )
        } else {
            None
        };
        let (values, joint_sample_covariance, sampling_unit_values) =
            statistics::evaluate_retaining(
                self.execution_policy,
                count,
                components,
                direction_start..direction_start + covariance_dimension,
                &retained_indices,
                |index| match self.engine {
                    EngineConfig::PseudoMonteCarlo(engine) => {
                        let z = lv.plan.path_shocks(
                            engine.master_seed(),
                            index,
                            RandomDomain::Valuation,
                            reduction.brownian_bridge(),
                        )?;
                        paired(&z)
                    }
                    EngineConfig::RandomizedQuasiMonteCarlo(_) => Ok(rqmc_values
                        .as_ref()
                        .expect("compiled RQMC scrambles")[index as usize]
                        .clone()),
                },
            )?;
        let mut audits = Vec::new();
        for (j, &row) in rows.iter().enumerate() {
            let a = &values[1 + j * width..1 + (j + 1) * width];
            let masses = a[8..8 + b].to_vec();
            let q = norm(masses.iter().map(|v| v.mean()));
            let qse = norm(masses.iter().map(|v| v.standard_error().unwrap()));
            let relative = if q == 0. && qse == 0. {
                Some(0.)
            } else if q > 0. {
                (qse / q).is_finite().then_some(qse / q)
            } else {
                None
            };
            let recovery = upper(&a[2]) + upper(&a[5]) <= tolerance
                && upper(&a[3]) + upper(&a[4]) <= tolerance
                && relative.is_some_and(|x| x <= gamma_se_limit);
            let offset = 8 + 2 * b;
            let diffs = a[offset + 2 * m..offset + 3 * m].to_vec();
            let conditional_variances = a[offset + 3 * m..offset + 4 * m].to_vec();
            let node_se = diffs
                .iter()
                .zip(&conditional_variances)
                .map(|(v, c)| {
                    v.standard_error()
                        .unwrap()
                        .max((upper(c) / physical_units as f64).sqrt())
                })
                .collect::<Vec<_>>();
            let raw_validation = a[1].mean().abs()
                <= 6. * a[1].standard_error().unwrap() + tolerance
                && diffs
                    .iter()
                    .zip(&node_se)
                    .all(|(x, se)| x.mean().abs() <= 6. * se + tolerance);
            let integrated = a[offset + 4 * m..offset + 5 * m].to_vec();
            let conditional_differences = a[offset + 5 * m..offset + 6 * m].to_vec();
            let raw_variances = a[offset + 6 * m..offset + 7 * m].to_vec();
            let raw_differences = a[offset + 7 * m..offset + 8 * m].to_vec();
            let raw_se = raw_differences
                .iter()
                .zip(&raw_variances)
                .map(|(v, c)| {
                    v.standard_error()
                        .unwrap()
                        .max((upper(c) / physical_units as f64).sqrt())
                })
                .collect::<Vec<_>>();
            let conditional_gamma_passed = conditional_differences
                .iter()
                .all(|v| v.mean().abs() <= 6. * v.standard_error().unwrap() + tolerance);
            let raw_conditional_passed = raw_differences
                .iter()
                .zip(&raw_se)
                .all(|(v, se)| v.mean().abs() <= 6. * se + tolerance);
            let validation = a[1].mean().abs() <= 6. * a[1].standard_error().unwrap() + tolerance
                && conditional_gamma_passed
                && raw_conditional_passed
                && (row != n - 1 || raw_validation);
            audits.push(VegaKtPathGammaRow {
                variance_row: row,
                recovered_price: a[0],
                price_difference: a[1],
                variance_refinement: a[2],
                price_refinement: a[3],
                price_tail_bound: a[4],
                variance_tail_bound: a[5],
                negative_atoms: a[6],
                bin_mass_refinement: a[7],
                gamma_bin_masses: masses,
                gamma_bin_variance_sensitivities: a[8 + b..offset].to_vec(),
                recovered_variance_nodes: a[offset..offset + m].to_vec(),
                native_variance_nodes: a[offset + m..offset + 2 * m].to_vec(),
                node_differences: diffs,
                terminal_ad_conditional_variances: conditional_variances,
                node_validation_standard_errors: node_se,
                raw_node_validation_passed: raw_validation,
                conditional_native_variance_nodes: integrated,
                conditional_node_differences: conditional_differences,
                raw_ad_conditional_variances: raw_variances,
                raw_conditional_node_differences: raw_differences,
                raw_conditional_standard_errors: raw_se,
                conditional_gamma_validation_passed: conditional_gamma_passed,
                raw_conditional_validation_passed: raw_conditional_passed,
                gamma_relative_standard_error: relative,
                recovery_passed: recovery,
                validation_passed: validation,
            });
        }
        let start = 1 + rows.len() * width;
        let native = values[start..start + d].to_vec();
        let recovered = values[start + d..start + 2 * d].to_vec();
        let differences = values[start + 2 * d..start + 3 * d].to_vec();
        let allowances = values[start + 3 * d..].to_vec();
        let direction_norm = norm(recovered.iter().map(|v| v.mean()));
        let direction_se = norm(recovered.iter().map(|v| v.standard_error().unwrap()));
        let direction_relative = if direction_norm == 0. && direction_se == 0. {
            Some(0.)
        } else if direction_norm > 0. {
            (direction_se / direction_norm)
                .is_finite()
                .then_some(direction_se / direction_norm)
        } else {
            None
        };
        let recovery = audits.iter().all(|r| r.recovery_passed)
            && direction_relative.is_some_and(|v| v <= gamma_se_limit);
        let validation = audits.iter().all(|r| r.validation_passed)
            && differences.iter().zip(&allowances).all(|(v, e)| {
                v.mean().abs()
                    <= 6. * v.standard_error().unwrap() + upper(e) + 256. * f64::EPSILON * scale
            });
        Ok(VegaKtPathGammaChainAudit {
            request_fingerprint: self.request_fingerprint(),
            plan_fingerprint: self.plan_fingerprint(),
            time_nodes: times.to_vec(),
            gamma_bin_boundaries: boundaries,
            variance_risk_directions: directions,
            subdivisions,
            relative_tolerance,
            functional_tolerance: tolerance,
            gamma_relative_standard_error_limit: gamma_se_limit,
            independent_sampling_units: count,
            points_per_scramble: match self.engine {
                EngineConfig::PseudoMonteCarlo(_) => None,
                EngineConfig::RandomizedQuasiMonteCarlo(_) => Some(points),
            },
            conditional_ad_points_per_scramble: match self.engine {
                EngineConfig::PseudoMonteCarlo(_) => None,
                EngineConfig::RandomizedQuasiMonteCarlo(_) => Some(conditional_points),
            },
            antithetic,
            native_price: values[0],
            rows: audits,
            native_directions: native,
            recovered_directions: recovered.clone(),
            direction_differences: differences,
            direction_refinement_allowances: allowances,
            direction_relative_standard_error: direction_relative,
            joint_sample_covariance,
            sampling_unit_values: retain_samples.then_some(sampling_unit_values),
            recovery_passed: recovery,
            validation_passed: validation,
            estimate: (recovery && validation).then_some(recovered),
        })
    }
}
