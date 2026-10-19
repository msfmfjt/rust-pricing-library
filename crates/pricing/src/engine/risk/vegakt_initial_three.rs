//! Initial-state Gamma: both subsequent transitions and the first fixing move.
use super::vegakt_terminal::try_affine_observation_payoff_gamma_measure;
use crate::MonteCarloError;
use crate::engine::plan::simulation::SimulationPlan;
use crate::mc::{DeterministicExecutor, EngineConfig, RandomDomain};
use crate::risk::{
    RiskConfigError, VegaKtInitialSmileGammaAudit, VegaKtInitialSmileResolution,
    VegaKtThreeStepInitialGammaAudit, gaussian_legendre,
};
use pricing_numerics::{NeumaierSum, standard_normal_cdf as cdf, standard_normal_pdf as pdf};
mod bound;
fn invalid(s: &'static str) -> RiskConfigError {
    RiskConfigError::VegaKtChainInput { quantity: s }
}
fn sum(v: impl Iterator<Item = f64>) -> f64 {
    v.collect::<NeumaierSum>().total()
}
fn call(s: f64, k: f64, v: f64) -> f64 {
    let sd = v.sqrt();
    let d1 = (s.ln() - k.ln()) / sd + sd / 2.;
    let d2 = d1 - sd;
    if s >= k {
        s - k + k * cdf(-d2) - s * cdf(-d1)
    } else {
        s * cdf(d1) - k * cdf(d2)
    }
}
fn tail(sd: f64) -> f64 {
    cdf(-12. - sd) + cdf(-12. + sd)
}
fn tail_moment(sd: f64) -> f64 {
    (sd * sd + 2.) * tail(sd) + (12. + sd) * (pdf(12. - sd) + pdf(12. + sd))
}
fn cuts(mut v: Vec<f64>) -> Vec<f64> {
    v.sort_by(f64::total_cmp);
    v.dedup_by(|a, b| (*a - *b).abs() <= 32. * f64::EPSILON * (1. + a.abs() + b.abs()));
    v
}
fn diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(a, b)| (a - b).abs())
        .fold(0., f64::max)
}
fn extrap(a: f64, b: f64, c: f64) -> f64 {
    (a - 6. * b + 8. * c) / 3.
}
fn extrapolate(
    a: &VegaKtInitialSmileResolution,
    b: &VegaKtInitialSmileResolution,
    c: &VegaKtInitialSmileResolution,
) -> VegaKtInitialSmileResolution {
    let combine = |a: &[f64], b: &[f64], c: &[f64]| {
        a.iter()
            .zip(b)
            .zip(c)
            .map(|((a, b), c)| extrap(*a, *b, *c))
            .collect::<Vec<_>>()
    };
    let masses = combine(
        &a.gamma_bin_masses,
        &b.gamma_bin_masses,
        &c.gamma_bin_masses,
    );
    let risks = combine(
        &a.gamma_bin_variance_sensitivities,
        &b.gamma_bin_variance_sensitivities,
        &c.gamma_bin_variance_sensitivities,
    );
    VegaKtInitialSmileResolution {
        subdivisions: c.subdivisions,
        interpolation_nodes: c.interpolation_nodes,
        total_gamma_mass: sum(masses.iter().copied()),
        recovered_variance_sensitivity: sum(risks.iter().copied()),
        gamma_bin_masses: masses,
        gamma_bin_variance_sensitivities: risks,
        interpolated_price: extrap(
            a.interpolated_price,
            b.interpolated_price,
            c.interpolated_price,
        ),
    }
}
impl SimulationPlan {
    pub(in crate::engine) fn evaluate_vega_kt_three_step_initial_gamma_audit(
        &self,
        boundaries: Vec<f64>,
        subdivisions: usize,
        relative_tolerance: f64,
    ) -> Result<VegaKtThreeStepInitialGammaAudit, MonteCarloError> {
        let lv = self.state_only_path_gamma_runtime()?;
        let times = lv.plan.time_grid().nodes();
        let nodes = lv.grid.log_moneyness_nodes();
        let width = nodes.len();
        if times.len() != 4
            || times != lv.grid.time_nodes()
            || !(2..=33).contains(&width)
            || lv.grid.values()[..width]
                .iter()
                .any(|v| *v != lv.grid.values()[0])
        {
            return Err(invalid("three matching intervals, flat first variance and 2..33 nodes required for initial Gamma").into());
        }
        if !(64..=4096).contains(&subdivisions)
            || !(1e-12..=1e-4).contains(&relative_tolerance)
            || boundaries.is_empty()
            || boundaries.len() > 33
            || boundaries.iter().any(|x| !x.is_finite() || *x <= 0.)
            || boundaries.windows(2).any(|x| x[0] >= x[1])
        {
            return Err(invalid("three-step initial Gamma needs 64..4096 subdivisions, 1..33 positive ordered boundaries and tolerance in [1e-12,1e-4]").into());
        }
        let EngineConfig::PseudoMonteCarlo(engine) = self.engine else {
            return Err(invalid("pseudo-MC required for initial Gamma validation").into());
        };
        let count = engine.independent_sampling_units().get();
        if !(2..=10_000_000).contains(&count)
            || count.div_ceil(self.execution_policy.reduction_block_size().get()) as u128 * 4
                > 4_000_000
        {
            return Err(invalid("initial Gamma sample or storage budget exceeded").into());
        }
        let work = (8 * subdivisions + 120 * (width + boundaries.len() + 10)) as u128
            * 96
            * (width + 12) as u128
            * self.payoff.opcodes().len() as u128;
        if work > 2_000_000_000 {
            return Err(invalid("three-step initial Gamma work budget exceeded").into());
        }
        let observations = self.local_vol_state_observations(lv, &[self.spot; 4], self.spot)?;
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
        let growth = [envelope[0], envelope[1] + envelope[2] + envelope[3]];
        let scale = growth[0] + growth[1] * self.spot;
        let functional_tolerance = relative_tolerance * scale;
        let mass_tolerance = relative_tolerance.sqrt() * (growth[0] / self.spot + growth[1]);
        let maturity = times[1];
        let v0 = lv.grid.values()[0] * maturity;
        let sd = v0.sqrt();
        let mu = -v0 / 2.;
        let window = [mu - 12. * sd, mu + 12. * sd];
        let dt2 = times[2] - times[1];
        let dt3 = times[3] - times[2];
        let max_v2 = lv.grid.values()[width..2 * width]
            .iter()
            .copied()
            .fold(0., f64::max)
            * dt2;
        let min_v2 = lv.grid.values()[width..2 * width]
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min)
            * dt2;
        if min_v2 <= 0.
            || !min_v2.is_finite()
            || !scale.is_finite()
            || !v0.is_finite()
            || v0 <= 0.
            || sd > 4.
            || !max_v2.is_finite()
            || max_v2.sqrt() > 4.
        {
            return Err(
                invalid("initial Gamma growth or variance precision domain invalid").into(),
            );
        }
        let factor = maturity * (1. / (2. * v0) + 1. / (4. * sd));
        let a = envelope[0] * tail(0.);
        let b = envelope[1] * tail(0.) + (envelope[2] + envelope[3]) * tail(max_v2.sqrt());
        let inner_tail_price = a + b * self.spot;
        let inner_tail_risk = factor * (2. * a + b * self.spot * (v0 + 2.));
        let mut outer = vec![window[0], window[1]];
        outer.extend(nodes);
        outer.extend(boundaries.iter().map(|s| (s / self.spot).ln()));
        for second in [(0., 0.), (self.spot, 0.), (0., 1.)] {
            for third in [(0., 0.), (self.spot, 0.), (0., 1.)] {
                let measure =
                    try_affine_observation_payoff_gamma_measure(&self.payoff, |u, date| {
                        match node_for(u, date)? {
                            0 => Some((self.spot, 0.)),
                            1 => Some((0., 1.)),
                            2 => Some(second),
                            3 => Some(third),
                            _ => None,
                        }
                    })?
                    .ok_or_else(|| invalid("piecewise affine initial payoff required"))?;
                outer.extend(
                    measure
                        .gamma()
                        .atoms()
                        .iter()
                        .map(|a| (a.strike() / self.spot).ln()),
                );
            }
        }
        let outer = cuts(
            outer
                .into_iter()
                .filter(|x| window[0] <= *x && *x <= window[1])
                .collect(),
        );
        if outer.len() < 2 {
            return Err(invalid(
                "initial Gaussian integration window is below floating-point resolution",
            )
            .into());
        }
        let rules = [gaussian_legendre(32)?, gaussian_legendre(64)?];
        let mut evaluations = 0usize;
        let mut inner_refinement = 0f64;
        let mut continuation = |x: f64| -> Result<f64, MonteCarloError> {
            let s1 = self.spot * x.exp();
            let variance = lv.grid.interpolate(times[1], x)?.value * dt2;
            let sd2 = variance.sqrt();
            let mu2 = x - variance / 2.;
            let low = mu2 - 12. * sd2;
            let high = mu2 + 12. * sd2;
            let mut inner = vec![low, high, x];
            inner.extend(nodes);
            for terminal in [(0., 0.), (self.spot, 0.), (0., 1.)] {
                let measure =
                    try_affine_observation_payoff_gamma_measure(&self.payoff, |u, date| {
                        match node_for(u, date)? {
                            0 => Some((self.spot, 0.)),
                            1 => Some((s1, 0.)),
                            2 => Some((0., 1.)),
                            3 => Some(terminal),
                            _ => None,
                        }
                    })?
                    .ok_or_else(|| invalid("piecewise affine conditional payoff required"))?;
                inner.extend(
                    measure
                        .gamma()
                        .atoms()
                        .iter()
                        .map(|a| (a.strike() / self.spot).ln()),
                );
            }
            let inner = cuts(
                inner
                    .into_iter()
                    .filter(|x| low <= *x && *x <= high)
                    .collect(),
            );
            let mut result = [0.; 2];
            for (i, rule) in rules.iter().enumerate() {
                let mut p = NeumaierSum::new();
                let mut probability = NeumaierSum::new();
                let mut spot_moment = NeumaierSum::new();
                for &(z, w) in rule {
                    for cell in inner.windows(2) {
                        let y = (cell[0] + cell[1]) / 2. + (cell[1] - cell[0]) * z / 2.;
                        let s2 = self.spot * y.exp();
                        let payoff = try_affine_observation_payoff_gamma_measure(
                            &self.payoff,
                            |u, date| match node_for(u, date)? {
                                0 => Some((self.spot, 0.)),
                                1 => Some((s1, 0.)),
                                2 => Some((s2, 0.)),
                                3 => Some((0., 1.)),
                                _ => None,
                            },
                        )?
                        .ok_or_else(|| invalid("piecewise affine terminal payoff required"))?;
                        evaluations += 1;
                        if evaluations as u128 * self.payoff.opcodes().len() as u128 > 2_000_000_000
                        {
                            return Err(
                                invalid("initial Gamma continuation work budget exceeded").into()
                            );
                        }
                        let v3 = lv.grid.interpolate(times[2], y)?.value * dt3;
                        let value = sum(std::iter::once(payoff.cash())
                            .chain(std::iter::once(payoff.slope() * s2))
                            .chain(
                                payoff
                                    .gamma()
                                    .atoms()
                                    .iter()
                                    .map(|a| a.mass() * call(s2, a.strike(), v3)),
                            ));
                        if !value.is_finite() {
                            return Err(invalid("nonfinite initial continuation").into());
                        }
                        let density_weight =
                            pdf((y - mu2) / sd2) / sd2 * w * (cell[1] - cell[0]) / 2.;
                        probability.add(density_weight);
                        spot_moment.add(density_weight * (s2 / s1));
                        p.add(value * density_weight);
                    }
                }
                if !probability.total().is_finite()
                    || !spot_moment.total().is_finite()
                    || (probability.total() - 1.).abs() > 1e-10 + tail(0.)
                    || (spot_moment.total() - 1.).abs() > 1e-10 + tail(sd2)
                {
                    return Err(invalid(
                        "initial continuation source probability or martingale integration failed",
                    )
                    .into());
                }
                result[i] = p.total();
            }
            let norm = growth[0] + growth[1] * s1;
            let discrepancy = if norm > 0. {
                (result[0] - result[1]).abs() / norm
            } else {
                (result[0] - result[1]).abs()
            };
            if !discrepancy.is_finite() || !result[1].is_finite() {
                return Err(invalid("nonfinite initial continuation refinement").into());
            }
            inner_refinement = inner_refinement.max(discrepancy);
            Ok(result[1])
        };
        let mut quad = [[0.; 2]; 2];
        for (i, rule) in rules.iter().enumerate() {
            let mut price = NeumaierSum::new();
            let mut risk = NeumaierSum::new();
            for &(z, w) in rule {
                for c in outer.windows(2) {
                    let x = (c[0] + c[1]) / 2. + (c[1] - c[0]) * z / 2.;
                    let normal = (x - mu) / sd;
                    let value = continuation(x)? * pdf(normal) / sd * w * (c[1] - c[0]) / 2.;
                    price.add(value);
                    risk.add(
                        value
                            * maturity
                            * ((normal * normal - 1.) / (2. * v0) - normal / (2. * sd)),
                    );
                }
            }
            quad[i] = [price.total(), risk.total()];
        }
        let xs =
            cuts(
                outer
                    .iter()
                    .copied()
                    .chain((0..=subdivisions).map(|i| {
                        window[0] + (window[1] - window[0]) * i as f64 / subdivisions as f64
                    }))
                    .collect(),
            );
        let mut points = xs
            .into_iter()
            .map(|x| Ok((x, continuation(x)?)))
            .collect::<Result<Vec<_>, MonteCarloError>>()?;
        let recover =
            |n,
             points: &[(f64, f64)]|
             -> Result<(VegaKtInitialSmileResolution, [f64; 2]), MonteCarloError> {
                let spots: Vec<_> = points.iter().map(|(x, _)| self.spot * x.exp()).collect();
                let slopes: Vec<_> = points
                    .windows(2)
                    .zip(spots.windows(2))
                    .map(|(v, s)| (v[1].1 - v[0].1) / (s[1] - s[0]))
                    .collect();
                let mut masses = vec![NeumaierSum::new(); boundaries.len() + 1];
                let mut risks = masses.clone();
                let mut price = NeumaierSum::new();
                price.add(points[0].1);
                price.add(slopes[0] * (self.spot - spots[0]));
                for (i, pair) in slopes.windows(2).enumerate() {
                    let mass = pair[1] - pair[0];
                    let k = spots[i + 1];
                    let x = points[i + 1].0;
                    let bin = boundaries.partition_point(|b| {
                        (b / self.spot).ln() <= x + 32. * f64::EPSILON * (1. + x.abs())
                    });
                    masses[bin].add(mass);
                    risks[bin]
                        .add(mass * self.spot * pdf(-x / sd + sd / 2.) * maturity / (2. * sd));
                    price.add(mass * call(self.spot, k, v0));
                }
                let alpha = (points[0].1 - slopes[0] * spots[0]).abs().max(
                    (points.last().unwrap().1 - slopes.last().unwrap() * spots.last().unwrap())
                        .abs(),
                );
                let beta = slopes[0].abs().max(slopes.last().unwrap().abs());
                let a = growth[0] + alpha;
                let b = growth[1] + beta;
                let tp = a * tail(0.) + b * self.spot * tail(sd);
                let tr = factor * (a * tail_moment(0.) + b * self.spot * tail_moment(sd));
                let masses: Vec<_> = masses.into_iter().map(NeumaierSum::total).collect();
                let risks: Vec<_> = risks.into_iter().map(NeumaierSum::total).collect();
                if masses
                    .iter()
                    .chain(&risks)
                    .chain([price.total(), tp, tr].iter())
                    .any(|v| !v.is_finite())
                {
                    return Err(invalid("nonfinite initial Gamma measure").into());
                }
                Ok((
                    VegaKtInitialSmileResolution {
                        subdivisions: n,
                        interpolation_nodes: points.len(),
                        total_gamma_mass: sum(masses.iter().copied()),
                        recovered_variance_sensitivity: sum(risks.iter().copied()),
                        gamma_bin_masses: masses,
                        gamma_bin_variance_sensitivities: risks,
                        interpolated_price: price.total(),
                    },
                    [tp, tr],
                ))
            };
        let mut levels = vec![recover(subdivisions, &points)?];
        for level in 1..=3 {
            let mut next = Vec::with_capacity(2 * points.len() - 1);
            for pair in points.windows(2) {
                next.push(pair[0]);
                let x = (pair[0].0 + pair[1].0) / 2.;
                next.push((x, continuation(x)?));
            }
            next.push(*points.last().unwrap());
            points = next;
            levels.push(recover(subdivisions * (1 << level), &points)?);
        }
        let coarse = extrapolate(&levels[0].0, &levels[1].0, &levels[2].0);
        let fine = extrapolate(&levels[1].0, &levels[2].0, &levels[3].0);
        let inner_price_allowance = inner_refinement * scale;
        let inner_risk_allowance =
            inner_refinement * factor * (2. * growth[0] + growth[1] * self.spot * (v0 + 2.));
        let tail_price =
            (levels[1].1[0] + 6. * levels[2].1[0] + 8. * levels[3].1[0]) / 3. + inner_tail_price;
        let tail_variance =
            (levels[1].1[1] + 6. * levels[2].1[1] + 8. * levels[3].1[1]) / 3. + inner_tail_risk;
        let quad32 = quad[0];
        let quad64 = quad[1];
        let price_refinement_difference =
            (fine.interpolated_price - coarse.interpolated_price).abs();
        let variance_refinement_difference =
            (fine.recovered_variance_sensitivity - coarse.recovered_variance_sensitivity).abs();
        let bin_mass_refinement_difference = diff(&coarse.gamma_bin_masses, &fine.gamma_bin_masses);
        let bin_variance_refinement_difference = diff(
            &coarse.gamma_bin_variance_sensitivities,
            &fine.gamma_bin_variance_sensitivities,
        );
        let quad_price_diff = (quad32[0] - quad64[0]).abs();
        let quad_risk_diff = (quad32[1] - quad64[1]).abs();
        let recovery_passed = [
            tail_price,
            inner_price_allowance,
            inner_risk_allowance,
            tail_variance,
            quad_price_diff,
            quad_risk_diff,
            price_refinement_difference,
            variance_refinement_difference,
            (fine.interpolated_price - quad64[0]).abs(),
            (fine.recovered_variance_sensitivity - quad64[1]).abs(),
        ]
        .iter()
        .all(|v| v.is_finite() && *v <= functional_tolerance)
            && bin_mass_refinement_difference <= mass_tolerance
            && bin_variance_refinement_difference
                <= mass_tolerance * self.spot * maturity / (2. * sd) * pdf(0.);
        let sample = |z: &[f64]| -> Result<[f64; 4], MonteCarloError> {
            let path = lv.plan.evolve_path(&lv.grid, self.spot, z)?;
            let (price, seeds) = self.chain_discounted_payoff_and_seeds(lv, path.states())?;
            let ad = path.reverse_state_adjoints(&seeds, lv.grid.values().len(), width)?;
            let risk = sum(ad.local_variance_value_adjoints()[..width].iter().copied());
            Ok([
                price,
                risk,
                fine.interpolated_price - price,
                fine.recovered_variance_sensitivity - risk,
            ])
        };
        let antithetic = engine.variance_reduction().antithetic();
        let stats = DeterministicExecutor::new(self.execution_policy)?
            .try_map_reduce_statistics_vector(
                count,
                4,
                |i, out| -> Result<(), MonteCarloError> {
                    let z = lv.plan.path_shocks(
                        engine.master_seed(),
                        i,
                        RandomDomain::Valuation,
                        engine.variance_reduction().brownian_bridge(),
                    )?;
                    let a = sample(&z)?;
                    if antithetic {
                        let b = sample(&z.iter().map(|x| -x).collect::<Vec<_>>())?;
                        for j in 0..4 {
                            out[j] = 0.5 * a[j] + 0.5 * b[j];
                        }
                    } else {
                        out.copy_from_slice(&a);
                    }
                    Ok(())
                },
            )?;
        let m = stats
            .into_iter()
            .map(|s| super::vegakt_steps::moment(s, count))
            .collect::<Result<Vec<_>, _>>()?;
        let validation_passed = m[2..].iter().all(|m| {
            m.standard_error()
                .is_some_and(|se| m.mean().abs() <= 6. * se + functional_tolerance)
        });
        let estimate =
            (recovery_passed && validation_passed).then_some(fine.recovered_variance_sensitivity);
        let audit = VegaKtInitialSmileGammaAudit {
            request_fingerprint: self.request_fingerprint(),
            plan_fingerprint: self.plan_fingerprint(),
            maturity,
            first_step_variance: v0,
            gamma_bin_boundaries: boundaries,
            log_window: window,
            continuation_growth_bound: growth,
            relative_tolerance,
            functional_tolerance,
            mass_tolerance,
            price_tail_bound: tail_price,
            variance_tail_bound: tail_variance,
            quadrature_price: quad64[0],
            quadrature_variance_sensitivity: quad64[1],
            quadrature_price_difference: quad_price_diff,
            quadrature_variance_difference: quad_risk_diff,
            coarse,
            fine,
            price_refinement_difference,
            variance_refinement_difference,
            bin_mass_refinement_difference,
            bin_variance_refinement_difference,
            recovery_passed,
            independent_sampling_units: count,
            antithetic,
            native_price: m[0],
            native_initial_variance_sensitivity: m[1],
            price_difference: m[2],
            variance_difference: m[3],
            validation_passed,
            estimate,
        };
        Ok(VegaKtThreeStepInitialGammaAudit {
            audit,
            refinement_subdivisions: [
                subdivisions,
                2 * subdivisions,
                4 * subdivisions,
                8 * subdivisions,
            ],
            continuation_evaluations: evaluations,
            continuation_relative_refinement: inner_refinement,
            continuation_price_refinement_allowance: inner_price_allowance,
            continuation_variance_refinement_allowance: inner_risk_allowance,
            continuation_price_tail_bound: inner_tail_price,
            continuation_variance_tail_bound: inner_tail_risk,
        })
    }
}
