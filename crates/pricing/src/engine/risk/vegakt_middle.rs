//! Conditional middle-state Gamma, with signed Richardson measures on nested grids.
use super::vegakt_terminal::try_affine_observation_payoff_gamma_measure;
use crate::MonteCarloError;
use crate::engine::plan::simulation::SimulationPlan;
use crate::mc::{DeterministicExecutor, EngineConfig, RandomDomain};
use crate::risk::{
    RiskConfigError, VegaKtMiddleGammaResolution, VegaKtMiddleSmileGammaAudit, gaussian_legendre,
};
use pricing_numerics::{NeumaierSum, standard_normal_cdf as cdf, standard_normal_pdf as pdf};
mod bound;
fn invalid(s: &'static str) -> RiskConfigError {
    RiskConfigError::VegaKtChainInput { quantity: s }
}
fn sum(v: impl Iterator<Item = f64>) -> f64 {
    v.collect::<NeumaierSum>().total()
}
fn max_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(a, b)| (a - b).abs())
        .fold(0., f64::max)
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
fn cuts(mut xs: Vec<f64>) -> Vec<f64> {
    xs.sort_by(f64::total_cmp);
    xs.dedup_by(|a, b| (*a - *b).abs() <= 32. * f64::EPSILON * (1. + a.abs() + b.abs()));
    xs
}
struct Slice {
    price: f64,
    risk: f64,
    masses: Vec<f64>,
    bin_risks: Vec<f64>,
    tail_price: f64,
    tail_risk: f64,
    negative: usize,
}

impl SimulationPlan {
    pub(in crate::engine) fn evaluate_vega_kt_middle_smile_gamma_audit(
        &self,
        boundaries: Vec<f64>,
        subdivisions: usize,
        relative_tolerance: f64,
    ) -> Result<VegaKtMiddleSmileGammaAudit, MonteCarloError> {
        let lv = self.state_only_path_gamma_runtime()?;
        let times = lv.plan.time_grid().nodes();
        let nodes = lv.grid.log_moneyness_nodes();
        let m = nodes.len();
        if times.len() != 4
            || times != lv.grid.time_nodes()
            || !(2..=33).contains(&m)
            || lv.grid.values()[..m]
                .iter()
                .any(|v| *v != lv.grid.values()[0])
        {
            return Err(invalid("three matching intervals, flat first variance and 2..33 nodes required for middle Gamma").into());
        }
        if !(64..=4096).contains(&subdivisions)
            || !(1e-12..=1e-4).contains(&relative_tolerance)
            || boundaries.is_empty()
            || boundaries.len() > 33
            || boundaries.iter().any(|v| !v.is_finite() || *v <= 0.)
            || boundaries.windows(2).any(|v| v[0] >= v[1])
        {
            return Err(invalid("middle Gamma needs 64..4096 base subdivisions, 1..33 positive ordered bins and tolerance in [1e-12,1e-4]").into());
        }
        let EngineConfig::PseudoMonteCarlo(engine) = self.engine else {
            return Err(invalid("pseudo-MC required for middle Gamma validation").into());
        };
        let count = engine.independent_sampling_units().get();
        let components = 2 + 2 * m;
        if !(2..=10_000_000).contains(&count)
            || count.div_ceil(self.execution_policy.reduction_block_size().get()) as u128
                * components as u128
                > 4_000_000
        {
            return Err(invalid("middle Gamma sample or storage budget exceeded").into());
        }
        let estimated = (m + boundaries.len() + 3) as u128
            * 96
            * (8 * subdivisions + 160 * (m + boundaries.len() + 10)) as u128
            * self.payoff.opcodes().len() as u128;
        if estimated > 2_000_000_000 {
            return Err(invalid("middle Gamma deterministic work budget exceeded").into());
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
        let growth = [envelope[0], envelope[1], envelope[2] + envelope[3]];
        let scale = growth[0] + (growth[1] + growth[2]) * self.spot;
        let tolerance = relative_tolerance * scale;
        let mass_tolerance =
            relative_tolerance.sqrt() * (growth[0] / self.spot + growth[1] + growth[2]);
        let v0 = lv.grid.values()[0] * times[1];
        let sd0 = v0.sqrt();
        let dt = times[2] - times[1];
        let dt3 = times[3] - times[2];
        let min_v = lv.grid.values()[m..2 * m]
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min)
            * dt;
        let max_v = lv.grid.values()[m..2 * m]
            .iter()
            .copied()
            .fold(0., f64::max)
            * dt;
        if !scale.is_finite() || v0 <= 0. || min_v <= 0. || max_v.sqrt() > 4. || sd0 > 4. {
            return Err(invalid("middle Gamma growth, variance or tail domain invalid").into());
        }
        let factor = dt * (1. / (2. * min_v) + 1. / (4. * min_v.sqrt()));
        let outer_price_tail =
            growth[0] * tail(0.) + (growth[1] + growth[2]) * self.spot * tail(sd0);
        let outer_risk_tail = factor
            * (2. * growth[0] * tail(0.)
                + (2. * growth[1] + growth[2] * (max_v + 2.)) * self.spot * tail(sd0));
        let lo = -v0 / 2. - 12. * sd0;
        let hi = -v0 / 2. + 12. * sd0;
        let outer_cuts = cuts(
            vec![lo, hi]
                .into_iter()
                .chain(nodes.iter().copied())
                .chain(boundaries.iter().map(|b| (b / self.spot).ln()))
                .filter(|x| lo <= *x && *x <= hi)
                .collect(),
        );
        let integrate = |order| -> Result<VegaKtMiddleGammaResolution, MonteCarloError> {
            let mut price = NeumaierSum::new();
            let mut b = vec![NeumaierSum::new(); m];
            let bins = boundaries.len() + 1;
            let mut masses = vec![NeumaierSum::new(); bins];
            let mut bin_risks = masses.clone();
            let mut quad_price = NeumaierSum::new();
            let mut quad_b = b.clone();
            let (
                mut inner_p,
                mut inner_r,
                mut inner_mass,
                mut score_p,
                mut score_r,
                mut tp,
                mut tr,
            ) = (
                NeumaierSum::new(),
                NeumaierSum::new(),
                NeumaierSum::new(),
                NeumaierSum::new(),
                NeumaierSum::new(),
                NeumaierSum::new(),
                NeumaierSum::new(),
            );
            let mut histories = 0;
            let mut evaluations = 0;
            let mut negative = 0;
            for (z, w) in gaussian_legendre(order)? {
                for cell in outer_cuts.windows(2) {
                    let x1 = (cell[0] + cell[1]) / 2. + (cell[1] - cell[0]) * z / 2.;
                    let s1 = self.spot * x1.exp();
                    let weight = pdf((x1 + v0 / 2.) / sd0) / sd0 * w * (cell[1] - cell[0]) / 2.;
                    let j = nodes
                        .partition_point(|x| *x <= x1)
                        .saturating_sub(1)
                        .min(m - 2);
                    let h = ((x1 - nodes[j]) / (nodes[j + 1] - nodes[j])).clamp(0., 1.);
                    let v = lv.grid.interpolate(times[1], x1)?.value * dt;
                    let sd = v.sqrt();
                    let mu = x1 - v / 2.;
                    let window = [mu - 12. * sd, mu + 12. * sd];
                    let mut special = vec![window[0], window[1], x1];
                    special.extend(nodes.iter().copied());
                    special.extend(boundaries.iter().map(|k| (k / self.spot).ln()));
                    // These exact compiled slices expose payoff switching points;
                    // remaining smooth regions are checked by nested refinement.
                    for terminal in [(0., 0.), (self.spot, 0.), (0., 1.)] {
                        let measure = try_affine_observation_payoff_gamma_measure(
                            &self.payoff,
                            |u, date| match node_for(u, date)? {
                                0 => Some((self.spot, 0.)),
                                1 => Some((s1, 0.)),
                                2 => Some((0., 1.)),
                                3 => Some(terminal),
                                _ => None,
                            },
                        )?
                        .ok_or_else(|| invalid("piecewise affine middle payoff required"))?;
                        special.extend(
                            measure
                                .gamma()
                                .atoms()
                                .iter()
                                .map(|a| (a.strike() / self.spot).ln()),
                        );
                    }
                    let special = cuts(
                        special
                            .into_iter()
                            .filter(|x| window[0] <= *x && *x <= window[1])
                            .collect(),
                    );
                    let mut continuation = |x: f64| -> Result<f64, MonteCarloError> {
                        let s2 = self.spot * x.exp();
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
                        .ok_or_else(|| {
                            invalid("piecewise affine conditional terminal payoff required")
                        })?;
                        evaluations += 1;
                        if evaluations as u128 * self.payoff.opcodes().len() as u128 > 2_000_000_000
                        {
                            return Err(invalid("middle Gamma continuation budget exceeded").into());
                        }
                        let variance = lv.grid.interpolate(times[2], x)?.value * dt3;
                        let value = sum(std::iter::once(payoff.cash())
                            .chain(std::iter::once(payoff.slope() * s2))
                            .chain(
                                payoff
                                    .gamma()
                                    .atoms()
                                    .iter()
                                    .map(|a| a.mass() * call(s2, a.strike(), variance)),
                            ));
                        if !value.is_finite() {
                            return Err(invalid("nonfinite middle continuation").into());
                        }
                        Ok(value)
                    };
                    let mut quadrature = |n| -> Result<[f64; 2], MonteCarloError> {
                        let mut p = NeumaierSum::new();
                        let mut r = NeumaierSum::new();
                        for (z, w) in gaussian_legendre(n)? {
                            for c in special.windows(2) {
                                let x = (c[0] + c[1]) / 2. + (c[1] - c[0]) * z / 2.;
                                let normal = (x - mu) / sd;
                                let value =
                                    continuation(x)? * pdf(normal) / sd * w * (c[1] - c[0]) / 2.;
                                p.add(value);
                                r.add(
                                    value
                                        * dt
                                        * ((normal * normal - 1.) / (2. * v) - normal / (2. * sd)),
                                );
                            }
                        }
                        Ok([p.total(), r.total()])
                    };
                    let q32 = quadrature(32)?;
                    let q64 = quadrature(64)?;
                    let xs = cuts(
                        special
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
                    let recover = |points: &[(f64, f64)]| -> Slice {
                        let s: Vec<_> = points.iter().map(|(x, _)| self.spot * x.exp()).collect();
                        let slopes: Vec<_> = points
                            .windows(2)
                            .zip(s.windows(2))
                            .map(|(p, s)| (p[1].1 - p[0].1) / (s[1] - s[0]))
                            .collect();
                        let mut ms = vec![NeumaierSum::new(); bins];
                        let mut rs = ms.clone();
                        let mut price = NeumaierSum::new();
                        let mut negative = 0;
                        price.add(points[0].1);
                        price.add(slopes[0] * (s1 - s[0]));
                        for (i, pair) in slopes.windows(2).enumerate() {
                            let mass = pair[1] - pair[0];
                            negative += usize::from(mass < 0.);
                            let k = s[i + 1];
                            let x = points[i + 1].0;
                            let bin = boundaries.partition_point(|b| {
                                (b / self.spot).ln() <= x + 32. * f64::EPSILON * (1. + x.abs())
                            });
                            ms[bin].add(mass);
                            rs[bin].add(
                                mass * s1 * pdf((s1.ln() - k.ln()) / sd + sd / 2.) * dt / (2. * sd),
                            );
                            price.add(mass * call(s1, k, v));
                        }
                        let alpha = (points[0].1 - slopes[0] * s[0]).abs().max(
                            (points.last().unwrap().1 - slopes.last().unwrap() * s.last().unwrap())
                                .abs(),
                        );
                        let beta = slopes[0].abs().max(slopes.last().unwrap().abs());
                        let a = growth[0] + growth[1] * s1 + alpha;
                        let c = growth[2] + beta;
                        let masses: Vec<_> = ms.into_iter().map(NeumaierSum::total).collect();
                        let risks: Vec<_> = rs.into_iter().map(NeumaierSum::total).collect();
                        Slice {
                            price: price.total(),
                            risk: sum(risks.iter().copied()),
                            masses,
                            bin_risks: risks,
                            tail_price: a * tail(0.) + c * s1 * tail(sd),
                            tail_risk: dt
                                * (1. / (2. * v) + 1. / (4. * sd))
                                * (a * tail_moment(0.) + c * s1 * tail_moment(sd)),
                            negative,
                        }
                    };
                    let mut levels = vec![recover(&points)];
                    for _ in 0..3 {
                        let mut next = Vec::with_capacity(2 * points.len() - 1);
                        for pair in points.windows(2) {
                            next.push(pair[0]);
                            let x = (pair[0].0 + pair[1].0) / 2.;
                            next.push((x, continuation(x)?));
                        }
                        next.push(*points.last().unwrap());
                        points = next;
                        levels.push(recover(&points));
                    }
                    let [a, c, f, g] = levels.as_slice() else {
                        unreachable!()
                    };
                    // Bin-edge slope jumps have an O(h) term even when smooth
                    // weighted functionals have O(h^2) error. Two-stage signed
                    // Richardson cancels both on the same nested measure.
                    let extrap = |a: f64, c: f64, f: f64| (8. * f - 6. * c + a) / 3.;
                    let p = extrap(c.price, f.price, g.price);
                    let r = extrap(c.risk, f.risk, g.risk);
                    inner_p.add(weight * (p - extrap(a.price, c.price, f.price)).abs());
                    inner_r.add(weight * (r - extrap(a.risk, c.risk, f.risk)).abs());
                    inner_mass.add(
                        weight
                            * (0..bins)
                                .map(|i| {
                                    (extrap(c.masses[i], f.masses[i], g.masses[i])
                                        - extrap(a.masses[i], c.masses[i], f.masses[i]))
                                    .abs()
                                })
                                .fold(0., f64::max),
                    );
                    score_p.add(weight * (q32[0] - q64[0]).abs());
                    score_r.add(weight * (q32[1] - q64[1]).abs());
                    price.add(weight * p);
                    quad_price.add(weight * q64[0]);
                    for (index, hat) in [(j, 1. - h), (j + 1, h)] {
                        b[index].add(weight * hat * r);
                        quad_b[index].add(weight * hat * q64[1]);
                    }
                    for i in 0..bins {
                        masses[i].add(weight * extrap(c.masses[i], f.masses[i], g.masses[i]));
                        bin_risks[i]
                            .add(weight * extrap(c.bin_risks[i], f.bin_risks[i], g.bin_risks[i]));
                    }
                    tp.add(weight * (8. * g.tail_price + 6. * f.tail_price + c.tail_price) / 3.);
                    tr.add(weight * (8. * g.tail_risk + 6. * f.tail_risk + c.tail_risk) / 3.);
                    negative += g.negative;
                    histories += 1;
                }
            }
            Ok(VegaKtMiddleGammaResolution {
                history_order: order,
                subdivisions,
                histories,
                continuation_evaluations: evaluations,
                recovered_price: price.total(),
                variance_node_sensitivities: b.into_iter().map(NeumaierSum::total).collect(),
                history_averaged_gamma_bin_masses: masses
                    .into_iter()
                    .map(NeumaierSum::total)
                    .collect(),
                gamma_bin_variance_sensitivities: bin_risks
                    .into_iter()
                    .map(NeumaierSum::total)
                    .collect(),
                quadrature_price: quad_price.total(),
                quadrature_variance_node_sensitivities: quad_b
                    .into_iter()
                    .map(NeumaierSum::total)
                    .collect(),
                inner_price_refinement: inner_p.total(),
                inner_variance_refinement: inner_r.total(),
                inner_bin_mass_refinement: inner_mass.total(),
                score_price_refinement: score_p.total(),
                score_variance_refinement: score_r.total(),
                price_tail_bound: tp.total() + outer_price_tail,
                variance_tail_bound: tr.total() + outer_risk_tail,
                negative_fine_gamma_atoms: negative,
            })
        };
        let coarse = integrate(32)?;
        let fine = integrate(64)?;
        let outer_p = (coarse.recovered_price - fine.recovered_price).abs();
        let outer_r = max_diff(
            &coarse.variance_node_sensitivities,
            &fine.variance_node_sensitivities,
        );
        let outer_mass = max_diff(
            &coarse.history_averaged_gamma_bin_masses,
            &fine.history_averaged_gamma_bin_masses,
        );
        let recovery_passed = [&coarse, &fine].iter().all(|r| {
            [
                r.inner_price_refinement,
                r.inner_variance_refinement,
                r.score_price_refinement,
                r.score_variance_refinement,
                r.price_tail_bound,
                r.variance_tail_bound,
                (r.recovered_price - r.quadrature_price).abs(),
                max_diff(
                    &r.variance_node_sensitivities,
                    &r.quadrature_variance_node_sensitivities,
                ),
            ]
            .iter()
            .all(|x| x.is_finite() && *x <= tolerance)
                && r.inner_bin_mass_refinement <= mass_tolerance
        }) && outer_p <= tolerance
            && outer_r <= tolerance
            && outer_mass <= mass_tolerance;
        // Sweep every first-state cell on each independent sampling unit.
        // Uniform within-cell sampling is weighted by the original Gaussian
        // density; no normalization or Gamma fit is used. This avoids treating
        // an unvisited source cell as a zero-risk, zero-SE observation.
        let sample = |z: &[f64]| -> Result<Vec<f64>, MonteCarloError> {
            let u = cdf(z[0]);
            let mut price = NeumaierSum::new();
            let mut middle = vec![NeumaierSum::new(); m];
            let mut conditioned = z.to_vec();
            for cell in outer_cuts.windows(2) {
                let x = cell[0] + (cell[1] - cell[0]) * u;
                conditioned[0] = (x + v0 / 2.) / sd0;
                let weight = pdf(conditioned[0]) / sd0 * (cell[1] - cell[0]);
                let path = lv.plan.evolve_path(&lv.grid, self.spot, &conditioned)?;
                let (value, seeds) = self.chain_discounted_payoff_and_seeds(lv, path.states())?;
                let ad = path.reverse_state_adjoints(&seeds, lv.grid.values().len(), m)?;
                price.add(weight * value);
                for (out, b) in middle
                    .iter_mut()
                    .zip(&ad.local_variance_value_adjoints()[m..2 * m])
                {
                    out.add(weight * b);
                }
            }
            let price = price.total();
            let middle: Vec<_> = middle.into_iter().map(NeumaierSum::total).collect();
            Ok(std::iter::once(price)
                .chain(middle.iter().copied())
                .chain(std::iter::once(fine.recovered_price - price))
                .chain(
                    fine.variance_node_sensitivities
                        .iter()
                        .zip(&middle)
                        .map(|(a, b)| a - b),
                )
                .collect())
        };
        let antithetic = engine.variance_reduction().antithetic();
        let stats = DeterministicExecutor::new(self.execution_policy)?
            .try_map_reduce_statistics_vector(
                count,
                components,
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
                        for ((o, a), b) in out.iter_mut().zip(a).zip(b) {
                            *o = 0.5 * a + 0.5 * b;
                        }
                    } else {
                        out.copy_from_slice(&a)
                    }
                    Ok(())
                },
            )?;
        let stats = stats
            .into_iter()
            .map(|s| super::vegakt_steps::moment(s, count))
            .collect::<Result<Vec<_>, _>>()?;
        let validation_passed = stats[1 + m..].iter().all(|s| {
            s.standard_error()
                .is_some_and(|se| s.mean().abs() <= 6. * se + tolerance)
        });
        let estimate = (recovery_passed && validation_passed)
            .then(|| fine.variance_node_sensitivities.clone());
        Ok(VegaKtMiddleSmileGammaAudit {
            request_fingerprint: self.request_fingerprint(),
            plan_fingerprint: self.plan_fingerprint(),
            gamma_bin_boundaries: boundaries,
            continuation_growth_bound: growth,
            relative_tolerance,
            functional_tolerance: tolerance,
            mass_tolerance,
            coarse_history: coarse,
            fine_history: fine,
            outer_price_refinement: outer_p,
            outer_variance_refinement: outer_r,
            outer_bin_mass_refinement: outer_mass,
            recovery_passed,
            independent_sampling_units: count,
            validation_history_strata: outer_cuts.len() - 1,
            validation_history_log_window: [lo, hi],
            antithetic,
            native_price: stats[0],
            native_middle_variance_sensitivities: stats[1..1 + m].to_vec(),
            price_difference: stats[1 + m],
            variance_differences: stats[2 + m..].to_vec(),
            validation_passed,
            estimate,
        })
    }
}
