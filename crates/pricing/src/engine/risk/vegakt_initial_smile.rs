//! Initial Gamma from conditional continuation, including nonflat second variance.
//! A piecewise-linear interpolation in physical S1 has signed slope-jump atoms.
//! Refinement and a separate Gaussian score integral audit the weighted measure.
use super::vegakt_terminal::try_affine_observation_payoff_gamma_measure;
use crate::MonteCarloError;
use crate::engine::plan::simulation::SimulationPlan;
use crate::mc::{DeterministicExecutor, EngineConfig, RandomDomain};
use crate::risk::{
    RiskConfigError, VegaKtInitialSmileGammaAudit, VegaKtInitialSmileResolution, gaussian_legendre,
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
fn tail_probability(sd: f64) -> f64 {
    cdf(-12. - sd) + cdf(-12. + sd)
}
fn tail_second_plus_one(sd: f64) -> f64 {
    // Positive upper bound on E[(Z²+1) 1(|Z|>12)] for Z~N(sd,1).
    (sd * sd + 2.) * tail_probability(sd) + (12. + sd.abs()) * (pdf(12. - sd) + pdf(12. + sd))
}
fn difference(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(a, b)| (a - b).abs())
        .fold(0., f64::max)
}

impl SimulationPlan {
    pub(in crate::engine) fn evaluate_vega_kt_initial_smile_gamma_audit(
        &self,
        boundaries: Vec<f64>,
        subdivisions: usize,
        relative_tolerance: f64,
    ) -> Result<VegaKtInitialSmileGammaAudit, MonteCarloError> {
        let lv = self.state_only_path_gamma_runtime()?;
        let times = lv.plan.time_grid().nodes();
        let nodes = lv.grid.log_moneyness_nodes();
        let width = nodes.len();
        if times.len() != 3
            || times != lv.grid.time_nodes()
            || !(2..=129).contains(&width)
            || lv.grid.values()[..width]
                .iter()
                .any(|v| *v != lv.grid.values()[0])
            || lv.grid.values()[..2 * width].iter().any(|v| *v <= 0.)
        {
            return Err(invalid(
                "two matching intervals, positive variances and flat first row required",
            )
            .into());
        }
        if !(64..=32768).contains(&subdivisions)
            || !(1e-12..=1e-4).contains(&relative_tolerance)
            || boundaries.is_empty()
            || boundaries.len() > 129
            || boundaries.iter().any(|b| !b.is_finite() || *b <= 0.)
            || boundaries.windows(2).any(|b| b[0] >= b[1])
        {
            return Err(invalid("initial smile Gamma requires 64..32768 subdivisions, tolerance in [1e-12,1e-4], and 1..129 positive ordered bin boundaries").into());
        }
        let EngineConfig::PseudoMonteCarlo(engine) = self.engine else {
            return Err(invalid("pseudo-MC required for initial smile validation").into());
        };
        let count = engine.independent_sampling_units().get();
        if !(2..=10_000_000).contains(&count) {
            return Err(invalid("2..10000000 initial smile validation samples required").into());
        }
        if count.div_ceil(self.execution_policy.reduction_block_size().get()) as u128 * 4
            > 4_000_000
        {
            return Err(invalid("initial smile reduction storage budget exceeded").into());
        }
        if (2 * subdivisions + 20000) as u128 * self.payoff.opcodes().len() as u128 > 16_000_000 {
            return Err(invalid("initial smile payoff evaluation budget exceeded").into());
        }
        let observations = self.local_vol_state_observations(lv, &[self.spot; 3], self.spot)?;
        let node_for = |u, date| {
            if u != self.underlying {
                return None;
            }
            self.observation_dates
                .iter()
                .position(|d| *d == Some(date))
                .map(|i| observations[i].node_index)
        };
        let envelope = bound::growth_bound(&self.payoff, node_for, self.spot)?;
        let growth = [envelope[0], envelope[1] + envelope[2]];
        let scale = growth[0] + growth[1] * self.spot;
        let functional_tolerance = relative_tolerance * scale;
        let mass_tolerance = relative_tolerance.sqrt() * (growth[0] / self.spot + growth[1]);
        let maturity = times[1];
        let v0 = lv.grid.values()[0] * maturity;
        let sd = v0.sqrt();
        let mu = -v0 / 2.;
        let window = [mu - 12. * sd, mu + 12. * sd];
        let tail_price =
            growth[0] * tail_probability(0.) + growth[1] * self.spot * tail_probability(sd);
        let tail_variance = maturity
            * (1. / (2. * v0) + 1. / (4. * sd))
            * (growth[0] * tail_second_plus_one(0.)
                + growth[1] * self.spot * tail_second_plus_one(sd));
        if [
            v0,
            scale,
            functional_tolerance,
            mass_tolerance,
            tail_price,
            tail_variance,
        ]
        .iter()
        .any(|v| !v.is_finite())
            || v0 <= 0.
            || !window
                .iter()
                .all(|x| (self.spot * x.exp()).is_finite() && self.spot * x.exp() > 0.)
        {
            return Err(invalid("nonfinite initial smile integration domain or budget").into());
        }
        let mut cuts = vec![window[0], window[1]];
        cuts.extend(
            nodes
                .iter()
                .copied()
                .filter(|x| window[0] < *x && *x < window[1]),
        );
        cuts.extend(
            boundaries
                .iter()
                .map(|b| b.ln() - self.spot.ln())
                .filter(|x| window[0] < *x && *x < window[1]),
        );
        cuts.sort_by(f64::total_cmp);
        cuts.dedup();
        let mut work = 0usize;
        let mut continuation = |x: f64| -> Result<f64, MonteCarloError> {
            let s = self.spot * x.exp();
            let payoff =
                try_affine_observation_payoff_gamma_measure(
                    &self.payoff,
                    |u, date| match node_for(u, date)? {
                        0 => Some((self.spot, 0.)),
                        1 => Some((s, 0.)),
                        2 => Some((0., 1.)),
                        _ => None,
                    },
                )?
                .ok_or_else(|| invalid("conditional terminal payoff must be piecewise affine"))?;
            work += self.payoff.opcodes().len() + payoff.gamma().atoms().len();
            if work > 32_000_000 {
                return Err(
                    invalid("initial smile conditional compilation budget exceeded").into(),
                );
            }
            let variance = lv.grid.interpolate(times[1], x)?.value * (times[2] - times[1]);
            let value = sum(std::iter::once(payoff.cash())
                .chain(std::iter::once(payoff.slope() * s))
                .chain(
                    payoff
                        .gamma()
                        .atoms()
                        .iter()
                        .map(|a| a.mass() * call(s, a.strike(), variance)),
                ));
            if !value.is_finite() {
                return Err(invalid("nonfinite conditional initial smile price").into());
            }
            Ok(value)
        };
        let mut quadrature = |order| -> Result<[f64; 2], MonteCarloError> {
            let mut price = NeumaierSum::new();
            let mut risk = NeumaierSum::new();
            for (z, w) in gaussian_legendre(order)? {
                for c in cuts.windows(2) {
                    let x = (c[0] + c[1]) / 2. + (c[1] - c[0]) * z / 2.;
                    let normal = (x - mu) / sd;
                    let weighted = continuation(x)? * pdf(normal) / sd * w * (c[1] - c[0]) / 2.;
                    price.add(weighted);
                    risk.add(
                        weighted
                            * maturity
                            * ((normal * normal - 1.) / (2. * v0) - normal / (2. * sd)),
                    );
                }
            }
            Ok([price.total(), risk.total()])
        };
        let quad32 = quadrature(32)?;
        let quad64 = quadrature(64)?;
        let mut coarse_xs = cuts.clone();
        coarse_xs.extend(
            (0..=subdivisions)
                .map(|i| window[0] + (window[1] - window[0]) * i as f64 / subdivisions as f64),
        );
        coarse_xs.sort_by(f64::total_cmp);
        // Avoid a zero physical spacing at roundoff-coincident declared cuts.
        coarse_xs.dedup_by(|a, b| (*a - *b).abs() <= 32. * f64::EPSILON * (1. + a.abs() + b.abs()));
        // Bisect EVERY existing cell, including cells shortened by variance or
        // reporting knots. Doubling only the background grid can leave those
        // one-sided slopes unchanged and mask bin-mass discretization error.
        let mut fine_xs = coarse_xs.clone();
        fine_xs.extend(coarse_xs.windows(2).map(|c| c[0] + (c[1] - c[0]) / 2.));
        fine_xs.sort_by(f64::total_cmp);
        let mut recover =
            |n: usize, xs: &[f64]| -> Result<VegaKtInitialSmileResolution, MonteCarloError> {
                let spots: Vec<_> = xs.iter().map(|x| self.spot * x.exp()).collect();
                let values = xs
                    .iter()
                    .map(|&x| continuation(x))
                    .collect::<Result<Vec<_>, _>>()?;
                let slopes = values
                    .windows(2)
                    .zip(spots.windows(2))
                    .map(|(v, s)| (v[1] - v[0]) / (s[1] - s[0]))
                    .collect::<Vec<_>>();
                let mut masses = vec![NeumaierSum::new(); boundaries.len() + 1];
                let mut risks = masses.clone();
                let mut price = NeumaierSum::new();
                price.add(values[0]);
                price.add(slopes[0] * (self.spot - spots[0]));
                for (i, s) in slopes.windows(2).enumerate() {
                    let k = spots[i + 1];
                    let mass = s[1] - s[0];
                    let d1 = (self.spot.ln() - k.ln()) / sd + sd / 2.;
                    // Bin against log boundaries with a roundoff guard: an exact
                    // declared boundary belongs to its right-hand bin.
                    let x = xs[i + 1];
                    let bin = boundaries.partition_point(|b| {
                        b.ln() - self.spot.ln() <= x + 32. * f64::EPSILON * (1. + x.abs())
                    });
                    masses[bin].add(mass);
                    risks[bin].add(mass * self.spot * pdf(d1) * maturity / (2. * sd));
                    price.add(mass * call(self.spot, k, v0));
                }
                let masses = masses
                    .into_iter()
                    .map(NeumaierSum::total)
                    .collect::<Vec<_>>();
                let risks = risks
                    .into_iter()
                    .map(NeumaierSum::total)
                    .collect::<Vec<_>>();
                if masses
                    .iter()
                    .chain(&risks)
                    .chain(std::iter::once(&price.total()))
                    .any(|v| !v.is_finite())
                {
                    return Err(invalid("nonfinite initial smile Gamma reconstruction").into());
                }
                Ok(VegaKtInitialSmileResolution {
                    subdivisions: n,
                    interpolation_nodes: xs.len(),
                    total_gamma_mass: sum(masses.iter().copied()),
                    recovered_variance_sensitivity: sum(risks.iter().copied()),
                    gamma_bin_masses: masses,
                    gamma_bin_variance_sensitivities: risks,
                    interpolated_price: price.total(),
                })
            };
        let coarse = recover(subdivisions, &coarse_xs)?;
        let fine = recover(2 * subdivisions, &fine_xs)?;
        let price_refinement_difference =
            (fine.interpolated_price - coarse.interpolated_price).abs();
        let variance_refinement_difference =
            (fine.recovered_variance_sensitivity - coarse.recovered_variance_sensitivity).abs();
        let bin_mass_refinement_difference =
            difference(&coarse.gamma_bin_masses, &fine.gamma_bin_masses);
        let bin_variance_refinement_difference = difference(
            &coarse.gamma_bin_variance_sensitivities,
            &fine.gamma_bin_variance_sensitivities,
        );
        let quad_price_diff = (quad32[0] - quad64[0]).abs();
        let quad_risk_diff = (quad32[1] - quad64[1]).abs();
        let recovery_passed = [
            tail_price,
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
        Ok(VegaKtInitialSmileGammaAudit {
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
        })
    }
}
